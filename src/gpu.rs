//! Host side of the OpenCL port: device/platform management, batching, and the
//! per-primitive selftest harness that verifies each device function against
//! the CPU crates.
//!
//! Ported from the CUDA host (`cust` crate) to OpenCL so the search can run on
//! non-NVIDIA GPUs — developed against an Intel Arc B580. The kernels live in
//! `src/ocl/kernels.cl` and are compiled by the device's OpenCL compiler at
//! startup; no external toolchain is needed at build time.

use anyhow::{Context as _, Result};
use opencl3::command_queue::{CommandQueue, CL_BLOCKING};
use opencl3::context::Context;
use opencl3::device::{Device, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY};
use opencl3::platform::get_platforms;
use opencl3::program::Program;
use opencl3::types::cl_mem_flags;

/// OpenCL C source of `src/ocl/kernels.cl`.
const CL_SRC: &str = include_str!("ocl/kernels.cl");

/// Work-group size used by the selftest kernels.
const LOCAL_SIZE: usize = 256;
/// Fixed-base table of multiples of G: 64 windows x 15 entries, each an affine
/// point (2 x 32 bytes) = 61440 bytes.
const GTABLE_BYTES: usize = 64 * 15 * 64;

/// Owns the OpenCL context + compiled program for the lifetime of a run.
pub struct Gpu {
    device: Device,
    _context: Context,
    /// Queue for transfers + the cheap checksum filter pass.
    queue: CommandQueue,
    /// Queue for the heavy derivation pass. A separate queue lets a pipeline
    /// kernel run concurrently with the next batch's filter/transfers, keeping
    /// the device busy instead of stalling on per-batch sync points.
    pipeline_queue: CommandQueue,
    program: Program,
    /// The fixed-base window table of multiples of G, built once by
    /// `k_init_gtable` and read by every kernel that derives a public key.
    gtable: Buffer<u8>,
}

/// Picks the GPU to run on: the first Intel GPU (the intended target), else the
/// first GPU of any vendor.
fn find_gpu_device() -> Result<Device> {
    let platforms =
        get_platforms().context("no OpenCL platform found (is the GPU driver installed?)")?;
    let mut fallback: Option<Device> = None;
    for platform in &platforms {
        let name = platform.name().unwrap_or_default();
        let Ok(ids) = platform.get_devices(CL_DEVICE_TYPE_GPU) else {
            continue;
        };
        for id in ids {
            let device = Device::new(id);
            if name.contains("Intel") {
                return Ok(device);
            }
            if fallback.is_none() {
                fallback = Some(device);
            }
        }
    }
    fallback.context("no OpenCL GPU device found")
}

impl Gpu {
    pub fn new() -> Result<Self> {
        let device = find_gpu_device()?;
        let device_name = device.name().unwrap_or_default();
        let _context = Context::from_device(&device).context("OpenCL context creation failed")?;
        let queue =
            CommandQueue::create_default(&_context, 0).context("creating OpenCL command queue")?;
        let pipeline_queue = CommandQueue::create_default(&_context, 0)
            .context("creating OpenCL pipeline command queue")?;
        // Build the program. The Intel OpenCL driver accepts `-cl-intel-256-GRF-
        // per-thread` on Arc/Xe GPUs, which gives each work-item the larger
        // register file and is empirically ~3-5% faster on k_pipeline here
        // (measured on the B580). The option is ignored on non-Intel drivers
        // (NEO) and on older Intel drivers, so the first try uses it; if the
        // driver rejects it (very old IGC, non-Intel ICD), we fall back to the
        // default. Either way, the resulting binary is bit-exact: registers
        // change scheduling, not semantics, and the existing selftest covers
        // every primitive.
        let program = match Program::create_and_build_from_source(
            &_context,
            CL_SRC,
            "-cl-intel-256-GRF-per-thread",
        ) {
            Ok(p) => p,
            Err(_) => Program::create_and_build_from_source(&_context, CL_SRC, "")
                .map_err(|e| anyhow::anyhow!("building OpenCL kernels: {e}"))
                .context("compiling src/ocl/kernels.cl")?,
        };

        // Build the fixed-base window table of multiples of G. Every kernel that
        // derives a public key reads it, so it must run before anything else.
        let gtable = unsafe {
            Buffer::<u8>::create(&_context, CL_MEM_READ_WRITE, GTABLE_BYTES, std::ptr::null_mut())
        }
        .context("allocating G table buffer")?;

        let init = Kernel::create(&program, "k_init_gtable").context("loading k_init_gtable")?;
        unsafe {
            ExecuteKernel::new(&init)
                .set_arg(&gtable)
                .set_global_work_size(1024)
                .set_local_work_size(LOCAL_SIZE)
                .enqueue_nd_range(&queue)
        }
        .context("launching k_init_gtable")?;
        queue.finish().context("building G table")?;

        println!("OpenCL device: {device_name}");
        Ok(Self {
            device,
            _context,
            queue,
            pipeline_queue,
            program,
            gtable,
        })
    }

    /// Blocking write of `data` into `buffer` (which must hold at least that
    /// many elements).
    fn write<T>(&self, buffer: &mut Buffer<T>, data: &[T]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        unsafe { self.queue.enqueue_write_buffer(buffer, CL_BLOCKING, 0, data, &[])? };
        Ok(())
    }

    /// Blocking read of `buffer` into `data`.
    fn read<T>(&self, buffer: &Buffer<T>, data: &mut [T]) -> Result<()> {
        unsafe { self.queue.enqueue_read_buffer(buffer, CL_BLOCKING, 0, data, &[])? };
        Ok(())
    }

    fn new_buffer<T>(&self, count: usize, flags: cl_mem_flags) -> Result<Buffer<T>> {
        unsafe { Buffer::<T>::create(&self._context, flags, count, std::ptr::null_mut()) }
            .context("allocating device buffer")
    }

    fn buffer_from_slice<T: Copy>(&self, data: &[T], flags: cl_mem_flags) -> Result<Buffer<T>> {
        let mut buf = self.new_buffer::<T>(data.len().max(1), flags)?;
        self.write(&mut buf, data)?;
        Ok(buf)
    }

    /// Launches `kernel` over `n` one-dimensional work items in work-groups of
    /// `local` and waits for completion. Kernel arguments must already be set
    /// (via `Kernel::set_arg`).
    fn launch(&self, kernel: &Kernel, n: usize, local: usize) -> Result<()> {
        let global = n.div_ceil(local).max(1) * local;
        let g = [global];
        let l = [local];
        unsafe {
            self.queue.enqueue_nd_range_kernel(
                kernel.get(),
                1,
                std::ptr::null(),
                g.as_ptr(),
                l.as_ptr(),
                &[],
            )
        }?;
        self.queue.finish()?;
        Ok(())
    }

    /// Runs a one-message-per-thread hash kernel over `inputs`, returning a
    /// `digest_len`-byte digest per input. The kernel signature must be
    /// `(const u8* msgs, const u32* lens, u32 stride, u8* out, u32 n)`.
    fn hash_batch(&self, kernel: &str, inputs: &[Vec<u8>], digest_len: usize) -> Result<Vec<Vec<u8>>> {
        let n = inputs.len();
        let (packed, lens, stride) = pack(inputs);

        let d_msgs = self.buffer_from_slice(&packed, CL_MEM_READ_ONLY)?;
        let d_lens = self.buffer_from_slice(&lens, CL_MEM_READ_ONLY)?;
        let d_out = self.new_buffer::<u8>(n * digest_len, CL_MEM_WRITE_ONLY)?;

        let func = Kernel::create(&self.program, kernel)?;
        unsafe {
            func.set_arg(0, &d_msgs)?;
            func.set_arg(1, &d_lens)?;
            func.set_arg(2, &(stride as u32))?;
            func.set_arg(3, &d_out)?;
            func.set_arg(4, &(n as u32))?;
        }
        self.launch(&func, n, LOCAL_SIZE)?;

        let mut out = vec![0u8; n * digest_len];
        self.read(&d_out, &mut out)?;
        Ok(out.chunks(digest_len).map(|c| c.to_vec()).collect())
    }

    /// One HMAC-SHA512 (64-byte output) per (key, msg) pair.
    fn hmac_batch(&self, keys: &[Vec<u8>], msgs: &[Vec<u8>]) -> Result<Vec<Vec<u8>>> {
        let n = keys.len();
        let (pk, klens, kstride) = pack(keys);
        let (pm, mlens, mstride) = pack(msgs);

        let d_keys = self.buffer_from_slice(&pk, CL_MEM_READ_ONLY)?;
        let d_klens = self.buffer_from_slice(&klens, CL_MEM_READ_ONLY)?;
        let d_msgs = self.buffer_from_slice(&pm, CL_MEM_READ_ONLY)?;
        let d_mlens = self.buffer_from_slice(&mlens, CL_MEM_READ_ONLY)?;
        let d_out = self.new_buffer::<u8>(n * 64, CL_MEM_WRITE_ONLY)?;

        let func = Kernel::create(&self.program, "k_hmac_sha512")?;
        unsafe {
            func.set_arg(0, &d_keys)?;
            func.set_arg(1, &d_klens)?;
            func.set_arg(2, &(kstride as u32))?;
            func.set_arg(3, &d_msgs)?;
            func.set_arg(4, &d_mlens)?;
            func.set_arg(5, &(mstride as u32))?;
            func.set_arg(6, &d_out)?;
            func.set_arg(7, &(n as u32))?;
        }
        self.launch(&func, n, LOCAL_SIZE)?;

        let mut out = vec![0u8; n * 64];
        self.read(&d_out, &mut out)?;
        Ok(out.chunks(64).map(|c| c.to_vec()).collect())
    }

    /// One PBKDF2-HMAC-SHA512 (dkLen=64) per (password, salt) pair.
    fn pbkdf2_batch(&self, pws: &[Vec<u8>], salts: &[Vec<u8>], iters: u32) -> Result<Vec<Vec<u8>>> {
        let n = pws.len();
        let (pp, pwlens, pwstride) = pack(pws);
        let (ps, slens, sstride) = pack(salts);

        let d_pw = self.buffer_from_slice(&pp, CL_MEM_READ_ONLY)?;
        let d_pwlens = self.buffer_from_slice(&pwlens, CL_MEM_READ_ONLY)?;
        let d_salt = self.buffer_from_slice(&ps, CL_MEM_READ_ONLY)?;
        let d_slens = self.buffer_from_slice(&slens, CL_MEM_READ_ONLY)?;
        let d_out = self.new_buffer::<u8>(n * 64, CL_MEM_WRITE_ONLY)?;

        let func = Kernel::create(&self.program, "k_pbkdf2")?;
        unsafe {
            func.set_arg(0, &d_pw)?;
            func.set_arg(1, &d_pwlens)?;
            func.set_arg(2, &(pwstride as u32))?;
            func.set_arg(3, &d_salt)?;
            func.set_arg(4, &d_slens)?;
            func.set_arg(5, &(sstride as u32))?;
            func.set_arg(6, &iters)?;
            func.set_arg(7, &d_out)?;
            func.set_arg(8, &(n as u32))?;
        }
        self.launch(&func, n, LOCAL_SIZE)?;

        let mut out = vec![0u8; n * 64];
        self.read(&d_out, &mut out)?;
        Ok(out.chunks(64).map(|c| c.to_vec()).collect())
    }

    /// One 64-byte BIP-39 seed in -> 20-byte Ethereum address out.
    fn seed_to_eth_batch(&self, seeds: &[[u8; 64]]) -> Result<Vec<[u8; 20]>> {
        let n = seeds.len();
        let flat: Vec<u8> = seeds.iter().flatten().copied().collect();
        let d_seeds = self.buffer_from_slice(&flat, CL_MEM_READ_ONLY)?;
        let d_out = self.new_buffer::<u8>(n * 20, CL_MEM_WRITE_ONLY)?;

        let func = Kernel::create(&self.program, "k_seed_to_eth")?;
        unsafe {
            func.set_arg(0, &d_seeds)?;
            func.set_arg(1, &d_out)?;
            func.set_arg(2, &(n as u32))?;
            func.set_arg(3, &self.gtable)?;
        }
        self.launch(&func, n, LOCAL_SIZE)?;

        let mut out = vec![0u8; n * 20];
        self.read(&d_out, &mut out)?;
        Ok(out.chunks_exact(20).map(|c| c.try_into().unwrap()).collect())
    }

    /// One 32-byte big-endian private key in -> 33-byte compressed pubkey out.
    fn pubkey_batch(&self, privs: &[[u8; 32]]) -> Result<Vec<[u8; 33]>> {
        let n = privs.len();
        let flat: Vec<u8> = privs.iter().flatten().copied().collect();
        let d_privs = self.buffer_from_slice(&flat, CL_MEM_READ_ONLY)?;
        let d_out = self.new_buffer::<u8>(n * 33, CL_MEM_WRITE_ONLY)?;

        let func = Kernel::create(&self.program, "k_pubkey")?;
        unsafe {
            func.set_arg(0, &d_privs)?;
            func.set_arg(1, &d_out)?;
            func.set_arg(2, &(n as u32))?;
            func.set_arg(3, &self.gtable)?;
        }
        self.launch(&func, n, LOCAL_SIZE)?;

        let mut out = vec![0u8; n * 33];
        self.read(&d_out, &mut out)?;
        Ok(out.chunks_exact(33).map(|c| c.try_into().unwrap()).collect())
    }

    /// One 32-byte big-endian private key in -> 64-byte uncompressed X||Y out.
    fn pubkey_xy_batch(&self, privs: &[[u8; 32]]) -> Result<Vec<[u8; 64]>> {
        let n = privs.len();
        let flat: Vec<u8> = privs.iter().flatten().copied().collect();
        let d_privs = self.buffer_from_slice(&flat, CL_MEM_READ_ONLY)?;
        let d_out = self.new_buffer::<u8>(n * 64, CL_MEM_WRITE_ONLY)?;

        let func = Kernel::create(&self.program, "k_pubkey_xy")?;
        unsafe {
            func.set_arg(0, &d_privs)?;
            func.set_arg(1, &d_out)?;
            func.set_arg(2, &(n as u32))?;
            func.set_arg(3, &self.gtable)?;
        }
        self.launch(&func, n, LOCAL_SIZE)?;

        let mut out = vec![0u8; n * 64];
        self.read(&d_out, &mut out)?;
        Ok(out.chunks_exact(64).map(|c| c.try_into().unwrap()).collect())
    }

    /// (a + b) mod n per pair, all 32-byte big-endian.
    fn scalar_add_batch(&self, a: &[[u8; 32]], b: &[[u8; 32]]) -> Result<Vec<[u8; 32]>> {
        let n = a.len();
        let fa: Vec<u8> = a.iter().flatten().copied().collect();
        let fb: Vec<u8> = b.iter().flatten().copied().collect();
        let d_a = self.buffer_from_slice(&fa, CL_MEM_READ_ONLY)?;
        let d_b = self.buffer_from_slice(&fb, CL_MEM_READ_ONLY)?;
        let d_out = self.new_buffer::<u8>(n * 32, CL_MEM_WRITE_ONLY)?;

        let func = Kernel::create(&self.program, "k_scalar_add")?;
        unsafe {
            func.set_arg(0, &d_a)?;
            func.set_arg(1, &d_b)?;
            func.set_arg(2, &d_out)?;
            func.set_arg(3, &(n as u32))?;
        }
        self.launch(&func, n, LOCAL_SIZE)?;

        let mut out = vec![0u8; n * 32];
        self.read(&d_out, &mut out)?;
        Ok(out.chunks_exact(32).map(|c| c.try_into().unwrap()).collect())
    }
}

/// Packs variable-length byte vectors into a fixed-stride buffer plus lengths.
/// Returns (packed, lens, stride). Stride is at least 1 so device pointers stay valid.
fn pack(inputs: &[Vec<u8>]) -> (Vec<u8>, Vec<u32>, usize) {
    let n = inputs.len();
    let stride = inputs.iter().map(|m| m.len()).max().unwrap_or(0).max(1);
    let mut packed = vec![0u8; n * stride];
    let mut lens = vec![0u32; n];
    for (i, m) in inputs.iter().enumerate() {
        packed[i * stride..i * stride + m.len()].copy_from_slice(m);
        lens[i] = m.len() as u32;
    }
    (packed, lens, stride)
}

/// A wordlist prepared for the GPU: NFKD bytes packed at a fixed stride plus a
/// per-word byte length. BIP-39 wordlists are already NFKD-normalized, so the
/// canonical word strings can be used verbatim.
pub struct GpuWordlist {
    packed: Vec<u8>,
    lens: Vec<u8>,
    stride: usize,
}

impl GpuWordlist {
    pub fn new(words: &[&str]) -> Result<Self> {
        let stride = words.iter().map(|w| w.len()).max().unwrap_or(1).max(1);
        // Kernel's mnemonic buffer is 512 bytes: 12 words + 11 spaces must fit.
        anyhow::ensure!(
            stride * 12 + 11 <= 512,
            "wordlist word too long for GPU mnemonic buffer (stride {stride})"
        );
        anyhow::ensure!(stride < 256, "word length exceeds u8 length field");
        let mut packed = vec![0u8; words.len() * stride];
        let mut lens = vec![0u8; words.len()];
        for (i, w) in words.iter().enumerate() {
            let b = w.as_bytes();
            packed[i * stride..i * stride + b.len()].copy_from_slice(b);
            lens[i] = b.len() as u8;
        }
        Ok(Self { packed, lens, stride })
    }
}

/// Result of a successful GPU search: the global candidate index and its 12
/// word indices.
pub struct SearchHit {
    pub global_index: usize,
    pub indices: [u16; 12],
}

impl Gpu {
    /// Searches `candidates` (an iterator of 12 word-index arrays) for one whose
    /// derived Ethereum address equals `target`. Streams in batches so memory
    /// stays flat.
    pub fn search(
        &self,
        candidates: impl Iterator<Item = [u16; 12]> + Send + 'static,
        wordlist: &GpuWordlist,
        target: &[u8; 20],
        batch_size: usize,
        local_size: usize,
    ) -> Result<Option<SearchHit>> {
        anyhow::ensure!(batch_size > 0, "GPU batch size must be greater than zero");
        anyhow::ensure!(
            batch_size <= u32::MAX as usize,
            "GPU batch size exceeds the OpenCL kernel's u32 limit"
        );
        anyhow::ensure!(local_size > 0, "OpenCL work-group size must be greater than zero");
        let candidate_words = batch_size
            .checked_mul(12)
            .context("GPU candidate-buffer size overflow")?;

        let d_wordlist = self.buffer_from_slice(&wordlist.packed, CL_MEM_READ_ONLY)?;
        let d_lens = self.buffer_from_slice(&wordlist.lens, CL_MEM_READ_ONLY)?;
        let d_target = self.buffer_from_slice(target, CL_MEM_READ_ONLY)?;
        // Host scratch for zeroing device flags/counters. Lives for the whole
        // search so non-blocking writes can refer to it safely.
        let zero_host = [0u32];

        // Two slots ping-pong between the two queues. Each slot owns its device
        // buffers and kernel objects so a batch in flight never shares memory
        // or kernel-argument state with the batch overlapping it from the other
        // queue.
        struct Slot {
            d_cand: Buffer<u16>,
            d_surv: Buffer<u32>,
            d_counter: Buffer<u32>,
            d_found_flag: Buffer<u32>,
            d_found_idx: Buffer<u32>,
            k_filter: Kernel,
            k_pipe: Kernel,
            buf: Option<Vec<u16>>,
            n: usize,
            count: [u32; 1],
            flag: [u32; 1],
            ev_rc: Option<opencl3::event::Event>,
            ev_rf: Option<opencl3::event::Event>,
            piped: bool,
        }

        fn make_slot(g: &Gpu, batch_size: usize, candidate_words: usize) -> Result<Slot> {
            Ok(Slot {
                d_cand: g.new_buffer::<u16>(candidate_words, CL_MEM_READ_WRITE)?,
                d_surv: g.new_buffer::<u32>(batch_size, CL_MEM_READ_WRITE)?,
                d_counter: g.new_buffer::<u32>(1, CL_MEM_READ_WRITE)?,
                d_found_flag: g.new_buffer::<u32>(1, CL_MEM_READ_WRITE)?,
                d_found_idx: g.new_buffer::<u32>(1, CL_MEM_READ_WRITE)?,
                k_filter: Kernel::create(&g.program, "k_filter")?,
                k_pipe: Kernel::create(&g.program, "k_pipeline")?,
                buf: None,
                n: 0,
                count: [0u32; 1],
                flag: [0u32; 1],
                ev_rc: None,
                ev_rf: None,
                piped: false,
            })
        }

        // Generating a batch takes ~10-15% of the time the GPU spends on it, and
        // it used to run between launches with the device idle. A producer thread
        // fills the next batch while the current ones are in flight; buffers cycle
        // back over `empty` so nothing is reallocated. The pool depth (2) matches
        // the two double-buffered slots.
        let (full_tx, full_rx) = std::sync::mpsc::sync_channel::<Vec<u16>>(1);
        let (empty_tx, empty_rx) = std::sync::mpsc::channel::<Vec<u16>>();
        for _ in 0..2 {
            let _ = empty_tx.send(Vec::with_capacity(candidate_words));
        }
        let producer = std::thread::spawn(move || {
            let mut it = candidates;
            while let Ok(mut buf) = empty_rx.recv() {
                buf.clear();
                for _ in 0..batch_size {
                    match it.next() {
                        Some(c) => buf.extend_from_slice(&c),
                        None => break,
                    }
                }
                let last = buf.len() < candidate_words;
                // A send error just means the consumer stopped (hit found).
                if full_tx.send(buf).is_err() || last {
                    return;
                }
            }
        });
        let mut slots = [
            make_slot(self, batch_size, candidate_words)?,
            make_slot(self, batch_size, candidate_words)?,
        ];
        let filter_limit = slots[0].k_filter.get_work_group_size(self.device.id())?;
        let pipeline_limit = slots[0].k_pipe.get_work_group_size(self.device.id())?;
        let work_group_limit = filter_limit.min(pipeline_limit);
        anyhow::ensure!(
            local_size <= work_group_limit,
            "OpenCL work-group size {local_size} exceeds this kernel's limit ({work_group_limit})"
        );
        if std::env::var_os("WORDS_BREAKER_DIAGNOSTICS").is_some() {
            let preferred = slots[0]
                .k_pipe
                .get_work_group_size_multiple(self.device.id())?;
            let private_bytes = slots[0].k_pipe.get_private_mem_size(self.device.id())?;
            eprintln!(
                "OpenCL search config: batch={batch_size}, local={local_size}, \
                 kernel_limit={work_group_limit}, preferred_multiple={preferred}, \
                 pipeline_private_mem={private_bytes} B/work-item"
            );
        }
        // Pipeline register counters: `head` = next batch to fill into a slot,
        // `mid` = next batch to promote to the pipeline queue, `tail` = next
        // batch to drain. At most two batches are in flight (one per slot).
        let mut head = 0usize;
        let mut mid = 0usize;
        let mut tail = 0usize;
        let mut producer_done = false;
        let mut seen = 0usize; // candidates in drained batches

        let result = 'search: loop {
            // 1) Refill: enqueue the next batch's cheap pass (H2D writes + filter
            //    + count readback) on the filter queue, if a slot is free.
            if head - tail < 2 && !producer_done {
                let s = head % 2;
                match full_rx.recv() {
                    Ok(buf) if !buf.is_empty() => {
                        let n = buf.len() / 12;
                        slots[s].buf = Some(buf);
                        slots[s].n = n;
                        let slot = &mut slots[s];
                        let data = slot.buf.as_deref().unwrap();
                        let _ew = unsafe {
                            self.queue.enqueue_write_buffer(
                                &mut slot.d_cand,
                                0, // non-blocking; data lives in slot.buf
                                0,
                                data,
                                &[],
                            )?
                        };
                        let _ec = unsafe {
                            self.queue.enqueue_write_buffer(
                                &mut slot.d_counter,
                                0,
                                0,
                                &zero_host,
                                &[],
                            )?
                        };
                        let _ef0 = unsafe {
                            self.queue.enqueue_write_buffer(
                                &mut slot.d_found_flag,
                                0,
                                0,
                                &zero_host,
                                &[],
                            )?
                        };
                        unsafe {
                            slot.k_filter.set_arg(0, &slot.d_cand)?;
                            slot.k_filter.set_arg(1, &(n as u32))?;
                            slot.k_filter.set_arg(2, &slot.d_surv)?;
                            slot.k_filter.set_arg(3, &slot.d_counter)?;
                        }
                        let global = n.div_ceil(local_size).max(1) * local_size;
                        let garr = [global];
                        let larr = [local_size];
                        let ev_f = unsafe {
                            self.queue.enqueue_nd_range_kernel(
                                slot.k_filter.get(),
                                1,
                                std::ptr::null(),
                                garr.as_ptr(),
                                larr.as_ptr(),
                                &[],
                            )?
                        };
                        let waits = [ev_f.get()];
                        let ev_rc = unsafe {
                            self.queue.enqueue_read_buffer(
                                &mut slot.d_counter,
                                0,
                                0,
                                &mut slot.count,
                                &waits,
                            )?
                        };
                        slot.ev_rc = Some(ev_rc);
                        head += 1;
                    }
                    // Producer finished or sent the (empty) final marker.
                    _ => producer_done = true,
                }
                continue;
            }
// 2) Promote: promote the oldest batch to the heavy pipeline queue.
            if mid < head {
                let s = mid % 2;
                let slot = &mut slots[s];
                let rc_ev = slot.ev_rc.take();
                if let Some(ev) = &rc_ev {
                    ev.wait()?;
                }
                let count = slot.count[0] as usize;
                if count > 0 {
                    unsafe {
                        slot.k_pipe.set_arg(0, &slot.d_cand)?;
                        slot.k_pipe.set_arg(1, &slot.d_surv)?;
                        slot.k_pipe.set_arg(2, &(count as u32))?;
                        slot.k_pipe.set_arg(3, &d_wordlist)?;
                        slot.k_pipe.set_arg(4, &d_lens)?;
                        slot.k_pipe.set_arg(5, &(wordlist.stride as u32))?;
                        slot.k_pipe.set_arg(6, &d_target)?;
                        slot.k_pipe.set_arg(7, &slot.d_found_flag)?;
                        slot.k_pipe.set_arg(8, &slot.d_found_idx)?;
                        slot.k_pipe.set_arg(9, &self.gtable)?;
                    }
                    let global = count.div_ceil(local_size).max(1) * local_size;
                    let garr = [global];
                    let larr = [local_size];
                    let waits = rc_ev.iter().map(|e| e.get()).collect::<Vec<_>>();
                    let ev_p = unsafe {
                        self.pipeline_queue.enqueue_nd_range_kernel(
                            slot.k_pipe.get(),
                            1,
                            std::ptr::null(),
                            garr.as_ptr(),
                            larr.as_ptr(),
                            &waits,
                        )?
                    };
                    let pwaits = [ev_p.get()];
                    let ev_rf = unsafe {
                        self.pipeline_queue.enqueue_read_buffer(
                            &mut slot.d_found_flag,
                            0,
                            0,
                            &mut slot.flag,
                            &pwaits,
                        )?
                    };
                    slot.ev_rf = Some(ev_rf);
                    slot.piped = true;
                } else {
                    slot.piped = false;
                    slot.ev_rf = None;
                }
                mid += 1;
                continue;
            }

            // 3) Drain: reap the oldest batch once its found-flag readback is in.
            if tail < mid {
                let s = tail % 2;
                let mut hit: Option<SearchHit> = None;
                {
                    let slot = &mut slots[s];
                    if slot.piped {
                        if let Some(ev) = slot.ev_rf.take() {
                            ev.wait()?;
                        }
                    }
                    if slot.piped && slot.flag[0] != 0 {
                        let mut idx = [0u32; 1];
                        let ev = unsafe {
                            self.pipeline_queue.enqueue_read_buffer(
                                &mut slot.d_found_idx,
                                0,
                                0,
                                &mut idx,
                                &[],
                            )?
                        };
                        ev.wait()?;
                        let local = idx[0] as usize;
                        if let Some(buf) = &slot.buf {
                            if local < slot.n {
                                let mut indices = [0u16; 12];
                                let base = local * 12;
                                indices.copy_from_slice(&buf[base..base + 12]);
                                hit = Some(SearchHit {
                                    global_index: seen + local,
                                    indices,
                                });
                            }
                        }
                    }
                }
                // Hand the buffer back to the producer pool.
                if let Some(buf) = slots[s].buf.take() {
                    let _ = empty_tx.send(buf);
                }
                seen += slots[s].n;
                slots[s].n = 0;
                slots[s].count = [0u32; 1];
                slots[s].flag = [0u32; 1];
                slots[s].ev_rf = None;
                slots[s].piped = false;

                if let Some(h) = hit {
                    break 'search Some(h);
                }
                println!("Checked {} candidates...", crate::format_number(seen));
                use std::io::Write;
                let _ = std::io::stdout().flush();
                tail += 1;
                continue;
            }

            // 4) All batches processed.
            if producer_done && head == tail {
                break 'search None;
            }
        };

        // Break out of the loop early (a hit) and the producer may be parked in
        // either direction, so both of its peers have to go before the join:
        // dropping the receiver fails its pending send, dropping the sender ends
        // its wait for the next empty buffer.
        drop(full_rx);
        drop(empty_tx);
        let _ = producer.join();
        Ok(result)
    }

    /// Sweeps (batch, local) combinations in a single process, measuring the
    /// steady-state kernel throughput. Warm-up is the first completed batch of
    /// every configuration. Returns the best rate (candidates per second).
    pub fn bench(
        &self,
        wordlist: &GpuWordlist,
        target: &[u8; 20],
        configs: &[(usize, usize)],
        seconds: f64,
    ) -> Result<usize> {
        let d_wordlist = self.buffer_from_slice(&wordlist.packed, CL_MEM_READ_ONLY)?;
        let d_lens = self.buffer_from_slice(&wordlist.lens, CL_MEM_READ_ONLY)?;
        let d_target = self.buffer_from_slice(target, CL_MEM_READ_ONLY)?;
        let k_filter = Kernel::create(&self.program, "k_filter")?;
        let k_pipe = Kernel::create(&self.program, "k_pipeline")?;
        let limit = k_filter
            .get_work_group_size(self.device.id())?
            .min(k_pipe.get_work_group_size(self.device.id())?);
        let survivor_cap = configs.iter().map(|(b, _)| *b).max().unwrap_or(1);
        let d_surv = self.new_buffer::<u32>(survivor_cap, CL_MEM_READ_WRITE)?;
        let mut d_counter = self.buffer_from_slice(&[0u32], CL_MEM_READ_WRITE)?;
        let mut d_found_flag = self.buffer_from_slice(&[0u32], CL_MEM_READ_WRITE)?;
        let d_found_idx = self.buffer_from_slice(&[0u32], CL_MEM_READ_WRITE)?;
        let mut d_cand = self.new_buffer::<u16>(survivor_cap * 12, CL_MEM_READ_WRITE)?;
        let zero_host = [0u32];

        // Synthetic input: 100k candidates with u16 word indices in [0, 2047).
        // The exact contents do not matter for the throughput measurement; we
        // only need every thread to do real cryptographic work without ever
        // triggering a hit.
        let n_in = 100_000usize;
        let mut host: Vec<u16> = Vec::with_capacity(n_in * 12);
        let mut s: u32 = 0xc0ffee_u32;
        for _ in 0..n_in * 12 {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            host.push(((s >> 16) as u16) & 0x07ff);
        }
        self.write(&mut d_cand, &host)?;

        let mut rate_max = 0usize;
        for &(batch_size, local_size) in configs {
            if local_size > limit || batch_size == 0 || batch_size > survivor_cap {
                println!("  skip batch={batch_size} local={local_size}: out of range");
                continue;
            }
            unsafe {
                k_filter.set_arg(0, &d_cand)?;
                k_filter.set_arg(1, &(n_in as u32))?;
                k_filter.set_arg(2, &d_surv)?;
                k_filter.set_arg(3, &d_counter)?;
                k_pipe.set_arg(0, &d_cand)?;
                k_pipe.set_arg(1, &d_surv)?;
                k_pipe.set_arg(2, &(n_in as u32))?;
                k_pipe.set_arg(3, &d_wordlist)?;
                k_pipe.set_arg(4, &d_lens)?;
                k_pipe.set_arg(5, &(wordlist.stride as u32))?;
                k_pipe.set_arg(6, &d_target)?;
                k_pipe.set_arg(7, &d_found_flag)?;
                k_pipe.set_arg(8, &d_found_idx)?;
                k_pipe.set_arg(9, &self.gtable)?;
            }

            let global = n_in.div_ceil(local_size).max(1) * local_size;
            let garr = [global];
            let larr = [local_size];
            // Warm-up batch: primes the constant caches and any first-touch overhead.
            unsafe {
                self.queue.enqueue_write_buffer(&mut d_counter, 0, 0, &zero_host, &[])?;
                self.queue.enqueue_write_buffer(&mut d_found_flag, 0, 0, &zero_host, &[])?;
                self.queue.enqueue_nd_range_kernel(
                    k_filter.get(),
                    1,
                    std::ptr::null(),
                    garr.as_ptr(),
                    larr.as_ptr(),
                    &[],
                )?;
                self.queue.enqueue_nd_range_kernel(
                    k_pipe.get(),
                    1,
                    std::ptr::null(),
                    garr.as_ptr(),
                    larr.as_ptr(),
                    &[],
                )?;
            }
            self.queue.finish()?;

            let deadline = std::time::Instant::now() + std::time::Duration::from_secs_f64(seconds);
            let t_start = std::time::Instant::now();
            let mut total: u64 = 0;
            let mut batches: u32 = 0;
            loop {
                if std::time::Instant::now() >= deadline {
                    break;
                }
                unsafe {
                    self.queue.enqueue_write_buffer(&mut d_counter, 0, 0, &zero_host, &[])?;
                    self.queue.enqueue_write_buffer(&mut d_found_flag, 0, 0, &zero_host, &[])?;
                    self.queue.enqueue_nd_range_kernel(
                        k_filter.get(),
                        1,
                        std::ptr::null(),
                        garr.as_ptr(),
                        larr.as_ptr(),
                        &[],
                    )?;
                    self.queue.enqueue_nd_range_kernel(
                        k_pipe.get(),
                        1,
                        std::ptr::null(),
                        garr.as_ptr(),
                        larr.as_ptr(),
                        &[],
                    )?;
                }
                self.queue.finish()?;
                total = total.wrapping_add(n_in as u64);
                batches += 1;
            }
            let elapsed = t_start.elapsed().as_secs_f64();
            let rate = if elapsed > 0.0 {
                (total as f64 / elapsed) as usize
            } else {
                0
            };
            println!(
                "  batch={batch_size:>7} local={local_size:>3} -> {batches:>3} batches, {rate:>10} cand/s"
            );
            if rate > rate_max {
                rate_max = rate;
            }
        }
        Ok(rate_max)
    }
}

/// Runs all primitive selftests, printing PASS/FAIL per primitive. Returns
/// `Ok(true)` iff every check passed.
pub fn run_selftest() -> Result<bool> {
    use bitcoin::hashes::{sha256, sha512, Hash};

    let gpu = Gpu::new()?;
    let mut all_ok = true;

    // Messages chosen to exercise: empty, short, multi-block boundaries.
    let mut msgs: Vec<Vec<u8>> = vec![
        vec![],
        b"abc".to_vec(),
        b"message digest".to_vec(),
        b"The quick brown fox jumps over the lazy dog".to_vec(),
        vec![0x61u8; 55], // one-block-1-byte boundary for sha256
    ];
    msgs.push(vec![0x5au8; 64]);
    msgs.push(vec![0xa5u8; 119]);
    // Keccak's rate is 136 bytes, so straddle it in both directions: 135 leaves
    // exactly one byte of padding (where the two pad marks collide into 0x81),
    // 136 forces an entire extra padding block, and 137 spills into a second.
    msgs.push(vec![0x61u8; 135]);
    msgs.push(vec![0x61u8; 136]);
    msgs.push(vec![0x61u8; 137]);
    msgs.push(vec![0u8; 200]);

    // --- SHA-256 ---
    let got = gpu.hash_batch("k_sha256", &msgs, 32)?;
    let sha256_ok = msgs.iter().zip(&got).all(|(m, g)| {
        let want = sha256::Hash::hash(m).to_byte_array();
        g.as_slice() == want
    });
    report("SHA-256", sha256_ok, &mut all_ok);

    // --- SHA-512 ---
    let got = gpu.hash_batch("k_sha512", &msgs, 64)?;
    let sha512_ok = msgs.iter().zip(&got).all(|(m, g)| {
        let want = sha512::Hash::hash(m).to_byte_array();
        g.as_slice() == want
    });
    report("SHA-512", sha512_ok, &mut all_ok);

    // --- Keccak-256 (vs the sha3 crate, plus a hardcoded known-answer test) ---
    let got = gpu.hash_batch("k_keccak256", &msgs, 32)?;
    let mut keccak_ok = msgs
        .iter()
        .zip(&got)
        .all(|(m, g)| g.as_slice() == crate::eth::keccak256(m));
    // Independent of both implementations: if the sha3 crate were somehow the
    // NIST variant, every comparison above would still agree while every digest
    // was wrong. These two constants are the published Keccak-256 vectors.
    keccak_ok &= hex::encode(&got[0]) == "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470";
    keccak_ok &= hex::encode(&got[1]) == "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45";
    report("Keccak-256 (incl. rate-boundary + KAT)", keccak_ok, &mut all_ok);

    // --- HMAC-SHA512 (vs bitcoin_hashes HmacEngine) ---
    // Keys chosen to cross the 128-byte block boundary (short, exactly-block, oversized).
    let hkeys: Vec<Vec<u8>> = vec![
        b"key".to_vec(),
        b"Bitcoin seed".to_vec(),
        vec![0x0bu8; 20],
        vec![0xaau8; 131], // > block size -> key gets hashed first
    ];
    let hmsgs: Vec<Vec<u8>> = vec![
        b"The quick brown fox jumps over the lazy dog".to_vec(),
        vec![0x00u8; 64],
        b"Hi There".to_vec(),
        vec![0xddu8; 200],
    ];
    let got = gpu.hmac_batch(&hkeys, &hmsgs)?;
    let hmac_ok = hkeys.iter().zip(&hmsgs).zip(&got).all(|((k, m), g)| {
        use bitcoin::hashes::{Hmac, HmacEngine};
        use bitcoin::hashes::HashEngine;
        let mut eng = HmacEngine::<sha512::Hash>::new(k);
        eng.input(m);
        let want = Hmac::<sha512::Hash>::from_engine(eng).to_byte_array();
        g.as_slice() == want
    });
    report("HMAC-SHA512", hmac_ok, &mut all_ok);

    // --- PBKDF2-HMAC-SHA512 / BIP-39 seed (vs bip39 crate Mnemonic::to_seed) ---
    use bip39::{Language, Mnemonic};
    let phrases = [
        "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about",
        "legal winner thank year wave sausage worth useful legal winner thank yellow",
        "letter advice cage absurd amount doctor acoustic avoid letter advice cage above",
    ];
    let mut pws = Vec::new();
    let mut salts = Vec::new();
    let mut want_seeds = Vec::new();
    for p in phrases {
        let m = Mnemonic::parse_in_normalized(Language::English, p)?;
        pws.push(m.to_string().into_bytes());
        salts.push(b"mnemonic".to_vec());
        want_seeds.push(m.to_seed("").to_vec());
    }
    // A Japanese mnemonic is ~220 bytes of UTF-8, past HMAC's 128-byte block, so
    // the key gets hashed before use. English phrases never reach that path.
    // Built from entropy rather than a literal: the wordlist is NFKD, in which
    // dakuten are separate code points, so a composed literal would not match.
    let m = Mnemonic::from_entropy_in(Language::Japanese, &[0u8; 16])?;
    let jp_bytes = m.to_string().into_bytes();
    anyhow::ensure!(
        jp_bytes.len() > 128,
        "expected the Japanese mnemonic to exceed one HMAC block, got {}",
        jp_bytes.len()
    );
    pws.push(jp_bytes);
    salts.push(b"mnemonic".to_vec());
    want_seeds.push(m.to_seed("").to_vec());

    let got = gpu.pbkdf2_batch(&pws, &salts, 2048)?;
    let pbkdf2_ok = got.iter().zip(&want_seeds).all(|(g, w)| g == w);
    report("PBKDF2-HMAC-SHA512 / BIP-39 seed", pbkdf2_ok, &mut all_ok);

    // --- secp256k1: priv -> compressed pubkey (vs secp256k1 crate) ---
    use bitcoin::secp256k1::{PublicKey, Scalar, Secp256k1, SecretKey};
    let secp = Secp256k1::new();
    let mut rng = SplitMix64::new(0x9E3779B97F4A7C15);
    let mut privs: Vec<[u8; 32]> = Vec::new();
    // Anchor with priv == 1 (pubkey == G) for a deterministic sanity point.
    let mut one = [0u8; 32];
    one[31] = 1;
    privs.push(one);
    while privs.len() < 512 {
        let cand = rng.fill32();
        if SecretKey::from_slice(&cand).is_ok() {
            privs.push(cand);
        }
    }
    let got = gpu.pubkey_batch(&privs)?;
    let pubkey_ok = privs.iter().zip(&got).all(|(p, g)| {
        let sk = SecretKey::from_slice(p).unwrap();
        let want = PublicKey::from_secret_key(&secp, &sk).serialize();
        g == &want
    });
    report("secp256k1 priv->compressed pubkey (512 keys)", pubkey_ok, &mut all_ok);

    // --- secp256k1: priv -> uncompressed X||Y ---
    // The compressed form above keeps only the parity of Y, so it cannot catch a
    // Y coordinate left in a non-canonical (unreduced) representation. Ethereum
    // hashes all 32 bytes of Y, so every bit of it has to be checked.
    let got = gpu.pubkey_xy_batch(&privs)?;
    let pubkey_xy_ok = privs.iter().zip(&got).all(|(p, g)| {
        let sk = SecretKey::from_slice(p).unwrap();
        let want = PublicKey::from_secret_key(&secp, &sk).serialize_uncompressed();
        g[..] == want[1..] // drop the 0x04 prefix, which Ethereum does not hash
    });
    report("secp256k1 priv->uncompressed X||Y (512 keys)", pubkey_xy_ok, &mut all_ok);

    // --- scalar add mod n (vs SecretKey::add_tweak) ---
    let mut avec: Vec<[u8; 32]> = Vec::new();
    let mut bvec: Vec<[u8; 32]> = Vec::new();
    let mut want_sum: Vec<[u8; 32]> = Vec::new();
    while avec.len() < 256 {
        let a = rng.fill32();
        let b = rng.fill32();
        let (sk, tw) = match (SecretKey::from_slice(&a), Scalar::from_be_bytes(b)) {
            (Ok(s), Ok(t)) => (s, t),
            _ => continue,
        };
        let sum = match sk.add_tweak(&tw) {
            Ok(s) => s,
            Err(_) => continue, // result was zero; vanishingly rare
        };
        avec.push(a);
        bvec.push(b);
        want_sum.push(sum.secret_bytes());
    }
    let got = gpu.scalar_add_batch(&avec, &bvec)?;
    let addn_ok = got.iter().zip(&want_sum).all(|(g, w)| g == w);
    report("secp256k1 scalar add mod n (256 pairs)", addn_ok, &mut all_ok);

    // --- BIP32 m/44'/60'/0'/0/0 seed -> Ethereum address (vs the CPU path) ---
    use bitcoin::bip32::DerivationPath;
    let path: DerivationPath = crate::eth::ETH_PATH.parse()?;
    let mut seeds: Vec<[u8; 64]> = Vec::new();
    let mut want_addr: Vec<[u8; 20]> = Vec::new();
    for p in phrases {
        let m = Mnemonic::parse_in_normalized(Language::English, p)?;
        let seed = m.to_seed("");
        want_addr.push(crate::eth::address_from_seed(&secp, &path, &seed)?);
        seeds.push(seed);
    }
    let got = gpu.seed_to_eth_batch(&seeds)?;
    let bip32_ok = got.iter().zip(&want_addr).all(|(g, w)| g == w);
    report("BIP32 m/44'/60'/0'/0/0 seed->ETH address", bip32_ok, &mut all_ok);

    // The CPU reference above and the GPU share no code, but they do share this
    // author's reading of BIP-44. Anchor the whole chain to a value produced
    // outside both: the canonical BIP-39 test mnemonic's first MetaMask account.
    let kat_seed = Mnemonic::parse_in_normalized(Language::English, phrases[0])?.to_seed("");
    let kat = gpu.seed_to_eth_batch(&[kat_seed])?;
    let kat_ok = crate::eth::to_eip55(&kat[0]) == "0x9858EfFD232B4033E47d90003D41EC34EcaEda94";
    report("BIP32 known-answer ('abandon...about' -> MetaMask #0)", kat_ok, &mut all_ok);

    Ok(all_ok)
}

/// Minimal SplitMix64 PRNG — deterministic test inputs without a rand dependency.
struct SplitMix64 {
    state: u64,
}
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn fill32(&mut self) -> [u8; 32] {
        let mut out = [0u8; 32];
        for chunk in out.chunks_mut(8) {
            chunk.copy_from_slice(&self.next_u64().to_be_bytes());
        }
        out
    }
}

fn report(name: &str, ok: bool, all_ok: &mut bool) {
    println!("  [{}] {}", if ok { "PASS" } else { "FAIL" }, name);
    *all_ok &= ok;
}
