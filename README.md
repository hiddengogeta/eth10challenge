# Words Breaker — OpenCL (Intel Arc)

> **Port for Intel Arc GPU of [Words Breaker](https://github.com/lmajowka/eth10challenge)**
> Original code: [github.com/lmajowka/eth10challenge](https://github.com/lmajowka/eth10challenge)
> (NVIDIA/CUDA version: [words-breaker-gpu](https://github.com/lmajowka/words-breaker-gpu))

---

## Credits & Acknowledgments

This repository is a **port of the code by Mestre Cacatal (Leonardo Majowka,
[@lmajowka](https://github.com/lmajowka))** from a **CUDA (NVIDIA)** backend to
**OpenCL (any GPU — tested on an Intel Arc B580)**, using the same
MetaMask default derivation `m/44'/60'/0'/0/0`.

Thanks, **Mestre Cacatal**, for making the code and the
`--pattern`/`--pool` methodology available, which made it possible to test the
theories of the **8.6 ETH (~R$ 100K)** challenge that has been sitting
untouched for 6 years! 🧙🏻‍♂️

- 🔗 Original code: https://github.com/lmajowka/eth10challenge
- 🎥 Mestre Cacatal's YouTube channel: https://www.youtube.com/@investidorint
- 💬 Hunters community: https://maestroapp.cloud/invite/22b0978e49d1c06ce733fc6ef5701500
- 🎮 Community Discord: https://discord.com/invite/ZFP4QrWney
- ⛏️ Bitcoin Puzzles: https://bitcoinpuzzles.io
- 📘 Facebook group: https://www.facebook.com/groups/1436355617246687

## 🤖 Built with AI assistance

This port was developed with the help of generative AI models running in
**Cline** (AI coding agent in the editor):

- 🧠 **GLM 5.3 Flash (xhigh)** — helped with the **CUDA → OpenCL** portability;
- 🧠 **DeepSeek V4 Flash (xhigh)** — helped reviewing and optimizing the code.

Every change was reviewed and validated with the `--selftest` before being published.

This port:
- Replaces the `kernels.cu` (CUDA) kernels with `src/ocl/kernels.cl` (OpenCL C),
  compiled by the driver at runtime — **no nvcc / CUDA toolkit required**;
- Was validated with `--selftest` (10/10 bit-exact primitives vs. CPU reference);
- Ran on an **Intel Arc B580 at ~2.57M candidates/s** steady-state with the
  256-GRF build (double-buffered queue pipeline, 4M-batch / 64-thread kernel
  config). Latest measurements and the per-config sweep are in
  [Performance](#performance).

---

# Words Breaker (Ethereum)

A command-line tool that attempts to recover a BIP-39 mnemonic seed phrase by testing permutations of 12 known words against a target Ethereum address.

## Use Case

If you have 12 BIP-39 mnemonic words but don't remember the correct order, this tool will brute-force permutations to find the combination that derives to your known Ethereum address.

Addresses are derived at **`m/44'/60'/0'/0/0`** — the first account of the default
Ethereum wallet (MetaMask, Ledger Live, Trust, Rabby, ...) — with no BIP-39
passphrase. If your wallet used a passphrase or a non-default account index,
this tool will not find it as-is.

## Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) (1.80 or later recommended)
- A **GPU with OpenCL 3.0** and the matching vendor driver — the search runs on the
  GPU by default. Developed against an **Intel Arc B580** (via `intel-compute-runtime`,
  which provides the `Intel(R) OpenCL Graphics` platform); any OpenCL 3.0 GPU with
  32-bit atomics should work, NVIDIA/AMD included. The kernels
  (`src/ocl/kernels.cl`) are compiled by the device's OpenCL compiler at startup —
  no `nvcc`, CUDA toolkit or build-time GPU toolchain is needed. If no OpenCL GPU
  is present at runtime, the tool falls back to the CPU automatically (or pass
  `--cpu`).

The original NVIDIA/CUDA version is preserved in `src/cuda/kernels.cu` for
reference; it is not compiled by this tree anymore.

### Performance

Measured on the **Intel Arc B580** (160 CUs, driver `intel-compute-runtime`
26.27), 20s steady-state per config (`wb_bench.sh <batch> <local> 20`).

| Config | Throughput (cand/s) |
|---|---:|
| 1M × 16 | 2,413,872 |
| 1M × 32 | 2,413,955 |
| 1M × 64 | 2,368,023 |
| 2M × 64 (default GRF) | 2,481,386 |
| 2M × 64 (256-GRF) | **2,574,425** |
| 4M × 64 (default GRF) | 2,507,867 |
| 4M × 64 (256-GRF) | **2,569,103** |
| 8M × 64 | 2,514,725 |

The hot path is k_pipeline (PBKDF2-HMAC-SHA512, 2048 iterations) — ~94% of
runtime. The remaining ~6% is the cheap k_filter pass (BIP-39 checksum). The
double-buffered queue pipeline (filter queue + heavy pipeline queue) keeps
the device busy while the host pre-fills the next candidate batch. The 256-GRF
option (`-cl-intel-256-GRF-per-thread`) is auto-detected at startup: the
program tries it first and falls back silently to the default on older
drivers, so the binary remains portable.

The rate is also bounded by the **k_pipeline private memory of ~64 KB per
work-item** (reported by the OpenCL driver). Reducing that is what would
unlock the next tier; the kernel is bit-exact, so any further optimization
has to keep `--selftest` and the BIP-39 known-answer test green.

### OpenCL library path

The OpenCL vendor library (`libOpenCL.so`) is located through the ICD loader,
which reads `/etc/OpenCL/vendors/*.icd` — no environment variables are needed
on a standard install.

## Building

### Windows

```powershell
cargo build --release
```

The binary will be located at `target\release\words-breaker.exe`.

### Linux / macOS

```bash
cargo build --release
```

The binary will be located at `target/release/words-breaker`.

## Usage

```
words-breaker <TARGET_ADDRESS> <WORD1> <WORD2> ... <WORD12> [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `TARGET_ADDRESS` | Target Ethereum address, 40 hex characters with an optional `0x` prefix. Mixed-case input is verified against its EIP-55 checksum, so a typo is rejected up front instead of after an exhaustive search. |
| `WORD1..WORDN` | 10, 11, or 12 BIP-39 words in any order. With 10 or 11 words, the missing word(s) are completed from the 2048-word BIP-39 list. |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--pattern` | | 12 space-separated slots, `?` for an unknown one. Non-`?` slots are pinned to that position. Replaces the positional word list. |
| `--pool` | | Words that fill the `?` slots, each used at most once. Fewer pool words than `?` slots means the leftovers are drawn from the fill set. |
| `--fill` | whole wordlist | Restricts what a leftover `?` slot may hold. Takes literal words and `prefix*` patterns, e.g. `--fill "d*,f*"`. Each leftover slot multiplies the search by this set's size, so this is the strongest lever available. |
| `-l, --language` | `english` | BIP-39 wordlist language |
| `-t, --threads` | `0` (all cores) | CPU threads (CPU path only) |
| `--cpu` | off | Force the CPU (rayon) search instead of the GPU |
| `--selftest` | | Verify each GPU crypto primitive against the CPU reference and exit |
| `-h, --help` | | Print help |
| `-V, --version` | | Print version |

The search runs on the **GPU by default**, streaming candidates in batches. Each
batch is filtered by BIP-39 checksum on the GPU (a cheap pass that keeps ~1/16 of
candidates), then only the survivors run the full seed/derivation/address
pipeline. Run `--selftest` first: it re-derives every GPU primitive
(SHA-256/512, Keccak-256, HMAC, PBKDF2, secp256k1, BIP32) against the CPU
reference, plus a known-answer anchor independent of both implementations, and
only a fully passing selftest makes a search result trustworthy.

**Supported languages:** `english`, `portuguese`, `spanish`, `french`, `italian`, `czech`, `korean`, `japanese`, `chinese-simplified`, `chinese-traditional`

### Examples

**Windows:**
```powershell
.\target\release\words-breaker.exe 0xc01B0dEFB2D8767F9C9A59EB464437e62428A31d scene spell mask private regret soda spike coconut any little december bronze
```

**Linux / macOS:**
```bash
./target/release/words-breaker 0xc01B0dEFB2D8767F9C9A59EB464437e62428A31d scene spell mask private regret soda spike coconut any little december bronze
```

That example is a throwaway mnemonic generated for testing, with its first two
words swapped. It resolves to `spell scene mask ...` at candidate index
39,916,800 (= 11!, the lexicographic rank of a single leading transposition).

**Known positions plus a word pool:**
```bash
./target/release/words-breaker 0x… \
  --pattern "dutch ? ? ? fog ? ? ? ? ? ? parrot" \
  --pool "fork fiber forest dinner goat seed key lake"
```

Three words are pinned to positions 1, 5 and 12; the eight pool words permute
over the nine open slots, and the ninth slot — whichever it turns out to be — is
drawn from the full 2048-word list. That is `9!/1! x 2048 = 743,178,240`
candidates. The tool prints this count before starting, so a miscounted pool is
caught in a second rather than an hour in.

**Verify the GPU implementation:**
```bash
./target/release/words-breaker --selftest
```

**With Portuguese wordlist:**
```bash
./target/release/words-breaker <TARGET_ADDRESS> bexiga bonde curativo nevoeiro mundial vareta urubu megafone cozinha livro surpresa senador -l portuguese
```

## How It Works

Every mode is one template: 12 slots, each either pinned to a word or open, plus
a pool that fills the open ones without replacement. Passing 12 loose words is
just the case of 12 open slots and a 12-word pool, so both modes share a single
enumerator (`src/candidates.rs`).

With `h` open slots, a pool of `p` and a fill set of `f`, the space is
`p!/(p-h)!` when `p >= h` (which pool words, in which order), and
`h!/(h-p)! x f^(h-p)` when `p < h` (place the pool, then draw the rest from the
fill set).

That last exponent is where searches live or die. Two unknown slots over the
whole list is `2048^2` = 4.2M times the placement count; the same two slots
restricted with `--fill "d*,f*"` is `218^2` = 47.5K times — an 88x cut. If you
know anything at all about the missing words, spend it here.

1. Streams that space as compact 12-index arrays — nothing is held fully in memory
2. On the GPU, each candidate is checksum-filtered, then survivors are derived:
   PBKDF2-HMAC-SHA512 seed → BIP32 `m/44'/60'/0'/0/0` → secp256k1 public key →
   `keccak256(X ‖ Y)`
3. The low 20 bytes of that digest are the address, compared against the target
4. Stops and outputs the correct phrase when a match is found

Note that BIP32's master-key HMAC uses the literal string `"Bitcoin seed"` for
every coin — that constant is fixed by BIP32 itself, not by Bitcoin. The only
thing separating an Ethereum account from a BIP-44 Bitcoin one here is the coin
type (`60'` vs `0'`) and the final hash.

## Performance Notes

- 12 words have 479,001,600 (12!) possible permutations; supplying 10 or 11 words
  multiplies this by up to 2048 per missing word
- The whole space is streamed and searched (there is no fixed permutation cap)
- Invalid BIP-39 checksums are filtered out cheaply before the expensive work,
  which removes 15/16 of candidates for the cost of one SHA-256

On an RTX 3050 the full 12! space takes 197 s (measured, exhaustive run), at
roughly 2.42M permutations/s (~151k full derivations/s after checksum
filtering).

The cost per surviving candidate is dominated by PBKDF2-HMAC-SHA512, whose 2048
iterations are fixed by the BIP-39 spec — measured at ~92% of GPU time, with all
of BIP32 + secp256k1 + the address hash making up the rest. The address hash is
the smallest part of that: one Keccak-f[1600] permutation against PBKDF2's 4096
SHA-512 compressions. Three things matter most:

1. **PBKDF2 runs entirely in registers.** With `dkLen == 64` every one of the
   4096 SHA-512 compressions per candidate is a single block of fixed layout
   (`W[8] = 0x80…`, `W[9..14] = 0`, `W[15] = 1536`), so the loop carries its
   state and message as `u64` registers instead of streaming bytes through a
   context struct in local memory. This was worth ~6.5x on its own; a generic
   streaming SHA-512 spends more time on local-memory traffic than on hashing.
2. **All three scalar multiplications are fixed-base.** Both non-hardened BIP32
   levels and the final public key are `k*G`, so a precomputed table of multiples
   of G (4-bit windows, built once at startup by `k_init_gtable`) replaces
   double-and-add: 64 point additions and no doublings.
3. **Field inversion uses an addition chain** (255 squarings + 15 multiplies)
   rather than a full exponentiation by `p-2`.

Candidate batches are generated on a producer thread so the CPU-side permutation
stream overlaps with the GPU work rather than running between launches.

## License

MIT
