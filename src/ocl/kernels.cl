// OpenCL device implementation of the words-breaker hot path.
//
// Direct port of src/cuda/kernels.cu (CUDA) to OpenCL C, so the search can run
// on non-NVIDIA GPUs — developed against an Intel Arc B580 (needs OpenCL 3.0
// with the generic address space and 32-bit atomics; both are standard on
// Intel's compute runtime). Each primitive is verified against the CPU crates
// via --selftest (see src/gpu.rs): SHA-256, SHA-512, Keccak-256, HMAC-SHA512,
// PBKDF2, secp256k1 (priv->pubkey, compressed and uncompressed, and scalar add
// mod n), and BIP32 m/44'/60'/0'/0/0 seed->Ethereum address. The full search is
// split into k_filter (cheap BIP-39 checksum, compacting survivors) and
// k_pipeline (heavy derivation), so the heavy pass has no divergence.
//
// All multi-byte values follow the relevant standard's byte order (big-endian
// for SHA, little-endian for Keccak lanes), independent of GPU endianness, so
// the code is endianness-correct by construction.
//
// Differences from the CUDA original:
//   * unqualified pointer parameters rely on the OpenCL *generic address
//     space*, so they accept both private and __global pointers — the same
//     call sites as the CUDA original compile unchanged;
//   * the fixed-base table of multiples of G is an explicit __global kernel
//     argument threaded through the derivation instead of a device-side
//     global symbol (OpenCL has no __device__ globals);
//   * the 128-bit carry arithmetic is rebuilt on adc64/sbb64/mac64 + mul_hi;
//   * atomicCAS/atomicAdd become atomic_cmpxchg/atomic_add.

typedef uchar  u8;
typedef uint   u32;
typedef ulong  u64;

// 64-bit add with carry: *c carries 0/1 in and out.
static inline u64 adc64(u64 a, u64 b, u64* c) {
    u64 t  = a + b;
    u64 c1 = t < a;
    u64 t2 = t + *c;
    u64 c2 = t2 < t;
    *c = c1 + c2;
    return t2;
}

// 64-bit subtract with borrow: *br borrows 0/1 in and out.
static inline u64 sbb64(u64 a, u64 b, u64* br) {
    u64 t  = a - *br;
    u64 b1 = a < *br;
    u64 t2 = t - b;
    u64 b2 = t < b;
    *br = b1 + b2;
    return t2;
}

// (a * b + acc + *carry) -> low half; *carry <- high half.
static inline u64 mac64(u64 a, u64 b, u64 acc, u64* carry) {
    u64 lo = a * b;
    u64 hi = mul_hi(a, b);
    u64 t  = lo + acc;
    u64 c1 = t < lo;
    u64 t2 = t + *carry;
    u64 c2 = t2 < t;
    *carry = hi + c1 + c2;
    return t2;
}

static inline u32 rotr32(u32 x, u32 n) { return (x >> n) | (x << (32 - n)); }
static inline u64 rotr64(u64 x, u32 n) { return (x >> n) | (x << (64 - n)); }
// Guarded so a rotation of 0 (lane A[0], whose rho offset is 0) is not UB.
static inline u64 rotl64(u64 x, u32 n) { return n ? ((x << n) | (x >> (64 - n))) : x; }

static inline void dmemcpy(u8* dst, const u8* src, u32 n) {
    for (u32 i = 0; i < n; i++) dst[i] = src[i];
}

// ===========================================================================
// SHA-256 (FIPS 180-4) — streaming
// ===========================================================================

__constant u32 K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
};

typedef struct { u32 h[8]; u64 total; u8 buf[64]; u32 n; } sha256_ctx;

void sha256_init(sha256_ctx* c) {
    c->h[0]=0x6a09e667; c->h[1]=0xbb67ae85; c->h[2]=0x3c6ef372; c->h[3]=0xa54ff53a;
    c->h[4]=0x510e527f; c->h[5]=0x9b05688c; c->h[6]=0x1f83d9ab; c->h[7]=0x5be0cd19;
    c->total = 0; c->n = 0;
}

void sha256_transform(u32 h[8], const u8 block[64]) {
    // 16-word rolling message schedule: w[(i)&15] holds w[i-16] until overwritten,
    // so only 16 words are kept live (vs 64), cutting per-thread register/local
    // memory pressure. With #pragma unroll the (i&15) indices fold to constants.
    u32 w[16];
    #pragma unroll
    for (int i = 0; i < 16; i++)
        w[i] = ((u32)block[i*4]<<24)|((u32)block[i*4+1]<<16)|((u32)block[i*4+2]<<8)|((u32)block[i*4+3]);
    u32 a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        u32 wi;
        if (i < 16) {
            wi = w[i & 15];
        } else {
            u32 w15 = w[(i+1) & 15];   // w[i-15]
            u32 w2  = w[(i+14) & 15];  // w[i-2]
            u32 s0 = rotr32(w15,7) ^ rotr32(w15,18) ^ (w15>>3);
            u32 s1 = rotr32(w2,17) ^ rotr32(w2,19) ^ (w2>>10);
            wi = w[i & 15] + s0 + w[(i+9) & 15] + s1; // w[i-16] + s0 + w[i-7] + s1
            w[i & 15] = wi;
        }
        u32 S1 = rotr32(e,6) ^ rotr32(e,11) ^ rotr32(e,25);
        u32 ch = (e & f) ^ ((~e) & g);
        u32 t1 = hh + S1 + ch + K256[i] + wi;
        u32 S0 = rotr32(a,2) ^ rotr32(a,13) ^ rotr32(a,22);
        u32 maj = (a & b) ^ (a & c) ^ (b & c);
        u32 t2 = S0 + maj;
        hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
}

void sha256_update(sha256_ctx* c, const u8* data, u32 len) {
    c->total += len;
    while (len) {
        u32 take = (64 - c->n < len) ? (64 - c->n) : len;
        dmemcpy(c->buf + c->n, data, take);
        c->n += take; data += take; len -= take;
        if (c->n == 64) { sha256_transform(c->h, c->buf); c->n = 0; }
    }
}

void sha256_final(sha256_ctx* c, u8 out[32]) {
    u64 bits = c->total * 8;
    u32 n = c->n;
    c->buf[n++] = 0x80;
    if (n > 56) {
        while (n < 64) c->buf[n++] = 0;
        sha256_transform(c->h, c->buf);
        n = 0;
    }
    while (n < 56) c->buf[n++] = 0;
    for (int i = 0; i < 8; i++) c->buf[56 + i] = (u8)(bits >> (56 - i*8)); // big-endian
    sha256_transform(c->h, c->buf);
    for (int i = 0; i < 8; i++) {
        out[i*4]   = (u8)(c->h[i] >> 24);
        out[i*4+1] = (u8)(c->h[i] >> 16);
        out[i*4+2] = (u8)(c->h[i] >> 8);
        out[i*4+3] = (u8)(c->h[i]);
    }
}

void sha256(const u8* data, u32 len, u8 out[32]) {
    sha256_ctx c; sha256_init(&c); sha256_update(&c, data, len); sha256_final(&c, out);
}

// ===========================================================================
// SHA-512 (FIPS 180-4) — streaming
// ===========================================================================

__constant u64 K512[80] = {
    0x428a2f98d728ae22ULL,0x7137449123ef65cdULL,0xb5c0fbcfec4d3b2fULL,0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL,0x59f111f1b605d019ULL,0x923f82a4af194f9bULL,0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL,0x12835b0145706fbeULL,0x243185be4ee4b28cULL,0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL,0x80deb1fe3b1696b1ULL,0x9bdc06a725c71235ULL,0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL,0xefbe4786384f25e3ULL,0x0fc19dc68b8cd5b5ULL,0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL,0x4a7484aa6ea6e483ULL,0x5cb0a9dcbd41fbd4ULL,0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL,0xa831c66d2db43210ULL,0xb00327c898fb213fULL,0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL,0xd5a79147930aa725ULL,0x06ca6351e003826fULL,0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL,0x2e1b21385c26c926ULL,0x4d2c6dfc5ac42aedULL,0x53380d139d95b3dfULL,
    0x650a73548baf63deULL,0x766a0abb3c77b2a8ULL,0x81c2c92e47edaee6ULL,0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL,0xa81a664bbc423001ULL,0xc24b8b70d0f89791ULL,0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL,0xd69906245565a910ULL,0xf40e35855771202aULL,0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL,0x1e376c085141ab53ULL,0x2748774cdf8eeb99ULL,0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL,0x4ed8aa4ae3418acbULL,0x5b9cca4f7763e373ULL,0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL,0x78a5636f43172f60ULL,0x84c87814a1f0ab72ULL,0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL,0xa4506cebde82bde9ULL,0xbef9a3f7b2c67915ULL,0xc67178f2e372532bULL,
    0xca273eceea26619cULL,0xd186b8c721c0c207ULL,0xeada7dd6cde0eb1eULL,0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL,0x0a637dc5a2c898a6ULL,0x113f9804bef90daeULL,0x1b710b35131c471bULL,
    0x28db77f523047d84ULL,0x32caab7b40c72493ULL,0x3c9ebe0a15c9bebcULL,0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL,0x597f299cfc657e2aULL,0x5fcb6fab3ad6faecULL,0x6c44198c4a475817ULL,
};

typedef struct { u64 h[8]; u64 total; u8 buf[128]; u32 n; } sha512_ctx;

inline void sha512_iv(u64 h[8]) {
    h[0]=0x6a09e667f3bcc908ULL; h[1]=0xbb67ae8584caa73bULL;
    h[2]=0x3c6ef372fe94f82bULL; h[3]=0xa54ff53a5f1d36f1ULL;
    h[4]=0x510e527fade682d1ULL; h[5]=0x9b05688c2b3e6c1fULL;
    h[6]=0x1f83d9abfb41bd6bULL; h[7]=0x5be0cd19137e2179ULL;
}

void sha512_init(sha512_ctx* c) {
    sha512_iv(c->h);
    c->total = 0; c->n = 0;
}

// Compression function over a message block already held as 16 big-endian words.
// Taking `w` by value (rather than a `const u8[128]` buffer) is what keeps the
// hot PBKDF2 loop entirely in registers: no local-memory block to memcpy into,
// and no byte-at-a-time reassembly of the schedule. `w` is clobbered.
//
// Callers that pass compile-time-constant padding words (see
// pbkdf2_hmac_sha512_64, where w[8..15] are fixed) get the first schedule
// rounds constant-folded for free, because the full unroll makes every w index
// a constant.
inline void sha512_block(u64 h[8], u64 w[16]) {
    u64 a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    #pragma unroll
    for (int i = 0; i < 80; i++) {
        u64 wi;
        if (i < 16) {
            wi = w[i & 15];
        } else {
            u64 w15 = w[(i+1) & 15];   // w[i-15]
            u64 w2  = w[(i+14) & 15];  // w[i-2]
            u64 s0 = rotr64(w15,1) ^ rotr64(w15,8) ^ (w15>>7);
            u64 s1 = rotr64(w2,19) ^ rotr64(w2,61) ^ (w2>>6);
            wi = w[i & 15] + s0 + w[(i+9) & 15] + s1; // w[i-16] + s0 + w[i-7] + s1
            w[i & 15] = wi;
        }
        u64 S1 = rotr64(e,14) ^ rotr64(e,18) ^ rotr64(e,41);
        u64 ch = g ^ (e & (f ^ g));               // == (e&f) ^ (~e&g), one op cheaper
        u64 t1 = hh + S1 + ch + K512[i] + wi;
        u64 S0 = rotr64(a,28) ^ rotr64(a,34) ^ rotr64(a,39);
        u64 maj = (a & b) ^ (c & (a ^ b));        // == (a&b)^(a&c)^(b&c), two ops cheaper
        u64 t2 = S0 + maj;
        hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
}

// Byte-buffer wrapper, for the streaming ctx used by the selftest kernels and
// the (cold) BIP32 HMACs.
void sha512_transform(u64 h[8], const u8 block[128]) {
    u64 w[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        w[i] = ((u64)block[i*8]<<56)|((u64)block[i*8+1]<<48)|((u64)block[i*8+2]<<40)|((u64)block[i*8+3]<<32)
             |((u64)block[i*8+4]<<24)|((u64)block[i*8+5]<<16)|((u64)block[i*8+6]<<8)|((u64)block[i*8+7]);
    }
    sha512_block(h, w);
}

void sha512_update(sha512_ctx* c, const u8* data, u32 len) {
    c->total += len;
    while (len) {
        u32 take = (128 - c->n < len) ? (128 - c->n) : len;
        dmemcpy(c->buf + c->n, data, take);
        c->n += take; data += take; len -= take;
        if (c->n == 128) { sha512_transform(c->h, c->buf); c->n = 0; }
    }
}

void sha512_final(sha512_ctx* c, u8 out[64]) {
    // We only ever hash messages well under 2^64 bytes, so the high 64 bits of
    // the 128-bit length field are always zero.
    // Pad directly in the context buffer instead of streaming padding byte-by-byte.
    u64 bits = c->total * 8;
    u32 n = c->n;
    c->buf[n++] = 0x80;
    if (n > 112) {
        while (n < 128) c->buf[n++] = 0;
        sha512_transform(c->h, c->buf);
        n = 0;
    }
    while (n < 112) c->buf[n++] = 0;
    // 128-bit length: high 64 bits are always zero (messages are tiny), low 64
    // bits big-endian.
    for (int i = 0; i < 8; i++) c->buf[112 + i] = 0;
    for (int i = 0; i < 8; i++) c->buf[120 + i] = (u8)(bits >> (56 - i*8));
    sha512_transform(c->h, c->buf);
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++) out[i*8+j] = (u8)(c->h[i] >> (56 - j*8));
}

void sha512(const u8* data, u32 len, u8 out[64]) {
    sha512_ctx c; sha512_init(&c); sha512_update(&c, data, len); sha512_final(&c, out);
}

// ===========================================================================
// Keccak-256 (Ethereum's hash)
//
// This is *original* Keccak, not NIST SHA-3: the only difference is the domain
// separation byte in the padding (0x01 here, 0x06 for SHA-3), but it changes
// every digest, so SHA-3 can never be substituted.
//
// Rate is 136 bytes (1600 - 2*256 bits). The only call on the hot path hashes a
// 64-byte public key, i.e. a single padded block and one permutation — next to
// PBKDF2's 4096 SHA-512 compressions the cost is noise, so this is written for
// clarity rather than speed.
// ===========================================================================

__constant u64 KECCAK_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL,
};

// rho offsets, flattened to the same x + 5*y indexing as the state lanes A.
__constant int KECCAK_RHO[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
};

// Keccak-f[1600] on 25 lanes indexed A[x + 5*y]. The x/y loops are unrolled so
// every lane index and rho offset becomes a constant.
void keccak_f1600(u64 A[25]) {
    #pragma unroll 1
    for (int r = 0; r < 24; r++) {
        // theta
        u64 C[5], D[5];
        #pragma unroll
        for (int x = 0; x < 5; x++)
            C[x] = A[x] ^ A[x+5] ^ A[x+10] ^ A[x+15] ^ A[x+20];
        #pragma unroll
        for (int x = 0; x < 5; x++)
            D[x] = C[(x+4)%5] ^ rotl64(C[(x+1)%5], 1);
        #pragma unroll
        for (int y = 0; y < 5; y++)
            #pragma unroll
            for (int x = 0; x < 5; x++)
                A[x + 5*y] ^= D[x];

        // rho (rotate each lane) + pi (permute lane positions)
        u64 B[25];
        #pragma unroll
        for (int y = 0; y < 5; y++)
            #pragma unroll
            for (int x = 0; x < 5; x++)
                B[y + 5*((2*x + 3*y) % 5)] = rotl64(A[x + 5*y], KECCAK_RHO[x + 5*y]);

        // chi
        #pragma unroll
        for (int y = 0; y < 5; y++)
            #pragma unroll
            for (int x = 0; x < 5; x++)
                A[x + 5*y] = B[x + 5*y] ^ ((~B[(x+1)%5 + 5*y]) & B[(x+2)%5 + 5*y]);

        // iota
        A[0] ^= KECCAK_RC[r];
    }
}

#define KECCAK_RATE 136 // bytes absorbed per permutation for Keccak-256

// Lanes are little-endian by definition, so they are assembled with explicit
// shifts rather than by aliasing bytes (which would depend on GPU endianness).
// noinline for the same reason as pbkdf2_hmac_sha512_64 — see the note there.
__attribute__((noinline))
void keccak256(const u8* msg, u32 len, u8 out[32]) {
    u64 A[25];
    #pragma unroll
    for (int i = 0; i < 25; i++) A[i] = 0;

    u32 i = 0;
    while (len - i >= KECCAK_RATE) {
        #pragma unroll
        for (int l = 0; l < KECCAK_RATE/8; l++) {
            u64 v = 0;
            #pragma unroll
            for (int b = 0; b < 8; b++) v |= (u64)msg[i + l*8 + b] << (b*8);
            A[l] ^= v;
        }
        keccak_f1600(A);
        i += KECCAK_RATE;
    }

    // Final block: the tail plus pad10*1. When only one byte of padding is left
    // (rem == 135) both marks land on the same byte and OR together to 0x81,
    // which is exactly what the padding rule requires.
    u64 tail[KECCAK_RATE/8];
    #pragma unroll
    for (int l = 0; l < KECCAK_RATE/8; l++) tail[l] = 0;
    u32 rem = len - i;
    for (u32 j = 0; j < rem; j++)
        tail[j >> 3] |= (u64)msg[i + j] << ((j & 7) * 8);
    tail[rem >> 3] |= (u64)0x01 << ((rem & 7) * 8);  // start of padding
    tail[16]       |= (u64)0x80 << 56;               // final bit, last byte of the rate
    #pragma unroll
    for (int l = 0; l < KECCAK_RATE/8; l++) A[l] ^= tail[l];
    keccak_f1600(A);

    // Squeeze: 32 bytes is well under the rate, so one squeeze pass suffices.
    #pragma unroll
    for (int l = 0; l < 4; l++)
        #pragma unroll
        for (int b = 0; b < 8; b++) out[l*8 + b] = (u8)(A[l] >> (b*8));
}

// ===========================================================================
// HMAC-SHA512 (RFC 2104) with precomputed inner/outer midstates.
//
// The key-dependent first block of each SHA-512 (ipad/opad) is hashed once in
// hmac512_init; every subsequent MAC only streams the message. PBKDF2 reuses the
// same key 2048x, so this is the key optimization.
// ===========================================================================

typedef struct { sha512_ctx inner; sha512_ctx outer; } hmac512_ctx;

void hmac512_init(hmac512_ctx* h, const u8* key, u32 keylen) {
    u8 k[128];
    if (keylen > 128) {
        u8 t[64]; sha512(key, keylen, t);
        for (int i = 0; i < 64; i++) k[i] = t[i];
        for (int i = 64; i < 128; i++) k[i] = 0;
    } else {
        for (u32 i = 0; i < keylen; i++) k[i] = key[i];
        for (u32 i = keylen; i < 128; i++) k[i] = 0;
    }
    u8 pad[128];
    for (int i = 0; i < 128; i++) pad[i] = k[i] ^ 0x36;
    sha512_init(&h->inner); sha512_update(&h->inner, pad, 128);
    for (int i = 0; i < 128; i++) pad[i] = k[i] ^ 0x5c;
    sha512_init(&h->outer); sha512_update(&h->outer, pad, 128);
}

// MAC of a single message using the precomputed states (states are copied, not
// mutated, so the same ctx can be reused for many messages).
void hmac512_compute(const hmac512_ctx* h, const u8* msg, u32 msglen, u8 out[64]) {
    sha512_ctx in = h->inner;
    sha512_update(&in, msg, msglen);
    u8 ih[64]; sha512_final(&in, ih);
    sha512_ctx ou = h->outer;
    sha512_update(&ou, ih, 64);
    sha512_final(&ou, out);
}

void hmac_sha512(const u8* key, u32 keylen, const u8* msg, u32 msglen, u8 out[64]) {
    hmac512_ctx h; hmac512_init(&h, key, keylen);
    hmac512_compute(&h, msg, msglen, out);
}

// ===========================================================================
// PBKDF2-HMAC-SHA512, specialized to dkLen == 64 (one output block), as used by
// BIP-39 seed derivation (salt = "mnemonic" || passphrase, c = 2048).
// ===========================================================================

// IMPORTANT: noinline is required (here and on seed_to_eth_address/keccak256).
// The CUDA original documented an nvcc miscompile when these were all inlined
// into the single huge k_pipeline frame; keeping each as its own frame matches
// the individually-verified kernels bit-for-bit and keeps the k_pipeline frame
// manageable on any backend. Do not remove.
//
// This is ~92-94% of the search's total cost (measured on the CUDA original),
// so it is the one routine worth specializing hard. Two facts make the hot
// loop completely regular:
//
//   * dkLen == 64, so there is exactly one output block and no outer loop;
//   * every U_i is 64 bytes, so each of the two HMAC halves per iteration is a
//     SHA-512 over exactly one block, whose layout is fixed:
//         W[0..7] = U_i,  W[8] = 0x80<<56,  W[9..14] = 0,  W[15] = (128+64)*8.
//
// So the 2048 iterations reduce to 4096 calls of sha512_block on state and
// message words that live entirely in registers. U is carried as 8 u64 rather
// than 64 bytes, which removes the byte<->word repacking as well.
__attribute__((noinline))
void pbkdf2_hmac_sha512_64(
    const u8* pw, u32 pwlen, const u8* salt, u32 saltlen, u32 iters, u8 out[64]) {
    // ---- key -> ipad/opad midstates (each one compression, done once) ----
    u64 kw[16];
    if (pwlen > 128) {
        u8 t[64]; sha512(pw, pwlen, t);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            u64 v = 0;
            #pragma unroll
            for (int j = 0; j < 8; j++) v = (v << 8) | t[i*8 + j];
            kw[i] = v;
        }
        #pragma unroll
        for (int i = 8; i < 16; i++) kw[i] = 0;
    } else {
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            u64 v = 0;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                u32 o = (u32)(i*8 + j);
                v = (v << 8) | (u64)(o < pwlen ? pw[o] : 0);
            }
            kw[i] = v;
        }
    }

    u64 inner[8], outer[8], w[16];
    sha512_iv(inner);
    #pragma unroll
    for (int i = 0; i < 16; i++) w[i] = kw[i] ^ 0x3636363636363636ULL;
    sha512_block(inner, w);
    sha512_iv(outer);
    #pragma unroll
    for (int i = 0; i < 16; i++) w[i] = kw[i] ^ 0x5c5c5c5c5c5c5c5cULL;
    sha512_block(outer, w);

    // ---- U1 = HMAC(pw, salt || INT32BE(1)) ----
    // One compression out of ~4097, so a byte buffer here costs nothing and
    // keeps arbitrary salt lengths (BIP-39 salt = "mnemonic" || passphrase)
    // working. saltlen+4 <= 111 keeps it to the single-block case; longer salts
    // fall back to streaming.
    u64 u[8], acc[8], st[8];
    if (saltlen + 4 <= 111) {
        u8 blk[128];
        u32 n = 0;
        for (u32 i = 0; i < saltlen; i++) blk[n++] = salt[i];
        blk[n++] = 0; blk[n++] = 0; blk[n++] = 0; blk[n++] = 1;   // INT32BE(1)
        blk[n++] = 0x80;
        while (n < 120) blk[n++] = 0;
        u64 bits = (u64)(128 + saltlen + 4) * 8;
        #pragma unroll
        for (int i = 0; i < 8; i++) blk[120 + i] = (u8)(bits >> (56 - i*8));
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            u64 v = 0;
            #pragma unroll
            for (int j = 0; j < 8; j++) v = (v << 8) | blk[i*8 + j];
            w[i] = v;
        }
        #pragma unroll
        for (int i = 0; i < 8; i++) st[i] = inner[i];
        sha512_block(st, w);
    } else {
        // Long salt: stream it, starting from the ipad midstate (128 bytes of
        // ipad block are already absorbed, hence total = 128).
        sha512_ctx in;
        #pragma unroll
        for (int i = 0; i < 8; i++) in.h[i] = inner[i];
        in.total = 128; in.n = 0;
        u8 idx[4] = {0, 0, 0, 1};
        sha512_update(&in, salt, saltlen);
        sha512_update(&in, idx, 4);
        u8 ih[64]; sha512_final(&in, ih);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            u64 v = 0;
            #pragma unroll
            for (int j = 0; j < 8; j++) v = (v << 8) | ih[i*8 + j];
            st[i] = v;
        }
    }
    // outer half of U1: message is the 64-byte inner digest.
    #pragma unroll
    for (int i = 0; i < 8; i++) w[i] = st[i];
    w[8] = 0x8000000000000000ULL;
    #pragma unroll
    for (int i = 9; i < 15; i++) w[i] = 0;
    w[15] = (u64)(128 + 64) * 8;
    #pragma unroll
    for (int i = 0; i < 8; i++) u[i] = outer[i];
    sha512_block(u, w);
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = u[i];

    // ---- iterations 2..c: two fixed-layout compressions, all in registers ----
    for (u32 iter = 1; iter < iters; iter++) {
        u64 s[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) { s[i] = inner[i]; w[i] = u[i]; }
        w[8] = 0x8000000000000000ULL;
        #pragma unroll
        for (int i = 9; i < 15; i++) w[i] = 0;
        w[15] = (u64)(128 + 64) * 8;
        sha512_block(s, w);

        #pragma unroll
        for (int i = 0; i < 8; i++) { u[i] = outer[i]; w[i] = s[i]; }
        w[8] = 0x8000000000000000ULL;
        #pragma unroll
        for (int i = 9; i < 15; i++) w[i] = 0;
        w[15] = (u64)(128 + 64) * 8;
        sha512_block(u, w);

        #pragma unroll
        for (int i = 0; i < 8; i++) acc[i] ^= u[i];
    }

    #pragma unroll
    for (int i = 0; i < 8; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++) out[i*8 + j] = (u8)(acc[i] >> (56 - j*8));
}

// ===========================================================================
// secp256k1
//
// Field elements are 4x u64 limbs, little-endian: value = n[0] + n[1]*2^64 +
// n[2]*2^128 + n[3]*2^192. p = 2^256 - 0x1000003D1. Points use Jacobian
// coordinates (X,Y,Z) with the point at infinity represented by Z == 0.
// ===========================================================================

typedef struct { u64 n[4]; } fe;   // field element mod p
typedef struct { u64 n[4]; } scalar; // integer mod n (group order)
typedef struct { fe X, Y, Z; } jpoint;

// p (little-endian limbs)
__constant u64 P[4] = {
    0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};
// n, the group order (little-endian limbs)
__constant u64 N[4] = {
    0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL};
// Generator G in affine coordinates.
__constant u64 GX[4] = {
    0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL, 0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL};
__constant u64 GY[4] = {
    0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL, 0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL};

#define FE_C 0x1000003D1ULL   // p = 2^256 - FE_C

inline void fe_set(fe* r, const u64 v[4]) {
    r->n[0]=v[0]; r->n[1]=v[1]; r->n[2]=v[2]; r->n[3]=v[3];
}
inline void fe_zero(fe* r) { r->n[0]=r->n[1]=r->n[2]=r->n[3]=0; }
inline int fe_is_zero(const fe* a) {
    return (a->n[0]|a->n[1]|a->n[2]|a->n[3]) == 0;
}
inline void fe_one(fe* r) { r->n[0]=1; r->n[1]=r->n[2]=r->n[3]=0; }

// returns 1 if a >= b (treating both as 256-bit little-endian)
int ge256(const u64 a[4], const u64 b[4]) {
    for (int i = 3; i >= 0; i--) {
        if (a[i] != b[i]) return a[i] > b[i];
    }
    return 1; // equal
}

// r = a - m (assumes a >= m), 256-bit
void sub256(u64 r[4], const u64 a[4], const u64 m[4]) {
    u64 borrow = 0;
    for (int i = 0; i < 4; i++) r[i] = sbb64(a[i], m[i], &borrow);
}

inline void fe_reduce_p(fe* r) {
    // P lives in __constant space; ge256/sub256 take private pointers (the
    // default for unqualified params), so pass a private copy.
    u64 Pp[4] = {P[0], P[1], P[2], P[3]};
    if (ge256(r->n, Pp)) sub256(r->n, r->n, Pp);
}

void fe_add(fe* r, const fe* a, const fe* b) {
    u64 carry = 0;
    for (int i = 0; i < 4; i++) r->n[i] = adc64(a->n[i], b->n[i], &carry);
    // value = r + carry*2^256 ≡ r + carry*FE_C (mod p)
    if (carry) {
        u64 c2 = 0;
        r->n[0] = adc64(r->n[0], carry * FE_C, &c2);
        for (int i = 1; i < 4; i++) r->n[i] = adc64(r->n[i], 0, &c2);
        if (c2) { // extremely rare second wrap
            u64 c3 = 0;
            r->n[0] = adc64(r->n[0], c2 * FE_C, &c3);
            for (int i = 1; i < 4; i++) r->n[i] = adc64(r->n[i], 0, &c3);
        }
    }
    fe_reduce_p(r);
}

void fe_sub(fe* r, const fe* a, const fe* b) {
    // r = a - b mod p; if underflow add p
    u64 t[4];
    u64 borrow = 0;
    for (int i = 0; i < 4; i++) t[i] = sbb64(a->n[i], b->n[i], &borrow);
    if (borrow) {
        u64 carry = 0;
        for (int i = 0; i < 4; i++) t[i] = adc64(t[i], P[i], &carry);
    }
    r->n[0]=t[0]; r->n[1]=t[1]; r->n[2]=t[2]; r->n[3]=t[3];
}

// reduce a 512-bit product (8 little-endian limbs) mod p into r
void fe_reduce512(fe* r, const u64 t[8]) {
    // m[0..3] = t_lo + t_hi*FE_C, m[4] = carry
    u64 m[5];
    u64 carry = 0;
    for (int i = 0; i < 4; i++) m[i] = mac64(t[4+i], FE_C, t[i], &carry);
    m[4] = carry;
    // fold m[4]*FE_C back in
    u64 c = 0;
    r->n[0] = mac64(m[4], FE_C, m[0], &c);
    for (int i = 1; i < 4; i++) r->n[i] = adc64(m[i], 0, &c);
    u64 extra = c; // 0 or 1
    if (extra) {
        u64 c2 = 0;
        r->n[0] = mac64(extra, FE_C, r->n[0], &c2);
        for (int i = 1; i < 4; i++) r->n[i] = adc64(r->n[i], 0, &c2);
    }
    fe_reduce_p(r);
}

void fe_mul(fe* r, const fe* a, const fe* b) {
    u64 t[8]; for (int i = 0; i < 8; i++) t[i] = 0;
    for (int i = 0; i < 4; i++) {
        u64 carry = 0;
        for (int j = 0; j < 4; j++)
            t[i+j] = mac64(a->n[i], b->n[j], t[i+j], &carry);
        t[i+4] = carry;
    }
    fe_reduce512(r, t);
}

inline void fe_sqr(fe* r, const fe* a) { fe_mul(r, a, a); }

// Modular inverse a^(p-2) mod p via the libsecp256k1 addition chain: 255
// squarings + 15 multiplications, versus ~256 + ~250 for a plain square-and-
// multiply over the bits of p-2. The binary expansion of p-2 is 5 runs of 1s
// with lengths in {1, 2, 22, 223}, so the chain builds 2^n-1 for
// n = 1,2,3,6,9,11,22,44,88,176,220,223 and assembles the result from those.
#define FE_SQR_N(dst, src, n) do { dst = (src); for (int _i = 0; _i < (n); _i++) fe_sqr(&dst, &dst); } while (0)

void fe_inv(fe* r, const fe* a) {
    fe x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223, t;

    fe_sqr(&x2, a);        fe_mul(&x2, &x2, a);       // a^(2^2-1)
    fe_sqr(&x3, &x2);      fe_mul(&x3, &x3, a);       // a^(2^3-1)
    FE_SQR_N(x6,   x3,   3);   fe_mul(&x6,   &x6,   &x3);
    FE_SQR_N(x9,   x6,   3);   fe_mul(&x9,   &x9,   &x3);
    FE_SQR_N(x11,  x9,   2);   fe_mul(&x11,  &x11,  &x2);
    FE_SQR_N(x22,  x11, 11);   fe_mul(&x22,  &x22,  &x11);
    FE_SQR_N(x44,  x22, 22);   fe_mul(&x44,  &x44,  &x22);
    FE_SQR_N(x88,  x44, 44);   fe_mul(&x88,  &x88,  &x44);
    FE_SQR_N(x176, x88, 88);   fe_mul(&x176, &x176, &x88);
    FE_SQR_N(x220, x176, 44);  fe_mul(&x220, &x220, &x44);
    FE_SQR_N(x223, x220, 3);   fe_mul(&x223, &x223, &x3);

    FE_SQR_N(t, x223, 23);  fe_mul(&t, &t, &x22);
    FE_SQR_N(t, t,     5);  fe_mul(&t, &t, a);
    FE_SQR_N(t, t,     3);  fe_mul(&t, &t, &x2);
    FE_SQR_N(t, t,     2);  fe_mul(r, &t, a);
}

// ---- point operations (Jacobian) ----

inline void jp_set_infinity(jpoint* p) { fe_zero(&p->Z); }
inline int jp_is_infinity(const jpoint* p) { return fe_is_zero(&p->Z); }

// P3 = 2*P1 (dbl-2009-l, a=0). Writes to locals first so r may alias p.
void jp_double(jpoint* r, const jpoint* p) {
    if (jp_is_infinity(p) || fe_is_zero(&p->Y)) { jp_set_infinity(r); return; }
    fe A, B, C, D, E, F, t, t2, X3, Y3, Z3;
    fe_sqr(&A, &p->X);                 // A = X^2
    fe_sqr(&B, &p->Y);                 // B = Y^2
    fe_sqr(&C, &B);                    // C = B^2
    fe_add(&t, &p->X, &B); fe_sqr(&t, &t);   // (X+B)^2
    fe_sub(&t, &t, &A); fe_sub(&t, &t, &C);  // (X+B)^2 - A - C
    fe_add(&D, &t, &t);                // D = 2*((X+B)^2 - A - C)
    fe_add(&E, &A, &A); fe_add(&E, &E, &A);  // E = 3*A
    fe_sqr(&F, &E);                    // F = E^2
    fe_mul(&t, &p->Y, &p->Z); fe_add(&Z3, &t, &t); // Z3 = 2*Y*Z  (before Y is touched)
    fe_add(&t, &D, &D);                // 2D
    fe_sub(&X3, &F, &t);               // X3 = F - 2D
    fe_sub(&t, &D, &X3);               // D - X3
    fe_mul(&t, &E, &t);                // E*(D - X3)
    fe_add(&t2, &C, &C); fe_add(&t2, &t2, &t2); fe_add(&t2, &t2, &t2); // 8C
    fe_sub(&Y3, &t, &t2);              // Y3 = E*(D-X3) - 8C
    r->X = X3; r->Y = Y3; r->Z = Z3;
}

// P3 = P1 + Q where Q is affine (qx,qy). (madd-2007-bl). Handles P1 = infinity
// and the P1 == +/-Q cases.
void jp_add_affine(jpoint* r, const jpoint* p, const fe* qx, const fe* qy) {
    if (jp_is_infinity(p)) { r->X=*qx; r->Y=*qy; fe_one(&r->Z); return; }
    fe Z1Z1, U2, S2, H, HH, I, J, rr, V, t, t2;
    fe_sqr(&Z1Z1, &p->Z);              // Z1Z1 = Z1^2
    fe_mul(&U2, qx, &Z1Z1);            // U2 = X2*Z1Z1
    fe_mul(&S2, qy, &p->Z); fe_mul(&S2, &S2, &Z1Z1); // S2 = Y2*Z1^3
    fe_sub(&H, &U2, &p->X);            // H = U2 - X1
    fe_sub(&t, &S2, &p->Y);            // S2 - Y1
    if (fe_is_zero(&H)) {
        if (fe_is_zero(&t)) { jp_double(r, p); return; } // P == Q
        jp_set_infinity(r); return;                       // P == -Q
    }
    fe_sqr(&HH, &H);                   // HH = H^2
    fe_add(&I, &HH, &HH); fe_add(&I, &I, &I); // I = 4*HH
    fe_mul(&J, &H, &I);                // J = H*I
    fe_add(&rr, &t, &t);               // r = 2*(S2 - Y1)
    fe_mul(&V, &p->X, &I);             // V = X1*I
    fe X3, Y3, Z3;
    fe_sqr(&X3, &rr);                  // r^2
    fe_sub(&X3, &X3, &J);              // r^2 - J
    fe_add(&t2, &V, &V); fe_sub(&X3, &X3, &t2); // X3 = r^2 - J - 2V
    fe_sub(&t, &V, &X3);               // V - X3
    fe_mul(&t, &rr, &t);               // r*(V - X3)
    fe_mul(&t2, &p->Y, &J); fe_add(&t2, &t2, &t2); // 2*Y1*J
    fe_sub(&Y3, &t, &t2);              // Y3 = r*(V-X3) - 2*Y1*J
    // Z3 = (Z1+H)^2 - Z1Z1 - HH = 2*Z1*H
    fe_add(&t, &p->Z, &H); fe_sqr(&t, &t);
    fe_sub(&t, &t, &Z1Z1); fe_sub(&Z3, &t, &HH);
    r->X = X3; r->Y = Y3; r->Z = Z3;
}

// Convert big-endian 32-byte scalar to little-endian limbs.
void be32_to_limbs(const u8* b, u64 out[4]) {
    for (int i = 0; i < 4; i++) {
        u64 v = 0;
        for (int j = 0; j < 8; j++) v = (v << 8) | b[i*8 + j];
        out[3 - i] = v;
    }
}

// Reference R = k*G by plain double-and-add. Only used to build the window
// table below (once per process), so its cost does not matter; keeping it means
// the table is generated by the same code path the selftest already validates.
void scalar_mul_G_ref(jpoint* r, const u64 k[4]) {
    jpoint acc; jp_set_infinity(&acc);
    u64 gxa[4] = {GX[0], GX[1], GX[2], GX[3]};
    u64 gya[4] = {GY[0], GY[1], GY[2], GY[3]};
    fe gx, gy; fe_set(&gx, gxa); fe_set(&gy, gya);
    for (int limb = 3; limb >= 0; limb--) {
        for (int b = 63; b >= 0; b--) {
            jp_double(&acc, &acc);
            if ((k[limb] >> b) & 1) jp_add_affine(&acc, &acc, &gx, &gy);
        }
    }
    *r = acc;
}

// ---- fixed-base window table ----
//
// Every scalar multiplication in this program is against the fixed generator G
// (BIP32 CKD-priv for the two non-hardened levels, plus the final pubkey), so
// the multiples of G can be precomputed. With 4-bit windows,
//     k*G = sum over w of digit_w * 16^w * G,   digit_w in 0..15
// which is 64 point additions and *no* doublings, against 256 doublings + ~128
// additions for double-and-add. gt[w][j] holds (j+1) * 16^w * G in affine form;
// 64*15*64 B = 60 KiB, small enough to stay resident in cache.
//
// Unlike the CUDA original (a __device__ global), the table is an explicit
// __global buffer allocated by the host and passed down through the derivation
// call chain — OpenCL has no device-side global symbols.
typedef struct { fe x, y; } apoint;

// One thread per table entry. Must be launched once (with >= 960 threads)
// before any kernel that derives a public key; src/gpu.rs does this at startup.
__kernel void k_init_gtable(__global apoint* gt) {
    u32 t = get_global_id(0);
    if (t >= 64 * 15) return;
    u32 w = t / 15, j = t % 15;

    // scalar = (j+1) << (4*w)
    u64 k[4] = {0, 0, 0, 0};
    u32 shift = 4 * w, limb = shift >> 6, off = shift & 63;
    u64 v = (u64)(j + 1);
    k[limb] = v << off;
    if (off && limb < 3) k[limb + 1] = v >> (64 - off);

    jpoint p; scalar_mul_G_ref(&p, k);
    fe zinv, z2, z3;
    fe_inv(&zinv, &p.Z);
    fe_sqr(&z2, &zinv);
    fe_mul(&z3, &z2, &zinv);
    apoint ap;
    fe_mul(&ap.x, &p.X, &z2);
    fe_mul(&ap.y, &p.Y, &z3);
    gt[w*15 + j] = ap;
}

// R = k*G, k given as little-endian limbs (k must be in [1, n-1]).
void scalar_mul_G(jpoint* r, const u64 k[4], __global const apoint* gt) {
    jpoint acc; jp_set_infinity(&acc);
    #pragma unroll 1
    for (int w = 0; w < 64; w++) {
        u32 d = (u32)((k[w >> 4] >> ((w & 15) * 4)) & 15);
        if (d) {
            fe qx = gt[w*15 + (d-1)].x;
            fe qy = gt[w*15 + (d-1)].y;
            jp_add_affine(&acc, &acc, &qx, &qy);
        }
    }
    *r = acc;
}

// k*G in affine coordinates. Shared by both serializations below: the scalar
// multiplication and the single field inversion are the entire cost.
void pubkey_affine(const u64 k[4], fe* x, fe* y, __global const apoint* gt) {
    jpoint p; scalar_mul_G(&p, k, gt);
    if (jp_is_infinity(&p)) { fe_zero(x); fe_zero(y); return; }
    fe zinv, zinv2, zinv3;
    fe_inv(&zinv, &p.Z);
    fe_sqr(&zinv2, &zinv);
    fe_mul(&zinv3, &zinv2, &zinv);
    fe_mul(x, &p.X, &zinv2);
    fe_mul(y, &p.Y, &zinv3);
}

// Serialize k*G as a 33-byte compressed pubkey. Used by BIP32 CKD-priv, which
// hashes the parent public key in compressed form at every non-hardened level.
void pubkey_compressed(const u64 k[4], u8 out[33], __global const apoint* gt) {
    fe x, y; pubkey_affine(k, &x, &y, gt);
    out[0] = 0x02 | (u8)(y.n[0] & 1);
    // x big-endian
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 8; j++) out[1 + i*8 + j] = (u8)(x.n[3-i] >> (56 - j*8));
}

// Serialize k*G as the 64-byte uncompressed X||Y, big-endian, with no 0x04
// prefix — the exact preimage Ethereum hashes to derive an address.
void pubkey_xy(const u64 k[4], u8 out[64], __global const apoint* gt) {
    fe x, y; pubkey_affine(k, &x, &y, gt);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 8; j++) {
            out[     i*8 + j] = (u8)(x.n[3-i] >> (56 - j*8));
            out[32 + i*8 + j] = (u8)(y.n[3-i] >> (56 - j*8));
        }
}

// r = (a + b) mod n, all as big-endian 32-byte; returns 1 if result != 0.
void scalar_add_modn(const u64 a[4], const u64 b[4], u64 r[4]) {
    u64 carry = 0;
    for (int i = 0; i < 4; i++) r[i] = adc64(a[i], b[i], &carry);
    u64 Nn[4] = {N[0], N[1], N[2], N[3]};
    if (carry || ge256(r, Nn)) sub256(r, r, Nn);
}

// ===========================================================================
// BIP32 derivation of the fixed path m/44'/60'/0'/0/0 (BIP-44 coin type 60 =
// Ethereum, the default MetaMask account), ending in the 20-byte address.
// ===========================================================================

void limbs_to_be32(const u64 k[4], u8* out) {
    for (int j = 0; j < 4; j++)
        for (int b = 0; b < 8; b++) out[j*8 + b] = (u8)(k[3-j] >> (56 - b*8));
}

// CKD-priv for one level. key/cc are 32-byte big-endian buffers, updated in place.
void bip32_ckd_priv(u8 key[32], u8 cc[32], u32 index, __global const apoint* gt) {
    u8 data[37];
    if (index & 0x80000000u) {
        // hardened: 0x00 || ser256(key) || ser32(index)
        data[0] = 0x00;
        for (int i = 0; i < 32; i++) data[1+i] = key[i];
    } else {
        // normal: serP(point(key)) || ser32(index)
        u64 k[4]; be32_to_limbs(key, k);
        u8 pub33[33]; pubkey_compressed(k, pub33, gt);
        for (int i = 0; i < 33; i++) data[i] = pub33[i];
    }
    data[33] = (u8)(index >> 24); data[34] = (u8)(index >> 16);
    data[35] = (u8)(index >> 8);  data[36] = (u8)(index);

    u8 I[64];
    hmac_sha512(cc, 32, data, 37, I);

    // child key = (IL + parent key) mod n
    u64 il[4], pk[4], ck[4];
    be32_to_limbs(I, il);
    be32_to_limbs(key, pk);
    scalar_add_modn(il, pk, ck);
    limbs_to_be32(ck, key);
    // child chain code = IR
    for (int i = 0; i < 32; i++) cc[i] = I[32 + i];
}

// seed[64] -> 20-byte Ethereum address for m/44'/60'/0'/0/0.
// noinline required — see the note on pbkdf2_hmac_sha512_64.
//
// The master-key HMAC key is the literal string "Bitcoin seed" for every coin:
// it is fixed by BIP32 itself, not by Bitcoin, and Ethereum wallets use it too.
// Only the coin-type level (60' vs 0') differs from a BIP-44 Bitcoin account.
// (OpenCL C has no string literals in device code, hence the char array.)
__attribute__((noinline))
void seed_to_eth_address(const u8 seed[64], u8 addr[20], __global const apoint* gt) {
    const u8 bitcoin_seed[12] = {'B','i','t','c','o','i','n',' ','s','e','e','d'};
    u8 I[64];
    hmac_sha512(bitcoin_seed, 12, seed, 64, I);
    u8 key[32], cc[32];
    for (int i = 0; i < 32; i++) { key[i] = I[i]; cc[i] = I[32 + i]; }

    const u32 H = 0x80000000u;
    bip32_ckd_priv(key, cc, H + 44, gt);
    bip32_ckd_priv(key, cc, H + 60, gt);
    bip32_ckd_priv(key, cc, H + 0, gt);
    bip32_ckd_priv(key, cc, 0, gt);
    bip32_ckd_priv(key, cc, 0, gt);

    u64 k[4]; be32_to_limbs(key, k);
    u8 pub64[64]; pubkey_xy(k, pub64, gt);
    u8 h[32]; keccak256(pub64, 64, h);
    // The address is the low 20 bytes of the digest.
    for (int i = 0; i < 20; i++) addr[i] = h[12 + i];
}

// ===========================================================================
// Full search kernel: one candidate (12 word indices) per thread.
// Filters on the BIP-39 checksum, derives the seed and address, compares to the
// target address, and records the first match via atomic_cmpxchg.
// ===========================================================================

// Largest joined mnemonic ("word word ... word", 12 words + 11 spaces). 512 is
// comfortably above every BIP-39 language (host asserts the real bound fits).
#define MNEMONIC_BUF 512

// Returns 1 if the 12 indices form a valid BIP-39 checksum.
int bip39_checksum_ok(__global const ushort idx[12]) {
    u8 ent[17];
    for (int i = 0; i < 17; i++) ent[i] = 0;
    int bitpos = 0;
    for (int w = 0; w < 12; w++) {
        u32 v = idx[w]; // 11-bit word index
        for (int b = 10; b >= 0; b--) {
            if ((v >> b) & 1) ent[bitpos >> 3] |= (u8)(0x80 >> (bitpos & 7));
            bitpos++;
        }
    }
    // entropy = ent[0..16]; 4-bit checksum = high nibble of ent[16]
    u8 h[32]; sha256(ent, 16, h);
    return (ent[16] >> 4) == (h[0] >> 4);
}

// Pass 1: keep only candidates with a valid BIP-39 checksum (~1/16). Survivor
// candidate indices are compacted into `survivors` via an atomic counter, so the
// heavy second pass runs with no divergence.
__kernel void k_filter(__global const ushort* cand, u32 n,
                       __global u32* survivors, __global u32* counter) {
    u32 i = get_global_id(0);
    if (i >= n) return;
    if (bip39_checksum_ok(cand + (u64)i * 12)) {
        u32 slot = atomic_add(counter, 1u);
        survivors[slot] = i;
    }
}

// Pass 2: full derivation for each compacted survivor. Thread t handles
// survivors[t]; every thread does real work.
__kernel void k_pipeline(__global const ushort* cand, __global const u32* survivors, u32 count,
                __global const u8* wordlist, __global const u8* word_lens, u32 word_stride,
                __global const u8* target_addr,
                __global u32* found_flag, __global u32* found_idx,
                __global const apoint* gt) {
    u32 t = get_global_id(0);
    if (t >= count) return;
    if (*found_flag) return;

    u32 i = survivors[t];
    __global const ushort* idx = cand + (u64)i * 12;

    // Build the (already NFKD) mnemonic: words joined by single spaces.
    u8 msg[MNEMONIC_BUF];
    u32 mlen = 0;
    for (int w = 0; w < 12; w++) {
        u32 wi = idx[w];
        u8 wl = word_lens[wi];
        __global const u8* wp = wordlist + (u64)wi * word_stride;
        for (u32 c = 0; c < wl; c++) msg[mlen++] = wp[c];
        if (w < 11) msg[mlen++] = ' ';
    }

    u8 seed[64];
    const u8 salt[8] = {'m','n','e','m','o','n','i','c'};
    pbkdf2_hmac_sha512_64(msg, mlen, salt, 8, 2048, seed);

    u8 addr[20];
    seed_to_eth_address(seed, addr, gt);

    int eq = 1;
    for (int j = 0; j < 20; j++) if (addr[j] != target_addr[j]) { eq = 0; break; }
    if (eq) {
        if (atomic_cmpxchg(found_flag, 0u, 1u) == 0u) *found_idx = i;
    }
}

// ===========================================================================
// Selftest kernels: one input message per thread (stride-packed), one digest
// out. Unqualified pointer parameters default to __private on OpenCL (unlike
// CUDA), so global inputs are staged through private buffers before the
// primitives run and results are written back byte-wise. The private staging
// buffer bounds what these (non-hot-path) kernels accept.
// ===========================================================================

#define IN_BUF 512

__kernel void k_sha256(__global const u8* msgs, __global const u32* lens, u32 stride, __global u8* out, u32 n) {
    u32 i = get_global_id(0);
    if (i >= n) return;
    u32 len = lens[i] < IN_BUF ? lens[i] : IN_BUF; // host asserts stride <= IN_BUF
    u8 m[IN_BUF];
    for (u32 j = 0; j < len; j++) m[j] = msgs[(u64)i*stride + j];
    u8 d[32];
    sha256(m, len, d);
    for (int j = 0; j < 32; j++) out[(u64)i*32 + j] = d[j];
}

__kernel void k_sha512(__global const u8* msgs, __global const u32* lens, u32 stride, __global u8* out, u32 n) {
    u32 i = get_global_id(0);
    if (i >= n) return;
    u32 len = lens[i] < IN_BUF ? lens[i] : IN_BUF;
    u8 m[IN_BUF];
    for (u32 j = 0; j < len; j++) m[j] = msgs[(u64)i*stride + j];
    u8 d[64];
    sha512(m, len, d);
    for (int j = 0; j < 64; j++) out[(u64)i*64 + j] = d[j];
}

__kernel void k_keccak256(__global const u8* msgs, __global const u32* lens, u32 stride, __global u8* out, u32 n) {
    u32 i = get_global_id(0);
    if (i >= n) return;
    u32 len = lens[i] < IN_BUF ? lens[i] : IN_BUF;
    u8 m[IN_BUF];
    for (u32 j = 0; j < len; j++) m[j] = msgs[(u64)i*stride + j];
    u8 d[32];
    keccak256(m, len, d);
    for (int j = 0; j < 32; j++) out[(u64)i*32 + j] = d[j];
}

__kernel void k_hmac_sha512(__global const u8* keys, __global const u32* klens, u32 kstride,
                   __global const u8* msgs, __global const u32* mlens, u32 mstride,
                   __global u8* out, u32 n) {
    u32 i = get_global_id(0);
    if (i >= n) return;
    u32 klen = klens[i] < IN_BUF ? klens[i] : IN_BUF;
    u32 mlen = mlens[i] < IN_BUF ? mlens[i] : IN_BUF;
    u8 k[IN_BUF];
    for (u32 j = 0; j < klen; j++) k[j] = keys[(u64)i*kstride + j];
    u8 m[IN_BUF];
    for (u32 j = 0; j < mlen; j++) m[j] = msgs[(u64)i*mstride + j];
    u8 d[64];
    hmac_sha512(k, klen, m, mlen, d);
    for (int j = 0; j < 64; j++) out[(u64)i*64 + j] = d[j];
}

__kernel void k_pbkdf2(__global const u8* pws, __global const u32* pwlens, u32 pwstride,
              __global const u8* salts, __global const u32* saltlens, u32 sstride,
              u32 iters, __global u8* out, u32 n) {
    u32 i = get_global_id(0);
    if (i >= n) return;
    u32 plen = pwlens[i] < IN_BUF ? pwlens[i] : IN_BUF;
    u32 slen = saltlens[i] < IN_BUF ? saltlens[i] : IN_BUF;
    u8 pw[IN_BUF];
    for (u32 j = 0; j < plen; j++) pw[j] = pws[(u64)i*pwstride + j];
    u8 salt[IN_BUF];
    for (u32 j = 0; j < slen; j++) salt[j] = salts[(u64)i*sstride + j];
    u8 d[64];
    pbkdf2_hmac_sha512_64(pw, plen, salt, slen, iters, d);
    for (int j = 0; j < 64; j++) out[(u64)i*64 + j] = d[j];
}

// One 32-byte big-endian private key in -> 33-byte compressed pubkey out.
__kernel void k_pubkey(__global const u8* privs, __global u8* out, u32 n,
                       __global const apoint* gt) {
    u32 i = get_global_id(0);
    if (i >= n) return;
    u8 p[32];
    for (int j = 0; j < 32; j++) p[j] = privs[(u64)i*32 + j];
    u64 k[4]; be32_to_limbs(p, k);
    u8 pc[33];
    pubkey_compressed(k, pc, gt);
    for (int j = 0; j < 33; j++) out[(u64)i*33 + j] = pc[j];
}

// One 32-byte big-endian private key in -> 64-byte uncompressed X||Y out.
// Unlike the compressed form, this exposes every bit of Y, so it is what pins
// down the field arithmetic's canonical reduction on the Y coordinate.
__kernel void k_pubkey_xy(__global const u8* privs, __global u8* out, u32 n,
                          __global const apoint* gt) {
    u32 i = get_global_id(0);
    if (i >= n) return;
    u8 p[32];
    for (int j = 0; j < 32; j++) p[j] = privs[(u64)i*32 + j];
    u64 k[4]; be32_to_limbs(p, k);
    u8 px[64];
    pubkey_xy(k, px, gt);
    for (int j = 0; j < 64; j++) out[(u64)i*64 + j] = px[j];
}

// (a * b) mod p, all 32-byte big-endian. Debug helper for field arithmetic.
__kernel void k_fe_mul(__global const u8* a, __global const u8* b, __global u8* out, u32 n) {
    u32 i = get_global_id(0);
    if (i >= n) return;
    u8 ba[32], bb[32];
    for (int j = 0; j < 32; j++) ba[j] = a[(u64)i*32 + j];
    for (int j = 0; j < 32; j++) bb[j] = b[(u64)i*32 + j];
    u64 la[4], lb[4];
    be32_to_limbs(ba, la);
    be32_to_limbs(bb, lb);
    fe fa, fb, fr; fe_set(&fa, la); fe_set(&fb, lb);
    fe_mul(&fr, &fa, &fb);
    for (int j = 0; j < 4; j++)
        for (int k = 0; k < 8; k++) out[(u64)i*32 + j*8 + k] = (u8)(fr.n[3-j] >> (56 - k*8));
}

// One 64-byte seed in -> 20-byte Ethereum address (m/44'/60'/0'/0/0) out.
__kernel void k_seed_to_eth(__global const u8* seeds, __global u8* out, u32 n,
                            __global const apoint* gt) {
    u32 i = get_global_id(0);
    if (i >= n) return;
    u8 s[64];
    for (int j = 0; j < 64; j++) s[j] = seeds[(u64)i*64 + j];
    u8 a[20];
    seed_to_eth_address(s, a, gt);
    for (int j = 0; j < 20; j++) out[(u64)i*20 + j] = a[j];
}

// a^{-1} mod p, 32-byte big-endian. Debug helper.
__kernel void k_fe_inv(__global const u8* a, __global u8* out, u32 n) {
    u32 i = get_global_id(0);
    if (i >= n) return;
    u8 ba[32];
    for (int j = 0; j < 32; j++) ba[j] = a[(u64)i*32 + j];
    u64 la[4]; be32_to_limbs(ba, la);
    fe fa, fr; fe_set(&fa, la); fe_inv(&fr, &fa);
    for (int j = 0; j < 4; j++)
        for (int k = 0; k < 8; k++) out[(u64)i*32 + j*8 + k] = (u8)(fr.n[3-j] >> (56 - k*8));
}

// (a + b) mod n, all 32-byte big-endian.
__kernel void k_scalar_add(__global const u8* a, __global const u8* b, __global u8* out, u32 n) {
    u32 i = get_global_id(0);
    if (i >= n) return;
    u8 ba[32], bb[32];
    for (int j = 0; j < 32; j++) ba[j] = a[(u64)i*32 + j];
    for (int j = 0; j < 32; j++) bb[j] = b[(u64)i*32 + j];
    u64 la[4], lb[4], lr[4];
    be32_to_limbs(ba, la);
    be32_to_limbs(bb, lb);
    scalar_add_modn(la, lb, lr);
    for (int j = 0; j < 4; j++)
        for (int k = 0; k < 8; k++) out[(u64)i*32 + j*8 + k] = (u8)(lr[3-j] >> (56 - k*8));
}
