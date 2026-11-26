// SPDX-License-Identifier: MIT OR Apache-2.0
// Adapted from POMXARK/bcpack-gpu-sdk (4bca64a)

// Clean-room Xbox Series BCPack Full BC6H (Modes 1..14) GPU Block Rebuilder (HLSL SM 6.6)
// Reads 16-lane symbol rows (8 uints/row, 2 rows/block) emitted by bcpack_bc6h_rans.hlsl
// and writes complete 128-bit (16-byte) BC6H blocks using 256-block history & 255-entry dictionary.
// Endpoint bit packing follows Khronos ARB_texture_compression_bptc (BPTC_FLOAT / BC6H) Table.F.

cbuffer Bc6hRebuildCB : register(b0) {
    uint  g_NumChunks;
    uint3 _pad;
};

struct Bc6hChunkDesc {
    uint in_row_offset;
    uint blocks;
    uint out_block_offset;
    uint _pad0;
};

StructuredBuffer<uint>           g_InSyms    : register(t0); // 8 uints (16 uint16) per row
StructuredBuffer<Bc6hChunkDesc>  g_Chunks    : register(t1);
StructuredBuffer<uint>           g_Dict      : register(t2); // 255 * 15 uints

RWStructuredBuffer<uint4>        g_OutBlocks : register(u0); // 16 bytes per BC6H block
RWStructuredBuffer<uint>         g_Hist      : register(u1); // max_chunks * 256 * 15 uints

// 1-based modes 1..14; entry 0 unused.
static const uint BC6_BITS[15][4] = {
    {  0u,  0u,  0u,  0u },
    { 10u,  5u,  5u,  5u }, // 1  (MN 0)
    {  7u,  6u,  6u,  6u }, // 2  (MN 1)
    { 11u,  5u,  4u,  4u }, // 3  (MN 2)
    { 11u,  4u,  5u,  4u }, // 4  (MN 6)
    { 11u,  4u,  4u,  5u }, // 5  (MN 10)
    {  9u,  5u,  5u,  5u }, // 6  (MN 14)
    {  8u,  6u,  5u,  5u }, // 7  (MN 18)
    {  8u,  5u,  6u,  5u }, // 8  (MN 22)
    {  8u,  5u,  5u,  6u }, // 9  (MN 26)
    {  6u,  6u,  6u,  6u }, // 10 (MN 30)
    { 10u, 10u, 10u, 10u }, // 11 (MN 3)
    { 11u,  9u,  9u,  9u }, // 12 (MN 7)
    { 12u,  8u,  8u,  8u }, // 13 (MN 11)
    { 16u,  4u,  4u,  4u }  // 14 (MN 15)
};

static const uint BC6_MODE_CODE[15] = {
    0u, 0u, 1u, 2u, 6u, 10u, 14u, 18u, 22u, 26u, 30u, 3u, 7u, 11u, 15u
};

// Standard BPTC 2-subset anchor indices for 5-bit partition 0..31.
static const uint ANCHOR_2[32] = {
    15u,15u,15u,15u,15u,15u,15u,15u,15u,15u,15u,15u,15u,15u,15u,15u,
    15u, 2u, 8u, 2u, 2u, 8u, 8u,15u, 2u, 8u, 2u, 2u, 8u, 8u, 2u, 2u
};

uint read_sym(uint row, uint lane) {
    uint w = g_InSyms[row * 8u + (lane >> 1u)];
    return ((lane & 1u) == 0u) ? (w & 0xFFFFu) : (w >> 16u);
}

int unzigzag(uint v) {
    int iv = int(v);
    return ((iv & 1) != 0) ? -((iv + 1) >> 1) : (iv >> 1);
}

uint wide16(uint a, uint b) {
    uint acc = 0u;
    [unroll]
    for (uint k = 0u; k < 8u; ++k) {
        acc |= (((a >> k) & 1u) << (2u * k)) | (((b >> k) & 1u) << (2u * k + 1u));
    }
    return acc;
}

uint zz16(uint v) {
    return ((v >> 1u) ^ (((v & 1u) != 0u) ? 0xFFFFu : 0u)) & 0xFFFFu;
}

uint interleave4(uint even, uint odd) {
    uint o = 0u;
    [unroll]
    for (uint k = 0u; k < 4u; ++k) {
        o |= ((even >> k) & 1u) << (2u * k);
        o |= ((odd  >> k) & 1u) << (2u * k + 1u);
    }
    return o;
}

void deinterleave4(uint b, out uint even, out uint odd) {
    even = 0u;
    odd = 0u;
    [unroll]
    for (uint k = 0u; k < 4u; ++k) {
        even |= ((b >> (2u * k)) & 1u) << k;
        odd  |= ((b >> (2u * k + 1u)) & 1u) << k;
    }
}

void decode_refs(uint d, uint x, out uint er, out uint xr) {
    int delta = unzigzag(d & 255u);
    er = uint((int(x & 255u) + delta) & 255);
    xr = x & 255u;
}

void put_bits(inout uint4 v, inout uint at, uint val, uint n) {
    if (n > 0u && at < 128u) {
        uint masked = val & ((1u << n) - 1u);
        uint w = at >> 5u;
        uint b = at & 31u;
        if (w == 0u) {
            v.x |= (masked << b);
            if (b + n > 32u) v.y |= (masked >> (32u - b));
        } else if (w == 1u) {
            v.y |= (masked << b);
            if (b + n > 32u) v.z |= (masked >> (32u - b));
        } else if (w == 2u) {
            v.z |= (masked << b);
            if (b + n > 32u) v.w |= (masked >> (32u - b));
        } else {
            v.w |= (masked << b);
        }
    }
    at += n;
}

void load_state_words(uint chunk_id, uint i, uint r, out uint sw[15]) {
    if (r == 0u) {
        [unroll]
        for (uint k = 0u; k < 15u; ++k) sw[k] = 0u;
        sw[6] = 11u;
    } else if (r <= i) {
        uint base = (chunk_id * 256u + ((i - r) & 255u)) * 15u;
        [unroll]
        for (uint k = 0u; k < 15u; ++k) sw[k] = g_Hist[base + k];
    } else {
        uint base = (r - i - 1u) * 15u;
        [unroll]
        for (uint k = 0u; k < 15u; ++k) sw[k] = g_Dict[base + k];
    }
}

uint read_ep(uint sw[15], uint k, uint c) {
    uint idx = k * 3u + c;
    return (sw[idx >> 1u] >> ((idx & 1u) * 16u)) & 0xFFFFu;
}

int read_ix(uint sw[15], uint p) {
    uint w = sw[7u + (p >> 1u)];
    uint u = ((p & 1u) == 0u) ? (w & 0xFFFFu) : (w >> 16u);
    return int(u << 16u) >> 16;
}

int add_index_signed(int cur, uint bits, int refv, uint ref_bits) {
    uint up_cur = (bits == 0u) ? 0u : ((uint(cur) & 0xFFFFu) << (16u - bits));
    uint up_ref = (ref_bits == 0u) ? 0u : ((uint(refv) & 0xFFFFu) << (16u - ref_bits));
    uint sum16 = (up_cur + up_ref) & 0xFFFFu;
    int s16 = int(sum16 << 16u) >> 16;
    return (bits == 0u) ? 0 : (s16 >> int(16u - bits));
}

[numthreads(64, 1, 1)]
void CSMain(uint3 dtid : SV_GroupThreadID, uint3 gid : SV_GroupID) {
    uint chunk_id = gid.x * 64u + dtid.x;
    if (chunk_id >= g_NumChunks) {
        return;
    }

    Bc6hChunkDesc desc = g_Chunks[chunk_id];
    uint hist_chunk_base = chunk_id * 256u * 15u;

    for (uint i = 0u; i < desc.blocks; ++i) {
        uint r0 = desc.in_row_offset + 2u * i;
        uint r1 = r0 + 1u;

        uint er = 0u;
        uint xr = 0u;
        decode_refs(
            interleave4(read_sym(r0, 0u), read_sym(r1, 0u)),
            interleave4(read_sym(r0, 1u), read_sym(r1, 1u)),
            er, xr);

        uint mode = (read_sym(r0, 15u) + 1u) & 255u;
        if (mode < 1u || mode > 14u) {
            g_OutBlocks[desc.out_block_offset + i] = uint4(0u, 0u, 0u, 0u);
            uint hb0 = hist_chunk_base + (i & 255u) * 15u;
            [unroll]
            for (uint k = 0u; k < 15u; ++k) g_Hist[hb0 + k] = (k == 6u) ? 11u : 0u;
            continue;
        }

        bool two = (mode <= 10u);
        uint part = read_sym(r1, 15u) & 31u;
        uint b0 = BC6_BITS[mode][0];
        uint bc[3] = { BC6_BITS[mode][1], BC6_BITS[mode][2], BC6_BITS[mode][3] };
        uint sh0 = 16u - b0;
        uint shc[3] = { 16u - bc[0], 16u - bc[1], 16u - bc[2] };

        uint e0r[3] = {
            zz16(wide16(read_sym(r0, 2u) & 255u, read_sym(r1, 2u) & 255u)),
            zz16(wide16(read_sym(r0, 3u) & 255u, read_sym(r1, 3u) & 255u)),
            zz16(wide16(read_sym(r0, 4u) & 255u, read_sym(r1, 4u) & 255u))
        };
        uint e1r[3] = {
            zz16(wide16(read_sym(r0, 5u) & 255u, read_sym(r1, 5u) & 255u)),
            zz16(wide16(read_sym(r0, 6u) & 255u, read_sym(r1, 6u) & 255u)),
            zz16(wide16(read_sym(r0, 7u) & 255u, read_sym(r1, 7u) & 255u))
        };
        uint e2r[3] = {
            two ? zz16(read_sym(r0, 8u)  & 255u) : 0u,
            two ? zz16(read_sym(r0, 9u)  & 255u) : 0u,
            two ? zz16(read_sym(r0, 10u) & 255u) : 0u
        };
        uint e3r[3] = {
            two ? zz16(read_sym(r1, 8u)  & 255u) : 0u,
            two ? zz16(read_sym(r1, 9u)  & 255u) : 0u,
            two ? zz16(read_sym(r1, 10u) & 255u) : 0u
        };

        uint ref_ep[15];
        load_state_words(chunk_id, i, er, ref_ep);
        uint ref_ix[15];
        load_state_words(chunk_id, i, xr, ref_ix);

        uint ep[4][3];
        [unroll]
        for (uint c = 0u; c < 3u; ++c) {
            uint m0 = (0xFFFFu << sh0) & 0xFFFFu;
            uint mc = (0xFFFFu << shc[c]) & 0xFFFFu;
            ep[0][c] = (read_ep(ref_ep, 0u, c) + ((e0r[c] << sh0) & 0xFFFFu)) & m0;
            ep[1][c] = (read_ep(ref_ep, 1u, c) + ((e1r[c] << shc[c]) & 0xFFFFu)) & mc;
            if (two) {
                ep[2][c] = (read_ep(ref_ep, 2u, c) + ((e2r[c] << shc[c]) & 0xFFFFu)) & mc;
                ep[3][c] = (read_ep(ref_ep, 3u, c) + ((e3r[c] << shc[c]) & 0xFFFFu)) & mc;
            } else {
                ep[2][c] = ((ep[0][c] >> sh0) << shc[c]) & 0xFFFFu;
                ep[3][c] = ep[1][c];
            }
        }

        uint ibits = two ? 3u : 4u;
        uint rmode = ref_ix[6] & 255u;
        uint rbits = (rmode >= 1u && rmode <= 10u) ? 3u : ((rmode >= 11u && rmode <= 14u) ? 4u : 0u);
        int ix_v[16];
        [unroll]
        for (uint row = 0u; row < 4u; ++row) {
            uint ln = 11u + row;
            [unroll]
            for (uint half = 0u; half < 2u; ++half) {
                uint ev = 0u;
                uint od = 0u;
                deinterleave4(read_sym((half == 0u) ? r0 : r1, ln), ev, od);
                uint left_c  = (row < 3u) ? od : ev;
                uint right_c = (row < 3u) ? ev : od;
                uint px0 = 4u * row + 2u * half;
                uint px1 = px0 + 1u;
                ix_v[px0] = add_index_signed(unzigzag(left_c),  ibits, read_ix(ref_ix, px0), rbits);
                ix_v[px1] = add_index_signed(unzigzag(right_c), ibits, read_ix(ref_ix, px1), rbits);
            }
        }

        // Pack 15-uint history state
        uint st_w[15];
        [unroll]
        for (uint k = 0u; k < 15u; ++k) st_w[k] = 0u;
        [unroll]
        for (uint ek = 0u; ek < 4u; ++ek) {
            [unroll]
            for (uint c2 = 0u; c2 < 3u; ++c2) {
                uint idx = ek * 3u + c2;
                st_w[idx >> 1u] |= (ep[ek][c2] & 0xFFFFu) << ((idx & 1u) * 16u);
            }
        }
        st_w[6] = mode;
        [unroll]
        for (uint p = 0u; p < 16u; ++p) {
            st_w[7u + (p >> 1u)] |= (uint(ix_v[p]) & 0xFFFFu) << ((p & 1u) * 16u);
        }

        // Field helper: working values flip top bit
        uint r0f = (ep[0][0] ^ 0x8000u) >> sh0;
        uint g0f = (ep[0][1] ^ 0x8000u) >> sh0;
        uint b0f = (ep[0][2] ^ 0x8000u) >> sh0;
        uint r1f = (ep[1][0] ^ 0x8000u) >> shc[0];
        uint g1f = (ep[1][1] ^ 0x8000u) >> shc[1];
        uint b1f = (ep[1][2] ^ 0x8000u) >> shc[2];
        uint r2f = (ep[2][0] ^ 0x8000u) >> shc[0];
        uint g2f = (ep[2][1] ^ 0x8000u) >> shc[1];
        uint b2f = (ep[2][2] ^ 0x8000u) >> shc[2];
        uint r3f = (ep[3][0] ^ 0x8000u) >> shc[0];
        uint g3f = (ep[3][1] ^ 0x8000u) >> shc[1];
        uint b3f = (ep[3][2] ^ 0x8000u) >> shc[2];

        uint4 out_blk = uint4(0u, 0u, 0u, 0u);
        uint at = 0u;
        put_bits(out_blk, at, BC6_MODE_CODE[mode], (mode <= 2u) ? 2u : 5u);

        if (mode == 1u) {
            put_bits(out_blk, at, (g2f >> 4u) & 1u, 1u);
            put_bits(out_blk, at, (b2f >> 4u) & 1u, 1u);
            put_bits(out_blk, at, (b3f >> 4u) & 1u, 1u);
            put_bits(out_blk, at, r0f, 10u);
            put_bits(out_blk, at, g0f, 10u);
            put_bits(out_blk, at, b0f, 10u);
            put_bits(out_blk, at, r1f, 5u);
            put_bits(out_blk, at, (g3f >> 4u) & 1u, 1u);
            put_bits(out_blk, at, g2f, 4u);
            put_bits(out_blk, at, g1f, 5u);
            put_bits(out_blk, at, (b3f >> 0u) & 1u, 1u);
            put_bits(out_blk, at, g3f, 4u);
            put_bits(out_blk, at, b1f, 5u);
            put_bits(out_blk, at, (b3f >> 1u) & 1u, 1u);
            put_bits(out_blk, at, b2f, 4u);
            put_bits(out_blk, at, r2f, 5u);
            put_bits(out_blk, at, (b3f >> 2u) & 1u, 1u);
            put_bits(out_blk, at, r3f, 5u);
            put_bits(out_blk, at, (b3f >> 3u) & 1u, 1u);
        } else if (mode == 2u) {
            put_bits(out_blk, at, (g2f >> 5u) & 1u, 1u);
            put_bits(out_blk, at, (g3f >> 4u) & 1u, 1u);
            put_bits(out_blk, at, (g3f >> 5u) & 1u, 1u);
            put_bits(out_blk, at, r0f, 7u);
            put_bits(out_blk, at, (b3f >> 0u) & 1u, 1u);
            put_bits(out_blk, at, (b3f >> 1u) & 1u, 1u);
            put_bits(out_blk, at, (b2f >> 4u) & 1u, 1u);
            put_bits(out_blk, at, g0f, 7u);
            put_bits(out_blk, at, (b2f >> 5u) & 1u, 1u);
            put_bits(out_blk, at, (b3f >> 2u) & 1u, 1u);
            put_bits(out_blk, at, (g2f >> 4u) & 1u, 1u);
            put_bits(out_blk, at, b0f, 7u);
            put_bits(out_blk, at, (b3f >> 3u) & 1u, 1u);
            put_bits(out_blk, at, (b3f >> 5u) & 1u, 1u);
            put_bits(out_blk, at, (b3f >> 4u) & 1u, 1u);
            put_bits(out_blk, at, r1f, 6u);
            put_bits(out_blk, at, g2f, 4u);
            put_bits(out_blk, at, g1f, 6u);
            put_bits(out_blk, at, g3f, 4u);
            put_bits(out_blk, at, b1f, 6u);
            put_bits(out_blk, at, b2f, 4u);
            put_bits(out_blk, at, r2f, 6u);
            put_bits(out_blk, at, r3f, 6u);
        } else if (mode == 3u) {
            put_bits(out_blk, at, r0f, 10u);
            put_bits(out_blk, at, g0f, 10u);
            put_bits(out_blk, at, b0f, 10u);
            put_bits(out_blk, at, r1f, 5u);
            put_bits(out_blk, at, (r0f >> 10u) & 1u, 1u);
            put_bits(out_blk, at, g2f, 4u);
            put_bits(out_blk, at, g1f, 4u);
            put_bits(out_blk, at, (g0f >> 10u) & 1u, 1u);
            put_bits(out_blk, at, (b3f >> 0u) & 1u, 1u);
            put_bits(out_blk, at, g3f, 4u);
            put_bits(out_blk, at, b1f, 4u);
            put_bits(out_blk, at, (b0f >> 10u) & 1u, 1u);
            put_bits(out_blk, at, (b3f >> 1u) & 1u, 1u);
            put_bits(out_blk, at, b2f, 4u);
            put_bits(out_blk, at, r2f, 5u);
            put_bits(out_blk, at, (b3f >> 2u) & 1u, 1u);
            put_bits(out_blk, at, r3f, 5u);
            put_bits(out_blk, at, (b3f >> 3u) & 1u, 1u);
        } else if (mode == 4u) {
            put_bits(out_blk, at, r0f, 10u);
            put_bits(out_blk, at, g0f, 10u);
            put_bits(out_blk, at, b0f, 10u);
            put_bits(out_blk, at, r1f, 4u);
            put_bits(out_blk, at, (r0f >> 10u) & 1u, 1u);
            put_bits(out_blk, at, (g3f >> 4u) & 1u, 1u);
            put_bits(out_blk, at, g2f, 4u);
            put_bits(out_blk, at, g1f, 5u);
            put_bits(out_blk, at, (g0f >> 10u) & 1u, 1u);
            put_bits(out_blk, at, g3f, 4u);
            put_bits(out_blk, at, b1f, 4u);
            put_bits(out_blk, at, (b0f >> 10u) & 1u, 1u);
            put_bits(out_blk, at, (b3f >> 1u) & 1u, 1u);
            put_bits(out_blk, at, b2f, 4u);
            put_bits(out_blk, at, r2f, 4u);
            put_bits(out_blk, at, (b3f >> 0u) & 1u, 1u);
            put_bits(out_blk, at, (b3f >> 2u) & 1u, 1u);
            put_bits(out_blk, at, r3f, 4u);
            put_bits(out_blk, at, (g2f >> 4u) & 1u, 1u);
            put_bits(out_blk, at, (b3f >> 3u) & 1u, 1u);
        } else if (mode == 5u) {
            put_bits(out_blk, at, r0f, 10u);
            put_bits(out_blk, at, g0f, 10u);
            put_bits(out_blk, at, b0f, 10u);
            put_bits(out_blk, at, r1f, 4u);
            put_bits(out_blk, at, (r0f >> 10u) & 1u, 1u);
            put_bits(out_blk, at, (b2f >> 4u) & 1u, 1u);
            put_bits(out_blk, at, g2f, 4u);
            put_bits(out_blk, at, g1f, 4u);
            put_bits(out_blk, at, (g0f >> 10u) & 1u, 1u);
            put_bits(out_blk, at, (b3f >> 0u) & 1u, 1u);
            put_bits(out_blk, at, g3f, 4u);
            put_bits(out_blk, at, b1f, 5u);
            put_bits(out_blk, at, (b0f >> 10u) & 1u, 1u);
            put_bits(out_blk, at, b2f, 4u);
            put_bits(out_blk, at, r2f, 4u);
            put_bits(out_blk, at, (b3f >> 1u) & 1u, 1u);
            put_bits(out_blk, at, (b3f >> 2u) & 1u, 1u);
            put_bits(out_blk, at, r3f, 4u);
            put_bits(out_blk, at, (b3f >> 4u) & 1u, 1u);
            put_bits(out_blk, at, (b3f >> 3u) & 1u, 1u);
        } else if (mode == 6u) {
            put_bits(out_blk, at, r0f, 9u);
            put_bits(out_blk, at, (b2f >> 4u) & 1u, 1u);
            put_bits(out_blk, at, g0f, 9u);
            put_bits(out_blk, at, (g2f >> 4u) & 1u, 1u);
            put_bits(out_blk, at, b0f, 9u);
            put_bits(out_blk, at, (b3f >> 4u) & 1u, 1u);
            put_bits(out_blk, at, r1f, 5u);
            put_bits(out_blk, at, (g3f >> 4u) & 1u, 1u);
            put_bits(out_blk, at, g2f, 4u);
            put_bits(out_blk, at, g1f, 5u);
            put_bits(out_blk, at, (b3f >> 0u) & 1u, 1u);
            put_bits(out_blk, at, g3f, 4u);
            put_bits(out_blk, at, b1f, 5u);
            put_bits(out_blk, at, (b3f >> 1u) & 1u, 1u);
            put_bits(out_blk, at, b2f, 4u);
            put_bits(out_blk, at, r2f, 5u);
            put_bits(out_blk, at, (b3f >> 2u) & 1u, 1u);
            put_bits(out_blk, at, r3f, 5u);
            put_bits(out_blk, at, (b3f >> 3u) & 1u, 1u);
        } else if (mode == 7u) {
            put_bits(out_blk, at, r0f, 8u);
            put_bits(out_blk, at, (g3f >> 4u) & 1u, 1u);
            put_bits(out_blk, at, (b2f >> 4u) & 1u, 1u);
            put_bits(out_blk, at, g0f, 8u);
            put_bits(out_blk, at, (b3f >> 2u) & 1u, 1u);
            put_bits(out_blk, at, (g2f >> 4u) & 1u, 1u);
            put_bits(out_blk, at, b0f, 8u);
            put_bits(out_blk, at, (b3f >> 3u) & 1u, 1u);
            put_bits(out_blk, at, (b3f >> 4u) & 1u, 1u);
            put_bits(out_blk, at, r1f, 6u);
            put_bits(out_blk, at, g2f, 4u);
            put_bits(out_blk, at, g1f, 5u);
            put_bits(out_blk, at, (b3f >> 0u) & 1u, 1u);
            put_bits(out_blk, at, g3f, 4u);
            put_bits(out_blk, at, b1f, 5u);
            put_bits(out_blk, at, (b3f >> 1u) & 1u, 1u);
            put_bits(out_blk, at, b2f, 4u);
            put_bits(out_blk, at, r2f, 6u);
            put_bits(out_blk, at, r3f, 6u);
        } else if (mode == 8u) {
            put_bits(out_blk, at, r0f, 8u);
            put_bits(out_blk, at, (b3f >> 0u) & 1u, 1u);
            put_bits(out_blk, at, (b2f >> 4u) & 1u, 1u);
            put_bits(out_blk, at, g0f, 8u);
            put_bits(out_blk, at, (g2f >> 5u) & 1u, 1u);
            put_bits(out_blk, at, (g2f >> 4u) & 1u, 1u);
            put_bits(out_blk, at, b0f, 8u);
            put_bits(out_blk, at, (g3f >> 5u) & 1u, 1u);
            put_bits(out_blk, at, (b3f >> 4u) & 1u, 1u);
            put_bits(out_blk, at, r1f, 5u);
            put_bits(out_blk, at, (g3f >> 4u) & 1u, 1u);
            put_bits(out_blk, at, g2f, 4u);
            put_bits(out_blk, at, g1f, 6u);
            put_bits(out_blk, at, g3f, 4u);
            put_bits(out_blk, at, b1f, 5u);
            put_bits(out_blk, at, (b3f >> 1u) & 1u, 1u);
            put_bits(out_blk, at, b2f, 4u);
            put_bits(out_blk, at, r2f, 5u);
            put_bits(out_blk, at, (b3f >> 2u) & 1u, 1u);
            put_bits(out_blk, at, r3f, 5u);
            put_bits(out_blk, at, (b3f >> 3u) & 1u, 1u);
        } else if (mode == 9u) {
            put_bits(out_blk, at, r0f, 8u);
            put_bits(out_blk, at, (b3f >> 1u) & 1u, 1u);
            put_bits(out_blk, at, (b2f >> 4u) & 1u, 1u);
            put_bits(out_blk, at, g0f, 8u);
            put_bits(out_blk, at, (b2f >> 5u) & 1u, 1u);
            put_bits(out_blk, at, (g2f >> 4u) & 1u, 1u);
            put_bits(out_blk, at, b0f, 8u);
            put_bits(out_blk, at, (b3f >> 5u) & 1u, 1u);
            put_bits(out_blk, at, (b3f >> 4u) & 1u, 1u);
            put_bits(out_blk, at, r1f, 5u);
            put_bits(out_blk, at, (g3f >> 4u) & 1u, 1u);
            put_bits(out_blk, at, g2f, 4u);
            put_bits(out_blk, at, g1f, 5u);
            put_bits(out_blk, at, (b3f >> 0u) & 1u, 1u);
            put_bits(out_blk, at, g3f, 4u);
            put_bits(out_blk, at, b1f, 6u);
            put_bits(out_blk, at, b2f, 4u);
            put_bits(out_blk, at, r2f, 5u);
            put_bits(out_blk, at, (b3f >> 2u) & 1u, 1u);
            put_bits(out_blk, at, r3f, 5u);
            put_bits(out_blk, at, (b3f >> 3u) & 1u, 1u);
        } else if (mode == 10u) {
            put_bits(out_blk, at, r0f, 6u);
            put_bits(out_blk, at, (g3f >> 4u) & 1u, 1u);
            put_bits(out_blk, at, (b3f >> 0u) & 1u, 1u);
            put_bits(out_blk, at, (b3f >> 1u) & 1u, 1u);
            put_bits(out_blk, at, (b2f >> 4u) & 1u, 1u);
            put_bits(out_blk, at, g0f, 6u);
            put_bits(out_blk, at, (g2f >> 5u) & 1u, 1u);
            put_bits(out_blk, at, (b2f >> 5u) & 1u, 1u);
            put_bits(out_blk, at, (b3f >> 2u) & 1u, 1u);
            put_bits(out_blk, at, (g2f >> 4u) & 1u, 1u);
            put_bits(out_blk, at, b0f, 6u);
            put_bits(out_blk, at, (g3f >> 5u) & 1u, 1u);
            put_bits(out_blk, at, (b3f >> 3u) & 1u, 1u);
            put_bits(out_blk, at, (b3f >> 5u) & 1u, 1u);
            put_bits(out_blk, at, (b3f >> 4u) & 1u, 1u);
            put_bits(out_blk, at, r1f, 6u);
            put_bits(out_blk, at, g2f, 4u);
            put_bits(out_blk, at, g1f, 6u);
            put_bits(out_blk, at, g3f, 4u);
            put_bits(out_blk, at, b1f, 6u);
            put_bits(out_blk, at, b2f, 4u);
            put_bits(out_blk, at, r2f, 6u);
            put_bits(out_blk, at, r3f, 6u);
        } else if (mode == 11u) {
            put_bits(out_blk, at, r0f, 10u);
            put_bits(out_blk, at, g0f, 10u);
            put_bits(out_blk, at, b0f, 10u);
            put_bits(out_blk, at, r1f, 10u);
            put_bits(out_blk, at, g1f, 10u);
            put_bits(out_blk, at, b1f, 10u);
        } else if (mode == 12u) {
            put_bits(out_blk, at, r0f, 10u);
            put_bits(out_blk, at, g0f, 10u);
            put_bits(out_blk, at, b0f, 10u);
            put_bits(out_blk, at, r1f, 9u);
            put_bits(out_blk, at, (r0f >> 10u) & 1u, 1u);
            put_bits(out_blk, at, g1f, 9u);
            put_bits(out_blk, at, (g0f >> 10u) & 1u, 1u);
            put_bits(out_blk, at, b1f, 9u);
            put_bits(out_blk, at, (b0f >> 10u) & 1u, 1u);
        } else if (mode == 13u) {
            put_bits(out_blk, at, r0f, 10u);
            put_bits(out_blk, at, g0f, 10u);
            put_bits(out_blk, at, b0f, 10u);
            put_bits(out_blk, at, r1f, 8u);
            put_bits(out_blk, at, r0f >> 10u, 2u);
            put_bits(out_blk, at, g1f, 8u);
            put_bits(out_blk, at, g0f >> 10u, 2u);
            put_bits(out_blk, at, b1f, 8u);
            put_bits(out_blk, at, b0f >> 10u, 2u);
        } else { // mode == 14u
            put_bits(out_blk, at, r0f, 10u);
            put_bits(out_blk, at, g0f, 10u);
            put_bits(out_blk, at, b0f, 10u);
            put_bits(out_blk, at, r1f, 4u);
            put_bits(out_blk, at, r0f >> 10u, 6u);
            put_bits(out_blk, at, g1f, 4u);
            put_bits(out_blk, at, g0f >> 10u, 6u);
            put_bits(out_blk, at, b1f, 4u);
            put_bits(out_blk, at, b0f >> 10u, 6u);
        }

        if (two) {
            put_bits(out_blk, at, part, 5u);
        }

        uint anchor = two ? ANCHOR_2[part] : 0u;
        uint ix_bias = 1u << (ibits - 1u);
        for (uint p = 0u; p < 16u; ++p) {
            uint is_anchor = (p == 0u || (two && p == anchor)) ? 1u : 0u;
            uint n = ibits - is_anchor;
            uint v = uint(ix_v[p] + int(ix_bias));
            put_bits(out_blk, at, v, n);
        }

        g_OutBlocks[desc.out_block_offset + i] = out_blk;

        uint hb = hist_chunk_base + (i & 255u) * 15u;
        [unroll]
        for (uint k2 = 0u; k2 < 15u; ++k2) {
            g_Hist[hb + k2] = st_w[k2];
        }
    }
}
