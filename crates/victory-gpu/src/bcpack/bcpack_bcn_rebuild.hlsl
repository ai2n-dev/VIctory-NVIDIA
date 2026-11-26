// SPDX-License-Identifier: MIT OR Apache-2.0
// Adapted from POMXARK/bcpack-gpu-sdk (4bca64a)

// Clean-room Xbox Series BCPack (BC1 / BC3 / BC4 / BC5) GPU Block Rebuilder (HLSL SM 6.6)
// Exact bit-for-bit parity with Ahmet-Dedeler/bcpack-rs & DirectStorage BCPack format.
// Dispatches 1 thread per chunk over the 6-uint/row symbols produced by bcpack_bcn_rans.hlsl.

cbuffer BcnRebuildCB : register(b0) {
    uint  g_NumChunks;
    uint3 _pad;
};

struct BcnChunkDesc {
    uint in_row_offset;
    uint blocks;
    uint out_byte_offset;
    uint mode; // 0 = BC3, 1 = BC4, 2 = BC5, 3 = BC1
};

StructuredBuffer<uint>         g_InSyms : register(t0); // 6 uints (12 uint16) per row
StructuredBuffer<BcnChunkDesc> g_Chunks : register(t1);
StructuredBuffer<uint>         g_Pairs  : register(t2); // 3 * 64 entries: p0 | (p1 << 8)
StructuredBuffer<uint>         g_Quads  : register(t3); // 512 entries: 0..255 BC3, 256..511 BC1
StructuredBuffer<uint>         g_Dict   : register(t4); // 4 * 255 * 6 uints (packed State)

RWByteAddressBuffer            g_OutBlocks : register(u0); // 8B/block (BC4) or 16B/block (BC3/BC5)
RWStructuredBuffer<uint>       g_Hist      : register(u1); // max_chunks * 256 * 6 uints

static const uint ALPHA_CODE[8]  = { 0u, 2u, 3u, 4u, 5u, 6u, 7u, 1u };
static const uint COLOUR_CODE[4] = { 0u, 2u, 3u, 1u };
static const uint EP_MASK_BC3[8] = { 255u, 255u, 31u, 63u, 31u, 31u, 63u, 31u };

uint read_sym(uint row, uint lane) {
    uint w = g_InSyms[row * 6u + (lane >> 1u)];
    return ((lane & 1u) == 0u) ? (w & 0xFFFFu) : (w >> 16u);
}

int unzigzag(uint v) {
    int iv = int(v);
    return ((iv & 1) != 0) ? -((iv + 1) >> 1) : (iv >> 1);
}

uint residual_bits(uint v, uint bits) {
    uint mask = (1u << bits) - 1u;
    return uint(unzigzag(v)) & mask;
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

void decode_refs(uint d, uint x, out uint er, out uint xr) {
    int delta = unzigzag(d & 255u);
    er = uint((int(x & 255u) + delta) & 255);
    xr = x & 255u;
}

void put_bits_u2(inout uint lo, inout uint hi, uint bit_pos, uint val, uint n) {
    if (bit_pos < 32u) {
        lo |= (val << bit_pos);
        if (bit_pos + n > 32u) {
            hi |= (val >> (32u - bit_pos));
        }
    } else {
        hi |= (val << (bit_pos - 32u));
    }
}

uint2 pack_alpha(uint e0, uint e1, uint ix_words[2]) {
    uint lo = (e0 ^ 0x80u) | ((e1 ^ 0x80u) << 8u);
    uint hi = 0u;
    [unroll]
    for (uint p = 0u; p < 16u; ++p) {
        uint pos = (ix_words[p >> 3u] >> ((p & 7u) * 4u)) & 7u;
        uint code = ALPHA_CODE[pos];
        put_bits_u2(lo, hi, 16u + 3u * p, code, 3u);
    }
    return uint2(lo, hi);
}

uint2 pack_colour(uint ep[8], uint ix_words[2]) {
    uint c0 = (ep[2] << 11u) | (ep[3] << 5u) | ep[4];
    uint c1 = (ep[5] << 11u) | (ep[6] << 5u) | ep[7];
    uint w = 0u;
    [unroll]
    for (uint p = 0u; p < 16u; ++p) {
        uint pos = (ix_words[p >> 3u] >> ((p & 7u) * 4u)) & 3u;
        w |= COLOUR_CODE[pos] << (2u * p);
    }
    return uint2(c0 | (c1 << 16u), w);
}

void load_state_from_dict(uint dict_base, out uint sw[6]) {
    [unroll]
    for (uint k = 0u; k < 6u; ++k) {
        sw[k] = g_Dict[dict_base + k];
    }
}

void load_state_from_hist(uint hist_base, out uint sw[6]) {
    [unroll]
    for (uint k = 0u; k < 6u; ++k) {
        sw[k] = g_Hist[hist_base + k];
    }
}

void load_ref_state(uint chunk_id, uint mode, uint i, uint r, out uint sw[6]) {
    if (r == 0u) {
        [unroll]
        for (uint k = 0u; k < 6u; ++k) sw[k] = 0u;
    } else if (r <= i) {
        uint slot = (i - r) & 255u;
        load_state_from_hist((chunk_id * 256u + slot) * 6u, sw);
    } else {
        uint d_idx = r - i - 1u;
        load_state_from_dict((mode * 255u + d_idx) * 6u, sw);
    }
}

uint get_ep_from_words(uint w0, uint w1, uint k) {
    uint w = (k < 4u) ? w0 : w1;
    return (w >> ((k & 3u) * 8u)) & 0xFFu;
}

[numthreads(64, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    uint chunk_id = dtid.x;
    if (chunk_id >= g_NumChunks) {
        return;
    }

    BcnChunkDesc desc = g_Chunks[chunk_id];
    uint mode = desc.mode;
    uint pair_base = min(mode, 2u) * 64u;
    uint hist_chunk_base = chunk_id * 256u * 6u;

    for (uint i = 0u; i < desc.blocks; ++i) {
        uint ep[8];
        [unroll]
        for (uint k = 0u; k < 8u; ++k) ep[k] = 0u;
        uint ix_w[4] = { 0u, 0u, 0u, 0u };

        uint er = 0u;
        uint xr = 0u;

        if (mode == 1u) {
            // BC4: 1 symbol row per block (8B)
            uint row = desc.in_row_offset + i;
            decode_refs(read_sym(row, 0u), read_sym(row, 1u), er, xr);
            ep[0] = residual_bits(read_sym(row, 2u), 8u);
            ep[1] = residual_bits(read_sym(row, 3u), 8u);
            [unroll]
            for (uint l = 0u; l < 8u; ++l) {
                uint s = read_sym(row, 4u + l) & 63u;
                uint pr = g_Pairs[pair_base + s];
                uint p0 = 2u * l;
                uint p1 = p0 + 1u;
                ix_w[p0 >> 3u] |= (pr & 7u) << ((p0 & 7u) * 4u);
                ix_w[p1 >> 3u] |= ((pr >> 8u) & 7u) << ((p1 & 7u) * 4u);
            }
        } else if (mode == 2u) {
            // BC5: 2 symbol rows per block (16B)
            uint r0 = desc.in_row_offset + 2u * i;
            uint r1 = r0 + 1u;
            decode_refs(
                interleave4(read_sym(r0, 0u), read_sym(r1, 0u)),
                interleave4(read_sym(r0, 1u), read_sym(r1, 1u)),
                er, xr);
            ep[0] = residual_bits(read_sym(r0, 2u), 8u);
            ep[1] = residual_bits(read_sym(r1, 2u), 8u);
            ep[2] = residual_bits(read_sym(r0, 3u), 8u);
            ep[3] = residual_bits(read_sym(r1, 3u), 8u);
            [unroll]
            for (uint l = 0u; l < 4u; ++l) {
                uint pr_re = g_Pairs[pair_base + (read_sym(r0, 4u + l) & 63u)];
                uint pr_ro = g_Pairs[pair_base + (read_sym(r1, 4u + l) & 63u)];
                uint b0 = 4u * l;
                ix_w[b0 >> 3u]       |= (pr_re & 7u)        << (((b0)      & 7u) * 4u);
                ix_w[(b0 + 1u) >> 3u]|= ((pr_re >> 8u) & 7u)<< (((b0 + 1u) & 7u) * 4u);
                ix_w[(b0 + 2u) >> 3u]|= (pr_ro & 7u)        << (((b0 + 2u) & 7u) * 4u);
                ix_w[(b0 + 3u) >> 3u]|= ((pr_ro >> 8u) & 7u)<< (((b0 + 3u) & 7u) * 4u);

                uint pr_ge = g_Pairs[pair_base + (read_sym(r0, 8u + l) & 63u)];
                uint pr_go = g_Pairs[pair_base + (read_sym(r1, 8u + l) & 63u)];
                uint g0 = 16u + 4u * l;
                ix_w[g0 >> 3u]       |= (pr_ge & 7u)        << (((g0)      & 7u) * 4u);
                ix_w[(g0 + 1u) >> 3u]|= ((pr_ge >> 8u) & 7u)<< (((g0 + 1u) & 7u) * 4u);
                ix_w[(g0 + 2u) >> 3u]|= (pr_go & 7u)        << (((g0 + 2u) & 7u) * 4u);
                ix_w[(g0 + 3u) >> 3u]|= ((pr_go >> 8u) & 7u)<< (((g0 + 3u) & 7u) * 4u);
            }
        } else if (mode == 3u) {
            // BC1: 1 symbol row per block (8B)
            uint row = desc.in_row_offset + i;
            decode_refs(read_sym(row, 0u), read_sym(row, 1u), er, xr);
            ep[2] = residual_bits(read_sym(row, 2u), 5u);
            ep[3] = residual_bits(read_sym(row, 3u), 6u);
            ep[4] = residual_bits(read_sym(row, 4u), 5u);
            ep[5] = residual_bits(read_sym(row, 5u), 5u);
            ep[6] = residual_bits(read_sym(row, 6u), 6u);
            ep[7] = residual_bits(read_sym(row, 7u), 5u);

            uint qs[4] = {
                read_sym(row, 8u)  & 255u,
                read_sym(row, 9u)  & 255u,
                read_sym(row, 10u) & 255u,
                read_sym(row, 11u) & 255u
            };
            [unroll]
            for (uint q = 0u; q < 4u; ++q) {
                uint qv = g_Quads[256u + qs[q]];
                uint ox = (q & 1u) * 2u;
                uint oy = (q >> 1u) * 2u;
                uint p00 = 16u + 4u * oy + ox;
                uint p10 = p00 + 1u;
                uint p01 = p00 + 4u;
                uint p11 = p00 + 5u;
                ix_w[p00 >> 3u] |= ((qv >> 0u)  & 3u) << ((p00 & 7u) * 4u);
                ix_w[p10 >> 3u] |= ((qv >> 8u)  & 3u) << ((p10 & 7u) * 4u);
                ix_w[p01 >> 3u] |= ((qv >> 16u) & 3u) << ((p01 & 7u) * 4u);
                ix_w[p11 >> 3u] |= ((qv >> 24u) & 3u) << ((p11 & 7u) * 4u);
            }
        } else {
            // BC3: 2 symbol rows per block (16B)
            uint r0 = desc.in_row_offset + 2u * i;
            uint r1 = r0 + 1u;
            decode_refs(
                interleave4(read_sym(r0, 0u), read_sym(r1, 0u)),
                interleave4(read_sym(r0, 1u), read_sym(r1, 1u)),
                er, xr);
            ep[0] = residual_bits(read_sym(r0, 2u), 8u);
            ep[1] = residual_bits(read_sym(r1, 2u), 8u);
            [unroll]
            for (uint l = 0u; l < 4u; ++l) {
                uint pr_e = g_Pairs[pair_base + (read_sym(r0, 3u + l) & 63u)];
                uint pr_o = g_Pairs[pair_base + (read_sym(r1, 3u + l) & 63u)];
                uint a0 = 4u * l;
                ix_w[a0 >> 3u]       |= (pr_e & 7u)        << (((a0)      & 7u) * 4u);
                ix_w[(a0 + 1u) >> 3u]|= ((pr_e >> 8u) & 7u)<< (((a0 + 1u) & 7u) * 4u);
                ix_w[(a0 + 2u) >> 3u]|= (pr_o & 7u)        << (((a0 + 2u) & 7u) * 4u);
                ix_w[(a0 + 3u) >> 3u]|= ((pr_o >> 8u) & 7u)<< (((a0 + 3u) & 7u) * 4u);
            }
            ep[2] = residual_bits(read_sym(r0, 7u), 5u);
            ep[5] = residual_bits(read_sym(r1, 7u), 5u);
            ep[3] = residual_bits(read_sym(r0, 8u), 6u);
            ep[6] = residual_bits(read_sym(r1, 8u), 6u);
            ep[4] = residual_bits(read_sym(r0, 9u), 5u);
            ep[7] = residual_bits(read_sym(r1, 9u), 5u);

            uint qs[4] = {
                read_sym(r0, 10u) & 255u,
                read_sym(r1, 10u) & 255u,
                read_sym(r0, 11u) & 255u,
                read_sym(r1, 11u) & 255u
            };
            [unroll]
            for (uint q = 0u; q < 4u; ++q) {
                uint qv = g_Quads[qs[q]];
                uint ox = (q & 1u) * 2u;
                uint oy = (q >> 1u) * 2u;
                uint p00 = 16u + 4u * oy + ox;
                uint p10 = p00 + 1u;
                uint p01 = p00 + 4u;
                uint p11 = p00 + 5u;
                ix_w[p00 >> 3u] |= ((qv >> 0u)  & 3u) << ((p00 & 7u) * 4u);
                ix_w[p10 >> 3u] |= ((qv >> 8u)  & 3u) << ((p10 & 7u) * 4u);
                ix_w[p01 >> 3u] |= ((qv >> 16u) & 3u) << ((p01 & 7u) * 4u);
                ix_w[p11 >> 3u] |= ((qv >> 24u) & 3u) << ((p11 & 7u) * 4u);
            }
        }

        // Apply endpoint reference
        uint ref_ep[6];
        load_ref_state(chunk_id, mode, i, er, ref_ep);
        uint ne = (mode == 1u) ? 2u : ((mode == 2u) ? 4u : 8u);
        [unroll]
        for (uint e = 0u; e < 8u; ++e) {
            if (e < ne) {
                uint m = (mode == 0u || mode == 3u) ? EP_MASK_BC3[e] : 255u;
                ep[e] = (ep[e] + get_ep_from_words(ref_ep[0], ref_ep[1], e)) & m;
            }
        }

        // Apply index reference (alpha 3-bit, BC3 colour 2-bit)
        uint ref_ix[6];
        load_ref_state(chunk_id, mode, i, xr, ref_ix);
        if (mode == 1u) {
            [unroll]
            for (uint w_i = 0u; w_i < 2u; ++w_i) {
                uint cur = ix_w[w_i];
                uint refw = ref_ix[2u + w_i];
                uint sum_w = 0u;
                [unroll]
                for (uint nib = 0u; nib < 8u; ++nib) {
                    uint sh = nib * 4u;
                    sum_w |= (((cur >> sh) + (refw >> sh)) & 7u) << sh;
                }
                ix_w[w_i] = sum_w;
            }
        } else if (mode == 2u) {
            [unroll]
            for (uint w_i = 0u; w_i < 4u; ++w_i) {
                uint cur = ix_w[w_i];
                uint refw = ref_ix[2u + w_i];
                uint sum_w = 0u;
                [unroll]
                for (uint nib = 0u; nib < 8u; ++nib) {
                    uint sh = nib * 4u;
                    sum_w |= (((cur >> sh) + (refw >> sh)) & 7u) << sh;
                }
                ix_w[w_i] = sum_w;
            }
        } else {
            // BC3: words 0..1 alpha (mod 8), words 2..3 colour (mod 4)
            [unroll]
            for (uint w_a = 0u; w_a < 2u; ++w_a) {
                uint cur = ix_w[w_a];
                uint refw = ref_ix[2u + w_a];
                uint sum_w = 0u;
                [unroll]
                for (uint nib = 0u; nib < 8u; ++nib) {
                    uint sh = nib * 4u;
                    sum_w |= (((cur >> sh) + (refw >> sh)) & 7u) << sh;
                }
                ix_w[w_a] = sum_w;
            }
            [unroll]
            for (uint w_c = 2u; w_c < 4u; ++w_c) {
                uint cur = ix_w[w_c];
                uint refw = ref_ix[2u + w_c];
                uint sum_w = 0u;
                [unroll]
                for (uint nib = 0u; nib < 8u; ++nib) {
                    uint sh = nib * 4u;
                    sum_w |= (((cur >> sh) + (refw >> sh)) & 3u) << sh;
                }
                ix_w[w_c] = sum_w;
            }
        }

        // Write packed block
        if (mode == 1u) {
            uint aw[2] = { ix_w[0], ix_w[1] };
            g_OutBlocks.Store2(desc.out_byte_offset + i * 8u, pack_alpha(ep[0], ep[1], aw));
        } else if (mode == 2u) {
            uint aw0[2] = { ix_w[0], ix_w[1] };
            uint aw1[2] = { ix_w[2], ix_w[3] };
            uint2 a0 = pack_alpha(ep[0], ep[1], aw0);
            uint2 a1 = pack_alpha(ep[2], ep[3], aw1);
            g_OutBlocks.Store4(desc.out_byte_offset + i * 16u, uint4(a0.x, a0.y, a1.x, a1.y));
        } else if (mode == 3u) {
            uint cw[2] = { ix_w[2], ix_w[3] };
            g_OutBlocks.Store2(desc.out_byte_offset + i * 8u, pack_colour(ep, cw));
        } else {
            uint aw[2] = { ix_w[0], ix_w[1] };
            uint cw[2] = { ix_w[2], ix_w[3] };
            uint2 a0 = pack_alpha(ep[0], ep[1], aw);
            uint2 c0 = pack_colour(ep, cw);
            g_OutBlocks.Store4(desc.out_byte_offset + i * 16u, uint4(a0.x, a0.y, c0.x, c0.y));
        }

        // Update 256-entry ring history
        uint hb = hist_chunk_base + (i & 255u) * 6u;
        g_Hist[hb + 0u] = ep[0] | (ep[1] << 8u) | (ep[2] << 16u) | (ep[3] << 24u);
        g_Hist[hb + 1u] = ep[4] | (ep[5] << 8u) | (ep[6] << 16u) | (ep[7] << 24u);
        g_Hist[hb + 2u] = ix_w[0];
        g_Hist[hb + 3u] = ix_w[1];
        g_Hist[hb + 4u] = ix_w[2];
        g_Hist[hb + 5u] = ix_w[3];
    }
}
