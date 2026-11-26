// SPDX-License-Identifier: MIT OR Apache-2.0
// Adapted from POMXARK/bcpack-gpu-sdk (4bca64a)

// Clean-room Xbox Series BCPack Full Multi-Mode BC7 (Modes 0..7) GPU Block Rebuilder (HLSL SM 6.6)
// Exact bit-for-bit parity with Ahmet-Dedeler/bcpack-rs (src/bc7.rs).
// Reads 16-lane symbol rows (8 uints/row, 2 rows/block) emitted by bcpack_bc7_rans.hlsl
// and writes complete 128-bit (16-byte) BC7 blocks using 256-block history & 255-entry dictionary.

cbuffer Bc7RebuildCB : register(b0) {
    uint  g_NumChunks;
    uint3 _pad;
};

struct Bc7ChunkDesc {
    uint in_row_offset;
    uint blocks;
    uint out_block_offset;
    uint _pad0;
};

StructuredBuffer<uint>         g_InSyms    : register(t0); // 8 uints (16 uint16) per row
StructuredBuffer<Bc7ChunkDesc> g_Chunks    : register(t1);
StructuredBuffer<uint>         g_Controls  : register(t2); // 512 entries: bit31=valid | mode | (part<<8) | (rot<<16) | (sel<<24)
StructuredBuffer<uint>         g_Dict      : register(t3); // 255 * 11 uints

RWStructuredBuffer<uint4>      g_OutBlocks : register(u0); // 16 bytes per BC7 block
RWStructuredBuffer<uint>       g_Hist      : register(u1); // max_chunks * 256 * 11 uints

static const uint SUBSETS[8]        = { 3u, 2u, 3u, 2u, 1u, 1u, 1u, 2u };
static const uint PARTITION_BITS[8] = { 4u, 6u, 6u, 6u, 0u, 0u, 0u, 6u };
static const uint CHANNEL_BITS[8][4] = {
    { 4u, 4u, 4u, 0u },
    { 6u, 6u, 6u, 0u },
    { 5u, 5u, 5u, 0u },
    { 7u, 7u, 7u, 0u },
    { 5u, 5u, 5u, 6u },
    { 7u, 7u, 7u, 8u },
    { 7u, 7u, 7u, 7u },
    { 5u, 5u, 5u, 5u }
};
static const uint INDEX_BITS[8][2] = {
    { 3u, 0u },
    { 3u, 0u },
    { 2u, 0u },
    { 2u, 0u },
    { 3u, 2u },
    { 2u, 2u },
    { 4u, 0u },
    { 2u, 0u }
};

static const uint ANCHOR_2[64] = {
    15u,15u,15u,15u,15u,15u,15u,15u,15u,15u,15u,15u,15u,15u,15u,15u,
    15u, 2u, 8u, 2u, 2u, 8u, 8u,15u, 2u, 8u, 2u, 2u, 8u, 8u, 2u, 2u,
    15u,15u, 6u, 8u, 2u, 8u,15u,15u, 2u, 8u, 2u, 2u, 2u,15u,15u, 6u,
     6u, 2u, 6u, 8u,15u,15u, 2u, 2u,15u,15u,15u,15u,15u, 2u, 2u,15u
};
static const uint ANCHOR_3A[64] = {
     3u, 3u,15u,15u, 8u, 3u,15u,15u, 8u, 8u, 6u, 6u, 6u, 5u, 3u, 3u,
     3u, 3u, 8u,15u, 3u, 3u, 6u,10u, 5u, 8u, 8u, 6u, 8u, 5u,15u,15u,
     8u,15u, 3u, 5u, 6u,10u, 8u,15u,15u, 3u,15u, 5u,15u,15u,15u,15u,
     3u,15u, 5u, 5u, 5u, 8u, 5u,10u, 5u,10u, 8u,13u,15u,12u, 3u, 3u
};
static const uint ANCHOR_3B[64] = {
    15u, 8u, 8u, 3u,15u,15u, 3u, 8u,15u,15u,15u,15u,15u,15u,15u, 8u,
    15u, 8u,15u, 3u,15u, 8u,15u, 8u, 3u,15u, 6u,10u,15u,15u,10u, 8u,
    15u, 3u,15u,10u,10u, 8u, 9u,10u, 6u,15u, 8u,15u, 3u, 6u, 6u, 8u,
    15u, 3u,15u,15u,15u,15u,15u,15u,15u,15u,15u,15u, 3u,15u,15u, 8u
};

static const uint INDEX_LANES[2][4] = {
    { 11u, 12u, 13u, 14u },
    {  6u,  7u,  8u, 10u }
};

uint read_sym(uint row, uint lane) {
    uint w = g_InSyms[row * 8u + (lane >> 1u)];
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

int endpoint_lane(uint subsets, uint row, uint ch) {
    if (subsets == 3u && row < 2u && ch == 3u) return -1;
    if (row == 0u) return int(2u + ch);
    if (row == 1u) return int(6u + ch);
    if (row == 2u) {
        if (ch == 0u) return 10;
        if (ch == 1u && subsets == 3u) return 5;
        if (ch == 2u && subsets == 3u) return 9;
    }
    return -1;
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

void load_state_words(uint chunk_id, uint i, uint r, out uint sw[11]) {
    if (r == 0u) {
        [unroll]
        for (uint k = 0u; k < 11u; ++k) sw[k] = 0u;
    } else if (r <= i) {
        uint base = (chunk_id * 256u + ((i - r) & 255u)) * 11u;
        [unroll]
        for (uint k = 0u; k < 11u; ++k) sw[k] = g_Hist[base + k];
    } else {
        uint base = (r - i - 1u) * 11u;
        [unroll]
        for (uint k = 0u; k < 11u; ++k) sw[k] = g_Dict[base + k];
    }
}

uint read_ep(uint sw[11], uint row, uint col) {
    uint idx = row * 8u + col;
    return (sw[1u + (idx >> 2u)] >> ((idx & 3u) * 8u)) & 0xFFu;
}

uint read_ix(uint sw[11], uint set_idx, uint p) {
    uint idx = set_idx * 16u + p;
    return (sw[7u + (idx >> 3u)] >> ((idx & 7u) * 4u)) & 0xFu;
}

[numthreads(64, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    uint chunk_id = dtid.x;
    if (chunk_id >= g_NumChunks) {
        return;
    }

    Bc7ChunkDesc desc = g_Chunks[chunk_id];
    uint hist_chunk_base = chunk_id * 256u * 11u;

    for (uint i = 0u; i < desc.blocks; ++i) {
        uint r0 = desc.in_row_offset + 2u * i;
        uint r1 = r0 + 1u;

        uint er = 0u;
        uint xr = 0u;
        decode_refs(
            interleave4(read_sym(r0, 0u), read_sym(r1, 0u)),
            interleave4(read_sym(r0, 1u), read_sym(r1, 1u)),
            er, xr);

        uint code = ((read_sym(r1, 15u) & 15u) << 5u) | (read_sym(r0, 15u) & 31u);
        uint ctl = g_Controls[code];
        if ((ctl & 0x80000000u) == 0u) {
            g_OutBlocks[desc.out_block_offset + i] = uint4(0u, 0u, 0u, 0u);
            uint hb0 = hist_chunk_base + (i & 255u) * 11u;
            [unroll]
            for (uint k = 0u; k < 11u; ++k) g_Hist[hb0 + k] = 0u;
            continue;
        }

        uint mode      = ctl & 7u;
        uint partition = (ctl >> 8u) & 63u;
        uint rotation  = (ctl >> 16u) & 3u;
        uint idx_sel   = (ctl >> 24u) & 1u;

        uint subsets = SUBSETS[mode];
        bool has_pbits = (mode == 0u || mode == 1u || mode == 3u || mode == 6u || mode == 7u);

        uint prec[4] = {
            CHANNEL_BITS[mode][0],
            CHANNEL_BITS[mode][1],
            CHANNEL_BITS[mode][2],
            CHANNEL_BITS[mode][3]
        };
        if (rotation != 0u) {
            uint tmp = prec[3];
            prec[3] = prec[rotation - 1u];
            prec[rotation - 1u] = tmp;
        } else if (has_pbits) {
            prec[0] += 1u;
        }

        uint ref_ep[11];
        load_state_words(chunk_id, i, er, ref_ep);
        uint ref_ix[11];
        load_state_words(chunk_id, i, xr, ref_ix);

        uint st_w[11];
        [unroll]
        for (uint k = 0u; k < 11u; ++k) st_w[k] = 0u;
        st_w[0] = subsets;

        uint common = min(subsets, ref_ep[0]);

        // Reconstruct endpoint state
        for (uint ch = 0u; ch < 4u; ++ch) {
            uint p = prec[ch];
            if (p == 0u) {
                continue;
            }
            uint sh = 8u - p;
            uint mask = (1u << p) - 1u;
            for (uint row = 0u; row < subsets; ++row) {
                uint from_row = (row < common) ? row : 0u;
                int ln = endpoint_lane(subsets, row, ch);
                [unroll]
                for (uint e = 0u; e < 2u; ++e) {
                    uint col = 2u * ch + e;
                    uint sym_v = (ln >= 0) ? read_sym((e == 0u) ? r0 : r1, uint(ln)) : 0u;
                    uint res = (ln >= 0) ? residual_bits(sym_v, p) : 0u;
                    uint base_v = read_ep(ref_ep, from_row, col) >> sh;
                    uint new_ep = ((base_v + res) & mask) << sh;
                    uint ep_idx = row * 8u + col;
                    st_w[1u + (ep_idx >> 2u)] |= (new_ep & 0xFFu) << ((ep_idx & 3u) * 8u);
                }
            }
        }

        // Reconstruct index state
        uint ibits[2] = { INDEX_BITS[mode][0], INDEX_BITS[mode][1] };
        for (uint set_i = 0u; set_i < 2u; ++set_i) {
            uint b = ibits[set_i];
            if (b == 0u) {
                continue;
            }
            uint sh = 4u - b;
            uint mask = (1u << b) - 1u;
            for (uint row = 0u; row < 4u; ++row) {
                uint ln = INDEX_LANES[set_i][row];
                [unroll]
                for (uint half = 0u; half < 2u; ++half) {
                    uint ev = 0u;
                    uint od = 0u;
                    deinterleave4(read_sym((half == 0u) ? r0 : r1, ln), ev, od);
                    uint left_c  = (row < 3u) ? od : ev;
                    uint right_c = (row < 3u) ? ev : od;
                    uint codes[2] = { left_c, right_c };
                    [unroll]
                    for (uint h = 0u; h < 2u; ++h) {
                        uint px = 4u * row + 2u * half + h;
                        uint res = residual_bits(codes[h], b);
                        uint base_x = read_ix(ref_ix, set_i, px) >> sh;
                        uint new_ix = ((base_x + res) & mask) << sh;
                        uint ix_idx = set_i * 16u + px;
                        st_w[7u + (ix_idx >> 3u)] |= (new_ix & 0xFu) << ((ix_idx & 7u) * 4u);
                    }
                }
            }
        }

        // Pack 128-bit BC7 block
        uint4 out_blk = uint4(0u, 0u, 0u, 0u);
        uint at = 0u;
        put_bits(out_blk, at, 1u << mode, mode + 1u);
        put_bits(out_blk, at, partition, PARTITION_BITS[mode]);
        if (mode == 4u || mode == 5u) {
            put_bits(out_blk, at, rotation, 2u);
        }
        if (mode == 4u) {
            put_bits(out_blk, at, idx_sel, 1u);
        }

        uint ch_order[4] = { 0u, 1u, 2u, 3u };
        if (rotation != 0u) {
            uint tmp = ch_order[3];
            ch_order[3] = ch_order[rotation - 1u];
            ch_order[rotation - 1u] = tmp;
        }

        for (uint ci = 0u; ci < 4u; ++ci) {
            uint ch = ch_order[ci];
            uint p = prec[ch];
            if (p == 0u) {
                continue;
            }
            uint drop = (ch == 0u && has_pbits) ? 1u : 0u;
            uint sh = 8u - p;
            for (uint e = 0u; e < 2u * subsets; ++e) {
                uint fld = read_ep(st_w, e >> 1u, 2u * ch + (e & 1u)) >> sh;
                put_bits(out_blk, at, fld >> drop, p - drop);
            }
        }

        if (mode == 1u) {
            uint sh0 = 8u - prec[0];
            for (uint row = 0u; row < subsets; ++row) {
                uint pbit = (read_ep(st_w, row, 0u) >> sh0) & 1u;
                put_bits(out_blk, at, pbit, 1u);
            }
        } else if (has_pbits) {
            uint sh0 = 8u - prec[0];
            for (uint e = 0u; e < 2u * subsets; ++e) {
                uint pbit = (read_ep(st_w, e >> 1u, e & 1u) >> sh0) & 1u;
                put_bits(out_blk, at, pbit, 1u);
            }
        }

        uint anchors = 1u;
        uint part_m = partition & 63u;
        if (subsets == 2u) {
            anchors |= (1u << ANCHOR_2[part_m]);
        } else if (subsets == 3u) {
            anchors |= (1u << ANCHOR_3A[part_m]) | (1u << ANCHOR_3B[part_m]);
        }

        if (mode == 4u) {
            uint b1 = ibits[1];
            uint s1 = 4u - b1;
            for (uint p = 0u; p < 16u; ++p) {
                uint n = b1 - ((anchors >> p) & 1u);
                put_bits(out_blk, at, read_ix(st_w, 1u, p) >> s1, n);
            }
            uint b0 = ibits[0];
            uint s0 = 4u - b0;
            for (uint p = 0u; p < 16u; ++p) {
                uint n = b0 - ((anchors >> p) & 1u);
                put_bits(out_blk, at, read_ix(st_w, 0u, p) >> s0, n);
            }
        } else if (mode == 5u) {
            uint b0 = ibits[0];
            uint s0 = 4u - b0;
            for (uint p = 0u; p < 16u; ++p) {
                uint n = b0 - ((anchors >> p) & 1u);
                put_bits(out_blk, at, read_ix(st_w, 0u, p) >> s0, n);
            }
            uint b1 = ibits[1];
            uint s1 = 4u - b1;
            for (uint p = 0u; p < 16u; ++p) {
                uint n = b1 - ((anchors >> p) & 1u);
                put_bits(out_blk, at, read_ix(st_w, 1u, p) >> s1, n);
            }
        } else {
            uint b0 = ibits[0];
            uint s0 = 4u - b0;
            for (uint p = 0u; p < 16u; ++p) {
                uint n = b0 - ((anchors >> p) & 1u);
                put_bits(out_blk, at, read_ix(st_w, 0u, p) >> s0, n);
            }
        }

        g_OutBlocks[desc.out_block_offset + i] = out_blk;

        uint hb = hist_chunk_base + (i & 255u) * 11u;
        [unroll]
        for (uint k = 0u; k < 11u; ++k) {
            g_Hist[hb + k] = st_w[k];
        }
    }
}
