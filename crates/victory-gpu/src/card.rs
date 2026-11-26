//! Large title buffers resident in device-local VRAM (`D3D12_HEAP_TYPE_DEFAULT`)
//! with GPU-side Wave64 64 KiB page hashing (`CARD_COMPARE`) for copy-back.
//!
//! Instead of keeping large buffers (4–75+ MiB) in system RAM (`L0`) over PCIe
//! and taking multi-megabyte CPU snapshots + CPU diffs every submission:
//! - Shaders read and write `Shadow::buffer` directly in GDDR6 VRAM.
//! - `CardState::staging` (`L0` WRITE_BACK, mapped as `Shadow::data`) receives
//!   CPU title-memory updates and simultaneously serves as the exact
//!   pre-submission snapshot (zero snapshot `memcpy` in `Shadow::writing`!).
//! - Before a submission (`card_before`), only CPU-modified ranges are uploaded
//!   via `CopyBufferRegion` and re-hashed on the GPU (if the buffer has hashes).
//! - At the end of a submission (`card_after`), `CARD_COMPARE` dispatches one
//!   64-thread workgroup per declared 64 KiB page in VRAM (64 coalesced
//!   1024-byte bursts per page, 64x less sequential latency than 1 thread/page),
//!   writing a 4-byte dirty flag per page to a small readback buffer.
//! - After the fence (`card_back`), ONLY pages flagged dirty are copied from
//!   VRAM to a readback buffer and merged into title memory + `staging`.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use crate::host::Host;
use crate::shadow::merge_byte_ranges;
use crate::watch::PAGE;

/// Wave64 / 64-thread-per-page 64 KiB hashing compute shader (`cs_5_1`).
/// One workgroup (`SV_GroupID.x`) hashes one 64 KiB buffer-relative piece using
/// 64 threads reading coalesced 1024-byte strides (64 iterations per page).
const CARD_COMPARE_HLSL: &str = r#"
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), UAV(u0), UAV(u1)"
cbuffer Span : register(b0) {
    uint first;    // first 64 KiB piece index
    uint pad0;     // reserved (0)
    uint size;     // buffer size in bytes (multiple of 65536)
    uint pieces;   // number of pieces in this dispatch
    uint out_at;   // byte offset in `differs` for gid.x == 0
    uint compare;  // 0: store hashes; 1: compare only; 2: compare and store new hash
};
ByteAddressBuffer now : register(t0);
RWByteAddressBuffer hashes : register(u0);
RWByteAddressBuffer differs : register(u1);

groupshared uint4 s_acc[64];

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint tid : SV_GroupIndex) {
    if (gid.x >= pieces) return;
    uint piece = first + gid.x;
    uint lo = piece * 65536u;
    uint hi = min(lo + 65536u, size);

    uint4 h = uint4(
        0x9E3779B9u ^ (tid * 0x85EBCA6Bu),
        0x85EBCA6Bu ^ (tid * 0xC2B2AE35u),
        0xC2B2AE35u ^ (tid * 0x27D4EB2Fu),
        0x27D4EB2Fu ^ (tid * 0x165667B1u)
    );
    [loop]
    for (uint at = lo + tid * 16u; at < hi; at += 1024u) {
        h = (h ^ now.Load4(at)) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
    }
    s_acc[tid] = h;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint stride = 32u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            uint4 a = s_acc[tid];
            uint4 b = s_acc[tid + stride];
            uint4 m = (a ^ (b * uint4(0x27D4EB2Fu, 0x165667B1u, 0x9E3779B9u, 0x85EBCA6Bu)))
                    * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
            s_acc[tid] = m.yzwx ^ (m >> 15);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0u) {
        uint4 final_h = s_acc[0];
        if (compare != 0u) {
            uint4 prev = hashes.Load4(piece * 16u);
            differs.Store(out_at + gid.x * 4u, any(final_h != prev) ? 1u : 0u);
            if (compare == 2u) {
                hashes.Store4(piece * 16u, final_h);
            }
        } else {
            hashes.Store4(piece * 16u, final_h);
        }
    }
}
"#;

/// Bytes currently allocated in VRAM for card buffers (`[0]`) and their hash
/// tables (`[1]`).
static CARD_BYTES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Readback wait attempts, wait time (us), bytes requested, and copy commands.
#[cfg(feature = "diagnostics")]
static FETCH_STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

#[cfg(feature = "diagnostics")]
pub(crate) fn take_fetch_stats() -> [u64; 4] {
    std::array::from_fn(|i| FETCH_STATS[i].swap(0, Ordering::Relaxed))
}

/// Reserve device-local bytes without transiently exceeding the shared cap.
fn reserve_card_bytes(used: &AtomicU64, size: u64, limit: u64) -> bool {
    used.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
        current.checked_add(size).filter(|&next| next <= limit)
    })
    .is_ok()
}

fn card_ceiling(budget: u64) -> u64 {
    (u128::from(budget) * 75 / 100) as u64
}

/// With `VICTORY_RESERVE_CARD_VRAM=1`, optional GPU upload heaps leave room
/// for card buffers. Ordinary upload heaps preserve the same data in RAM.
pub(crate) fn gpu_upload_fits(used: u64, budget: u64, non_local_budget: u64, size: u64) -> bool {
    if budget == 0 {
        return true;
    }
    static RESERVE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let reserve = non_local_budget != 0
        && enabled()
        && *RESERVE.get_or_init(|| {
            std::env::var_os("VICTORY_RESERVE_CARD_VRAM").is_some_and(|v| v == "1")
        });
    let (card_limit, active_card_bytes) = if reserve {
        (limits().1, CARD_BYTES[0].load(Ordering::Relaxed))
    } else {
        (0, 0)
    };
    committed_upload_fits(used, gpu_upload_limit(budget, card_limit, active_card_bytes), size)
}

/// A committed D3D12 buffer consumes whole 64 KiB allocation units.
fn committed_upload_fits(used: u64, limit: u64, size: u64) -> bool {
    size.max(1)
        .checked_next_multiple_of(D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT as u64)
        .and_then(|bytes| used.checked_add(bytes))
        .is_some_and(|total| total < limit)
}

fn gpu_upload_limit(budget: u64, card_limit: u64, active_card_bytes: u64) -> u64 {
    let baseline = (u128::from(budget) * 85 / 100) as u64;
    let remaining = card_limit.min(budget / 6).saturating_sub(active_card_bytes);
    if remaining == 0 {
        baseline
    } else {
        card_ceiling(budget).saturating_sub(remaining)
    }
}

/// A host's card buffers with pending CPU-to-VRAM uploads.
pub(crate) type DirtyQueue = Mutex<Vec<Weak<CardState>>>;

/// Whether VRAM card buffers are enabled (`VICTORY_CARD_BUFFERS=0` disables).
pub(crate) fn enabled() -> bool {
    if cfg!(test) {
        return false;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_BUFFERS").is_none_or(|v| v != "0"))
}

/// `(min_bytes, max_total_bytes)` for card buffers:
/// `VICTORY_CARD_BUFFERS_MIN_MB` (default 4 MiB) and
/// `VICTORY_CARD_BUFFERS_MB` (default 2048 MiB).
fn limits() -> (u64, u64) {
    static LIMITS: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    *LIMITS.get_or_init(|| {
        let mb = |name: &str, default: u64| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.trim().parse::<u64>().ok())
                .unwrap_or(default)
                .saturating_mul(1 << 20)
        };
        (
            mb("VICTORY_CARD_BUFFERS_MIN_MB", 4).max(1 << 20),
            mb("VICTORY_CARD_BUFFERS_MB", 2048),
        )
    })
}

pub(crate) fn hash_table_bytes(size: u64) -> u64 {
    ((size / PAGE as u64 + 1) * 16).next_multiple_of(256)
}

/// Per-`Shadow` state when a buffer lives in device-local VRAM (`DEFAULT`).
pub(crate) struct CardState {
    /// CPU-mapped staging mirror (`L0` WRITE_BACK; `Shadow::data` points here).
    pub(crate) staging: ID3D12Resource,
    /// Device-local VRAM buffer (`DEFAULT`, `ALLOW_UNORDERED_ACCESS`; same COM
    /// resource as `Shadow::buffer`).
    pub(crate) vram: ID3D12Resource,
    /// Buffer width in bytes (multiple of `PAGE`).
    pub(crate) size: u64,
    /// Lazily created `DEFAULT` UAV buffer holding 16-byte hashes per 64 KiB piece.
    hashes: Mutex<Option<ID3D12Resource>>,
    /// True once `hashes` exists and has been populated for the whole buffer.
    hashes_ready: AtomicBool,
    /// Buffer-relative byte ranges `[from, to)` written to `staging` by the CPU
    /// that must be copied to `vram` (and re-hashed if `hashes_ready`) before
    /// the next GPU submission.
    dirty_up: Mutex<Vec<(usize, usize)>>,
    dirty_queue: Arc<DirtyQueue>,
    /// Deduplication flag for `dirty_queue`.
    queued: AtomicBool,
}

// SAFETY: D3D12 resources are free-threaded; mutable fields are synchronized.
unsafe impl Send for CardState {}
// SAFETY: as above.
unsafe impl Sync for CardState {}

impl Drop for CardState {
    fn drop(&mut self) {
        CARD_BYTES[0].fetch_sub(self.size, Ordering::Relaxed);
        let hashes = self.hashes.get_mut().unwrap_or_else(|e| e.into_inner()).take();
        let mut held = self.size;
        if hashes.is_some() {
            CARD_BYTES[1].fetch_sub(hash_table_bytes(self.size), Ordering::Relaxed);
            held += hash_table_bytes(self.size);
        }
        // The video memory stays with the retire queue until the GPU is done.
        crate::host::retire_bytes((self.staging.clone(), self.vram.clone(), hashes), held);
    }
}

impl CardState {
    /// Allocates a VRAM card buffer for `size` bytes (must be a multiple of
    /// `PAGE`) if enabled and within VRAM budget.
    pub(crate) fn try_new(host: &Host, staging: &ID3D12Resource, size: u64) -> Option<Arc<Self>> {
        if !enabled() {
            return None;
        }
        let (min_bytes, max_bytes) = limits();
        if size < min_bytes || size >= (1 << 30) || !size.is_multiple_of(PAGE as u64) {
            return None;
        }
        if !host.card_ready() {
            return None;
        }
        if let Some([(used, budget), _]) = crate::host::video_memory()
            && budget > 0
            && used.saturating_add(size) > card_ceiling(budget)
        {
            crate::memstat::add(&crate::memstat::CARD_REFUSED[0], size);
            crate::memstat::add(&crate::memstat::CARD_REFUSED[1], 1);
            return None;
        }
        if !reserve_card_bytes(&CARD_BYTES[0], size, max_bytes) {
            crate::memstat::add(&crate::memstat::CARD_REFUSED[2], size);
            crate::memstat::add(&crate::memstat::CARD_REFUSED[3], 1);
            return None;
        }
        Self::new_unchecked(host, staging, size, true)
    }

    /// Creates a `CardState` directly (used by `try_new` and WARP unit tests).
    pub(crate) fn new_unchecked(
        host: &Host,
        staging: &ID3D12Resource,
        size: u64,
        counted: bool,
    ) -> Option<Arc<Self>> {
        let Some(vram) = host.uav_buffer(size) else {
            if counted {
                CARD_BYTES[0].fetch_sub(size, Ordering::Relaxed);
            }
            return None;
        };
        if !counted {
            CARD_BYTES[0].fetch_add(size, Ordering::Relaxed);
        }
        Some(Arc::new(Self {
            staging: staging.clone(),
            vram,
            size,
            hashes: Mutex::new(None),
            hashes_ready: AtomicBool::new(false),
            dirty_up: Mutex::new(Vec::new()),
            dirty_queue: host.card_dirty.clone(),
            queued: AtomicBool::new(false),
        }))
    }

    /// Notes that CPU copied new bytes into `staging[from..to]`.
    pub(crate) fn mark_dirty(self: &Arc<Self>, from: usize, to: usize) {
        if from >= to {
            return;
        }
        {
            let mut dirty = self.dirty_up.lock().unwrap_or_else(|e| e.into_inner());
            dirty.push((from, to));
        }
        if !self.queued.swap(true, Ordering::AcqRel) {
            self.dirty_queue
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(Arc::downgrade(self));
        }
    }

    /// Ensures the GPU hash table exists; returns `(hashes_resource, needs_full_init)`.
    fn ensure_hashes(&self, host: &Host) -> Option<(ID3D12Resource, bool)> {
        let mut h = self.hashes.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(res) = h.as_ref() {
            return Some((res.clone(), !self.hashes_ready.load(Ordering::Acquire)));
        }
        let bytes = hash_table_bytes(self.size);
        let res = host.uav_buffer(bytes)?;
        crate::host::name(&res, || format!("card hashes {bytes:#x}"));
        CARD_BYTES[1].fetch_add(bytes, Ordering::Relaxed);
        *h = Some(res.clone());
        Some((res, true))
    }

    fn take_dirty_up(&self) -> Vec<(usize, usize)> {
        self.queued.store(false, Ordering::Release);
        let mut ranges = {
            let mut dirty = self.dirty_up.lock().unwrap_or_else(|e| e.into_inner());
            std::mem::take(&mut *dirty)
        };
        // The detached ranges no longer need the per-card mutex while sorting.
        merge_byte_ranges(&mut ranges);
        ranges
    }

    pub(crate) fn total_pieces(&self) -> u32 {
        (self.size / PAGE as u64) as u32
    }
}

/// Groups consecutive runs by entry without allocating a vector per shadow.
/// Entries arrive in plan order; the returned spans index the flat pieces.
type ReadbackGroup = (usize, std::ops::Range<usize>);
type ReadbackPiece<'a> = (u32, &'a [u8]);
type DirtyCard = (Arc<CardState>, Vec<(usize, usize)>);
type RehashUpload = (Arc<CardState>, ID3D12Resource, Vec<(u32, u32)>);

fn group_readback_runs<'a>(
    runs: impl IntoIterator<Item = (usize, u32, &'a [u8])>,
) -> (Vec<ReadbackGroup>, Vec<ReadbackPiece<'a>>) {
    let mut groups: Vec<ReadbackGroup> = Vec::new();
    let runs = runs.into_iter();
    let mut pieces = Vec::with_capacity(runs.size_hint().0);
    for (entry, first, bytes) in runs {
        let at = pieces.len();
        pieces.push((first, bytes));
        if let Some((_, span)) = groups.last_mut().filter(|(i, _)| *i == entry) {
            span.end = at + 1;
        } else {
            groups.push((entry, at..at + 1));
        }
    }
    (groups, pieces)
}

/// Optional extra pages to hash between upload rehash runs. Never apply this
/// to the post-submission compare: that must retain exact declared coverage.
fn rehash_gap() -> u32 {
    static GAP: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *GAP.get_or_init(|| {
        std::env::var("VICTORY_CARD_REHASH_GAP")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .unwrap_or(0)
            .min(16)
    })
}

/// Input runs are sorted, disjoint and bounded by the card size. Bridging
/// small gaps trades extra hash work for fewer dispatches. Rehash only.
fn join_rehash_runs(runs: &mut Vec<(u32, u32)>, gap: u32) {
    if gap == 0 || runs.len() < 2 {
        return;
    }
    let mut out = 0;
    for i in 1..runs.len() {
        let (first, count) = runs[i];
        let (base, span) = runs[out];
        let end = base + span;
        if first.saturating_sub(end) <= gap {
            runs[out].1 = end.max(first + count) - base;
        } else {
            out += 1;
            runs[out] = (first, count);
        }
    }
    runs.truncate(out + 1);
}

/// Converts byte ranges `[from, to)` into sorted, merged 64 KiB piece runs `(first_piece, count)`.
fn byte_ranges_to_piece_runs(ranges: &[(usize, usize)], max_pieces: u32) -> Vec<(u32, u32)> {
    if ranges.is_empty() || max_pieces == 0 {
        return Vec::new();
    }
    let mut spans = Vec::with_capacity(ranges.len());
    for &(from, to) in ranges {
        if from >= to {
            continue;
        }
        // Clip before narrowing, so large offsets cannot wrap into valid pages.
        let first = (from / PAGE).min(max_pieces as usize) as u32;
        let end = to.div_ceil(PAGE).min(max_pieces as usize) as u32;
        if first < end {
            spans.push((first, end));
        }
    }
    spans.sort_unstable();
    // Compact in place instead of allocating a second vector of runs.
    let mut used = 0;
    for i in 0..spans.len() {
        let (first, end) = spans[i];
        if used > 0 && first <= spans[used - 1].1 {
            spans[used - 1].1 = spans[used - 1].1.max(end);
        } else {
            spans[used] = (first, end);
            used += 1;
        }
    }
    spans.truncate(used);
    for (first, end) in &mut spans {
        *end -= *first;
    }
    spans
}

/// Per-submission plan built in `card_before` and consumed in `card_after` + `card_back`.
pub(crate) struct CardPlan {
    entries: Vec<CardWriteEntry>,
    total_pieces: u32,
    uploads: Vec<DirtyCard>,
    prefetched: Option<(ID3D12Resource, usize)>,
}

#[derive(Debug, PartialEq, Eq)]
struct DirtyRun {
    entry_idx: usize,
    first: u32,
    count: u32,
    offset: usize,
}

/// Select dirty pages and coalesce adjacent copies. The optimistic readback
/// uses the same packed offsets as the flags, including holes for clean pages.
fn dirty_readback_runs(
    entries: impl IntoIterator<Item = (u32, impl IntoIterator<Item = (u32, u32)>)>,
    flags: &[u32],
    prefetched: bool,
) -> (Vec<DirtyRun>, usize) {
    let mut dirty: Vec<DirtyRun> = Vec::new();
    let mut bytes = 0;
    for (entry_idx, (out_start, runs)) in entries.into_iter().enumerate() {
        let mut index = out_start as usize;
        for (first, count) in runs {
            for piece in first..first + count {
                let offset = if prefetched { index * PAGE } else { bytes };
                let changed = flags[index] != 0;
                index += 1;
                if !changed {
                    continue;
                }
                bytes += PAGE;
                if let Some(last) = dirty.last_mut()
                    && last.entry_idx == entry_idx
                    && last.first + last.count == piece
                    && last.offset + last.count as usize * PAGE == offset
                {
                    last.count += 1;
                } else {
                    dirty.push(DirtyRun { entry_idx, first: piece, count: 1, offset });
                }
            }
        }
    }
    (dirty, bytes)
}

fn prefetch_bytes(pieces: u32, limit: u64) -> Option<u64> {
    let bytes = u64::from(pieces) * PAGE as u64;
    (bytes != 0 && bytes <= limit).then_some(bytes)
}

struct CardWriteEntry {
    shadow: Arc<crate::shadow::Shadow>,
    card: Arc<CardState>,
    hashes: ID3D12Resource,
    /// Original declared title ranges `(address, len)` for clearing `pending`.
    declared: Vec<(usize, usize)>,
    /// Merged 64 KiB buffer-relative piece runs `(first_piece, count)`.
    runs: Vec<(u32, u32)>,
    /// First dword index in `differs` for this entry.
    out_start: u32,
}

impl CardPlan {
    pub(crate) fn has_writes(&self) -> bool {
        self.total_pieces > 0
    }

    pub(crate) fn clear_pending(self) {
        for entry in &self.entries {
            entry.shadow.finish_card_pending(&entry.declared);
        }
    }
}

/// Per-`Host` reusable GPU pipelines, command lists, and readback scratch buffers
/// for VRAM card buffer uploads, Wave64 hashing, and dirty-page fetch.
pub(crate) struct CardEngine {
    root_sig: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    before_alloc: ID3D12CommandAllocator,
    before_list: ID3D12GraphicsCommandList,
    after_alloc: ID3D12CommandAllocator,
    after_list: ID3D12GraphicsCommandList,
    fetch_alloc: ID3D12CommandAllocator,
    fetch_list: ID3D12GraphicsCommandList,
    differs_gpu: ID3D12Resource,
    differs_rb: (ID3D12Resource, usize, u64),
    direct_flags: bool,
    prefetch_limit: u64,
    dummy_uav: ID3D12Resource,
    fetch_rb: Option<(ID3D12Resource, usize, u64)>,
    dirty_queue: Vec<Weak<CardState>>,
}

// SAFETY: accessed only under `Host::submitted` lock or `CardEngine` mutex;
// D3D12 interfaces are free-threaded.
unsafe impl Send for CardEngine {}

impl Drop for CardEngine {
    fn drop(&mut self) {
        crate::cardmeter::release(&self.before_list);
        crate::cardmeter::release(&self.after_list);
    }
}

impl CardEngine {
    pub(crate) fn new(host: &Host) -> windows::core::Result<Self> {
        let direct = std::env::var_os("VICTORY_CARD_DIRECT_FLAGS").is_some_and(|v| v == "1");
        let prefetch = !cfg!(test)
            && std::env::var_os("VICTORY_CARD_PREFETCH_READBACK").is_some_and(|v| v == "1");
        Self::new_with_options(host, direct, prefetch)
    }

    fn new_with_options(host: &Host, direct: bool, prefetch: bool) -> windows::core::Result<Self> {
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: valid HLSL source and entry point.
        unsafe {
            D3DCompile(
                CARD_COMPARE_HLSL.as_ptr().cast(),
                CARD_COMPARE_HLSL.len(),
                windows::core::s!("card_compare_wave64"),
                None,
                None,
                windows::core::s!("main"),
                windows::core::s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )?;
        }
        let code = code.ok_or_else(windows::core::Error::empty)?;
        // SAFETY: valid compiled DXBC blob from D3DCompile.
        let bytecode = unsafe {
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
        };
        // SAFETY: bytecode embeds `[RootSignature(RS)]`.
        let root_sig: ID3D12RootSignature =
            unsafe { host.device.CreateRootSignature(0, bytecode)? };
        let pso_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::ManuallyDrop::new(Some(root_sig.clone())),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytecode.as_ptr().cast(),
                BytecodeLength: bytecode.len(),
            },
            ..Default::default()
        };
        // SAFETY: valid compute pipeline description.
        let pso_res: windows::core::Result<ID3D12PipelineState> =
            unsafe { host.device.CreateComputePipelineState(&pso_desc) };
        // Release our extra ref from `ManuallyDrop` above even on error.
        drop(std::mem::ManuallyDrop::into_inner(pso_desc.pRootSignature));
        let pso = pso_res?;

        let make_pair = || -> windows::core::Result<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)> {
            // SAFETY: plain direct command allocator and list creation.
            unsafe {
                let alloc: ID3D12CommandAllocator = host
                    .device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                let list: ID3D12GraphicsCommandList = host.device.CreateCommandList(
                    0,
                    D3D12_COMMAND_LIST_TYPE_DIRECT,
                    &alloc,
                    None,
                )?;
                list.Close()?;
                Ok((alloc, list))
            }
        };
        let (before_alloc, before_list) = make_pair()?;
        let (after_alloc, after_list) = make_pair()?;
        let (fetch_alloc, fetch_list) = make_pair()?;

        let init_diff_bytes = 256 << 10; // 65,536 pieces (4 GiB of ranges)
        let (differs_gpu, rb_res, rb_ptr, direct_flags) =
            Self::allocate_flags(host, init_diff_bytes, direct)
                .ok_or_else(windows::core::Error::empty)?;
        let dummy_uav = host
            .uav_buffer(256)
            .ok_or_else(windows::core::Error::empty)?;
        tracing::debug!(direct_flags, "card compare flag buffer");
        crate::host::name(&dummy_uav, || "card dummy uav".into());

        Ok(Self {
            root_sig,
            pso,
            before_alloc,
            before_list,
            after_alloc,
            after_list,
            fetch_alloc,
            fetch_list,
            differs_gpu,
            differs_rb: (rb_res, rb_ptr, init_diff_bytes),
            direct_flags,
            prefetch_limit: if prefetch { 1 << 20 } else { 0 },
            dummy_uav,
            fetch_rb: None,
            dirty_queue: Vec::new(),
        })
    }

    /// Prefer a CPU-mapped L0 UAV for the small dirty-flag array when opted in.
    /// If it is unavailable, retain the default device-local UAV + readback.
    fn allocate_flags(
        host: &Host,
        size: u64,
        direct: bool,
    ) -> Option<(ID3D12Resource, ID3D12Resource, usize, bool)> {
        if direct && let Some((resource, ptr)) = host.shared_buffer(size) {
            crate::host::name(&resource, || "card direct L0 flags".into());
            return Some((resource.clone(), resource, ptr as usize, true));
        }
        let gpu = host.uav_buffer(size)?;
        let (rb, ptr) = host.readback_buffer(size)?;
        crate::host::name(&gpu, || "card differs gpu".into());
        crate::host::name(&rb, || "card differs readback".into());
        Some((gpu, rb, ptr as usize, false))
    }

    fn ensure_differs_capacity(&mut self, host: &Host, needed_bytes: u64) -> bool {
        if needed_bytes <= self.differs_rb.2 {
            return true;
        }
        let size = needed_bytes.next_multiple_of(256 << 10);
        let Some((gpu, rb, ptr, direct_flags)) =
            Self::allocate_flags(host, size, self.direct_flags)
        else {
            return false;
        };
        // Only this engine records these resources. The previous submission
        // has finished using them before this command list is recorded.
        self.differs_gpu = gpu;
        self.differs_rb = (rb, ptr, size);
        self.direct_flags = direct_flags;
        true
    }

    fn ensure_fetch_capacity(&mut self, host: &Host, needed_bytes: u64) -> Option<(ID3D12Resource, usize)> {
        if let Some((ref res, ptr, cap)) = self.fetch_rb
            && needed_bytes <= cap
        {
            return Some((res.clone(), ptr));
        }
        let size = needed_bytes.next_multiple_of(if needed_bytes <= 1 << 20 {
            1 << 20
        } else {
            8 << 20
        });
        let (rb, ptr) = host.readback_buffer(size)?;
        crate::host::name(&rb, || format!("card fetch readback {size:#x}"));
        // The previous fetch and merge finished under the submission lock.
        self.fetch_rb = Some((rb.clone(), ptr as usize, size));
        Some((rb, ptr as usize))
    }

    /// Dispatches `CARD_COMPARE` for `count` contiguous 64 KiB pieces starting at `first`.
    unsafe fn dispatch_hash(
        &self,
        list: &ID3D12GraphicsCommandList,
        vram: &ID3D12Resource,
        hashes: &ID3D12Resource,
        differs: &ID3D12Resource,
        constants: [u32; 6],
    ) {
        let count = constants[3];
        if count == 0 {
            return;
        }
        // SAFETY: caller ensures `list` is open with `root_sig` and `pso` bound
        // and all 3 buffers are live with valid states.
        unsafe {
            list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
            list.SetComputeRootShaderResourceView(1, vram.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(2, hashes.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(3, differs.GetGPUVirtualAddress());
            list.Dispatch(count, 1, 1);
        }
    }

    /// Uploads all CPU-modified `Card` ranges to VRAM, initializes hashes for
    /// any `Card` shadow about to be written by shaders, and re-hashes uploaded
    /// pieces of hashed shadows. Returns `(optional before command list, CardPlan)`.
    pub(crate) fn record_before(
        &mut self,
        host: &Host,
        writes: &[(Arc<crate::shadow::Shadow>, usize, usize)],
    ) -> (Option<ID3D12CommandList>, CardPlan) {
        // 1. Drain all card shadows with pending CPU -> VRAM uploads.
        let mut dirty_cards: Vec<DirtyCard> = Vec::new();
        {
            let mut q = host.card_dirty.lock().unwrap_or_else(|e| e.into_inner());
            // Swap reusable queue storage, then release the global lock
            // before inspecting cards or sorting their dirty ranges.
            std::mem::swap(&mut *q, &mut self.dirty_queue);
        }
        for weak in self.dirty_queue.drain(..) {
            if let Some(card) = weak.upgrade() {
                let ranges = card.take_dirty_up();
                if !ranges.is_empty() {
                    dirty_cards.push((card, ranges));
                }
            }
        }

        // 2. Group declared `Step::GpuWrite` ranges by `Card` shadow and ensure
        //    each written card shadow has its hash table allocated.
        let mut entries: Vec<CardWriteEntry> = Vec::new();
        let mut entry_indices = std::collections::HashMap::<_, usize>::new();
        let mut need_full_hash: Vec<(Arc<CardState>, ID3D12Resource)> = Vec::new();
        let mut full_hash_cards = std::collections::HashSet::new();
        let mut total_pieces = 0u32;

        for (shadow, address, len) in writes {
            let key = Arc::as_ptr(shadow);
            if let Some(&index) = entry_indices.get(&key) {
                entries[index].declared.push((*address, *len));
                continue;
            }
            let Some(card) = shadow.card_state() else {
                continue;
            };
            let Some((hashes, full_init)) = card.ensure_hashes(host) else {
                crate::host::fail(format_args!("cannot allocate CARD_COMPARE hashes"));
            };
            if full_init && full_hash_cards.insert(Arc::as_ptr(&card)) {
                need_full_hash.push((card.clone(), hashes.clone()));
            }
            entry_indices.insert(key, entries.len());
            entries.push(CardWriteEntry {
                shadow: shadow.clone(),
                card,
                hashes,
                declared: vec![(*address, *len)],
                runs: Vec::new(),
                out_start: 0,
            });
        }

        for entry in &mut entries {
            let mut byte_ranges = Vec::with_capacity(entry.declared.len());
            for &(addr, len) in &entry.declared {
                let (from, to) = entry.shadow.inside_offsets(addr, len);
                if from < to {
                    byte_ranges.push((from, to));
                }
            }
            entry.runs = byte_ranges_to_piece_runs(&byte_ranges, entry.card.total_pieces());
            entry.out_start = total_pieces;
            total_pieces = total_pieces
                .checked_add(entry.runs.iter().map(|&(_, c)| c).sum::<u32>())
                .filter(|&pieces| pieces <= u32::MAX / 4)
                .unwrap_or_else(|| crate::host::fail(format_args!("CARD_COMPARE flag offset overflow")));
        }

        // Also check which dirty-uploaded shadows already have hashes (so we
        // re-hash their newly uploaded pieces after CopyBufferRegion).
        let mut rehash_uploads: Vec<RehashUpload> = Vec::new();
        let mut partial_hash_cards = std::collections::HashSet::new();
        for (card, ranges) in &dirty_cards {
            // Skip partial rehash if this card is already getting a full hash below.
            if full_hash_cards.contains(&Arc::as_ptr(card)) {
                continue;
            }
            if card.hashes_ready.load(Ordering::Acquire)
                && let Some(hashes) = card
                    .hashes
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone()
            {
                let mut runs = byte_ranges_to_piece_runs(ranges, card.total_pieces());
                join_rehash_runs(&mut runs, rehash_gap());
                if !runs.is_empty() {
                    partial_hash_cards.insert(Arc::as_ptr(card));
                    rehash_uploads.push((card.clone(), hashes, runs));
                }
            }
        }

        let plan = CardPlan {
            entries,
            total_pieces,
            uploads: dirty_cards,
            prefetched: None,
        };

        if plan.uploads.is_empty() && need_full_hash.is_empty() {
            return (None, plan);
        }

        // SAFETY: allocator & list are idle under `Host::submitted`.
        let list = unsafe {
            if self.before_alloc.Reset().is_err()
                || self.before_list.Reset(&self.before_alloc, None).is_err()
            {
                crate::host::fail(format_args!("cannot reset CARD_COMPARE upload list"));
            }
            &self.before_list
        };
        let timed = crate::cardmeter::begin(list, crate::cardmeter::Kind::Sync);

        // Step A: Copy dirty staging ranges -> VRAM.
        for (card, ranges) in &plan.uploads {
            for &(from, to) in ranges {
                // SAFETY: both buffers cover `[from, to)`; implicit COMMON promotion.
                unsafe {
                    list.CopyBufferRegion(
                        &card.vram,
                        from as u64,
                        &card.staging,
                        from as u64,
                        (to - from) as u64,
                    );
                }
            }
        }

        // Step B: Transition uploaded VRAM buffers COPY_DEST -> NON_PIXEL_SHADER_RESOURCE
        // (or COMMON if no hashing follows).
        let mut to_srv = Vec::new();
        let mut to_common = Vec::new();
        for (card, _) in &plan.uploads {
            let key = Arc::as_ptr(card);
            let will_hash = full_hash_cards.contains(&key) || partial_hash_cards.contains(&key);
            let after = if will_hash {
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
            } else {
                D3D12_RESOURCE_STATE_COMMON
            };
            to_srv.push(transition_barrier(
                &card.vram,
                D3D12_RESOURCE_STATE_COPY_DEST,
                after,
            ));
            if will_hash {
                to_common.push(transition_barrier(
                    &card.vram,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                ));
            }
        }
        if !to_srv.is_empty() {
            // SAFETY: valid resource transitions.
            unsafe { list.ResourceBarrier(&to_srv) };
            drop_barriers(to_srv);
        }

        // Step C: Dispatch Wave64 CARD_COMPARE (compare = 0) to store hashes.
        if !need_full_hash.is_empty() || !rehash_uploads.is_empty() {
            // SAFETY: live root signature and PSO.
            unsafe {
                list.SetComputeRootSignature(&self.root_sig);
                list.SetPipelineState(&self.pso);
            }
            for (card, hashes) in &need_full_hash {
                // SAFETY: buffer is readable as SRV (either transitioned above or COMMON).
                unsafe {
                    self.dispatch_hash(
                        list,
                        &card.vram,
                        hashes,
                        &self.dummy_uav,
                        [0, 0, card.size as u32, card.total_pieces(), 0, 0],
                    );
                }
            }
            for (card, hashes, runs) in &rehash_uploads {
                for &(first, count) in runs {
                    // SAFETY: buffer is in NON_PIXEL_SHADER_RESOURCE.
                    unsafe {
                        self.dispatch_hash(
                            list,
                            &card.vram,
                            hashes,
                            &self.dummy_uav,
                            [first, 0, card.size as u32, count, 0, 0],
                        );
                    }
                }
            }
            // Flush hash UAV writes and return transitioned VRAM buffers to COMMON.
            to_common.push(uav_barrier());
        }

        if !to_common.is_empty() {
            // SAFETY: valid barriers.
            unsafe { list.ResourceBarrier(&to_common) };
            drop_barriers(to_common);
        }

        crate::cardmeter::end(list, timed);
        // SAFETY: close command list.
        let cmd = unsafe { list.Close() }.and_then(|()| list.cast()).unwrap_or_else(|e| {
            crate::host::fail(format_args!("cannot close CARD_COMPARE upload list: {e}"))
        });
        for (card, _) in need_full_hash {
            card.hashes_ready.store(true, Ordering::Release);
        }
        (Some(cmd), plan)
    }

    /// Records the post-submission GPU hash compare (`compare = 2`) and copies
    /// the tiny `differs` array to `differs_rb`, or writes it directly into
    /// a mapped L0 UAV. For small opt-in plans, also copy the declared pages
    /// into the readback buffer before that same submission fence.
    pub(crate) fn record_after(&mut self, host: &Host, plan: &mut CardPlan) -> Option<ID3D12CommandList> {
        if plan.total_pieces == 0 {
            return None;
        }
        let needed_bytes = u64::from(plan.total_pieces) * 4;
        if !self.ensure_differs_capacity(host, needed_bytes) {
            crate::host::fail(format_args!("cannot allocate CARD_COMPARE flags ({needed_bytes} bytes)"));
        }
        plan.prefetched = prefetch_bytes(plan.total_pieces, self.prefetch_limit)
            .and_then(|bytes| self.ensure_fetch_capacity(host, bytes));
        // SAFETY: allocator & list are idle under `Host::submitted`.
        let cmd: windows::core::Result<ID3D12CommandList> = (|| unsafe {
            self.after_alloc.Reset()?;
            self.after_list.Reset(&self.after_alloc, None)?;
            let timed = crate::cardmeter::begin(&self.after_list, crate::cardmeter::Kind::Check);
            // Global UAV barrier ensures all shader writes from `runs` are visible.
            self.after_list.ResourceBarrier(&[uav_barrier()]);
            self.after_list.SetComputeRootSignature(&self.root_sig);
            self.after_list.SetPipelineState(&self.pso);

            for entry in &plan.entries {
                let mut out_idx = entry.out_start;
                for &(first, count) in &entry.runs {
                    self.dispatch_hash(
                        &self.after_list,
                        &entry.card.vram,
                        &entry.hashes,
                        &self.differs_gpu,
                        // Mode 2 compares and stores the updated hash.
                        [first, 0, entry.card.size as u32, count, out_idx * 4, 2],
                    );
                    out_idx += count;
                }
            }

            if let Some((readback, _)) = &plan.prefetched {
                // Hash SRV use precedes copies from the same buffers. Leave
                // each resource in COMMON for the next submission.
                for entry in &plan.entries {
                    if entry.runs.is_empty() {
                        continue;
                    }
                    let to_copy = [transition_barrier(
                        &entry.card.vram,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    )];
                    self.after_list.ResourceBarrier(&to_copy);
                    drop_barriers(to_copy);
                    let mut offset = u64::from(entry.out_start) * PAGE as u64;
                    for &(first, count) in &entry.runs {
                        let bytes = u64::from(count) * PAGE as u64;
                        self.after_list.CopyBufferRegion(
                            readback, offset, &entry.card.vram, u64::from(first) * PAGE as u64, bytes,
                        );
                        offset += bytes;
                    }
                    let to_common = [transition_barrier(
                        &entry.card.vram,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_COMMON,
                    )];
                    self.after_list.ResourceBarrier(&to_common);
                    drop_barriers(to_common);
                }
            }

            if self.direct_flags {
                // The CPU reads the mapped L0 flags only after the existing
                // submission fence. No GPU-to-readback copy is needed.
                let common = [transition_barrier(
                    &self.differs_gpu,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COMMON,
                )];
                self.after_list.ResourceBarrier(&common);
                drop_barriers(common);
            } else {
                let to_copy = [transition_barrier(
                    &self.differs_gpu,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )];
                self.after_list.ResourceBarrier(&to_copy);
                drop_barriers(to_copy);

                self.after_list.CopyBufferRegion(
                    &self.differs_rb.0,
                    0,
                    &self.differs_gpu,
                    0,
                    needed_bytes,
                );

                let back_common = [transition_barrier(
                    &self.differs_gpu,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                )];
                self.after_list.ResourceBarrier(&back_common);
                drop_barriers(back_common);
            }

            crate::cardmeter::end(&self.after_list, timed);
            self.after_list.Close()?;
            self.after_list.cast()
        })();
        Some(cmd.unwrap_or_else(|e| {
            crate::host::fail(format_args!("cannot record CARD_COMPARE results: {e}"))
        }))
    }

    /// Reads the `differs` results after the submission fence completes, fetches
    /// the 64 KiB pages the GPU actually modified if they were not prefetched,
    /// and merges only those pages into title memory and `staging`.
    pub(crate) fn finish_back(
        &mut self,
        host: &Host,
        plan: CardPlan,
        submitted: &mut u64,
    ) -> usize {
        if plan.entries.is_empty() {
            return 0;
        }
        let differs: &[u32] = if plan.total_pieces > 0 {
            // SAFETY: the caller waits for the compare list's fence; the
            // allocated mapping covers plan.total_pieces u32 flag values.
            unsafe {
                std::slice::from_raw_parts(
                    self.differs_rb.1 as *const u32,
                    plan.total_pieces as usize,
                )
            }
        } else {
            &[]
        };

        let prefetched = plan.prefetched.is_some();
        let entries = plan.entries.iter().map(|e| (e.out_start, e.runs.iter().copied()));
        let (dirty_runs, total_bytes) = dirty_readback_runs(entries, differs, prefetched);
        #[cfg(feature = "diagnostics")]
        if prefetched {
            FETCH_STATS[2].fetch_add(u64::from(plan.total_pieces) * PAGE as u64, Ordering::Relaxed);
            let copies = plan.entries.iter().map(|entry| entry.runs.len() as u64).sum();
            FETCH_STATS[3].fetch_add(copies, Ordering::Relaxed);
        }
        if dirty_runs.is_empty() {
            plan.clear_pending();
            return 0;
        }

        let (readback, ptr, span) = if let Some((resource, ptr)) = &plan.prefetched {
            (resource.clone(), *ptr, plan.total_pieces as usize * PAGE)
        } else {
            let Some((resource, ptr)) = self.ensure_fetch_capacity(host, total_bytes as u64) else {
                crate::host::fail(format_args!("cannot allocate CARD_COMPARE data readback ({total_bytes} bytes)"));
            };
            // SAFETY: both command objects are idle under Host::submitted.
            // The plan holds every source and the destination is owned here.
            let cmd: windows::core::Result<ID3D12CommandList> = (|| unsafe {
                self.fetch_alloc.Reset()?;
                self.fetch_list.Reset(&self.fetch_alloc, None)?;
                for run in &dirty_runs {
                    self.fetch_list.CopyBufferRegion(
                        &resource,
                        run.offset as u64,
                        &plan.entries[run.entry_idx].card.vram,
                        u64::from(run.first) * PAGE as u64,
                        u64::from(run.count) * PAGE as u64,
                    );
                }
                self.fetch_list.Close()?;
                self.fetch_list.cast()
            })();
            let cmd = cmd.unwrap_or_else(|e| {
                crate::host::fail(format_args!("cannot record CARD_COMPARE readback: {e}"))
            });
            #[cfg(feature = "diagnostics")]
            let fetch_started = std::time::Instant::now();
            host.execute_and_wait_locked(cmd, submitted);
            #[cfg(feature = "diagnostics")]
            {
                FETCH_STATS[0].fetch_add(1, Ordering::Relaxed);
                FETCH_STATS[1].fetch_add(fetch_started.elapsed().as_micros() as u64, Ordering::Relaxed);
                FETCH_STATS[2].fetch_add(total_bytes as u64, Ordering::Relaxed);
                FETCH_STATS[3].fetch_add(dirty_runs.len() as u64, Ordering::Relaxed);
            }
            (resource, ptr, total_bytes)
        };

        // SAFETY: the completed fence covers the copies into this live mapping.
        // Both layouts wrote the entire packed span. Submission serialization
        // prevents any reset or subsequent GPU writes while workers borrow it.
        let data = unsafe { std::slice::from_raw_parts(ptr as *const u8, span) };
        let (per_entry, pieces) = group_readback_runs(dirty_runs.iter().map(|run| {
            let end = run.offset + run.count as usize * PAGE;
            (run.entry_idx, run.first, &data[run.offset..end])
        }));
        let total_written = AtomicU64::new(0);
        crate::host::in_parallel(&per_entry, 4, |chunk| {
            for (entry_idx, span) in chunk {
                let entry = &plan.entries[*entry_idx];
                let n = entry.shadow.merge_card_pieces(&entry.card, &pieces[span.clone()]);
                total_written.fetch_add(n as u64, Ordering::Relaxed);
            }
        });
        drop(readback);
        let written = total_written.into_inner() as usize;


        // Clear all declared pending ranges on each card shadow.
        for entry in &plan.entries {
            entry.shadow.finish_card_pending(&entry.declared);
        }
        written
    }
}

fn transition_barrier(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: std::mem::ManuallyDrop::new(Some(resource.clone())),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    }
}

fn drop_barriers(barriers: impl IntoIterator<Item = D3D12_RESOURCE_BARRIER>) {
    for b in barriers {
        if b.Type == D3D12_RESOURCE_BARRIER_TYPE_TRANSITION {
            // SAFETY: we constructed this TRANSITION barrier with `ManuallyDrop`.
            let t = unsafe { std::mem::ManuallyDrop::into_inner(b.Anonymous.Transition) };
            let _ = std::mem::ManuallyDrop::into_inner(t.pResource);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_upload_budget_leaves_room_for_unallocated_card_buffers() {
        const GIB: u64 = 1 << 30;
        let budget = 12 * GIB;
        let target = 2 * GIB;
        assert_eq!(gpu_upload_limit(budget, target, 0), 7 * GIB);
        assert_eq!(gpu_upload_limit(budget, target, GIB), 8 * GIB);
        assert_eq!(gpu_upload_limit(budget, target, target), budget * 85 / 100);
        assert_eq!(gpu_upload_limit(budget, 0, 0), budget * 85 / 100);

        let budget = 6 * GIB;
        assert_eq!(gpu_upload_limit(budget, target, 0), 7 * GIB / 2);
        assert_eq!(gpu_upload_limit(0, target, 0), 0);
        assert!(gpu_upload_limit(u64::MAX, u64::MAX, 0) < u64::MAX);
    }

    #[test]
    fn committed_upload_budget_counts_the_whole_allocation() {
        let unit = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT as u64;
        assert!(!committed_upload_fits(unit, 2 * unit, 1));
        assert!(committed_upload_fits(unit, 2 * unit + 1, 0));
        assert!(!committed_upload_fits(unit, 3 * unit, unit + 1));
        assert!(!committed_upload_fits(0, u64::MAX, u64::MAX));
        assert!(!committed_upload_fits(u64::MAX - 1, u64::MAX, 1));
    }

    #[test]
    fn card_reservations_respect_limit_and_overflow() {
        let used = AtomicU64::new(u64::MAX - 8);
        assert!(!reserve_card_bytes(&used, 9, u64::MAX));
        assert_eq!(used.load(Ordering::Relaxed), u64::MAX - 8);
        assert!(reserve_card_bytes(&used, 8, u64::MAX));
        assert!(!reserve_card_bytes(&used, 1, u64::MAX));
        used.fetch_sub(8, Ordering::Relaxed);
        assert!(reserve_card_bytes(&used, 8, u64::MAX));
    }

    #[test]
    fn concurrent_card_reservations_share_one_limit() {
        let used = AtomicU64::new(0);
        let gate = std::sync::Barrier::new(8);
        let accepted = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        gate.wait();
                        reserve_card_bytes(&used, 64, 192)
                    })
                })
                .collect();
            workers
                .into_iter()
                .map(|worker| worker.join().unwrap())
                .filter(|&ok| ok)
                .count()
        });
        assert_eq!(accepted, 3);
        assert_eq!(used.load(Ordering::Relaxed), 192);
    }

    #[test]
    fn piece_runs_match_bitmap_with_unsorted_and_clipped_ranges() {
        let mut seed = 0x2934_5abcu32;
        for count in 0..128 {
            let mut ranges = Vec::new();
            let mut expected = [false; 31];
            for _ in 0..count {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let from = seed as usize % (40 * PAGE);
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let to = from + seed as usize % (4 * PAGE);
                ranges.push((from, to));
                if from < to {
                    for page in expected.iter_mut().take(to.div_ceil(PAGE)).skip(from / PAGE) {
                        *page = true;
                    }
                }
            }
            ranges.push((usize::MAX - PAGE, usize::MAX));
            ranges.push((PAGE, PAGE));
            let runs = byte_ranges_to_piece_runs(&ranges, expected.len() as u32);
            let mut actual = [false; 31];
            let mut previous_end = None;
            for (first, count) in runs {
                assert!(count > 0);
                assert!(previous_end.is_none_or(|end| first > end));
                let end = first + count;
                assert!(end <= actual.len() as u32);
                actual[first as usize..end as usize].fill(true);
                previous_end = Some(end);
            }
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn readback_groups_preserve_order_and_offsets() {
        let data = [1, 2, 3, 4, 5];
        let input = [
            (0, 1, &data[0..1]),
            (0, 5, &data[1..2]),
            (2, 0, &data[2..3]),
            (7, 9, &data[3..4]),
            (7, 11, &data[4..5]),
        ];
        let (groups, pieces) = group_readback_runs(input);
        assert_eq!(groups, vec![(0, 0..2), (2, 2..3), (7, 3..5)]);
        assert_eq!(
            pieces,
            input.map(|(_, first, bytes)| (first, bytes))
        );
        let (groups, pieces) = group_readback_runs(std::iter::empty());
        assert!(groups.is_empty());
        assert!(pieces.is_empty());
    }

    #[test]
    fn dirty_readback_layout_preserves_clean_holes_only_when_prefetched() {
        let entries = [(0, [(1, 3), (5, 2)]), (5, [(2, 2), (7, 1)])];
        let flags = [0, 1, 1, 1, 0, 0, 1, 1];
        for prefetched in [false, true] {
            let (runs, bytes) = dirty_readback_runs(entries, &flags, prefetched);
            assert_eq!(bytes, 5 * PAGE);
            assert_eq!(
                runs.iter().map(|r| (r.entry_idx, r.first, r.count)).collect::<Vec<_>>(),
                [(0, 2, 2), (0, 5, 1), (1, 3, 1), (1, 7, 1)]
            );
            let offsets = runs.iter().map(|r| r.offset / PAGE).collect::<Vec<_>>();
            assert_eq!(offsets, if prefetched { [1, 3, 6, 7] } else { [0, 2, 3, 4] });
        }
        assert_eq!(prefetch_bytes(0, 1 << 20), None);
        assert_eq!(prefetch_bytes(1, 0), None);
        assert_eq!(prefetch_bytes(16, 1 << 20), Some(1 << 20));
        assert_eq!(prefetch_bytes(17, 1 << 20), None);
    }

    #[test]
    fn rehash_gap_coalesces_only_small_gaps() {
        let original = vec![(0, 2), (4, 3), (8, 1), (20, 2)];
        let mut baseline = original.clone();
        join_rehash_runs(&mut baseline, 0);
        assert_eq!(baseline, original);

        let mut one = original.clone();
        join_rehash_runs(&mut one, 1);
        assert_eq!(one, vec![(0, 2), (4, 5), (20, 2)]);

        let mut two = original;
        join_rehash_runs(&mut two, 2);
        assert_eq!(two, vec![(0, 9), (20, 2)]);
        join_rehash_runs(&mut Vec::new(), 4);
    }

    #[test]
    fn range_helpers_merge_and_convert_pieces() {
        let mut r = vec![(100, 200), (50, 120), (300, 400), (200, 300)];
        merge_byte_ranges(&mut r);
        assert_eq!(r, vec![(50, 400)]);

        let runs = byte_ranges_to_piece_runs(
            &[(0, 10), (PAGE - 1, PAGE + 1), (3 * PAGE, 4 * PAGE)],
            8,
        );
        assert_eq!(runs, vec![(0, 2), (3, 1)]);
    }

    #[test]
    fn warp_card_roundtrip_upload_wave64_hash_and_concurrent_merge() {
        card_roundtrip(false, false);
        card_roundtrip(true, true);
    }

    fn card_roundtrip(direct: bool, prefetch: bool) {
        let Some(host) = Host::new_warp() else {
            return;
        };
        let mut engine = CardEngine::new_with_options(&host, direct, prefetch)
            .expect("CardEngine should build on WARP");
        assert_eq!(engine.direct_flags, direct);
        assert!(engine.ensure_differs_capacity(&host, (256 << 10) + 4));
        assert_eq!(engine.direct_flags, direct);
        let pages = 4usize;
        let size = (pages * PAGE) as u64;
        let mut title = vec![0u8; size as usize];
        for (i, b) in title.iter_mut().enumerate() {
            *b = ((i.wrapping_mul(31) ^ (i >> 8)) & 0xFF) as u8;
        }
        let base = title.as_ptr() as usize;

        let (staging, staging_ptr) = host.shared_buffer(size).expect("staging buffer");
        let card = CardState::new_unchecked(&host, &staging, size, false).expect("card buffer");
        let shadow = crate::shadow::Shadow::new_card_test(
            card.clone(),
            staging_ptr,
            base as u64,
            size,
        );
        let mut submitted = 0u64;

        // Phase 1: Initial CPU copy -> upload to VRAM + Wave64 hash, no GPU writes.
        shadow.copy_for_test(base, base + size as usize);
        shadow.writing(base, size as usize);
        let (before, mut plan) = engine.record_before(&host, &[(shadow.clone(), base, size as usize)]);
        assert!(before.is_some(), "initial upload + hash must emit before list");
        assert_eq!(plan.total_pieces, pages as u32);
        let after = engine.record_after(&host, &mut plan).expect("after list");
        host.execute_and_wait_locked(before.unwrap(), &mut submitted);
        host.execute_and_wait_locked(after, &mut submitted);

        // SAFETY: command queue finished `after`, so `differs_rb` is coherent on CPU.
        let differs_1 = unsafe {
            std::slice::from_raw_parts(engine.differs_rb.1 as *const u32, pages).to_vec()
        };
        assert_eq!(differs_1, vec![0, 0, 0, 0]);
        let written_1 = engine.finish_back(&host, plan, &mut submitted);
        assert_eq!(written_1, 0);

        // Phase 2: GPU writes 4 bytes on page 2 while CPU concurrently writes
        // 4 different bytes on the same page 2.
        let (patch_buf, patch_ptr) = host.shared_buffer(256).expect("patch buffer");
        let gpu_bytes = [0xDEu8, 0xAD, 0xBE, 0xEF];
        // SAFETY: `patch_ptr` points to a 256-byte persistently mapped upload buffer.
        unsafe { std::ptr::copy_nonoverlapping(gpu_bytes.as_ptr(), patch_ptr, 4) };
        // SAFETY: valid D3D12 device/command list calls on live test resources.
        unsafe {
            let alloc: ID3D12CommandAllocator = host
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .expect("alloc");
            let list: ID3D12GraphicsCommandList = host
                .device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
                .expect("list");
            list.CopyBufferRegion(
                &card.vram,
                (2 * PAGE + 128) as u64,
                &patch_buf,
                0,
                4,
            );
            list.Close().expect("close");
            let cmd: ID3D12CommandList = list.cast().expect("cast");
            // Execute immediately after record_before while alloc stays alive.
            shadow.writing(base, size as usize);
            let (before2, mut plan2) =
                engine.record_before(&host, &[(shadow.clone(), base, size as usize)]);
            assert!(before2.is_none(), "no CPU dirty uploads expected in phase 2");
            let after2 = engine.record_after(&host, &mut plan2).expect("after2");
            assert_eq!(plan2.prefetched.is_some(), prefetch);
            // Simulate concurrent CPU modification on page 2 while GPU runs.
            title[2 * PAGE + 256..2 * PAGE + 260].copy_from_slice(&[0x11, 0x22, 0x33, 0x44]);
            host.execute_and_wait_locked(cmd, &mut submitted);
            drop((list, alloc));
            host.execute_and_wait_locked(after2, &mut submitted);

            let differs_2 =
                std::slice::from_raw_parts(engine.differs_rb.1 as *const u32, pages).to_vec();
            assert_eq!(differs_2, vec![0, 0, 1, 0], "only page 2 should differ");
            let compare_fence = submitted;
            let written_2 = engine.finish_back(&host, plan2, &mut submitted);
            assert_eq!(submitted, compare_fence + u64::from(!prefetch));
            // The initial pattern already contains 0xBE at the third byte.
            assert_eq!(written_2, 3, "copy-back counts only changed bytes");
            assert_eq!(&title[2 * PAGE + 128..2 * PAGE + 132], &gpu_bytes);
            assert_eq!(&title[2 * PAGE + 256..2 * PAGE + 260], &[0x11, 0x22, 0x33, 0x44]);

            // Phase 3: Next submission uploads merged concurrent CPU write on page 2
            // back to VRAM and re-hashes page 2 so subsequent compare reports 0 diffs.
            shadow.writing(base, size as usize);
            let (before3, mut plan3) =
                engine.record_before(&host, &[(shadow.clone(), base, size as usize)]);
            assert!(
                before3.is_some(),
                "concurrent CPU write must queue page 2 for re-upload and re-hash"
            );
            let after3 = engine.record_after(&host, &mut plan3).expect("after3");
            host.execute_and_wait_locked(before3.unwrap(), &mut submitted);
            host.execute_and_wait_locked(after3, &mut submitted);
            let written_3 = engine.finish_back(&host, plan3, &mut submitted);
            assert_eq!(written_3, 0, "after re-upload + re-hash, 0 bytes should differ");

            // Read back page 2 directly from VRAM and verify both GPU and CPU writes exist there.
            let (rb, rb_ptr) = host.readback_buffer(PAGE as u64).expect("verify rb");
            let v_alloc: ID3D12CommandAllocator = host
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .expect("v_alloc");
            let v_list: ID3D12GraphicsCommandList = host
                .device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &v_alloc, None)
                .expect("v_list");
            v_list.CopyBufferRegion(&rb, 0, &card.vram, (2 * PAGE) as u64, PAGE as u64);
            v_list.Close().expect("v_close");
            let v_cmd: ID3D12CommandList = v_list.cast().expect("v_cast");
            host.execute_and_wait_locked(v_cmd, &mut submitted);
            let vram_page2 = std::slice::from_raw_parts(rb_ptr, PAGE);
            assert_eq!(&vram_page2[128..132], &gpu_bytes);
            assert_eq!(&vram_page2[256..260], &[0x11, 0x22, 0x33, 0x44]);
        }
    }
}
