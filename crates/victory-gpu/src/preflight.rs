//! Pre-submission D3D12 safety checks: DXIL container sanity, descriptor and
//! GPU virtual-address bounds, resource lifetimes, command-list replay rules,
//! fence cycle detection, resource barrier consistency, and bounded waits.
//!
//! Kept free of D3D12 device/driver calls so every invariant can be verified
//! exhaustively in CPU-only test runs and invoked before driver submission.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

/// Maximum thread groups per dispatch dimension guaranteed by D3D12 (`CS 5.0+`).
pub const MAX_DISPATCH_GROUPS_PER_DIM: u32 = 65_535;
/// Default finite fence wait ceiling in test builds when `VICTORY_FENCE_TIMEOUT_MS`
/// is unset, preventing infinite stalls on hung queues/fences.
pub const DEFAULT_TEST_FENCE_TIMEOUT_MS: u64 = 30_000;

/// Structured pre-flight rejection reason.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreflightError {
    pub stage: &'static str,
    pub detail: String,
}

impl PreflightError {
    fn new(stage: &'static str, detail: impl Into<String>) -> Self {
        Self {
            stage,
            detail: detail.into(),
        }
    }
}

impl fmt::Display for PreflightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "preflight [{}]: {}", self.stage, self.detail)
    }
}

impl std::error::Error for PreflightError {}

fn read_u32_le(bytes: &[u8], offset: usize) -> Option<u32> {
    let slice = bytes.get(offset..offset.checked_add(4)?)?;
    Some(u32::from_le_bytes(slice.try_into().ok()?))
}

/// Validates a DXBC/DXIL container layout before handing bytecode to a D3D12
/// driver (`CreateComputePipelineState` / `CreateGraphicsPipelineState` /
/// `CreatePipelineState`).
pub fn validate_dxil_container(bytes: &[u8]) -> Result<(), PreflightError> {
    if bytes.len() < 32 {
        return Err(PreflightError::new(
            "dxil",
            format!("container too short: {} bytes (< 32)", bytes.len()),
        ));
    }
    if &bytes[0..4] != b"DXBC" {
        return Err(PreflightError::new(
            "dxil",
            format!("missing DXBC container magic: {:?}", &bytes[0..4]),
        ));
    }
    let declared_size = read_u32_le(bytes, 24).ok_or_else(|| {
        PreflightError::new("dxil", "missing DXBC total size header")
    })? as usize;
    if declared_size < 32 || declared_size > bytes.len() {
        return Err(PreflightError::new(
            "dxil",
            format!(
                "invalid DXBC container size {declared_size} for {}-byte buffer",
                bytes.len()
            ),
        ));
    }
    let part_count = read_u32_le(bytes, 28).ok_or_else(|| {
        PreflightError::new("dxil", "missing DXBC part count")
    })? as usize;
    if part_count == 0 || part_count > 64 {
        return Err(PreflightError::new(
            "dxil",
            format!("implausible DXBC part count: {part_count}"),
        ));
    }
    let table_end = 32usize
        .checked_add(part_count.checked_mul(4).ok_or_else(|| {
            PreflightError::new("dxil", "part table size overflow")
        })?)
        .ok_or_else(|| PreflightError::new("dxil", "part table end overflow"))?;
    if table_end > declared_size {
        return Err(PreflightError::new(
            "dxil",
            format!("part offset table ({table_end}) exceeds container size ({declared_size})"),
        ));
    }

    let mut has_shader_part = false;
    let mut prev_offset = table_end.saturating_sub(1);
    for idx in 0..part_count {
        let off = read_u32_le(bytes, 32 + idx * 4).ok_or_else(|| {
            PreflightError::new("dxil", format!("missing part {idx} offset"))
        })? as usize;
        if off < table_end || off <= prev_offset && idx > 0 {
            return Err(PreflightError::new(
                "dxil",
                format!("part {idx} offset {off:#x} is out of order or overlaps table"),
            ));
        }
        prev_offset = off;
        let part_end_hdr = off.checked_add(8).ok_or_else(|| {
            PreflightError::new("dxil", "part header end overflow")
        })?;
        if part_end_hdr > declared_size {
            return Err(PreflightError::new(
                "dxil",
                format!("part {idx} header truncated at {off:#x}"),
            ));
        }
        let fourcc = &bytes[off..off + 4];
        let part_size = read_u32_le(bytes, off + 4).ok_or_else(|| {
            PreflightError::new("dxil", "missing part size")
        })? as usize;
        let payload_start = off + 8;
        let payload_end = payload_start.checked_add(part_size).ok_or_else(|| {
            PreflightError::new("dxil", "part payload end overflow")
        })?;
        if payload_end > declared_size {
            return Err(PreflightError::new(
                "dxil",
                format!(
                    "part {idx} ({:?}) size {part_size} exceeds container ({declared_size})",
                    String::from_utf8_lossy(fourcc)
                ),
            ));
        }
        if matches!(fourcc, b"DXIL" | b"ILDB" | b"SHEX" | b"SHDR") {
            has_shader_part = true;
            if fourcc == b"DXIL" {
                validate_dxil_program_header(&bytes[payload_start..payload_end])?;
            }
        }
    }
    if !has_shader_part {
        return Err(PreflightError::new(
            "dxil",
            "container contains no shader bytecode chunk (DXIL/ILDB/SHEX/SHDR)",
        ));
    }
    Ok(())
}

/// Validates a DXBC/DXIL container structurally and, whenever `dxil.dll`
/// (`IDxcValidator`) is available and `VICTORY_VALIDATE_DXIL != "0"`, runs
/// full semantic DXIL verification before passing the shader to the driver.
pub fn validate_dxil_semantic(bytes: &[u8]) -> Result<(), PreflightError> {
    validate_dxil_container(bytes)?;
    if std::env::var_os("VICTORY_VALIDATE_DXIL").is_some_and(|v| v == "0") {
        return Ok(());
    }
    static VALIDATOR: OnceLock<
        Result<Mutex<victory_shader::emit::validator::Validator>, String>,
    > = OnceLock::new();
    let loaded = VALIDATOR.get_or_init(|| {
        let path = victory_shader::emit::validator::Validator::default_path();
        victory_shader::emit::validator::Validator::load(path.as_deref()).map(Mutex::new)
    });
    match loaded {
        Ok(lock) => {
            let mut validator = lock.lock().unwrap_or_else(|e| e.into_inner());
            validator
                .validate_readonly(bytes)
                .map_err(|err| PreflightError::new("dxil_semantic", err))
        }
        Err(load_err) => {
            if std::env::var_os("VICTORY_REQUIRE_DXIL_VALIDATOR").is_some_and(|v| v == "1") {
                return Err(PreflightError::new(
                    "dxil_semantic",
                    format!("IDxcValidator unavailable: {load_err}"),
                ));
            }
            static WARNED: OnceLock<()> = OnceLock::new();
            WARNED.get_or_init(|| {
                tracing::warn!(error = %load_err, "dxil.dll unavailable; semantic DXIL validation skipped");
            });
            Ok(())
        }
    }
}

/// Persists the exact pre-GPU operation into `VICTORY_LAST_OP_FILE` (when set
/// by `cargo xtask gpu-gate`) so post-reboot crash/hang triage sees the real
/// GPU call that executed last.
pub fn record_last_op(op: impl fmt::Display) {
    static TARGET: OnceLock<Option<Mutex<PathBuf>>> = OnceLock::new();
    let target = TARGET.get_or_init(|| {
        std::env::var_os("VICTORY_LAST_OP_FILE")
            .filter(|v| !v.is_empty())
            .map(|v| Mutex::new(PathBuf::from(v)))
    });
    let Some(lock) = target else {
        return;
    };
    let text = op.to_string();
    if let Ok(path) = lock.lock() {
        record_last_op_path(Some(&path), &text);
    }
}

pub(crate) fn record_last_op_path(path: Option<&Path>, op: &str) {
    let Some(path) = path else {
        return;
    };
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)
    {
        let _ = file.write_all(op.as_bytes());
        let _ = file.sync_data();
    }
}

fn validate_dxil_program_header(part: &[u8]) -> Result<(), PreflightError> {
    if part.len() < 24 {
        return Err(PreflightError::new(
            "dxil",
            format!("DXIL program header too short: {} bytes", part.len()),
        ));
    }
    let size_in_words = read_u32_le(part, 4).unwrap_or(0) as usize;
    let bitcode_offset = read_u32_le(part, 16).unwrap_or(0) as usize;
    let bitcode_size = read_u32_le(part, 20).unwrap_or(0) as usize;
    if size_in_words.saturating_mul(4) > part.len() {
        return Err(PreflightError::new(
            "dxil",
            format!("DXIL word size ({} bytes) exceeds chunk ({})", size_in_words * 4, part.len()),
        ));
    }
    // BitcodeOffset is relative to byte 8 (the DxilProgramHeader start after
    // the DXIL container's ProgramVersion + SizeInUint32 fields, or 16..24
    // header). Both common encodings place bitcode inside `part` with `'BC\xc0\xde'`.
    let bc_start = bitcode_offset
        .checked_add(8)
        .filter(|&s| part.get(s..s + 4) == Some(&[b'B', b'C', 0xC0, 0xDE]))
        .or_else(|| {
            (bitcode_offset + 4 <= part.len()
                && part.get(bitcode_offset..bitcode_offset + 4) == Some(&[b'B', b'C', 0xC0, 0xDE]))
                .then_some(bitcode_offset)
        })
        .ok_or_else(|| {
            PreflightError::new(
                "dxil",
                format!("DXIL chunk missing LLVM bitcode magic at offset {bitcode_offset}"),
            )
        })?;
    let bc_end = bc_start.checked_add(bitcode_size).ok_or_else(|| {
        PreflightError::new("dxil", "DXIL bitcode end overflow")
    })?;
    if bitcode_size < 4 || bc_end > part.len() {
        return Err(PreflightError::new(
            "dxil",
            format!(
                "DXIL bitcode range {bc_start}..{bc_end} out of {}-byte chunk",
                part.len()
            ),
        ));
    }
    Ok(())
}

/// Validates a single descriptor heap slot write (not exceeding heap bounds or
/// mutating a reserved capped null slot as dynamic data).
pub fn validate_descriptor_slot(
    index: u32,
    total_slots: u32,
    capped_null_slot: Option<u32>,
) -> Result<(), PreflightError> {
    if index >= total_slots {
        return Err(PreflightError::new(
            "descriptor",
            format!("descriptor index {index} out of bounds (total {total_slots})"),
        ));
    }
    if capped_null_slot == Some(index) {
        return Err(PreflightError::new(
            "descriptor",
            format!("attempted dynamic write into reserved capped null slot {index}"),
        ));
    }
    Ok(())
}

/// Validates a contiguous dynamic descriptor allocation range.
pub fn validate_descriptor_run(
    first: u32,
    count: u32,
    heap_base: u32,
    heap_spare: u32,
    capped_null_slot: Option<u32>,
) -> Result<(), PreflightError> {
    if count == 0 {
        return Err(PreflightError::new("descriptor", "empty descriptor run"));
    }
    let end = first.checked_add(count).ok_or_else(|| {
        PreflightError::new("descriptor", "descriptor run overflow")
    })?;
    let limit = heap_base.checked_add(heap_spare).ok_or_else(|| {
        PreflightError::new("descriptor", "descriptor heap limit overflow")
    })?;
    if first < heap_base || end > limit {
        return Err(PreflightError::new(
            "descriptor",
            format!("descriptor run {first}..{end} outside dynamic region {heap_base}..{limit}"),
        ));
    }
    if let Some(null_slot) = capped_null_slot
        && (first..end).contains(&null_slot)
    {
        return Err(PreflightError::new(
            "descriptor",
            format!("descriptor run {first}..{end} overlaps capped null slot {null_slot}"),
        ));
    }
    Ok(())
}

/// Validates a GPU virtual-address range for alignment and overflow.
pub fn validate_gpu_range(va: u64, size: u64, align: u64) -> Result<(), PreflightError> {
    if size == 0 {
        return Err(PreflightError::new("memory", "zero-sized GPU range"));
    }
    if va == 0 {
        return Err(PreflightError::new("memory", "null GPU virtual address"));
    }
    if align > 1 && !align.is_power_of_two() {
        return Err(PreflightError::new(
            "memory",
            format!("non-power-of-two alignment requirement: {align}"),
        ));
    }
    if align > 1 && !va.is_multiple_of(align) {
        return Err(PreflightError::new(
            "memory",
            format!("GPU address {va:#x} is not {align}-byte aligned"),
        ));
    }
    if va.checked_add(size).is_none() {
        return Err(PreflightError::new(
            "memory",
            format!("GPU range {va:#x} + {size:#x} overflows 64-bit address space"),
        ));
    }
    Ok(())
}

/// Validates a buffer-to-buffer copy against both buffer capacities.
pub fn validate_buffer_copy(
    src_offset: u64,
    dst_offset: u64,
    len: u64,
    src_size: u64,
    dst_size: u64,
) -> Result<(), PreflightError> {
    let src_end = src_offset.checked_add(len).ok_or_else(|| {
        PreflightError::new("memory", "source copy range overflow")
    })?;
    let dst_end = dst_offset.checked_add(len).ok_or_else(|| {
        PreflightError::new("memory", "destination copy range overflow")
    })?;
    if src_end > src_size {
        return Err(PreflightError::new(
            "memory",
            format!("source copy {src_offset}..{src_end} exceeds buffer size {src_size}"),
        ));
    }
    if dst_end > dst_size {
        return Err(PreflightError::new(
            "memory",
            format!("destination copy {dst_offset}..{dst_end} exceeds buffer size {dst_size}"),
        ));
    }
    Ok(())
}

/// Snapshot of a command list submitted in an `ExecuteCommandLists` batch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ListExecMeta {
    pub key: usize,
    pub closed: bool,
    pub host_fence: usize,
    pub list_fence: usize,
    pub last_fence: u64,
    pub completed_fence: u64,
    pub live_kept_resources: bool,
}

/// Validates whether a batch of closed command lists may safely be submitted or
/// re-executed.
pub fn validate_submission_batch(lists: &[ListExecMeta]) -> Result<(), PreflightError> {
    let mut seen = HashSet::with_capacity(lists.len());
    for (idx, meta) in lists.iter().enumerate() {
        if meta.key == 0 {
            return Err(PreflightError::new(
                "command_list",
                format!("list {idx} has null command-list handle"),
            ));
        }
        if !seen.insert(meta.key) {
            return Err(PreflightError::new(
                "command_list",
                format!(
                    "duplicate closed command list {:#x} in the same submission batch",
                    meta.key
                ),
            ));
        }
        if !meta.closed {
            return Err(PreflightError::new(
                "command_list",
                format!("list {:#x} is not closed before submission", meta.key),
            ));
        }
        if !meta.live_kept_resources {
            return Err(PreflightError::new(
                "lifetime",
                format!("list {:#x} has released resources before submission", meta.key),
            ));
        }
        if meta.list_fence != 0
            && meta.list_fence != meta.host_fence
            && meta.last_fence > meta.completed_fence
        {
            return Err(PreflightError::new(
                "lifetime",
                format!(
                    "list {:#x} still has pending fence {} (> completed {}) on another queue",
                    meta.key, meta.last_fence, meta.completed_fence
                ),
            ));
        }
    }
    Ok(())
}

/// Directed fence synchronization step on a queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FenceOpKind {
    Wait,
    Signal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FenceEdge {
    pub queue_id: usize,
    pub fence_id: usize,
    pub value: u64,
    pub completed: u64,
    pub kind: FenceOpKind,
}

/// Checks a sequence of pending queue fence operations for backwards/stalled
/// signal values and cross-queue wait/signal deadlocks.
pub fn validate_fence_graph(ops: &[FenceEdge]) -> Result<(), PreflightError> {
    let mut last_signal_on_fence: HashMap<usize, u64> = HashMap::new();
    let mut waiters_by_fence: HashMap<(usize, u64), Vec<usize>> = HashMap::new();
    let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();

    for op in ops {
        match op.kind {
            FenceOpKind::Signal => {
                if op.value <= op.completed {
                    return Err(PreflightError::new(
                        "fence",
                        format!(
                            "fence {:#x} signal value {} is not ahead of completed {}",
                            op.fence_id, op.value, op.completed
                        ),
                    ));
                }
                if let Some(&prev) = last_signal_on_fence.get(&op.fence_id)
                    && op.value <= prev
                {
                    return Err(PreflightError::new(
                        "fence",
                        format!(
                            "non-monotonic signal {} after {} on fence {:#x}",
                            op.value, prev, op.fence_id
                        ),
                    ));
                }
                last_signal_on_fence.insert(op.fence_id, op.value);
                if let Some(waiters) = waiters_by_fence.get(&(op.fence_id, op.value)) {
                    for &waiter_q in waiters {
                        if waiter_q == op.queue_id {
                            return Err(PreflightError::new(
                                "fence",
                                format!(
                                    "self-deadlock on queue {:#x}: wait({:#x}, {}) precedes signal on the same queue",
                                    op.queue_id, op.fence_id, op.value
                                ),
                            ));
                        }
                        adj.entry(waiter_q).or_default().push(op.queue_id);
                    }
                }
            }
            FenceOpKind::Wait => {
                if op.value > op.completed {
                    let already_emitted = last_signal_on_fence
                        .get(&op.fence_id)
                        .is_some_and(|&s| s >= op.value);
                    if !already_emitted {
                        waiters_by_fence
                            .entry((op.fence_id, op.value))
                            .or_default()
                            .push(op.queue_id);
                    }
                }
            }
        }
    }

    // Detect cyclic queue wait -> signal dependencies across queues.
    let mut state: HashMap<usize, u8> = HashMap::new(); // 1 = visiting, 2 = done
    let nodes: Vec<usize> = adj.keys().copied().collect();
    for start in nodes {
        if has_cycle(start, &adj, &mut state) {
            return Err(PreflightError::new(
                "fence",
                format!("cyclic cross-queue fence dependency involving queue {start:#x}"),
            ));
        }
    }
    Ok(())
}

fn has_cycle(
    node: usize,
    adj: &HashMap<usize, Vec<usize>>,
    state: &mut HashMap<usize, u8>,
) -> bool {
    match state.get(&node).copied() {
        Some(1) => return true,
        Some(2) => return false,
        _ => {}
    }
    state.insert(node, 1);
    if let Some(nexts) = adj.get(&node) {
        for &next in nexts {
            if has_cycle(next, adj, state) {
                return true;
            }
        }
    }
    state.insert(node, 2);
    false
}

/// Abstract D3D12 resource state transition for static/runtime barrier checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BarrierTransition {
    pub resource: usize,
    pub subresource: u32,
    pub before: u32,
    pub after: u32,
}

/// Checks resource transition batches for null resources, no-op barriers, and
/// contradictory consecutive transitions on the same subresource.
pub fn validate_resource_barriers(
    barriers: &[BarrierTransition],
) -> Result<(), PreflightError> {
    let mut tracked: HashMap<(usize, u32), u32> = HashMap::new();
    for (idx, b) in barriers.iter().enumerate() {
        if b.resource == 0 {
            return Err(PreflightError::new(
                "barrier",
                format!("barrier {idx} targets null resource"),
            ));
        }
        if b.before == b.after {
            return Err(PreflightError::new(
                "barrier",
                format!(
                    "barrier {idx} on resource {:#x} subresource {} has identical before/after state {:#x}",
                    b.resource, b.subresource, b.before
                ),
            ));
        }
        if let Some(&prev_after) = tracked.get(&(b.resource, b.subresource))
            && prev_after != b.before
        {
            return Err(PreflightError::new(
                "barrier",
                format!(
                    "barrier {idx} mismatch on resource {:#x} sub {}: previous after={:#x}, next before={:#x}",
                    b.resource, b.subresource, prev_after, b.before
                ),
            ));
        }
        tracked.insert((b.resource, b.subresource), b.after);
    }
    Ok(())
}

/// Validates dispatch dimensions against D3D12 per-axis caps and an overall
/// thread-group TDR budget.
pub fn validate_dispatch(
    x: u32,
    y: u32,
    z: u32,
    max_total_groups: u64,
) -> Result<(), PreflightError> {
    for (axis, dim) in [('x', x), ('y', y), ('z', z)] {
        if dim == 0 || dim > MAX_DISPATCH_GROUPS_PER_DIM {
            return Err(PreflightError::new(
                "dispatch",
                format!("dispatch {axis}={dim} outside 1..={MAX_DISPATCH_GROUPS_PER_DIM}"),
            ));
        }
    }
    let total = u64::from(x)
        .saturating_mul(u64::from(y))
        .saturating_mul(u64::from(z));
    if total > max_total_groups {
        return Err(PreflightError::new(
            "dispatch",
            format!("dispatch ({x}, {y}, {z}) = {total} groups exceeds safety limit {max_total_groups}"),
        ));
    }
    Ok(())
}

/// Parses `VICTORY_FENCE_TIMEOUT_MS`; returns a bounded finite wait in tests or
/// whenever explicitly configured (`0` disables timeout explicitly).
pub(crate) fn effective_fence_timeout_ms(env_val: Option<&str>, in_tests: bool) -> Option<u64> {
    match env_val.map(str::trim) {
        Some("") | None => in_tests.then_some(DEFAULT_TEST_FENCE_TIMEOUT_MS),
        Some("0") => None,
        Some(text) => match text.parse::<u64>() {
            Ok(ms) => Some(ms),
            Err(_) => Some(DEFAULT_TEST_FENCE_TIMEOUT_MS),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_dxbc_container(parts: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
        let header_size = 32usize;
        let table_size = parts.len() * 4;
        let mut out = vec![0u8; header_size + table_size];
        out[0..4].copy_from_slice(b"DXBC");
        let mut offsets = Vec::with_capacity(parts.len());
        for (fourcc, payload) in parts {
            offsets.push(out.len() as u32);
            out.extend_from_slice(*fourcc);
            out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            out.extend_from_slice(payload);
        }
        let total = out.len() as u32;
        out[24..28].copy_from_slice(&total.to_le_bytes());
        out[28..32].copy_from_slice(&(parts.len() as u32).to_le_bytes());
        for (i, off) in offsets.into_iter().enumerate() {
            out[32 + i * 4..36 + i * 4].copy_from_slice(&off.to_le_bytes());
        }
        out
    }

    fn build_dxil_chunk(bitcode: &[u8]) -> Vec<u8> {
        let mut chunk = vec![0u8; 24];
        chunk[16..20].copy_from_slice(&16u32.to_le_bytes()); // offset relative to byte 8 -> 24
        chunk[20..24].copy_from_slice(&(bitcode.len() as u32).to_le_bytes());
        chunk.extend_from_slice(bitcode);
        let words = (chunk.len() / 4) as u32;
        chunk[4..8].copy_from_slice(&words.to_le_bytes());
        chunk
    }

    #[test]
    fn dxil_container_validation_accepts_valid_and_rejects_corrupt_blobs() {
        let dxil_part = build_dxil_chunk(b"BC\xc0\xde\x01\x02\x03\x04");
        let valid = build_dxbc_container(&[(b"DXIL", &dxil_part)]);
        assert!(validate_dxil_container(&valid).is_ok());

        // Wrong magic
        let mut bad_magic = valid.clone();
        bad_magic[0..4].copy_from_slice(b"NOPE");
        assert!(validate_dxil_container(&bad_magic).is_err());

        // Truncated container
        assert!(validate_dxil_container(&valid[..16]).is_err());
        assert!(validate_dxil_container(&valid[..valid.len() - 4]).is_err());

        // Container with no shader chunk
        let no_shader = build_dxbc_container(&[(b"RDEF", b"1234")]);
        assert!(validate_dxil_container(&no_shader).is_err());

        // Corrupt DXIL bitcode magic
        let bad_bc = build_dxil_chunk(b"XXXX\x01\x02\x03\x04");
        let corrupt_bc = build_dxbc_container(&[(b"DXIL", &bad_bc)]);
        assert!(validate_dxil_container(&corrupt_bc).is_err());
    }

    #[test]
    fn descriptor_and_memory_bounds_checks_reject_overflow_and_null_slot() {
        assert!(validate_descriptor_slot(5, 16, Some(10)).is_ok());
        assert!(validate_descriptor_slot(10, 16, Some(10)).is_err());
        assert!(validate_descriptor_slot(16, 16, Some(10)).is_err());

        assert!(validate_descriptor_run(100, 10, 100, 64, Some(120)).is_ok());
        assert!(validate_descriptor_run(115, 10, 100, 64, Some(120)).is_err());
        assert!(validate_descriptor_run(160, 10, 100, 64, None).is_err());

        assert!(validate_gpu_range(0x10000, 4096, 256).is_ok());
        assert!(validate_gpu_range(0, 4096, 256).is_err());
        assert!(validate_gpu_range(0x10080, 4096, 256).is_err());
        assert!(validate_gpu_range(u64::MAX - 10, 64, 1).is_err());

        assert!(validate_buffer_copy(0, 128, 256, 512, 512).is_ok());
        assert!(validate_buffer_copy(300, 0, 256, 512, 512).is_err());
        assert!(validate_buffer_copy(0, 300, 256, 512, 512).is_err());
    }

    #[test]
    fn command_list_reexecution_and_lifetime_checks_catch_duplicates_and_hazards() {
        let ok = ListExecMeta {
            key: 0x1000,
            closed: true,
            host_fence: 1,
            list_fence: 1,
            last_fence: 2,
            completed_fence: 1,
            live_kept_resources: true,
        };
        assert!(validate_submission_batch(&[ok]).is_ok());

        // Duplicate same list in one batch is invalid for ExecuteCommandLists.
        assert!(validate_submission_batch(&[ok, ok]).is_err());

        // Unclosed list.
        let unclosed = ListExecMeta { closed: false, ..ok };
        assert!(validate_submission_batch(&[unclosed]).is_err());

        // Released resources.
        let dead = ListExecMeta {
            live_kept_resources: false,
            ..ok
        };
        assert!(validate_submission_batch(&[dead]).is_err());

        // Re-execution across different fence while previous fence is still pending.
        let cross_queue_pending = ListExecMeta {
            host_fence: 2,
            list_fence: 1,
            last_fence: 5,
            completed_fence: 4,
            ..ok
        };
        assert!(validate_submission_batch(&[cross_queue_pending]).is_err());
    }

    #[test]
    fn fence_graph_and_barrier_validators_detect_deadlocks_and_conflicts() {
        // Same-queue wait-before-signal deadlock.
        let self_deadlock = [
            FenceEdge {
                queue_id: 1,
                fence_id: 10,
                value: 1,
                completed: 0,
                kind: FenceOpKind::Wait,
            },
            FenceEdge {
                queue_id: 1,
                fence_id: 10,
                value: 1,
                completed: 0,
                kind: FenceOpKind::Signal,
            },
        ];
        assert!(validate_fence_graph(&self_deadlock).is_err());

        // Two-queue cycle: Q1 waits for F2 signalled by Q2; Q2 waits for F1 signalled by Q1.
        let cycle = [
            FenceEdge {
                queue_id: 1,
                fence_id: 20,
                value: 1,
                completed: 0,
                kind: FenceOpKind::Wait,
            },
            FenceEdge {
                queue_id: 2,
                fence_id: 10,
                value: 1,
                completed: 0,
                kind: FenceOpKind::Wait,
            },
            FenceEdge {
                queue_id: 1,
                fence_id: 10,
                value: 1,
                completed: 0,
                kind: FenceOpKind::Signal,
            },
            FenceEdge {
                queue_id: 2,
                fence_id: 20,
                value: 1,
                completed: 0,
                kind: FenceOpKind::Signal,
            },
        ];
        assert!(validate_fence_graph(&cycle).is_err());

        // Valid acyclic cross-queue handoff: Q1 signals F1, Q2 waits F1.
        let acyclic = [
            FenceEdge {
                queue_id: 1,
                fence_id: 10,
                value: 1,
                completed: 0,
                kind: FenceOpKind::Signal,
            },
            FenceEdge {
                queue_id: 2,
                fence_id: 10,
                value: 1,
                completed: 0,
                kind: FenceOpKind::Wait,
            },
        ];
        assert!(validate_fence_graph(&acyclic).is_ok());

        // Barrier chain consistency.
        assert!(
            validate_resource_barriers(&[
                BarrierTransition {
                    resource: 0xAA,
                    subresource: 0,
                    before: 1,
                    after: 8,
                },
                BarrierTransition {
                    resource: 0xAA,
                    subresource: 0,
                    before: 8,
                    after: 1,
                },
            ])
            .is_ok()
        );
        assert!(
            validate_resource_barriers(&[
                BarrierTransition {
                    resource: 0xAA,
                    subresource: 0,
                    before: 1,
                    after: 8,
                },
                BarrierTransition {
                    resource: 0xAA,
                    subresource: 0,
                    before: 4,
                    after: 1,
                },
            ])
            .is_err()
        );
    }

    #[test]
    fn dispatch_limits_and_fence_timeouts_are_bounded() {
        assert!(validate_dispatch(64, 1, 1, 1024).is_ok());
        assert!(validate_dispatch(0, 1, 1, 1024).is_err());
        assert!(validate_dispatch(65_536, 1, 1, 100_000).is_err());
        assert!(validate_dispatch(1024, 1024, 1, 100_000).is_err());

        assert_eq!(
            effective_fence_timeout_ms(None, true),
            Some(DEFAULT_TEST_FENCE_TIMEOUT_MS)
        );
        assert_eq!(effective_fence_timeout_ms(None, false), None);
        assert_eq!(effective_fence_timeout_ms(Some("0"), true), None);
        assert_eq!(effective_fence_timeout_ms(Some("5000"), false), Some(5000));
    }

    #[test]
    fn dxil_semantic_validator_rejects_structurally_valid_corrupt_bitcode() {
        let dxil_part = build_dxil_chunk(b"BC\xc0\xde\x01\x02\x03\x04\x05\x06\x07\x08");
        let container = build_dxbc_container(&[(b"DXIL", &dxil_part)]);
        assert!(validate_dxil_container(&container).is_ok());

        let path = victory_shader::emit::validator::Validator::default_path();
        if victory_shader::emit::validator::Validator::load(path.as_deref()).is_ok() {
            let err = validate_dxil_semantic(&container)
                .expect_err("IDxcValidator must reject truncated/corrupt DXIL bitcode");
            assert_eq!(err.stage, "dxil_semantic");
        }
    }

    #[test]
    fn last_op_recorder_persists_latest_gpu_operation_to_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("last_op.txt");
        record_last_op_path(Some(&path), "device_init:NVIDIA");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "device_init:NVIDIA");
        record_last_op_path(Some(&path), "execute_lists:count=2:sub=7");
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "execute_lists:count=2:sub=7"
        );
    }
}
