//! Host-memory pressure for reusable CPU and upload storage.
//!
//! Hysteresis adapted from codec_d's cool_fox, f4b48ad (GPL-3.0).
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

static PRESSURED: AtomicBool = AtomicBool::new(false);
static NEXT_COMMIT_SAMPLE: AtomicU64 = AtomicU64::new(0);
static AVAILABLE_COMMIT: AtomicU64 = AtomicU64::new(u64::MAX);
static RETRY_UPLOAD_AT: AtomicU64 = AtomicU64::new(0);

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

fn coarse_ms() -> u64 {
    static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
    EPOCH
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_millis() as u64
}

fn memory_status() -> Option<MEMORYSTATUSEX> {
    let mut status = MEMORYSTATUSEX {
        dwLength: size_of::<MEMORYSTATUSEX>() as u32,
        ..Default::default()
    };
    // SAFETY: the API fills a writable record sized by dwLength.
    (unsafe { GlobalMemoryStatusEx(&mut status) } != 0).then_some(status)
}

pub(crate) fn allocation_error(error: &windows::core::Error) {
    if error.code() == windows::Win32::Foundation::E_OUTOFMEMORY {
        RETRY_UPLOAD_AT.fetch_max(coarse_ms().saturating_add(500), Ordering::Relaxed);
    }
}

pub(crate) fn upload_refused() -> bool {
    coarse_ms() < RETRY_UPLOAD_AT.load(Ordering::Relaxed)
}

fn sample_commit(now: u64) -> u64 {
    let next = NEXT_COMMIT_SAMPLE.load(Ordering::Relaxed);
    if now >= next
        && NEXT_COMMIT_SAMPLE
            .compare_exchange(
                next,
                now.saturating_add(32),
                Ordering::Relaxed,
                Ordering::Relaxed,
            )
            .is_ok()
        && let Some(status) = memory_status()
    {
        AVAILABLE_COMMIT.store(status.ullAvailPageFile, Ordering::Relaxed);
    }
    AVAILABLE_COMMIT.load(Ordering::Relaxed)
}

pub(crate) fn critical_commit() -> bool {
    if !adapt_textures() {
        return false;
    }
    upload_refused() || sample_commit(coarse_ms()) < 1536 * MIB
}

pub(crate) fn reclaim_on_oom() -> bool {
    let mut freed = false;
    if trim_snapshots() {
        freed |= crate::shadow::reclaim_idle_snapshots() != 0;
    }
    if adapt_textures() {
        freed |= crate::texture::reclaim_idle_staging() != 0;
    }
    freed
}

pub(crate) fn texture_pressure() -> bool {
    adapt_textures() && PRESSURED.load(Ordering::Relaxed)
}

fn adapt_textures() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ADAPT_TEXTURE_STAGING").is_some_and(|v| v == "1"))
}

fn trim_snapshots() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TRIM_SNAPSHOTS").is_some_and(|v| v == "1"))
}

/// Sample once per sweep whether either cache has opted into adaptation.
/// A failed query leaves the previous good sample in effect.
pub(crate) fn refresh() {
    if !adapt_textures() && !trim_snapshots() {
        return;
    }
    let Some(status) = memory_status() else {
        return;
    };
    AVAILABLE_COMMIT.store(status.ullAvailPageFile, Ordering::Relaxed);
    NEXT_COMMIT_SAMPLE.store(coarse_ms().saturating_add(32), Ordering::Relaxed);
    let previous = PRESSURED.load(Ordering::Relaxed);
    let pressure = next_pressure(previous, status.ullAvailPhys, status.ullAvailPageFile);
    PRESSURED.store(pressure, Ordering::Relaxed);
    if pressure != previous {
        tracing::info!(
            pressure,
            free_ram_mib = status.ullAvailPhys >> 20,
            free_commit_mib = status.ullAvailPageFile >> 20,
            "host memory pressure changed"
        );
    }
}

pub(crate) fn snapshot_pressure() -> bool {
    trim_snapshots() && PRESSURED.load(Ordering::Relaxed)
}

fn next_pressure(previous: bool, free_ram: u64, free_commit: u64) -> bool {
    if previous {
        free_ram < 5 * GIB || free_commit < 6 * GIB
    } else {
        free_ram <= 3 * GIB || free_commit <= 4 * GIB
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_pressure_requires_both_resources_to_recover() {
        assert!(next_pressure(false, 3 * GIB, 8 * GIB));
        assert!(next_pressure(false, 8 * GIB, 4 * GIB));
        assert!(!next_pressure(false, 4 * GIB, 5 * GIB));
        assert!(next_pressure(true, 4 * GIB, 8 * GIB));
        assert!(next_pressure(true, 8 * GIB, 5 * GIB));
        assert!(!next_pressure(true, 5 * GIB, 6 * GIB));
    }

    #[test]
    fn oom_sets_upload_backoff_window() {
        let e_ok = windows::core::Error::from(windows::Win32::Foundation::E_INVALIDARG);
        allocation_error(&e_ok);
        assert!(!upload_refused());

        let e_oom = windows::core::Error::from(windows::Win32::Foundation::E_OUTOFMEMORY);
        allocation_error(&e_oom);
        assert!(upload_refused());
        RETRY_UPLOAD_AT.store(0, Ordering::Relaxed);
        assert!(!upload_refused());
    }
}
