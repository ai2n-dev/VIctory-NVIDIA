//! Host-memory pressure for reusable CPU and upload storage.
//!
//! Hysteresis adapted from codec_d's cool_fox, f4b48ad (GPL-3.0).
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

static PRESSURED: AtomicBool = AtomicBool::new(false);

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
    let mut status = MEMORYSTATUSEX {
        dwLength: size_of::<MEMORYSTATUSEX>() as u32,
        ..Default::default()
    };
    // SAFETY: the API fills a writable record sized by dwLength.
    if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 {
        return;
    }
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
    const GIB: u64 = 1 << 30;
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
        const GIB: u64 = 1 << 30;
        assert!(next_pressure(false, 3 * GIB, 8 * GIB));
        assert!(next_pressure(false, 8 * GIB, 4 * GIB));
        assert!(!next_pressure(false, 4 * GIB, 5 * GIB));
        assert!(next_pressure(true, 4 * GIB, 8 * GIB));
        assert!(next_pressure(true, 8 * GIB, 5 * GIB));
        assert!(!next_pressure(true, 5 * GIB, 6 * GIB));
    }
}
