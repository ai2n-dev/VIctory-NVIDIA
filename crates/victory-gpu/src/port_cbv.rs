//! Opt-in zero-copy constant buffers (`VICTORY_PORT_CBV=1`, off by default).
//! Adapted from AvatarManz 3920dc26 (Renderer Port 5, GPL-3.0).
//!
//! When enabled, translated shaders read constant buffers as raw byte views
//! (table/heap CBVs as raw SRVs, root CBVs as root SRVs). Because raw views
//! require 4-byte (and console 16-byte) alignment rather than 256-byte CBV
//! alignment, in-place title constant buffers can be bound directly without
//! per-submission staging copies.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

pub(crate) fn on() -> bool {
    victory_shader::emit::constants_raw()
}

pub(crate) fn in_place_on() -> bool {
    static PLACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    on() && *PLACE
        .get_or_init(|| std::env::var_os("VICTORY_PORT_CBV_PLACE").is_none_or(|v| v != "0"))
}

static STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

pub(crate) fn in_place(size: u32) {
    STATS[0].fetch_add(1, Relaxed);
    STATS[1].fetch_add(u64::from(size), Relaxed);
}

pub(crate) fn copied(aligned: bool) {
    STATS[if aligned { 2 } else { 3 }].fetch_add(1, Relaxed);
}

pub(crate) fn raw_ok(gpu: u64) -> bool {
    gpu.is_multiple_of(16)
}

pub(crate) fn log(frames: f64) {
    if !on() {
        return;
    }
    let s = STATS
        .each_ref()
        .map(|a| a.swap(0, Relaxed) as f64 / frames.max(1e-9));
    tracing::info!(
        "port 5 (VICTORY_PORT_CBV=1) per frame: {:.0} table CBVs in place ({:.1} KB not copied), {:.0} copied not in place, {:.0} copied unaligned",
        s[0],
        s[1] / 1024.0,
        s[2],
        s[3],
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_views_need_sixteen_bytes() {
        assert!(raw_ok(0x3500_0010));
        assert!(raw_ok(0x3500_0100));
        assert!(!raw_ok(0x3500_0008));
        assert!(!raw_ok(0x3500_0004));
    }
}
