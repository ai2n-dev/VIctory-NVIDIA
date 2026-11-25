//! The graphics driver's system errors, logged as Windows records them.
//!
//! Device losses are often preceded seconds earlier by display-driver entries
//! in the Windows `System` event log (`amdkmdag` / `amdwddmg` / `nvlddmkm` /
//! `igfx` / `Display`), giving the host log a timestamped warning correlated
//! with what the layer was doing at that moment.
//!
//! On by default; `VICTORY_DRIVER_WATCH=0` disables the subscription.

use std::ffi::OsStr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};

use windows_sys::Win32::System::EventLog::{
    EVT_HANDLE, EVT_SUBSCRIBE_NOTIFY_ACTION, EvtSubscribe, EvtSubscribeActionDeliver,
    EvtSubscribeToFutureEvents,
};

/// Display-driver event-log entries seen since the start.
static ERRORS: AtomicU64 = AtomicU64::new(0);
/// Whether at least one driver error was seen in this run.
static TRIPPED: AtomicBool = AtomicBool::new(false);

/// Total display-driver errors observed in this run.
pub(crate) fn errors() -> u64 {
    ERRORS.load(Relaxed)
}

fn should_watch(value: Option<&OsStr>) -> bool {
    value.is_none_or(|v| v != "0")
}

fn record_seen(errors: &AtomicU64, tripped: &AtomicBool) -> (u64, bool) {
    let n = errors.fetch_add(1, Relaxed) + 1;
    let first = !tripped.swap(true, Relaxed);
    (n, first)
}

fn seen() {
    let (n, first) = record_seen(&ERRORS, &TRIPPED);
    tracing::error!(
        n,
        first,
        "the graphics driver reported an error to the system event log"
    );
}

unsafe extern "system" fn on_event(
    action: EVT_SUBSCRIBE_NOTIFY_ACTION,
    _context: *const core::ffi::c_void,
    _event: EVT_HANDLE,
) -> u32 {
    if action == EvtSubscribeActionDeliver {
        seen();
    }
    0
}

/// Subscribes to future display-driver events in the Windows `System` log, once.
pub(crate) fn start() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if !should_watch(std::env::var_os("VICTORY_DRIVER_WATCH").as_deref()) {
            return;
        }
        let wide = |s: &str| -> Vec<u16> { s.encode_utf16().chain(std::iter::once(0)).collect() };
        let channel = wide("System");
        let query = wide(
            "*[System[Provider[@Name='nvlddmkm' or @Name='amdkmdag' or @Name='amdwddmg' or @Name='igfx' or @Name='igfxn' \
             or @Name='Display']]]",
        );
        // SAFETY: NUL-terminated UTF-16 strings; `on_event` only updates lock-free atomics and logs.
        let subscription = unsafe {
            EvtSubscribe(
                0,
                std::ptr::null_mut(),
                channel.as_ptr(),
                query.as_ptr(),
                0,
                std::ptr::null(),
                Some(on_event),
                EvtSubscribeToFutureEvents,
            )
        };
        if subscription == 0 {
            tracing::warn!(
                error = %std::io::Error::last_os_error(),
                "cannot subscribe to System event log for graphics driver errors"
            );
        } else {
            tracing::info!(
                "watching System event log for graphics driver errors (VICTORY_DRIVER_WATCH=0: disabled)"
            );
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_error_trips_the_switch_and_every_one_is_counted() {
        let count = AtomicU64::new(0);
        let flag = AtomicBool::new(false);
        assert_eq!(record_seen(&count, &flag), (1, true));
        assert_eq!(record_seen(&count, &flag), (2, false));

        // Verify global `seen()` / `on_event` integration too.
        let before = errors();
        // Non-deliver action is ignored.
        // SAFETY: pure callback exercising action filtering.
        let r0 = unsafe { on_event(0, std::ptr::null(), 0) };
        assert_eq!(r0, 0);
        assert_eq!(errors(), before);
        // Deliver action increments global counter and sets `TRIPPED`.
        // SAFETY: pure callback with `EvtSubscribeActionDeliver`.
        let r1 = unsafe { on_event(EvtSubscribeActionDeliver, std::ptr::null(), 0) };
        assert_eq!(r1, 0);
        assert_eq!(errors(), before + 1);
        assert!(TRIPPED.load(Relaxed));
        seen();
        assert_eq!(errors(), before + 2);
    }

    #[test]
    fn watch_switch_honors_env_override() {
        assert!(should_watch(None));
        assert!(should_watch(Some(OsStr::new("1"))));
        assert!(!should_watch(Some(OsStr::new("0"))));
    }

    #[test]
    fn the_system_takes_the_subscription() {
        let wide = |s: &str| -> Vec<u16> { s.encode_utf16().chain(std::iter::once(0)).collect() };
        let (channel, query) = (
            wide("System"),
            wide("*[System[Provider[@Name='amdkmdag' or @Name='nvlddmkm' or @Name='Display']]]"),
        );
        // SAFETY: valid NUL-terminated strings; subscription closed immediately.
        let subscription = unsafe {
            EvtSubscribe(
                0,
                std::ptr::null_mut(),
                channel.as_ptr(),
                query.as_ptr(),
                0,
                std::ptr::null(),
                Some(on_event),
                EvtSubscribeToFutureEvents,
            )
        };
        assert_ne!(subscription, 0, "{}", std::io::Error::last_os_error());
        // SAFETY: valid subscription handle returned above.
        unsafe {
            windows_sys::Win32::System::EventLog::EvtClose(subscription);
        }
    }
}
