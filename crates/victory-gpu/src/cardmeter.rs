//! What the card's time goes to, by kind of list, in every run.
//!
//! The timing run (`VICTORY_GPU_TIMING`) names every title command but sends
//! each list on its own, which changes the frame, and it never timed the lists
//! the layer adds for itself: card shadow sync/hash (`record_before`) and
//! post-submission hash compare (`record_after`).
//!
//! Here each list gets one timestamp where it begins and one where it ends,
//! resolved into one persistent read-back buffer by the list itself. A pair is
//! read a second and a half after it was recorded, when every flight of that
//! time has long landed, and its length goes to the frame log's totals by
//! kind. So the frame log's card columns reflect the frames of a moment ago:
//! accurate for the usual frame, rather than placing one long frame.
//!
//! On by default in diagnostics builds; `VICTORY_CARD_METER=0` or `1` says
//! otherwise. Two queries and a resolve per list.

use std::sync::Mutex;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

/// What a timed list is.
#[derive(Clone, Copy)]
pub(crate) enum Kind {
    /// Ours: dirty staging ranges uploaded to card VRAM and hashed before execution.
    Sync,
    /// Ours: post-submission GPU hash compare of shader-written card ranges.
    Check,
    /// Every other host command list: the title's own and our present copy lists.
    Lists,
}

/// Pairs of timestamps in the ring.
const PAIRS: u32 = 8192;
/// A pair is read this long after it was recorded.
const AFTER: std::time::Duration = std::time::Duration::from_millis(1500);

/// A pair being recorded: its place in the ring, or none.
#[derive(Clone, Copy)]
pub(crate) struct Span(u32);

impl Span {
    pub(crate) const NONE: Span = Span(u32::MAX);

    /// For keeping in an atomic.
    pub(crate) fn raw(self) -> u32 {
        self.0
    }

    pub(crate) fn from_raw(raw: u32) -> Span {
        Span(raw)
    }
}

struct Meter {
    heap: ID3D12QueryHeap,
    back: ID3D12Resource,
    /// The read-back buffer's bytes, mapped permanently.
    mapped: usize,
    /// Ticks per second of the queue's timestamp clock.
    frequency: u64,
    next: u32,
    /// Pairs recorded and not read yet: slot, kind, when recorded.
    pending: std::collections::VecDeque<(u32, Kind, std::time::Instant)>,
    /// What each slot held when last read: an unwritten slot retains its old
    /// pair and is not counted twice.
    seen: Vec<(u64, u64)>,
}

// SAFETY: D3D12 query heap/resource are free-threaded and `mapped` is only read under `METER`.
unsafe impl Send for Meter {}

static METER: Mutex<Option<Option<Meter>>> = Mutex::new(None);

/// Whether lists are timed.
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var_os("VICTORY_CARD_METER") {
        Some(v) => v != "0",
        None => cfg!(feature = "diagnostics"),
    })
}

fn make() -> Option<Meter> {
    let host = crate::host::host()?;
    let device = host.device();
    let mut heap: Option<ID3D12QueryHeap> = None;
    let mut back: Option<ID3D12Resource> = None;
    let readback = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: u64::from(PAIRS) * 16,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    // SAFETY: plain D3D12 resource/heap creation and permanent readback mapping.
    unsafe {
        device
            .CreateQueryHeap(
                &D3D12_QUERY_HEAP_DESC {
                    Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                    Count: PAIRS * 2,
                    NodeMask: 0,
                },
                &mut heap,
            )
            .ok()?;
        device
            .CreateCommittedResource(
                &readback,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut back,
            )
            .ok()?;
        let (heap, back) = (heap?, back?);
        let mut mapped = std::ptr::null_mut();
        back.Map(0, None, Some(&mut mapped)).ok()?;
        let frequency = host.timestamp_frequency();
        if mapped.is_null() || frequency == 0 {
            return None;
        }
        tracing::info!(
            "the card's time is measured by kind of list (VICTORY_CARD_METER=0: disabled)"
        );
        Some(Meter {
            heap,
            back,
            mapped: mapped as usize,
            frequency,
            next: 0,
            pending: std::collections::VecDeque::new(),
            seen: vec![(0, 0); PAIRS as usize],
        })
    }
}

/// Begins timing a list of `kind`: records the opening timestamp query.
pub(crate) fn begin(list: &ID3D12GraphicsCommandList, kind: Kind) -> Span {
    if !on() {
        return Span::NONE;
    }
    let mut meter = METER.lock().unwrap_or_else(|e| e.into_inner());
    let Some(m) = meter.get_or_insert_with(make) else {
        return Span::NONE;
    };
    if m.pending.len() as u32 >= PAIRS - 64 {
        return Span::NONE;
    }
    let pair = m.next;
    m.next = (m.next + 1) % PAIRS;
    m.pending.push_back((pair, kind, std::time::Instant::now()));
    // SAFETY: an open list and our live query heap.
    unsafe { list.EndQuery(&m.heap, D3D12_QUERY_TYPE_TIMESTAMP, pair * 2) };
    Span(pair)
}

/// Ends timing a list: records the closing timestamp and resolves both into the read-back ring.
pub(crate) fn end(list: &ID3D12GraphicsCommandList, span: Span) {
    if span.0 == u32::MAX {
        return;
    }
    let meter = METER.lock().unwrap_or_else(|e| e.into_inner());
    let Some(Some(m)) = meter.as_ref() else {
        return;
    };
    // SAFETY: an open list, our live query heap, and our 16-byte slice of the read-back buffer.
    unsafe {
        list.EndQuery(&m.heap, D3D12_QUERY_TYPE_TIMESTAMP, span.0 * 2 + 1);
        list.ResolveQueryData(
            &m.heap,
            D3D12_QUERY_TYPE_TIMESTAMP,
            span.0 * 2,
            2,
            &m.back,
            u64::from(span.0) * 16,
        );
    }
}

/// Microseconds between two timestamps of a clock of `frequency` ticks per second.
fn micros(begin: u64, end: u64, frequency: u64) -> u64 {
    if end <= begin || frequency == 0 {
        return 0;
    }
    ((end - begin) as u128 * 1_000_000 / frequency as u128).min(10_000_000) as u64
}

/// Collects resolved pairs older than [`AFTER`] into the frame log's totals.
pub(crate) fn collect() {
    if !on() {
        return;
    }
    let mut meter = METER.lock().unwrap_or_else(|e| e.into_inner());
    let Some(Some(m)) = meter.as_mut() else {
        return;
    };
    let now = std::time::Instant::now();
    while let Some(&(pair, kind, at)) = m.pending.front() {
        if now.duration_since(at) < AFTER {
            break;
        }
        m.pending.pop_front();
        // SAFETY: `pair` indexes within `m.back`'s `PAIRS * 16` mapped bytes.
        let (begin, end) = unsafe {
            let p = (m.mapped as *const u64).add(pair as usize * 2);
            (p.read_volatile(), p.add(1).read_volatile())
        };
        if std::mem::replace(&mut m.seen[pair as usize], (begin, end)) == (begin, end) {
            continue;
        }
        let us = micros(begin, end, m.frequency);
        use crate::framelog::{T, add};
        match kind {
            Kind::Sync => {
                add(T::CardSyncUs, us);
                add(T::CardSyncLists, 1);
            }
            Kind::Check => {
                add(T::CardCheckUs, us);
                add(T::CardCheckLists, 1);
            }
            Kind::Lists => {
                add(T::CardListsUs, us);
                add(T::CardLists, 1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pairs_length_is_in_microseconds_of_the_queues_clock() {
        assert_eq!(micros(1_000, 11_000, 10_000_000), 1_000);
        assert_eq!(micros(0, 25_000_000, 25_000_000), 1_000_000);
        assert_eq!(micros(5, 5, 1_000), 0);
        assert_eq!(micros(9, 5, 1_000), 0);
        assert_eq!(micros(1, 2, 0), 0);
        assert_eq!(micros(1, u64::MAX, 1), 10_000_000);
    }
}
