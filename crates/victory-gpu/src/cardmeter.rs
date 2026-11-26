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

use std::collections::VecDeque;
use std::sync::Mutex;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::core::Interface;

/// What a timed list is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
/// Minimum headroom kept free in the ring before dropping new pairs.
const HEADROOM: u32 = 64;
/// A pair is read this long after it was recorded.
const AFTER: std::time::Duration = std::time::Duration::from_millis(1500);

/// A pair being recorded: its place in the ring, or none.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    device: usize,
    heap: ID3D12QueryHeap,
    back: ID3D12Resource,
    /// The read-back buffer's bytes, mapped permanently.
    mapped: usize,
    /// Ticks per second of the queue's timestamp clock.
    frequency: u64,
    next: u32,
    /// Pairs recorded and not read yet: slot, kind, when recorded.
    pending: VecDeque<(u32, Kind, std::time::Instant)>,
    /// What each slot held when last read: an unwritten slot retains its old
    /// pair and is not counted twice.
    seen: Vec<(u64, u64)>,
}

// SAFETY: D3D12 query heap/resource are free-threaded and `mapped` is only read under `METER`.
unsafe impl Send for Meter {}

static METER: Mutex<Option<Option<Meter>>> = Mutex::new(None);

fn parse_on(value: Option<&std::ffi::OsStr>, diagnostics: bool) -> bool {
    match value {
        Some(v) => v != "0",
        None => diagnostics,
    }
}

/// Whether lists are timed.
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        parse_on(
            std::env::var_os("VICTORY_CARD_METER").as_deref(),
            cfg!(feature = "diagnostics"),
        )
    })
}

fn make() -> Option<Meter> {
    let host = crate::host::host()?;
    Meter::make_on(host.device(), host.timestamp_frequency())
}

impl Meter {
    fn make_on(device: &ID3D12Device, frequency: u64) -> Option<Meter> {
        if frequency == 0 {
            return None;
        }
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
            if mapped.is_null() {
                return None;
            }
            tracing::info!(
                "the card's time is measured by kind of list (VICTORY_CARD_METER=0: disabled)"
            );
            Some(Meter {
                device: device.as_raw() as usize,
                heap,
                back,
                mapped: mapped as usize,
                frequency,
                next: 0,
                pending: VecDeque::new(),
                seen: vec![(0, 0); PAIRS as usize],
            })
        }
    }

    fn alloc_slot(&mut self, kind: Kind, at: std::time::Instant) -> Option<u32> {
        if self.pending.len() as u32 >= PAIRS - HEADROOM {
            return None;
        }
        let pair = self.next;
        self.next = (self.next + 1) % PAIRS;
        self.pending.push_back((pair, kind, at));
        Some(pair)
    }

    fn begin_list(&mut self, list: &ID3D12GraphicsCommandList, kind: Kind) -> Span {
        let Some(pair) = self.alloc_slot(kind, std::time::Instant::now()) else {
            return Span::NONE;
        };
        // SAFETY: an open list and our live query heap.
        unsafe { list.EndQuery(&self.heap, D3D12_QUERY_TYPE_TIMESTAMP, pair * 2) };
        Span(pair)
    }

    fn end_list(&self, list: &ID3D12GraphicsCommandList, span: Span) {
        if span.0 == u32::MAX {
            return;
        }
        // SAFETY: an open list, our live query heap, and our 16-byte slice of the read-back buffer.
        unsafe {
            list.EndQuery(&self.heap, D3D12_QUERY_TYPE_TIMESTAMP, span.0 * 2 + 1);
            list.ResolveQueryData(
                &self.heap,
                D3D12_QUERY_TYPE_TIMESTAMP,
                span.0 * 2,
                2,
                &self.back,
                u64::from(span.0) * 16,
            );
        }
    }

    #[cfg(test)]
    fn read_pair(&self, pair: u32) -> (u64, u64) {
        read_pair_raw(self.mapped, pair)
    }
}

fn read_pair_raw(mapped: usize, pair: u32) -> (u64, u64) {
    // SAFETY: `pair` indexes within `PAIRS * 16` permanently mapped bytes.
    unsafe {
        let p = (mapped as *const u64).add(pair as usize * 2);
        (p.read_volatile(), p.add(1).read_volatile())
    }
}

fn list_device(list: &ID3D12GraphicsCommandList) -> Option<usize> {
    let mut dev: Option<ID3D12Device> = None;
    // SAFETY: `list` is a live D3D12 command list.
    unsafe { list.GetDevice(&mut dev).ok()? };
    Some(dev?.as_raw() as usize)
}

/// Begins timing a list of `kind`: records the opening timestamp query.
pub(crate) fn begin(list: &ID3D12GraphicsCommandList, kind: Kind) -> Span {
    if !on() {
        return Span::NONE;
    }
    let Some(dev) = list_device(list) else {
        return Span::NONE;
    };
    let mut meter = METER.lock().unwrap_or_else(|e| e.into_inner());
    let Some(m) = meter.get_or_insert_with(make) else {
        return Span::NONE;
    };
    if m.device != dev {
        return Span::NONE;
    }
    m.begin_list(list, kind)
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
    m.end_list(list, span);
}

/// Microseconds between two timestamps of a clock of `frequency` ticks per second.
fn micros(begin: u64, end: u64, frequency: u64) -> u64 {
    if end <= begin || frequency == 0 {
        return 0;
    }
    ((end - begin) as u128 * 1_000_000 / frequency as u128).min(10_000_000) as u64
}

fn collect_ready(
    pending: &mut VecDeque<(u32, Kind, std::time::Instant)>,
    seen: &mut [(u64, u64)],
    frequency: u64,
    now: std::time::Instant,
    mut read: impl FnMut(u32) -> (u64, u64),
) {
    while let Some(&(pair, kind, at)) = pending.front() {
        if now.duration_since(at) < AFTER {
            break;
        }
        pending.pop_front();
        let (begin, end) = read(pair);
        if std::mem::replace(&mut seen[pair as usize], (begin, end)) == (begin, end) {
            continue;
        }
        let us = micros(begin, end, frequency);
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

/// Collects resolved pairs older than [`AFTER`] into the frame log's totals.
pub(crate) fn collect() {
    if !on() {
        return;
    }
    let mut meter = METER.lock().unwrap_or_else(|e| e.into_inner());
    let Some(Some(m)) = meter.as_mut() else {
        return;
    };
    let (pending, seen, frequency, mapped) =
        (&mut m.pending, &mut m.seen, m.frequency, m.mapped);
    collect_ready(pending, seen, frequency, std::time::Instant::now(), |pair| {
        read_pair_raw(mapped, pair)
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;

    #[test]
    fn span_round_trips_raw_slot() {
        assert_eq!(Span::NONE.raw(), u32::MAX);
        assert_eq!(Span::from_raw(0).raw(), 0);
        assert_eq!(Span::from_raw(42).raw(), 42);
    }

    #[test]
    fn card_meter_switch_honors_env_and_feature_default() {
        assert!(!parse_on(None, false));
        assert!(parse_on(None, true));
        assert!(!parse_on(Some(OsStr::new("0")), true));
        assert!(parse_on(Some(OsStr::new("1")), false));
    }

    #[test]
    fn a_pairs_length_is_in_microseconds_of_the_queues_clock() {
        assert_eq!(micros(1_000, 11_000, 10_000_000), 1_000);
        assert_eq!(micros(0, 25_000_000, 25_000_000), 1_000_000);
        assert_eq!(micros(5, 5, 1_000), 0);
        assert_eq!(micros(9, 5, 1_000), 0);
        assert_eq!(micros(1, 2, 0), 0);
        assert_eq!(micros(1, u64::MAX, 1), 10_000_000);
    }

    #[test]
    fn collect_ready_respects_delay_dedups_unchanged_and_routes_kinds() {
        use crate::framelog::{T, total};
        let t0 = std::time::Instant::now();
        let mut pending = VecDeque::new();
        let mut seen = vec![(0, 0); 8];
        pending.push_back((0, Kind::Sync, t0));
        pending.push_back((1, Kind::Check, t0));
        pending.push_back((2, Kind::Lists, t0 + std::time::Duration::from_millis(500)));

        let b_sync = (total(T::CardSyncUs), total(T::CardSyncLists));
        let b_check = (total(T::CardCheckUs), total(T::CardCheckLists));
        let b_lists = (total(T::CardListsUs), total(T::CardLists));

        // Just before AFTER: nothing collected yet.
        collect_ready(
            &mut pending,
            &mut seen,
            1_000_000,
            t0 + AFTER - std::time::Duration::from_millis(1),
            |_| (100, 200),
        );
        assert_eq!(pending.len(), 3);

        // At AFTER for pairs 0 and 1 (pair 2 not ready yet).
        collect_ready(&mut pending, &mut seen, 1_000_000, t0 + AFTER, |pair| {
            match pair {
                0 => (10, 60),
                1 => (20, 90),
                _ => unreachable!("pair 2 is still immature"),
            }
        });
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].0, 2);
        assert_eq!(
            (
                total(T::CardSyncUs) - b_sync.0,
                total(T::CardSyncLists) - b_sync.1
            ),
            (50, 1)
        );
        assert_eq!(
            (
                total(T::CardCheckUs) - b_check.0,
                total(T::CardCheckLists) - b_check.1
            ),
            (70, 1)
        );
        assert_eq!(total(T::CardLists) - b_lists.1, 0);

        // Duplicate timestamps at slot 0 must not double-count; slot 2 matures at +500ms.
        pending.push_back((0, Kind::Sync, t0));
        collect_ready(
            &mut pending,
            &mut seen,
            1_000_000,
            t0 + AFTER + std::time::Duration::from_millis(500),
            |pair| match pair {
                2 => (5, 30),
                0 => (10, 60),
                _ => unreachable!(),
            },
        );
        assert!(pending.is_empty());
        assert_eq!(
            (
                total(T::CardListsUs) - b_lists.0,
                total(T::CardLists) - b_lists.1
            ),
            (25, 1)
        );
        // Unchanged slot 0 pair did not add another Sync entry.
        assert_eq!(total(T::CardSyncLists) - b_sync.1, 1);
    }

    #[test]
    fn gpu_meter_records_and_resolves_real_timestamps() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
        use windows::Win32::Graphics::Dxgi::*;
        use windows::core::Interface;
        let mut device: Option<ID3D12Device> = None;
        // SAFETY: probe hardware adapter first, then WARP fallback.
        unsafe {
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device).is_err() {
                let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("dxgi factory");
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().expect("warp adapter");
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_12_0, &mut device)
                    .expect("warp d3d12 device");
            }
        }
        let device = device.expect("d3d12 device");
        assert!(Meter::make_on(&device, 0).is_none());
        // SAFETY: valid device/queue/allocator/list creation.
        unsafe {
            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .expect("queue");
            let freq = queue.GetTimestampFrequency().expect("timestamp freq");
            let mut meter = Meter::make_on(&device, freq).expect("timestamp meter");
            let alloc: ID3D12CommandAllocator = device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .expect("alloc");
            let list: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
                .expect("list");
            // Advance past slot 0 so `span` is slot 1 (non-zero query/byte offsets).
            assert_eq!(
                meter.alloc_slot(Kind::Lists, std::time::Instant::now()),
                Some(0)
            );
            let span = meter.begin_list(&list, Kind::Lists);
            assert_eq!(span.raw(), 1);
            meter.end_list(&list, span);
            meter.end_list(&list, Span::NONE);
            list.Close().expect("close");
            let cmd: ID3D12CommandList = list.cast().unwrap();
            queue.ExecuteCommandLists(&[Some(cmd)]);
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            queue.Signal(&fence, 1).expect("signal");
            while fence.GetCompletedValue() < 1 {
                std::thread::yield_now();
            }
            let (b, e) = meter.read_pair(1);
            assert!(b > 0 && e >= b, "expected resolved timestamps at slot 1: {b}..{e}");

            // Ring wraps at PAIRS -> 0 and stops when PAIRS - HEADROOM slots are pending.
            meter.pending.clear();
            meter.next = PAIRS - 1;
            assert_eq!(
                meter.alloc_slot(Kind::Lists, std::time::Instant::now()),
                Some(PAIRS - 1)
            );
            assert_eq!(
                meter.alloc_slot(Kind::Lists, std::time::Instant::now()),
                Some(0)
            );
            while meter.pending.len() < (PAIRS - HEADROOM) as usize {
                assert!(meter.alloc_slot(Kind::Lists, std::time::Instant::now()).is_some());
            }
            assert!(meter.alloc_slot(Kind::Lists, std::time::Instant::now()).is_none());
        }
    }
}
