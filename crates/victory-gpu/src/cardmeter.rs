//! What the card's time goes to, by kind of list, in every run.
//!
//! The timing run (`VICTORY_GPU_TIMING`) names every title command but sends
//! each list on its own, which changes the frame, and it never timed the lists
//! the layer adds for itself: card shadow sync/hash (`record_before`) and
//! post-submission hash compare (`record_after`).
//!
//! Here each list gets one timestamp where it begins and one where it ends,
//! resolved into one persistent read-back buffer by the list itself. Slots
//! belong to a recording until reset/drop, not until the first execution:
//! a closed command list can be submitted again. Read only after the actual
//! submission fence completes. No extra GPU wait is introduced.
//!
//! On by default in diagnostics builds; `VICTORY_CARD_METER=0` or `1` says
//! otherwise. Two queries and a resolve per list.

use std::collections::{HashMap, VecDeque};
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
    free: VecDeque<u32>,
    recordings: HashMap<usize, Recording>,
}

/// A slot remains owned by the recording after a sample is collected.
struct Recording {
    pair: u32,
    kind: Kind,
    state: State,
}

enum State {
    Recorded,
    Running(ID3D12Fence),
    Submitted(ID3D12Fence, u64),
    Sampled,
}

impl Recording {
    fn sample(&mut self, read: impl FnOnce(u32) -> (u64, u64)) -> Option<(Kind, u64, u64)> {
        let State::Submitted(fence, value) = &self.state else { return None };
        // SAFETY: a live fence retained by this recording.
        let completed = unsafe { fence.GetCompletedValue() };
        if completed == u64::MAX {
            self.state = State::Sampled;
            return None;
        }
        if completed < *value {
            return None;
        }
        let (begin, end) = read(self.pair);
        self.state = State::Sampled;
        Some((self.kind, begin, end))
    }

    fn recyclable(&self) -> bool {
        matches!(self.state, State::Recorded | State::Sampled)
    }
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
                free: (0..PAIRS - HEADROOM).collect(),
                recordings: HashMap::new(),
            })
        }
    }

    fn release(&mut self, key: usize) {
        if let Some(mut recording) = self.recordings.remove(&key) {
            let mapped = self.mapped;
            if let Some((kind, begin, end)) = recording.sample(|pair| read_pair_raw(mapped, pair)) {
                account(kind, micros(begin, end, self.frequency));
            }
            // An unsignalled or incomplete submission quarantines the slot.
            // Losing diagnostics is safer than aliasing a GPU write.
            if recording.recyclable() {
                self.free.push_back(recording.pair);
            }
        }
    }

    fn alloc_slot(&mut self, key: usize, kind: Kind) -> Option<u32> {
        // Only called after reset of the previous recording, if any.
        self.release(key);
        let pair = self.free.pop_front()?;
        self.recordings.insert(key, Recording { pair, kind, state: State::Recorded });
        Some(pair)
    }

    fn begin_list(&mut self, list: &ID3D12GraphicsCommandList, kind: Kind) -> Span {
        let Some(key) = list_key(list) else { return Span::NONE };
        let Some(pair) = self.alloc_slot(key, kind) else { return Span::NONE };
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

fn list_key(list: &ID3D12GraphicsCommandList) -> Option<usize> {
    list.cast::<ID3D12CommandList>().ok().map(|l| l.as_raw() as usize)
}

/// Release after reset or destruction, never merely after execution.
pub(crate) fn release(list: &ID3D12GraphicsCommandList) {
    if !on() {
        return;
    }
    let Some(key) = list_key(list) else { return };
    let mut meter = METER.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(Some(m)) = meter.as_mut() {
        m.release(key);
    }
}

/// Mark before ExecuteCommandLists so collection cannot race a new write.
/// Multiple executions of one recording in a batch keep the last sample.
pub(crate) fn executing(lists: &[Option<ID3D12CommandList>], fence: &ID3D12Fence) {
    if !on() {
        return;
    }
    let mut meter = METER.lock().unwrap_or_else(|e| e.into_inner());
    let Some(Some(m)) = meter.as_mut() else { return };
    let mapped = m.mapped;
    let frequency = m.frequency;
    for list in lists.iter().flatten() {
        if let Some(r) = m.recordings.get_mut(&(list.as_raw() as usize)) {
            if let Some((kind, begin, end)) = r.sample(|pair| read_pair_raw(mapped, pair)) {
                account(kind, micros(begin, end, frequency));
            }
            r.state = State::Running(fence.clone());
        }
    }
}

/// Attach a successfully signalled fence to the recordings it covers.
pub(crate) fn signalled(fence: &ID3D12Fence, value: u64) {
    if !on() {
        return;
    }
    let mut meter = METER.lock().unwrap_or_else(|e| e.into_inner());
    let Some(Some(m)) = meter.as_mut() else { return };
    for r in m.recordings.values_mut() {
        if matches!(&r.state, State::Running(owner) if owner.as_raw() == fence.as_raw()) {
            r.state = State::Submitted(fence.clone(), value);
        }
    }
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

fn account(kind: Kind, us: u64) {
    use crate::framelog::{T, add};
    let (time, count) = match kind {
        Kind::Sync => (T::CardSyncUs, T::CardSyncLists),
        Kind::Check => (T::CardCheckUs, T::CardCheckLists),
        Kind::Lists => (T::CardListsUs, T::CardLists),
    };
    add(time, us);
    add(count, 1);
}

/// Collect completed samples without releasing their recorded slots.
pub(crate) fn collect() {
    if !on() {
        return;
    }
    let mut meter = METER.lock().unwrap_or_else(|e| e.into_inner());
    let Some(Some(m)) = meter.as_mut() else { return };
    let mapped = m.mapped;
    let frequency = m.frequency;
    for r in m.recordings.values_mut() {
        if let Some((kind, begin, end)) = r.sample(|pair| read_pair_raw(mapped, pair)) {
            account(kind, micros(begin, end, frequency));
        }
    }
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
    fn unsubmitted_recording_is_not_read() {
        let mut r = Recording { pair: 1, kind: Kind::Lists, state: State::Recorded };
        assert!(r.sample(|_| panic!("unsubmitted slot")).is_none());
        assert!(r.recyclable());
    }

    #[test]
    fn gpu_meter_records_and_resolves_real_timestamps() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
        use windows::Win32::Graphics::Dxgi::*;
        use windows::core::Interface;
        // The default adapter may also be WARP on a software-only runner.
        // Keep the guard until every test-owned D3D12 object is dropped.
        let _serial = crate::host::warp_serial();
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
                meter.alloc_slot(usize::MAX, Kind::Lists),
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

            let key = list_key(&list).unwrap();
            let r = meter.recordings.get_mut(&key).unwrap();
            r.state = State::Running(fence.clone());
            assert!(r.sample(|_| panic!("unsignalled slot")).is_none());
            assert!(!r.recyclable());
            r.state = State::Submitted(fence.clone(), 2);
            assert!(r.sample(|_| panic!("incomplete fence")).is_none());
            fence.Signal(2).expect("complete fence");
            assert_eq!(r.sample(|_| (10, 20)), Some((Kind::Lists, 10, 20)));
            assert!(r.sample(|_| panic!("already sampled")).is_none());
            // Even after collection, the slot belongs to the closed list.
            assert!(!meter.free.contains(&span.raw()));
            meter.release(key);
            assert!(meter.free.contains(&span.raw()));
            while meter.free.front().is_some() {
                let key = meter.recordings.len() + 10;
                assert!(meter.alloc_slot(key, Kind::Lists).is_some());
            }
            assert!(meter.alloc_slot(usize::MAX - 1, Kind::Lists).is_none());
        }
    }
}
