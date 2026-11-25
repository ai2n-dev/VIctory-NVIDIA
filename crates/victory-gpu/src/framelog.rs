//! Per-frame duration breakdown and long-frame diagnostics.
//!
//! Monotonic running counters avoid per-second resets so individual frame spikes
//! can be compared column-by-column against the usual frame mean.
//!
//! On by default in `diagnostics` builds; set `VICTORY_FRAME_LOG=1` or `0` to
//! override. Unconfigured builds still pay only plain relaxed atomic adds.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Monotonic totals since start (microseconds unless stated otherwise).
#[derive(Clone, Copy)]
#[repr(usize)]
pub enum T {
    /// Total time in `queue::present`, and GPU present-copy time in `Host::present`.
    PresentUs,
    PresentShowUs,
    /// Total time in `queue::execute` across all threads.
    ExecuteUs,
    /// Submissions sent; CPU prepare time and GPU wait time.
    Submissions,
    PrepareUs,
    CardWaitUs,
    /// Title-memory copies for submissions (us, bytes) and shader copy-back (us, bytes).
    CopyUs,
    CopyBytes,
    CopyBackUs,
    CopyBackBytes,
    /// Virtual-memory remapping invalidations (`title_memory_remapped`).
    Remaps,
    RemapUs,
    /// Texture staging refreshes and uploads recorded.
    RefreshUs,
    Uploads,
    /// Time waiting on host pipeline creation workers.
    PipelineWaitUs,
    /// Write faults caught by view-memory guards.
    GuardFaults,
    /// Frame-pacing sleep duration across threads.
    PaceUs,
    /// DirectStorage completed requests, bytes, and worker time.
    DiskReads,
    DiskBytes,
    DiskUs,
    /// GPU timestamp intervals collected from [`crate::cardmeter`] (delayed ~1.5 s).
    CardSyncUs,
    CardSyncLists,
    CardCheckUs,
    CardCheckLists,
    CardListsUs,
    CardLists,
    /// Core-efficiency checks on the presenting thread (`GetSystemCpuSetInformation`).
    RenderLooks,
    RenderSlow,
    Count,
}

const N: usize = T::Count as usize;
static TOTAL: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];

/// Adds `n` to counter `t`.
pub fn add(t: T, n: u64) {
    TOTAL[t as usize].fetch_add(n, Relaxed);
}

#[cfg(test)]
pub(crate) fn total(t: T) -> u64 {
    TOTAL[t as usize].load(Relaxed)
}

/// Thread-local counters accumulated on the presenting thread between presents.
#[derive(Clone, Copy)]
#[repr(usize)]
pub enum H {
    /// `queue::execute` microseconds on this thread.
    ExecuteUs,
    /// Frame-wait sleep microseconds on this thread.
    PaceUs,
    Count,
}

const HN: usize = H::Count as usize;

thread_local! {
    static HERE: std::cell::Cell<[u64; HN]> = const { std::cell::Cell::new([0; HN]) };
}

/// Adds `n` to the calling thread's counter `h`.
pub fn here(h: H, n: u64) {
    HERE.with(|c| {
        let mut v = c.get();
        v[h as usize] += n;
        c.set(v);
    });
}

/// Returns whether the calling thread currently runs on a lower-efficiency logical core
/// (always `false` on symmetric CPUs).
pub fn on_slow_core() -> bool {
    use windows_sys::Win32::System::Threading::GetCurrentProcessorNumber;
    static SLOW: std::sync::OnceLock<Vec<bool>> = std::sync::OnceLock::new();
    let slow = SLOW.get_or_init(|| {
        let sets = cpu_sets();
        let best = sets.iter().map(|c| c.2).max().unwrap_or(0);
        let mut slow = vec![false; sets.iter().map(|c| c.1 as usize + 1).max().unwrap_or(0)];
        for (_, processor, class) in &sets {
            slow[*processor as usize] = *class < best;
        }
        slow
    });
    // SAFETY: plain processor-index query with no side effects.
    let processor = unsafe { GetCurrentProcessorNumber() } as usize;
    slow.get(processor).copied().unwrap_or(false)
}

fn cpu_sets() -> Vec<(u32, u8, u8)> {
    use windows_sys::Win32::System::SystemInformation::{
        GetSystemCpuSetInformation, SYSTEM_CPU_SET_INFORMATION,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut bytes = 0u32;
    // SAFETY: size query on our current process.
    unsafe { GetSystemCpuSetInformation(std::ptr::null_mut(), 0, &mut bytes, GetCurrentProcess(), 0) };
    let mut buffer = vec![0u64; (bytes as usize).div_ceil(8).max(1)];
    // SAFETY: `buffer` holds `bytes` aligned bytes.
    let ok = unsafe {
        GetSystemCpuSetInformation(
            buffer.as_mut_ptr().cast(),
            bytes,
            &mut bytes,
            GetCurrentProcess(),
            0,
        )
    };
    let mut sets = Vec::new();
    let mut at = 0usize;
    while ok != 0 && at + size_of::<SYSTEM_CPU_SET_INFORMATION>() <= bytes as usize {
        // SAFETY: record lies within `[0..bytes]` and reports its own `Size`.
        let info = unsafe {
            &*(buffer
                .as_ptr()
                .cast::<u8>()
                .add(at)
                .cast::<SYSTEM_CPU_SET_INFORMATION>())
        };
        if info.Size == 0 {
            break;
        }
        // SAFETY: `SYSTEM_CPU_SET_INFORMATION_TYPE::CpuSetInformation` is the sole union member.
        let set = unsafe { &info.Anonymous.CpuSet };
        if set.Group == 0 {
            sets.push((set.Id, set.LogicalProcessorIndex, set.EfficiencyClass));
        }
        at += info.Size as usize;
    }
    sets
}

fn parse_on(value: Option<&std::ffi::OsStr>, diagnostics: bool) -> bool {
    match value {
        Some(v) => v != "0",
        None => diagnostics,
    }
}

/// Whether per-frame breakdown logging is active.
pub fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        parse_on(
            std::env::var_os("VICTORY_FRAME_LOG").as_deref(),
            cfg!(feature = "diagnostics"),
        )
    })
}

fn micros(t: windows_sys::Win32::Foundation::FILETIME) -> u64 {
    // Disjoint 32-bit halves: `+` matches `|` and avoids equivalent `|` -> `^` mutants.
    ((u64::from(t.dwHighDateTime) << 32) + u64::from(t.dwLowDateTime)) / 10
}

fn thread_times() -> (u64, u64) {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentThread, GetThreadTimes};
    let zero = FILETIME {
        dwLowDateTime: 0,
        dwHighDateTime: 0,
    };
    let (mut created, mut exited, mut kernel, mut user) = (zero, zero, zero, zero);
    // SAFETY: current thread handle and valid local output `FILETIME` structs.
    unsafe { GetThreadTimes(GetCurrentThread(), &mut created, &mut exited, &mut kernel, &mut user) };
    (micros(user), micros(kernel))
}

fn process_times() -> (u64, u64) {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};
    let zero = FILETIME {
        dwLowDateTime: 0,
        dwHighDateTime: 0,
    };
    let (mut created, mut exited, mut kernel, mut user) = (zero, zero, zero, zero);
    // SAFETY: current process handle and valid local output `FILETIME` structs.
    unsafe {
        GetProcessTimes(
            GetCurrentProcess(),
            &mut created,
            &mut exited,
            &mut kernel,
            &mut user,
        )
    };
    (micros(user), micros(kernel))
}

fn page_faults() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    // SAFETY: plain POD struct sized by `cb`.
    let mut c: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
    c.cb = size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    // SAFETY: current process handle and properly sized output struct.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    u64::from(c.PageFaultCount)
}

fn free_mb() -> u64 {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: plain POD struct sized by `dwLength`.
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: valid sized `MEMORYSTATUSEX`.
    unsafe { GlobalMemoryStatusEx(&mut m) };
    m.ullAvailPhys >> 20
}

const S: usize = 10;
const S_THREAD_USER: usize = 0;
const S_THREAD_KERNEL: usize = 1;
const S_PROCESS_USER: usize = 2;
const S_PROCESS_KERNEL: usize = 3;
const S_PAGE_FAULTS: usize = 4;
const S_DRAWS: usize = 5;
const S_DISPATCHES: usize = 6;
const S_PIPELINES_MADE: usize = 7;
const S_PIPELINES_LOADED: usize = 8;
const S_FREE_MB: usize = 9;

fn sample() -> [u64; S] {
    let (tu, tk) = thread_times();
    let (pu, pk) = process_times();
    let (made, loaded) = crate::pipeline::frame_log_counts();
    [
        tu,
        tk,
        pu,
        pk,
        page_faults(),
        crate::draw::drawn(),
        crate::draw::dispatched(),
        made,
        loaded,
        free_mb(),
    ]
}

#[derive(Clone)]
struct Frame {
    number: u64,
    at: f64,
    us: f64,
    t: [f64; N],
    s: [f64; S],
    h: [f64; HN],
}

struct State {
    first: std::time::Instant,
    last: Option<(std::time::Instant, [u64; N], [u64; S], u32)>,
    number: u64,
    frames: Vec<Frame>,
    since: std::time::Instant,
    recent: std::collections::VecDeque<f64>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

fn median(values: impl Iterator<Item = f64>) -> f64 {
    let mut v: Vec<f64> = values.collect();
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn text(f: &Frame) -> String {
    let t = |x: T| f.t[x as usize];
    let ms = |x: T| f.t[x as usize] / 1e3;
    let s = |i: usize| f.s[i];
    let h = |x: H| f.h[x as usize];
    format!(
        "{:.0} ms | render thread: ran {:.0} ms (user {:.0}, kernel {:.0}); present {:.1} ms (show {:.1}), \
         execute {:.1} ms, paced {:.1} ms | submissions: {:.0}, preparing {:.1} ms, card wait {:.1} ms; \
         copies {:.1} ms {:.1} MB; copy-back {:.1} ms {:.2} MB | cardmeter (delayed): \
         sync {:.1} ms in {:.0}, check {:.1} ms in {:.0}, lists {:.1} ms in {:.0} | \
         execute all {:.1} ms | textures: refresh {:.1} ms, {:.0} uploads | \
         remaps: {:.0} ({:.1} ms) | pipelines: {:.0} made, {:.0} loaded, {:.1} ms waited | \
         disk: {:.0} reads {:.1} MB {:.1} worker ms | faults: {:.0} page, {:.0} guard | \
         process: {:.0} user {:.0} kernel ms | pacing all {:.1} ms | {:.0} draws, {:.0} dispatches | \
         {:.0} MB free | slow core render {:.0}/{:.0}",
        f.us / 1e3,
        (s(S_THREAD_USER) + s(S_THREAD_KERNEL)) / 1e3,
        s(S_THREAD_USER) / 1e3,
        s(S_THREAD_KERNEL) / 1e3,
        ms(T::PresentUs),
        ms(T::PresentShowUs),
        h(H::ExecuteUs) / 1e3,
        h(H::PaceUs) / 1e3,
        t(T::Submissions),
        ms(T::PrepareUs),
        ms(T::CardWaitUs),
        ms(T::CopyUs),
        t(T::CopyBytes) / 1e6,
        ms(T::CopyBackUs),
        t(T::CopyBackBytes) / 1e6,
        ms(T::CardSyncUs),
        t(T::CardSyncLists),
        ms(T::CardCheckUs),
        t(T::CardCheckLists),
        ms(T::CardListsUs),
        t(T::CardLists),
        ms(T::ExecuteUs),
        ms(T::RefreshUs),
        t(T::Uploads),
        t(T::Remaps),
        ms(T::RemapUs),
        s(S_PIPELINES_MADE),
        s(S_PIPELINES_LOADED),
        ms(T::PipelineWaitUs),
        t(T::DiskReads),
        t(T::DiskBytes) / 1e6,
        ms(T::DiskUs),
        s(S_PAGE_FAULTS),
        t(T::GuardFaults),
        s(S_PROCESS_USER) / 1e3,
        s(S_PROCESS_KERNEL) / 1e3,
        ms(T::PaceUs),
        s(S_DRAWS),
        s(S_DISPATCHES),
        s(S_FREE_MB),
        t(T::RenderSlow),
        t(T::RenderLooks),
    )
}

fn mean(frames: &[&Frame]) -> Frame {
    let n = frames.len() as f64;
    let mut out = Frame {
        number: 0,
        at: 0.0,
        us: 0.0,
        t: [0.0; N],
        s: [0.0; S],
        h: [0.0; HN],
    };
    for f in frames {
        out.us += f.us / n;
        for i in 0..N {
            out.t[i] += f.t[i] / n;
        }
        for i in 0..S {
            out.s[i] += f.s[i] / n;
        }
        for i in 0..HN {
            out.h[i] += f.h[i] / n;
        }
    }
    out
}

fn long_over(usual: f64) -> f64 {
    (usual * 1.5).max(usual + 40_000.0)
}

fn lines(frames: &[Frame], usual: f64, seconds: f64) -> Vec<String> {
    let over = long_over(usual);
    let (long, plain): (Vec<&Frame>, Vec<&Frame>) = frames.iter().partition(|f| f.us > over);
    let each: Vec<String> = frames.iter().map(|f| format!("{:.0}", f.us / 1e3)).collect();
    let shortest = frames.iter().map(|f| f.us).fold(f64::INFINITY, f64::min);
    let longest = frames.iter().map(|f| f.us).fold(0.0, f64::max);
    let mut out = vec![format!(
        "frame times: {} frames in {seconds:.1} s: shortest {:.0} ms, median {:.0}, longest {:.0}; \
         {} long (over {:.0} ms, the usual frame of the last {RECENT} being {:.0}); each: {}",
        frames.len(),
        shortest / 1e3,
        median(frames.iter().map(|f| f.us)) / 1e3,
        longest / 1e3,
        long.len(),
        over / 1e3,
        usual / 1e3,
        each.join(" ")
    )];
    if !plain.is_empty() {
        out.push(format!(
            "usual frame (the mean of {} not long): {}",
            plain.len(),
            text(&mean(&plain))
        ));
    }
    for f in long.iter().take(LONG_MOST) {
        out.push(format!(
            "long frame {} at {:.1} s: {}",
            f.number,
            f.at,
            text(f)
        ));
    }
    if long.len() > LONG_MOST {
        out.push(format!(
            "long frames: {} more in this reading are not listed",
            long.len() - LONG_MOST
        ));
    }
    out
}

const RECENT: usize = 64;
const LONG_MOST: usize = 8;

/// Records a finished frame at present time and emits a 5-second breakdown when enabled.
pub fn frame() {
    if !on() {
        return;
    }
    crate::cardmeter::collect();
    add(T::RenderLooks, 1);
    if on_slow_core() {
        add(T::RenderSlow, 1);
    }
    let now = std::time::Instant::now();
    let totals: [u64; N] = std::array::from_fn(|i| TOTAL[i].load(Relaxed));
    let sampled = sample();
    let here = HERE.with(|c| c.replace([0; HN]));
    // SAFETY: plain current-thread ID query.
    let thread = unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() };
    let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let state = state.get_or_insert_with(|| State {
        first: now,
        last: None,
        number: 0,
        frames: Vec::new(),
        since: now,
        recent: std::collections::VecDeque::new(),
    });
    state.number += 1;
    if let Some((then, t0, s0, thread0)) = state.last.replace((now, totals, sampled, thread)) {
        let mut f = Frame {
            number: state.number,
            at: (now - state.first).as_secs_f64(),
            us: (now - then).as_micros() as f64,
            t: std::array::from_fn(|i| totals[i].wrapping_sub(t0[i]) as i64 as f64),
            s: std::array::from_fn(|i| sampled[i].wrapping_sub(s0[i]) as i64 as f64),
            h: here.map(|v| v as f64),
        };
        f.s[S_FREE_MB] = sampled[S_FREE_MB] as f64;
        if thread != thread0 {
            (f.s[S_THREAD_USER], f.s[S_THREAD_KERNEL]) = (0.0, 0.0);
        }
        state.frames.push(f);
    }
    let elapsed = state.since.elapsed();
    if elapsed < std::time::Duration::from_secs(5) || state.frames.is_empty() {
        return;
    }
    let frames = std::mem::take(&mut state.frames);
    state.since = now;
    for f in &frames {
        if state.recent.len() == RECENT {
            state.recent.pop_front();
        }
        state.recent.push_back(f.us);
    }
    let usual = median(state.recent.iter().copied());
    for line in lines(&frames, usual, elapsed.as_secs_f64()) {
        tracing::info!("{line}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(number: u64, ms: f64) -> Frame {
        let mut f = Frame {
            number,
            at: number as f64,
            us: ms * 1e3,
            t: [0.0; N],
            s: [0.0; S],
            h: [0.0; HN],
        };
        f.t[T::PresentUs as usize] = ms * 100.0;
        f.t[T::Uploads as usize] = ms;
        f
    }

    #[test]
    fn a_long_frame_is_listed_beside_the_usual_one() {
        let frames: Vec<Frame> = [150.0, 160.0, 140.0, 400.0, 150.0]
            .iter()
            .enumerate()
            .map(|(i, &ms)| frame(i as u64, ms))
            .collect();
        let out = lines(&frames, 150_000.0, 1.0);
        assert_eq!(out.len(), 3, "{out:#?}");
        assert!(
            out[0].starts_with(
                "frame times: 5 frames in 1.0 s: shortest 140 ms, median 150, longest 400; 1 long (over 225 ms"
            ),
            "{}",
            out[0]
        );
        assert!(out[0].ends_with("each: 150 160 140 400 150"), "{}", out[0]);
        assert!(
            out[1].starts_with("usual frame (the mean of 4 not long): 150 ms |"),
            "{}",
            out[1]
        );
        assert!(out[1].contains("present 15.0 ms"), "{}", out[1]);
        assert!(out[1].contains("150 uploads"), "{}", out[1]);
        assert!(
            out[2].starts_with("long frame 3 at 3.0 s: 400 ms |"),
            "{}",
            out[2]
        );
        assert!(out[2].contains("present 40.0 ms"), "{}", out[2]);
        assert!(out[2].contains("400 uploads"), "{}", out[2]);
    }

    #[test]
    fn long_is_half_as_long_again_and_at_least_40_ms_more() {
        assert_eq!(long_over(150_000.0), 225_000.0);
        assert_eq!(long_over(30_000.0), 70_000.0);
    }

    #[test]
    fn more_long_frames_than_are_listed_are_counted() {
        let exact: Vec<Frame> = (0..LONG_MOST).map(|i| frame(i as u64, 500.0)).collect();
        let exact_out = lines(&exact, 100_000.0, 5.0);
        assert_eq!(
            exact_out.iter().filter(|l| l.starts_with("long frame ")).count(),
            LONG_MOST
        );
        assert!(!exact_out.iter().any(|l| l.starts_with("long frames:")));
        let mut frames: Vec<Frame> = (0..12).map(|i| frame(i, 500.0)).collect();
        frames.push(frame(12, 100.0));
        let out = lines(&frames, 100_000.0, 5.0);
        assert_eq!(
            out.iter().filter(|l| l.starts_with("long frame ")).count(),
            LONG_MOST
        );
        assert!(
            out.last().unwrap().starts_with("long frames: 4 more"),
            "{out:#?}"
        );
    }

    #[test]
    fn totals_and_a_threads_own_counts_add_up() {
        let before = TOTAL[T::DiskReads as usize].load(Relaxed);
        add(T::DiskReads, 3);
        assert_eq!(total(T::DiskReads) - before, 3);
        here(H::PaceUs, 5);
        here(H::PaceUs, 6);
        assert_eq!(HERE.with(|c| c.get())[H::PaceUs as usize], 11);
    }

    #[test]
    fn frame_log_helpers_cover_switches_filetime_median_and_means() {
        use std::ffi::OsStr;
        assert!(!parse_on(None, false));
        assert!(parse_on(None, true));
        assert!(!parse_on(Some(OsStr::new("0")), true));
        assert!(parse_on(Some(OsStr::new("1")), false));

        assert_eq!(median(std::iter::empty()), 0.0);
        assert_eq!(median([30.0, 10.0, 20.0].into_iter()), 20.0);
        assert_eq!(median([10.0, 40.0, 20.0, 30.0].into_iter()), 30.0);
        assert_eq!(long_over(0.0), 40_000.0);
        assert_eq!(long_over(100_000.0), 150_000.0);
        // Boundary: equal to `long_over(100_000.0) == 150_000.0` is not marked long (`>` strict).
        let boundary = lines(&[frame(0, 150.0)], 100_000.0, 1.0);
        assert!(boundary[0].contains("0 long"), "{}", boundary[0]);
        assert!(boundary.iter().all(|l| !l.starts_with("long frame")));

        let ft = windows_sys::Win32::Foundation::FILETIME {
            dwLowDateTime: 1_000_000,
            dwHighDateTime: 1,
        };
        assert_eq!(micros(ft), ((1u64 << 32) + 1_000_000) / 10);

        let mut a = frame(1, 100.0);
        a.s = [
            20_000.0, 2_000.0, 50_000.0, 5_000.0, 12.0, 100.0, 20.0, 2.0, 4.0, 4096.0,
        ];
        a.h = [3_000.0, 4_000.0];
        a.t[T::PresentShowUs as usize] = 2_000.0;
        a.t[T::Submissions as usize] = 4.0;
        a.t[T::PrepareUs as usize] = 6_000.0;
        a.t[T::CardWaitUs as usize] = 8_000.0;
        a.t[T::CopyUs as usize] = 1_000.0;
        a.t[T::CopyBytes as usize] = 2_000_000.0;
        a.t[T::CopyBackUs as usize] = 500.0;
        a.t[T::CopyBackBytes as usize] = 100_000.0;
        a.t[T::CardSyncUs as usize] = 4_000.0;
        a.t[T::CardSyncLists as usize] = 2.0;
        a.t[T::CardCheckUs as usize] = 2_000.0;
        a.t[T::CardCheckLists as usize] = 1.0;
        a.t[T::CardListsUs as usize] = 12_000.0;
        a.t[T::CardLists as usize] = 6.0;
        a.t[T::ExecuteUs as usize] = 9_000.0;
        a.t[T::RefreshUs as usize] = 3_000.0;
        a.t[T::Remaps as usize] = 2.0;
        a.t[T::RemapUs as usize] = 1_000.0;
        a.t[T::PipelineWaitUs as usize] = 5_000.0;
        a.t[T::DiskReads as usize] = 10.0;
        a.t[T::DiskBytes as usize] = 4_000_000.0;
        a.t[T::DiskUs as usize] = 2_000.0;
        a.t[T::GuardFaults as usize] = 6.0;
        a.t[T::PaceUs as usize] = 8_000.0;
        a.t[T::RenderLooks as usize] = 1.0;
        let mut b = a.clone();
        b.number = 2;
        b.us = 200_000.0;
        b.t[T::PresentUs as usize] = 20_000.0;
        b.t[T::Uploads as usize] = 200.0;
        b.t[T::CardSyncUs as usize] = 6_000.0;
        b.t[T::CardSyncLists as usize] = 4.0;
        let avg = mean(&[&a, &b]);
        assert_eq!(avg.us, 150_000.0);
        assert_eq!(avg.s[S_THREAD_USER], 20_000.0);
        assert_eq!(avg.h[H::ExecuteUs as usize], 3_000.0);
        assert_eq!(avg.t[T::CardSyncLists as usize], 3.0);
        let summary = text(&avg);
        for needle in [
            "150 ms |",
            "ran 22 ms (user 20, kernel 2)",
            "present 15.0 ms (show 2.0)",
            "execute 3.0 ms, paced 4.0 ms",
            "submissions: 4, preparing 6.0 ms, card wait 8.0 ms",
            "copies 1.0 ms 2.0 MB",
            "copy-back 0.5 ms 0.10 MB",
            "sync 5.0 ms in 3, check 2.0 ms in 1, lists 12.0 ms in 6",
            "execute all 9.0 ms",
            "refresh 3.0 ms, 150 uploads",
            "remaps: 2 (1.0 ms)",
            "pipelines: 2 made, 4 loaded, 5.0 ms waited",
            "disk: 10 reads 4.0 MB 2.0 worker ms",
            "faults: 12 page, 6 guard",
            "process: 50 user 5 kernel ms",
            "pacing all 8.0 ms",
            "100 draws, 20 dispatches",
            "4096 MB free",
            "slow core render 0/1",
        ] {
            assert!(summary.contains(needle), "missing {needle:?} in {summary}");
        }
    }
}
