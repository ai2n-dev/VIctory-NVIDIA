//! The host GPU: one D3D12 device and direct queue (decision 0007), and a
//! window to present on.
//!
//! shortcut: every submission is waited for, so CPU and GPU never overlap;
//! overlap them with per-submission fences once more than clears run.

use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Foundation::{HANDLE, HWND, LPARAM, LRESULT, RECT, WAIT_FAILED, WPARAM};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::{Interface, PCWSTR, w};

/// Resource states without the console's own bits (compression hints from
/// bit 25 up), which the host rejects.
pub(crate) fn pc_state(state: u32) -> u32 {
    state & 0x01ff_ffff
}

/// A console state as a host texture state: PC bits only, and none of the
/// buffer-only ones the console allows on textures (vertex, constant and
/// index buffer, stream out, indirect arguments, as in "generic read").
pub(crate) fn texture_state(state: u32) -> u32 {
    const BUFFER_ONLY: u32 = 0x1 | 0x2 | 0x100 | 0x200;
    pc_state(state) & !BUFFER_ONLY
}

/// Render target views live only until their command is recorded, so a
/// ring of host descriptors is enough.
const RTV_RING: u32 = 1024;

/// An asynchronous submission still running on the GPU: its staging textures
/// and bindless plans stay alive until `fence` completes.
struct InFlight {
    fence: u64,
    submission: u64,
    _early_copies: Vec<(
        std::sync::Arc<crate::texture::Staging>,
        std::sync::Arc<crate::resource::HostTexture>,
    )>,
    _plans: Vec<std::sync::Arc<crate::descriptor::host::BindlessPlan>>,
    /// What is done once it has run ([`Host::finish_submission`]): its
    /// lists, kept alive for that, and the ranges its shaders wrote.
    finish: Option<Finish>,
}

/// See [`InFlight::finish`].
struct Finish {
    lists: Vec<std::sync::Arc<HostList>>,
    back: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
}
// SAFETY: host lists are free-threaded objects, run by one thread once
// closed; the rest is ours and locked.
unsafe impl Send for Finish {}

/// Whether the title's submissions may return before the host GPU is done
/// (`VICTORY_ASYNC_SUBMIT=1`; off by default until RC0 async baseline passes).
#[cfg(not(test))]
fn async_submit() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASYNC_SUBMIT").is_some_and(|v| v == "1"))
}

// Tests run submissions synchronously, whatever the environment says,
// unless a test of the asynchronous path turns it on for its own thread
// (a flag for the whole process made the title submissions of the queue
// tests running beside it asynchronous too).
#[cfg(test)]
thread_local! {
    static ASYNC_IN_TESTS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
fn async_submit() -> bool {
    ASYNC_IN_TESTS.with(std::cell::Cell::get)
}

/// Microseconds spent bringing buffer copies up to date while the
/// submission before ran ([`Host::prewarm`]), since the last frame line.
static PREWARM_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

thread_local! {
    /// This thread is putting back the results of an unwaited submission,
    /// holding `Host::submitted`: what it calls meanwhile must not wait for
    /// that submission or take the lock again.
    static FINISHING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Marks the thread as finishing a submission until dropped.
struct Finishing;

impl Finishing {
    fn enter() -> Finishing {
        FINISHING.with(|finishing| finishing.set(true));
        Finishing
    }
}

impl Drop for Finishing {
    fn drop(&mut self) {
        FINISHING.with(|finishing| finishing.set(false));
    }
}

fn finishing_here() -> bool {
    FINISHING.with(std::cell::Cell::get)
}

pub(crate) struct Host {
    pub(crate) device: ID3D12Device,
    pub(crate) depth_bounds_supported: bool,
    /// Resizable BAR (`D3D12_HEAP_TYPE_GPU_UPLOAD`) supported by device/OS.
    pub(crate) gpu_upload: bool,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    /// The last value signalled on `fence`, behind the submission lock.
    submitted: Mutex<u64>,
    /// Async submissions not yet retired, oldest first.
    in_flight: Mutex<std::collections::VecDeque<InFlight>>,
    /// The newest fence value of `fence` whose submission is finished: its
    /// results are put back, not just run. One per host, as each counts its
    /// own fence.
    finished_fence: std::sync::atomic::AtomicU64,
    event: OwnedHandle,
    rtvs: ID3D12DescriptorHeap,
    rtv_size: u32,
    /// A ring of depth views, used as `rtvs`.
    dsvs: ID3D12DescriptorHeap,
    dsv_size: u32,
    next_dsv: AtomicU32,
    /// With `VICTORY_D3D12_DEBUG` set (diagnostics builds): the debug
    /// layer's messages.
    messages: Option<ID3D12InfoQueue>,
    next_rtv: AtomicU32,
    presenter: Mutex<Option<Presenter>>,
    /// Device-local VRAM card buffer engine (Wave64 GPU page hashing), lazily initialized.
    card: OnceLock<Option<Mutex<crate::card::CardEngine>>>,
    /// Tests only: held by a WARP host while it lives, so that one WARP
    /// device works at a time ([`warp_serial`]). Declared last: it is
    /// released after every object of the host.
    #[cfg(test)]
    warp_serial: Option<WarpSerial>,
}

// SAFETY: D3D12 devices, queues, fences and descriptor heaps are
// free-threaded; the event handle is only waited on under `submitted`.
unsafe impl Send for Host {}
// SAFETY: as above.
unsafe impl Sync for Host {}

/// Host objects the title let go of while a submission may still use them
/// (a texture its descriptors name, a descriptor slot's old resource), by
/// the submission count they wait for. The title's descriptor writes and
/// releases reach the host at once, from its own threads, while the host
/// runs a submission for most of a second in the world: freeing there was
/// a GPU page fault on the freed texture (DXGI_ERROR_DEVICE_HUNG,
/// 2026-11-24, RX 7600).
static RETIRED: Mutex<std::collections::VecDeque<Retired>> =
    Mutex::new(std::collections::VecDeque::new());
/// Submissions started and finished (waited for).
static STARTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FINISHED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Presented host frames a retired object is also kept for. Worker threads
/// record lists ahead of submission and a title frame spans many
/// submissions, so finishing the submissions running when it was retired
/// is not enough on its own. Ages are in frames presented to the host, not
/// the title's 60 Hz clock (which ticks many times per host frame while
/// the host runs at 1-2 fps).
const KEEP_FRAMES: u64 = 32;

/// And the time a retired object is kept for at least: in the menus 32
/// frames are a third of a second, and lists recorded ahead, or descriptors
/// the title left in a heap, named copies freed by then (with
/// `VICTORY_RAYS=1`: page faults inside destroyed 4 and 10.78 MiB buffers,
/// Radeon GPU Detective, runs rt2 and rt6; with nothing freed the run went
/// on into the world, rt3).
/// shortcut: a time, not a count of what still names the object.
const KEEP_TIME: std::time::Duration = std::time::Duration::from_secs(8);

/// A retired object, dropped on whatever thread finishes the submission.
struct Retired {
    /// The submission count it waits for.
    submission: u64,
    /// The presented frame it was retired in.
    frame: u64,
    /// When.
    at: std::time::Instant,
    /// The video memory it holds as far as the retiring code knows (0 if
    /// it did not say), counted in [`crate::memstat::RETIRED_BYTES`].
    bytes: u64,
    #[allow(dead_code)]
    object: Box<dyn std::any::Any>,
}
// SAFETY: what is retired is D3D12 objects (free-threaded) and the
// reference-counted wrappers holding them, dropped once.
unsafe impl Send for Retired {}

/// Drops `object` once every submission running now has finished and
/// [`KEEP_FRAMES`] more frames were presented.
pub(crate) fn retire<T: 'static>(object: T) {
    retire_bytes(object, 0);
}

/// [`retire`] for an object that holds `bytes` of video memory, so that what
/// is kept waiting for the GPU shows in the memory report and in the
/// pressure that shortens the wait ([`KeepPolicy`]).
pub(crate) fn retire_bytes<T: 'static>(object: T, bytes: u64) {
    // An object's destructor can retire resources with a later GPU use
    // than the object itself. Every retirement takes its own snapshot.
    let submission = STARTED.load(Ordering::Acquire);
    let frame = crate::queue::presented();
    crate::memstat::add(&crate::memstat::RETIRED_BYTES, bytes);
    crate::memstat::add(&crate::memstat::RETIRED_COUNT, 1);
    RETIRED.lock().unwrap_or_else(|e| e.into_inner()).push_back(Retired {
        submission,
        frame,
        at: std::time::Instant::now(),
        bytes,
        object: Box::new(object),
    });
}

/// The longest wait `VICTORY_KEEP_SECS` can ask for (larger values, and values
/// [`Duration`] cannot hold, are cut to it).
const LONGEST_KEEP: std::time::Duration = std::time::Duration::from_secs(24 * 60 * 60);

/// How long retired objects wait, and whether video memory pressure may
/// shorten it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct KeepPolicy {
    /// At least this long after they were retired ([`KEEP_TIME`] by default,
    /// `VICTORY_KEEP_SECS=<seconds>`).
    time: std::time::Duration,
    /// `VICTORY_KEEP_STRICT=1`: the time is kept whatever the video memory
    /// does, for comparing runs.
    strict: bool,
}

impl KeepPolicy {
    fn parse(secs: Option<&str>, strict: Option<&str>) -> Self {
        let time = secs
            .and_then(|secs| secs.trim().parse::<f64>().ok())
            .filter(|secs| secs.is_finite() && *secs >= 0.0)
            .map_or(KEEP_TIME, |secs| {
                std::time::Duration::try_from_secs_f64(secs).unwrap_or(LONGEST_KEEP).min(LONGEST_KEEP)
            });
        KeepPolicy { time, strict: strict.is_some_and(|v| !v.is_empty() && v != "0") }
    }

    fn current() -> Self {
        static POLICY: OnceLock<KeepPolicy> = OnceLock::new();
        *POLICY.get_or_init(|| {
            KeepPolicy::parse(
                std::env::var("VICTORY_KEEP_SECS").ok().as_deref(),
                std::env::var("VICTORY_KEEP_STRICT").ok().as_deref(),
            )
        })
    }
}

/// Whether only the time still keeps `r`: its submissions finished and the
/// frames passed. Only then can video memory pressure free it, and only
/// then is the driver asked about the pressure.
fn waits_for_time_only(r: &Retired, finished: u64, frame: u64, keep_time: std::time::Duration) -> bool {
    r.submission <= finished && frame.saturating_sub(r.frame) >= KEEP_FRAMES && r.at.elapsed() < keep_time
}

/// Percent of the video memory budget from which retired objects stop
/// waiting for [`KEEP_TIME`] (the host already keeps uploads out of video memory
/// above 85 %).
const PRESSURE_PERCENT: u64 = 90;

/// Whether video memory is used up to [`PRESSURE_PERCENT`] of its budget, asked of
/// the driver at most every 200 ms.
fn video_memory_pressure() -> bool {
    static CHECKED: Mutex<Option<(std::time::Instant, bool)>> = Mutex::new(None);
    let mut checked = CHECKED.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((at, pressure)) = *checked
        && at.elapsed() < std::time::Duration::from_millis(200)
    {
        return pressure;
    }
    let pressure = video_memory().is_some_and(|[(used, budget), _]| {
        budget != 0 && used.saturating_mul(100) >= budget.saturating_mul(PRESSURE_PERCENT)
    });
    *checked = Some((std::time::Instant::now(), pressure));
    pressure
}

/// How many objects at the front of `retired` (oldest first) may be dropped:
/// those whose submissions are among the `finished` ones, that were retired
/// `keep_frames` presented frames and `keep_time` ago.
fn expired(
    retired: &std::collections::VecDeque<Retired>,
    finished: u64,
    frame: u64,
    keep_time: std::time::Duration,
) -> usize {
    retired
        .iter()
        .take_while(|r| {
            r.submission <= finished
                && frame.saturating_sub(r.frame) >= KEEP_FRAMES
                && r.at.elapsed() >= keep_time
        })
        .count()
}

/// Drops what waited for submissions up to `finished` and is old enough
/// in presented frames.
fn bury(finished: u64) {
    // Diagnostics (`VICTORY_KEEP_RETIRED`): nothing is freed, to tell a use
    // after free from a wild address.
    if cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_KEEP_RETIRED").is_some() {
        return;
    }
    let now = crate::queue::presented();
    let policy = KeepPolicy::current();
    let done: Vec<Retired> = {
        let mut retired = RETIRED.lock().unwrap_or_else(|e| e.into_inner());
        // In the order they came: the old ones are at the front.
        let mut n = expired(&retired, finished, now, policy.time);
        if !policy.strict
            && retired.get(n).is_some_and(|r| waits_for_time_only(r, finished, now, policy.time))
        {
            // Resource releases can enqueue while DXGI queries the driver.
            drop(retired);
            let time = if video_memory_pressure() {
                std::time::Duration::ZERO
            } else {
                policy.time
            };
            retired = RETIRED.lock().unwrap_or_else(|e| e.into_inner());
            n = expired(&retired, finished, now, time);
        }
        retired.drain(..n).collect()
    };
    for r in &done {
        crate::memstat::sub(&crate::memstat::RETIRED_BYTES, r.bytes);
        crate::memstat::sub(&crate::memstat::RETIRED_COUNT, 1);
    }
    drop(done);
}

/// Time spent in submissions since the last frame-rate line, in
/// microseconds: preparing them on the CPU (title memory copies, texture
/// detiling), then waiting for the GPU.
static SUBMISSION_US: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// After submissions: shader-written ranges compared and copied back into
/// title memory (microseconds, bytes that differed), since the last
/// frame-rate line.
static COPY_BACK: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// Bindless heaps' slots made ready (microseconds), since the last
/// frame-rate line: plans at submission (part of the preparing), and
/// placed textures as draws are recorded.
pub(crate) static BINDLESS_US: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// Of the preparing: title memory copies (microseconds, bytes) and texture
/// refreshes (microseconds, bytes hashed), since the last frame-rate line.
pub(crate) static PREPARING: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// Adds to [`SUBMISSION_US`] and the frame log's submission counters (`0` prepare, `1` wait).
fn submission_us(i: usize, us: u64) {
    SUBMISSION_US[i].fetch_add(us, Ordering::Relaxed);
    crate::framelog::add(
        [
            crate::framelog::T::PrepareUs,
            crate::framelog::T::CardWaitUs,
        ][i],
        us,
    );
    if i == 0 {
        crate::framelog::add(crate::framelog::T::Submissions, 1);
    }
}

/// Adds to [`PREPARING`] copy counters (`0` microseconds, `1` bytes) and the frame log.
fn preparing(i: usize, n: u64) {
    PREPARING[i].fetch_add(n, Ordering::Relaxed);
    crate::framelog::add(
        [crate::framelog::T::CopyUs, crate::framelog::T::CopyBytes][i],
        n,
    );
}

/// Adds to [`COPY_BACK`] (`0` microseconds, `1` bytes) and the frame log.
fn copy_back(i: usize, n: u64) {
    COPY_BACK[i].fetch_add(n, Ordering::Relaxed);
    crate::framelog::add(
        [
            crate::framelog::T::CopyBackUs,
            crate::framelog::T::CopyBackBytes,
        ][i],
        n,
    );
}

/// What the runtime does when the host GPU fails for good: report it like a
/// crash and end the run. The console has no such state, and carrying on
/// draws nothing at seconds per frame.
static ON_FAILURE: OnceLock<fn(&str) -> !> = OnceLock::new();

/// Sets what a host GPU failure does (the runtime's crash report).
pub fn on_failure(report: fn(&str) -> !) {
    let _ = ON_FAILURE.set(report);
}

/// Ends the run on a host GPU failure: what failed, the removal reason and
/// the debug layer's first error, if any (`VICTORY_D3D12_DEBUG=1` in a
/// diagnostics build names the command).
pub(crate) fn fail(what: std::fmt::Arguments) -> ! {
    let mut text = what.to_string();
    if let Some(host) = host() {
        // SAFETY: a plain query.
        if let Err(e) = unsafe { host.device.GetDeviceRemovedReason() } {
            text.push_str(&format!("; device removed: {e}"));
        }
        match host.log_messages() {
            Some(first) => text.push_str(&format!("; debug layer: {first}")),
            None if host.messages.is_none() => text.push_str(
                "; a diagnostics build with VICTORY_D3D12_DEBUG=1 names the command",
            ),
            None => {}
        }
    }
    tracing::error!("{text}");
    match ON_FAILURE.get() {
        Some(report) => report(&text),
        None => {
            eprintln!("victory-gpu host::fail: {text}");
            std::process::abort();
        }
    }
}

/// The host GPU, created on first use; `None` without one (a run then
/// behaves as before: nothing executes).
static HOST: OnceLock<Option<Host>> = OnceLock::new();

/// The host GPU's memory use as its driver reports it, `[(used, budget)]`
/// in video memory and in system memory; `None` before the device is
/// made (asking does not make it).
pub(crate) fn video_memory() -> Option<[(u64, u64); 2]> {
    static ADAPTER: OnceLock<Option<IDXGIAdapter3>> = OnceLock::new();
    let host = HOST.get()?.as_ref()?;
    let adapter = ADAPTER
        .get_or_init(|| {
            // SAFETY: plain DXGI queries on a live device.
            unsafe {
                let luid = host.device.GetAdapterLuid();
                let factory: IDXGIFactory4 =
                    CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
                factory.EnumAdapterByLuid::<IDXGIAdapter3>(luid).ok()
            }
        })
        .as_ref()?;
    let query = |group| {
        let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: node 0 of the device's own adapter, a sized out structure.
        unsafe { adapter.QueryVideoMemoryInfo(0, group, &mut info) }.ok()?;
        Some((info.CurrentUsage, info.Budget))
    };
    Some([
        query(DXGI_MEMORY_SEGMENT_GROUP_LOCAL)?,
        query(DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL)?,
    ])
}

pub(crate) fn host() -> Option<&'static Host> {
    HOST.get_or_init(|| match create() {
        Ok(host) => {
            // Its threads start now, not inside the first submission that
            // splits a copy.
            crate::workpool::threads();
            Some(host)
        }
        Err(e) => {
            tracing::warn!(error = %e, "no host D3D12 device; command lists will not run");
            None
        }
    })
    .as_ref()
}

/// Whether a submission is still to be waited for and finished.
pub(crate) fn has_in_flight() -> bool {
    matches!(HOST.get(), Some(Some(h)) if h.has_in_flight())
}

/// Waits for any in-flight async host GPU submissions without creating the
/// host device if it has not been initialised yet.
pub(crate) fn wait_in_flight_if_ready() {
    if let Some(Some(h)) = HOST.get() {
        h.wait_in_flight();
    }
}

fn create() -> windows::core::Result<Host> {
    crate::drvwatch::start();
    // SAFETY: plain D3D12 object creation with valid descriptions.
    unsafe {
        // The debug layer only in builds with diagnostics.
        let debug =
            cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_D3D12_DEBUG").is_some();
        if debug {
            let mut layer: Option<ID3D12Debug> = None;
            match D3D12GetDebugInterface(&mut layer) {
                Ok(()) => layer.iter().for_each(|l| l.EnableDebugLayer()),
                Err(e) => tracing::warn!(error = %e, "no D3D12 debug layer"),
            }
            // GPU-based validation (`VICTORY_D3D12_GBV`): out-of-range
            // descriptor heap indices and the like become messages naming
            // the command instead of page faults. Without state tracking
            // (directly indexed textures are not transitioned).
            if std::env::var_os("VICTORY_D3D12_GBV").is_some()
                && let Some(l) = &layer
            {
                if let Ok(d1) = l.cast::<ID3D12Debug1>() {
                    d1.SetEnableGPUBasedValidation(true);
                }
                if let Ok(d2) = l.cast::<ID3D12Debug2>() {
                    d2.SetGPUBasedValidationFlags(
                        D3D12_GPU_BASED_VALIDATION_FLAGS_DISABLE_STATE_TRACKING,
                    );
                }
                tracing::info!("GPU-based validation on");
            }
        }
        // Diagnostics: device-removed data (the command that faulted, the
        // faulting address), read when the device is removed.
        if dred_on() {
            let mut dred: Option<ID3D12DeviceRemovedExtendedDataSettings> = None;
            if D3D12GetDebugInterface(&mut dred).is_ok()
                && let Some(dred) = dred
            {
                dred.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                dred.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            }
        }
        let mut device: Option<ID3D12Device> = None;
        // Diagnostics: the software adapter, on which a shader's wild access
        // is an access violation in this process (a crash report with the
        // address and instruction) instead of a GPU fault and driver reset.
        if (cfg!(feature = "diagnostics") || cfg!(test))
            && std::env::var_os("VICTORY_D3D12_WARP").is_some()
        {
            let adapter = warp_adapter()?;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            tracing::info!("host device on the software adapter (WARP)");
        } else {
            match chosen_adapter() {
                Ok(adapter) => D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)?,
                Err(e) => {
                    tracing::warn!(error = %e, "no adapter preference; the default is used");
                    D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device)?;
                }
            }
        }
        let device = device.ok_or_else(windows::core::Error::empty)?;
        let messages = debug
            .then(|| device.cast::<ID3D12InfoQueue>().ok())
            .flatten();
        let host = Host::from_device(device, messages)?;
        if std::env::var_os("VICTORY_LAST_OP_FILE").is_some_and(|v| !v.is_empty()) {
            let report = probe_device(&host.device);
            crate::preflight::record_last_op(format_args!("device_init:{}", report.name));
            eprintln!("[gpu-gate adapter] {}", report.name);
        }
        tracing::info!("host D3D12 device ready");
        Ok(host)
    }
}

impl Host {
    fn from_device(
        device: ID3D12Device,
        messages: Option<ID3D12InfoQueue>,
    ) -> windows::core::Result<Host> {
        // SAFETY: plain D3D12 feature query and queue, fence, event and
        // descriptor heap creation.
        unsafe {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS2::default();
            let depth_bounds_supported = device
                .CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS2,
                    (&raw mut options).cast(),
                    size_of_val(&options) as u32,
                )
                .is_ok()
                && options.DepthBoundsTestSupported.as_bool();
            let report = probe_device(&device);
            let gpu_upload = !cfg!(test)
                && std::env::var_os("VICTORY_GPU_UPLOAD").is_some_and(|v| v == "1")
                && report.rebar_supported;
            tracing::info!(
                adapter = %report.name,
                vendor = format_args!("{:#06x}", report.vendor),
                device = format_args!("{:#06x}", report.device),
                vram_mb = report.vram_mb,
                luid = format_args!("{:08x}:{:08x}", report.luid_high, report.luid_low),
                wave_min = report.wave_min,
                wave_max = report.wave_max,
                binding_tier = report.binding_tier,
                mesh_tier = report.mesh_tier,
                shader_model = format_args!("{:#x}", report.shader_model),
                rebar_supported = report.rebar_supported,
                gpu_upload,
                depth_bounds_supported,
                "host D3D12 adapter confirmed"
            );
            let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })?;
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
            // SAFETY: CreateEventW returns a newly owned handle on success.
            // Own it immediately, including if later heap creation fails.
            let event = OwnedHandle::from_raw_handle(CreateEventW(None, false, false, None)?.0);
            let rtvs: ID3D12DescriptorHeap =
                device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    NumDescriptors: RTV_RING,
                    ..Default::default()
                })?;
            let rtv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
            let dsvs: ID3D12DescriptorHeap =
                device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                    NumDescriptors: RTV_RING,
                    ..Default::default()
                })?;
            let dsv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
            Ok(Host {
                device,
                depth_bounds_supported,
                gpu_upload,
                queue,
                fence,
                submitted: Mutex::new(0),
                in_flight: Mutex::new(std::collections::VecDeque::new()),
                finished_fence: std::sync::atomic::AtomicU64::new(0),
                event,
                rtvs,
                rtv_size,
                dsvs,
                dsv_size,
                next_dsv: AtomicU32::new(0),
                messages,
                next_rtv: AtomicU32::new(0),
                presenter: Mutex::new(None),
                card: OnceLock::new(),
                #[cfg(test)]
                warp_serial: None,
            })
        }
    }

    fn card_engine(&self) -> Option<&Mutex<crate::card::CardEngine>> {
        self.card
            .get_or_init(|| match crate::card::CardEngine::new(self) {
                Ok(engine) => {
                    tracing::info!("VRAM card buffer engine ready (Wave64 GPU hashing)");
                    Some(Mutex::new(engine))
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "VRAM card buffer engine unavailable; falling back to host RAM"
                    );
                    None
                }
            })
            .as_ref()
    }

    pub(crate) fn card_ready(&self) -> bool {
        self.card_engine().is_some()
    }

    /// Signals `self.fence`, updates [`crate::cardmeter`], and waits for
    /// completion. Caller MUST hold `self.submitted`.
    fn signal_and_wait_locked(&self, submitted: &mut u64, warn_msg: &str) {
        *submitted += 1;
        let fence_val = *submitted;
        crate::preflight::record_last_op(format_args!("fence_signal:{fence_val}"));
        // SAFETY: caller holds `self.submitted`; fence & event are ours.
        match unsafe { self.queue.Signal(&self.fence, fence_val) } {
            Ok(()) => {
                crate::cardmeter::signalled(&self.fence, fence_val);
                crate::preflight::record_last_op(format_args!("fence_wait:{fence_val}"));
                wait_for_value(&self.fence, Some(HANDLE(self.event.as_raw_handle())), fence_val);
            }
            Err(e) => tracing::warn!(error = %e, "{warn_msg}"),
        }
    }

    /// Executes a single command list on the direct queue and waits for it to
    /// finish. Caller MUST hold `self.submitted`.
    pub(crate) fn execute_and_wait_locked(&self, cmd: ID3D12CommandList, submitted: &mut u64) {
        let lists = [Some(cmd)];
        crate::cardmeter::executing(&lists, &self.fence);
        crate::preflight::record_last_op("execute_and_wait:1");
        // SAFETY: caller holds `self.submitted` and `cmd` is closed.
        unsafe { self.queue.ExecuteCommandLists(&lists) };
        self.signal_and_wait_locked(submitted, "card GPU wait failed");
    }

    /// Creates an isolated WARP-backed `Host` for unit tests without touching
    /// the process-global host singleton.
    #[cfg(test)]
    pub(crate) fn new_warp() -> Option<Host> {
        let serial = warp_serial();
        let adapter = warp_adapter().ok()?;
        let mut device: Option<ID3D12Device> = None;
        // SAFETY: `adapter` is a verified software WARP adapter.
        unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).ok()? };
        let mut host = Host::from_device(device?, None).ok()?;
        host.warp_serial = Some(serial);
        Some(host)
    }
}

impl Drop for Host {
    fn drop(&mut self) {
        self.wait_in_flight();
    }
}

/// Tests only: taken by everything that works on a WARP device for as long
/// as it does. Tests run on several threads and the WARP adapter hands every
/// one of them the same software device; three of the harnesses of
/// `indirect::tests` running at once ended in an access violation inside
/// d3d10warp.dll about once in thirty runs (13 of 400, and 0 of 300 when
/// the tests ran one at a time or 0 of 400 with this lock). The GPU
/// objects of each test are all released before the lock is.
#[cfg(test)]
pub(crate) fn warp_serial() -> WarpSerial {
    let (busy, freed) = &WARP_BUSY;
    let mut busy = busy.lock().unwrap_or_else(|e| e.into_inner());
    while *busy {
        busy = freed.wait(busy).unwrap_or_else(|e| e.into_inner());
    }
    *busy = true;
    WarpSerial(())
}

/// The turn of [`warp_serial`]: a flag and a condition, not a `MutexGuard`,
/// which must be released by the thread that took it. A host (and the
/// objects of a test) may be dropped on another thread than the one that
/// made it.
#[cfg(test)]
static WARP_BUSY: (Mutex<bool>, std::sync::Condvar) = (Mutex::new(false), std::sync::Condvar::new());

/// Holds the turn until dropped.
#[cfg(test)]
pub(crate) struct WarpSerial(());

#[cfg(test)]
impl Drop for WarpSerial {
    fn drop(&mut self) {
        let (busy, freed) = &WARP_BUSY;
        *busy.lock().unwrap_or_else(|e| e.into_inner()) = false;
        freed.notify_one();
    }
}

/// Waits until `fence` reaches `value`, on `event`, which nothing else
/// waits on meanwhile.
///
/// A device that was removed signals every fence, so the wait ends then too.
/// A device that is alive is waited for as long as it takes: what the
/// caller releases next may still be in use by the GPU. A fence whose event
/// cannot be armed is polled instead of being taken for reached (it was, and
/// the resources of the submission were released under the running GPU). A
/// wait of ten seconds and more is logged every ten seconds.
fn wait_for_value(fence: &ID3D12Fence, event: Option<HANDLE>, value: u64) {
    // SAFETY: plain fence and device queries; the event is the caller's.
    unsafe {
        if fence.GetCompletedValue() >= value {
            return;
        }
        let removed = || {
            let mut device: Option<ID3D12Device> = None;
            fence.GetDevice(&mut device).is_ok()
                && device.is_some_and(|device| device.GetDeviceRemovedReason().is_err())
        };
        // SetEventOnCompletion with a null event is itself a blocking wait.
        let armed = event.is_some_and(|event| match fence.SetEventOnCompletion(value, event) {
            Ok(()) => true,
            Err(e) => {
                tracing::warn!(error = %e, value, "fence event not armed: polling the fence");
                false
            }
        });
        let timeout_ms = crate::preflight::effective_fence_timeout_ms(
            std::env::var("VICTORY_FENCE_TIMEOUT_MS").ok().as_deref(),
            cfg!(test),
        );
        let began = std::time::Instant::now();
        let mut logged = 0;
        while fence.GetCompletedValue() < value {
            if removed() {
                return;
            }
            if let Some(limit) = timeout_ms
                && began.elapsed().as_millis() >= u128::from(limit)
            {
                let completed = fence.GetCompletedValue();
                fail(format_args!(
                    "fence wait timed out after {limit} ms (target={value}, completed={completed})"
                ));
            }
            if !armed || event.is_none_or(|event| WaitForSingleObject(event, 1000) == WAIT_FAILED) {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            let tens = began.elapsed().as_secs() / 10;
            if tens > logged {
                logged = tens;
                tracing::warn!(
                    value,
                    completed = fence.GetCompletedValue(),
                    seconds = began.elapsed().as_secs(),
                    "still waiting for the host GPU"
                );
            }
        }
    }
}

/// Hardware details and D3D12 capabilities queried directly from a created
/// `ID3D12Device` via its LUID, confirming which physical adapter runs.
#[derive(Debug, Clone)]
pub(crate) struct AdapterReport {
    pub(crate) name: String,
    pub(crate) vendor: u32,
    pub(crate) device: u32,
    pub(crate) vram_mb: usize,
    pub(crate) luid_high: i32,
    pub(crate) luid_low: u32,
    pub(crate) wave_min: u32,
    pub(crate) wave_max: u32,
    pub(crate) binding_tier: i32,
    pub(crate) mesh_tier: i32,
    pub(crate) shader_model: i32,
    pub(crate) rebar_supported: bool,
}

fn probe_device(device: &ID3D12Device) -> AdapterReport {
    // SAFETY: plain D3D12 / DXGI queries on a valid live device.
    unsafe {
        let luid = device.GetAdapterLuid();
        let (mut name, mut vendor, mut dev_id, mut vram_mb) = (String::new(), 0, 0, 0);
        if let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            && let Ok(adapter) = factory.EnumAdapterByLuid::<IDXGIAdapter1>(luid)
            && let Ok(desc) = adapter.GetDesc1()
        {
            let end = desc
                .Description
                .iter()
                .position(|&c| c == 0)
                .unwrap_or(desc.Description.len());
            name = String::from_utf16_lossy(&desc.Description[..end]);
            vendor = desc.VendorId;
            dev_id = desc.DeviceId;
            vram_mb = desc.DedicatedVideoMemory >> 20;
        }
        let mut opts0 = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
        let binding_tier = if device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS,
                (&raw mut opts0).cast(),
                size_of_val(&opts0) as u32,
            )
            .is_ok()
        {
            opts0.ResourceBindingTier.0
        } else {
            0
        };
        let mut opts1 = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
        let (wave_min, wave_max) = if device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS1,
                (&raw mut opts1).cast(),
                size_of_val(&opts1) as u32,
            )
            .is_ok()
            && opts1.WaveOps.as_bool()
        {
            (opts1.WaveLaneCountMin, opts1.WaveLaneCountMax)
        } else {
            (0, 0)
        };
        let mut opts7 = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
        let mesh_tier = if device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS7,
                (&raw mut opts7).cast(),
                size_of_val(&opts7) as u32,
            )
            .is_ok()
        {
            opts7.MeshShaderTier.0
        } else {
            0
        };
        let mut sm = D3D12_FEATURE_DATA_SHADER_MODEL {
            HighestShaderModel: D3D_SHADER_MODEL_6_8,
        };
        if device
            .CheckFeatureSupport(
                D3D12_FEATURE_SHADER_MODEL,
                (&raw mut sm).cast(),
                size_of_val(&sm) as u32,
            )
            .is_err()
        {
            sm.HighestShaderModel = D3D_SHADER_MODEL_6_6;
            let _ = device.CheckFeatureSupport(
                D3D12_FEATURE_SHADER_MODEL,
                (&raw mut sm).cast(),
                size_of_val(&sm) as u32,
            );
        }
        let mut opts16 = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
        let rebar_supported = device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS16,
                (&raw mut opts16).cast(),
                size_of_val(&opts16) as u32,
            )
            .is_ok()
            && opts16.GPUUploadHeapSupported.as_bool();
        AdapterReport {
            name,
            vendor,
            device: dev_id,
            vram_mb,
            luid_high: luid.HighPart,
            luid_low: luid.LowPart,
            wave_min,
            wave_max,
            binding_tier,
            mesh_tier,
            shader_model: sm.HighestShaderModel.0,
            rebar_supported,
        }
    }
}

/// The adapter `VICTORY_ADAPTER` names (a part of its description, ignoring
/// case), or the one Windows ranks fastest. A machine with an integrated GPU
/// as well as a discrete one can thus be told which to run on; a name that
/// matches nothing falls back to the fastest, with a warning.
fn chosen_adapter() -> windows::core::Result<IDXGIAdapter1> {
    match std::env::var("VICTORY_ADAPTER") {
        Ok(name) if !name.trim().is_empty() => named_adapter(name.trim()).or_else(|e| {
            tracing::warn!(name, error = %e, "no adapter matches VICTORY_ADAPTER; the fastest is used");
            fastest_adapter()
        }),
        _ => fastest_adapter(),
    }
}

/// The first hardware adapter whose description holds `name`, ignoring case.
fn named_adapter(name: &str) -> windows::core::Result<IDXGIAdapter1> {
    // SAFETY: plain DXGI adapter queries.
    unsafe {
        let factory: IDXGIFactory6 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let mut i = 0;
        loop {
            let adapter: IDXGIAdapter1 = factory.EnumAdapters1(i)?;
            let desc = adapter.GetDesc1()?;
            let end = desc
                .Description
                .iter()
                .position(|&c| c == 0)
                .unwrap_or(desc.Description.len());
            let description = String::from_utf16_lossy(&desc.Description[..end]);
            if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0
                && adapter_matches(&description, name)
            {
                tracing::info!(
                    adapter = description,
                    vendor = format_args!("{:#06x}", desc.VendorId),
                    device = format_args!("{:#06x}", desc.DeviceId),
                    vram_mb = desc.DedicatedVideoMemory >> 20,
                    "host adapter chosen by VICTORY_ADAPTER"
                );
                return Ok(adapter);
            }
            i += 1;
        }
    }
}

/// Whether an adapter description holds `wanted`, ignoring case.
fn adapter_matches(description: &str, wanted: &str) -> bool {
    description.to_lowercase().contains(&wanted.to_lowercase())
}

/// The adapter Windows ranks fastest, so a machine with an integrated GPU
/// too runs on its discrete one.
fn fastest_adapter() -> windows::core::Result<IDXGIAdapter1> {
    // SAFETY: plain DXGI adapter queries.
    unsafe {
        let factory: IDXGIFactory6 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let adapter: IDXGIAdapter1 =
            factory.EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE)?;
        let desc = adapter.GetDesc1()?;
        let end = desc.Description.iter().position(|&c| c == 0).unwrap_or(desc.Description.len());
        tracing::info!(
            name = String::from_utf16_lossy(&desc.Description[..end]),
            vendor = format_args!("{:#06x}", desc.VendorId),
            device = format_args!("{:#06x}", desc.DeviceId),
            vram_mb = desc.DedicatedVideoMemory >> 20,
            "host adapter"
        );
        Ok(adapter)
    }
}

/// DXGI may return hardware for `EnumWarpAdapter` (notably under Proton).
/// Reject it before creating a device in diagnostics' software-only mode.
fn warp_adapter() -> windows::core::Result<IDXGIAdapter> {
    // SAFETY: plain DXGI adapter queries.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let adapter: IDXGIAdapter = factory.EnumWarpAdapter()?;
        let desc = adapter.cast::<IDXGIAdapter1>()?.GetDesc1()?;
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
            tracing::warn!(
                vendor = format_args!("{:#x}", desc.VendorId),
                flags = format_args!("{:#x}", desc.Flags),
                "DXGI did not identify the WARP adapter as software; refusing device"
            );
            return Err(windows::core::Error::from_hresult(
                windows::Win32::Foundation::E_FAIL,
            ));
        }
        Ok(adapter)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn sparse_copies_preserve_islands_and_observe_remaps() {
        use windows_sys::Win32::System::Memory::*;
        const N: usize = 5 << 16;
        // SAFETY: we reserve a private region, commit selected pages inside it,
        // and only directly access the pages while they are writable.
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), N, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            let mut expected = vec![0; N];
            for (offset, value) in [(0, 9), (2 << 16, 37), (N - 4096, 85)] {
                let at = base.add(offset);
                assert_eq!(VirtualAlloc(at.cast(), 4096, MEM_COMMIT, PAGE_READWRITE), at.cast());
                at.write_bytes(value, 4096);
                expected[offset..offset + 4096].fill(value);
            }
            for remap in [false, true] {
                if remap {
                    let mut old = 0;
                    assert_ne!(VirtualProtect(base.cast(), 4096, PAGE_NOACCESS, &mut old), 0);
                    expected[..4096].fill(0);
                    let at = base.add(1 << 16);
                    assert_eq!(VirtualAlloc(at.cast(), 4096, MEM_COMMIT, PAGE_READWRITE), at.cast());
                    at.write_bytes(0xc7, 4096);
                    expected[1 << 16..(1 << 16) + 4096].fill(0xc7);
                }
                let mut out = vec![0xee; N];
                super::copy_mapped(out.as_mut_ptr().add(27), base.add(27), N - 54, false);
                assert_eq!(out[27..N - 27], expected[27..N - 27]);
                assert!(out[..27].iter().chain(&out[N - 27..]).all(|&v| v == 0xee));
            }
            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
        }
    }

    use super::*;

    /// Constants and texture footprints share scratch chunks without overlap.
    #[test]
    fn scratch_uploads_keep_footprint_alignment_and_disjoint_ranges() {
        use windows::core::Interface;
        let Some(host) = host() else { return };
        let Ok(list) = host.new_list() else { return };
        let (gpu, constant) = list.scratch(1).unwrap();
        let (first, offset, bytes) = list.upload(257).unwrap();
        assert_eq!(offset % 512, 0);
        // SAFETY: a live upload buffer.
        assert_eq!(unsafe { first.GetGPUVirtualAddress() } + offset, gpu + 512);
        assert_eq!(bytes as usize - constant as usize, 512);

        let (second, next, next_bytes) = list.upload(1).unwrap();
        assert_eq!(first.as_raw(), second.as_raw());
        assert_eq!(next, offset + 512);
        assert_eq!(next_bytes as usize - bytes as usize, 512);

        let (large, large_offset, _) = list.upload(SCRATCH_CHUNK + 1).unwrap();
        assert_ne!(large.as_raw(), first.as_raw());
        assert_eq!(large_offset, 0);

        let (new_chunk, new_offset, _) = list.upload(SCRATCH_CHUNK).unwrap();
        assert_ne!(new_chunk.as_raw(), first.as_raw());
        assert_eq!(new_offset, 0);
    }

    /// A list with no work before its waits, as the title's queues meet one
    /// another halfway: each write goes as soon as the waits before it are
    /// met, and work holds the later ones back.
    #[test]
    fn writes_before_an_unmet_wait_go_early() {
        let Some(host) = host() else { return };
        let Ok(list) = host.new_list() else { return };
        let mut memory = [0u64; 4];
        let at = memory.as_mut_ptr();
        // SAFETY: inside the array.
        let slot = |i: usize| unsafe { at.add(i) };
        let wait = |i: usize| crate::memwait::Wait {
            address: slot(i) as usize,
            comparison: 7,
            reference: 5,
            bytes: 8,
        };
        let write = |i: usize| list.defer_write(slot(i).cast(), crate::queue::Written::Value(5), 8);
        // SAFETY: as above.
        let value = |i: usize| unsafe { slot(i).read_volatile() };
        list.note_wait(wait(0));
        write(1);
        list.note_wait(wait(2));
        list.count_work();
        write(3);
        assert!(!list.write_early() && value(1) == 0);
        // SAFETY: as above.
        unsafe { slot(0).write_volatile(5) };
        assert_eq!(list.unmet(), [wait(2)]);
        assert!(list.write_early() && !list.write_early());
        assert_eq!((value(1), value(3)), (5, 0));
        // SAFETY: as above.
        unsafe { slot(2).write_volatile(5) };
        assert!(list.unmet().is_empty() && !list.write_early());
        // Run: what is left, then all of it anew for another submission.
        list.close();
        host.execute(&[&list]);
        assert_eq!(value(3), 5);
        // SAFETY: as above.
        unsafe { slot(1).write_volatile(0) };
        host.execute(&[&list]);
        assert_eq!(value(1), 5);
    }

    /// A list of another host than the process's waits on that host's fence,
    /// not on the process's.
    #[test]
    fn a_list_waits_on_the_fence_of_its_own_host() {
        let Some(warp) = Host::new_warp() else { return };
        let Ok(list) = warp.new_list() else { return };
        // A value the process's host (a hardware device, if there is one)
        // never reaches: waiting for it there would not return.
        let value = 1 << 40;
        // SAFETY: our queue and fence.
        unsafe { warp.queue.Signal(&warp.fence, value).unwrap() };
        list.last_fence.store(value, Ordering::Release);
        list.close();
        list.reset();
        drop(list);
    }

    /// With asynchronous submission on, a title submission may return before
    /// its results are put back; they are there once it is waited for.
    #[test]
    fn an_unwaited_submission_is_finished_when_waited_for() {
        let Some(host) = host() else { return };
        let Ok(list) = host.new_list() else { return };
        let mut memory = [0u64; 2];
        let at = memory.as_mut_ptr();
        list.count_work();
        list.defer_write(at.cast(), crate::queue::Written::Value(5), 8);
        list.close();
        let list = std::sync::Arc::new(list);

        ASYNC_IN_TESTS.with(|on| on.set(true));
        host.execute_title(std::slice::from_ref(&list));
        ASYNC_IN_TESTS.with(|on| on.set(false));

        host.wait_in_flight();
        // SAFETY: inside the array.
        assert_eq!(unsafe { at.read_volatile() }, 5);
        let (finished, last) = (
            host.finished_fence.load(Ordering::Acquire),
            list.last_fence.load(Ordering::Acquire),
        );
        assert!(finished >= last, "finished {finished}, list's last fence {last}");
    }

    fn retired_at(submission: u64, frame: u64, age: std::time::Duration) -> Retired {
        Retired {
            submission,
            frame,
            at: std::time::Instant::now() - age,
            bytes: 0,
            object: Box::new(()),
        }
    }

    /// An object leaves the queue only when its submissions finished, the
    /// frames passed and the time passed; time that is not asked for
    /// (pressure) leaves the other two as they are.
    #[test]
    fn retired_objects_wait_for_submissions_and_frames_whatever_the_time() {
        use std::time::Duration;
        let old = Duration::from_secs(60);
        let queue: std::collections::VecDeque<Retired> = [
            retired_at(3, 10, old),
            retired_at(4, 10, Duration::ZERO),
            retired_at(9, 10, old),
        ]
        .into();
        let all_frames = 10 + KEEP_FRAMES;
        // Nothing finished: nothing goes.
        assert_eq!(expired(&queue, 2, all_frames, KEEP_TIME), 0);
        // The first has everything; the second is too young.
        assert_eq!(expired(&queue, 9, all_frames, KEEP_TIME), 1);
        // Without the wait for time, the second goes too, but not the
        // third (its submission is not finished) ...
        assert_eq!(expired(&queue, 8, all_frames, Duration::ZERO), 2);
        // ... nor does an object whose frames have not passed.
        assert_eq!(expired(&queue, 9, all_frames - 1, Duration::ZERO), 0);
        assert_eq!(expired(&queue, 9, all_frames, Duration::ZERO), 3);
    }

    fn retired_object<T: 'static>(object: T) -> Retired {
        Retired {
            submission: 0,
            frame: 0,
            at: std::time::Instant::now(),
            bytes: 0,
            object: Box::new(object),
        }
    }

    /// The wait and the strict mode come from the environment's texts; a
    /// value too large for a duration is cut instead of panicking.
    #[test]
    fn keep_policy_reads_its_settings() {
        use std::time::Duration;
        let default = KeepPolicy::parse(None, None);
        assert_eq!(default, KeepPolicy { time: KEEP_TIME, strict: false });
        assert_eq!(KeepPolicy::parse(Some(" 2.5 "), Some("1")), KeepPolicy { time: Duration::from_millis(2500), strict: true });
        assert_eq!(KeepPolicy::parse(Some("0"), Some("0")).time, Duration::ZERO);
        assert!(!KeepPolicy::parse(None, Some("0")).strict);
        for huge in ["1e300", "1e19", "99999999999", "inf"] {
            let time = KeepPolicy::parse(Some(huge), None).time;
            assert!(time == LONGEST_KEEP || time == KEEP_TIME, "{huge}: {time:?}");
        }
        for bad in ["", "abc", "-3", "NaN"] {
            assert_eq!(KeepPolicy::parse(Some(bad), None).time, KEEP_TIME, "{bad}");
        }
    }

    /// Pressure is asked about only for an object that nothing but the time
    /// keeps: not before its submissions finished or its frames passed.
    #[test]
    fn pressure_is_asked_for_objects_kept_by_time_only() {
        use std::time::Duration;
        let young = retired_at(3, 10, Duration::ZERO);
        let old = retired_at(3, 10, Duration::from_secs(60));
        let frames = 10 + KEEP_FRAMES;
        assert!(waits_for_time_only(&young, 3, frames, KEEP_TIME));
        assert!(!waits_for_time_only(&young, 2, frames, KEEP_TIME), "its submission has not finished");
        assert!(!waits_for_time_only(&young, 3, frames - 1, KEEP_TIME), "its frames have not passed");
        assert!(!waits_for_time_only(&old, 3, frames, KEEP_TIME), "the time has passed");
    }

    /// A name picks an adapter by a part of its description, ignoring case.
    #[test]
    fn adapter_names_match_a_part_ignoring_case() {
        assert!(adapter_matches("AMD Radeon RX 9070 XT", "radeon"));
        assert!(adapter_matches("NVIDIA GeForce RTX 4070", "RTX 4070"));
        assert!(!adapter_matches("AMD Radeon RX 9070 XT", "nvidia"));
        assert!(adapter_matches("Intel(R) UHD Graphics", "intel"));
    }

    /// Confirms the physical adapter backing the live `ID3D12Device` via its
    /// LUID and reports wave range, binding tier, mesh tier, and shader model.
    #[test]
    fn physical_adapter_details_match_dxgi_luid() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let report = probe_device(&host.device);
        eprintln!("physical adapter report: {report:?}");
        assert!(!report.name.is_empty());
        assert_ne!(report.vendor, 0);
        assert!(report.wave_min > 0 && report.wave_max >= report.wave_min);
        assert!(report.shader_model >= D3D_SHADER_MODEL_6_6.0);
        if let Ok(wanted) = std::env::var("VICTORY_ADAPTER") {
            let w = wanted.trim();
            if !w.is_empty() && named_adapter(w).is_ok() {
                assert!(
                    adapter_matches(&report.name, w),
                    "expected adapter matching {w:?}, got {:?}",
                    report.name
                );
            }
        }
    }

    #[test]
    fn warp_adapter_accepts_only_software() {
        // This probes DXGI without creating a D3D12 device or running a title.
        // SAFETY: plain DXGI adapter queries.
        let raw = unsafe {
            let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            else {
                return;
            };
            factory.EnumWarpAdapter::<IDXGIAdapter>()
        };
        let Ok(raw) = raw else { return };
        // SAFETY: EnumWarpAdapter returns a DXGI adapter, which implements
        // IDXGIAdapter1; `raw` owns that interface.
        let desc = unsafe { raw.cast::<IDXGIAdapter1>().unwrap().GetDesc1().unwrap() };
        eprintln!(
            "EnumWarpAdapter: vendor={:#x}, flags={:#x}",
            desc.VendorId, desc.Flags
        );
        assert_eq!(
            warp_adapter().is_ok(),
            desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
        );
    }

    #[test]
    #[ignore = "quarantined per 2026-11-26 incident policy: GPU/WARP re-execution test requires isolated/preflighted execution"]
    fn reexecuting_closed_list_and_unsubmitted_deferred_drop_work_in_sync_and_async() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicUsize;

        struct KeepProbe(Arc<AtomicUsize>);
        impl Drop for KeepProbe {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        struct AsyncReset;
        impl Drop for AsyncReset {
            fn drop(&mut self) {
                ASYNC_IN_TESTS.with(|on| on.set(false));
            }
        }

        let Some(host) = Host::new_warp() else { return };

        // 1. An empty execution returns cleanly without hanging or leaking fence waits.
        host.execute(&[]);

        // 2. An unsubmitted closed list drops its deferred & kept state on drop without running writes.
        let unsubmitted_drops = Arc::new(AtomicUsize::new(0));
        let mut untouched = [0u64; 1];
        {
            let list = host.new_list().expect("unsubmitted list");
            list.keep_until_run(Box::new(KeepProbe(unsubmitted_drops.clone())));
            list.defer_write(untouched.as_mut_ptr().cast(), crate::queue::Written::Value(99), 8);
            list.close();
            assert_eq!(list.last_fence.load(Ordering::Acquire), 0);
            assert_eq!(unsubmitted_drops.load(Ordering::SeqCst), 0);
        }
        assert_eq!(unsubmitted_drops.load(Ordering::SeqCst), 1);
        // SAFETY: isolated local stack word never written by unsubmitted list.
        assert_eq!(unsafe { untouched.as_ptr().read_volatile() }, 0);

        // 3. Re-executing the same closed list synchronously keeps held state until reset/drop.
        let kept_drops = Arc::new(AtomicUsize::new(0));
        let list = Arc::new(host.new_list().expect("reused list"));
        list.keep_until_run(Box::new(KeepProbe(kept_drops.clone())));
        list.count_work();
        list.close();

        host.execute(&[&list]);
        host.execute(&[&list]);
        assert_eq!(kept_drops.load(Ordering::SeqCst), 0);

        // 4. Re-executing the same closed list asynchronously advances last_fence and finishes on wait.
        ASYNC_IN_TESTS.with(|on| on.set(true));
        let _reset_async = AsyncReset;
        host.execute_title(std::slice::from_ref(&list));
        let fence_1 = list.last_fence.load(Ordering::Acquire);
        host.execute_title(std::slice::from_ref(&list));
        let fence_2 = list.last_fence.load(Ordering::Acquire);
        drop(_reset_async);
        assert!(fence_2 > fence_1 && fence_1 > 0, "fences {fence_1} -> {fence_2}");

        host.wait_in_flight();
        assert!(host.finished_fence.load(Ordering::Acquire) >= fence_2);
        list.reset();
        assert_eq!(kept_drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    #[ignore = "quarantined after 2026-11-26 freeze/0xC0000409 incident: 200-cycle WARP handle test aborted intermittently under parallel runs; verify off-host"]
    fn repeated_host_list_and_wait_cycles_do_not_leak_handles() {
        let Some(host) = Host::new_warp() else { return };
        let handle_count = || -> u32 {
            let mut count = 0u32;
            // SAFETY: valid process handle and out pointer.
            let ok = unsafe {
                windows_sys::Win32::System::Threading::GetProcessHandleCount(
                    windows_sys::Win32::System::Threading::GetCurrentProcess(),
                    &mut count,
                )
            };
            assert_ne!(ok, 0);
            count
        };

        // Warm up once before measuring baseline handle count.
        {
            let warmup = host.new_list().expect("warmup list");
            warmup.count_work();
            warmup.close();
            host.execute(&[&warmup]);
            warmup.reset();
        }
        let before = handle_count();
        const CYCLES: usize = 200;
        for _ in 0..CYCLES {
            let list = host.new_list().expect("cycle list");
            list.count_work();
            list.close();
            host.execute(&[&list]);
            // Exercise `wait_for_last_submission`'s temporary WaitForSingleObject event path.
            list.last_fence.store(1, Ordering::Release);
            list.wait_for_last_submission();
            list.reset();
            drop(list);
        }
        let after = handle_count();
        let growth = after.saturating_sub(before);
        assert!(
            growth < 64,
            "handle count grew by {growth} ({before} -> {after}) over {CYCLES} list+wait cycles"
        );
    }

    #[test]
    fn retirement_queues_nested_drops_and_tracks_retired_bytes() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;

        struct Marker(Arc<AtomicBool>);
        impl Drop for Marker {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }
        struct Owner(Option<Marker>, u64);
        impl Drop for Owner {
            fn drop(&mut self) {
                retire_bytes(self.0.take(), self.1);
            }
        }

        let dropped = Arc::new(AtomicBool::new(false));

        let before_count = crate::memstat::RETIRED_COUNT.load(Ordering::Relaxed);
        let before_bytes = crate::memstat::RETIRED_BYTES.load(Ordering::Relaxed);
        drop(retired_object(Owner(
            Some(Marker(dropped.clone())),
            4096,
        )));
        assert!(
            !dropped.load(Ordering::SeqCst),
            "a nested retirement must take its own GPU completion snapshot"
        );
        assert!(crate::memstat::RETIRED_COUNT.load(Ordering::Relaxed) > before_count);
        assert!(crate::memstat::RETIRED_BYTES.load(Ordering::Relaxed) >= before_bytes + 4096);
    }
}

/// A host command list and its allocator, always used together.
pub(crate) struct HostList {
    allocator: ID3D12CommandAllocator,
    pub(crate) list: ID3D12GraphicsCommandList,
    pub(crate) depth_bounds: Option<ID3D12GraphicsCommandList1>,
    /// Work for submission time, in recorded order.
    deferred: Mutex<Deferred>,
    /// Upload memory for data the list's draws make (expanded vertices),
    /// reused after a reset.
    scratch: Mutex<Scratch>,
    /// Host texture states the list assumes, by host copy: at its start and
    /// at its end. Lists are recorded on many threads and run in submission
    /// order, so the transitions from a texture's real state are made at
    /// submission, in `prelude`. Holding the copies keeps every texture the
    /// list names alive until it is reset, even if the title releases the
    /// resource first.
    states: Mutex<States>,
    /// The list that runs those transitions just before this one.
    prelude: (ID3D12CommandAllocator, ID3D12GraphicsCommandList),
    /// What the title's list held for this recording, if the title reset or
    /// released it while a submission of it was still to run.
    kept: Mutex<Vec<Box<dyn std::any::Any + Send>>>,
    /// Draws and dispatches recorded.
    work: std::sync::atomic::AtomicUsize,
    /// The pipelines the list binds, for reports of slow submissions.
    pipelines: Mutex<std::collections::BTreeSet<String>>,
    /// Queries begun in the list and not ended yet: host heap, type, index,
    /// and whether a draw since was not run on the host (`missed_draw`).
    queries: Mutex<Vec<(ID3D12QueryHeap, D3D12_QUERY_TYPE, u32, bool)>>,
    /// The work its indirect executions may still do (`indirect`): a
    /// shader-writable word, zeroed when the list is submitted.
    budget: OnceLock<Option<(ID3D12Resource, usize)>>,
    /// Device-local scratch buffer for clamped indirect arguments (`indirect`),
    /// grown to the largest indirect execution recorded on the list.
    indirect_args: Mutex<Option<(ID3D12Resource, u64)>>,
    /// Host fence value signalled by the last submission that ran this list.
    last_fence: std::sync::atomic::AtomicU64,
    /// The fence of the host that made the list, which `last_fence` counts
    /// on. The process's host is not the only one (tests make their own).
    fence: ID3D12Fence,
    /// Dispatches and buffer copies recorded since the reset: what may have
    /// written indirect arguments (`crate::indirect` clamps hoisted before).
    writes: std::sync::atomic::AtomicU64,
    /// Buffers for clamped indirect arguments when this list holds the
    /// clamps of the list after it (`clamp_room`).
    clamp_chunks: Mutex<ClampChunks>,
    /// Timestamp slot pair for [`crate::cardmeter`] (`u32::MAX` when unstarted).
    meter: std::sync::atomic::AtomicU32,
    /// Tracks whether `close()` finished recording without a subsequent `reset()`.
    closed: std::sync::atomic::AtomicBool,
    /// The texts of the markers `note_op` set, in order (for device-removed
    /// data without marker texts), with the host GPU range a copy reads.
    #[cfg(feature = "diagnostics")]
    notes: Mutex<Vec<Note>>,
    /// With `VICTORY_GPU_TIMING`: a GPU timestamp before each noted command
    /// and at the end, and where they are read back.
    #[cfg(feature = "diagnostics")]
    stamps: OnceLock<Option<(ID3D12QueryHeap, ID3D12Resource)>>,
}

pub(crate) fn warn_depth_bounds() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| tracing::warn!("the host GPU has no depth bounds test; depth bounds are ignored"));
}

/// A list's buffers for hoisted clamped indirect arguments with their
/// states, how many are in use since the reset, the bytes taken of the last
/// of those, whether the list has the barrier that precedes the clamps, and
/// whether it has the clamps' root signature, pipeline and budget bound.
#[derive(Default)]
struct ClampChunks {
    all: Vec<(ID3D12Resource, D3D12_RESOURCE_STATES)>,
    used: usize,
    at: u64,
    barrier: bool,
    bound: bool,
}

/// Timestamps per list (diagnostics); commands past them are not timed.
#[cfg(feature = "diagnostics")]
const STAMPS: u32 = 16384;

impl Drop for HostList {
    fn drop(&mut self) {
        self.wait_for_last_submission();
        crate::cardmeter::release(&self.list);
        crate::memstat::sub(&crate::memstat::LISTS, 1);
        self.release_kept();
    }
}

impl HostList {
    /// Waits until the last submission that ran the list is done, on the
    /// host that ran it: the process's host through its own wait (which also
    /// finishes unwaited submissions), any other on its fence directly.
    fn wait_for_last_submission(&self) {
        use windows::core::Interface;
        let value = self.last_fence.load(Ordering::Acquire);
        if value == 0 {
            return;
        }
        if let Some(Some(host)) = HOST.get()
            && host.fence.as_raw() == self.fence.as_raw()
        {
            host.wait_fence(value);
            return;
        }
        // SAFETY: CreateEventW returns a newly owned handle on success.
        let event = unsafe {
            match CreateEventW(None, false, false, None) {
                Ok(event) => Some(OwnedHandle::from_raw_handle(event.0)),
                Err(error) => {
                    tracing::warn!(%error, value, "cannot create list fence event: polling the fence");
                    None
                }
            }
        };
        wait_for_value(
            &self.fence,
            event.as_ref().map(|event| HANDLE(event.as_raw_handle())),
            value,
        );
    }
}

// SAFETY: D3D12 lists and allocators may be used from any thread, one at
// a time: the title records a list on one thread at a time, and a
// submission runs it after its recording ended (a reset while one is to
// run moves recording to a new list). The rest is locked.
unsafe impl Send for HostList {}
// SAFETY: as above.
unsafe impl Sync for HostList {}

/// A marker's text and the host range its command reads (diagnostics).
#[cfg(feature = "diagnostics")]
type Note = (String, Option<(u64, u64)>);

type States = std::collections::HashMap<
    usize,
    (u32, u32, std::sync::Arc<crate::resource::HostTexture>),
>;

/// Whether device-removed data is on (`VICTORY_D3D12_DRED`, diagnostics
/// builds).
pub(crate) fn dred_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_D3D12_DRED").is_some()
    })
}

/// Whether each list of a submission runs and is timed on its own, slow
/// ones logged with their commands (`VICTORY_GPU_TIMING`, diagnostics
/// builds; the commands are noted with `VICTORY_D3D12_DRED` on).
#[cfg(feature = "diagnostics")]
fn timing() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GPU_TIMING").is_some())
}

/// Notes how long the host GPU took for a submission: how many took how
/// long, and of those of 3 ms and more which pipelines they ran.
#[cfg(feature = "diagnostics")]
fn gpu_waits(gpu: std::time::Duration, runs: &[&HostList]) {
    let mut waits = GPU_WAITS.lock().unwrap_or_else(|e| e.into_inner());
    let us = gpu.as_micros() as u64;
    let bucket = [200, 1000, 3000, 10_000, 30_000].iter().position(|&m| us < m).unwrap_or(5);
    waits.0[bucket].0 += 1;
    waits.0[bucket].1 += us;
    if us >= 3000 {
        let mut pipelines: Vec<String> = runs
            .iter()
            .flat_map(|l| l.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone())
            .collect();
        pipelines.sort_unstable();
        pipelines.dedup();
        let mut key = format!("{} lists {} pipelines:", runs.len(), pipelines.len());
        for p in pipelines.iter().take(12) {
            key.push(' ');
            key.push_str(p.trim_start_matches("[compute ").trim_end_matches(']'));
        }
        let e = waits.1.entry(key).or_insert((0, 0));
        e.0 += 1;
        e.1 += us;
    }
}

#[cfg(feature = "diagnostics")]
#[allow(clippy::type_complexity)]
static GPU_WAITS: Mutex<([(u32, u64); 6], std::collections::BTreeMap<String, (u32, u64)>)> =
    Mutex::new(([(0, 0); 6], std::collections::BTreeMap::new()));

/// Logs and forgets what [`gpu_waits`] noted.
#[cfg(feature = "diagnostics")]
fn log_gpu_waits() {
    let (buckets, slow) = std::mem::take(&mut *GPU_WAITS.lock().unwrap_or_else(|e| e.into_inner()));
    let mut text = String::new();
    for (name, (n, us)) in ["<0.2", "<1", "<3", "<10", "<30", "more"].iter().zip(buckets) {
        text.push_str(&format!(" {name} ms: {n} x, {:.0} ms;", us as f64 / 1e3));
    }
    tracing::info!("GPU waits since:{text}");
    let mut slow: Vec<_> = slow.into_iter().collect();
    slow.sort_unstable_by_key(|(_, (_, us))| std::cmp::Reverse(*us));
    for (key, (n, us)) in slow.into_iter().take(8) {
        tracing::info!("GPU wait {:.0} ms in {n} x: {key}", us as f64 / 1e3);
    }
}

/// With `VICTORY_GPU_TIMING`: GPU milliseconds and commands by what ran
/// (kind and pipeline), and the frames presented meanwhile.
#[cfg(feature = "diagnostics")]
static GPU_TIMES: Mutex<Option<std::collections::HashMap<String, (u32, f64)>>> = Mutex::new(None);
#[cfg(feature = "diagnostics")]
static TIMED_FRAMES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Logs and forgets the GPU time per kind of command, per frame.
#[cfg(feature = "diagnostics")]
fn log_gpu_times() {
    let frames = f64::from(TIMED_FRAMES.swap(0, Ordering::Relaxed).max(1));
    let Some(times) = GPU_TIMES.lock().unwrap_or_else(|e| e.into_inner()).take() else {
        return;
    };
    let total: f64 = times.iter().filter(|(k, _)| k.starts_with("all: ")).map(|(_, t)| t.1).sum();
    let mut all: Vec<_> = times.into_iter().collect();
    all.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
    tracing::info!("GPU time: {:.1} ms/frame in {} kinds of commands", total / frames, all.len());
    for (key, (n, ms)) in all.into_iter().take(45) {
        tracing::info!("GPU time {:7.2} ms/frame, {:7.1} x: {key}", ms / frames, f64::from(n) / frames);
    }
}

/// Whether each noted command waits for those before it (`VICTORY_GPU_STEP`,
/// diagnostics builds), so that a list stops at the command at fault.
#[cfg(feature = "diagnostics")]
fn step() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GPU_STEP").is_some())
}

/// Whether submitted lists are not run on the host GPU (`VICTORY_NO_GPU_WORK`,
/// diagnostics builds).
#[cfg(feature = "diagnostics")]
fn no_work() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_NO_GPU_WORK").is_some())
}

/// Names a host object for the device-removed report, which names the
/// allocations a page fault hit (only with device-removed data on).
pub(crate) fn name(object: &impl Interface, text: impl FnOnce() -> String) {
    if dred_on()
        && let Ok(object) = object.cast::<ID3D12Object>()
    {
        // SAFETY: a plain call on a live object.
        let _ = unsafe { object.SetName(&windows::core::HSTRING::from(text())) };
    }
}

/// Upload chunks handed out front to back: the chunks, the one in use and
/// the bytes used in it.
#[derive(Default)]
struct Scratch {
    chunks: Vec<(ID3D12Resource, usize, u64)>,
    current: usize,
    used: u64,
}

// SAFETY: the chunk pointers are persistent mappings of upload buffers.
unsafe impl Send for Scratch {}

const SCRATCH_CHUNK: u64 = 512 << 10;

/// Scratch chunks no list uses: lists give theirs back when reset. Each list
/// keeping the chunks of its largest use added up to gigabytes, as the title
/// has hundreds of lists.
static SCRATCH_FREE: Mutex<Vec<(ID3D12Resource, usize, u64)>> = Mutex::new(Vec::new());
/// Free chunks kept; more are released.
const SCRATCH_KEPT: usize = 256;

impl Scratch {
    fn give_back(&mut self) {
        (self.current, self.used) = (0, 0);
        let mut free = SCRATCH_FREE.lock().unwrap_or_else(|e| e.into_inner());
        free.append(&mut self.chunks);
        let over = free.len().saturating_sub(SCRATCH_KEPT);
        let over: Vec<_> = free.drain(..over).collect();
        drop(free);
        drop(over);
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if self.chunks.is_empty() {
            return;
        }
        // A submission may still read them.
        let chunks = std::mem::take(&mut self.chunks);
        retire(chunks);
    }
}

/// One piece of submission-time work, in recorded order.
enum Step {
    /// A copy into title memory (destination, source, bytes): the title's
    /// buffer copies, which later refreshes must see.
    Copy(usize, usize, usize),
    /// A copy of title memory into the list's own upload memory.
    Fill(usize, usize, usize),
    /// Brings a range (address, bytes) of a buffer's host copy up to date:
    /// once per submission, whichever lists ask, until a `Copy` or `Run`
    /// intervenes.
    Refresh(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Work that may write title memory.
    Run(Box<dyn Fn() + Send>),
    /// Work that writes host memory only.
    Host(Box<dyn Fn() + Send>),
    /// Shaders write a range (address, bytes) of a buffer's host copy.
    GpuWrite(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Brings a texture's staging buffer up to date and, when dirty, copies
    /// it into the host texture on the list's prelude: once per submission,
    /// whichever lists sample it, until a `Copy` or `Run` intervenes.
    Texture(
        std::sync::Arc<crate::texture::Staging>,
        std::sync::Arc<crate::resource::HostTexture>,
    ),
}

/// A texture-to-buffer readback copied from a mapped readback buffer into
/// title memory once the GPU submission completes.
struct Readback {
    dst: usize,
    src: usize,
    dst_pitch: usize,
    src_pitch: usize,
    lines: usize,
    row_bytes: usize,
}

/// What a list does on the CPU when it is submitted: copies between host
/// memory regions (title buffers, upload buffers and buffer copies), and
/// the upload/readback buffers its GPU commands name, kept until reset.
#[derive(Default)]
struct Deferred {
    /// Destination, source, bytes.
    copies: Vec<Step>,
    /// Post-submission readbacks from GPU textures into title buffers.
    readbacks: Vec<Readback>,
    /// Title textures of kinds copied back (`crate::texback`) written via UAV in this list.
    back_wanted: Vec<usize>,
    /// Readback copies recorded when closing the list, landed into title memory after GPU completion.
    texture_backs: Vec<crate::texback::Back>,
    uploads: Vec<ID3D12Resource>,
    /// Host objects the list's commands name (pipelines, root signatures,
    /// descriptor heaps): the title may release its objects before it
    /// submits the list.
    kept: Vec<windows::core::IUnknown>,
    /// The host pipelines `bind_pipeline` kept (by address), so a pipeline
    /// the list binds again is not kept again.
    bound: std::collections::HashSet<usize>,
    /// Buffer ranges refreshed since the last title copy (host copy,
    /// address, bytes): refreshing them again would find the same bytes.
    refreshed: std::collections::HashSet<(usize, usize, usize)>,
    /// Buffer ranges declared written by shaders since the last title copy
    /// (host copy, address, bytes): declaring them again would change nothing.
    gpu_written: std::collections::HashSet<(usize, usize, usize)>,
    /// Upload memory filled from title memory since the last title write
    /// (source, bytes): its host address, which a second fill would find
    /// holding the same bytes.
    filled: std::collections::HashMap<(usize, usize), u64>,
    /// View heaps the list's draws index directly (`ResourceDescriptorHeap[]`),
    /// and whether any of their shaders may write memory: their slots are
    /// brought up to date once per submission, after every other step.
    bindless: Vec<(std::sync::Arc<crate::descriptor::host::HostHeap>, bool)>,
    /// Textures whose contents the list brings up to date.
    textures: std::collections::HashSet<usize>,
    /// Bottom-of-pipe writes, done once the list has run: address, what,
    /// bytes.
    writes: Vec<(usize, crate::queue::Written, usize)>,
    /// Waits for memory values, with how many `writes` precede them.
    waits: Vec<(usize, crate::memwait::Wait)>,
    /// The work before each of `writes`.
    marks: Vec<usize>,
    /// How many of `writes` this submission has done already.
    early: usize,
    wait_marks: Vec<usize>,
    /// Whether a step zeroes the indirect-work budget.
    budget_zeroed: bool,
}

impl Host {
    /// Whether the host runs mesh shaders at shader model 6.6 (translated
    /// mesh shaders need both); logged once.
    pub(crate) fn mesh_shaders(&self) -> bool {
        static MESH: OnceLock<bool> = OnceLock::new();
        *MESH.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
            let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
                HighestShaderModel: D3D_SHADER_MODEL_6_6,
            };
            // SAFETY: each query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS7,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
                    && self
                        .device
                        .CheckFeatureSupport(
                            D3D12_FEATURE_SHADER_MODEL,
                            (&raw mut model).cast(),
                            size_of_val(&model) as u32,
                        )
                        .is_ok()
            };
            let mesh = ok
                && options.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED
                && model.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_6.0;
            tracing::info!(mesh, "host mesh shader support");
            mesh
        })
    }

    /// The wave sizes the host runs (D3D12 options 1), which translated
    /// compute shaders that declare the console's wave size must fit;
    /// logged once. Unknown (the query fails): any.
    pub(crate) fn target(&self) -> victory_shader::emit::Target {
        static TARGET: OnceLock<victory_shader::emit::Target> = OnceLock::new();
        *TARGET.get_or_init(|| {
            crate::descriptor::host::clamp_shader_indices(&self.device);
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
            // SAFETY: the query writes the structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS1,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
            };
            let target = if ok && options.WaveOps.as_bool() && options.WaveLaneCountMin > 0 {
                victory_shader::emit::Target {
                    lanes_min: options.WaveLaneCountMin,
                    lanes_max: options.WaveLaneCountMax,
                }
            } else {
                victory_shader::emit::Target::ANY
            };
            tracing::info!(
                lanes_min = target.lanes_min,
                lanes_max = target.lanes_max,
                "host wave sizes"
            );
            if !target.runs(64) {
                tracing::warn!(
                    "the host has no 64-lane waves: console wave64 compute programs that read                      other lanes do not translate, and their dispatches are dropped"
                );
            }
            target
        })
    }

    /// A new list, open for recording.
    pub(crate) fn new_list(&self) -> windows::core::Result<HostList> {
        // SAFETY: plain object creation.
        unsafe {
            let allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None,
            )?;
            let prelude_allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let prelude: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &prelude_allocator,
                None,
            )?;
            prelude.Close()?;
            crate::memstat::add(&crate::memstat::LISTS, 1);
            Ok(HostList {
                allocator,
                depth_bounds: self.depth_bounds_supported.then(|| list.cast()).transpose()?,
                list,
                deferred: Mutex::default(),
                scratch: Mutex::default(),
                states: Mutex::default(),
                prelude: (prelude_allocator, prelude),
                kept: Mutex::default(),
                work: Default::default(),
                writes: Default::default(),
                clamp_chunks: Mutex::default(),
                pipelines: Mutex::default(),
                queries: Mutex::default(),
                budget: OnceLock::new(),
                indirect_args: Mutex::default(),
                last_fence: std::sync::atomic::AtomicU64::new(0),
                fence: self.fence.clone(),
                meter: std::sync::atomic::AtomicU32::new(u32::MAX),
                closed: std::sync::atomic::AtomicBool::new(false),
                #[cfg(feature = "diagnostics")]
                notes: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                stamps: OnceLock::new(),
            })
        }
    }

    /// Finishes the unwaited submissions that have run, oldest first, and
    /// buries objects retired before them. The caller holds `submitted`:
    /// finishing puts back what shaders wrote, as a submission does.
    fn reap_completed(&self) {
        // SAFETY: a plain fence query.
        let completed = unsafe { self.fence.GetCompletedValue() };
        let mut last_sub = 0u64;
        loop {
            // The entry stays queued until it is finished: a thread that
            // looks for unfinished submissions meanwhile (a fence about to
            // be signalled) has to wait for this one, whoever finishes it.
            let ready = {
                let mut q = self.in_flight.lock().unwrap_or_else(|e| e.into_inner());
                match q.front_mut() {
                    Some(front) if front.fence <= completed => {
                        Some((front.fence, front.submission, front.finish.take()))
                    }
                    _ => None,
                }
            };
            let Some((fence, submission, finish)) = ready else { break };
            last_sub = last_sub.max(submission);
            if let Some(Finish { lists, back }) = finish {
                let _finishing = Finishing::enter();
                {
                    let runs: Vec<&HostList> = lists.iter().map(|l| &**l).collect();
                    self.fail_if_device_removed(&runs);
                    self.finish_submission(&runs, back);
                    self.log_messages();
                }
                // Before its lists go: a list that goes with it waits for
                // this fence, with `submitted` held here.
                self.finished_fence.fetch_max(fence, Ordering::AcqRel);
                drop(lists);
            } else {
                self.finished_fence.fetch_max(fence, Ordering::AcqRel);
            }
            self.in_flight.lock().unwrap_or_else(|e| e.into_inner()).pop_front();
        }
        if last_sub != 0 {
            FINISHED.fetch_max(last_sub, Ordering::AcqRel);
            bury(last_sub);
        }
    }

    /// Whether a submission is still to be waited for and finished.
    fn has_in_flight(&self) -> bool {
        !finishing_here() && !self.in_flight.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
    }

    /// Waits until the host GPU fence reaches at least `target`.
    pub(crate) fn wait_fence(&self, target: u64) {
        if target == 0 || finishing_here() || self.finished_fence.load(Ordering::Acquire) >= target {
            return;
        }
        let _guard = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        wait_for_value(&self.fence, Some(HANDLE(self.event.as_raw_handle())), target);
        self.reap_completed();
    }

    /// Waits for all in-flight asynchronous submissions to complete.
    pub(crate) fn wait_in_flight(&self) {
        let target = self
            .in_flight
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .back()
            .map(|f| f.fence)
            .unwrap_or(0);
        if target != 0 {
            self.wait_fence(target);
        }
    }

    /// Waits for any in-flight async submission while `self.submitted` is already held.
    fn drain_in_flight_locked(&self) {
        let target = self
            .in_flight
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .back()
            .map(|f| f.fence)
            .unwrap_or(0);
        if target == 0 {
            return;
        }
        // SAFETY: a plain fence query.
        if unsafe { self.fence.GetCompletedValue() } < target {
            let waited = std::time::Instant::now();
            // The fence and event are ours; the caller holds `submitted`.
            wait_for_value(&self.fence, Some(HANDLE(self.event.as_raw_handle())), target);
            submission_us(1, waited.elapsed().as_micros() as u64);
        }
        self.reap_completed();
    }

    /// Runs closed lists; waits synchronously when CPU copy-back, readbacks, or
    /// bottom-of-pipe writes require GPU completion, or returns asynchronously
    /// so CPU recording overlaps GPU execution.
    pub(crate) fn execute(&self, runs: &[&HostList]) {
        self.execute_inner(runs, None);
    }

    /// As [`Host::execute`] for the title's own submissions, the only ones
    /// that may return before the host GPU is done
    /// (`VICTORY_ASYNC_SUBMIT=1`): the title waits for its fences, which are
    /// signalled after the wait (`crate::timeline`). Our own submissions
    /// (present, captures, read-backs, texture uploads) read or free what
    /// they made as soon as the call returns: unwaited, the present's
    /// capture buffer was freed with its copy still pending, which the
    /// debug layer names (#921, object deleted while still in use) and the
    /// AMD driver answered by removing the device at the first frame.
    ///
    /// Such a submission is finished (its shaders' writes put back, its
    /// read-backs and bottom-of-pipe writes done) when it is waited for:
    /// before the next submission's own preparation, a title fence's
    /// signal, a present, or a reset of one of its lists. Meanwhile the
    /// next submission's buffer copies are brought up to date, so the
    /// processor and the card work at the same time for that long.
    pub(crate) fn execute_title(&self, runs: &[std::sync::Arc<HostList>]) {
        let lists: Vec<&HostList> = runs.iter().map(|l| &**l).collect();
        self.execute_inner(&lists, Some(runs));
    }

    /// Brings the buffer copies `runs` will ask for up to date while the
    /// submission before still runs: the same refreshes as the lists'
    /// steps make, early. The ranges that submission's shaders write are
    /// left alone (`Shadow::update` goes around them), and the steps in
    /// their order do it all again after the wait, finding little left.
    fn prewarm(&self, runs: &[&HostList]) {
        let started = std::time::Instant::now();
        let mut batch = Batch::default();
        for list in runs {
            for step in &list.deferred().copies {
                if let Step::Refresh(shadow, address, len) = step {
                    batch.add(shadow, *address, *len);
                }
            }
        }
        batch.flush();
        PREWARM_US.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
    }

    fn execute_inner(&self, runs: &[&HostList], keep: Option<&[std::sync::Arc<HostList>]>) {
        #[cfg(feature = "diagnostics")]
        let _inside = crate::profile::here();
        // One submission at a time from preparation to copy-back: snapshots
        // and the ranges shaders write are per host copy, not per submission.
        let mut submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        // Wait for any previous async submission before mutating shared Shadow
        // buffers in `prepare`. During the gap between submissions, the GPU
        // ran in parallel with the title's CPU command-list recording, and
        // runs on while this submission's buffer copies are refreshed.
        if keep.is_some() && self.has_in_flight() {
            self.prewarm(runs);
        }
        self.drain_in_flight_locked();
        if !runs.is_empty() {
            use windows::core::Interface;
            let host_fence = self.fence.as_raw() as usize;
            // SAFETY: plain completed-value query on live host fence.
            let completed_fence = unsafe { self.fence.GetCompletedValue() };
            let batch_meta: Vec<crate::preflight::ListExecMeta> = runs
                .iter()
                .map(|list| crate::preflight::ListExecMeta {
                    key: list.list.as_raw() as usize,
                    closed: list.closed.load(Ordering::Acquire),
                    host_fence,
                    list_fence: list.fence.as_raw() as usize,
                    last_fence: list.last_fence.load(Ordering::Acquire),
                    completed_fence,
                    live_kept_resources: true,
                })
                .collect();
            if let Err(err) = crate::preflight::validate_submission_batch(&batch_meta) {
                fail(format_args!("host submission rejected by preflight: {err}"));
            }
        }
        let this_submission = STARTED.fetch_add(1, Ordering::AcqRel) + 1;
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::Submissions, 1);
        let mut lists: Vec<Option<ID3D12CommandList>> = Vec::with_capacity(2 * runs.len());
        // Where each run's lists start in `lists`.
        let mut firsts = Vec::with_capacity(runs.len());
        let mut batch = Batch::default();
        let mut early = Early::default();
        if !runs.is_empty() {
            early.take(&mut batch, runs, false);
        }
        for list in runs {
            let uploads = list.run_deferred(&mut batch);
            firsts.push(lists.len());
            if let Some(prelude) = list.prelude(&uploads) {
                lists.push(Some(prelude));
            }
            lists.push(list.list.cast().ok());
        }
        batch.prepare_bindless();
        if !runs.is_empty() {
            early.take(&mut batch, runs, true);
        }
        // The textures live (in `early`) until the submission is done.
        let first = early.record(self);
        // Deduplicate GPU write ranges once before execution so VRAM card
        // shadows get pre-uploaded / Wave64-hashed in the same GPU wait.
        let mut card_writes: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> =
            Vec::new();
        let mut cpu_back: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = Vec::new();
        // A copy is either a card's or not, so one set of ranges serves both.
        let mut declared = std::collections::HashSet::new();
        for list in runs {
            for step in &list.deferred().copies {
                if let Step::GpuWrite(shadow, address, len) = step
                    && declared.insert((std::sync::Arc::as_ptr(shadow) as usize, *address, *len))
                {
                    let target = if shadow.is_card() {
                        &mut card_writes
                    } else {
                        &mut cpu_back
                    };
                    target.push((shadow.clone(), *address, *len));
                }
            }
        }
        for (shadow, address, len) in batch.bindless_writes.drain(..) {
            if declared.insert((std::sync::Arc::as_ptr(&shadow) as usize, address, len)) {
                let target = if shadow.is_card() {
                    &mut card_writes
                } else {
                    &mut cpu_back
                };
                target.push((shadow, address, len));
            }
        }
        let (card_before, card_plan, mut card_after) =
            if self.card.get().and_then(|o| o.as_ref()).is_some() || !card_writes.is_empty() {
                if let Some(engine_lock) = self.card_engine() {
                    let mut engine = engine_lock.lock().unwrap_or_else(|e| e.into_inner());
                    let (before, plan) = engine.record_before(self, &card_writes);
                    let after = if plan.has_writes() {
                        engine.record_after(self, &plan)
                    } else {
                        None
                    };
                    (before, Some(plan), after)
                } else {
                    (None, None, None)
                }
            } else {
                (None, None, None)
            };
        let prepared = std::time::Instant::now();
        // Each run (prelude and list) is a submission of its own: the host
        // resets the device when one submission runs for about 2 s, and
        // the world's frames (a second or more over dozens of lists) fit
        // that only list by list. One wait at the end; with
        // `VICTORY_GPU_TIMING` (diagnostics) each run is waited for and
        // timed on its own.
        let groups: Vec<&[Option<ID3D12CommandList>]> = firsts
            .iter()
            .zip(firsts.iter().skip(1).chain([&lists.len()]))
            .map(|(&a, &b)| &lists[a..b])
            .collect();
        #[cfg(feature = "diagnostics")]
        let each = timing();
        #[cfg(not(feature = "diagnostics"))]
        let each = false;
        let can_async = keep.is_some()
            && !each
            && async_submit()
            && card_writes.is_empty()
            && runs.iter().all(|list| list.deferred().texture_backs.is_empty());
        let mut unsignalled_pre = false;
        if let Some(first) = first {
            let lists = [Some(first)];
            crate::cardmeter::executing(&lists, &self.fence);
            // SAFETY: a closed list.
            unsafe { self.queue.ExecuteCommandLists(&lists) };
            unsignalled_pre = true;
        }
        if let Some(card_before) = card_before {
            let lists = [Some(card_before)];
            crate::cardmeter::executing(&lists, &self.fence);
            // SAFETY: a closed list.
            unsafe { self.queue.ExecuteCommandLists(&lists) };
            unsignalled_pre = true;
        }
        let mut async_signalled = false;
        let mut card_after_ran = false;
        for (i, group) in groups.iter().enumerate() {
            // Diagnostics: everything but the GPU's work, to look at the
            // order of the title's submissions.
            #[cfg(feature = "diagnostics")]
            if no_work() {
                continue;
            }
            *submitted += 1;
            let started = std::time::Instant::now();
            crate::cardmeter::executing(group, &self.fence);
            crate::preflight::record_last_op(format_args!(
                "execute_lists:group_len={}:sub={this_submission}",
                group.len()
            ));
            // SAFETY: closed lists.
            unsafe { self.queue.ExecuteCommandLists(group) };
            if !each && i + 1 < groups.len() {
                continue;
            }
            if !each
                && let Some(after) = card_after.take()
            {
                let lists = [Some(after)];
                crate::cardmeter::executing(&lists, &self.fence);
                // SAFETY: closed list; runs immediately after the final group under the same fence.
                unsafe { self.queue.ExecuteCommandLists(&lists) };
                card_after_ran = true;
            }
            if can_async {
                *submitted += 1;
                let fence_val = *submitted;
                // SAFETY: the fence is ours.
                match unsafe { self.queue.Signal(&self.fence, fence_val) } {
                    Ok(()) => {
                        crate::cardmeter::signalled(&self.fence, fence_val);
                        unsignalled_pre = false;
                        for list in runs {
                            list.last_fence.store(fence_val, Ordering::Release);
                        }
                        async_signalled = true;
                        continue;
                    }
                    // Waited for below like a synchronous submission: its
                    // resources must not be released while the GPU runs.
                    Err(e) => tracing::warn!(error = %e, "host async submission not signalled"),
                }
            }
            self.signal_and_wait_locked(&mut submitted, "host submission not waited for");
            unsignalled_pre = false;
            #[cfg(feature = "diagnostics")]
            if each {
                // SAFETY: a plain query.
                let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
                runs[i].add_times(frequency);
            }
            #[cfg(feature = "diagnostics")]
            if each && started.elapsed() > std::time::Duration::from_millis(50) {
                // SAFETY: a plain query.
                let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
                let commands = runs[i].notes.lock().unwrap_or_else(|e| e.into_inner()).len();
                tracing::warn!(ms = started.elapsed().as_millis(), commands, "slow host list");
                for (ms, op) in runs[i].slowest(frequency, 6) {
                    tracing::warn!("slow list command, {ms:.1} ms: {op}");
                }
            }
            let _ = (i, started);
        }
        if let Some(after) = card_after.take() {
            #[cfg(feature = "diagnostics")]
            let skip = no_work();
            #[cfg(not(feature = "diagnostics"))]
            let skip = false;
            if !skip {
                self.execute_and_wait_locked(after, &mut submitted);
                unsignalled_pre = false;
                card_after_ran = true;
            }
        }
        if unsignalled_pre && !async_signalled {
            self.signal_and_wait_locked(&mut submitted, "host prelude wait failed");
        }
        let _ = &firsts;
        if async_signalled {
            if let Some(plan) = card_plan {
                plan.clear_pending();
            }
            submission_us(0, (prepared - started).as_micros() as u64);
            self.in_flight
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push_back(InFlight {
                    fence: *submitted,
                    submission: this_submission,
                    _early_copies: early.copies,
                    _plans: batch.plans,
                    finish: Some(Finish {
                        lists: keep.map(<[_]>::to_vec).unwrap_or_default(),
                        back: cpu_back,
                    }),
                });
            return;
        }
        drop(early);
        FINISHED.fetch_max(this_submission, Ordering::AcqRel);
        bury(this_submission);
        let gpu = prepared.elapsed();
        submission_us(0, (prepared - started).as_micros() as u64);
        submission_us(1, gpu.as_micros() as u64);
        #[cfg(feature = "diagnostics")]
        gpu_waits(gpu, runs);
        // The host resets a device whose submission runs for about 2 s.
        if gpu > std::time::Duration::from_millis(250) {
            let pipelines: std::collections::BTreeSet<String> = runs
                .iter()
                .flat_map(|l| l.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone())
                .collect();
            tracing::warn!(ms = gpu.as_millis(), lists = runs.len(), ?pipelines, "slow host submission");
        }
        self.fail_if_device_removed(runs);
        // What shaders wrote goes back before the bottom-of-pipe writes,
        // which tell the title the GPU is done; each range once per
        // submission, however many dispatches declared it (the world's
        // passes declare the same big ranges dozens of times).
        let copying = std::time::Instant::now();
        if let Some(plan) = card_plan {
            if card_after_ran {
                if let Some(engine_lock) = self.card_engine() {
                    let mut engine = engine_lock.lock().unwrap_or_else(|e| e.into_inner());
                    let written = engine.finish_back(self, plan, &mut submitted);
                    copy_back(1, written as u64);
                } else {
                    plan.clear_pending();
                }
            } else {
                plan.clear_pending();
            }
        }
        COPY_BACK[0].fetch_add(copying.elapsed().as_micros() as u64, Ordering::Relaxed);
        self.finish_submission(runs, cpu_back);
        self.finished_fence.fetch_max(*submitted, Ordering::AcqRel);
        self.log_messages();
    }

    /// Stops the process with the removal data when the host device was
    /// removed while `runs` ran.
    fn fail_if_device_removed(&self, runs: &[&HostList]) {
        // SAFETY: a plain query.
        if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
            fail(format_args!(
                "host GPU device removed after a submission of {} lists{}{}",
                runs.len(),
                self.removal_data(runs),
                last_notes(runs)
            ));
        }
    }
    /// After a submission ran: what its shaders wrote goes back into title
    /// memory, then its read-backs, then the bottom-of-pipe writes that
    /// tell the title the GPU is done.
    fn finish_submission(
        &self,
        runs: &[&HostList],
        back: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    ) {
        let copying = std::time::Instant::now();
        if back.len() <= 1 {
            for (shadow, address, len) in &back {
                let written = shadow.copy_back(*address, *len);
                copy_back(1, written as u64);
            }
        } else {
            // Group by Shadow pointer so distinct shadows run in parallel
            // across CPU cores without lock contention on `before`/`pending`.
            let mut groups: Vec<(std::sync::Arc<crate::shadow::Shadow>, Vec<_>)> = Vec::new();
            let mut indices = std::collections::HashMap::<_, usize>::new();
            for (shadow, address, len) in back {
                // Sparse aliases share one snapshot and pending queue.
                let owner = shadow.copy_back_owner();
                let index = *indices.entry(std::sync::Arc::as_ptr(owner)).or_insert_with(|| {
                    groups.push((owner.clone(), Vec::new()));
                    groups.len() - 1
                });
                groups[index].1.push((address, address + len));
            }
            for (_, ranges) in &mut groups {
                crate::shadow::merge_byte_ranges(ranges);
            }
            in_parallel(&groups, 8, |chunk| {
                let mut written = 0;
                for (shadow, ranges) in chunk {
                    for &(start, end) in ranges {
                        written += shadow.copy_back(start, end - start) as u64;
                    }
                }
                copy_back(1, written);
            });
        }
        for list in runs {
            for back in &list.deferred().texture_backs {
                crate::texback::land(back);
            }
            for rb in &list.deferred().readbacks {
                let written = if rb.dst_pitch == rb.row_bytes && rb.src_pitch == rb.row_bytes {
                    let bytes = rb.row_bytes * rb.lines;
                    crate::watch::writing(rb.dst, bytes);
                    write_mapped(rb.dst as *mut u8, rb.src as *const u8, bytes);
                    bytes
                } else {
                    let mut n = 0;
                    for line in 0..rb.lines {
                        let to = rb.dst + line * rb.dst_pitch;
                        let from = rb.src + line * rb.src_pitch;
                        crate::watch::writing(to, rb.row_bytes);
                        write_mapped(to as *mut u8, from as *const u8, rb.row_bytes);
                        n += rb.row_bytes;
                    }
                    n
                };
                copy_back(1, written as u64);
            }
        }
        copy_back(0, copying.elapsed().as_micros() as u64);
        // Host timestamps the lists resolved, on the title's GPU clock.
        let mut clock = None;
        for list in runs {
            let mut deferred = list.deferred();
            for &(at, what, bytes) in &deferred.writes[deferred.early..] {
                let what = match what {
                    crate::queue::Written::HostTimestamp(raw) => match clock.get_or_insert_with(|| self.clock()) {
                        // SAFETY: the list's mapped read-back memory, which
                        // the submission that ran resolved into.
                        Some(c) => crate::queue::Written::Value(c.convert(unsafe { (raw as *const u64).read_volatile() })),
                        None => crate::queue::Written::GpuTimestamp,
                    },
                    what => what,
                };
                what.write(at as *mut u8, bytes);
            }
            deferred.early = 0;
        }
    }

    /// How host GPU timestamps map onto the title's GPU clock
    /// ([`crate::queue::Written::GpuTimestamp`]): the queue's calibration
    /// of its timestamps against the CPU's performance counter, which
    /// `Instant` reads too.
    fn clock(&self) -> Option<Clock> {
        use windows_sys::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};
        let (mut gpu, mut cpu, mut now, mut cpu_frequency) = (0, 0, 0, 0);
        // SAFETY: plain queries into our locals.
        let frequency = unsafe {
            let frequency = self.queue.GetTimestampFrequency().ok()?;
            self.queue.GetClockCalibration(&mut gpu, &mut cpu).ok()?;
            QueryPerformanceCounter(&mut now);
            QueryPerformanceFrequency(&mut cpu_frequency);
            frequency
        };
        let title_now = crate::queue::gpu_clock();
        let ticks = i128::from(crate::queue::TIMESTAMP_FREQUENCY);
        // The title's clock when the host GPU's read `gpu`.
        let since = (i128::from(now) - i128::from(cpu)) * ticks / i128::from(cpu_frequency.max(1));
        (frequency > 0).then_some(Clock { gpu, at: i128::from(title_now) - since, frequency, now: title_now })
    }

    /// With `VICTORY_D3D12_DRED`: the faulting address and the allocations
    /// it lies in (by the names we give them) or the title memory copy just
    /// below it; for each list that did not finish, the command it stopped
    /// at and what the draws and dispatches in flight there run (or, in
    /// lists that never started, the pipelines they bind).
    fn removal_data(&self, runs: &[&HostList]) -> String {
        let Ok(dred) = self.device.cast::<ID3D12DeviceRemovedExtendedData1>() else {
            return String::new();
        };
        let mut text = String::new();
        let mut fault_va = None;
        // SAFETY: the outputs are the runtime's, valid while the device
        // lives; nodes are walked as linked, histories have their counts.
        unsafe {
            if let Ok(fault) = dred.GetPageFaultAllocationOutput1()
                && fault.PageFaultVA != 0
            {
                let va = fault.PageFaultVA;
                fault_va = Some(va);
                text.push_str(&format!("; page fault at {va:#x}"));
                for (what, head) in [
                    ("inside", fault.pHeadExistingAllocationNode),
                    ("inside freed", fault.pHeadRecentFreedAllocationNode),
                ] {
                    let mut names = Vec::new();
                    let mut node = head;
                    while let Some(n) = node.as_ref()
                        && names.len() < 8
                    {
                        let name = if n.ObjectNameW.is_null() {
                            "unnamed".to_owned()
                        } else {
                            n.ObjectNameW.to_string().unwrap_or_default()
                        };
                        names.push(format!("{name} (type {})", n.AllocationType.0));
                        node = n.pNext;
                    }
                    if !names.is_empty() {
                        text.push_str(&format!("; {what} {names:?}"));
                    }
                }
                text.push_str(&crate::shadow::near(va));
                text.push_str(&crate::descriptor::host::near_heaps(va));
            }
            // The plain output: the extended one (with marker texts) came
            // back without stopped lists on this host.
            let crumbs = match dred.GetAutoBreadcrumbsOutput() {
                Ok(crumbs) => crumbs,
                Err(e) => {
                    text.push_str(&format!("; no breadcrumbs ({e})"));
                    return text;
                }
            };
            // Every list's progress (done/count), in the runtime's order.
            let mut progress = Vec::new();
            let mut node = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = node.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                let ours = p.pCommandList.as_ref().map(|l| l.as_raw());
                let index = runs.iter().position(|r| Some(r.list.as_raw()) == ours);
                progress.push(format!("{done}/{} {index:?}", p.BreadcrumbCount));
                node = p.pNext;
            }
            text.push_str(&format!("; breadcrumbs (done/count, list) {progress:?}"));
            let mut plain = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = plain.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                if done < p.BreadcrumbCount && !p.pCommandHistory.is_null() {
                    let history =
                        std::slice::from_raw_parts(p.pCommandHistory, p.BreadcrumbCount as usize);
                    text.push_str(&format!(
                        "; a list stopped at command {done} of {} (op {})",
                        p.BreadcrumbCount, history[done as usize].0
                    ));
                    let list = p.pCommandList.as_ref().map(|l| l.as_raw());
                    let run = runs.iter().find(|r| Some(r.list.as_raw()) == list);
                    // The k-th marker command is the k-th note.
                    #[cfg(feature = "diagnostics")]
                    let markers: Vec<(u32, String)> = run.map_or_else(Vec::new, |r| {
                        let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
                        history
                            .iter()
                            .enumerate()
                            .filter(|(_, op)| **op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
                            .zip(notes.iter())
                            .map(|((at, _), (note, range))| {
                                let place = fault_va.zip(*range).map(|(va, r)| fault_in(va, r));
                                (at as u32, format!("{note}{}", place.unwrap_or_default()))
                            })
                            .collect()
                    });
                    #[cfg(not(feature = "diagnostics"))]
                    let markers: Vec<(u32, String)> = Vec::new();
                    #[cfg(not(feature = "diagnostics"))]
                    let _ = fault_va;
                    if done > 0 {
                        text.push_str(&in_flight(history, &markers, done as usize));
                    } else if let Some(r) = run {
                        let pipelines = r.pipelines.lock().unwrap_or_else(|e| e.into_inner());
                        text.push_str(&format!(", its pipelines {pipelines:?}"));
                    }
                }
                plain = p.pNext;
            }
        }
        text
    }

    /// Moves the debug layer's messages into our log; returns the first
    /// error among them.
    pub(crate) fn log_messages(&self) -> Option<String> {
        let Some(queue) = &self.messages else {
            return None;
        };
        let mut first_error = None;
        // SAFETY: each message is read into a buffer of the size the queue
        // reports, then the queue is emptied.
        unsafe {
            for i in 0..queue.GetNumStoredMessages() {
                let mut size = 0;
                if queue.GetMessage(i, None, &mut size).is_err() || size == 0 {
                    continue;
                }
                let mut buffer = vec![0u64; size.div_ceil(8)];
                let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if queue.GetMessage(i, Some(message), &mut size).is_ok() {
                    let m = &*message;
                    let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                    let text = String::from_utf8_lossy(text);
                    let text = text.trim_end_matches('\0');
                    tracing::warn!(id = m.ID.0, severity = m.Severity.0, "D3D12: {text}");
                    if m.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 && first_error.is_none() {
                        first_error = Some(text.to_owned());
                    }
                }
            }
            queue.ClearStoredMessages();
        }
        first_error
    }

    /// Logs and drains accumulated D3D12 debug layer messages during live runs.
    pub(crate) fn drain_messages(&self) {
        let Some(queue) = &self.messages else {
            return;
        };
        static SEEN: Mutex<std::collections::BTreeMap<i32, (u64, i32)>> =
            Mutex::new(std::collections::BTreeMap::new());
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: each stored message is read into a buffer of its reported size, then cleared.
        let stored = unsafe {
            let stored = queue.GetNumStoredMessages();
            for i in 0..stored {
                let mut size = 0;
                if queue.GetMessage(i, None, &mut size).is_err() || size == 0 {
                    continue;
                }
                let mut buffer = vec![0u64; size.div_ceil(8)];
                let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if queue.GetMessage(i, Some(message), &mut size).is_ok() {
                    let m = &*message;
                    let kind = seen.entry(m.ID.0).or_insert((0, m.Severity.0));
                    kind.0 += 1;
                    if kind.0 <= 5 {
                        let text =
                            std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                        let text = String::from_utf8_lossy(text);
                        let text = text.trim_end_matches('\0');
                        tracing::warn!(
                            id = m.ID.0,
                            severity = m.Severity.0,
                            nth = kind.0,
                            "D3D12 debug layer: {text}"
                        );
                    }
                }
            }
            queue.ClearStoredMessages();
            stored
        };
        if stored > 0 {
            let counts: Vec<String> = seen
                .iter()
                .map(|(id, (n, severity))| format!("id {id} (severity {severity}) x{n}"))
                .collect();
            tracing::info!(
                "D3D12 debug layer messages so far by kind: {}",
                counts.join(", ")
            );
        }
    }

    pub(crate) fn device(&self) -> &ID3D12Device {
        &self.device
    }

    pub(crate) fn timestamp_frequency(&self) -> u64 {
        // SAFETY: plain D3D12 timestamp frequency query on our direct queue.
        unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(0)
    }

    /// A host render target view of `resource`, valid until it is recorded.
    pub(crate) fn rtv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_RENDER_TARGET_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_rtv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.rtvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.rtv_size) as usize;
            self.device
                .CreateRenderTargetView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }

    /// A host depth view of `resource`, valid until it is recorded.
    pub(crate) fn dsv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_dsv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.dsvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.dsv_size) as usize;
            self.device
                .CreateDepthStencilView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }
}

/// A host GPU timestamp `gpu` taken when the title's GPU clock read `at`,
/// the host's ticks per second, and the title's clock when taken.
struct Clock {
    gpu: u64,
    at: i128,
    frequency: u64,
    now: u64,
}

impl Clock {
    /// A host GPU timestamp on the title's clock: never past the time it
    /// was read, so bottom-of-pipe timestamps written after stay later.
    fn convert(&self, raw: u64) -> u64 {
        let ticks = i128::from(crate::queue::TIMESTAMP_FREQUENCY);
        let since = (i128::from(raw) - i128::from(self.gpu)) * ticks / i128::from(self.frequency);
        (self.at + since).clamp(0, i128::from(self.now)) as u64
    }
}

impl HostList {
    /// The timestamp heap and its read-back buffer (diagnostics, timing on).
    #[cfg(feature = "diagnostics")]
    fn stamps(&self) -> Option<&(ID3D12QueryHeap, ID3D12Resource)> {
        if !timing() {
            return None;
        }
        self.stamps
            .get_or_init(|| {
                let device = &host()?.device;
                let mut heap: Option<ID3D12QueryHeap> = None;
                let mut back: Option<ID3D12Resource> = None;
                let readback = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: u64::from(STAMPS) * 8,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                };
                // SAFETY: plain object creation.
                unsafe {
                    device
                        .CreateQueryHeap(
                            &D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: STAMPS, NodeMask: 0 },
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
                }
                Some((heap?, back?))
            })
            .as_ref()
    }

    /// The noted commands that took longest on the GPU, longest first, as
    /// (milliseconds, text) (diagnostics, timing on, after the list ran).
    #[cfg(feature = "diagnostics")]
    fn slowest(&self, frequency: u64, n: usize) -> Vec<(f64, String)> {
        let notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
        let (Some((_, back)), count) = (self.stamps(), notes.len().min(STAMPS as usize - 1)) else {
            return Vec::new();
        };
        let mut data = std::ptr::null_mut();
        let range = D3D12_RANGE { Begin: 0, End: (count + 1) * 8 };
        // SAFETY: a read-back buffer the finished list resolved into.
        let stamps: Vec<u64> = unsafe {
            if back.Map(0, Some(&range), Some(&mut data)).is_err() {
                return Vec::new();
            }
            let s = std::slice::from_raw_parts(data.cast::<u64>(), count + 1).to_vec();
            back.Unmap(0, None);
            s
        };
        let mut times: Vec<(f64, String)> = (0..count)
            .map(|i| {
                let ticks = stamps[i + 1].saturating_sub(stamps[i]);
                (ticks as f64 * 1e3 / frequency.max(1) as f64, notes[i].0.chars().take(300).collect())
            })
            .collect();
        times.sort_by(|a, b| b.0.total_cmp(&a.0));
        times.truncate(n);
        times
    }

    /// Adds the GPU time of each noted command to the totals by kind
    /// (diagnostics, timing on, after the list ran).
    #[cfg(feature = "diagnostics")]
    fn add_times(&self, frequency: u64) {
        let notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
        let (Some((_, back)), count) = (self.stamps(), notes.len().min(STAMPS as usize - 1)) else {
            return;
        };
        if count == 0 {
            return;
        }
        let mut data = std::ptr::null_mut();
        let range = D3D12_RANGE { Begin: 0, End: (count + 1) * 8 };
        // SAFETY: a read-back buffer the finished list resolved into.
        let stamps: Vec<u64> = unsafe {
            if back.Map(0, Some(&range), Some(&mut data)).is_err() {
                return;
            }
            let s = std::slice::from_raw_parts(data.cast::<u64>(), count + 1).to_vec();
            back.Unmap(0, None);
            s
        };
        let mut times = GPU_TIMES.lock().unwrap_or_else(|e| e.into_inner());
        let times = times.get_or_insert_with(Default::default);
        for i in 0..count {
            let ms = stamps[i + 1].saturating_sub(stamps[i]) as f64 * 1e3 / frequency.max(1) as f64;
            // The kind without its counts: "dispatch [compute ...]".
            let text = &notes[i].0;
            let kind = text.split(' ').next().unwrap_or("");
            let rest = text.find('[').map_or("", |at| &text[at..]);
            let key: String = format!("{kind} {rest}").chars().take(150).collect();
            let k = times.entry(format!("all: {kind}")).or_insert((0, 0.0));
            k.0 += 1;
            k.1 += ms;
            let e = times.entry(key).or_insert((0, 0.0));
            e.0 += 1;
            e.1 += ms;
        }
    }

    pub(crate) fn close(&self) {
        // The host closes no list with queries running; their title ends,
        // in a later list, are dropped (`end_query`).
        for (heap, kind, index, _) in self.queries.lock().unwrap_or_else(|e| e.into_inner()).drain(..) {
            // SAFETY: an open list and a query begun in it.
            unsafe { self.list.EndQuery(&heap, kind, index) };
        }
        #[cfg(feature = "diagnostics")]
        if let Some((heap, back)) = self.stamps() {
            let count = (self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32).min(STAMPS - 1);
            // SAFETY: an open list, our heap and read-back buffer.
            unsafe {
                self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, count);
                self.list.ResolveQueryData(heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, count + 1, back, 0);
            }
        }
        self.textures_back();
        crate::cardmeter::end(
            &self.list,
            crate::cardmeter::Span::from_raw(
                self.meter.swap(u32::MAX, std::sync::atomic::Ordering::Relaxed),
            ),
        );
        // SAFETY: our list.
        if let Err(e) = unsafe { self.list.Close() } {
            fail(format_args!(
                "a host command list did not close ({e}): a command recorded into it was invalid"
            ));
        }
        self.closed.store(true, Ordering::Release);
    }

    pub(crate) fn keep_until_run(&self, what: Box<dyn std::any::Any + Send>) {
        self.kept.lock().unwrap_or_else(|e| e.into_inner()).push(what);
    }

    pub(crate) fn release_kept(&self) {
        let kept = std::mem::take(&mut *self.kept.lock().unwrap_or_else(|e| e.into_inner()));
        drop(kept);
    }

    /// Only after the list's last submission finished.
    pub(crate) fn reset(&self) {
        self.wait_for_last_submission();
        self.release_kept();
        *self.deferred() = Deferred::default();
        self.work.store(0, Ordering::Relaxed);
        self.writes.store(0, Ordering::Relaxed);
        {
            let mut chunks = self.clamp_chunks.lock().unwrap_or_else(|e| e.into_inner());
            chunks.used = 0;
            chunks.at = 0;
            chunks.barrier = false;
            chunks.bound = false;
        }
        self.states.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.queries.lock().unwrap_or_else(|e| e.into_inner()).clear();
        #[cfg(feature = "diagnostics")]
        self.notes.lock().unwrap_or_else(|e| e.into_inner()).clear();
        {
            let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
            scratch.give_back();
        }
        // SAFETY: the list is not executing.
        let reset = unsafe {
            self.allocator
                .Reset()
                .and_then(|()| self.list.Reset(&self.allocator, None))
        };
        if let Err(e) = reset {
            fail(format_args!("a host command list did not reset ({e})"));
        }
        self.closed.store(false, Ordering::Release);
        self.meter.store(
            crate::cardmeter::begin(&self.list, crate::cardmeter::Kind::Lists).raw(),
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    /// Moves a title texture's host copy to `wanted` (a host state) in this
    /// list: the host texture and the state it leaves, if that is a change.
    /// The list's first use of a texture records nothing: its prelude brings
    /// the texture there at submission.
    pub(crate) fn transition(
        &self,
        resource: crate::object::This,
        wanted: u32,
    ) -> Option<(&'static ID3D12Resource, u32)> {
        let texture = crate::resource::host_resource(resource)?;
        let copy = crate::resource::host_copy(resource)?;
        self.transition_copy(&copy, wanted).map(|before| (texture, before))
    }

    /// As [`HostList::transition`], for a host copy held by the caller (the
    /// title's object may be gone): the state it leaves, if that is a change.
    pub(crate) fn transition_copy(
        &self,
        copy: &std::sync::Arc<crate::resource::HostTexture>,
        wanted: u32,
    ) -> Option<u32> {
        let wanted = legal_state(&copy.texture, wanted);
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        match states.entry(std::sync::Arc::as_ptr(copy) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert((wanted, wanted, copy.clone()));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().1, wanted);
                (before != wanted).then_some(before)
            }
        }
    }

    /// Records that this list starts and ends with `copy` in `wanted` state
    /// (test helper matching `transition`'s first-use tracking).
    #[cfg(test)]
    pub(crate) fn expect_texture_state(
        &self,
        copy: &std::sync::Arc<crate::resource::HostTexture>,
        wanted: u32,
    ) {
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        states
            .entry(std::sync::Arc::as_ptr(copy) as usize)
            .or_insert_with(|| (wanted, wanted, copy.clone()));
    }

    #[cfg(test)]
    pub(crate) fn has_deferred_work(&self) -> bool {
        !self.deferred().copies.is_empty()
    }

    /// At submission: the list that copies any newly dirtied textures from
    /// their staging buffers into their host textures and moves each texture
    /// this list uses from its real state to the one the list starts with;
    /// the textures' real states become those the list ends with.
    fn prelude(
        &self,
        uploads: &[(
            std::sync::Arc<crate::texture::Staging>,
            std::sync::Arc<crate::resource::HostTexture>,
        )],
    ) -> Option<ID3D12CommandList> {
        const COPY_DEST: u32 = D3D12_RESOURCE_STATE_COPY_DEST.0 as u32;
        let mut pre_barriers = Vec::new();
        for (_, copy) in uploads {
            let real = copy.state.swap(COPY_DEST, Ordering::Relaxed);
            if real != COPY_DEST {
                pre_barriers.push(transition(
                    &copy.texture,
                    D3D12_RESOURCE_STATES(real as i32),
                    D3D12_RESOURCE_STATE_COPY_DEST,
                ));
            }
        }
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        let post_barriers: Vec<D3D12_RESOURCE_BARRIER> = states
            .values()
            .filter_map(|(first, last, copy)| {
                let real = copy.state.swap(*last, Ordering::Relaxed);
                (real != *first).then(|| {
                    transition(
                        &copy.texture,
                        D3D12_RESOURCE_STATES(real as i32),
                        D3D12_RESOURCE_STATES(*first as i32),
                    )
                })
            })
            .collect();
        if uploads.is_empty() && post_barriers.is_empty() {
            return None;
        }
        let validate_barriers = |barriers: &[D3D12_RESOURCE_BARRIER]| {
            use windows::core::Interface;
            let checks: Vec<crate::preflight::BarrierTransition> = barriers
                .iter()
                .filter(|b| b.Type == D3D12_RESOURCE_BARRIER_TYPE_TRANSITION)
                .map(|b| {
                    // SAFETY: `Type == TRANSITION` guarantees `Anonymous.Transition` is active.
                    let t = unsafe { &b.Anonymous.Transition };
                    crate::preflight::BarrierTransition {
                        resource: t.pResource.as_ref().map_or(0, |r| r.as_raw() as usize),
                        subresource: t.Subresource,
                        before: t.StateBefore.0 as u32,
                        after: t.StateAfter.0 as u32,
                    }
                })
                .collect();
            if let Err(err) = crate::preflight::validate_resource_barriers(&checks) {
                fail(format_args!("host prelude barrier rejected by preflight: {err}"));
            }
        };
        validate_barriers(&pre_barriers);
        validate_barriers(&post_barriers);
        let (allocator, list) = &self.prelude;
        // SAFETY: the prelude is not executing (every submission is waited
        // for); the textures and staging buffers are live.
        let recorded = unsafe {
            allocator.Reset().and_then(|()| {
                list.Reset(allocator, None)?;
                if !pre_barriers.is_empty() {
                    list.ResourceBarrier(&pre_barriers);
                }
                for (staging, copy) in uploads {
                    staging.record_copies(list, &copy.texture);
                }
                if !post_barriers.is_empty() {
                    list.ResourceBarrier(&post_barriers);
                }
                list.Close()
            })
        };
        if let Err(e) = recorded {
            fail(format_args!("a host state prelude did not record ({e})"));
        }
        list.cast().ok()
    }

    /// Notes what the command about to be recorded does, as a marker
    /// command whose text the list keeps (diagnostics, with device-removed
    /// data on): the report pairs the breadcrumbs' marker commands with the
    /// texts in order and names the commands a fault stopped among.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_op(&self, what: impl FnOnce() -> String) {
        self.note(what, None);
    }

    /// As [`HostList::note_op`], for a copy that reads the host GPU range
    /// `(address, bytes)`: the report places the faulting address in it.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_read(&self, what: impl FnOnce() -> String, range: (u64, u64)) {
        self.note(what, Some(range));
    }

    #[cfg(feature = "diagnostics")]
    fn note(&self, what: impl FnOnce() -> String, range: Option<(u64, u64)>) {
        if dred_on() {
            if step() {
                let barrier = D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                    ..Default::default()
                };
                // SAFETY: an open list; the barrier names no resource.
                unsafe { self.list.ResourceBarrier(&[barrier]) };
            }
            let at = self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32;
            if let Some((heap, _)) = self.stamps() && at < STAMPS - 1 {
                // SAFETY: an open list and our query heap.
                unsafe { self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, at) };
            }
            let what = what();
            let text: Vec<u16> = what.encode_utf16().chain([0]).collect();
            const UNICODE: u32 = 0;
            // SAFETY: an open list; the marker is copied when recorded.
            unsafe {
                self.list.SetMarker(UNICODE, Some(text.as_ptr().cast()), 2 * text.len() as u32)
            };
            self.notes
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((what, range));
        }
    }

    /// Begins query `index` of `heap` (occlusion or statistics, never a
    /// timestamp) unless it is running in the list already. Whether it
    /// began.
    pub(crate) fn begin_query(&self, heap: &ID3D12QueryHeap, kind: u32, index: u32) -> bool {
        let mut queries = self.queries.lock().unwrap_or_else(|e| e.into_inner());
        if kind == crate::query::TIMESTAMP || queries.iter().any(|(h, _, i, _)| h == heap && *i == index) {
            return false;
        }
        let kind = D3D12_QUERY_TYPE(kind as i32);
        // SAFETY: an open list and a live heap, kept until the list is reset.
        unsafe { self.list.BeginQuery(heap, kind, index) };
        queries.push((heap.clone(), kind, index, false));
        drop(queries);
        self.keep(heap);
        true
    }

    /// Ends query `index` of `heap`: a timestamp is taken; another query
    /// ends only if it began in this list, as the type it began as. The
    /// type it ended as, if it did and its result is the host's to give: an
    /// occlusion query with a draw inside that the host did not run counts
    /// too few samples, and a title would cull by it what the draw shows.
    pub(crate) fn end_query(&self, heap: &ID3D12QueryHeap, kind: u32, index: u32) -> Option<u32> {
        let (kind, missed) = if kind == crate::query::TIMESTAMP {
            (D3D12_QUERY_TYPE(kind as i32), false)
        } else {
            let mut queries = self.queries.lock().unwrap_or_else(|e| e.into_inner());
            let at = queries.iter().position(|(h, _, i, _)| h == heap && *i == index)?;
            let (_, kind, _, missed) = queries.swap_remove(at);
            (kind, missed)
        };
        // SAFETY: an open list and a live heap, kept until the list is reset.
        unsafe { self.list.EndQuery(heap, kind, index) };
        self.keep(heap);
        let occlusion = matches!(kind.0 as u32, crate::query::OCCLUSION | crate::query::BINARY_OCCLUSION);
        (!(missed && occlusion)).then_some(kind.0 as u32)
    }

    /// A draw the title recorded was not run on the host (no pipeline yet,
    /// a shader that does not translate, ...): the queries running around
    /// it did not see it.
    pub(crate) fn missed_draw(&self) {
        for q in self.queries.lock().unwrap_or_else(|e| e.into_inner()).iter_mut() {
            q.3 = true;
        }
    }

    /// Notes a pipeline the list binds and keeps it and its root signature
    /// until the list is reset, the first time only: the list keeps the
    /// first, so its address names the same pipeline from then on.
    pub(crate) fn bind_pipeline(&self, p: &crate::pipeline::host::HostPipeline) {
        let mut deferred = self.deferred();
        if !deferred.bound.insert(p.pso.as_raw() as usize) {
            return;
        }
        deferred.kept.push(p.pso.clone().into());
        deferred.kept.push(p.root.clone().into());
        drop(deferred);
        self.pipelines
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(p.name.clone());
    }

    fn deferred(&self) -> std::sync::MutexGuard<'_, Deferred> {
        self.deferred.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Copies `len` bytes from `src` to `dst` when the list is submitted.
    pub(crate) fn defer_copy(&self, dst: *mut u8, src: *const u8, len: usize) {
        let mut deferred = self.deferred();
        deferred
            .copies
            .push(Step::Copy(dst as usize, src as usize, len));
        deferred.refreshed.clear();
        deferred.gpu_written.clear();
        deferred.filled.clear();
    }

    /// Copies `len` bytes of title memory at `src` into the list's upload
    /// memory at `dst` when the list is submitted.
    pub(crate) fn defer_fill(&self, dst: *mut u8, src: *const u8, len: usize) {
        self.deferred()
            .copies
            .push(Step::Fill(dst as usize, src as usize, len));
    }

    /// The host GPU address of `len` bytes of upload memory holding the title
    /// memory at `src` as of the list's submission: one copy for every ask
    /// until a title copy intervenes.
    pub(crate) fn fill_scratch(&self, src: *const u8, len: u64) -> Option<u64> {
        let key = (src as usize, len as usize);
        let known = self.deferred().filled.get(&key).copied();
        if known.is_some() {
            return known;
        }
        let (gpu, to) = self.scratch(len)?;
        self.defer_fill(to, src, len as usize);
        self.deferred().filled.insert(key, gpu);
        Some(gpu)
    }

    /// Brings `[address, address + len)` of a buffer's host copy up to date
    /// when the list is submitted (once per range until a title copy
    /// intervenes), keeping the copy alive until the list is reset.
    pub(crate) fn defer_refresh(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(shadow) as usize, address, len);
        if deferred.refreshed.insert(key) {
            deferred
                .copies
                .push(Step::Refresh(shadow.clone(), address, len));
        }
    }

    /// Shaders of the list write `[address, address + len)` of a buffer's
    /// host copy: copied back into title memory once the list has run.
    pub(crate) fn defer_gpu_write(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(shadow) as usize, address, len);
        if deferred.gpu_written.insert(key) {
            deferred
                .copies
                .push(Step::GpuWrite(shadow.clone(), address, len));
        }
    }

    /// A wait for a memory value.
    pub(crate) fn note_wait(&self, wait: crate::memwait::Wait) {
        let mut deferred = self.deferred();
        let before = deferred.writes.len();
        deferred.waits.push((before, wait));
        let work = self.work.load(Ordering::Relaxed);
        deferred.wait_marks.push(work);
    }

    /// Stores `what` (`bytes` of it) at `dst` once the list has run.
    ///
    /// Work is draws, dispatches, clears and copies: what has to run first.
    pub(crate) fn defer_write(&self, dst: *mut u8, what: crate::queue::Written, bytes: usize) {
        let work = self.work.load(Ordering::Relaxed);
        let mut deferred = self.deferred();
        deferred.writes.push((dst as usize, what, bytes));
        deferred.marks.push(work);
    }

    /// The list's memory waits and writes in order, with how many draws and
    /// dispatches precede each (for the log).
    pub(crate) fn script(&self) -> String {
        let deferred = self.deferred();
        let mut events: Vec<(usize, u8, String)> = Vec::new();
        for (i, (at, what, _)) in deferred.writes.iter().enumerate() {
            events.push((i, 1, format!("W[{at:x}]={what:x?}@{}", deferred.marks.get(i).copied().unwrap_or(0))));
        }
        for (before, w) in &deferred.waits {
            let mark = deferred.wait_marks.get(events.len().saturating_sub(deferred.writes.len())).copied().unwrap_or(0);
            events.push((*before, 0, format!("R[{:x}]>={:x}@{mark}", w.address, w.reference)));
        }
        events.sort_by_key(|e| (e.0, e.1));
        let work = self.work.load(Ordering::Relaxed);
        let texts: Vec<_> = events.into_iter().map(|e| e.2).collect();
        format!("{} ({work} in all)", texts.join(" "))
    }

    /// Whether the list has work: draws, dispatches, clears or copies.
    pub(crate) fn worked(&self) -> bool {
        self.work.load(Ordering::Relaxed) > 0
    }

    /// How many queries begun in the list still run.
    pub(crate) fn open_queries(&self) -> usize {
        self.queries.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Counts a dispatch or a copy: what may write indirect arguments.
    pub(crate) fn count_write(&self) {
        self.writes.fetch_add(1, Ordering::Relaxed);
    }

    /// Dispatches and copies since the reset.
    pub(crate) fn writes(&self) -> u64 {
        self.writes.load(Ordering::Relaxed)
    }

    /// Room for `bytes` of clamped indirect arguments in this list's
    /// buffers, writable by shaders from here on in the list: the buffer and
    /// the offset. None for more than a buffer holds.
    pub(crate) fn clamp_room(&self, bytes: u64) -> Option<(ID3D12Resource, u64)> {
        const CHUNK: u64 = 256 << 10;
        let bytes = bytes.next_multiple_of(16);
        if bytes > CHUNK {
            return None;
        }
        let mut chunks = self.clamp_chunks.lock().unwrap_or_else(|e| e.into_inner());
        if chunks.used > 0 && chunks.at + bytes <= CHUNK {
            let at = chunks.at;
            chunks.at += bytes;
            return Some((chunks.all[chunks.used - 1].0.clone(), at));
        }
        // The next buffer, writable.
        if chunks.used == chunks.all.len() {
            let buffer = host()?.uav_buffer(CHUNK)?;
            name(&buffer, || "hoisted indirect arguments".to_string());
            chunks.all.push((buffer, D3D12_RESOURCE_STATE_COMMON));
        }
        let i = chunks.used;
        chunks.used += 1;
        chunks.at = bytes;
        let mut barriers = Vec::new();
        if !chunks.barrier {
            // What earlier lists' shaders wrote first.
            chunks.barrier = true;
            barriers.push(D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                ..Default::default()
            });
        }
        let (buffer, state) = &mut chunks.all[i];
        if *state != D3D12_RESOURCE_STATE_UNORDERED_ACCESS {
            barriers.push(transition(buffer, *state, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
            *state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        }
        let buffer = buffer.clone();
        if !barriers.is_empty() {
            // SAFETY: an open list and live buffers.
            unsafe { self.list.ResourceBarrier(&barriers) };
        }
        Some((buffer, 0))
    }

    /// Whether the list's clamps still have to bind their root signature,
    /// pipeline and budget: the first after the reset does, and the rest
    /// find them bound (nothing else records compute state in the list).
    pub(crate) fn clamp_unbound(&self) -> bool {
        let mut chunks = self.clamp_chunks.lock().unwrap_or_else(|e| e.into_inner());
        !std::mem::replace(&mut chunks.bound, true)
    }

    /// The clamped arguments written so far become readable as indirect
    /// arguments by the lists after this one.
    pub(crate) fn clamp_end(&self) {
        let mut chunks = self.clamp_chunks.lock().unwrap_or_else(|e| e.into_inner());
        let used = chunks.used;
        let barriers: Vec<_> = chunks.all[..used]
            .iter_mut()
            .filter(|(_, state)| *state == D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
            .map(|(buffer, state)| {
                *state = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
                transition(
                    buffer,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                )
            })
            .collect();
        if !barriers.is_empty() {
            // SAFETY: an open list and live buffers.
            unsafe { self.list.ResourceBarrier(&barriers) };
        }
    }

    /// Counts a draw, dispatch, clear or copy.
    pub(crate) fn count_work(&self) {
        self.work.fetch_add(1, Ordering::Relaxed);
    }

    /// Does the list's writes that come before its first wait not met, if no
    /// work of the list precedes them; whether there were any.
    pub(crate) fn write_early(&self) -> bool {
        let mut deferred = self.deferred();
        let first = deferred.waits.iter().find(|(_, wait)| !wait.met(wait.value()));
        let Some(&(before, _)) = first else {
            return false;
        };
        let done = deferred.early;
        // Those before any work (marks only grow).
        let before = before.min(deferred.marks.iter().take_while(|&&m| m == 0).count());
        if before <= done {
            return false;
        }
        for &(at, what, bytes) in &deferred.writes[done..before] {
            what.write(at as *mut u8, bytes);
        }
        deferred.early = before;
        true
    }

    /// The waits of the list that are not met yet, apart from those its own
    /// earlier writes meet.
    pub(crate) fn unmet(&self) -> Vec<crate::memwait::Wait> {
        let deferred = self.deferred();
        deferred
            .waits
            .iter()
            .filter(|(before, wait)| {
                let own = deferred.writes[..*before]
                    .iter()
                    .rev()
                    .find(|w| w.0 == wait.address);
                match own {
                    Some((_, crate::queue::Written::Value(v), _)) => !wait.met(*v),
                    _ => !wait.met(wait.value()),
                }
            })
            .map(|w| w.1)
            .collect()
    }

    /// Copies a GPU texture readback from `src` into title memory at `dst`
    /// once the list has finished running on the host GPU.
    pub(crate) fn defer_readback(
        &self,
        dst: *mut u8,
        src: *const u8,
        dst_pitch: usize,
        src_pitch: usize,
        lines: usize,
        row_bytes: usize,
    ) {
        self.deferred().readbacks.push(Readback {
            dst: dst as usize,
            src: src as usize,
            dst_pitch,
            src_pitch,
            lines,
            row_bytes,
        });
    }

    /// Records that a draw or dispatch in this list writes `resource` through
    /// unordered access; wanted kinds are copied back to title memory on close.
    pub(crate) fn texture_written(&self, resource: crate::object::This) {
        if !crate::texback::wanted(resource) {
            return;
        }
        let mut deferred = self.deferred();
        if !deferred.back_wanted.contains(&(resource as usize)) {
            deferred.back_wanted.push(resource as usize);
        }
    }

    /// Before the list closes: copies wanted UAV-written textures into readback buffers.
    fn textures_back(&self) {
        const COPY_SOURCE: u32 = 0x800;
        let wanted = std::mem::take(&mut self.deferred().back_wanted);
        for resource in wanted {
            let resource = resource as crate::object::This;
            let Some(copy) = crate::resource::host_copy(resource) else {
                continue;
            };
            crate::list::ensure_state(self, resource, COPY_SOURCE);
            if let Some(back) = crate::texback::record(&self.list, resource, &copy.texture) {
                self.deferred().texture_backs.push(back);
            }
        }
    }
    /// `size` bytes of mapped upload memory, 512-byte aligned for a placed
    /// footprint, alive until the list is reset: the buffer, the offset in
    /// it and the CPU pointer.
    pub(crate) fn upload(&self, size: u64) -> Option<(ID3D12Resource, u64, *mut u8)> {
        self.scratch_in(size, 512, |buffer, at, _, data| (buffer.clone(), at, data))
    }

    /// A mapped readback buffer of `size` bytes, alive until the list is
    /// reset.
    pub(crate) fn readback(&self, host: &Host, size: u64) -> Option<(ID3D12Resource, *const u8)> {
        let (buffer, data) = host.readback_buffer(size)?;
        name(&buffer, || format!("list readback {size:#x}"));
        self.deferred().uploads.push(buffer.clone());
        Some((buffer, data))
    }

    /// `len` bytes of upload memory (256-byte aligned) until the list is
    /// reset: the host GPU address and the CPU pointer.
    pub(crate) fn scratch(&self, len: u64) -> Option<(u64, *mut u8)> {
        self.scratch_in(len, 256, |_, _, gpu, data| (gpu, data))
    }

    /// `len` bytes of upload memory at a multiple of `align` until the list
    /// is reset, handed to `f` with the buffer, the offset in it, the host GPU
    /// address and the CPU pointer.
    fn scratch_in<R>(
        &self,
        len: u64,
        align: u64,
        f: impl FnOnce(&ID3D12Resource, u64, u64, *mut u8) -> R,
    ) -> Option<R> {
        let host = host()?;
        let mut s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let len = len.max(1).next_multiple_of(256);
        if len > SCRATCH_CHUNK {
            let (buffer, data) = host.upload_buffer(len)?;
            name(&buffer, || format!("list scratch {len:#x}"));
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            let made = f(&buffer, 0, gpu, data);
            self.deferred().uploads.push(buffer);
            return Some(made);
        }
        if s.used.next_multiple_of(align) + len > SCRATCH_CHUNK && s.current < s.chunks.len() {
            (s.current, s.used) = (s.current + 1, 0);
        }
        if s.current == s.chunks.len() {
            let free = SCRATCH_FREE.lock().unwrap_or_else(|e| e.into_inner()).pop();
            let chunk = match free {
                Some(chunk) => chunk,
                None => {
                    let (buffer, data) = host.upload_buffer(SCRATCH_CHUNK)?;
                    name(&buffer, || "list scratch".into());
                    // SAFETY: a live buffer.
                    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
                    (buffer, data as usize, gpu)
                }
            };
            s.chunks.push(chunk);
            s.used = 0;
        }
        let (buffer, data, gpu) = &s.chunks[s.current];
        let at = s.used.next_multiple_of(align);
        let made = f(buffer, at, gpu + at, (data + at as usize) as *mut u8);
        s.used = at + len;
        Some(made)
    }

    /// The host address of the list's indirect-work budget, zeroed at each
    /// submission of the list from the first use after a reset on.
    pub(crate) fn budget(&self) -> Option<u64> {
        self.budget_with_resource().map(|(gpu, _)| gpu)
    }

    /// The budget address and its backing resource, for scoped UAV ordering.
    pub(crate) fn budget_with_resource(&self) -> Option<(u64, &ID3D12Resource)> {
        let (buffer, data) = self
            .budget
            .get_or_init(|| {
                let (b, d) = host()?.shared_buffer(256)?;
                Some((b, d as usize))
            })
            .as_ref()?;
        let mut deferred = self.deferred();
        if !deferred.budget_zeroed {
            deferred.budget_zeroed = true;
            let at = *data;
            // SAFETY: the list's own mapped word, written before it runs.
            deferred
                .copies
                .push(Step::Host(Box::new(move || unsafe { (at as *mut u32).write_volatile(0) })));
        }
        // SAFETY: a live buffer.
        Some((unsafe { buffer.GetGPUVirtualAddress() }, buffer))
    }

    /// A device-local unordered-access buffer of at least `size` bytes for
    /// clamped indirect arguments, reused across executions and resets of the
    /// list. Smaller buffers replaced mid-list stay alive until
    /// [`HostList::reset`].
    pub(crate) fn indirect_scratch(&self, size: u64) -> Option<ID3D12Resource> {
        let needed = size.max(64 << 10);
        let mut slot = self.indirect_args.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((buffer, cap)) = &*slot
            && *cap >= needed
        {
            return Some(buffer.clone());
        }
        let prev = slot.as_ref().map_or(0, |(_, cap)| *cap);
        let alloc = prev
            .saturating_mul(2)
            .max(needed)
            .next_multiple_of(64 << 10);
        let buffer = host()?.uav_buffer(alloc)?;
        name(&buffer, || format!("indirect arguments {alloc:#x}"));
        let old = slot.replace((buffer.clone(), alloc));
        drop(slot);
        if let Some((old_buf, _)) = old {
            self.keep(&old_buf);
        }
        Some(buffer)
    }

    /// Keeps `object` alive until the list is reset.
    pub(crate) fn keep(&self, object: &impl Interface) {
        if let Ok(object) = object.cast::<windows::core::IUnknown>() {
            self.deferred().kept.push(object);
        }
    }

    /// Whether this is the first time since the reset that the list asks
    /// about `key` (a texture to bring up to date).
    pub(crate) fn first_use(&self, key: usize) -> bool {
        self.deferred().textures.insert(key)
    }

    /// Runs `work` when the list is submitted, in order with its copies.
    /// `work` may write title memory (refreshes after it copy again).
    pub(crate) fn defer(&self, work: impl Fn() + Send + 'static) {
        let mut deferred = self.deferred();
        deferred.copies.push(Step::Run(Box::new(work)));
        deferred.refreshed.clear();
        deferred.gpu_written.clear();
        deferred.filled.clear();
    }

    /// Queues submission-time materialization and refresh of the bindless
    /// slots in `heap`, once per list; `writes`: the draw's shaders may write
    /// memory, so the slots' unordered access ranges go back to title memory.
    pub(crate) fn defer_bindless(
        &self,
        heap: &std::sync::Arc<crate::descriptor::host::HostHeap>,
        writes: bool,
    ) {
        let mut deferred = self.deferred();
        match deferred
            .bindless
            .iter_mut()
            .find(|(h, _)| std::sync::Arc::ptr_eq(h, heap))
        {
            Some((_, w)) => *w |= writes,
            None => deferred.bindless.push((heap.clone(), writes)),
        }
    }

    /// Whether each heap queued for bindless preparation may be written.
    #[cfg(test)]
    pub(crate) fn bindless_queued(&self) -> Vec<bool> {
        self.deferred().bindless.iter().map(|&(_, w)| w).collect()
    }

    /// As [`HostList::defer`], for work that writes only host memory.
    pub(crate) fn defer_host(&self, work: impl Fn() + Send + 'static) {
        self.deferred().copies.push(Step::Host(Box::new(work)));
    }

    /// Brings a texture's staging buffer up to date when the list is
    /// submitted (and copies it to `texture` on the prelude only if dirty),
    /// keeping both alive until the list is reset.
    pub(crate) fn defer_texture(
        &self,
        staging: std::sync::Arc<crate::texture::Staging>,
        texture: std::sync::Arc<crate::resource::HostTexture>,
    ) {
        self.deferred()
            .copies
            .push(Step::Texture(staging, texture));
    }

    /// Runs the list's submission-time steps; refreshes go into `batch`
    /// (the title's big buffers, a 174 MB one among them, are read by most
    /// lists of a submission, through many ranges). Returns the textures
    /// whose staging buffers changed and must be copied onto the prelude.
    fn run_deferred(
        &self,
        batch: &mut Batch,
    ) -> Vec<(
        std::sync::Arc<crate::texture::Staging>,
        std::sync::Arc<crate::resource::HostTexture>,
    )> {
        #[cfg(feature = "diagnostics")]
        use crate::copystat::{C, add};
        let mut uploads = Vec::new();
        let deferred = self.deferred();
        for (heap, writes) in &deferred.bindless {
            batch.bindless(heap, *writes);
        }
        let mut copy_bytes = 0u64;
        let mut copy_us = 0u64;
        #[cfg(feature = "diagnostics")]
        let mut copy_count = 0u64;
        #[cfg(feature = "diagnostics")]
        let mut refresh_count = 0u64;
        #[cfg(feature = "diagnostics")]
        let mut refresh_bytes = 0u64;
        for step in &deferred.copies {
            match step {
                &Step::Copy(dst, src, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        copy_count += 1;
                    }
                    let started = std::time::Instant::now();
                    batch.title_written();
                    copy_mapped(dst as *mut u8, src as *const u8, len, true);
                    copy_us += started.elapsed().as_micros() as u64;
                    copy_bytes += len as u64;
                }
                &Step::Fill(dst, src, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        copy_count += 1;
                    }
                    let started = std::time::Instant::now();
                    copy_mapped(dst as *mut u8, src as *const u8, len, false);
                    copy_us += started.elapsed().as_micros() as u64;
                    copy_bytes += len as u64;
                }
                Step::Refresh(shadow, address, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        refresh_count += 1;
                        refresh_bytes += *len as u64;
                    }
                    batch.add(shadow, *address, *len);
                }
                Step::Run(work) | Step::Host(work) => {
                    #[cfg(feature = "diagnostics")]
                    let started = std::time::Instant::now();
                    if matches!(step, Step::Run(_)) {
                        batch.title_written();
                    }
                    work();
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Runs, 1);
                        add(C::RunUs, started.elapsed().as_micros() as u64);
                    }
                }
                Step::GpuWrite(shadow, address, len) => {
                    // The shaders start from the title's data.
                    batch.flush();
                    shadow.writing(*address, *len);
                }
                Step::Texture(staging, copy) => {
                    if batch.texture_due(staging) && staging.refresh() {
                        uploads.push((staging.clone(), copy.clone()));
                    }
                }
            }
        }
        if copy_bytes > 0 || copy_us > 0 {
            preparing(0, copy_us);
            preparing(1, copy_bytes);
            #[cfg(feature = "diagnostics")]
            {
                add(C::Copies, copy_count);
                add(C::CopyBytes, copy_bytes);
            }
        }
        #[cfg(feature = "diagnostics")]
        if refresh_count > 0 {
            add(C::Refreshes, refresh_count);
            add(C::RefreshBytes, refresh_bytes);
        }
        uploads
    }
}

/// A submission's refreshes not done yet, merged per buffer copy into the
/// range that covers them: done before a step that writes title memory or
/// gives a range to shaders, and before the lists run.
///
/// Also the textures brought up to date in the submission so far: most
/// lists of a world submission sample the same textures, and a texture in
/// physical-page memory is hashed whole each time it is checked. Checked
/// once, it is checked again only after a step that writes title memory.
#[derive(Default)]
struct Batch {
    refreshes: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    /// The same of sparse copies.
    pieces: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    /// Staging buffers (by address) refreshed since the last title write.
    textures: std::collections::HashSet<usize>,
    /// View heaps the submission's draws index directly, and whether their
    /// shaders may write memory.
    bindless: Vec<(std::sync::Arc<crate::descriptor::host::HostHeap>, bool)>,
    /// Their plans, which hold the host resources the slots view until the
    /// submission has run.
    plans: Vec<std::sync::Arc<crate::descriptor::host::BindlessPlan>>,
    /// Bindless unordered access ranges, copied back after the submission.
    bindless_writes: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
}

impl Batch {
    fn add(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>, address: usize, len: usize) {
        let end = address + len;
        if shadow.is_sparse() {
            // It stands for far more memory than a list reads.
            self.pieces.push((shadow.clone(), address, end));
            return;
        }
        match self
            .refreshes
            .iter_mut()
            .find(|(s, ..)| std::sync::Arc::ptr_eq(s, shadow))
        {
            Some((_, from, to)) => (*from, *to) = ((*from).min(address), (*to).max(end)),
            None => self.refreshes.push((shadow.clone(), address, end)),
        }
    }

    /// Whether `staging` is still to be refreshed in this submission (and
    /// notes it refreshed).
    fn texture_due<T>(&mut self, staging: &std::sync::Arc<T>) -> bool {
        self.textures.insert(std::sync::Arc::as_ptr(staging) as *const () as usize)
    }

    /// A list of the submission indexes `heap` directly.
    fn bindless(&mut self, heap: &std::sync::Arc<crate::descriptor::host::HostHeap>, writes: bool) {
        match self
            .bindless
            .iter_mut()
            .find(|(h, _)| std::sync::Arc::ptr_eq(h, heap))
        {
            Some((_, w)) => *w |= writes,
            None => self.bindless.push((heap.clone(), writes)),
        }
    }

    /// Before a step that writes title memory: pending refreshes are done
    /// first, and textures are checked again after it.
    fn title_written(&mut self) {
        self.flush();
        self.textures.clear();
    }

    fn flush(&mut self) {
        for (shadow, from, to) in std::mem::take(&mut self.refreshes) {
            update(&shadow, from, to - from);
        }
        let mut pieces = std::mem::take(&mut self.pieces);
        pieces.sort_unstable_by_key(|(s, from, _)| (s.root_key(), *from));
        let mut run: Option<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = None;
        for (shadow, from, to) in pieces {
            if let Some((s, _, end)) = &mut run
                && s.root_key() == shadow.root_key()
                && from <= *end + 4096
            {
                *end = (*end).max(to);
                continue;
            }
            if let Some((s, a, b)) = run.replace((shadow, from, to)) {
                update(&s, a, b - a);
            }
        }
        if let Some((s, a, b)) = run {
            update(&s, a, b - a);
        }
    }

    /// After every list's steps (no title copy follows): each directly
    /// indexed heap's slots, once per submission however many lists and
    /// draws use it. What they read is refreshed (unchanged pages cost only
    /// the write-watch's answer); their unordered access ranges are kept
    /// for copy-back only when a shader that may write memory uses the heap.
    fn prepare_bindless(&mut self) {
        self.flush();
        for (heap, writes) in std::mem::take(&mut self.bindless) {
            let started = std::time::Instant::now();
            let plan = heap.bindless_plan();
            BINDLESS_US[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
            let Some(plan) = plan else {
                continue;
            };
            let (reads, all) = plan.copy_constants();
            if !reads.is_empty() {
                let started = std::time::Instant::now();
                let copied = std::sync::atomic::AtomicU64::new(0);
                in_parallel(reads, 8, |reads| {
                    for (shadow, address, len) in reads {
                        #[cfg(feature = "diagnostics")]
                        {
                            use crate::copystat::{C, add};
                            add(C::Refreshes, 1);
                            add(C::RefreshBytes, *len as u64);
                        }
                        let n = shadow.update_lazily(*address, *len, all);
                        copied.fetch_add(n as u64, Ordering::Relaxed);
                    }
                });
                preparing(0, started.elapsed().as_micros() as u64);
                preparing(1, copied.into_inner());
            }
            // shortcut: what shaders write through directly indexed slots into
            // copied memory stays in the copy (title memory used in place
            // needs no copy back); copying all of it back cost ~0.3 s a frame.
            if writes && cfg!(test) {
                for (shadow, address, len) in plan.writes.iter() {
                    if !self.bindless_writes.iter().any(|(s, a, l)| {
                        std::sync::Arc::ptr_eq(s, shadow) && (*a, *l) == (*address, *len)
                    }) {
                        shadow.writing(*address, *len);
                        self.bindless_writes.push((shadow.clone(), *address, *len));
                    }
                }
            }
            self.plans.push(plan);
        }
    }
}

/// Uploads before a submission's lists ([`crate::texture::upload_first`]).
#[derive(Default)]
struct Early {
    copies: Vec<(
        std::sync::Arc<crate::texture::Staging>,
        std::sync::Arc<crate::resource::HostTexture>,
    )>,
    before: Vec<D3D12_RESOURCE_BARRIER>,
    after: Vec<D3D12_RESOURCE_BARRIER>,
}

impl Early {
    /// Takes the textures waiting; `late`: after the lists' preludes took
    /// the textures' states, so not those the lists use (they upload them).
    fn take(&mut self, batch: &mut Batch, runs: &[&HostList], late: bool) {
        let first = crate::texture::take_first();
        if first.is_empty() {
            return;
        }
        let mut used = std::collections::HashSet::new();
        if late {
            for list in runs {
                let states = list.states.lock().unwrap_or_else(|e| e.into_inner());
                used.extend(states.values().map(|s| std::sync::Arc::as_ptr(&s.2) as usize));
            }
        }
        const COPY_DEST: i32 = D3D12_RESOURCE_STATE_COPY_DEST.0;
        const SHADER_RESOURCE: u32 = 0x40 | 0x80;
        for (staging, copy) in first {
            if used.contains(&(std::sync::Arc::as_ptr(&copy) as usize))
                || !batch.texture_due(&staging)
                || !staging.refresh()
            {
                continue;
            }
            let real = copy.state.swap(SHADER_RESOURCE, Ordering::Relaxed) as i32;
            let (to, from) = (D3D12_RESOURCE_STATES(COPY_DEST), D3D12_RESOURCE_STATES(real));
            if real != COPY_DEST {
                self.before.push(transition(&copy.texture, from, to));
            }
            self.after
                .push(transition(&copy.texture, to, D3D12_RESOURCE_STATES(SHADER_RESOURCE as i32)));
            self.copies.push((staging, copy));
        }
    }

    /// The list with the uploads, if there are any. One submission runs at a
    /// time and is waited for, so there is one list.
    fn record(&self, host: &Host) -> Option<ID3D12CommandList> {
        if self.copies.is_empty() {
            return None;
        }
        struct List(ID3D12CommandAllocator, ID3D12GraphicsCommandList);
        // SAFETY: free-threaded objects, used under the submission's lock.
        unsafe impl Send for List {}
        // SAFETY: as above.
        unsafe impl Sync for List {}
        static LIST: OnceLock<Option<List>> = OnceLock::new();
        // SAFETY: the list is not executing (every submission is waited
        // for); the textures and staging buffers are live.
        unsafe {
            let List(allocator, list) = LIST
                .get_or_init(|| {
                    let allocator: ID3D12CommandAllocator = host
                        .device
                        .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                        .ok()?;
                    let list: ID3D12GraphicsCommandList = host
                        .device
                        .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                        .ok()?;
                    list.Close().ok()?;
                    Some(List(allocator, list))
                })
                .as_ref()?;
            let recorded = allocator.Reset().and_then(|()| {
                list.Reset(allocator, None)?;
                if !self.before.is_empty() {
                    list.ResourceBarrier(&self.before);
                }
                for (staging, copy) in &self.copies {
                    staging.record_copies(list, &copy.texture);
                }
                list.ResourceBarrier(&self.after);
                list.Close()
            });
            if let Err(e) = recorded {
                fail(format_args!("texture uploads did not record ({e})"));
            }
            list.cast().ok()
        }
    }
}

/// Runs `f` on pieces of `list`, of `least` items or more, on the threads of
/// the work pool and the calling one.
pub(crate) fn in_parallel<T: Sync>(list: &[T], least: usize, f: impl Fn(&[T]) + Sync) {
    if list.len() <= least {
        return f(list);
    }
    let pool_threads = crate::workpool::threads();
    // Small pieces, taken in turn: they are not equally long to do.
    let piece = (list.len() / (pool_threads * 8)).max(least);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let work = || {
        loop {
            let from = next.fetch_add(piece, Ordering::Relaxed);
            if from >= list.len() {
                break;
            }
            f(&list[from..(from + piece).min(list.len())]);
        }
    };
    let threads = pool_threads.min(list.len().div_ceil(piece));
    crate::workpool::broadcast(threads - 1, &work);
}

/// Brings `[from, from + len)` of a host copy up to date now.
fn update(shadow: &crate::shadow::Shadow, from: usize, len: usize) {
    let started = std::time::Instant::now();
    let copied = shadow.update(from, len);
    let us = started.elapsed().as_micros() as u64;
    #[cfg(feature = "diagnostics")]
    crate::copystat::range(from, len, us, 0);
    preparing(0, us);
    preparing(1, copied as u64);
}

/// Copies title memory at submission. The ranges were checked against their
/// resources when recorded, but the title may since have unmapped pages of
/// them (tiled memory, page mappings), which read as zeros here; a title
/// destination (`title_dst`) that is not all mapped is left alone.
///
/// The source is read through the kernel (`ReadProcessMemory`), which
/// reports an unmapped page instead of faulting. Asking `VirtualQuery` first
/// cost about 2 ms per copy in the title's big pools (it walks the page
/// tables to the region's end), which made copies run at 100-200 MB/s.
pub(crate) fn copy_mapped(dst: *mut u8, src: *const u8, len: usize, title_dst: bool) {
    #[cfg(feature = "diagnostics")]
    match crate::copystat::mode() {
        1 => {
            // SAFETY: timing only: the caller vouches for the source.
            unsafe { std::ptr::copy(src, dst, len) };
            return;
        }
        2 => return,
        _ => {}
    }
    if len == 0 || title_dst && !crate::texture::committed(dst, len) {
        return;
    }
    if title_dst {
        // The kernel writes the destination: it must not be guarded.
        crate::watch::writing(dst as usize, len);
    }
    #[cfg(feature = "diagnostics")]
    let moved = std::time::Instant::now();
    let whole = read(src, dst, len);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::MoveUs, moved.elapsed().as_micros() as u64);
    if whole {
        return;
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::Fallbacks, 1);
    if !title_dst && len >= 64 << 10 {
        read_sparse(src, dst, len);
    } else {
        read_pieces(src, dst, len);
    }
}

/// After a failed bulk read into host memory, skip contiguous holes. Never
/// cache the query across copies: the title may remap the same address.
fn read_sparse(src: *const u8, dst: *mut u8, len: usize) {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, VirtualQuery,
    };
    let mut offset = 0;
    while offset < len {
        let at = src.wrapping_add(offset);
        let to = dst.wrapping_add(offset);
        let remaining = len - offset;
        // SAFETY: the query inspects an address and fills a local structure.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: `info` is a local structure of the size passed.
        let size = unsafe { VirtualQuery(at.cast(), &mut info, size_of_val(&info)) };
        let n = (info.BaseAddress as usize).saturating_add(info.RegionSize)
            .saturating_sub(at as usize).min(remaining);
        if size != size_of_val(&info) || n == 0 {
            read_pieces(at, to, remaining);
            break;
        }
        if info.State != MEM_COMMIT || info.Protect == PAGE_NOACCESS {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
            // SAFETY: this path is only used with a writable host destination.
            unsafe { to.write_bytes(0, n) };
        } else if !read(at, to, n) {
            // A mapping may change between query and read.
            read_pieces(at, to, n);
        }
        offset += n;
    }
}

fn read_pieces(src: *const u8, dst: *mut u8, len: usize) {
    // Something is unmapped: page by page (64 KiB, then 4 KiB in a piece
    // that fails), zeros for what cannot be read.
    for_pieces(src as usize, len, 64 << 10, |at, n| {
        let to = dst.wrapping_add(at - src as usize);
        if !read(at as *const u8, to, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                let to = dst.wrapping_add(at - src as usize);
                if !read(at as *const u8, to, n) {
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
                    // SAFETY: inside the destination, which is mapped.
                    unsafe { to.write_bytes(0, n) };
                }
            });
        }
    });
}

#[cfg(target_arch = "x86_64")]
core::arch::global_asm!(
    ".globl victory_safe_copy",
    ".globl victory_safe_copy_start",
    ".globl victory_safe_copy_end",
    ".globl victory_safe_copy_fault",
    "victory_safe_copy:",
    "victory_safe_copy_start:",
    "cmp r8, 64",
    "jb 2f",
    "1:",
    "movdqu xmm0, xmmword ptr [rdx]",
    "movdqu xmm1, xmmword ptr [rdx + 16]",
    "movdqu xmm2, xmmword ptr [rdx + 32]",
    "movdqu xmm3, xmmword ptr [rdx + 48]",
    "movdqu xmmword ptr [rcx], xmm0",
    "movdqu xmmword ptr [rcx + 16], xmm1",
    "movdqu xmmword ptr [rcx + 32], xmm2",
    "movdqu xmmword ptr [rcx + 48], xmm3",
    "add rdx, 64",
    "add rcx, 64",
    "sub r8, 64",
    "cmp r8, 64",
    "jae 1b",
    "2:",
    "cmp r8, 16",
    "jb 3f",
    "movdqu xmm0, xmmword ptr [rdx]",
    "movdqu xmmword ptr [rcx], xmm0",
    "add rdx, 16",
    "add rcx, 16",
    "sub r8, 16",
    "jmp 2b",
    "3:",
    "cmp r8, 8",
    "jb 4f",
    "mov rax, qword ptr [rdx]",
    "mov qword ptr [rcx], rax",
    "add rdx, 8",
    "add rcx, 8",
    "sub r8, 8",
    "4:",
    "test r8, r8",
    "jz 5f",
    "mov al, byte ptr [rdx]",
    "mov byte ptr [rcx], al",
    "inc rdx",
    "inc rcx",
    "dec r8",
    "jnz 4b",
    "5:",
    "victory_safe_copy_end:",
    "mov eax, 1",
    "ret",
    "victory_safe_copy_fault:",
    "xor eax, eax",
    "ret",
);

#[cfg(target_arch = "x86_64")]
unsafe extern "C" {
    fn victory_safe_copy(dst: *mut u8, src: *const u8, len: usize) -> u32;
    static victory_safe_copy_start: u8;
    static victory_safe_copy_end: u8;
    static victory_safe_copy_fault: u8;
}

/// Installs our Vectored Exception Handler once, ahead of the crash reporter,
/// so `safe_copy` can copy mapped memory in pure user mode without kernel
/// syscalls and recover cleanly if a page is unmapped.
#[cfg(target_arch = "x86_64")]
fn ensure_safe_copy_veh() {
    use windows_sys::Win32::Foundation::{
        EXCEPTION_ACCESS_VIOLATION, EXCEPTION_GUARD_PAGE, EXCEPTION_IN_PAGE_ERROR, NTSTATUS,
    };
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn handler(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: Windows passes valid exception and context pointers.
        let r = unsafe { &*(*info).ExceptionRecord };
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            && r.ExceptionCode != EXCEPTION_IN_PAGE_ERROR as NTSTATUS
            && r.ExceptionCode != EXCEPTION_GUARD_PAGE as NTSTATUS
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        // SAFETY: `ContextRecord` is valid for x86_64 exceptions.
        let ctx = unsafe { &mut *(*info).ContextRecord };
        let rip = ctx.Rip as usize;
        let start = (&raw const victory_safe_copy_start) as usize;
        let end = (&raw const victory_safe_copy_end) as usize;
        if rip >= start && rip < end {
            const WRITE: usize = 1;
            if r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION as NTSTATUS
                && r.NumberParameters >= 2
                && r.ExceptionInformation[0] == WRITE
                && crate::watch::try_open_guarded(r.ExceptionInformation[1])
            {
                return EXCEPTION_CONTINUE_EXECUTION;
            }
            ctx.Rip = (&raw const victory_safe_copy_fault) as u64;
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        EXCEPTION_CONTINUE_SEARCH
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: registers a read-only IP-range check handler ahead of crash reporting.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(handler));
    });
}

/// Whether to force kernel `ReadProcessMemory` / `WriteProcessMemory` copies
/// (`VICTORY_KERNEL_COPY=1`, diagnostic escape hatch).
fn kernel_copy() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_KERNEL_COPY").is_some())
}

/// Zero-syscall user-mode copy for non-overlapping ranges; returns `false` if
/// either side hits an unmapped or inaccessible page.
#[cfg(target_arch = "x86_64")]
fn try_user_copy(dst: *mut u8, src: *const u8, len: usize) -> Option<bool> {
    if kernel_copy() {
        return None;
    }
    let (d, s) = (dst as usize, src as usize);
    if len == 0 {
        return Some(true);
    }
    // Reject obviously invalid low addresses without even taking a fault.
    if d < 0x10000 || s < 0x10000 {
        return Some(false);
    }
    let (Some(d_end), Some(s_end)) = (d.checked_add(len), s.checked_add(len)) else {
        return Some(false);
    };
    // Overlapping ranges fall back to kernel copy (memmove semantics).
    if d < s_end && s < d_end {
        return None;
    }
    ensure_safe_copy_veh();
    // SAFETY: leaf assembly uses only volatile registers and recovers via VEH
    // if `src` or `dst` hits an unmapped page.
    Some(unsafe { victory_safe_copy(dst, src, len) != 0 })
}

/// Reads title memory at `src` into `dst`; false if a page of it is not
/// readable.
pub(crate) fn read_into(src: *const u8, dst: &mut [u8]) -> bool {
    read(src, dst.as_mut_ptr(), dst.len())
}

/// Copies `len` bytes from `src` to `dst` unless a source page is not
/// readable.
fn read(src: *const u8, dst: *mut u8, len: usize) -> bool {
    #[cfg(target_arch = "x86_64")]
    if let Some(ok) = try_user_copy(dst, src, len) {
        return ok;
    }
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut done = 0;
    // SAFETY: the destination is mapped writable memory of `len` bytes; the
    // kernel checks the source.
    unsafe { ReadProcessMemory(GetCurrentProcess(), src.cast(), dst.cast(), len, &mut done) != 0 }
}

/// Copies host memory into title memory after a submission (what shaders
/// wrote), leaving pages that are not mapped alone.
pub(crate) fn write_mapped(dst: *mut u8, src: *const u8, len: usize) {
    use windows_sys::Win32::System::Diagnostics::Debug::WriteProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let write = |at: usize, n: usize| {
        let from = src.wrapping_add(at - dst as usize);
        #[cfg(target_arch = "x86_64")]
        if let Some(ok) = try_user_copy(at as *mut u8, from, n) {
            return ok;
        }
        let mut done = 0;
        // SAFETY: the source is our mapped host copy; the kernel checks the
        // destination.
        unsafe { WriteProcessMemory(GetCurrentProcess(), at as _, from.cast(), n, &mut done) != 0 }
    };
    if len == 0 || write(dst as usize, len) {
        return;
    }
    for_pieces(dst as usize, len, 64 << 10, |at, n| {
        if !write(at, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                write(at, n);
            });
        }
    });
}

/// Calls `f` with the pieces of `[start, start + len)` cut at multiples of
/// `step`.
fn for_pieces(start: usize, len: usize, step: usize, mut f: impl FnMut(usize, usize)) {
    let end = start + len;
    let mut at = start;
    while at < end {
        let next = ((at / step + 1) * step).min(end);
        f(at, next - at);
        at = next;
    }
}

impl Host {
    /// A buffer in GPU-upload (Resizable BAR VRAM, when supported and within
    /// 85% of the live VRAM budget) or standard upload heap, mapped for CPU
    /// writing.
    pub(crate) fn upload_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
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
        let use_gpu_upload = self.gpu_upload
            && video_memory().is_none_or(|[(used, budget), (_, non_local_budget)]| {
                budget == 0
                    || used.saturating_add(size) < crate::card::gpu_upload_ceiling(budget, non_local_budget)
            });
        let try_make = |heap_type: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES| {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: heap_type,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            let mut data = std::ptr::null_mut();
            // SAFETY: plain creation and mapping of an upload buffer.
            let made = unsafe {
                self.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        state,
                        None,
                        &mut buffer,
                    )
                    .and_then(|()| {
                        let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                        b.Map(0, None, Some(&mut data))
                    })
            };
            made.map(|()| (buffer, data))
        };
        let made = if use_gpu_upload {
            try_make(D3D12_HEAP_TYPE_GPU_UPLOAD, D3D12_RESOURCE_STATE_COMMON)
                .or_else(|_| try_make(D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ))
        } else {
            try_make(D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ)
        };
        match made {
            Ok((buffer, data)) => {
                let buffer = buffer?;
                crate::memstat::track(&buffer, 0, size);
                Some((buffer, data.cast()))
            }
            Err(e) => {
                wait_if_removed(&self.device, &e);
                tracing::warn!(error = %e, size, "no upload buffer");
                None
            }
        }
    }

    /// A buffer in the readback heap, mapped for reading.
    pub(crate) fn readback_buffer(&self, size: u64) -> Option<(ID3D12Resource, *const u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
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
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a readback buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => {
                let buffer = buffer?;
                crate::memstat::track(&buffer, 1, size);
                Some((buffer, data.cast()))
            }
            Err(e) => {
                tracing::warn!(error = %e, size, "no readback buffer");
                None
            }
        }
    }
}

/// The swap chain formats a presented texture can be copied into: its own
/// family's.
fn swap_format(format: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(match format.0 {
        27..=32 => DXGI_FORMAT_R8G8B8A8_UNORM,
        87 | 90 | 91 => DXGI_FORMAT_B8G8R8A8_UNORM,
        23..=25 => DXGI_FORMAT_R10G10B10A2_UNORM,
        9..=14 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        _ => return None,
    })
}

const BUFFERS: u32 = 2;

struct Presenter {
    swap_chain: IDXGISwapChain3,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
    copy: HostList,
    hwnd: HWND,
    meter: Meter,
    /// One render target view, for the back buffer the readout is drawn on.
    rtv: ID3D12DescriptorHeap,
}

impl Presenter {
    /// Writes the frame rate and frame time in the top left corner of
    /// `back`, which is a render target on `list`.
    ///
    /// # Safety
    /// `back` is this swap chain's buffer, in the render target state.
    unsafe fn readout(&self, device: &ID3D12Device, list: &ID3D12GraphicsCommandList, back: &ID3D12Resource) {
        let Some((fps, ms)) = self.meter.shown else { return };
        if !crate::readout::on() {
            return;
        }
        let text = format!("{fps:.1} FPS  {ms:.0} MS");
        // Dots that keep their share of the picture at any size.
        let dot = (self.height as i32 / 216).max(2);
        let at = (dot * 3, dot * 3);
        let (w, h) = crate::readout::size(&text);
        let behind = RECT {
            left: at.0,
            top: at.1,
            right: at.0 + w * dot,
            bottom: at.1 + h * dot,
        };
        let marks = crate::readout::rects(&text, at, dot);
        // SAFETY: our heap of one view, and rectangles inside the picture
        // (a text of a few characters at this size).
        unsafe {
            let view = self.rtv.GetCPUDescriptorHandleForHeapStart();
            device.CreateRenderTargetView(back, None, view);
            if behind.right <= self.width as i32 && behind.bottom <= self.height as i32 {
                list.ClearRenderTargetView(view, &[0.0, 0.0, 0.0, 1.0], Some(&[behind]));
                list.ClearRenderTargetView(view, &[1.0, 0.9, 0.1, 1.0], Some(&marks));
            }
        }
    }
}

/// The frame rate, shown in the window's caption every second and logged
/// every few seconds.
struct Meter {
    since: std::time::Instant,
    frames: u32,
    drawn: u64,
    logged: std::time::Instant,
    /// The caption the window had, once read.
    caption: Option<Vec<u16>>,
    /// The frames a second and milliseconds a frame of the last second.
    shown: Option<(f64, f64)>,
}

impl Meter {
    fn new() -> Self {
        let now = std::time::Instant::now();
        Meter {
            since: now,
            frames: 0,
            drawn: crate::draw::drawn(),
            logged: now,
            caption: None,
            shown: None,
        }
    }

    /// Counts a presented frame.
    fn frame(&mut self, hwnd: HWND) {
        #[cfg(feature = "diagnostics")]
        tracing::debug!(target: "victory_gpu::frame", token = crate::frame::current(), "present");
        self.frames += 1;
        #[cfg(feature = "diagnostics")]
        TIMED_FRAMES.fetch_add(1, Ordering::Relaxed);
        let elapsed = self.since.elapsed();
        if elapsed < std::time::Duration::from_secs(1) {
            return;
        }
        let drawn = crate::draw::drawn();
        let frames = f64::from(self.frames);
        let fps = frames / elapsed.as_secs_f64();
        let ms = elapsed.as_secs_f64() * 1000.0 / frames;
        let draws = (drawn - self.drawn) as f64 / frames;
        (self.since, self.frames, self.drawn) = (std::time::Instant::now(), 0, drawn);
        self.shown = Some((fps, ms));
        // Of a frame, ms preparing submissions and ms waiting for the GPU.
        let [cpu, gpu] = [0, 1].map(|i| {
            SUBMISSION_US[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames
        });
        let [copy_ms, copy_mb, texture_ms, texture_mb] = [0, 1, 2, 3].map(|i| {
            let scale = if i % 2 == 0 { 1e3 } else { 1e6 };
            PREPARING[i].swap(0, Ordering::Relaxed) as f64 / scale / frames
        });
        let [back_ms, back_mb] = [0, 1].map(|i| {
            COPY_BACK[i].swap(0, Ordering::Relaxed) as f64 / [1e3, 1e6][i] / frames
        });
        let [plans_ms, records_ms] =
            [0, 1].map(|i| BINDLESS_US[i].swap(0, Ordering::Relaxed) as f64 / 1e3 / frames);
        let dropped = crate::draw::take_dropped() as f64 / frames;
        let indirect = crate::list::take_indirect() as f64 / frames;
        // Over the same frames as the counts above.
        #[cfg(feature = "diagnostics")]
        let drops = crate::pipeline::take_drops();
        if self.logged.elapsed() >= std::time::Duration::from_secs(5) {
            self.logged = std::time::Instant::now();
            if let Some(host) = host() {
                host.drain_messages();
            }
            if crate::drvwatch::errors() > 0 {
                tracing::warn!(
                    errors = crate::drvwatch::errors(),
                    "graphics driver errors reported to Windows System event log since start"
                );
            }
            #[cfg(feature = "diagnostics")]
            tracing::info!("sparse writes since: {}", crate::shadow::sparse_writes_text());
            #[cfg(feature = "diagnostics")]
            {
                let [waits, wait_us, bytes, copies] = crate::card::take_fetch_stats();
                if waits != 0 {
                    tracing::info!(
                        waits,
                        wait_ms = wait_us as f64 / 1000.0,
                        requested_mb = bytes as f64 / 1e6,
                        copies,
                        "card readback since last report (includes queue execution)"
                    );
                }
            }
            #[cfg(feature = "diagnostics")]
            log_gpu_waits();
            #[cfg(feature = "diagnostics")]
            log_gpu_times();
            tracing::info!(
                "[{} {}] {fps:.1} fps, {ms:.1} ms/frame ({cpu:.1} preparing: copies {copy_ms:.1} ms \
                 {copy_mb:.1} MB, textures {texture_ms:.1} ms {texture_mb:.1} MB, bindless plans \
                 {plans_ms:.1} ms; {gpu:.1} on the GPU; {back_ms:.1} copying back {back_mb:.1} MB; \
                 bindless records {records_ms:.1} ms), {draws:.0} draws/frame, \
                 dropped {dropped:.0} without a host pipeline ({making} being made) and \
                 {indirect:.0} indirect",
                crate::descriptor::host::BACKLOG[0].load(Ordering::Relaxed),
                crate::descriptor::host::BACKLOG[1].load(Ordering::Relaxed),
                making = crate::pipeline::making()
            );
            #[cfg(feature = "diagnostics")]
            {
                tracing::info!(
                    written = crate::resource::rewrites(),
                    evicted = crate::resource::EVICTED.load(Ordering::Relaxed),
                    late = crate::texture::CHANGED[0].load(Ordering::Relaxed),
                    at_once = crate::texture::CHANGED[1].load(Ordering::Relaxed),
                    "textures changed in place"
                );
                crate::texture::flips::log();
                {
                    static CENSUS: AtomicU32 = AtomicU32::new(0);
                    if CENSUS.fetch_add(1, Ordering::Relaxed) % 6 == 5 {
                        crate::resource::placed_census();
                    }
                }
                crate::copystat::log(frames);
            }
            crate::launch::line(frames);
            #[cfg(feature = "diagnostics")]
            {
                for (why, n) in drops.iter().take(40) {
                    tracing::info!(
                        "dropped {:.1}/frame without a host pipeline: {why}",
                        *n as f64 / frames
                    );
                }
            }
        }
        if self.caption.is_none() {
            // The window's own text, read without a message.
            let mut buffer = vec![0u16; 512];
            // SAFETY: a window handle and a buffer of that length.
            let n = unsafe { InternalGetWindowText(hwnd, &mut buffer) };
            buffer.truncate(n.max(0) as usize);
            self.caption = Some(buffer);
        }
        let Some(caption) = &self.caption else { return };
        let text = format!(" | VIctory {fps:.0} fps {ms:.1} ms/frame {draws:.0} draws/frame");
        let mut text: Vec<u16> = caption.iter().copied().chain(text.encode_utf16()).collect();
        text.push(0);
        // Setting the text waits for the window's thread (a title thread,
        // which may not pump for a while), so a thread of ours does it and
        // presenting never waits; it takes the newest text when free.
        static PENDING: Mutex<Option<(usize, Vec<u16>)>> = Mutex::new(None);
        static SETTER: std::sync::Once = std::sync::Once::new();
        *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = Some((hwnd.0 as usize, text));
        SETTER.call_once(|| {
            let _ = std::thread::Builder::new()
                .name("victory-caption".into())
                .spawn(|| {
                    loop {
                        let next = PENDING.lock().unwrap_or_else(|e| e.into_inner()).take();
                        match next {
                            // SAFETY: a window handle and a NUL-terminated text.
                            Some((hwnd, text)) => unsafe {
                                let _ = SetWindowTextW(HWND(hwnd as *mut _), PCWSTR(text.as_ptr()));
                            },
                            None => std::thread::sleep(std::time::Duration::from_millis(100)),
                        }
                    }
                });
        });
    }
}

impl Host {
    /// Shows `texture`, which is in `state` (a host state) and stays in it,
    /// as frame `frame` (saved if captures are on). `None` if nothing was
    /// shown; else, in diagnostics builds, whether the frame was black.
    pub(crate) fn present(&self, texture: &ID3D12Resource, state: u32, frame: u64) -> Option<bool> {
        let showing = std::time::Instant::now();
        let shown = self.present_now(texture, state, frame);
        crate::framelog::add(
            crate::framelog::T::PresentShowUs,
            showing.elapsed().as_micros() as u64,
        );
        shown
    }

    fn present_now(&self, texture: &ID3D12Resource, state: u32, frame: u64) -> Option<bool> {
        self.wait_in_flight();
        // SAFETY: a live resource.
        let desc = unsafe { texture.GetDesc() };
        let Some(format) = swap_format(desc.Format) else {
            tracing::warn!(format = desc.Format.0, "cannot present this format");
            return None;
        };
        let (width, height) = (desc.Width as u32, desc.Height);
        let mut presenter = self.presenter.lock().unwrap_or_else(|e| e.into_inner());
        if presenter.is_none() {
            *presenter = match self.presenter(width, height, format) {
                Ok(p) => Some(p),
                Err(e) => {
                    tracing::warn!(error = %e, "no swap chain");
                    return None;
                }
            };
        }
        let p = presenter.as_mut()?;
        if (p.width, p.height, p.format) != (width, height, format) {
            // SAFETY: no back buffer is held; every submission was waited for.
            let resized = unsafe {
                p.swap_chain
                    .ResizeBuffers(BUFFERS, width, height, format, DXGI_SWAP_CHAIN_FLAG(0))
            };
            match resized {
                Ok(()) => (p.width, p.height, p.format) = (width, height, format),
                Err(e) => {
                    tracing::warn!(error = %e, width, height, "swap chain not resized");
                    return None;
                }
            }
        }
        let state = D3D12_RESOURCE_STATES(state as i32);
        // SAFETY: our swap chain, list and a live source texture.
        unsafe {
            let index = p.swap_chain.GetCurrentBackBufferIndex();
            let Ok(back) = p.swap_chain.GetBuffer::<ID3D12Resource>(index) else {
                return None;
            };
            p.copy.reset();
            let list = &p.copy.list;
            list.ResourceBarrier(&[
                transition(texture, state, D3D12_RESOURCE_STATE_COPY_SOURCE),
                transition(
                    &back,
                    D3D12_RESOURCE_STATE_PRESENT,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                ),
            ]);
            list.CopyResource(&back, texture);
            #[cfg(feature = "diagnostics")]
            let capture = crate::capture::wanted(frame)
                .then(|| crate::capture::read_back(&self.device, list, texture))
                .flatten();
            #[cfg(feature = "diagnostics")]
            let probe = crate::capture::probe(&self.device, list, texture);
            #[cfg(not(feature = "diagnostics"))]
            let _ = frame;
            let target = D3D12_RESOURCE_STATE_RENDER_TARGET;
            list.ResourceBarrier(&[
                transition(texture, D3D12_RESOURCE_STATE_COPY_SOURCE, state),
                transition(&back, D3D12_RESOURCE_STATE_COPY_DEST, target),
            ]);
            p.readout(&self.device, list, &back);
            list.ResourceBarrier(&[transition(&back, target, D3D12_RESOURCE_STATE_PRESENT)]);
            p.copy.close();
            self.execute(&[&p.copy]);
            #[cfg(feature = "diagnostics")]
            if let Some((buffer, footprint)) = capture {
                crate::capture::save_read_back(frame, &buffer, &footprint);
            }
            let _ = p.swap_chain.Present(0, DXGI_PRESENT(0));
            p.meter.frame(p.hwnd);
            #[cfg(feature = "diagnostics")]
            return Some(probe.is_some_and(|b| crate::capture::black(&b)));
            #[cfg(not(feature = "diagnostics"))]
            Some(false)
        }
    }

    /// As [`Host::present`], with the texture `over` put over `scene` (each
    /// with the host state it is in and stays in).
    pub(crate) fn present_over(
        &self,
        scene: Option<(&ID3D12Resource, u32)>,
        over: (&ID3D12Resource, u32),
        frame: u64,
    ) -> Option<()> {
        let showing = std::time::Instant::now();
        let out = self.present_over_now(scene, over, frame);
        crate::framelog::add(
            crate::framelog::T::PresentShowUs,
            showing.elapsed().as_micros() as u64,
        );
        out
    }

    fn present_over_now(
        &self,
        scene: Option<(&ID3D12Resource, u32)>,
        over: (&ID3D12Resource, u32),
        frame: u64,
    ) -> Option<()> {
        self.wait_in_flight();
        // SAFETY: live resources.
        let (a, b) = unsafe { (scene.map(|s| s.0.GetDesc()), over.0.GetDesc()) };
        // The scene's size; with a smaller one (or none) the screen's.
        let (width, height) = match a {
            Some(a) if a.Height >= 1080 => (a.Width as u32, a.Height),
            _ => (1920, 1080),
        };
        let format = DXGI_FORMAT_R8G8B8A8_UNORM;
        let mut presenter = self.presenter.lock().unwrap_or_else(|e| e.into_inner());
        if presenter.is_none() {
            *presenter = self.presenter(width, height, format).ok();
        }
        let p = presenter.as_mut()?;
        if (p.width, p.height, p.format) != (width, height, format) {
            // SAFETY: no back buffer is held; every submission was waited for.
            unsafe {
                p.swap_chain
                    .ResizeBuffers(BUFFERS, width, height, format, DXGI_SWAP_CHAIN_FLAG(0))
            }
            .ok()?;
            (p.width, p.height, p.format) = (width, height, format);
        }
        let view = |d: &D3D12_RESOURCE_DESC| swap_format(d.Format);
        let planes = [
            scene.and_then(|s| Some((s.0, view(&a?)?))),
            Some((over.0, view(&b)?)),
        ];
        // SAFETY: our swap chain, list and live textures.
        unsafe {
            let index = p.swap_chain.GetCurrentBackBufferIndex();
            let back = p.swap_chain.GetBuffer::<ID3D12Resource>(index).ok()?;
            p.copy.reset();
            let list = &p.copy.list;
            let read = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
            let target = D3D12_RESOURCE_STATE_RENDER_TARGET;
            let mut into = vec![transition(&back, D3D12_RESOURCE_STATE_PRESENT, target)];
            let mut out = vec![transition(&back, target, D3D12_RESOURCE_STATE_PRESENT)];
            for (texture, state) in scene.iter().chain([&over]) {
                let state = D3D12_RESOURCE_STATES(*state as i32);
                if state != read {
                    into.push(transition(texture, state, read));
                    out.push(transition(texture, read, state));
                }
            }
            list.ResourceBarrier(&into);
            crate::compose::record(self, list, planes, &back, (width, height, format));
            p.readout(&self.device, list, &back);
            #[cfg(feature = "diagnostics")]
            let capture = {
                list.ResourceBarrier(&[transition(&back, target, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                let capture = crate::capture::wanted(frame)
                    .then(|| crate::capture::read_back(&self.device, list, &back))
                    .flatten();
                list.ResourceBarrier(&[transition(&back, D3D12_RESOURCE_STATE_COPY_SOURCE, target)]);
                capture
            };
            #[cfg(not(feature = "diagnostics"))]
            let _ = frame;
            list.ResourceBarrier(&out);
            p.copy.close();
            self.execute(&[&p.copy]);
            #[cfg(feature = "diagnostics")]
            if let Some((buffer, footprint)) = capture {
                crate::capture::save_read_back(frame, &buffer, &footprint);
            }
            let _ = p.swap_chain.Present(0, DXGI_PRESENT(0));
            p.meter.frame(p.hwnd);
        }
        Some(())
    }

    fn presenter(
        &self,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
    ) -> windows::core::Result<Presenter> {
        let hwnd = match title_window() {
            Some(hwnd) => hwnd,
            None => window(width, height)?,
        };
        // SAFETY: plain object creation for our window.
        unsafe {
            let factory: IDXGIFactory2 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
            let desc = DXGI_SWAP_CHAIN_DESC1 {
                Width: width,
                Height: height,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount: BUFFERS,
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                ..Default::default()
            };
            let swap_chain = factory
                .CreateSwapChainForHwnd(&self.queue, hwnd, &desc, None, None)?
                .cast::<IDXGISwapChain3>()?;
            let copy = self.new_list()?;
            copy.close();
            let rtv: ID3D12DescriptorHeap =
                self.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    NumDescriptors: 1,
                    ..Default::default()
                })?;
            tracing::info!(width, height, format = format.0, "presenting to a window");
            Ok(Presenter {
                swap_chain,
                width,
                height,
                format,
                copy,
                hwnd,
                meter: Meter::new(),
                rtv,
            })
        }
    }
}

/// A visible top-level window of this process: the one the title made, if
/// it made one.
fn title_window() -> Option<HWND> {
    extern "system" fn visit(hwnd: HWND, found: LPARAM) -> windows::core::BOOL {
        let mut process = 0;
        // SAFETY: a window handle from the enumeration; `found` is ours.
        unsafe {
            GetWindowThreadProcessId(hwnd, Some(&mut process));
            if process == windows::Win32::System::Threading::GetCurrentProcessId()
                && IsWindowVisible(hwnd).as_bool()
            {
                *(found.0 as *mut Option<HWND>) = Some(hwnd);
                return false.into();
            }
        }
        true.into()
    }
    let mut found: Option<HWND> = None;
    // SAFETY: the callback only writes `found`, which outlives the call.
    let _ = unsafe { EnumWindows(Some(visit), LPARAM(&raw mut found as isize)) };
    found
}

/// `state` without the bits the host texture's flags forbid: the console
/// lets the title name states its own resources allow (unordered access on
/// a depth target, whose host copy cannot have both).
pub(crate) fn legal_state(texture: &ID3D12Resource, state: u32) -> u32 {
    // SAFETY: a plain query.
    legal_for(unsafe { texture.GetDesc() }.Flags.0 as u32, state)
}

/// [`legal_state`] for a texture with resource flags `flags`.
pub(crate) fn legal_for(flags: u32, state: u32) -> u32 {
    // (flag, set or clear forbids, state bits): render target, unordered
    // access, depth, denied shader resource.
    const FORBIDDEN: [(u32, bool, u32); 4] = [
        (0x1, false, 0x4),
        (0x4, false, 0x8),
        (0x2, false, 0x10 | 0x20),
        (0x8, true, 0x40 | 0x80),
    ];
    FORBIDDEN
        .iter()
        .filter(|&&(flag, deny, _)| (flags & flag != 0) == deny)
        .fold(state, |s, &(_, _, bits)| s & !bits)
}

impl Host {
    pub(crate) fn queue(&self) -> &ID3D12CommandQueue {
        &self.queue
    }

    /// A buffer in system memory that the CPU maps (write-back, so reading
    /// it is fast too) and shaders read and write: for host copies of title
    /// memory that shaders write, which upload memory cannot be. Starts in
    /// the common state (buffers move out of it implicitly).
    pub(crate) fn shared_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a CPU-visible buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => {
                let buffer = buffer?;
                crate::memstat::track(&buffer, 2, size);
                Some((buffer, data.cast()))
            }
            Err(e) => {
                wait_if_removed(&self.device, &e);
                tracing::warn!(error = %e, size, "no shared buffer");
                None
            }
        }
    }

    /// A device-local buffer shaders may write (unordered access) and indirect
    /// execution may read, starting in the common state.
    pub(crate) fn uav_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a device-local buffer.
        let made = unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut buffer,
            )
        };
        match made {
            Ok(()) => {
                if let Some(b) = &buffer {
                    crate::memstat::track(b, 3, size);
                }
                buffer
            }
            Err(e) => {
                tracing::warn!(error = %e, size, "no unordered-access buffer");
                None
            }
        }
    }
}

/// For adapters without device-removed data (the software one): the last
/// commands each list of the failed submission noted (diagnostics, with
/// `VICTORY_D3D12_DRED`).
fn last_notes(runs: &[&HostList]) -> String {
    #[cfg(feature = "diagnostics")]
    {
        let mut text = String::new();
        for (i, r) in runs.iter().enumerate() {
            let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
            for (n, _) in notes.iter().rev().take(6).rev() {
                text.push_str(&format!("; list {i} of {}, of its {}: {n}", runs.len(), notes.len()));
            }
        }
        text
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        let _ = runs;
        String::new()
    }
}

/// Where the faulting address `va` lies against a range `(start, bytes)` a
/// command reads (diagnostics).
#[cfg(feature = "diagnostics")]
fn fault_in(va: u64, (start, bytes): (u64, u64)) -> String {
    let end = start + bytes;
    if va < start {
        format!(" (fault {:#x} below its source)", start - va)
    } else if va < end {
        format!(" (fault at source +{:#x})", va - start)
    } else {
        format!(" (fault {:#x} past its source's end)", va - end)
    }
}

/// The commands that may have been running where a list stopped at
/// command `done`: those between the barriers around it (work between
/// barriers overlaps on the GPU), by kind, and what the markers just
/// before them say (`HostList::note_op`: diagnostics builds with
/// device-removed data on).
fn in_flight(history: &[D3D12_AUTO_BREADCRUMB_OP], markers: &[(u32, String)], done: usize) -> String {
    let barrier = |op: &D3D12_AUTO_BREADCRUMB_OP| *op == D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER;
    let from = history[..done].iter().rposition(barrier).map_or(0, |b| b + 1);
    let to = history[done..]
        .iter()
        .position(barrier)
        .map_or(history.len(), |b| (done + b).max(done + 1));
    let mut text = format!(", commands {from}..{to} between its barriers:");
    let before = markers.iter().filter(|(at, _)| (*at as usize) < from);
    for (at, marker) in before.rev().take(6) {
        let short: String = marker.chars().take(90).collect();
        text.push_str(&format!(" (before, {at}: {short})"));
    }
    let mut shown = 0;
    for i in from..to {
        let op = history[i];
        if op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER || shown == 100 {
            continue;
        }
        shown += 1;
        let marker = i
            .checked_sub(1)
            .filter(|&m| history[m] == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
            .and_then(|m| markers.iter().find(|(at, _)| *at as usize == m))
            .map_or("no marker", |(_, s)| s.as_str());
        text.push_str(&format!("; command {i} (op {}): {marker}", op.0));
        // Its constants as they are now: the title may have overwritten them.
        #[cfg(feature = "diagnostics")]
        for c in marker.split("Constant(0x").skip(1).take(4) {
            let mut parts = c.split([',', ')']);
            let at = parts.next().and_then(|a| u64::from_str_radix(a, 16).ok());
            let size = parts.next().and_then(|s| s.trim().parse::<usize>().ok());
            if let (Some(at), Some(size)) = (at, size) {
                let words = crate::shadow::words_text(at, size);
                let same = c.contains(&words);
                text.push_str(&format!(" NOW {at:#x} same={same} {words}"));
                if shown == 1 {
                    let mut slots: Vec<u32> = words
                        .split(['[', ',', ']'])
                        .filter_map(|w| u32::from_str_radix(w.trim(), 16).ok())
                        .filter(|w| (0x1000..0x13_0000).contains(w))
                        .collect();
                    for u in marker.split(" u").skip(1) {
                        let v = u.split_once("=0x").map(|(_, v)| v.split(' ').next().unwrap_or(""));
                        if let Some(v) = v.and_then(|v| u32::from_str_radix(v.trim(), 16).ok())
                            && v % 32 == 0
                            && v >= 0x10000
                        {
                            slots.extend([0, 1, 2, 3, 6, 24, 25].map(|k| v / 32 + k));
                        }
                    }
                    text.push_str(&crate::draw::report_slots(&slots));
                }
            }
        }
        // The records a skinning program takes its views from: those whose
        // slot holds no buffer.
        #[cfg(feature = "diagnostics")]
        if shown <= 3 && marker.contains("CSComputeSkinning") {
            let number = |key: &str, radix: u32, end: char| {
                marker
                    .split_once(key)
                    .and_then(|(_, v)| u64::from_str_radix(v.split(end).next()?, radix).ok())
            };
            let list = marker.split_once("Srv s1+16: placed").and_then(|(_, t)| {
                u64::from_str_radix(t.split_once(" at 0x")?.1.split('"').next()?, 16).ok()
            });
            if let (Some(list), Some(first), Some(count)) =
                (list, number(" u6=0x", 16, ' '), number(" u7=0x", 16, ' '))
            {
                text.push_str(&format!(" RECORDS {first}+{count} of {list:#x}"));
                for k in 0..count.min(64) {
                    let at = list + 192 + (first + k) * 176;
                    let words = crate::shadow::words_text(at, 128);
                    let w: Vec<u32> = words
                        .split(['[', ',', ']'])
                        .filter_map(|w| u32::from_str_radix(w.trim(), 16).ok())
                        .collect();
                    if w.len() < 32 {
                        text.push_str(&format!(" REC {k} unreadable"));
                        continue;
                    }
                    let slots = [w[18], w[15] & 0xff_ffff, w[19] & 0xff_ffff];
                    let what = crate::draw::report_slots(&slots);
                    if k < 1 || what.contains("null") || what.contains("Shader(") || what.contains("DIRTY") {
                        text.push_str(&format!(" REC {k} {w:x?}{what}"));
                    }
                }
            }
        }
        // What its root views start with now.
        #[cfg(feature = "diagnostics")]
        if shown <= 6 {
            for r in marker.split(" root").skip(1).take(10) {
                if let Some((slot, rest)) = r.split_once("=0x")
                    && let Ok(at) = u64::from_str_radix(rest.split(' ').next().unwrap_or(""), 16)
                {
                    text.push_str(&format!(" ROOT {slot} {at:#x} {}", crate::shadow::words_text(at, 80)));
                    if let Some(s) = crate::shadow::find(at, 80)
                        && s.is_sparse()
                    {
                        text.push_str(&format!(" {}", crate::shadow::args_text(at)));
                    }
                }
            }
        }
        // What an indirect command got, as the title's memory has it now.
        if let Some(at) = marker
            .strip_prefix("indirect kind ")
            .and_then(|m| m.split(" at 0x").nth(1))
            .and_then(|m| u64::from_str_radix(m.split(' ').next().unwrap_or(""), 16).ok())
        {
            text.push_str(&format!(" args now {}", crate::shadow::args_text(at)));
        }
    }
    text
}

pub(crate) fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// A window of the given client size, owned by its own message thread.
fn window(width: u32, height: u32) -> windows::core::Result<HWND> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("victory-window".into())
        .spawn(move || {
            let made = make_window(width, height);
            let ok = made.is_ok();
            let _ = sender.send(made.map(|h| h.0 as usize));
            if ok {
                pump();
            }
        })
        .map_err(|_| windows::core::Error::empty())?;
    let hwnd = receiver
        .recv()
        .map_err(|_| windows::core::Error::empty())??;
    Ok(HWND(hwnd as *mut _))
}

extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // SAFETY: the default handling for our window.
    unsafe { DefWindowProcW(hwnd, message, wparam, lparam) }
}

fn make_window(width: u32, height: u32) -> windows::core::Result<HWND> {
    const CLASS: PCWSTR = w!("VIctory");
    // SAFETY: registers our class once and creates a window of it.
    unsafe {
        let instance = GetModuleHandleW(None)?;
        let class = WNDCLASSEXW {
            cbSize: size_of::<WNDCLASSEXW>() as u32,
            lpfnWndProc: Some(window_proc),
            hInstance: instance.into(),
            lpszClassName: CLASS,
            hCursor: LoadCursorW(None, IDC_ARROW)?,
            ..Default::default()
        };
        RegisterClassExW(&class);
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: width as i32,
            bottom: height as i32,
        };
        AdjustWindowRect(&mut rect, WS_OVERLAPPEDWINDOW, false)?;
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            CLASS,
            w!("VIctory"),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            rect.right - rect.left,
            rect.bottom - rect.top,
            None,
            None,
            Some(instance.into()),
            None,
        )
    }
}

fn pump() {
    let mut message = MSG::default();
    // SAFETY: the standard message loop of this thread's window.
    unsafe {
        while GetMessageW(&mut message, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
}

#[cfg(test)]
mod batch {   // Batch::texture_due lives here; `mod tests` is already taken at the top of this file
    use super::Batch;
    use std::sync::Arc;

    #[test]
    fn a_texture_is_checked_once_per_submission() {
        let mut batch = Batch::default();
        let (a, b) = (Arc::new([0u8; 16]), Arc::new([0u8; 16]));
        // The first list sampling each texture checks it ...
        assert!(batch.texture_due(&a));
        assert!(batch.texture_due(&b));
        // ... later lists of the same submission do not.
        assert!(!batch.texture_due(&a));
        assert!(!batch.texture_due(&a.clone()));
        assert!(!batch.texture_due(&b));
    }

    #[test]
    fn title_writes_make_textures_due_again() {
        let mut batch = Batch::default();
        let a = Arc::new(1u32);
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A copy into title memory or work that may write it may have
        // changed the texture's memory.
        batch.title_written();
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A fresh submission starts over.
        let mut next = Batch::default();
        assert!(next.texture_due(&a));
    }

    /// The hashing a submission does for textures in physical-page memory,
    /// with every list checking the textures it samples (before) and with
    /// each checked once (after). Run with
    /// `cargo test --release -p victory-gpu -- --ignored --nocapture texture_checks`.
    #[test]
    #[ignore = "benchmark"]
    fn texture_checks_per_submission() {
        const TEXTURES: usize = 48;
        const SIZE: usize = 4 << 20;
        const LISTS: usize = 12;
        let textures: Vec<Arc<Vec<u8>>> = (0..TEXTURES)
            .map(|t| Arc::new((0..SIZE).map(|i| (i * 31 + t) as u8).collect()))
            .collect();
        let submit = |dedupe: bool| {
            let mut batch = Batch::default();
            let mut hashed = 0usize;
            let started = std::time::Instant::now();
            for _ in 0..LISTS {
                for t in &textures {
                    if !dedupe || batch.texture_due(t) {
                        std::hint::black_box(blake3::hash(t));
                        hashed += 1;
                    }
                }
            }
            (started.elapsed(), hashed)
        };
        submit(false); // warm up
        let (before, n_before) = submit(false);
        let (after, n_after) = submit(true);
        assert_eq!((n_before, n_after), (TEXTURES * LISTS, TEXTURES));
        println!(
            "{LISTS} lists sampling {TEXTURES} textures of {} MiB: \
             before {n_before} hashes in {before:?}, after {n_after} in {after:?} ({:.1}x)",
            SIZE >> 20,
            before.as_secs_f64() / after.as_secs_f64()
        );
    }

    #[test]
    fn safe_copy_handles_valid_unaligned_and_unmapped_memory() {
        use super::{copy_mapped, read_into, write_mapped};
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
            VirtualFree,
        };
        for len in [0usize, 1, 7, 8, 15, 16, 31, 63, 64, 65, 129, 4096, 70_000] {
            let src: Vec<u8> = (0..len).map(|i| ((i * 31 + 7) & 0xff) as u8).collect();
            let mut dst = vec![0xAAu8; len];
            assert!(read_into(src.as_ptr(), &mut dst), "valid read len={len}");
            assert_eq!(dst, src, "mismatch len={len}");
        }
        // Reserve 2 pages: page 0 committed RW, page 1 uncommitted (faults on access).
        // SAFETY: fresh reservation, freed below.
        let base = unsafe {
            VirtualAlloc(std::ptr::null(), 2 * 4096, MEM_RESERVE, PAGE_NOACCESS) as usize
        };
        assert_ne!(base, 0);
        // SAFETY: commit only page 0.
        let committed = unsafe {
            VirtualAlloc(base as *mut _, 4096, MEM_COMMIT, PAGE_READWRITE) as usize
        };
        assert_eq!(committed, base);
        // SAFETY: page 0 is committed RW.
        unsafe { std::ptr::write_bytes(base as *mut u8, 0x5A, 4096) };

        let mut buf = vec![0u8; 2 * 4096];
        // Reading across into uncommitted page 1 must fail cleanly without crashing.
        assert!(!read_into(base as *const u8, &mut buf));
        // copy_mapped fills page 0 with 0x5A and zeroes uncommitted page 1.
        copy_mapped(buf.as_mut_ptr(), base as *const u8, 2 * 4096, false);
        assert!(buf[..4096].iter().all(|&b| b == 0x5A));
        assert!(buf[4096..].iter().all(|&b| b == 0));

        // write_mapped writes page 0 and safely skips uncommitted page 1.
        let src = vec![0x3Cu8; 2 * 4096];
        write_mapped(base as *mut u8, src.as_ptr(), 2 * 4096);
        // SAFETY: page 0 is committed RW.
        let page0 = unsafe { std::slice::from_raw_parts(base as *const u8, 4096) };
        assert!(page0.iter().all(|&b| b == 0x3C));

        // SAFETY: our reservation.
        unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
    }

    #[test]
    fn safe_copy_handles_guard_pages() {
        use super::read_into;
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_GUARD, PAGE_READWRITE, VirtualAlloc,
            VirtualFree, VirtualProtect,
        };
        // SAFETY: fresh committed page freed below.
        let base = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4096,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            ) as usize
        };
        assert_ne!(base, 0);
        // SAFETY: page is committed RW.
        unsafe { std::ptr::write_bytes(base as *mut u8, 42, 4096) };
        let mut prev = 0u32;
        // SAFETY: protecting our own committed page with PAGE_GUARD.
        let guarded = unsafe { VirtualProtect(base as *mut _, 4096, PAGE_READWRITE | PAGE_GUARD, &mut prev) };
        assert_ne!(guarded, 0);
        let mut dst = [0u8; 4];
        assert!(!read_into(base as *const u8, &mut dst));
        // SAFETY: freeing our own allocation.
        unsafe { VirtualFree(base as *mut _, 0, MEM_RELEASE) };
    }
}

/// The device is gone: why is logged, and the caller waits, so the thread
/// that runs the submission can report it before the title fails.
fn wait_if_removed(device: &ID3D12Device, error: &windows::core::Error) {
    if error.code().0 as u32 != 0x887A_0005 {
        return;
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        // SAFETY: a plain query.
        let reason = unsafe { device.GetDeviceRemovedReason() };
        tracing::error!(?reason, layer = ?host().and_then(|h| h.log_messages()), "host GPU device gone");
    });
    std::thread::sleep(std::time::Duration::from_secs(20));
}
