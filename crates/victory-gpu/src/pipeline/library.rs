//! Compiled host pipelines kept on disk, and the title pipelines worth
//! starting early (decision 0007, "Pipeline cache").
//!
//! 1. Every host pipeline goes into the host's pipeline library, named by
//!    a hash of all that its host description holds (the translated
//!    shaders' code, which changes when a translation does, our root
//!    signature, the graphics state). The library is saved under the cache
//!    folder; the host refuses a library of another adapter or driver, and
//!    an entry whose description differs, so neither is ever used.
//! 2. The title pipelines a run drew or dispatched with are listed by a key
//!    known when the title creates them (their shader containers and
//!    graphics state); the next run hands those to the workers at
//!    creation, during loading.
//!
//! All library calls are serialised: the interface does not promise that
//! stores, loads and serialising may overlap.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{HSTRING, Interface, PCWSTR};

static DIR: OnceLock<PathBuf> = OnceLock::new();

/// Keeps the pipeline library and the early list in `dir` from now on.
pub fn use_pipeline_cache(dir: PathBuf) {
    let _ = DIR.set(dir);
}

/// Host pipelines created, loaded from the library, and title pipelines
/// started at creation, since the start.
pub(crate) static MADE: AtomicU64 = AtomicU64::new(0);
pub(crate) static LOADED: AtomicU64 = AtomicU64::new(0);
pub(crate) static EARLY: AtomicU64 = AtomicU64::new(0);

struct Library {
    library: ID3D12PipelineLibrary1,
    /// The blob the library was made from: it must outlive the library.
    _blob: Vec<u8>,
    /// Pipelines stored since the last save.
    added: u64,
}

// SAFETY: D3D12 objects are free-threaded; every call is made under the
// mutex that holds the library.
unsafe impl Send for Library {}

const LIBRARY_FILE: &str = "host-pipelines.bin";
const EARLY_FILE: &str = "early-pipelines.txt";
/// Title pipelines listed for an early start, at most.
const EARLY_MAX: usize = 8192;

fn library() -> Option<&'static Mutex<Library>> {
    static LIBRARY: OnceLock<Option<Mutex<Library>>> = OnceLock::new();
    LIBRARY
        .get_or_init(|| {
            let device = crate::host::host()?.device.cast::<ID3D12Device1>().ok()?;
            let blob = DIR
                .get()
                .and_then(|d| std::fs::read(d.join(LIBRARY_FILE)).ok())
                .unwrap_or_default();
            let library = open(&device, blob);
            if library.is_none() {
                tracing::warn!("no host pipeline library: pipelines are not kept");
            }
            library.map(Mutex::new)
        })
        .as_ref()
}

/// The library in `blob`, or an empty one if the host rejects the blob (an
/// adapter or driver change, damage).
fn open(device: &ID3D12Device1, blob: Vec<u8>) -> Option<Library> {
    let make = |bytes: &[u8]| -> windows::core::Result<ID3D12PipelineLibrary1> {
        // SAFETY: the bytes outlive the library (kept beside it).
        unsafe {
            device.CreatePipelineLibrary::<ID3D12PipelineLibrary>(bytes)?.cast()
        }
    };
    if !blob.is_empty() {
        match make(&blob) {
            Ok(library) => {
                tracing::info!(bytes = blob.len(), "host pipeline library loaded");
                return Some(Library { library, _blob: blob, added: 0 });
            }
            Err(e) => tracing::info!(error = %e, "host pipeline library rejected; starting an empty one"),
        }
    } else {
        tracing::info!("no host pipeline library yet; starting an empty one");
    }
    let library = make(&[]).ok()?;
    Some(Library { library, _blob: Vec::new(), added: 0 })
}

/// A pipeline's name in the library: the hash of what its host
/// description holds.
pub(crate) fn name(kind: &str, root: &[u8], code: &[&[u8]], state: &str) -> String {
    let mut h = blake3::Hasher::new();
    for part in [kind.as_bytes(), root].into_iter().chain(code.iter().copied()).chain([state.as_bytes()]) {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part);
    }
    h.finalize().to_hex()[..32].to_owned()
}

/// Pipelines a driver's compiler crashed on (names, one per line).
const CRASHED_FILE: &str = "crashed.txt";

/// The pipeline named `name` is left out from the next run on.
fn note_crashed(name: &str) {
    use std::io::Write;
    let Some(dir) = DIR.get() else { return };
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join(CRASHED_FILE));
    if let Ok(mut f) = file {
        let _ = writeln!(f, "{name}");
    }
}

/// Whether the driver crashed making the pipeline named `name` before.
fn crashed(name: &str) -> bool {
    static NAMES: OnceLock<std::collections::HashSet<String>> = OnceLock::new();
    NAMES
        .get_or_init(|| {
            DIR.get()
                .and_then(|d| std::fs::read_to_string(d.join(CRASHED_FILE)).ok())
                .map(|t| t.lines().map(str::to_owned).collect())
                .unwrap_or_default()
        })
        .contains(name)
}

thread_local! {
    /// The pipeline the driver makes on this thread.
    static MAKING: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
}

/// The driver makes a pipeline on this thread. If its compiler crashes there
/// (AMD's does on a few of the world's compute shaders), the pipeline is
/// noted and left out from then on, and the thread parked: the others, and
/// the title, go on.
struct Making;

impl Making {
    fn new(name: &str) -> Making {
        use windows_sys::Win32::System::Diagnostics::Debug::AddVectoredExceptionHandler;
        static HANDLER: std::sync::Once = std::sync::Once::new();
        // SAFETY: a handler that lives as long as the process.
        HANDLER.call_once(|| unsafe {
            AddVectoredExceptionHandler(1, Some(compiler_crashed));
        });
        MAKING.with(|m| name.clone_into(&mut m.borrow_mut()));
        endless(name, true);
        Making
    }
}

/// The process's private bytes.
fn committed() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    // SAFETY: a structure of the size passed.
    unsafe {
        let mut info: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
        let size = size_of_val(&info) as u32;
        if GetProcessMemoryInfo(GetCurrentProcess(), &mut info, size) == 0 {
            return 0;
        }
        info.PagefileUsage as u64
    }
}

/// Workers (threads) lost to compiles that crashed, not replaced yet.
static LOST: Mutex<Vec<u32>> = Mutex::new(Vec::new());

/// Suspends thread `id`, at work in the driver's compiler (never in a heap
/// function, whose lock it would keep); false if it was not caught there.
fn suspend(id: u32) -> bool {
    use windows::Win32::Foundation::{CloseHandle, HMODULE};
    use windows_sys::Win32::System::Diagnostics::Debug::{CONTEXT, GetThreadContext};
    use windows::Win32::System::LibraryLoader::{
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        GetModuleHandleExW, GetModuleHandleW,
    };
    use windows::Win32::System::Threading::{
        OpenThread, ResumeThread, SuspendThread, THREAD_GET_CONTEXT, THREAD_SUSPEND_RESUME,
    };
    const CONTEXT_CONTROL_AMD64: u32 = 0x10_0001;
    #[repr(align(16))]
    struct Aligned(CONTEXT);
    // SAFETY: plain calls on a thread of ours; the context is whole and
    // aligned; nothing that may need the suspended thread's locks is called
    // while it is suspended (the modules are asked about after it runs on).
    unsafe {
        let Ok(thread) = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT, false, id) else {
            return false;
        };
        let module = |at: u64| {
            let mut m = HMODULE::default();
            let _ = GetModuleHandleExW(
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                PCWSTR(at as *const u16),
                &mut m,
            );
            m.0 as usize
        };
        let named = |s: &str| GetModuleHandleW(&HSTRING::from(s)).map_or(0, |m| m.0 as usize);
        let (ntdll, ucrt) = (named("ntdll.dll"), named("ucrtbase.dll"));
        let mut caught = false;
        let mut last = Vec::new();
        for _ in 0..500 {
            if SuspendThread(thread) == u32::MAX {
                break;
            }
            let mut context: Aligned = std::mem::zeroed();
            context.0.ContextFlags = CONTEXT_CONTROL_AMD64;
            let known = GetThreadContext(thread.0, &mut context.0) != 0;
            let at = context.0.Rip >> 12;
            if known && last.contains(&at) {
                caught = true;
                break;
            }
            ResumeThread(thread);
            // A page of the driver's own code: caught there next time.
            let m = module(at << 12);
            if known && m != 0 && m != ntdll && m != ucrt {
                last.push(at);
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let _ = CloseHandle(thread);
        caught
    }
}

/// Notes pipelines the driver's compiler has worked on for long
/// ([`ENDLESS`]): they are left out from the next run on. AMD's never ends
/// some of the world's ray tracing programs, and takes memory all the while
/// (16 GB in two minutes), so their threads are stopped.
fn endless(name: &str, begins: bool) {
    use std::time::{Duration, Instant};
    // Some long programs take a while (the upscaler's 10,000 instructions);
    // what takes memory all along goes first.
    // Not sooner, nor for less memory: while the world loads the process
    // takes gigabytes by itself and every core is busy, and four pipelines
    // the driver makes in seconds were left out for good (20 s and 3 GiB;
    // run amd1, 2026-11-25). A compile that never ends takes 8 GiB a minute.
    const ENDLESS: Duration = Duration::from_secs(60);
    const EVEN_SO: Duration = Duration::from_secs(300);
    const GROWTH: u64 = 8 << 30;
    static AT_WORK: Mutex<Vec<(String, Instant, u32)>> = Mutex::new(Vec::new());
    static WATCH: std::sync::Once = std::sync::Once::new();
    {
        let mut at_work = AT_WORK.lock().unwrap_or_else(|e| e.into_inner());
        if begins {
            // SAFETY: a plain call.
            let id = unsafe { windows::Win32::System::Threading::GetCurrentThreadId() };
            at_work.push((name.to_owned(), Instant::now(), id));
        } else if let Some(i) = at_work.iter().position(|w| w.0 == name) {
            at_work.swap_remove(i);
        }
    }
    WATCH.call_once(|| {
        let _ = std::thread::Builder::new()
            .name("victory-compiles".into())
            .spawn(|| {
                let mut noted: Vec<String> = Vec::new();
                // The process's commit when a compile was first seen late.
                let mut late: Option<u64> = None;
                loop {
                    std::thread::sleep(Duration::from_secs(2));
                    let lost = std::mem::take(&mut *LOST.lock().unwrap_or_else(|e| e.into_inner()));
                    for id in lost {
                        super::make::worker_lost(id);
                    }
                    let long: Vec<(String, u32)> = AT_WORK
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .iter()
                        .filter(|w| w.1.elapsed() > ENDLESS && !noted.contains(&w.0))
                        .map(|w| (w.0.clone(), w.2))
                        .collect();
                    if long.is_empty() {
                        late = None;
                        continue;
                    }
                    let now = committed();
                    let grown = now.saturating_sub(*late.get_or_insert(now)) > GROWTH;
                    let long: Vec<(String, u32)> = if grown {
                        late = None;
                        long
                    } else {
                        AT_WORK
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .iter()
                            .filter(|w| w.1.elapsed() > EVEN_SO && !noted.contains(&w.0))
                            .map(|w| (w.0.clone(), w.2))
                            .collect()
                    };
                    for (name, id) in long {
                        let stopped = suspend(id);
                        if stopped {
                            super::make::worker_lost(id);
                        }
                        tracing::error!(
                            name,
                            stopped,
                            "the driver's compiler does not end a pipeline: it is left out from the next run on"
                        );
                        note_crashed(&name);
                        noted.push(name);
                    }
                }
            });
    });
}

impl Drop for Making {
    fn drop(&mut self) {
        let name = MAKING.with(|m| std::mem::take(&mut *m.borrow_mut()));
        endless(&name, false);
    }
}

unsafe extern "system" fn compiler_crashed(
    info: *mut windows_sys::Win32::System::Diagnostics::Debug::EXCEPTION_POINTERS,
) -> i32 {
    const ACCESS_VIOLATION: i32 = 0xc000_0005_u32 as i32;
    // SAFETY: the system passes a live record.
    let record = unsafe { &*(*info).ExceptionRecord };
    let name = MAKING.try_with(|m| m.try_borrow().map(|m| m.clone()).unwrap_or_default());
    let name = name.unwrap_or_default();
    // Only reads or writes of the first pages: our own guards' faults are
    // elsewhere.
    if record.ExceptionCode != ACCESS_VIOLATION
        || name.is_empty()
        || record.ExceptionInformation[1] >= 0x10000
    {
        return 0; // not ours
    }
    tracing::error!(name, "the driver's compiler crashed making a pipeline: it is left out");
    note_crashed(&name);
    endless(&name, false);
    // SAFETY: a plain call.
    let id = unsafe { windows::Win32::System::Threading::GetCurrentThreadId() };
    LOST.lock().unwrap_or_else(|e| e.into_inner()).push(id);
    loop {
        std::thread::sleep(std::time::Duration::from_secs(3600));
    }
}

/// The pipeline named `name`: loaded from the library by `load` if it is
/// there with the same description, else made by `create` and stored.
pub(crate) fn pipeline(
    name: &str,
    load: impl FnOnce(&ID3D12PipelineLibrary1, PCWSTR) -> windows::core::Result<ID3D12PipelineState>,
    create: impl FnOnce() -> windows::core::Result<ID3D12PipelineState>,
) -> windows::core::Result<ID3D12PipelineState> {
    let wide = HSTRING::from(name);
    if crashed(name) {
        // E_FAIL.
        return Err(windows::core::Error::from_hresult(windows::core::HRESULT(0x8000_4005_u32 as i32)));
    }
    let create = || {
        let _making = Making::new(name);
        create()
    };
    let Some(library) = library() else {
        MADE.fetch_add(1, Ordering::Relaxed);
        return create();
    };
    if let Ok(pso) = load(&library.lock().unwrap_or_else(|e| e.into_inner()).library, PCWSTR(wide.as_ptr())) {
        LOADED.fetch_add(1, Ordering::Relaxed);
        return Ok(pso);
    }
    // Compiled outside the lock: compiles take up to seconds.
    let pso = create()?;
    MADE.fetch_add(1, Ordering::Relaxed);
    if stores_later() {
        store_or_defer(
            library,
            &DEFERRED,
            Deferred(wide, pso.clone()),
            |l, Deferred(wide, pso)| store_pipeline(l, &wide, &pso),
        );
    } else {
        let mut l = library.lock().unwrap_or_else(|e| e.into_inner());
        store_pipeline(&mut l, &wide, &pso);
    }
    Ok(pso)
}

fn store_pipeline(l: &mut Library, wide: &HSTRING, pso: &ID3D12PipelineState) {
    // SAFETY: a live pipeline; all library calls remain serialized.
    if unsafe { l.library.StorePipeline(PCWSTR(wide.as_ptr()), pso) }.is_ok() {
        l.added += 1;
    }
}

// Adapted from DoubleGhosty/cool_fox 4a028f31 (GPL-3.0), retaining this
// branch's single library mutex. Only cache stores may be deferred.
fn stores_later() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LIBRARY_STORE_LATER").is_some_and(|v| v == "1"))
}

struct Deferred(HSTRING, ID3D12PipelineState);
// SAFETY: D3D12 objects are free-threaded; the queue owns its reference.
unsafe impl Send for Deferred {}
static DEFERRED: Mutex<Vec<Deferred>> = Mutex::new(Vec::new());
const DEFERRED_LIMIT: usize = 256;

fn store_or_defer<L, T>(
    library: &Mutex<L>,
    queue: &Mutex<Vec<T>>,
    item: T,
    store: impl FnOnce(&mut L, T),
) {
    let mut l = match library.try_lock() {
        Ok(l) => l,
        Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
        Err(std::sync::TryLockError::WouldBlock) => {
            let mut q = queue.lock().unwrap_or_else(|e| e.into_inner());
            // A full queue skips only cache publication, never the live PSO.
            if q.len() < DEFERRED_LIMIT {
                q.push(item);
            }
            return;
        }
    };
    store(&mut l, item);
}

fn drain_deferred<T>(queue: &Mutex<Vec<T>>, mut store: impl FnMut(T)) {
    let items = std::mem::take(&mut *queue.lock().unwrap_or_else(|e| e.into_inner()));
    for item in items {
        store(item);
    }
}

fn save_every() -> std::time::Duration {
    static EVERY: OnceLock<std::time::Duration> = OnceLock::new();
    *EVERY.get_or_init(|| {
        let seconds = std::env::var("VICTORY_LIBRARY_SAVE_S")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(20)
            .max(5);
        std::time::Duration::from_secs(seconds)
    })
}

/// Periodic saves must not park other pipeline workers behind disk I/O.
fn save_guard<T>(lock: &Mutex<T>, now: bool) -> Option<std::sync::MutexGuard<'_, T>> {
    if now {
        return Some(lock.lock().unwrap_or_else(|e| e.into_inner()));
    }
    match lock.try_lock() {
        Ok(guard) => Some(guard),
        Err(std::sync::TryLockError::Poisoned(e)) => Some(e.into_inner()),
        Err(std::sync::TryLockError::WouldBlock) => None,
    }
}

/// Writes only the dirty cache components. Early-use records can change
/// without a new host PSO (including on hosts without pipeline libraries).
pub(crate) fn save(now: bool) {
    // Held over serialization and writes, so overlapping saves cannot
    // replace files under each other.
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    let Some(dir) = DIR.get() else { return };
    let Some(mut last) = save_guard(&LAST, now) else { return };
    if !now && last.is_some_and(|t| t.elapsed() < save_every()) {
        return;
    }
    *last = Some(std::time::Instant::now());

    if let Some(library) = library() {
        let snapshot = {
            let mut l = library.lock().unwrap_or_else(|e| e.into_inner());
            if stores_later() {
                drain_deferred(&DEFERRED, |Deferred(wide, pso)| {
                    store_pipeline(&mut l, &wide, &pso);
                });
            }
            serialize_if_dirty(l.added, || {
                // SAFETY: a buffer of the size the library asks for.
                let bytes = unsafe {
                    let mut bytes = vec![0u8; l.library.GetSerializedSize()];
                    l.library.Serialize(&mut bytes).map(|()| bytes)
                };
                match bytes {
                    Ok(bytes) => Some(bytes),
                    Err(e) => {
                        tracing::warn!(error = %e, "host pipeline library not serialised");
                        None
                    }
                }
            })
        };
        if let Some((blob, saved_additions)) = snapshot {
            match std::fs::create_dir_all(dir)
                .and_then(|()| write_atomically(&dir.join(LIBRARY_FILE), &blob))
            {
                Ok(()) => {
                    let mut l = library.lock().unwrap_or_else(|e| e.into_inner());
                    // Keep pipelines stored while the file was being written.
                    l.added = l.added.saturating_sub(saved_additions);
                    tracing::info!(
                        bytes = blob.len(),
                        made = MADE.load(Ordering::Relaxed),
                        loaded = LOADED.load(Ordering::Relaxed),
                        started_early = EARLY.load(Ordering::Relaxed),
                        "host pipeline library saved"
                    );
                }
                Err(e) => tracing::warn!(error = %e, "host pipeline library not saved"),
            }
        }
    }

    // Snapshot under USED: a note arriving after the snapshot will leave
    // the dirty flag set for a later save. Failed writes also restore it.
    let snapshot = {
        let used = USED.lock().unwrap_or_else(|e| e.into_inner());
        USED_CHANGED.swap(false, Ordering::Relaxed).then(|| {
            let mut keys: Vec<&str> = used.iter().flatten().map(String::as_str).collect();
            keys.sort_unstable();
            (keys.join("\n"), keys.len())
        })
    };
    if let Some((bytes, count)) = snapshot {
        match std::fs::create_dir_all(dir)
            .and_then(|()| write_atomically(&dir.join(EARLY_FILE), bytes.as_bytes()))
        {
            Ok(()) => tracing::info!(early = count, "early pipeline list saved"),
            Err(e) => {
                USED_CHANGED.store(true, Ordering::Relaxed);
                tracing::warn!(error = %e, "early pipeline list not saved");
            }
        }
    }
}

fn serialize_if_dirty<T>(added: u64, serialize: impl FnOnce() -> Option<T>) -> Option<(T, u64)> {
    if added == 0 {
        return None;
    }
    serialize().map(|value| (value, added))
}

/// Writes `bytes` to a unique temporary file beside `path`, then renames it
/// over `path`: a reader never sees a part, and overlapping saves do not
/// share a temporary file.
fn write_atomically(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let extension = format!(
        "tmp-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    );
    let tmp = path.with_extension(extension);
    let written = std::fs::write(&tmp, bytes).and_then(|()| rename_over(&tmp, path));
    if written.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    written
}

/// Renames `from` over the file `to`. Windows answers "access denied" when
/// another rename onto `to` or a reader that opened it is in the way at that
/// moment, and the same call succeeds a moment later, so a few attempts are
/// made. A `to` that is a directory fails at once: no wait changes that.
fn rename_over(from: &std::path::Path, to: &std::path::Path) -> std::io::Result<()> {
    const ATTEMPTS: u32 = 20;
    let mut attempt = 0;
    loop {
        match std::fs::rename(from, to) {
            Err(e)
                if e.kind() == std::io::ErrorKind::PermissionDenied
                    && attempt < ATTEMPTS
                    && !to.is_dir() =>
            {
                attempt += 1;
                std::thread::sleep(std::time::Duration::from_millis(attempt.into()));
            }
            done => return done,
        }
    }
}

/// Title pipeline keys drawn or dispatched with (this run and the last).
static USED: Mutex<Option<HashSet<String>>> = Mutex::new(None);
static USED_CHANGED: AtomicBool = AtomicBool::new(false);

fn used() -> std::sync::MutexGuard<'static, Option<HashSet<String>>> {
    let mut used = USED.lock().unwrap_or_else(|e| e.into_inner());
    if used.is_none() {
        let listed = DIR
            .get()
            .and_then(|d| std::fs::read_to_string(d.join(EARLY_FILE)).ok())
            .unwrap_or_default();
        let set: HashSet<String> = listed.lines().filter(|l| !l.is_empty()).map(str::to_owned).collect();
        tracing::info!(pipelines = set.len(), "title pipelines to start early");
        *used = Some(set);
    }
    used
}

/// Whether the title pipeline with `key` was drawn with in an earlier run.
pub(crate) fn wanted_early(key: &str) -> bool {
    used().as_ref().is_some_and(|s| s.contains(key))
}

/// The title pipeline with `key` was drawn or dispatched with.
pub(crate) fn note_used(key: &str) {
    let mut used = used();
    let set = used.get_or_insert_with(HashSet::new);
    if set.len() < EARLY_MAX && !set.contains(key) {
        set.insert(key.to_owned());
        USED_CHANGED.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_library_never_calls_serializer() {
        assert_eq!(serialize_if_dirty::<Vec<u8>>(0, || panic!("clean library")), None);
        assert_eq!(serialize_if_dirty(3, || Some(vec![1u8, 2])), Some((vec![1, 2], 3)));
        assert_eq!(serialize_if_dirty::<Vec<u8>>(3, || None), None);
    }

    #[test]
    fn deferred_cache_stores_are_bounded_and_drained() {
        let library = Mutex::new(Vec::new());
        let queue = Mutex::new(Vec::new());
        store_or_defer(&library, &queue, 0usize, |l, item| l.push(item));
        let mut held = library.lock().unwrap();
        for n in 1..=DEFERRED_LIMIT + 1 {
            store_or_defer(&library, &queue, n, |l, item| l.push(item));
        }
        assert_eq!(held.as_slice(), &[0]);
        assert_eq!(queue.lock().unwrap().len(), DEFERRED_LIMIT);
        drain_deferred(&queue, |item| held.push(item));
        assert_eq!(*held, (0..=DEFERRED_LIMIT).collect::<Vec<_>>());
        assert!(queue.lock().unwrap().is_empty());
    }

    #[test]
    fn a_periodic_save_does_not_wait_for_another_save() {
        let lock = Mutex::new(0);
        let held = lock.lock().unwrap();
        std::thread::scope(|scope| {
            scope.spawn(|| assert!(save_guard(&lock, false).is_none()))
                .join()
                .unwrap();
        });
        drop(held);
        assert!(save_guard(&lock, false).is_some());
        assert!(save_guard(&lock, true).is_some());
    }

    #[test]
    fn names_change_with_what_goes_into_the_pipeline() {
        let a = name("graphics", b"root", &[b"vs", b"ps"], "state");
        assert_eq!(a, name("graphics", b"root", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps2"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps"], "state2"));
        assert_ne!(a, name("graphics", b"root2", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("mesh", b"root", &[b"vs", b"ps"], "state"));
        // Parts do not run into each other.
        assert_ne!(a, name("graphics", b"root", &[b"vsp", b"s"], "state"));
    }

    #[test]
    fn a_rejected_blob_gives_an_empty_library() {
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok()) else {
            return; // no host GPU here
        };
        let library = open(&device, b"not a pipeline library".repeat(16)).expect("an empty library");
        // SAFETY: a plain query on a live library.
        assert!(unsafe { library.library.GetSerializedSize() } > 0);
        assert!(library._blob.is_empty());
    }

    #[test]
    fn concurrent_atomic_writes_publish_one_complete_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(LIBRARY_FILE);
        let start = std::sync::Arc::new(std::sync::Barrier::new(8));
        let mut threads = Vec::new();
        for thread in 0..8 {
            let path = path.clone();
            let start = start.clone();
            threads.push(std::thread::spawn(move || {
                start.wait();
                let byte = thread as u8 + 1;
                for _ in 0..32 {
                    write_atomically(&path, &[byte; 64 * 1024]).unwrap();
                }
            }));
        }
        for thread in threads {
            thread.join().unwrap();
        }

        let published = std::fs::read(&path).unwrap();
        assert_eq!(published.len(), 64 * 1024);
        assert!(published.iter().all(|&byte| byte == published[0]));
        assert!((1..=8).contains(&published[0]));
        assert!(!std::fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .any(|entry| entry.file_name().to_string_lossy().contains(".tmp-")));
    }

    #[test]
    fn failed_atomic_writes_remove_their_temporary_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(LIBRARY_FILE);
        std::fs::create_dir(&path).unwrap();

        assert!(write_atomically(&path, b"not published").is_err());
        assert!(!std::fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .any(|entry| entry.file_name().to_string_lossy().contains(".tmp-")));
    }
}
