//! Host pipelines are made by workers, not by the title thread that first
//! records a draw with them. Translating a shader and the host driver's
//! compile take from milliseconds to seconds each (structurisation of the
//! biggest compute programs ~0.4 s, some driver compiles 3.4 s), and Grand
//! Theft Auto VI's first world frame wants over 1,200 new pipelines: made
//! on its recording threads, that frame took more than 30 s and the
//! title's GPU watchdog ended the run (decision 0007, "Pipelines made off
//! the title's threads").
//!
//! A first use queues the pipeline and waits for it within a budget per
//! frame, so the cheap ones (menus, one-off passes such as a colour-grading
//! bake) are there at once as before; past the budget a draw or dispatch
//! whose pipeline is not ready is dropped, as an untranslated one is.
//!
//! Passes over the screen (no vertex input, no depth) have workers and a
//! longer budget of their own: the title fills its lighting tables with
//! such draws once, at the world's first frame, and behind a world's worth
//! of pipelines they waited minutes; a run that lost them had no diffuse
//! light at all.
//!
//! shortcut: the budgets (half a second of waiting per title frame, 20 s
//! for screen passes) are guesses; another pass the title runs only once
//! and whose pipeline takes longer is lost.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::Shader;
use super::host::{GraphicsState, HostPipeline};

/// What a host pipeline is made from, and the host pipeline once made.
pub(crate) struct Pending {
    shaders: Vec<Arc<Shader>>,
    pub(crate) graphics: Option<Box<GraphicsState>>,
    queued: AtomicBool,
    made: OnceLock<Option<HostPipeline>>,
    /// Its worker was lost to it: it will not be made.
    lost: AtomicBool,
    /// A pass over the screen (see [`GraphicsState::screen_pass`]).
    screen: bool,
    /// What identifies it across runs (shader containers and state), and
    /// whether a draw has used it yet.
    key: String,
    used: AtomicBool,
    /// Reserved or spent lifetime wait in the optional bounded policy.
    wait_us: AtomicU64,
}

// SAFETY: D3D12 objects are free-threaded; the graphics state (whose input
// elements point into its own names) is only read after creation.
unsafe impl Send for Pending {}
// SAFETY: as above.
unsafe impl Sync for Pending {}

impl Pending {
    pub(crate) fn new(shaders: Vec<Arc<Shader>>, graphics: Option<Box<GraphicsState>>) -> Pending {
        let mut key = blake3::Hasher::new();
        for s in &shaders {
            key.update(s.stage.as_bytes());
            key.update(s.hash.as_bytes());
        }
        key.update(graphics.as_ref().map(|g| g.describe()).unwrap_or_default().as_bytes());
        Pending {
            screen: graphics.as_ref().is_some_and(|g| g.screen_pass()),
            shaders,
            graphics,
            queued: AtomicBool::new(false),
            made: OnceLock::new(),
            lost: AtomicBool::new(false),
            key: key.finalize().to_hex()[..32].to_owned(),
            used: AtomicBool::new(false),
            wait_us: AtomicU64::new(0),
        }
    }

    fn make(&self) {
        // Not `get_or_init`: a worker lost in it would leave it locked.
        let made = HostPipeline::new(&self.shaders, self.graphics.as_deref());
        let _ = self.made.set(made);
        let (lock, done) = &DONE;
        let _held = lock.lock().unwrap_or_else(|e| e.into_inner());
        done.notify_all();
    }
}

/// Why a pipeline has no host pipeline (diagnostics): its shaders that do
/// not translate, else "refused" or "being made"; `None` if it has one.
#[cfg(feature = "diagnostics")]
pub(crate) fn missing(p: &Pending) -> Option<String> {
    match p.made.get() {
        Some(Some(_)) => None,
        None => Some("being made".into()),
        Some(None) => {
            let failed: Vec<String> = p
                .shaders
                .iter()
                .filter(|s| matches!(s.translated.get(), Some(None)))
                .map(|s| format!("{} {}", s.stage, s.hash))
                .collect();
            Some(if failed.is_empty() {
                format!("refused {:?}", p.shaders)
            } else {
                failed.join(" + ")
            })
        }
    }
}

/// Signalled whenever a worker has made a pipeline.
static DONE: (Mutex<()>, Condvar) = (Mutex::new(()), Condvar::new());

/// Waiting allowed per title frame, over all threads.
const BUDGET: Duration = Duration::from_millis(500);

/// The same for passes over the screen (below the title's 30 s watchdog).
const SCREEN_BUDGET: Duration = Duration::from_secs(20);

/// Pipelines queued and not made yet.
static WAITING: AtomicU64 = AtomicU64::new(0);

/// Pipelines queued and not made yet (for the frame line).
pub(crate) fn waiting() -> u64 {
    WAITING.load(Ordering::Relaxed)
}

/// The host pipeline if it is made (or is made within the frame's budget);
/// queues it on first use.
pub(crate) fn ready(p: &Arc<Pending>) -> Option<&HostPipeline> {
    if p.made.get().is_none() {
        start(p);
        if p.lost.load(Ordering::Relaxed) {
            return None;
        }
        // Also with a world's worth queued: passes the title runs only once
        // set up what later ones loop over.
        wait(p, crate::frame::current());
    }
    let made = p.made.get()?.as_ref();
    // A load first: after the first draw this only reads.
    if made.is_some() && !p.used.load(Ordering::Relaxed) && !p.used.swap(true, Ordering::Relaxed) {
        super::library::note_used(&p.key);
    }
    made
}

/// Queues `p` for the workers, once.
fn start(p: &Arc<Pending>) {
    // A load first: every draw that finds it not made yet comes here.
    if !p.queued.load(Ordering::Relaxed) && !p.queued.swap(true, Ordering::Relaxed) {
        PEAK.fetch_max(WAITING.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);
        queue(p.clone());
    }
}

/// At the title's creation of a pipeline: starts it at once if an earlier
/// run drew with it (during loading, so the world's first frames find it).
pub(crate) fn early(p: &Arc<Pending>) {
    if !p.shaders.is_empty() && super::library::wanted_early(&p.key) {
        super::library::EARLY.fetch_add(1, Ordering::Relaxed);
        start(p);
    }
}

/// Inspired by trustno1-coder/VIctory-NVIDIA 77e13a97 (GPL-3.0).
/// In the experimental policy a pending pipeline cannot consume a fresh
/// full budget every frame. Reserve both budgets before sleeping so
/// concurrent recording threads cannot all spend the same allowance.
fn bounded_wait_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_WAIT_ONCE").is_some_and(|v| v == "1"))
}

#[derive(Default)]
struct WaitBudget {
    frame: Option<u64>,
    spent: u64,
}

impl WaitBudget {
    // Every access to own is under the same per-kind mutex.
    fn reserve(&mut self, frame: u64, budget: u64, own: &AtomicU64) -> u64 {
        if self.frame.is_some_and(|current| frame < current) {
            return 0;
        }
        if self.frame != Some(frame) {
            self.frame = Some(frame);
            self.spent = 0;
        }
        let left = budget
            .saturating_sub(self.spent)
            .min(budget.saturating_sub(own.load(Ordering::Relaxed)));
        self.spent += left;
        own.fetch_add(left, Ordering::Relaxed);
        left
    }

    fn settle(&mut self, frame: u64, own: &AtomicU64, reserved: u64, elapsed: u64) {
        let unused = reserved.saturating_sub(elapsed);
        own.fetch_sub(unused, Ordering::Relaxed);
        // A waiter from a previous frame must never refund the new frame.
        if self.frame == Some(frame) {
            self.spent = self.spent.saturating_sub(unused);
        }
    }
}

fn wait_bounded(p: &Pending, frame: u64) {
    static SPENT: [Mutex<WaitBudget>; 2] =
        [const { Mutex::new(WaitBudget { frame: None, spent: 0 }) }; 2];
    let kind = usize::from(p.screen);
    let budget = [BUDGET, SCREEN_BUDGET][kind].as_micros() as u64;
    if p.wait_us.load(Ordering::Relaxed) >= budget {
        return;
    }
    let reserved = SPENT[kind]
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .reserve(frame, budget, &p.wait_us);
    if reserved == 0 {
        return;
    }
    let elapsed = wait_for_pipeline(p, Duration::from_micros(reserved));
    SPENT[kind]
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .settle(frame, &p.wait_us, reserved, elapsed);
}

/// Waits for `p` while frame `frame` has budget of its kind left.
fn wait(p: &Pending, frame: u64) {
    if bounded_wait_on() {
        wait_bounded(p, frame);
        return;
    }
    static FRAME: [AtomicU64; 2] = [const { AtomicU64::new(u64::MAX) }; 2];
    static WAITED_US: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
    let kind = usize::from(p.screen);
    let budget = [BUDGET, SCREEN_BUDGET][kind];
    let waited = &WAITED_US[kind];
    if FRAME[kind].swap(frame, Ordering::Relaxed) != frame {
        waited.store(0, Ordering::Relaxed);
    }
    let left = budget.saturating_sub(Duration::from_micros(waited.load(Ordering::Relaxed)));
    // Spent: the loop cannot run, and time added to a spent budget changes
    // nothing, so neither the lock nor the clock is needed.
    if left.is_zero() {
        return;
    }
    let elapsed = wait_for_pipeline(p, left);
    waited.fetch_add(elapsed, Ordering::Relaxed);
}

fn wait_for_pipeline(p: &Pending, left: Duration) -> u64 {
    let started = Instant::now();
    let (lock, done) = &DONE;
    let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
    while p.made.get().is_none() {
        let rest = left.saturating_sub(started.elapsed());
        if rest.is_zero() {
            break;
        }
        held = done.wait_timeout(held, rest).unwrap_or_else(|e| e.into_inner()).0;
    }
    let elapsed = started.elapsed().as_micros() as u64;
    crate::framelog::add(crate::framelog::T::PipelineWaitUs, elapsed);
    elapsed
}

/// What the workers take from: the ordinary pipelines, the screen passes.
static RECEIVER: [OnceLock<Mutex<std::sync::mpsc::Receiver<Arc<Pending>>>>; 2] =
    [const { OnceLock::new() }; 2];

/// What the workers (by thread) are making.
static AT_WORK: Mutex<Vec<(u32, Arc<Pending>)>> = Mutex::new(Vec::new());

/// Starts worker `i`, for screen passes or for the others.
fn worker(i: usize, screen: bool) {
    let spawned = std::thread::Builder::new()
        .name(format!("victory-pipelines-{i}"))
        .spawn(move || {
            // Below the title's own threads (loading, recording).
            // SAFETY: a plain call on our own thread.
            unsafe {
                use windows::Win32::System::Threading::*;
                let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
            }
            let Some(receiver) = RECEIVER[usize::from(screen)].get() else { return };
            loop {
                let next = receiver.lock().unwrap_or_else(|e| e.into_inner()).recv();
                let Ok(p) = next else { return };
                // SAFETY: a plain call.
                let id = unsafe { windows::Win32::System::Threading::GetCurrentThreadId() };
                AT_WORK.lock().unwrap_or_else(|e| e.into_inner()).push((id, p.clone()));
                p.make();
                AT_WORK.lock().unwrap_or_else(|e| e.into_inner()).retain(|w| w.0 != id);
                if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
                    drained();
                } else {
                    super::library::save(false);
                }
            }
        });
    if let Err(e) = spawned {
        tracing::warn!(error = %e, "no pipeline worker");
    }
}

/// A worker is lost to a compile that crashed or never ends: another takes
/// its place (its pipeline stays unmade). With all of them lost, the rest of
/// a world's pipelines waited forever.
pub(super) fn worker_lost(id: u32) {
    static LOST: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(100);
    let mut lost = None;
    AT_WORK.lock().unwrap_or_else(|e| e.into_inner()).retain(|w| {
        if w.0 == id {
            w.1.lost.store(true, Ordering::Relaxed);
            lost = Some(w.1.screen);
        }
        w.0 != id
    });
    if let Some(screen) = lost {
        WAITING.fetch_sub(1, Ordering::Relaxed);
        worker(LOST.fetch_add(1, Ordering::Relaxed), screen);
    }
}

/// Hands `p` to the workers (half the logical processors, 2 to 8, and two
/// for screen passes), started on first use.
fn queue(p: Arc<Pending>) {
    static SENDER: [OnceLock<std::sync::mpsc::Sender<Arc<Pending>>>; 2] =
        [const { OnceLock::new() }; 2];
    let screen = p.screen;
    let sender = SENDER[usize::from(screen)].get_or_init(|| {
        let (sender, receiver) = std::sync::mpsc::channel::<Arc<Pending>>();
        let _ = RECEIVER[usize::from(screen)].set(Mutex::new(receiver));
        let workers = if screen {
            2
        } else {
            std::thread::available_parallelism().map_or(2, |n| (n.get() / 2).clamp(2, 8))
        };
        for i in 0..workers {
            worker(i + if screen { 50 } else { 0 }, screen);
        }
        sender
    });
    if let Err(e) = sender.send(p) {
        // No worker: made here instead.
        e.0.make();
        WAITING.fetch_sub(1, Ordering::Relaxed);
    }
}

/// The most pipelines waiting at once since the workers last ran out.
static PEAK: AtomicU64 = AtomicU64::new(0);

/// The workers ran out of work: logged after a batch of 10 or more, and
/// the cache saved (at most every 20 s).
fn drained() {
    use super::library::{EARLY, LOADED, MADE};
    if PEAK.swap(0, Ordering::Relaxed) >= 10 {
        tracing::info!(
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "host pipelines all made"
        );
    }
    super::library::save(false);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_wait_reserves_both_budgets_and_refunds_only_unused_time() {
        let mut shared = WaitBudget::default();
        let a = AtomicU64::new(0);
        let b = AtomicU64::new(0);
        assert_eq!(shared.reserve(1, 100, &a), 100);
        assert_eq!(shared.reserve(1, 100, &b), 0);
        shared.settle(1, &a, 100, 40);
        assert_eq!(shared.reserve(1, 100, &b), 60);
        shared.settle(1, &b, 60, 80);
        assert_eq!(shared.reserve(2, 100, &a), 60);
        shared.settle(2, &a, 60, 60);
        assert_eq!(shared.reserve(3, 100, &a), 0);
        assert_eq!(shared.reserve(3, 100, &b), 40);
    }

    #[test]
    fn old_waiter_does_not_refund_or_reopen_new_frame() {
        let mut shared = WaitBudget::default();
        let a = AtomicU64::new(0);
        let b = AtomicU64::new(0);
        assert_eq!(shared.reserve(1, 100, &a), 100);
        assert_eq!(shared.reserve(2, 100, &b), 100);
        shared.settle(1, &a, 100, 10);
        assert_eq!(a.load(Ordering::Relaxed), 10);
        assert_eq!(shared.reserve(2, 100, &a), 0);
        assert_eq!(shared.reserve(1, 100, &a), 0);
        shared.settle(2, &b, 100, 20);
        assert_eq!(shared.reserve(2, 100, &a), 80);
    }

    /// A first use queues the pipeline and, within the budget, gets the
    /// worker's answer (here: no host pipeline for no shaders).
    #[test]
    fn first_use_is_made_by_a_worker() {
        let p = Arc::new(Pending::new(Vec::new(), None));
        assert!(ready(&p).is_none());
        assert!(p.queued.load(Ordering::Relaxed));
        let started = Instant::now();
        while p.made.get().is_none() && started.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        assert!(matches!(p.made.get(), Some(None)));
    }
}
