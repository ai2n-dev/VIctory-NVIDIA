//! A persistent pool of helper threads for fork-join work on borrowed data.
//!
//! [`broadcast`] runs one closure on the calling thread and on up to a given number
//! of pool threads, and returns once every thread that joined has left it. The
//! caller always does a share of the work itself, so a call finishes even when
//! every helper is busy, and calls nested in a running closure cannot
//! deadlock.

use std::any::Any;
use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};

/// Most threads, the caller included, that work on one call. Copying between
/// memory pools stops scaling well before the core count of a large desktop
/// part, and extra threads only fight over memory bandwidth and the shared
/// caches.
const MAX_THREADS: usize = 8;

/// Overrides the number of threads (the caller included) for tuning on a
/// given CPU.
const THREADS_VARIABLE: &str = "VICTORY_COPY_THREADS";

type Body = dyn Fn() + Sync;

/// One call of [`broadcast`].
struct Job {
    /// The closure of the call, with its lifetime erased. It is only used by
    /// threads counted in `State::running`, and the caller waits for that
    /// count to reach zero before the closure goes out of scope.
    body: *const Body,
    state: Mutex<State>,
    left: Condvar,
}

struct State {
    /// Whether new helpers may still join.
    open: bool,
    /// Helpers inside the closure.
    running: usize,
    /// The first panic of a helper, passed on to the caller.
    panic: Option<Box<dyn Any + Send>>,
}

// SAFETY: `body` points at a `Sync` closure and is dereferenced only while the
// caller is blocked in `Job::close`, which keeps the closure alive.
unsafe impl Send for Job {}
// SAFETY: as for `Send`; the rest of the job is behind a mutex.
unsafe impl Sync for Job {}

impl Job {
    /// Joins the job if it is still open and runs the closure to its end.
    fn help(&self) {
        {
            let mut state = lock(&self.state);
            if !state.open {
                return;
            }
            state.running += 1;
        }
        // SAFETY: registered in `running` while the job was open, so `close`
        // has not returned and the closure is alive.
        let body = unsafe { &*self.body };
        let outcome = catch_unwind(AssertUnwindSafe(body));
        let mut state = lock(&self.state);
        if let Err(payload) = outcome {
            state.panic.get_or_insert(payload);
        }
        state.running -= 1;
        if state.running == 0 {
            self.left.notify_all();
        }
    }

    /// Stops new helpers from joining and waits for those inside to leave.
    fn close(&self) {
        let mut state = lock(&self.state);
        state.open = false;
        while state.running > 0 {
            state = self.left.wait(state).unwrap_or_else(|e| e.into_inner());
        }
    }
}

/// Closes the job when the caller leaves, also by unwinding, so that no
/// helper outlives the closure it runs.
struct CloseOnExit<'a>(&'a Job);

impl Drop for CloseOnExit<'_> {
    fn drop(&mut self) {
        self.0.close();
    }
}

/// One entry per helper a job asked for.
static TICKETS: Mutex<VecDeque<Arc<Job>>> = Mutex::new(VecDeque::new());
static WAKE: Condvar = Condvar::new();
static HELPERS: OnceLock<usize> = OnceLock::new();

/// The number of threads a call may use, the caller included.
pub(crate) fn threads() -> usize {
    helpers() + 1
}

/// The number of pool threads, started on first use.
fn helpers() -> usize {
    *HELPERS.get_or_init(|| {
        let helpers = configured_threads() - 1;
        let started = (0..helpers)
            .filter(|index| {
                std::thread::Builder::new()
                    .name(format!("victory-copy {index}"))
                    .spawn(serve)
                    .is_ok()
            })
            .count();
        tracing::debug!(helpers = started, "work pool: helpers");
        started
    })
}

fn configured_threads() -> usize {
    let requested = std::env::var(THREADS_VARIABLE)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok());
    let available = std::thread::available_parallelism().map_or(4, |n| n.get());
    requested.unwrap_or(available).clamp(1, MAX_THREADS)
}

/// A helper: joins one job after another for as long as the process lives.
fn serve() {
    loop {
        let job = {
            let mut tickets = lock(&TICKETS);
            loop {
                if let Some(job) = tickets.pop_front() {
                    break job;
                }
                tickets = WAKE.wait(tickets).unwrap_or_else(|e| e.into_inner());
            }
        };
        job.help();
    }
}

/// Runs `work` on this thread and on up to `extra` pool threads at once,
/// and returns when all of them have left it. `work` is expected to split the
/// job between its callers (for example by taking pieces from a shared
/// counter), because each thread runs it once. A panic of any thread is
/// raised here after the others are done.
pub(crate) fn broadcast(extra: usize, work: &(dyn Fn() + Sync)) {
    let extra = extra.min(helpers());
    if extra == 0 {
        return work();
    }
    // SAFETY: only the lifetime bound of the pointee changes. `CloseOnExit`
    // below waits for every thread that took the pointer before `work` is
    // released.
    let body: *const Body = unsafe {
        std::mem::transmute::<*const (dyn Fn() + Sync + '_), *const Body>(work)
    };
    let job = Arc::new(Job {
        body,
        state: Mutex::new(State {
            open: true,
            running: 0,
            panic: None,
        }),
        left: Condvar::new(),
    });
    lock(&TICKETS).extend((0..extra).map(|_| job.clone()));
    for _ in 0..extra {
        WAKE.notify_one();
    }

    let closing = CloseOnExit(&job);
    work();
    drop(closing);

    // Tickets nobody took are of no use any more.
    lock(&TICKETS).retain(|ticket| !Arc::ptr_eq(ticket, &job));
    if let Some(payload) = lock(&job.state).panic.take() {
        resume_unwind(payload);
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Splits `items` in pieces of `piece` and adds them up on the threads of
    /// one call.
    fn parallel_sum(items: &[u64], piece: usize) -> u64 {
        let next = AtomicUsize::new(0);
        let total = std::sync::atomic::AtomicU64::new(0);
        broadcast(MAX_THREADS, &|| {
            loop {
                let from = next.fetch_add(piece, Ordering::Relaxed);
                if from >= items.len() {
                    break;
                }
                let to = (from + piece).min(items.len());
                total.fetch_add(items[from..to].iter().sum(), Ordering::Relaxed);
            }
        });
        total.into_inner()
    }

    #[test]
    fn borrowed_data_is_processed_exactly_once() {
        let items: Vec<u64> = (1..=100_000).collect();
        assert_eq!(parallel_sum(&items, 97), 100_000 * 100_001 / 2);
    }

    #[test]
    fn a_call_without_helpers_runs_on_the_caller() {
        let runs = AtomicUsize::new(0);
        broadcast(0, &|| {
            runs.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(runs.into_inner(), 1);
    }

    #[test]
    fn no_thread_is_still_inside_after_the_call() {
        for _ in 0..200 {
            let inside = AtomicUsize::new(0);
            let entered = AtomicUsize::new(0);
            broadcast(MAX_THREADS, &|| {
                inside.fetch_add(1, Ordering::SeqCst);
                entered.fetch_add(1, Ordering::SeqCst);
                std::thread::yield_now();
                inside.fetch_sub(1, Ordering::SeqCst);
            });
            assert_eq!(inside.load(Ordering::SeqCst), 0);
            assert!(entered.load(Ordering::SeqCst) >= 1);
        }
    }

    #[test]
    fn concurrent_callers_do_not_mix_their_jobs() {
        std::thread::scope(|scope| {
            for caller in 0..6u64 {
                scope.spawn(move || {
                    let items: Vec<u64> = (0..20_000).map(|i| i * (caller + 1)).collect();
                    let expected: u64 = items.iter().sum();
                    for _ in 0..50 {
                        assert_eq!(parallel_sum(&items, 31), expected);
                    }
                });
            }
        });
    }

    #[test]
    fn nested_calls_finish_when_every_helper_is_busy() {
        let outer = AtomicUsize::new(0);
        let inner = AtomicUsize::new(0);
        broadcast(MAX_THREADS, &|| {
            outer.fetch_add(1, Ordering::Relaxed);
            broadcast(MAX_THREADS, &|| {
                inner.fetch_add(1, Ordering::Relaxed);
            });
        });
        assert!(outer.into_inner() >= 1);
        assert!(inner.into_inner() >= 1);
    }

    #[test]
    fn a_helper_panic_reaches_the_caller_and_the_pool_survives() {
        let calling = std::thread::current().id();
        let outcome = catch_unwind(|| {
            broadcast(MAX_THREADS, &|| {
                if std::thread::current().id() != calling {
                    panic!("helper failed");
                }
                // Give the helpers time to join.
                std::thread::sleep(std::time::Duration::from_millis(50));
            });
        });
        if helpers() > 0 {
            assert!(outcome.is_err());
        }
        let items: Vec<u64> = (1..=1000).collect();
        assert_eq!(parallel_sum(&items, 7), 500_500);
    }

    #[test]
    fn a_caller_panic_waits_for_the_helpers() {
        let inside = AtomicUsize::new(0);
        let calling = std::thread::current().id();
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            broadcast(MAX_THREADS, &|| {
                if std::thread::current().id() == calling {
                    panic!("caller failed");
                }
                inside.fetch_add(1, Ordering::SeqCst);
                std::thread::sleep(std::time::Duration::from_millis(20));
                inside.fetch_sub(1, Ordering::SeqCst);
            });
        }));
        assert!(outcome.is_err());
        assert_eq!(inside.load(Ordering::SeqCst), 0);
    }
}
