//! Benchmark of the work pool: `cargo test --release -p victory-gpu --lib
//! workpool::bench -- --ignored --nocapture`.
//!
//! It prints, for each thread count (the caller included), the speed of
//! the copies the pool serves (large and small pieces of title memory) and
//! the cost of one call that has almost nothing to do. `VICTORY_COPY_THREADS`
//! sets how many threads the pool has, the sweep goes up to that number.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use super::{broadcast, threads};

const ROUNDS: usize = 9;

/// Runs `round` `ROUNDS` times and returns the median.
fn median(mut round: impl FnMut() -> Duration) -> Duration {
    let mut times: Vec<Duration> = (0..ROUNDS).map(|_| round()).collect();
    times.sort();
    times[ROUNDS / 2]
}

/// Copies `source` to `target` in pieces of `piece` bytes taken in turn by
/// `threads` threads.
fn copy_with(threads: usize, source: &[u8], target: &mut [u8], piece: usize) -> Duration {
    let next = AtomicUsize::new(0);
    let to = target.as_mut_ptr() as usize;
    let started = Instant::now();
    broadcast(threads - 1, &|| {
        loop {
            let from = next.fetch_add(piece, Ordering::Relaxed);
            if from >= source.len() {
                break;
            }
            let len = piece.min(source.len() - from);
            // SAFETY: each piece is taken by one thread and lies inside both
            // buffers, which do not overlap.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    source.as_ptr().add(from),
                    (to as *mut u8).add(from),
                    len,
                )
            };
        }
    });
    started.elapsed()
}

/// A call whose work is a handful of counter bumps: the cost of the pool.
fn empty_call(threads: usize) -> Duration {
    const CALLS: u32 = 2000;
    let counter = AtomicUsize::new(0);
    let started = Instant::now();
    for _ in 0..CALLS {
        broadcast(threads - 1, &|| {
            counter.fetch_add(1, Ordering::Relaxed);
        });
    }
    started.elapsed() / CALLS
}

#[test]
#[ignore = "a benchmark: run it in release mode with --nocapture"]
fn sweep_thread_counts() {
    let most = threads();
    println!(
        "pool of {most} threads, {} logical processors",
        std::thread::available_parallelism().map_or(0, |n| n.get())
    );
    const SIZE: usize = 256 << 20;
    let source: Vec<u8> = (0..SIZE).map(|i| i as u8).collect();
    let mut target = vec![0u8; SIZE];
    // Touch every page once, so that no round pays for faulting them in.
    target.copy_from_slice(&source);

    // The same 16 MiB again and again: it stays in the processor's caches.
    const CACHED: usize = 16 << 20;
    const CACHED_REPEATS: usize = 16;

    println!(
        "threads  256 MiB in 1 MiB pieces  256 MiB in 64 KiB pieces  16 MiB cached, 64 KiB pieces  empty call"
    );
    for threads in 1..=most {
        let mut cells = Vec::new();
        for piece in [1 << 20, 64 << 10] {
            let time = median(|| copy_with(threads, &source, &mut target, piece));
            cells.push(format!(
                "{:8.1} GB/s",
                SIZE as f64 / time.as_secs_f64() / 1e9
            ));
        }
        let cached = median(|| {
            (0..CACHED_REPEATS)
                .map(|_| copy_with(threads, &source[..CACHED], &mut target[..CACHED], 64 << 10))
                .sum()
        });
        let cached = format!(
            "{:8.1} GB/s",
            (CACHED * CACHED_REPEATS) as f64 / cached.as_secs_f64() / 1e9
        );
        let call = median(|| empty_call(threads));
        println!(
            "{threads:7}  {:>22}  {:>24}  {cached:>27}  {call:>9.1?}",
            cells[0], cells[1]
        );
    }
    assert_eq!(source, target);
}
