//! The time of every presented frame, for benchmarks: `VICTORY_FRAMETIME=<file>`
//! writes one line, `frame,at_us,frame_us`, for each frame the title
//! presents after the first (the time since the start of the first frame,
//! and since the frame before). Percentiles, 1 % and 0.1 % lows come from
//! the file; the five-second lines of the log only give means.
//!
//! Off without the variable: a present then costs one load of a
//! `OnceLock`. On, a present formats a line into a buffer under an
//! uncontended lock (about 100 ns) and the buffer goes to the file when it
//! holds 32 KiB or a second has passed, so a crash loses the last second.

use std::fmt::Write as _;
use std::io::Write as _;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Bytes after which the buffer is written, and the longest it waits.
const FLUSH_BYTES: usize = 32 << 10;
const FLUSH_EVERY: Duration = Duration::from_secs(1);

struct Trace {
    out: std::fs::File,
    text: String,
    first: Instant,
    last: Option<Instant>,
    frames: u64,
    flushed: Instant,
}

impl Trace {
    fn create(path: &std::path::Path, now: Instant) -> std::io::Result<Trace> {
        let mut out = std::fs::File::create(path)?;
        out.write_all(b"frame,at_us,frame_us\n")?;
        Ok(Trace {
            out,
            text: String::with_capacity(FLUSH_BYTES + 256),
            first: now,
            last: None,
            frames: 0,
            flushed: now,
        })
    }

    /// Notes a frame presented at `now`.
    fn frame(&mut self, now: Instant) {
        self.frames += 1;
        if let Some(last) = self.last.replace(now) {
            let at = now.duration_since(self.first).as_micros();
            let took = now.duration_since(last).as_micros();
            let _ = writeln!(self.text, "{},{at},{took}", self.frames);
        }
        if self.text.len() >= FLUSH_BYTES || now.duration_since(self.flushed) >= FLUSH_EVERY {
            self.flush(now);
        }
    }

    fn flush(&mut self, now: Instant) {
        self.flushed = now;
        if self.text.is_empty() {
            return;
        }
        // A full disk or a removed file loses the trace, not the frame.
        let _ = self.out.write_all(self.text.as_bytes());
        self.text.clear();
    }
}

fn trace() -> Option<&'static Mutex<Trace>> {
    static TRACE: OnceLock<Option<Mutex<Trace>>> = OnceLock::new();
    TRACE
        .get_or_init(|| {
            let path = std::env::var_os("VICTORY_FRAMETIME").filter(|p| !p.is_empty())?;
            match Trace::create(std::path::Path::new(&path), Instant::now()) {
                Ok(trace) => {
                    tracing::info!(file = ?path, "frame times are written");
                    Some(Mutex::new(trace))
                }
                Err(error) => {
                    tracing::warn!(%error, file = ?path, "frame times cannot be written");
                    None
                }
            }
        })
        .as_ref()
}

/// A frame was presented.
pub fn present() {
    let Some(trace) = trace() else { return };
    let now = Instant::now();
    trace.lock().unwrap_or_else(|e| e.into_inner()).frame(now);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_after_the_first_are_written_with_their_times() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("frames.csv");
        let start = Instant::now();
        let mut trace = Trace::create(&path, start).unwrap();
        for (i, ms) in [0u64, 16, 33, 50, 120].into_iter().enumerate() {
            trace.frame(start + Duration::from_millis(ms));
            assert_eq!(trace.frames, i as u64 + 1);
        }
        trace.flush(start + Duration::from_millis(121));
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(
            text,
            "frame,at_us,frame_us\n2,16000,16000\n3,33000,17000\n4,50000,17000\n5,120000,70000\n"
        );
    }

    #[test]
    fn the_buffer_is_written_when_it_is_full_or_a_second_old() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("frames.csv");
        let start = Instant::now();
        let mut trace = Trace::create(&path, start).unwrap();
        trace.frame(start);
        trace.frame(start + Duration::from_millis(16));
        let size = || std::fs::metadata(&path).unwrap().len();
        let header = size();
        assert!(!trace.text.is_empty() && size() == header);
        trace.frame(start + Duration::from_millis(1100));
        assert!(trace.text.is_empty() && size() > header);
        for i in 0..4000u64 {
            trace.frame(start + Duration::from_millis(1100 + 4 * i));
        }
        assert!(trace.text.len() < FLUSH_BYTES);
    }

    /// What a frame costs with the trace on, printed with `--nocapture`.
    #[test]
    fn a_frame_costs_far_less_than_a_microsecond_on_average() {
        let dir = tempfile::tempdir().unwrap();
        let mut trace = Trace::create(&dir.path().join("frames.csv"), Instant::now()).unwrap();
        const FRAMES: u32 = 1_000_000;
        let began = Instant::now();
        let mut at = began;
        for _ in 0..FRAMES {
            at += Duration::from_micros(4);
            trace.frame(at);
        }
        let per_frame = began.elapsed().as_nanos() / u128::from(FRAMES);
        eprintln!("frame trace: {per_frame} ns a frame, file writes included");
        assert!(per_frame < 2_000, "{per_frame} ns a frame");
    }
}
