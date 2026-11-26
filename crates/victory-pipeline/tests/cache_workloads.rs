//! Synthetic, CPU-only cache workloads. No title resources or GPU required.
//! Timing is reported for observation, not used as a flaky CI threshold.
use std::path::Path;
use std::sync::Barrier;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use victory_pipeline::{Cache, Converter};

const WORKERS: usize = 8;
const UNIQUE: usize = 64;

struct Synthetic {
    version: u32,
    runs: AtomicUsize,
}

fn payload(id: usize) -> Vec<u8> {
    let mut bytes = vec![0x5a; 4096];
    bytes[..8].copy_from_slice(&(id as u64).to_le_bytes());
    bytes
}

impl Converter for Synthetic {
    fn id(&self) -> &'static str {
        "test.cache_workload"
    }

    fn version(&self) -> u32 {
        self.version
    }

    fn convert(&self, input: &[u8]) -> anyhow::Result<Vec<u8>> {
        self.runs.fetch_add(1, Ordering::Relaxed);
        Ok(blake3::hash(input).as_bytes().to_vec())
    }
}

fn batch(root: &Path, converter: &Synthetic, first: usize) -> Duration {
    let start = Barrier::new(WORKERS);
    let began = Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..WORKERS {
            scope.spawn(|| {
                let cache = Cache::open(root).unwrap();
                start.wait();
                for id in first..first + UNIQUE {
                    let input = payload(id);
                    let path = cache.get_or_convert(converter, &input).unwrap();
                    let expected = blake3::hash(&input);
                    assert_eq!(std::fs::read(path).unwrap().as_slice(), expected.as_bytes());
                }
            });
        }
    });
    began.elapsed()
}

#[test]
fn cold_warm_and_partially_changed_workloads_convert_only_missing_keys() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("cache");
    let converter = Synthetic {
        version: 1,
        runs: AtomicUsize::new(0),
    };

    let cold = batch(&root, &converter, 0);
    assert_eq!(converter.runs.swap(0, Ordering::Relaxed), UNIQUE);

    // Reopen the cache and converter independently, as on the next launch.
    let converter = Synthetic {
        version: 1,
        runs: AtomicUsize::new(0),
    };
    let warm = batch(&root, &converter, 0);
    assert_eq!(converter.runs.swap(0, Ordering::Relaxed), 0);

    let mixed = batch(&root, &converter, UNIQUE / 2);
    assert_eq!(converter.runs.load(Ordering::Relaxed), UNIQUE / 2);

    let incompatible = Synthetic {
        version: 2,
        runs: AtomicUsize::new(0),
    };
    let new_version = batch(&root, &incompatible, 0);
    assert_eq!(incompatible.runs.load(Ordering::Relaxed), UNIQUE);

    // Version 2 must not overwrite the still-compatible version 1.
    converter.runs.store(0, Ordering::Relaxed);
    let _ = batch(&root, &converter, 0);
    assert_eq!(converter.runs.load(Ordering::Relaxed), 0);

    println!(
        "synthetic cache workload: {} requests per batch; cold {:?} ({} conversions), warm {:?} (0), mixed {:?} ({}), new version {:?} ({})",
        WORKERS * UNIQUE, cold, UNIQUE, warm, mixed, UNIQUE / 2, new_version, UNIQUE
    );
}

#[test]
fn unsuccessful_conversion_can_be_retried_and_then_cached() {
    struct Retry(AtomicUsize);
    impl Converter for Retry {
        fn id(&self) -> &'static str {
            "test.cache_retry"
        }

        fn version(&self) -> u32 {
            1
        }

        fn convert(&self, input: &[u8]) -> anyhow::Result<Vec<u8>> {
            if self.0.fetch_add(1, Ordering::Relaxed) == 0 {
                anyhow::bail!("synthetic conversion failure");
            }
            Ok(input.to_vec())
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let cache = Cache::open(dir.path().join("cache")).unwrap();
    let converter = Retry(AtomicUsize::new(0));
    let input = b"retry";
    assert!(cache.get_or_convert(&converter, input).is_err());
    let path = cache.get_or_convert(&converter, input).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), input);
    assert_eq!(cache.get_or_convert(&converter, input).unwrap(), path);
    assert_eq!(converter.0.load(Ordering::Relaxed), 2);
}
