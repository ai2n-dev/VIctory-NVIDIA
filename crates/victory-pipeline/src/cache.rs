use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use anyhow::{Context, Result};

use crate::Converter;

/// Identifies one conversion result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheKey(blake3::Hash);

impl CacheKey {
    pub fn new(converter: &dyn Converter, input: &[u8]) -> Self {
        let mut h = blake3::Hasher::new();
        h.update(converter.id().as_bytes());
        h.update(&[0]);
        h.update(&converter.version().to_le_bytes());
        h.update(input);
        CacheKey(h.finalize())
    }

    pub fn to_hex(&self) -> String {
        self.0.to_hex().to_string()
    }
}

/// Content-addressed store under the user's cache directory.
#[derive(Debug, Clone)]
pub struct Cache {
    root: PathBuf,
}

impl Cache {
    /// Opens (creating) a cache at `root`. Refuses locations inside the repo.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        victory_core::ensure_outside_repo(&root)?;
        std::fs::create_dir_all(&root).with_context(|| format!("creating {}", root.display()))?;
        Ok(Cache { root })
    }

    pub fn path_for(&self, key: &CacheKey) -> PathBuf {
        let hex = key.to_hex();
        self.root.join(&hex[..2]).join(&hex[2..])
    }

    /// Returns the cached output for `input`, converting it first if needed.
    pub fn get_or_convert(&self, converter: &dyn Converter, input: &[u8]) -> Result<PathBuf> {
        let key = CacheKey::new(converter, input);
        let path = self.path_for(&key);
        if path.is_file() {
            tracing::trace!(converter = converter.id(), key = %key.to_hex(), "cache hit");
            return Ok(path);
        }
        // The same missing shader can be requested by many pipeline workers.
        // Serialize only identical output paths, then recheck the disk. Hits
        // do not touch the registry and unrelated conversions run in parallel.
        let flight = flight_for(&path);
        let _guard = flight.lock().unwrap_or_else(|e| e.into_inner());
        if path.is_file() {
            tracing::trace!(converter = converter.id(), key = %key.to_hex(), "cache hit after concurrent conversion");
            return Ok(path);
        }
        let output = converter
            .convert(input)
            .with_context(|| format!("converter {} v{}", converter.id(), converter.version()))?;
        let dir = path.parent().expect("hashed path has a parent");
        std::fs::create_dir_all(dir)?;
        // Write-then-rename so readers never see partial output.
        let (tmp, mut file) = create_staging(|| tempfile_in(dir))?;
        let written = file.write_all(&output);
        drop(file);
        let published = written.and_then(|()| std::fs::rename(&tmp, &path));
        if published.is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
        published?;
        Ok(path)
    }
}

// Weak references keep only in-flight locks alive. Pruning on each miss
// bounds the registry by the concurrent requests, not the number of shaders.
// The map is process-wide so independently opened caches can also cooperate.
// It never stays locked during conversion or disk IO.
type Flights = Mutex<HashMap<PathBuf, Weak<Mutex<()>>>>;

fn flight_for(path: &Path) -> Arc<Mutex<()>> {
    static FLIGHTS: OnceLock<Flights> = OnceLock::new();
    let mut flights = FLIGHTS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    flights.retain(|_, lock| lock.strong_count() != 0);
    if let Some(lock) = flights.get(path).and_then(Weak::upgrade) {
        return lock;
    }
    let lock = Arc::new(Mutex::new(()));
    flights.insert(path.to_owned(), Arc::downgrade(&lock));
    lock
}

/// Skip a stale staging name after PID reuse, without truncating another
/// writer's file. Bound retries so a corrupt cache cannot spin indefinitely.
fn create_staging(mut next: impl FnMut() -> PathBuf) -> std::io::Result<(PathBuf, std::fs::File)> {
    for _ in 0..64 {
        let path = next();
        match std::fs::OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => return Ok((path, file)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    Err(std::io::Error::new(std::io::ErrorKind::AlreadyExists, "staging name retry limit"))
}

fn tempfile_in(dir: &Path) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    dir.join(format!(".tmp-{}-{n}", std::process::id()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Upper {
        version: u32,
        runs: AtomicUsize,
    }

    impl Converter for Upper {
        fn id(&self) -> &'static str {
            "test.upper"
        }
        fn version(&self) -> u32 {
            self.version
        }
        fn convert(&self, input: &[u8]) -> Result<Vec<u8>> {
            self.runs.fetch_add(1, Ordering::Relaxed);
            Ok(input.to_ascii_uppercase())
        }
    }

    #[test]
    fn converts_once_then_hits() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = Cache::open(tmp.path().join("cache")).unwrap();
        let conv = Upper {
            version: 1,
            runs: AtomicUsize::new(0),
        };
        let a = cache.get_or_convert(&conv, b"abc").unwrap();
        let b = cache.get_or_convert(&conv, b"abc").unwrap();
        assert_eq!(a, b);
        assert_eq!(std::fs::read(&a).unwrap(), b"ABC");
        assert_eq!(conv.runs.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn concurrent_misses_convert_once_across_independently_opened_caches() {
        struct Slow(AtomicUsize);
        impl Converter for Slow {
            fn id(&self) -> &'static str {
                "test.concurrent"
            }
            fn version(&self) -> u32 {
                1
            }
            fn convert(&self, input: &[u8]) -> Result<Vec<u8>> {
                self.0.fetch_add(1, Ordering::Relaxed);
                std::thread::sleep(std::time::Duration::from_millis(25));
                Ok(input.to_ascii_uppercase())
            }
        }
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("cache");
        let converter = Slow(AtomicUsize::new(0));
        let start = std::sync::Barrier::new(8);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    let cache = Cache::open(&root).unwrap();
                    start.wait();
                    let path = cache.get_or_convert(&converter, b"same shader").unwrap();
                    assert_eq!(std::fs::read(path).unwrap(), b"SAME SHADER");
                });
            }
        });
        assert_eq!(converter.0.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn flights_are_shared_only_for_identical_output_paths() {
        let tmp = tempfile::tempdir().unwrap();
        let a = tmp.path().join("a");
        let b = tmp.path().join("b");
        let first = flight_for(&a);
        let same = flight_for(&a);
        let other = flight_for(&b);
        assert!(Arc::ptr_eq(&first, &same));
        assert!(!Arc::ptr_eq(&first, &other));
        assert!(other.try_lock().is_ok());
    }

    #[test]
    fn stale_staging_file_is_skipped_without_modification() {
        let dir = tempfile::tempdir().unwrap();
        let old = dir.path().join("old");
        let fresh = dir.path().join("fresh");
        std::fs::write(&old, b"partial").unwrap();
        let mut names = [old.clone(), fresh.clone()].into_iter();
        let (path, mut file) = create_staging(|| names.next().unwrap()).unwrap();
        assert_eq!(path, fresh);
        file.write_all(b"new").unwrap();
        drop(file);
        assert_eq!(std::fs::read(&old).unwrap(), b"partial");
        assert_eq!(std::fs::read(&fresh).unwrap(), b"new");
        assert_eq!(create_staging(|| old.clone()).unwrap_err().kind(), std::io::ErrorKind::AlreadyExists);
    }

    #[test]
    fn failed_publication_removes_the_staging_file() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = Cache::open(tmp.path().join("cache")).unwrap();
        let conv = Upper {
            version: 1,
            runs: AtomicUsize::new(0),
        };
        let path = cache.path_for(&CacheKey::new(&conv, b"abc"));
        std::fs::create_dir_all(&path).unwrap();

        assert!(cache.get_or_convert(&conv, b"abc").is_err());
        assert!(path.is_dir());
        assert_eq!(
            std::fs::read_dir(path.parent().unwrap()).unwrap().count(),
            1
        );
    }

    #[test]
    fn version_bump_invalidates() {
        let v1 = Upper {
            version: 1,
            runs: AtomicUsize::new(0),
        };
        let v2 = Upper {
            version: 2,
            runs: AtomicUsize::new(0),
        };
        assert_ne!(CacheKey::new(&v1, b"x"), CacheKey::new(&v2, b"x"));
    }

    #[test]
    fn refuses_cache_inside_repo() {
        let inside = Path::new(env!("CARGO_MANIFEST_DIR")).join("cache");
        assert!(Cache::open(inside).is_err());
    }
}
