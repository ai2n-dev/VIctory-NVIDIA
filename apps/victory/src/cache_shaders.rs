//! Offline CPU-only population of the ordinary runtime shader cache.
use std::path::PathBuf;

use anyhow::Result;

#[cfg(windows)]
fn exact_container(bytes: &[u8]) -> Option<&[u8]> {
    // Raw scanner sources may include bytes after the DXBC container.
    let size = u32::from_le_bytes(bytes.get(24..28)?.try_into().ok()?) as usize;
    let bytes = bytes.get(..size)?;
    victory_shader::Container::parse(bytes).ok()?;
    Some(bytes)
}

#[cfg(windows)]
pub fn run(paths: Vec<PathBuf>, host_lanes: (u32, u32), heap_clamp: u32, jobs: usize) -> Result<()> {
    use std::collections::HashSet;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use victory_gpu::shader_cache;
    use victory_shader::emit::{self, Target};

    anyhow::ensure!((1..=8).contains(&jobs), "--jobs must be between 1 and 8");
    anyhow::ensure!(heap_clamp > 0 && heap_clamp < emit::OWN_SLOTS, "invalid --heap-clamp");
    for path in &paths {
        anyhow::ensure!(path.exists(), "input does not exist: {}", path.display());
    }
    let cfg = victory_core::Config::load()?;
    // Use the same default directory as Run; an explicit shader-only
    // override never changes the pipeline library or persistent title data.
    shader_cache::open_shader_cache(cfg.cache_subdir("shaders")?)?;
    emit::set_heap_clamp(heap_clamp);
    let target = Target { lanes_min: host_lanes.0, lanes_max: host_lanes.1 };
    println!("CPU-only shader cache: waves {}-{}, clamp {}, workers {}",
        target.lanes_min, target.lanes_max, heap_clamp, jobs);
    let seen = Mutex::new(HashSet::new());
    let translated = AtomicUsize::new(0);
    let failed = AtomicUsize::new(0);
    let duplicates = AtomicUsize::new(0);
    let invalid = AtomicUsize::new(0);
    let started = std::time::Instant::now();
    for root in paths {
        for entry in walkdir::WalkDir::new(root) {
            let entry = entry?;
            if !entry.file_type().is_file() || entry.metadata()?.len() > crate::shader::MAX_FILE {
                continue;
            }
            let data = std::fs::read(entry.path())?;
            let sources = crate::shader::sources(&data);
            let next = AtomicUsize::new(0);
            std::thread::scope(|scope| {
                for _ in 0..jobs.min(sources.len()) {
                    scope.spawn(|| {
                        while let Some(source) = sources.get(next.fetch_add(1, Ordering::Relaxed)) {
                            let Some(bytes) = source.bytes() else {
                                invalid.fetch_add(1, Ordering::Relaxed);
                                continue;
                            };
                            let Some(bytes) = exact_container(&bytes) else {
                                invalid.fetch_add(1, Ordering::Relaxed);
                                continue;
                            };
                            if !seen.lock().unwrap_or_else(|e| e.into_inner()).insert(blake3::hash(bytes)) {
                                duplicates.fetch_add(1, Ordering::Relaxed);
                                continue;
                            }
                            match shader_cache::precompile(bytes, target) {
                                Ok(()) => { translated.fetch_add(1, Ordering::Relaxed); }
                                Err(error) => {
                                    if failed.fetch_add(1, Ordering::Relaxed) < 10 {
                                        eprintln!("shader translation failed: {error}");
                                    }
                                }
                            }
                        }
                    });
                }
            });
        }
    }
    let (hits, misses) = shader_cache::counts();
    let ok = translated.load(Ordering::Relaxed);
    let failed = failed.load(Ordering::Relaxed);
    let invalid = invalid.load(Ordering::Relaxed);
    println!("ready {ok}; cache hits {hits}; cache misses {misses}; failed {failed}; invalid {invalid}; duplicates {}; elapsed {:.2}s",
        duplicates.load(Ordering::Relaxed), started.elapsed().as_secs_f64());
    anyhow::ensure!(ok > 0, "no shaders were cached successfully");
    anyhow::ensure!(failed == 0 && invalid == 0, "shader cache incomplete; see counts above");
    Ok(())
}

#[cfg(not(windows))]
pub fn run(_: Vec<PathBuf>, _: (u32, u32), _: u32, _: usize) -> Result<()> {
    anyhow::bail!("runtime shader cache is currently supported on Windows")
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    #[test]
    fn exact_container_excludes_trailing_bytes_and_rejects_bad_length() {
        let mut bytes = [0u8; 64];
        bytes[..4].copy_from_slice(b"DXBC");
        bytes[24..28].copy_from_slice(&32u32.to_le_bytes());
        assert_eq!(exact_container(&bytes).unwrap().len(), 32);
        bytes[24..28].copy_from_slice(&65u32.to_le_bytes());
        assert!(exact_container(&bytes).is_none());
        bytes[24..28].copy_from_slice(&16u32.to_le_bytes());
        assert!(exact_container(&bytes).is_none());
    }
}
