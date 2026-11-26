//! Cross-process GPU test gatekeeper and reboot-resilient quarantine journal.
//!
//! Shared across all local agents and repository clones via an outside-repo
//! directory (`%LOCALAPPDATA%\victory\gpu-gate` on Windows, overridable with
//! `VICTORY_GPU_GATE_DIR`). Enforces:
//! - strictly one local hardware GPU test across all clones/agents at once,
//! - cross-process serialized atomic updates to `inflight.txt`, `quarantine.tsv`,
//!   and `journal.tsv` (`StateLock`),
//! - remote CI green-SHA verification and clean-tree check before local GPU runs,
//! - zero-race `CreateProcessW(CREATE_SUSPENDED)` + `JobObject` (`KILL_ON_JOB_CLOSE`)
//!   assignment and flushed `inflight.txt` before the child executes any instruction,
//! - scrubbing hazardous experimental env vars while wiring `VICTORY_LAST_OP_FILE`
//!   to capture the real pre-GPU operation on disk,
//! - verifying `1 passed` + exact test name + actual DXGI adapter telemetry after exit.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use clap::{Subcommand, ValueEnum};

const DEFAULT_TIMEOUT_SECS: u64 = 60;
const MAX_TIMEOUT_SECS: u64 = 180;
const MIN_FREE_RAM_MB: u64 = 2048;
const MIN_FREE_DISK_MB: u64 = 1024;
const STATE_LOCK_WAIT_MS: u32 = 10_000;
const CI_REPO: &str = "ai2n-dev/VIctory-NVIDIA";

const KNOWN_INCIDENT_QUARANTINES: &[(&str, &str)] = &[(
    "host::tests::repeated_host_list_and_wait_cycles_do_not_leak_handles",
    "quarantined after 2026-11-26 incident (intermittent 0xC0000409 + 13:38 freeze)",
)];

/// Experimental or hazardous overrides stripped from gated child runs so one
/// agent/session cannot silently run local hardware under unsafe flags.
pub const HAZARDOUS_EXPERIMENTAL_ENV_VARS: &[&str] = &[
    "VICTORY_D3D12_DEBUG",
    "VICTORY_D3D12_GBV",
    "VICTORY_D3D12_DRED",
    "VICTORY_ASYNC_SUBMIT",
    "VICTORY_KEEP_RETIRED",
    "VICTORY_KEEP_SECS",
    "VICTORY_KEEP_STRICT",
    "VICTORY_GPU_TIMING",
    "VICTORY_NO_WORK",
    "VICTORY_CARD_DIRECT_FLAGS",
    "VICTORY_CARD_METER",
    "VICTORY_RAYS",
    "VICTORY_HOIST_SCOPED_BUDGET",
    "VICTORY_PARALLEL_SNAPSHOTS",
    "VICTORY_FENCE_TIMEOUT_MS",
    "VICTORY_VALIDATE_DXIL",
    "VICTORY_ALLOW_UNSAFE_TEST",
    "VICTORY_WARP",
    "VICTORY_D3D12_WARP",
    "VICTORY_ADAPTER",
];

static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum GateMode {
    Smoke,
    Targeted,
}

impl GateMode {
    fn as_str(self) -> &'static str {
        match self {
            GateMode::Smoke => "smoke",
            GateMode::Targeted => "targeted",
        }
    }
}

#[derive(Subcommand)]
pub enum GateCmd {
    /// Reconcile any interrupted inflight run and print lock/quarantine/journal status.
    Status,
    /// Run mandatory preflight checks for one exact GPU test without launching it.
    Preflight {
        #[arg(long)]
        test: String,
        #[arg(long)]
        adapter: String,
        #[arg(long, value_enum, default_value_t = GateMode::Smoke)]
        mode: GateMode,
        #[arg(long, default_value = "")]
        last_op: String,
    },
    /// Acquire cross-process lock, verify remote CI for HEAD SHA, record inflight state, and run ONE `--exact --test-threads=1` test.
    Run {
        #[arg(long = "crate", default_value = "victory-gpu")]
        pkg: String,
        #[arg(long)]
        test: String,
        #[arg(long)]
        adapter: String,
        #[arg(long, value_enum, default_value_t = GateMode::Smoke)]
        mode: GateMode,
        #[arg(long)]
        last_op: String,
        #[arg(long, default_value_t = DEFAULT_TIMEOUT_SECS)]
        timeout: u64,
        #[arg(long)]
        all_features: bool,
    },
    /// Explicitly place a suspicious test into persistent quarantine.
    Quarantine {
        #[arg(long)]
        test: String,
        #[arg(long)]
        reason: String,
        #[arg(long, default_value = "unknown")]
        adapter: String,
        #[arg(long, default_value = "manual")]
        last_op: String,
    },
    /// Remove a test from quarantine only with documented isolated-system verification evidence.
    Unquarantine {
        #[arg(long)]
        test: String,
        #[arg(long)]
        evidence: String,
    },
}

#[derive(Debug, Clone)]
struct GatePaths {
    root: PathBuf,
    lock_file: PathBuf,
    state_lock_file: PathBuf,
    inflight_file: PathBuf,
    quarantine_file: PathBuf,
    journal_file: PathBuf,
    logs_dir: PathBuf,
}

impl GatePaths {
    fn resolve() -> Result<Self> {
        let root = if let Some(dir) = std::env::var_os("VICTORY_GPU_GATE_DIR")
            && !dir.is_empty()
        {
            PathBuf::from(dir)
        } else {
            default_gate_dir()?
        };
        Self::from_dir(root)
    }

    fn from_dir(root: PathBuf) -> Result<Self> {
        let logs_dir = root.join("logs");
        fs::create_dir_all(&logs_dir)
            .with_context(|| format!("creating gate directory {}", root.display()))?;
        let paths = Self {
            lock_file: root.join("gate.lock"),
            state_lock_file: root.join("state.lock"),
            inflight_file: root.join("inflight.txt"),
            quarantine_file: root.join("quarantine.tsv"),
            journal_file: root.join("journal.tsv"),
            logs_dir,
            root,
        };
        let _state = StateLock::acquire(&paths)?;
        paths.ensure_seeded_quarantine()?;
        Ok(paths)
    }

    fn ensure_seeded_quarantine(&self) -> Result<()> {
        if self.quarantine_file.exists() {
            return Ok(());
        }
        let now = now_epoch_s();
        let mut entries = BTreeMap::new();
        for &(test, reason) in KNOWN_INCIDENT_QUARANTINES {
            entries.insert(
                test.to_string(),
                QuarantineEntry {
                    test: test.to_string(),
                    reason: reason.to_string(),
                    sha: "seed".to_string(),
                    adapter: "all".to_string(),
                    last_op: "2026-11-26 incident quarantine seed".to_string(),
                    epoch_s: now,
                },
            );
        }
        save_quarantine(&self.quarantine_file, &entries)
    }
}

fn default_gate_dir() -> Result<PathBuf> {
    if let Some(local) = std::env::var_os("LOCALAPPDATA")
        && !local.is_empty()
    {
        return Ok(PathBuf::from(local).join("victory").join("gpu-gate"));
    }
    if let Some(home) = std::env::var_os("HOME")
        && !home.is_empty()
    {
        return Ok(PathBuf::from(home)
            .join(".local")
            .join("share")
            .join("victory")
            .join("gpu-gate"));
    }
    Ok(std::env::temp_dir().join("victory-gpu-gate"))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct InflightRecord {
    test: String,
    pkg: String,
    sha: String,
    adapter: String,
    mode: String,
    repo_root: String,
    pid: u32,
    started_epoch_s: u64,
    last_op: String,
    last_op_file: String,
    log_path: String,
}

impl InflightRecord {
    fn encode(&self) -> String {
        format!(
            "test={}\npkg={}\nsha={}\nadapter={}\nmode={}\nrepo_root={}\npid={}\nstarted_epoch_s={}\nlast_op={}\nlast_op_file={}\nlog_path={}\n",
            sanitize_field(&self.test),
            sanitize_field(&self.pkg),
            sanitize_field(&self.sha),
            sanitize_field(&self.adapter),
            sanitize_field(&self.mode),
            sanitize_field(&self.repo_root),
            self.pid,
            self.started_epoch_s,
            sanitize_field(&self.last_op),
            sanitize_field(&self.last_op_file),
            sanitize_field(&self.log_path),
        )
    }

    fn decode(text: &str) -> Result<Self> {
        let mut map = BTreeMap::new();
        for line in text.lines() {
            if let Some((k, v)) = line.split_once('=') {
                map.insert(k.trim(), v.trim().to_string());
            }
        }
        let req = |k: &str| {
            map.get(k)
                .cloned()
                .with_context(|| format!("inflight record missing '{k}'"))
        };
        Ok(Self {
            test: req("test")?,
            pkg: map
                .get("pkg")
                .cloned()
                .unwrap_or_else(|| "victory-gpu".into()),
            sha: req("sha")?,
            adapter: req("adapter")?,
            mode: req("mode")?,
            repo_root: map.get("repo_root").cloned().unwrap_or_default(),
            pid: req("pid")?.parse().context("invalid inflight pid")?,
            started_epoch_s: req("started_epoch_s")?
                .parse()
                .context("invalid inflight timestamp")?,
            last_op: map.get("last_op").cloned().unwrap_or_default(),
            last_op_file: map.get("last_op_file").cloned().unwrap_or_default(),
            log_path: map.get("log_path").cloned().unwrap_or_default(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct QuarantineEntry {
    test: String,
    reason: String,
    sha: String,
    adapter: String,
    last_op: String,
    epoch_s: u64,
}

/// Cross-process metadata serialization lock protecting `quarantine.tsv`,
/// `journal.tsv`, and `inflight.txt` read-modify-write transitions.
struct StateLock {
    file_path: Option<PathBuf>,
    #[cfg(windows)]
    mutex_handle: Option<windows_sys::Win32::Foundation::HANDLE>,
}

impl StateLock {
    fn acquire(paths: &GatePaths) -> Result<Self> {
        // When `VICTORY_GPU_GATE_DIR` is overridden (for hermetic temp tests),
        // avoid colliding on the global Windows mutex across distinct temp dirs.
        #[cfg(windows)]
        let mutex_handle = if std::env::var_os("VICTORY_GPU_GATE_DIR")
            .is_none_or(|v| v.is_empty())
        {
            use windows_sys::Win32::Foundation::{
                CloseHandle, WAIT_ABANDONED, WAIT_OBJECT_0,
            };
            use windows_sys::Win32::System::Threading::{CreateMutexW, WaitForSingleObject};

            let name: Vec<u16> = "Local\\VictoryGpuGateStateV1\0".encode_utf16().collect();
            // SAFETY: valid null-terminated local mutex name.
            let handle = unsafe { CreateMutexW(std::ptr::null(), 0, name.as_ptr()) };
            if handle.is_null() {
                anyhow::bail!("failed to create cross-process GPU state mutex");
            }
            // SAFETY: valid mutex handle with bounded 10s wait.
            let wait = unsafe { WaitForSingleObject(handle, STATE_LOCK_WAIT_MS) };
            if wait != WAIT_OBJECT_0 && wait != WAIT_ABANDONED {
                // SAFETY: valid handle owned here.
                unsafe { CloseHandle(handle) };
                anyhow::bail!("timed out waiting for cross-process GPU state mutex");
            }
            Some(handle)
        } else {
            None
        };

        // Exclusive file lock on `state.lock` works even across isolated temp dirs.
        let deadline = Instant::now() + Duration::from_millis(u64::from(STATE_LOCK_WAIT_MS));
        loop {
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&paths.state_lock_file)
            {
                Ok(mut f) => {
                    let _ = writeln!(f, "pid={}", std::process::id());
                    return Ok(Self {
                        file_path: Some(paths.state_lock_file.clone()),
                        #[cfg(windows)]
                        mutex_handle,
                    });
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    if let Ok(text) = fs::read_to_string(&paths.state_lock_file)
                        && let Some(pid) = parse_pid_line(&text)
                        && !process_alive(pid)
                    {
                        let _ = fs::remove_file(&paths.state_lock_file);
                        continue;
                    }
                    if Instant::now() >= deadline {
                        #[cfg(windows)]
                        if let Some(handle) = mutex_handle {
                            // SAFETY: release mutex before failing cleanly.
                            unsafe {
                                windows_sys::Win32::System::Threading::ReleaseMutex(handle);
                                windows_sys::Win32::Foundation::CloseHandle(handle);
                            }
                        }
                        anyhow::bail!(
                            "timed out waiting for state lock {}",
                            paths.state_lock_file.display()
                        );
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(e) => {
                    #[cfg(windows)]
                    if let Some(handle) = mutex_handle {
                        // SAFETY: release mutex before failing cleanly.
                        unsafe {
                            windows_sys::Win32::System::Threading::ReleaseMutex(handle);
                            windows_sys::Win32::Foundation::CloseHandle(handle);
                        }
                    }
                    return Err(e).context("creating state lock");
                }
            }
        }
    }
}

impl Drop for StateLock {
    fn drop(&mut self) {
        if let Some(path) = self.file_path.take() {
            let _ = fs::remove_file(path);
        }
        #[cfg(windows)]
        if let Some(handle) = self.mutex_handle.take() {
            // SAFETY: owns both mutex lock and handle.
            unsafe {
                let _ = windows_sys::Win32::System::Threading::ReleaseMutex(handle);
                let _ = windows_sys::Win32::Foundation::CloseHandle(handle);
            }
        }
    }
}

/// Exclusive cross-process GPU execution lock held across all clones/agents.
struct GateLock {
    lock_file: PathBuf,
    #[cfg(windows)]
    mutex_handle: Option<windows_sys::Win32::Foundation::HANDLE>,
}

impl GateLock {
    fn try_acquire(paths: &GatePaths, owner: &InflightRecord) -> Result<Self> {
        #[cfg(windows)]
        let mutex_handle = {
            use windows_sys::Win32::Foundation::{
                CloseHandle, ERROR_ALREADY_EXISTS, GetLastError, WAIT_ABANDONED, WAIT_OBJECT_0,
            };
            use windows_sys::Win32::System::Threading::{CreateMutexW, WaitForSingleObject};
            let name: Vec<u16> = "Local\\VictoryGpuGateV1\0".encode_utf16().collect();
            // SAFETY: null-terminated mutex name and valid Win32 handle handling.
            let handle = unsafe { CreateMutexW(std::ptr::null(), 1, name.as_ptr()) };
            if handle.is_null() {
                anyhow::bail!("failed to create cross-process GPU gate mutex");
            }
            // SAFETY: immediate last-error check on current thread.
            if unsafe { GetLastError() } == ERROR_ALREADY_EXISTS {
                // SAFETY: non-blocking check whether another process owns the mutex.
                let wait = unsafe { WaitForSingleObject(handle, 0) };
                if wait != WAIT_OBJECT_0 && wait != WAIT_ABANDONED {
                    // SAFETY: releasing duplicate handle cleanly without owning mutex.
                    unsafe {
                        let _ = CloseHandle(handle);
                    }
                    anyhow::bail!(
                        "another GPU test or clone already holds Local\\VictoryGpuGateV1 ({})",
                        read_lock_summary(&paths.lock_file)
                    );
                }
            }
            Some(handle)
        };

        if paths.lock_file.exists()
            && let Ok(existing) = fs::read_to_string(&paths.lock_file)
            && let Some(pid) = parse_pid_line(&existing)
            && process_alive(pid)
        {
            anyhow::bail!(
                "cross-process GPU gate lock held by live PID {pid}: {}",
                existing.replace('\n', "; ")
            );
        }

        let _ = fs::remove_file(&paths.lock_file);
        let mut f = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&paths.lock_file)
            .with_context(|| {
                format!(
                    "acquiring exclusive lock file {}",
                    paths.lock_file.display()
                )
            })?;
        f.write_all(owner.encode().as_bytes())?;
        f.sync_all()?;

        Ok(Self {
            lock_file: paths.lock_file.clone(),
            #[cfg(windows)]
            mutex_handle,
        })
    }
}

impl Drop for GateLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.lock_file);
        #[cfg(windows)]
        if let Some(handle) = self.mutex_handle.take() {
            // SAFETY: this guard owns both lock acquisition and handle closure.
            unsafe {
                let _ = windows_sys::Win32::System::Threading::ReleaseMutex(handle);
                let _ = windows_sys::Win32::Foundation::CloseHandle(handle);
            }
        }
    }
}

pub fn run(workspace_root: &Path, cmd: GateCmd) -> Result<()> {
    let paths = GatePaths::resolve()?;
    match cmd {
        GateCmd::Status => print_status(workspace_root, &paths),
        GateCmd::Preflight {
            test,
            adapter,
            mode,
            last_op,
        } => {
            let quarantined = {
                let _state = StateLock::acquire(&paths)?;
                reconcile_inflight_locked(&paths)?
            };
            validate_gate_request(&test, &adapter, mode, DEFAULT_TIMEOUT_SECS, &last_op)?;
            check_not_quarantined(&test, &quarantined)?;
            check_no_live_holder(&paths)?;
            let health = probe_system_health(&paths.root)?;
            ensure_health_ok(&health)?;
            println!(
                "gpu-gate preflight ok: test={test} adapter={adapter} mode={} free_ram_mb={} free_disk_mb={}",
                mode.as_str(),
                health.free_ram_mb,
                health.free_disk_mb
            );
            Ok(())
        }
        GateCmd::Quarantine {
            test,
            reason,
            adapter,
            last_op,
        } => quarantine_test(workspace_root, &paths, &test, &reason, &adapter, &last_op),
        GateCmd::Unquarantine { test, evidence } => {
            let _state = StateLock::acquire(&paths)?;
            let _ = reconcile_inflight_locked(&paths)?;
            let trimmed = evidence.trim();
            anyhow::ensure!(
                trimmed.len() >= 16,
                "unquarantine requires documented isolated-system verification evidence (>= 16 chars)"
            );
            let mut entries = load_quarantine(&paths.quarantine_file)?;
            anyhow::ensure!(
                entries.remove(&test).is_some(),
                "test '{test}' is not in quarantine"
            );
            save_quarantine(&paths.quarantine_file, &entries)?;
            let sha = short_sha(workspace_root);
            append_journal_locked(
                &paths.journal_file,
                now_epoch_s(),
                &sha,
                workspace_root,
                "-",
                "audit",
                &test,
                std::process::id(),
                "unquarantined",
                0,
                trimmed,
                "-",
            )?;
            println!("unquarantined {test}: {trimmed}");
            Ok(())
        }
        GateCmd::Run {
            pkg,
            test,
            adapter,
            mode,
            last_op,
            timeout,
            all_features,
        } => run_gated(
            workspace_root,
            &paths,
            &pkg,
            &test,
            &adapter,
            mode,
            &last_op,
            timeout,
            all_features,
        ),
    }
}

fn quarantine_test(
    workspace_root: &Path,
    paths: &GatePaths,
    test: &str,
    reason: &str,
    adapter: &str,
    last_op: &str,
) -> Result<()> {
    let _state = StateLock::acquire(paths)?;
    let _ = reconcile_inflight_locked(paths)?;
    let mut entries = load_quarantine(&paths.quarantine_file)?;
    let sha = short_sha(workspace_root);
    entries.insert(
        test.to_string(),
        QuarantineEntry {
            test: test.to_string(),
            reason: reason.to_string(),
            sha: sha.clone(),
            adapter: adapter.to_string(),
            last_op: last_op.to_string(),
            epoch_s: now_epoch_s(),
        },
    );
    save_quarantine(&paths.quarantine_file, &entries)?;
    append_journal_locked(
        &paths.journal_file,
        now_epoch_s(),
        &sha,
        workspace_root,
        adapter,
        "manual",
        test,
        std::process::id(),
        "quarantined",
        0,
        last_op,
        "-",
    )?;
    println!("quarantined {test}: {reason}");
    Ok(())
}

fn print_status(workspace_root: &Path, paths: &GatePaths) -> Result<()> {
    let (quarantined, journal_tail) = {
        let _state = StateLock::acquire(paths)?;
        let q = reconcile_inflight_locked(paths)?;
        let tail = if paths.journal_file.exists() {
            fs::read_to_string(&paths.journal_file)
                .unwrap_or_default()
                .lines()
                .rev()
                .take(10)
                .map(str::to_string)
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        (q, tail)
    };
    let health = probe_system_health(&paths.root)?;
    let holder = if paths.lock_file.exists() {
        read_lock_summary(&paths.lock_file)
    } else {
        "unlocked".to_string()
    };
    println!("gpu-gate dir: {}", paths.root.display());
    println!("workspace sha: {}", short_sha(workspace_root));
    println!("lock state: {holder}");
    println!(
        "system health: free_ram_mb={} free_disk_mb={}",
        health.free_ram_mb, health.free_disk_mb
    );
    println!("quarantined tests ({}):", quarantined.len());
    for entry in quarantined.values() {
        println!(
            "  - {} | adapter={} sha={} last_op='{}' | reason={}",
            entry.test, entry.adapter, entry.sha, entry.last_op, entry.reason
        );
    }
    println!("recent journal entries ({}):", journal_tail.len());
    for line in journal_tail.into_iter().rev() {
        println!("  {line}");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_gated(
    workspace_root: &Path,
    paths: &GatePaths,
    pkg: &str,
    test: &str,
    adapter: &str,
    mode: GateMode,
    last_op: &str,
    timeout_secs: u64,
    all_features: bool,
) -> Result<()> {
    let quarantined = {
        let _state = StateLock::acquire(paths)?;
        reconcile_inflight_locked(paths)?
    };
    validate_gate_request(test, adapter, mode, timeout_secs, last_op)?;
    check_not_quarantined(test, &quarantined)?;
    let health = probe_system_health(&paths.root)?;
    ensure_health_ok(&health)?;

    // Ensure workspace matches clean committed SHA and remote CI succeeded for that SHA.
    let full_sha = full_head_sha(workspace_root)?;
    verify_clean_worktree(workspace_root)?;
    verify_remote_ci_for_sha(workspace_root, &full_sha)?;
    let sha_short = full_sha.chars().take(12).collect::<String>();

    let now = now_epoch_s();
    let tag = sanitize_filename(adapter);
    let log_path = paths
        .logs_dir
        .join(format!("gate_{now}_{tag}_{}.log", std::process::id()));
    let last_op_path = paths
        .logs_dir
        .join(format!("last_op_{now}_{tag}_{}.txt", std::process::id()));
    write_atomic(&last_op_path, last_op.as_bytes())?;

    let mut record = InflightRecord {
        test: test.to_string(),
        pkg: pkg.to_string(),
        sha: sha_short.clone(),
        adapter: adapter.to_string(),
        mode: mode.as_str().to_string(),
        repo_root: workspace_root.display().to_string(),
        pid: std::process::id(),
        started_epoch_s: now,
        last_op: last_op.to_string(),
        last_op_file: last_op_path.display().to_string(),
        log_path: log_path.display().to_string(),
    };

    let _lock = GateLock::try_acquire(paths, &record)?;

    let mut args = vec![
        "test".to_string(),
        "-p".to_string(),
        pkg.to_string(),
        "--locked".to_string(),
    ];
    if all_features {
        args.push("--all-features".to_string());
    }
    args.push(test.to_string());
    args.push("--".to_string());
    args.push("--exact".to_string());
    args.push("--test-threads=1".to_string());
    args.push("--nocapture".to_string());

    let parent_env: Vec<(String, String)> = std::env::vars().collect();
    let child_env = build_sanitized_child_env(&parent_env, adapter, &last_op_path);

    let started = Instant::now();
    let outcome = spawn_suspended_attach_job_and_wait(
        workspace_root,
        paths,
        &mut record,
        &args,
        &child_env,
        &log_path,
        Duration::from_secs(timeout_secs),
    )?;
    let elapsed_s = started.elapsed().as_secs();

    let observed_last_op = fs::read_to_string(&last_op_path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| last_op.to_string());

    let log_text = fs::read_to_string(&log_path).unwrap_or_default();

    let (status_label, should_quarantine, reason) = match outcome {
        ChildOutcome::Exited(0) => {
            match verify_test_execution_log(&log_text, test, adapter, &observed_last_op) {
                Ok(()) => ("ok", false, String::new()),
                Err(verify_err) => ("failed_verification", true, verify_err.to_string()),
            }
        }
        ChildOutcome::Exited(code) => ("failed", true, format!("non-zero exit code {code}")),
        ChildOutcome::TimedOut => ("timeout", true, format!("timed out after {timeout_secs}s")),
    };

    {
        let _state = StateLock::acquire(paths)?;
        if should_quarantine {
            let mut entries = load_quarantine(&paths.quarantine_file)?;
            entries.insert(
                test.to_string(),
                QuarantineEntry {
                    test: test.to_string(),
                    reason: reason.clone(),
                    sha: sha_short.clone(),
                    adapter: adapter.to_string(),
                    last_op: observed_last_op.clone(),
                    epoch_s: now_epoch_s(),
                },
            );
            save_quarantine(&paths.quarantine_file, &entries)?;
        }
        append_journal_locked(
            &paths.journal_file,
            now_epoch_s(),
            &sha_short,
            workspace_root,
            adapter,
            mode.as_str(),
            test,
            record.pid,
            status_label,
            elapsed_s,
            &observed_last_op,
            &record.log_path,
        )?;
        let _ = fs::remove_file(&paths.inflight_file);
    }

    let _ = fs::remove_file(&last_op_path);
    println!(
        "gpu-gate {status_label}: test={test} adapter={adapter} last_op='{observed_last_op}' elapsed={elapsed_s}s log={}",
        log_path.display()
    );
    if should_quarantine {
        anyhow::bail!("gated test {test} failed ({reason}) and was quarantined");
    }
    Ok(())
}

/// Strips hazardous experimental overrides before setting adapter and last-op env vars.
fn build_sanitized_child_env(
    parent_env: &[(String, String)],
    adapter: &str,
    last_op_file: &Path,
) -> BTreeMap<String, String> {
    let mut env = BTreeMap::new();
    for (k, v) in parent_env {
        if HAZARDOUS_EXPERIMENTAL_ENV_VARS
            .iter()
            .any(|bad| bad.eq_ignore_ascii_case(k))
        {
            continue;
        }
        env.insert(k.clone(), v.clone());
    }
    if adapter.eq_ignore_ascii_case("warp") {
        env.insert("VICTORY_WARP".to_string(), "1".to_string());
        env.insert("VICTORY_D3D12_WARP".to_string(), "1".to_string());
    } else {
        env.insert("VICTORY_ADAPTER".to_string(), adapter.to_string());
    }
    env.insert(
        "VICTORY_LAST_OP_FILE".to_string(),
        last_op_file.display().to_string(),
    );
    env
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChildOutcome {
    Exited(i32),
    TimedOut,
}

#[cfg(windows)]
fn spawn_suspended_attach_job_and_wait(
    workspace_root: &Path,
    paths: &GatePaths,
    record: &mut InflightRecord,
    args: &[String],
    child_env: &BTreeMap<String, String>,
    log_path: &Path,
    timeout: Duration,
) -> Result<ChildOutcome> {
    use std::os::windows::ffi::OsStrExt;
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::{
        CloseHandle, HANDLE_FLAG_INHERIT, SetHandleInformation, WAIT_OBJECT_0, WAIT_TIMEOUT,
    };
    use windows_sys::Win32::Security::SECURITY_ATTRIBUTES;
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
        SetInformationJobObject, TerminateJobObject,
    };
    use windows_sys::Win32::System::Threading::{
        CREATE_NO_WINDOW, CREATE_SUSPENDED, CREATE_UNICODE_ENVIRONMENT, CreateProcessW,
        GetExitCodeProcess, PROCESS_INFORMATION, ResumeThread, STARTF_USESTDHANDLES,
        STARTUPINFOW, TerminateProcess, WaitForSingleObject,
    };

    let log_file = File::create(log_path)
        .with_context(|| format!("creating log file {}", log_path.display()))?;
    let log_handle = log_file.as_raw_handle() as windows_sys::Win32::Foundation::HANDLE;
    // SAFETY: `log_handle` is a valid open file handle; mark it inheritable before CreateProcessW.
    if unsafe { SetHandleInformation(log_handle, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT) } == 0 {
        anyhow::bail!(
            "SetHandleInformation failed: {}",
            std::io::Error::last_os_error()
        );
    }

    let mut cmd_line = String::from("cargo.exe");
    for arg in args {
        cmd_line.push(' ');
        push_windows_arg(&mut cmd_line, arg);
    }
    let mut cmd_wide: Vec<u16> = cmd_line.encode_utf16().chain([0]).collect();

    let mut env_block: Vec<u16> = Vec::new();
    for (k, v) in child_env {
        if k.contains('=') || k.contains('\0') || v.contains('\0') {
            continue;
        }
        env_block.extend(k.encode_utf16());
        env_block.push(b'=' as u16);
        env_block.extend(v.encode_utf16());
        env_block.push(0);
    }
    env_block.push(0);

    let cwd_wide: Vec<u16> = workspace_root
        .as_os_str()
        .encode_wide()
        .chain([0])
        .collect();

    // SAFETY: zero-initialized Win32 STARTUPINFOW with explicit inheritable log handles.
    let mut si = unsafe { std::mem::zeroed::<STARTUPINFOW>() };
    si.cb = std::mem::size_of::<STARTUPINFOW>() as u32;
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = std::ptr::null_mut();
    si.hStdOutput = log_handle;
    si.hStdError = log_handle;

    // SAFETY: zero-initialized PROCESS_INFORMATION out parameter.
    let mut pi = unsafe { std::mem::zeroed::<PROCESS_INFORMATION>() };
    let sa = SECURITY_ATTRIBUTES {
        nLength: std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
        lpSecurityDescriptor: std::ptr::null_mut(),
        bInheritHandle: 1,
    };

    // SAFETY: `cmd_wide`, `env_block`, `cwd_wide`, `si`, `pi` are valid for the call;
    // child starts SUSPENDED before executing any user/GPU instruction.
    let created = unsafe {
        CreateProcessW(
            std::ptr::null(),
            cmd_wide.as_mut_ptr(),
            &sa,
            std::ptr::null(),
            1,
            CREATE_SUSPENDED | CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT,
            env_block.as_ptr().cast(),
            cwd_wide.as_ptr(),
            &si,
            &raw mut pi,
        )
    };
    if created == 0 {
        anyhow::bail!(
            "CreateProcessW(CREATE_SUSPENDED) failed: {}",
            std::io::Error::last_os_error()
        );
    }

    // SAFETY: valid process/thread handles from CreateProcessW.
    let job = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
    if job.is_null() {
        // SAFETY: abort suspended child cleanly if job creation fails.
        unsafe {
            let _ = TerminateProcess(pi.hProcess, 1);
            let _ = CloseHandle(pi.hThread);
            let _ = CloseHandle(pi.hProcess);
        }
        anyhow::bail!("CreateJobObjectW failed");
    }

    // SAFETY: sized JOBOBJECT_EXTENDED_LIMIT_INFORMATION buffer.
    let limit_ok = unsafe {
        let mut info = std::mem::zeroed::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            (&raw const info).cast(),
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        )
    };
    if limit_ok == 0 {
        // SAFETY: terminate suspended child and close handles on error.
        unsafe {
            let _ = TerminateProcess(pi.hProcess, 1);
            let _ = CloseHandle(job);
            let _ = CloseHandle(pi.hThread);
            let _ = CloseHandle(pi.hProcess);
        }
        anyhow::bail!("SetInformationJobObject failed");
    }

    // SAFETY: assign child to kill-on-close JobObject BEFORE resuming primary thread.
    if unsafe { AssignProcessToJobObject(job, pi.hProcess) } == 0 {
        // SAFETY: terminate suspended child and close handles on error.
        unsafe {
            let _ = TerminateProcess(pi.hProcess, 1);
            let _ = CloseHandle(job);
            let _ = CloseHandle(pi.hThread);
            let _ = CloseHandle(pi.hProcess);
        }
        anyhow::bail!("AssignProcessToJobObject failed");
    }

    record.pid = pi.dwProcessId;
    {
        let _state = StateLock::acquire(paths)?;
        write_atomic(&paths.inflight_file, record.encode().as_bytes())?;
    }

    // Drop our file handle now that child owns inherited copy, so logs unlock cleanly when child exits.
    drop(log_file);

    // SAFETY: `pi.hThread` is suspended; JobObject and `inflight.txt` are now both active.
    if unsafe { ResumeThread(pi.hThread) } == u32::MAX {
        // SAFETY: kill job cleanly if ResumeThread fails.
        unsafe {
            let _ = TerminateJobObject(job, 1);
            let _ = CloseHandle(job);
            let _ = CloseHandle(pi.hThread);
            let _ = CloseHandle(pi.hProcess);
        }
        anyhow::bail!("ResumeThread failed");
    }

    let ms = timeout.as_millis().min(u128::from(u32::MAX - 1)) as u32;
    // SAFETY: finite WaitForSingleObject on child process handle.
    let wait = unsafe { WaitForSingleObject(pi.hProcess, ms) };
    let outcome = if wait == WAIT_OBJECT_0 {
        let mut code = 1u32;
        // SAFETY: process exited; query exact exit status.
        if unsafe { GetExitCodeProcess(pi.hProcess, &raw mut code) } == 0 {
            anyhow::bail!("GetExitCodeProcess failed");
        }
        ChildOutcome::Exited(code as i32)
    } else if wait == WAIT_TIMEOUT {
        // SAFETY: hard-terminate entire child job tree on timeout.
        if unsafe { TerminateJobObject(job, 124) } == 0 {
            anyhow::bail!("TerminateJobObject failed on timeout");
        }
        // SAFETY: brief wait for terminated child process teardown.
        let _ = unsafe { WaitForSingleObject(pi.hProcess, 5_000) };
        ChildOutcome::TimedOut
    } else {
        // SAFETY: terminate job tree cleanly on wait failure.
        unsafe {
            let _ = TerminateJobObject(job, 1);
            let _ = CloseHandle(job);
            let _ = CloseHandle(pi.hThread);
            let _ = CloseHandle(pi.hProcess);
        }
        anyhow::bail!("WaitForSingleObject returned unexpected status {wait}");
    };

    // SAFETY: release job and process/thread handles after exit.
    unsafe {
        let _ = CloseHandle(job);
        let _ = CloseHandle(pi.hThread);
        let _ = CloseHandle(pi.hProcess);
    }
    Ok(outcome)
}

#[cfg(not(windows))]
fn spawn_suspended_attach_job_and_wait(
    workspace_root: &Path,
    paths: &GatePaths,
    record: &mut InflightRecord,
    args: &[String],
    child_env: &BTreeMap<String, String>,
    log_path: &Path,
    timeout: Duration,
) -> Result<ChildOutcome> {
    let log_file = File::create(log_path)?;
    let log_err = log_file.try_clone()?;
    let mut cmd = Command::new("cargo");
    cmd.args(args)
        .current_dir(workspace_root)
        .env_clear()
        .envs(child_env)
        .stdout(log_file)
        .stderr(log_err);
    let mut child = cmd.spawn().context("spawning gated cargo test")?;
    record.pid = child.id();
    {
        let _state = StateLock::acquire(paths)?;
        write_atomic(&paths.inflight_file, record.encode().as_bytes())?;
    }
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(ChildOutcome::Exited(status.code().unwrap_or(1)));
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Ok(ChildOutcome::TimedOut);
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[cfg(windows)]
fn push_windows_arg(dst: &mut String, arg: &str) {
    if !arg.is_empty() && !arg.contains([' ', '\t', '"']) {
        dst.push_str(arg);
        return;
    }
    dst.push('"');
    let mut backslashes = 0usize;
    for ch in arg.chars() {
        if ch == '\\' {
            backslashes += 1;
        } else if ch == '"' {
            for _ in 0..=backslashes * 2 {
                dst.push('\\');
            }
            dst.push('"');
            backslashes = 0;
        } else {
            for _ in 0..backslashes {
                dst.push('\\');
            }
            dst.push(ch);
            backslashes = 0;
        }
    }
    for _ in 0..backslashes * 2 {
        dst.push('\\');
    }
    dst.push('"');
}

fn full_head_sha(workspace_root: &Path) -> Result<String> {
    let out = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(workspace_root)
        .output()
        .context("running git rev-parse HEAD")?;
    anyhow::ensure!(out.status.success(), "git rev-parse HEAD failed");
    let sha = String::from_utf8_lossy(&out.stdout).trim().to_string();
    anyhow::ensure!(sha.len() == 40, "unexpected SHA length: '{sha}'");
    Ok(sha)
}

fn verify_clean_worktree(workspace_root: &Path) -> Result<()> {
    let out = Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(workspace_root)
        .output()
        .context("running git status --porcelain")?;
    anyhow::ensure!(out.status.success(), "git status failed");
    check_worktree_clean(&String::from_utf8_lossy(&out.stdout))
}

fn check_worktree_clean(porcelain: &str) -> Result<()> {
    let dirty: Vec<&str> = porcelain
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    if !dirty.is_empty() {
        anyhow::bail!(
            "local repository has uncommitted changes not yet verified on remote CI ({} entries, e.g. '{}')",
            dirty.len(),
            dirty[0]
        );
    }
    Ok(())
}

fn verify_remote_ci_for_sha(workspace_root: &Path, full_sha: &str) -> Result<()> {
    let out = Command::new("gh")
        .args([
            "run",
            "list",
            "-R",
            CI_REPO,
            "--commit",
            full_sha,
            "--workflow",
            "ci.yml",
            "--status",
            "completed",
            "--json",
            "conclusion,headSha,status,databaseId",
        ])
        .current_dir(workspace_root)
        .output()
        .context("querying GitHub Actions via `gh run list`")?;
    anyhow::ensure!(
        out.status.success(),
        "gh run list failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    parse_remote_ci_result(&String::from_utf8_lossy(&out.stdout), full_sha)
}

fn parse_remote_ci_result(json: &str, expected_sha: &str) -> Result<()> {
    let expected = expected_sha.trim();
    if expected.is_empty() {
        anyhow::bail!("missing expected SHA");
    }
    let has_exact_sha = json.contains(&format!("\"headSha\":\"{expected}\""))
        || json.contains(&format!("\"headSha\": \"{expected}\""));
    let has_success = json.contains("\"conclusion\":\"success\"")
        || json.contains("\"conclusion\": \"success\"");
    let is_completed = json.contains("\"status\":\"completed\"")
        || json.contains("\"status\": \"completed\"");
    if !(has_exact_sha && has_success && is_completed) {
        anyhow::bail!(
            "remote CI (`{CI_REPO}` ci.yml) has not completed with `success` for exact SHA `{expected}`"
        );
    }
    Ok(())
}

/// Ensures the gated binary actually executed 1 unignored test matching
/// `expected_test` on `expected_adapter` (using both log output and `last_op`).
fn verify_test_execution_log(
    log_text: &str,
    expected_test: &str,
    expected_adapter: &str,
    observed_last_op: &str,
) -> Result<()> {
    let ran_test_ok = {
        let prefix = format!("test {expected_test} ...");
        let lines: Vec<&str> = log_text.lines().map(str::trim).collect();
        lines.iter().enumerate().any(|(idx, line)| {
            line.starts_with(&prefix)
                && (line.ends_with("... ok")
                    || line.ends_with(" ok")
                    || lines
                        .get(idx + 1..)
                        .into_iter()
                        .flatten()
                        .take_while(|l| !l.starts_with("test ") && !l.starts_with("test result:"))
                        .any(|next| *next == "ok"))
        })
    };
    anyhow::ensure!(
        ran_test_ok,
        "test `{expected_test}` did not execute and pass (missing `test ... ok` in log)"
    );

    let passed_one = log_text.lines().any(|line| {
        let t = line.trim();
        t.starts_with("test result: ok. 1 passed; 0 failed; 0 ignored;")
    });
    anyhow::ensure!(
        passed_one,
        "expected exact count `1 passed; 0 failed; 0 ignored;` in test summary"
    );

    let reported_adapter = extract_reported_adapter(log_text, observed_last_op).ok_or_else(|| {
        anyhow::anyhow!(
            "missing confirmed DXGI/WARP adapter report in log or last_op (`{observed_last_op}`)"
        )
    })?;
    anyhow::ensure!(
        adapter_matches_expected(&reported_adapter, expected_adapter),
        "adapter mismatch: expected `{expected_adapter}`, observed `{reported_adapter}`"
    );
    Ok(())
}

fn extract_reported_adapter(log_text: &str, observed_last_op: &str) -> Option<String> {
    if let Some(after) = observed_last_op.strip_prefix("device_init:")
        && !after.trim().is_empty()
    {
        return Some(after.trim().to_string());
    }
    for line in log_text.lines() {
        if let Some(at) = line.find("[gpu-gate adapter]") {
            let name = line[at + "[gpu-gate adapter]".len()..].trim();
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
        if let Some(at) = line.find("AdapterReport { name: \"") {
            let rest = &line[at + "AdapterReport { name: \"".len()..];
            if let Some(end) = rest.find('"') {
                return Some(rest[..end].to_string());
            }
        }
    }
    None
}

fn adapter_matches_expected(actual: &str, expected: &str) -> bool {
    let actual_lc = actual.to_ascii_lowercase();
    let expected_lc = expected.trim().to_ascii_lowercase();
    if expected_lc == "warp" {
        return actual_lc.contains("basic render") || actual_lc.contains("warp");
    }
    actual_lc.contains(&expected_lc)
}

fn validate_gate_request(
    test: &str,
    adapter: &str,
    _mode: GateMode,
    timeout_secs: u64,
    last_op: &str,
) -> Result<()> {
    let t = test.trim();
    anyhow::ensure!(
        !t.is_empty() && t.contains("::") && !t.chars().any(char::is_whitespace),
        "gpu-gate requires a single exact module-qualified test name (`mod::test`), got '{test}'"
    );
    anyhow::ensure!(
        !t.contains('*') && !t.contains('?'),
        "gpu-gate forbids wildcard test filters ('{test}')"
    );
    anyhow::ensure!(!adapter.trim().is_empty(), "adapter must not be empty");
    anyhow::ensure!(
        !last_op.trim().is_empty(),
        "--last-op must describe the exact pre-GPU operation for reboot triage"
    );
    anyhow::ensure!(
        (1..=MAX_TIMEOUT_SECS).contains(&timeout_secs),
        "timeout must be between 1 and {MAX_TIMEOUT_SECS}s (got {timeout_secs}s)"
    );
    Ok(())
}

/// Reconciles `inflight.txt` under `StateLock`: if a previous process crashed,
/// aborted, or suffered a hard reboot, automatically quarantines the unfinished
/// test with its real `last_op_file` telemetry if available.
fn reconcile_inflight_locked(paths: &GatePaths) -> Result<BTreeMap<String, QuarantineEntry>> {
    let mut quarantine = load_quarantine(&paths.quarantine_file)?;
    if !paths.inflight_file.exists() {
        return Ok(quarantine);
    }
    let raw = fs::read_to_string(&paths.inflight_file)
        .with_context(|| format!("reading {}", paths.inflight_file.display()))?;
    let inflight = match InflightRecord::decode(&raw) {
        Ok(rec) => rec,
        Err(err) => {
            quarantine.insert(
                "corrupt_inflight_record".to_string(),
                QuarantineEntry {
                    test: "corrupt_inflight_record".to_string(),
                    reason: format!("unreadable inflight after crash/reboot: {err}"),
                    sha: "unknown".to_string(),
                    adapter: "unknown".to_string(),
                    last_op: "corrupt inflight".to_string(),
                    epoch_s: now_epoch_s(),
                },
            );
            save_quarantine(&paths.quarantine_file, &quarantine)?;
            let _ = fs::remove_file(&paths.inflight_file);
            return Ok(quarantine);
        }
    };

    if inflight.pid != std::process::id() && process_alive(inflight.pid) {
        anyhow::bail!(
            "live GPU test PID {} is currently running `{}` on `{}`",
            inflight.pid,
            inflight.test,
            inflight.adapter
        );
    }

    let real_last_op = if !inflight.last_op_file.is_empty() {
        fs::read_to_string(&inflight.last_op_file)
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| inflight.last_op.clone())
    } else {
        inflight.last_op.clone()
    };

    let entry = QuarantineEntry {
        test: inflight.test.clone(),
        reason: format!(
            "unclean exit or system reboot while running pid={} on adapter={}",
            inflight.pid, inflight.adapter
        ),
        sha: inflight.sha.clone(),
        adapter: inflight.adapter.clone(),
        last_op: real_last_op.clone(),
        epoch_s: now_epoch_s(),
    };
    quarantine.insert(inflight.test.clone(), entry);
    save_quarantine(&paths.quarantine_file, &quarantine)?;
    append_journal_locked(
        &paths.journal_file,
        now_epoch_s(),
        &inflight.sha,
        Path::new(&inflight.repo_root),
        &inflight.adapter,
        &inflight.mode,
        &inflight.test,
        inflight.pid,
        "aborted_or_rebooted",
        now_epoch_s().saturating_sub(inflight.started_epoch_s),
        &real_last_op,
        &inflight.log_path,
    )?;
    let _ = fs::remove_file(&paths.inflight_file);
    Ok(quarantine)
}

fn check_not_quarantined(
    test: &str,
    quarantine: &BTreeMap<String, QuarantineEntry>,
) -> Result<()> {
    if let Some(entry) = quarantine.get(test) {
        anyhow::bail!(
            "test '{test}' is QUARANTINED (adapter={}, sha={}, last_op='{}', reason='{}'); \
             resolve off-host and run `cargo xtask gpu-gate unquarantine` with isolated evidence",
            entry.adapter,
            entry.sha,
            entry.last_op,
            entry.reason
        );
    }
    Ok(())
}

fn check_no_live_holder(paths: &GatePaths) -> Result<()> {
    if paths.lock_file.exists()
        && let Ok(text) = fs::read_to_string(&paths.lock_file)
        && let Some(pid) = parse_pid_line(&text)
        && process_alive(pid)
    {
        anyhow::bail!(
            "GPU gate already locked by live PID {pid}: {}",
            text.replace('\n', "; ")
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct SystemHealth {
    free_ram_mb: u64,
    free_disk_mb: u64,
}

fn probe_system_health(path: &Path) -> Result<SystemHealth> {
    #[cfg(windows)]
    {
        use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;
        use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

        // SAFETY: valid MEMORYSTATUSEX struct initialized with dwLength.
        let free_ram_mb = unsafe {
            let mut mem = std::mem::zeroed::<MEMORYSTATUSEX>();
            mem.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
            if GlobalMemoryStatusEx(&raw mut mem) != 0 {
                mem.ullAvailPhys >> 20
            } else {
                0
            }
        };

        let mut wide: Vec<u16> = path.as_os_str().encode_wide_vec();
        wide.push(0);
        // SAFETY: valid null-terminated directory path and u64 out pointer.
        let free_disk_mb = unsafe {
            let mut avail = 0u64;
            if GetDiskFreeSpaceExW(
                wide.as_ptr(),
                &raw mut avail,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            ) != 0
            {
                avail >> 20
            } else {
                0
            }
        };
        Ok(SystemHealth {
            free_ram_mb,
            free_disk_mb,
        })
    }
    #[cfg(not(windows))]
    {
        let _ = path;
        Ok(SystemHealth {
            free_ram_mb: MIN_FREE_RAM_MB,
            free_disk_mb: MIN_FREE_DISK_MB,
        })
    }
}

#[cfg(windows)]
trait EncodeWideExt {
    fn encode_wide_vec(&self) -> Vec<u16>;
}

#[cfg(windows)]
impl EncodeWideExt for std::ffi::OsStr {
    fn encode_wide_vec(&self) -> Vec<u16> {
        use std::os::windows::ffi::OsStrExt;
        self.encode_wide().collect()
    }
}

fn ensure_health_ok(h: &SystemHealth) -> Result<()> {
    anyhow::ensure!(
        h.free_ram_mb >= MIN_FREE_RAM_MB,
        "insufficient free RAM ({} MiB < {} MiB minimum)",
        h.free_ram_mb,
        MIN_FREE_RAM_MB
    );
    anyhow::ensure!(
        h.free_disk_mb >= MIN_FREE_DISK_MB,
        "insufficient free disk space ({} MiB < {} MiB minimum)",
        h.free_disk_mb,
        MIN_FREE_DISK_MB
    );
    Ok(())
}

fn load_quarantine(path: &Path) -> Result<BTreeMap<String, QuarantineEntry>> {
    let mut entries = BTreeMap::new();
    if !path.exists() {
        return Ok(entries);
    }
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    for line in BufReader::new(file).lines() {
        let line = line?;
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() >= 6 {
            let test = cols[0].to_string();
            entries.insert(
                test.clone(),
                QuarantineEntry {
                    test,
                    reason: cols[1].to_string(),
                    sha: cols[2].to_string(),
                    adapter: cols[3].to_string(),
                    last_op: cols[4].to_string(),
                    epoch_s: cols[5].parse().unwrap_or(0),
                },
            );
        }
    }
    Ok(entries)
}

fn save_quarantine(path: &Path, entries: &BTreeMap<String, QuarantineEntry>) -> Result<()> {
    let mut buf = String::from("# test\treason\tsha\tadapter\tlast_op\tepoch_s\n");
    for e in entries.values() {
        buf.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\n",
            sanitize_field(&e.test),
            sanitize_field(&e.reason),
            sanitize_field(&e.sha),
            sanitize_field(&e.adapter),
            sanitize_field(&e.last_op),
            e.epoch_s
        ));
    }
    write_atomic(path, buf.as_bytes())
}

#[allow(clippy::too_many_arguments)]
fn append_journal_locked(
    path: &Path,
    epoch_s: u64,
    sha: &str,
    repo_root: &Path,
    adapter: &str,
    mode: &str,
    test: &str,
    pid: u32,
    status: &str,
    elapsed_s: u64,
    last_op: &str,
    log_path: &str,
) -> Result<()> {
    let line = format!(
        "{epoch_s}\t{}\t{}\t{}\t{}\t{}\t{pid}\t{status}\t{elapsed_s}\t{}\t{}\n",
        sanitize_field(sha),
        sanitize_field(&repo_root.display().to_string()),
        sanitize_field(adapter),
        sanitize_field(mode),
        sanitize_field(test),
        sanitize_field(last_op),
        sanitize_field(log_path)
    );
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    file.write_all(line.as_bytes())?;
    file.sync_all()?;
    Ok(())
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let seq = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let tmp = path.with_extension(format!("tmp-{}-{seq}", std::process::id()));
    {
        let mut f = File::create(&tmp)
            .with_context(|| format!("creating temp file {}", tmp.display()))?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, path).with_context(|| format!("renaming {} -> {}", tmp.display(), path.display()))
}

fn short_sha(workspace_root: &Path) -> String {
    Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .current_dir(workspace_root)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map_or_else(|| "unknown".to_string(), |s| s.trim().to_string())
}

fn parse_pid_line(text: &str) -> Option<u32> {
    text.lines()
        .find_map(|line| line.strip_prefix("pid="))
        .and_then(|v| v.trim().parse().ok())
}

fn read_lock_summary(path: &Path) -> String {
    fs::read_to_string(path)
        .map(|s| s.lines().collect::<Vec<_>>().join(", "))
        .unwrap_or_else(|_| "locked".to_string())
}

fn process_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::Foundation::{CloseHandle, STILL_ACTIVE};
        use windows_sys::Win32::System::Threading::{
            GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
        };
        // SAFETY: querying existence and exit status of an OS PID.
        unsafe {
            let h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
            if h.is_null() {
                return false;
            }
            let mut code = 0u32;
            let ok = GetExitCodeProcess(h, &raw mut code) != 0;
            let _ = CloseHandle(h);
            ok && code == STILL_ACTIVE as u32
        }
    }
    #[cfg(not(windows))]
    {
        Path::new("/proc").join(pid.to_string()).exists()
    }
}

fn now_epoch_s() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn sanitize_field(s: &str) -> String {
    s.replace(['\t', '\r', '\n'], " ").trim().to_string()
}

fn sanitize_filename(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interrupted_inflight_with_dead_pid_auto_quarantines_and_uses_real_last_op() {
        let dir = tempfile::tempdir().unwrap();
        let paths = GatePaths::from_dir(dir.path().to_path_buf()).unwrap();
        let last_op_file = dir.path().join("last_op.txt");
        write_atomic(&last_op_file, b"execute_lists:group_len=2:sub=42").unwrap();

        let crashed = InflightRecord {
            test: "host::tests::sample_gpu_test".to_string(),
            pkg: "victory-gpu".to_string(),
            sha: "abc1234".to_string(),
            adapter: "NVIDIA".to_string(),
            mode: "smoke".to_string(),
            repo_root: dir.path().display().to_string(),
            pid: u32::MAX - 3,
            started_epoch_s: 1000,
            last_op: "cli-placeholder".to_string(),
            last_op_file: last_op_file.display().to_string(),
            log_path: "gate.log".to_string(),
        };
        write_atomic(&paths.inflight_file, crashed.encode().as_bytes()).unwrap();

        let quarantine = {
            let _state = StateLock::acquire(&paths).unwrap();
            reconcile_inflight_locked(&paths).unwrap()
        };
        assert!(!paths.inflight_file.exists());
        let q = quarantine.get("host::tests::sample_gpu_test").unwrap();
        assert_eq!(q.last_op, "execute_lists:group_len=2:sub=42");
        assert!(check_not_quarantined("host::tests::sample_gpu_test", &quarantine).is_err());

        let journal = fs::read_to_string(&paths.journal_file).unwrap();
        assert!(journal.contains("aborted_or_rebooted"));
        assert!(journal.contains("execute_lists:group_len=2:sub=42"));
    }

    #[test]
    fn remote_ci_verification_requires_exact_sha_and_success_and_clean_tree() {
        let sha = "30f9d81a47ab8188375c049ce9a27a13c2706d28";
        let ok_json = format!(
            r#"[{{"conclusion":"success","databaseId":45788731593,"headSha":"{sha}","status":"completed"}}]"#
        );
        assert!(parse_remote_ci_result(&ok_json, sha).is_ok());

        let wrong_sha = r#"[{"conclusion":"success","headSha":"1111111111111111111111111111111111111111","status":"completed"}]"#.to_string();
        assert!(parse_remote_ci_result(&wrong_sha, sha).is_err());

        let failed_json = format!(
            r#"[{{"conclusion":"failure","headSha":"{sha}","status":"completed"}}]"#
        );
        assert!(parse_remote_ci_result(&failed_json, sha).is_err());
        assert!(parse_remote_ci_result("[]", sha).is_err());

        assert!(check_worktree_clean("").is_ok());
        assert!(check_worktree_clean(" M crates/victory-gpu/src/host.rs\n").is_err());
    }

    #[test]
    fn child_env_sanitization_strips_all_hazardous_experimental_vars() {
        let parent = vec![
            ("PATH".to_string(), "C:\\Windows".to_string()),
            ("VICTORY_D3D12_DEBUG".to_string(), "1".to_string()),
            ("VICTORY_D3D12_GBV".to_string(), "1".to_string()),
            ("VICTORY_ASYNC_SUBMIT".to_string(), "1".to_string()),
            ("VICTORY_KEEP_STRICT".to_string(), "1".to_string()),
            ("VICTORY_RAYS".to_string(), "1".to_string()),
            ("VICTORY_ADAPTER".to_string(), "WARP".to_string()),
        ];
        let op_file = Path::new("C:\\temp\\last_op.txt");
        let sanitized = build_sanitized_child_env(&parent, "NVIDIA", op_file);
        assert_eq!(sanitized.get("PATH").map(String::as_str), Some("C:\\Windows"));
        assert_eq!(sanitized.get("VICTORY_ADAPTER").map(String::as_str), Some("NVIDIA"));
        assert_eq!(
            sanitized.get("VICTORY_LAST_OP_FILE").map(String::as_str),
            Some("C:\\temp\\last_op.txt")
        );
        for bad in [
            "VICTORY_D3D12_DEBUG",
            "VICTORY_D3D12_GBV",
            "VICTORY_ASYNC_SUBMIT",
            "VICTORY_KEEP_STRICT",
            "VICTORY_RAYS",
            "VICTORY_WARP",
        ] {
            assert!(!sanitized.contains_key(bad), "{bad} leaked into child env");
        }
    }

    #[test]
    fn log_verification_requires_exact_single_pass_and_matching_adapter() {
        let good_log = "running 1 test\ntest host::tests::foo ... ok\n\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 100 filtered out;\n";
        assert!(verify_test_execution_log(
            good_log,
            "host::tests::foo",
            "NVIDIA",
            "device_init:NVIDIA GeForce RTX 5070 Ti Laptop GPU"
        ).is_ok());

        // 0 passed (typo/filtered) must fail even with exit code 0.
        let zero_log = "running 0 tests\ntest result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 100 filtered out;\n";
        assert!(verify_test_execution_log(
            zero_log,
            "host::tests::foo",
            "NVIDIA",
            "device_init:NVIDIA GeForce RTX 5070 Ti Laptop GPU"
        ).is_err());

        // Adapter mismatch must fail.
        assert!(verify_test_execution_log(
            good_log,
            "host::tests::foo",
            "NVIDIA",
            "device_init:AMD Radeon(TM) 610M"
        ).is_err());

        // Fallback via AdapterReport in stdout works too.
        let adapter_log = "running 1 test\ntest host::tests::foo ... physical adapter report: AdapterReport { name: \"AMD Radeon(TM) 610M\", vendor: 4098 }\nok\n\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 100 filtered out;\n";
        assert!(verify_test_execution_log(adapter_log, "host::tests::foo", "Radeon", "cli-smoke").is_ok());
    }

    #[test]
    fn concurrent_quarantine_and_journal_updates_preserve_all_records() {
        let dir = tempfile::tempdir().unwrap();
        let paths = GatePaths::from_dir(dir.path().to_path_buf()).unwrap();
        let mut threads = Vec::new();
        for tid in 0..8usize {
            let p = paths.clone();
            let root = dir.path().to_path_buf();
            threads.push(std::thread::spawn(move || {
                for iter in 0..10usize {
                    let test_name = format!("mod{tid}::tests::case_{iter}");
                    let _state = StateLock::acquire(&p).unwrap();
                    let mut entries = load_quarantine(&p.quarantine_file).unwrap();
                    entries.insert(
                        test_name.clone(),
                        QuarantineEntry {
                            test: test_name.clone(),
                            reason: "concurrent test".to_string(),
                            sha: "deadbeef".to_string(),
                            adapter: "NVIDIA".to_string(),
                            last_op: format!("op_{tid}_{iter}"),
                            epoch_s: 123,
                        },
                    );
                    save_quarantine(&p.quarantine_file, &entries).unwrap();
                    append_journal_locked(
                        &p.journal_file,
                        123,
                        "deadbeef",
                        &root,
                        "NVIDIA",
                        "smoke",
                        &test_name,
                        1,
                        "ok",
                        0,
                        &format!("op_{tid}_{iter}"),
                        "-",
                    )
                    .unwrap();
                }
            }));
        }
        for t in threads {
            t.join().unwrap();
        }
        let entries = load_quarantine(&paths.quarantine_file).unwrap();
        // 1 seeded + 8*10 concurrent entries = 81 unique entries, none overwritten/lost.
        assert_eq!(entries.len(), 81);
        let journal = fs::read_to_string(&paths.journal_file).unwrap();
        assert_eq!(journal.lines().count(), 80);
    }

    #[test]
    fn request_validation_and_exclusive_gate_lock_behave_strictly() {
        assert!(validate_gate_request("host::tests::foo", "NVIDIA", GateMode::Smoke, 30, "smoke op").is_ok());
        assert!(validate_gate_request("host::*", "NVIDIA", GateMode::Smoke, 30, "op").is_err());

        let dir = tempfile::tempdir().unwrap();
        let paths = GatePaths::from_dir(dir.path().to_path_buf()).unwrap();
        let rec = InflightRecord {
            test: "host::tests::smoke".to_string(),
            pkg: "victory-gpu".to_string(),
            sha: "abc".to_string(),
            adapter: "NVIDIA".to_string(),
            mode: "smoke".to_string(),
            repo_root: dir.path().display().to_string(),
            pid: std::process::id(),
            started_epoch_s: 1,
            last_op: "init".to_string(),
            last_op_file: String::new(),
            log_path: "-".to_string(),
        };
        let guard = GateLock::try_acquire(&paths, &rec).unwrap();
        assert!(check_no_live_holder(&paths).is_err());
        drop(guard);
        assert!(check_no_live_holder(&paths).is_ok());
    }

    #[test]
    fn verify_test_execution_log_handles_multiline_nocapture_stdout() {
        let log = "\
running 1 test
test host::tests::physical_adapter_details_match_dxgi_luid ... [gpu-gate adapter] NVIDIA GeForce RTX 5070 Ti Laptop GPU
physical adapter report: AdapterReport { name: \"NVIDIA GeForce RTX 5070 Ti Laptop GPU\" }
ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 197 filtered out; finished in 0.24s
";
        assert!(
            verify_test_execution_log(
                log,
                "host::tests::physical_adapter_details_match_dxgi_luid",
                "NVIDIA",
                "device_init:NVIDIA GeForce RTX 5070 Ti Laptop GPU",
            )
            .is_ok()
        );
    }
}
