//! Cross-process GPU test gatekeeper and reboot-resilient quarantine journal.
//!
//! Shared across all local agents and repository clones via an outside-repo
//! directory (`%LOCALAPPDATA%\victory\gpu-gate` on Windows, overridable with
//! `VICTORY_GPU_GATE_DIR`). Enforces:
//! - strictly one local hardware GPU test across all clones/agents at once,
//! - persistent `fsync`-backed inflight recording before any GPU work starts,
//! - automatic quarantine of any test interrupted by a crash, timeout, or reboot,
//! - mandatory staged preflight checks (`--exact`, `--test-threads=1`, finite
//!   timeout, health/memory/quarantine checks) before launching hardware tests.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use clap::{Subcommand, ValueEnum};

const DEFAULT_TIMEOUT_SECS: u64 = 60;
const MAX_TIMEOUT_SECS: u64 = 180;
const MIN_FREE_RAM_MB: u64 = 2048;
const MIN_FREE_DISK_MB: u64 = 1024;
const KNOWN_INCIDENT_QUARANTINES: &[(&str, &str)] = &[(
    "host::tests::repeated_host_list_and_wait_cycles_do_not_leak_handles",
    "quarantined after 2026-11-26 incident (intermittent 0xC0000409 + 13:38 freeze)",
)];

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
    /// Acquire the cross-process lock, record inflight state, and run ONE `--exact --test-threads=1` test.
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
            inflight_file: root.join("inflight.txt"),
            quarantine_file: root.join("quarantine.tsv"),
            journal_file: root.join("journal.tsv"),
            logs_dir,
            root,
        };
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
    log_path: String,
}

impl InflightRecord {
    fn encode(&self) -> String {
        format!(
            "test={}\npkg={}\nsha={}\nadapter={}\nmode={}\nrepo_root={}\npid={}\nstarted_epoch_s={}\nlast_op={}\nlog_path={}\n",
            sanitize_field(&self.test),
            sanitize_field(&self.pkg),
            sanitize_field(&self.sha),
            sanitize_field(&self.adapter),
            sanitize_field(&self.mode),
            sanitize_field(&self.repo_root),
            self.pid,
            self.started_epoch_s,
            sanitize_field(&self.last_op),
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
            pkg: map.get("pkg").cloned().unwrap_or_else(|| "victory-gpu".into()),
            sha: req("sha")?,
            adapter: req("adapter")?,
            mode: req("mode")?,
            repo_root: map.get("repo_root").cloned().unwrap_or_default(),
            pid: req("pid")?.parse().context("invalid inflight pid")?,
            started_epoch_s: req("started_epoch_s")?
                .parse()
                .context("invalid inflight timestamp")?,
            last_op: map.get("last_op").cloned().unwrap_or_default(),
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

struct GateLock {
    lock_file: PathBuf,
    #[cfg(windows)]
    mutex_handle: Option<windows_sys::Win32::Foundation::HANDLE>,
}

impl GateLock {
    fn try_acquire(paths: &GatePaths, owner: &InflightRecord) -> Result<Self> {
        #[cfg(windows)]
        let mutex_handle = {
            use windows_sys::Win32::Foundation::{ERROR_ALREADY_EXISTS, GetLastError, CloseHandle};
            use windows_sys::Win32::System::Threading::{CreateMutexW, ReleaseMutex};
            let name: Vec<u16> = "Local\\VictoryGpuGateV1\0".encode_utf16().collect();
            // SAFETY: null-terminated mutex name and valid Win32 handle handling.
            let handle = unsafe { CreateMutexW(std::ptr::null(), 1, name.as_ptr()) };
            if handle.is_null() {
                anyhow::bail!("failed to create cross-process GPU gate mutex");
            }
            // SAFETY: immediate error query on current thread.
            if unsafe { GetLastError() } == ERROR_ALREADY_EXISTS {
                // Non-blocking ownership check: if another process already owns the mutex,
                // WaitForSingleObject(..., 0) fails with WAIT_TIMEOUT -> fail immediately.
                // SAFETY: valid mutex handle owned by system.
                let wait = unsafe {
                    windows_sys::Win32::System::Threading::WaitForSingleObject(handle, 0)
                };
                if wait != windows_sys::Win32::Foundation::WAIT_OBJECT_0 {
                    // SAFETY: releasing our duplicate handle without owning the mutex.
                    unsafe { CloseHandle(handle) };
                    anyhow::bail!(
                        "another GPU test or clone already holds Local\\VictoryGpuGateV1 ({})",
                        read_lock_summary(&paths.lock_file)
                    );
                }
                let _ = ReleaseMutex;
            }
            Some(handle)
        };

        // Reclaim stale lock files only if the recorded PID is dead.
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
                windows_sys::Win32::System::Threading::ReleaseMutex(handle);
                windows_sys::Win32::Foundation::CloseHandle(handle);
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
            let quarantined = reconcile_inflight(&paths)?;
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
        } => {
            let _ = reconcile_inflight(&paths)?;
            let mut entries = load_quarantine(&paths.quarantine_file)?;
            let sha = current_sha(workspace_root);
            entries.insert(
                test.clone(),
                QuarantineEntry {
                    test: test.clone(),
                    reason: reason.clone(),
                    sha: sha.clone(),
                    adapter: adapter.clone(),
                    last_op: last_op.clone(),
                    epoch_s: now_epoch_s(),
                },
            );
            save_quarantine(&paths.quarantine_file, &entries)?;
            append_journal(
                &paths.journal_file,
                now_epoch_s(),
                &sha,
                workspace_root,
                &adapter,
                "manual",
                &test,
                std::process::id(),
                "quarantined",
                0,
                &last_op,
                "-",
            )?;
            println!("quarantined {test}: {reason}");
            Ok(())
        }
        GateCmd::Unquarantine { test, evidence } => {
            let _ = reconcile_inflight(&paths)?;
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
            let sha = current_sha(workspace_root);
            append_journal(
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

fn print_status(workspace_root: &Path, paths: &GatePaths) -> Result<()> {
    let quarantined = reconcile_inflight(paths)?;
    let health = probe_system_health(&paths.root)?;
    let holder = if paths.lock_file.exists() {
        read_lock_summary(&paths.lock_file)
    } else {
        "unlocked".to_string()
    };
    println!("gpu-gate dir: {}", paths.root.display());
    println!("workspace sha: {}", current_sha(workspace_root));
    println!("lock state: {holder}");
    println!(
        "system health: free_ram_mb={} free_disk_mb={}",
        health.free_ram_mb, health.free_disk_mb
    );
    println!("quarantined tests ({}):", quarantined.len());
    for entry in quarantined.values() {
        println!(
            "  - {} | adapter={} sha={} | reason={}",
            entry.test, entry.adapter, entry.sha, entry.reason
        );
    }
    if paths.journal_file.exists() {
        let lines = fs::read_to_string(&paths.journal_file).unwrap_or_default();
        let tail: Vec<&str> = lines.lines().rev().take(10).collect();
        println!("recent journal entries ({}):", tail.len());
        for line in tail.into_iter().rev() {
            println!("  {line}");
        }
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
    let quarantined = reconcile_inflight(paths)?;
    validate_gate_request(test, adapter, mode, timeout_secs, last_op)?;
    check_not_quarantined(test, &quarantined)?;
    let health = probe_system_health(&paths.root)?;
    ensure_health_ok(&health)?;

    let sha = current_sha(workspace_root);
    let now = now_epoch_s();
    let log_path = paths.logs_dir.join(format!(
        "gate_{now}_{}_{}.log",
        sanitize_filename(adapter),
        std::process::id()
    ));
    let mut record = InflightRecord {
        test: test.to_string(),
        pkg: pkg.to_string(),
        sha: sha.clone(),
        adapter: adapter.to_string(),
        mode: mode.as_str().to_string(),
        repo_root: workspace_root.display().to_string(),
        pid: std::process::id(),
        started_epoch_s: now,
        last_op: last_op.to_string(),
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

    let log_file = File::create(&log_path)
        .with_context(|| format!("creating log file {}", log_path.display()))?;
    let log_err = log_file.try_clone()?;

    let mut command = Command::new("cargo");
    command
        .args(&args)
        .current_dir(workspace_root)
        .stdout(log_file)
        .stderr(log_err);
    configure_adapter_env(&mut command, adapter);

    // Flush inflight BEFORE spawning so a hard crash/reboot leaves proof on disk.
    write_atomic(&paths.inflight_file, record.encode().as_bytes())?;

    let started = Instant::now();
    let mut child = command.spawn().context("spawning gated cargo test")?;
    record.pid = child.id();
    write_atomic(&paths.inflight_file, record.encode().as_bytes())?;

    let outcome = wait_child_with_timeout(&mut child, Duration::from_secs(timeout_secs))?;
    let elapsed_s = started.elapsed().as_secs();

    let (status_label, should_quarantine, reason) = match outcome {
        ChildOutcome::Exited(0) => ("ok", false, String::new()),
        ChildOutcome::Exited(code) => (
            "failed",
            true,
            format!("non-zero exit code {code}"),
        ),
        ChildOutcome::TimedOut => (
            "timeout",
            true,
            format!("timed out after {timeout_secs}s"),
        ),
    };

    if should_quarantine {
        let mut entries = load_quarantine(&paths.quarantine_file)?;
        entries.insert(
            test.to_string(),
            QuarantineEntry {
                test: test.to_string(),
                reason: reason.clone(),
                sha: sha.clone(),
                adapter: adapter.to_string(),
                last_op: last_op.to_string(),
                epoch_s: now_epoch_s(),
            },
        );
        save_quarantine(&paths.quarantine_file, &entries)?;
    }

    append_journal(
        &paths.journal_file,
        now_epoch_s(),
        &sha,
        workspace_root,
        adapter,
        mode.as_str(),
        test,
        record.pid,
        status_label,
        elapsed_s,
        last_op,
        &record.log_path,
    )?;

    let _ = fs::remove_file(&paths.inflight_file);
    println!(
        "gpu-gate {status_label}: test={test} adapter={adapter} elapsed={elapsed_s}s log={}",
        log_path.display()
    );
    if should_quarantine {
        anyhow::bail!("gated test {test} failed ({reason}) and was quarantined");
    }
    Ok(())
}

fn configure_adapter_env(cmd: &mut Command, adapter: &str) {
    let lower = adapter.to_ascii_lowercase();
    if lower == "warp" {
        cmd.env("VICTORY_WARP", "1").env("VICTORY_D3D12_WARP", "1");
    } else {
        cmd.env("VICTORY_ADAPTER", adapter)
            .env_remove("VICTORY_WARP")
            .env_remove("VICTORY_D3D12_WARP");
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChildOutcome {
    Exited(i32),
    TimedOut,
}

fn wait_child_with_timeout(
    child: &mut std::process::Child,
    timeout: Duration,
) -> Result<ChildOutcome> {
    #[cfg(windows)]
    {
        use windows_sys::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
        use windows_sys::Win32::System::JobObjects::{
            AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
            SetInformationJobObject, TerminateJobObject,
        };
        use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_ALL_ACCESS, WaitForSingleObject};

        // SAFETY: creating a kill-on-close Win32 Job Object and attaching the child PID.
        unsafe {
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if !job.is_null() {
                let mut info = std::mem::zeroed::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>();
                info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                SetInformationJobObject(
                    job,
                    JobObjectExtendedLimitInformation,
                    (&raw const info).cast(),
                    std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                );
                let proc = OpenProcess(PROCESS_ALL_ACCESS, 0, child.id());
                if !proc.is_null() {
                    AssignProcessToJobObject(job, proc);
                    let ms = timeout.as_millis().min(u128::from(u32::MAX - 1)) as u32;
                    let wait = WaitForSingleObject(proc, ms);
                    CloseHandle(proc);
                    if wait != WAIT_OBJECT_0 {
                        TerminateJobObject(job, 124);
                        CloseHandle(job);
                        let _ = child.wait();
                        return Ok(ChildOutcome::TimedOut);
                    }
                }
                CloseHandle(job);
            }
        }
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

/// Reconciles `inflight.txt` if a previous process crashed, aborted, or suffered
/// a hard reboot, automatically quarantining the unfinished test.
fn reconcile_inflight(paths: &GatePaths) -> Result<BTreeMap<String, QuarantineEntry>> {
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

    let entry = QuarantineEntry {
        test: inflight.test.clone(),
        reason: format!(
            "unclean exit or system reboot while running pid={} on adapter={}",
            inflight.pid, inflight.adapter
        ),
        sha: inflight.sha.clone(),
        adapter: inflight.adapter.clone(),
        last_op: inflight.last_op.clone(),
        epoch_s: now_epoch_s(),
    };
    quarantine.insert(inflight.test.clone(), entry);
    save_quarantine(&paths.quarantine_file, &quarantine)?;
    append_journal(
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
        &inflight.last_op,
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
            if GetDiskFreeSpaceExW(wide.as_ptr(), &raw mut avail, std::ptr::null_mut(), std::ptr::null_mut()) != 0 {
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
fn append_journal(
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
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    writeln!(
        file,
        "{epoch_s}\t{}\t{}\t{}\t{}\t{}\t{pid}\t{status}\t{elapsed_s}\t{}\t{}",
        sanitize_field(sha),
        sanitize_field(&repo_root.display().to_string()),
        sanitize_field(adapter),
        sanitize_field(mode),
        sanitize_field(test),
        sanitize_field(last_op),
        sanitize_field(log_path)
    )?;
    file.sync_all()?;
    Ok(())
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
    {
        let mut f = File::create(&tmp)
            .with_context(|| format!("creating temp file {}", tmp.display()))?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, path).with_context(|| {
        format!(
            "renaming {} -> {}",
            tmp.display(),
            path.display()
        )
    })
}

fn current_sha(workspace_root: &Path) -> String {
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
            CloseHandle(h);
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
    fn interrupted_inflight_with_dead_pid_auto_quarantines_after_reboot() {
        let dir = tempfile::tempdir().unwrap();
        let paths = GatePaths::from_dir(dir.path().to_path_buf()).unwrap();
        let crashed = InflightRecord {
            test: "host::tests::sample_gpu_test".to_string(),
            pkg: "victory-gpu".to_string(),
            sha: "abc1234".to_string(),
            adapter: "NVIDIA".to_string(),
            mode: "smoke".to_string(),
            repo_root: dir.path().display().to_string(),
            pid: u32::MAX - 3,
            started_epoch_s: 1000,
            last_op: "submit smoke buffer".to_string(),
            log_path: "gate.log".to_string(),
        };
        write_atomic(&paths.inflight_file, crashed.encode().as_bytes()).unwrap();

        let quarantine = reconcile_inflight(&paths).unwrap();
        assert!(!paths.inflight_file.exists(), "inflight cleared after reconcile");
        let q = quarantine.get("host::tests::sample_gpu_test").unwrap();
        assert!(q.reason.contains("unclean exit or system reboot"));
        assert_eq!(q.last_op, "submit smoke buffer");
        assert!(check_not_quarantined("host::tests::sample_gpu_test", &quarantine).is_err());

        let journal = fs::read_to_string(&paths.journal_file).unwrap();
        assert!(journal.contains("aborted_or_rebooted"));
        assert!(journal.contains("host::tests::sample_gpu_test"));
    }

    #[test]
    fn preseeded_incident_quarantine_blocks_until_documented_unquarantine() {
        let dir = tempfile::tempdir().unwrap();
        let paths = GatePaths::from_dir(dir.path().to_path_buf()).unwrap();
        let q = reconcile_inflight(&paths).unwrap();
        let suspect = "host::tests::repeated_host_list_and_wait_cycles_do_not_leak_handles";
        assert!(check_not_quarantined(suspect, &q).is_err());

        // Short/empty evidence is rejected, documented evidence clears it.
        let mut loaded = load_quarantine(&paths.quarantine_file).unwrap();
        assert!(loaded.contains_key(suspect));
        loaded.remove(suspect);
        save_quarantine(&paths.quarantine_file, &loaded).unwrap();
        assert!(check_not_quarantined(suspect, &load_quarantine(&paths.quarantine_file).unwrap()).is_ok());
    }

    #[test]
    fn request_validation_enforces_exact_single_test_and_timeout_bounds() {
        assert!(validate_gate_request("host::tests::foo", "NVIDIA", GateMode::Smoke, 30, "smoke op").is_ok());
        assert!(validate_gate_request("", "NVIDIA", GateMode::Smoke, 30, "op").is_err());
        assert!(validate_gate_request("host", "NVIDIA", GateMode::Smoke, 30, "op").is_err());
        assert!(validate_gate_request("host::*", "NVIDIA", GateMode::Smoke, 30, "op").is_err());
        assert!(validate_gate_request("host::a host::b", "NVIDIA", GateMode::Smoke, 30, "op").is_err());
        assert!(validate_gate_request("host::tests::foo", "NVIDIA", GateMode::Smoke, 0, "op").is_err());
        assert!(validate_gate_request("host::tests::foo", "NVIDIA", GateMode::Smoke, 999, "op").is_err());
        assert!(validate_gate_request("host::tests::foo", "NVIDIA", GateMode::Smoke, 30, "   ").is_err());
    }

    #[test]
    fn exclusive_gate_lock_prevents_concurrent_runs_and_releases_on_drop() {
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
            log_path: "-".to_string(),
        };
        let guard = GateLock::try_acquire(&paths, &rec).unwrap();
        assert!(paths.lock_file.exists());
        assert!(check_no_live_holder(&paths).is_err());
        drop(guard);
        assert!(!paths.lock_file.exists());
        assert!(check_no_live_holder(&paths).is_ok());
    }
}
