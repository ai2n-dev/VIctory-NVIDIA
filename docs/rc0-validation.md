# RC0 — first integrated hardware validation

Goal: a reproducible compatibility-layer test candidate, not a claim of playability or a public game release. The owner expects test inputs within approximately five hours. Until then, prioritize integration, regression fixes and diagnostics over additional speculative optimizations.

## Existing branches and ownership

- `dev/nvidia-blackwell`: candidate integration target; Antigravity owns Windows hardware validation and conflict resolution.
- `cto/shader-hash-perf`: incremental contributions from ChatGPT and Codex; PR #1 targets the Blackwell branch.
- `main`: synchronized upstream reference, not the candidate. Do not merge it blindly into the candidate during this window.
- No extra branches/worktrees/PRs, no force push. Preserve current local changes. Publish focused fixes promptly. Use the existing PR to coordinate; the owner/CTO approves the exact candidate SHA before merging.
- Codex: finish async safety, outstanding compilation/lints and regressions. If async cannot pass, keep it disabled by default and retain a working synchronous path.
- Antigravity: complete the current patch and test it; reconcile both branch tips using the latest merge-tree report. Resolve DXIL flag/DCE overlap and audit auto-merged memory/queue files. Do not silently discard either implementation.
- ChatGPT: review memory/card/descriptor changes and integration evidence; defer new architecture experiments unless they block validation.

## Proposed five-hour window (targets, not guarantees)

| Elapsed | Objective | Exit evidence |
|---|---|---|
| 0–1 h | Finish in-flight changes; inventory defaults and known failures; select a candidate source SHA | Both branch tips recorded; outstanding risks assigned |
| 1–2 h | Reconcile branches; validate policy, formatting, build and lint | Exact merged tree reviewed and built; any remaining failure logged |
| 2–3 h | Run workspace and focused regression tests on Windows | Exact SHA, adapter, test counts, failures and ignored reasons |
| 3–4 h | NVIDIA Release/diagnostics check, AMD/WARP cross-check | D3D12 validation output, no new corruption/deadlock; experimental switches off |
| 4–5 h | Preserve tested binaries and logs; prepare test-session checklist and fallback | Binary SHA-256, source SHA, build profile, adapter/driver, options and a known-good fallback |

Do not shorten a correctness gate to hit the time budget. If integration is blocked, retain the newest actually validated Blackwell build as fallback and report the precise missing gate.

## Candidate gates

Run against the same immutable source tree, on the local Windows toolchain. Use bounded subprocess timeouts; capture output/exit status. Do not mark skipped checks as passed. Distinguish the baseline warning inventory from newly introduced warnings.

- cargo xtask policy
- cargo fmt --all -- --check
- cargo clippy --workspace --all-targets --locked -- -D warnings
- cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
- cargo build --release --locked -p victory
- cargo build --profile diag --locked -p victory --features diagnostics
- cargo test --workspace --locked --no-fail-fast
- cargo test --release -p victory-gpu --all-features --locked --no-fail-fast
- cargo test -p victory-shader --features host,diagnostics --locked
- Focused card, shadow, descriptor, queue, watch, loader and HLE regressions; compare NVIDIA and Radeon; use WARP only for applicable tests.
- Capture D3D12 debug-layer results on NVIDIA. Test nondefault flags individually only after baseline passes.
- Record CI failures separately from local hardware results. Do not conceal lint errors or make them nonblocking simply to obtain a green check.

## First test session

Use independently obtained, authorized test inputs and existing supported configuration. Keep all input files, generated caches, saves, logs and binaries outside the source repository in accordance with the asset policy. Record only non-sensitive measurements and metadata in public reports.

1. Confirm build identity and configuration; collect adapter/driver, memory budget, baseline options and available disk/commit headroom.
2. Check the initialization path, menu or equivalent first render, and input/audio where applicable. Record the first blocking failure rather than changing multiple switches at once.
3. Verify scene load, first rendered frames and continued execution. Record shader/PSO failures, D3D12 errors, crash stage, RAM/VRAM and frame timings.
4. Once a repeatable scene exists, compare baseline and one experimental option at a time under the same scene, resolution, cache state and duration. Record average FPS, median/p95 frame time, peak memory, and failures. Keep cold-cache and warm-cache measurements separate.
5. A candidate that fails to initialize is not a playable build. A passing synthetic suite is not proof of scene correctness or real FPS.

## Report template

Source SHA / binary hash / adapter + driver / profile / options / cache state / test counts / ignored reasons / first failure or scene reached / memory / frame-time percentiles / next fix.

The target is a validated test candidate plus actionable logs. No FPS or playability guarantee before actual scene measurements.
