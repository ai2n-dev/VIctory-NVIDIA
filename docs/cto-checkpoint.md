# CTO checkpoint — 2026-11-26

This compact handoff is a snapshot, not a substitute for git history, CI logs
or docs/agent.md. On every resume, fetch dev/nvidia-blackwell and read the
latest commits, CI results and PR #1 before editing. No need to replay the
long conversation.

## Goal and ownership
- Stable native 30+ FPS across representative gameplay, not generated frames.
  Preserve correctness and Windows stability. No validated gameplay FPS
  for this branch yet.
- ChatGPT: shader lowering/DXIL, D16, Wave64, PSO and descriptor work.
- Codex: memory ownership, CARD_COMPARE/readback, CBV zero-copy, refresh,
  residency. Antigravity: compatibility, composition, RT and staged QA;
  ZCode: QA when available. Check current agent work before touching files.
- Branch dev/nvidia-blackwell; PR #1; no force pushes or uncoordinated merges
  to main. Publish scoped patches, cite compatible licenses and upstream SHA.

## State
- Snapshot HEAD before this documentation commit: 11c074e0cafe8ae6aa35cfa19023b0717ab7754c.
- e591cae1: experimental AvatarManz Port 6 guard/refresh, default off.
- ad06476c: OOM backoff and nonblocking reclamation; CI 45819237315 success.
- 0ac7ea76: bounded descriptor allocator; CI 45818533524 success.
- 75fdc78e: parallel texture refresh; CI 45817765238 success.
- 1b28a86d: RAM/commit pressure and staging trim; CI 45816984716 success.
- f3aaae05: fix isolated WARP test using global-host upload; subsequent
  descendant CI passed. b17fbe58 introduced experimental async CARD_COMPARE.
- fdcc01aa: configurable capped descriptor reserve; default unchanged.
  028e211c: experimental bounded PSO wait; both in passing descendants.
- b7004030 / b44b4cb0: strict/independent/paired Wave64 policies, auto unchanged.
  50e43c40: SSA mask analysis; 666900d6: TablePlan; 24510d25: BC7 rotation;
  9693c9f4: legacy FMA input FTZ.
- 11c074e0: static D16 pack reuses input Vec and adds signed/unsigned tests;
  no new runtime format support; CI pending unless updated in Actions.

## Next technical work
1. D16 dynamic integer views: compare Flamingo a1178ce8 and trustno1 77e13a97.
   Static MTBUF packing already exists. Our runtime stride table is a
   different ABI (stride/count, OWN_SLOTS); inspect every reader, producer,
   copy/null/reuse path and cache identity before adding format metadata.
   Do not copy a float handle trick without DXIL and device validation.
   Add typed UINT/SINT vs float/norm differential tests and prove dynamic
   table slot mapping and bounds. Preserve prior defaults until validated.
2. Flamingo 89e5d6c3 (AvatarManz #204): mixed LDS atomics need ordering
   across halves. Our lds_meet already treats non-read LDS operations as
   writes and orders accesses before lane guards in its applicable path.
   Audit independent/console flow coverage; don't add redundant barriers
   or assume multi-wave convergence.
3. Codex: finish opt-in async readback, verify ownership and CBV zero-copy;
   continue budgeted texture refresh and residency A/B. GPU BCPack unfinished.
4. Preserve last validated builds. Use measured frame time, 1% low,
   draw/PSO errors, memory, stalls and correctness, not synthetic FPS claims.

## Monitoring
Hourly automation already configured for PortWatch and public GitHub.
Direct site unavailable in latest fetch; tracker data timestamp
2026-11-26T10:23:53.338Z. Inspect current source/diffs, not just leaderboard.
Latest observed public heads: trustno1-coder/VIctory-NVIDIA 033be27f,
avatarmanz/VIctory-7900XTX a9400a6d, fxceless/victory-fork ceb9e1f7,
realblastex/flamingo 89e5d6c3, DoubleGhosty/cool_fox f4b48ad7.
Also monitor Belobug, Kelvaroq, AnyXboxX, LNX, VItoriaBr and new leaders.
Only public compatible source; no game assets, binaries or leaked material.
Distinguish native FPS, frame generation, settings and hardware.

## Non-negotiable safety
Read docs/agent.md. Two system freezes occurred; exact kernel cause
unconfirmed. No local reproduction of suspect tests; watchdog cannot
protect against a kernel/driver hang. Static analysis -> CPU-only ->
remote CI -> Gatekeeper preflight -> one targeted hardware test where
admitted; no repeat/stress loops on workstation. Quarantine stays until
isolated verification. No TDR, BIOS, drivers or system changes. Do not
touch user processes/files. Keep only agreed development directories;
no deletion or overwriting of user files. Never publish local private logs.
No local GPU/game launch by ChatGPT in this checkpoint.
