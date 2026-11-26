# CTO #012 — upstream integration inventory (2026-11-26)

This is a source audit, not a merged or hardware-tested candidate.

## Pinned inputs

- main: 4f62c2a5e27e7beebf3e8d68f77898e43dc7b77c
- dev/nvidia-blackwell: fa1320d9e05fe2c9a713f4bc22f8f9fdfb48e81a
- cto/shader-hash-perf: afe40f55caef6250c2ea7ed6b832982eb90707ce
- shared ancestor with main: 71652a2c15a0da68cd801890e1f71109670f9ca5
- AvatarManz reference: https://github.com/avatarmanz/VIctory-7900XTX/tree/e2a28ec70632f5112be085abfa7d0a557b25e8d8

GitHub compare: CTO ahead 79 / behind 49; Blackwell ahead 40 / behind 49, relative to the pinned main. Counts are ancestry, not unique feature counts.

Compared the complete blob inventories of all four trees (no truncation): 59 paths changed on at least one working branch relative to the common ancestor, 38 of these unchanged on main, 21 differing on main and at least one working branch. These are review candidates, NOT a computed merge conflict count. Even an upstream-unchanged path can have competing edits on both working branches.

## Correction to the earlier upstream hypothesis

The pinned main still describes Belobug's 5–8 FPS world and has its original card.rs (identical blob to the shared ancestor). Its host.rs has ASYNC_SUBMIT but inspection did not establish the complete AvatarManz THIN / RELAY / LAND / CARD_STEPS architecture. Do not infer feature equivalence from a fork name or a README seen at another ref. Compare the actual source of each candidate.

## Preserve / reconcile / defer

| Area | Evidence at pinned refs | Action |
|---|---|---|
| card.rs | main unchanged from common ancestor; CTO has direct L0 flags, rehash-gap, flattened grouping, hash-pass indexing; Blackwell has a separate edit | Preserve CTO changes and reconcile Blackwell edit; keep experiments off; run card roundtrip, resize, dirty-range and alias checks |
| watch.rs | main still has Mutex<Journal>; CTO has fully_clean and expire_pins, absent from main | Retain kernel-pin fix and regression test; reconcile remap/section changes rather than replace file |
| descriptor/host.rs | main has died_seen / direct-slot invalidation, CTO has free_dynamic_run and oracle test | Retain both changes; review invalidation and allocation independently |
| host.rs / timeline.rs / queue.rs / list.rs | all are changed by upstream and both working branches | Joint integration only: Codex owns async safety, Antigravity owns integration; no independent competing rewrite |
| HLE dstorage.rs / xasync.rs / xmem.rs | main unchanged from ancestor | Preserve our safety fixes; reconcile Blackwell and CTO where both edited |
| workpool.rs + workpool/bench.rs | CTO additions absent from ancestor and main | Preserve if callers still match; benchmark against alternative pools; do not layer pools blindly |
| shader lower.rs, lower/alu.rs, emit/host.rs, container.rs | main unchanged from ancestor; both working branches differ | Keep Blackwell hardware fixes and CTO tests, reconcile carefully; upstream DCE/metadata must retain correct semantics |
| texture.rs / texback.rs | main unchanged from ancestor; Blackwell introduces texback and middle-mip handling | Preserve texture lifetime and state rules; no duplicate readback |
| bitcode.rs / shader.rs / emit/mod.rs | upstream and both working branches changed | Audit DCE, constant interning and UAV metadata together; validate DXIL and hardware |
| CI / diagnostics / cardmeter / framelog | Blackwell changes/additions; main unchanged from ancestor | Preserve validation and measurement tools; keep lint failures visible |
| AvatarManz THIN, JOIN_PARTS, RELAY, LAND, CARD_STEPS | separate upstream architecture; not proven in pinned main | Evaluate after RC0, or only port a proven blocker fix; do not transplant intertwined subsystems blindly |

## Async gate

Pinned main host.rs: async_submit() defaults on unless VICTORY_ASYNC_SUBMIT=0; can_async explicitly excludes cfg!(test), requires owned lists and empty card_writes. CTO host.rs: production async requires VICTORY_ASYNC_SUBMIT=1; separate test gate ASYNC_IN_TESTS and test an_unwaited_submission_is_finished_when_waited_for. Preserve opt-in default and explicit asynchronous regression tests. A normal unit-test pass on main does not validate the production async path.

## Integration sequence

1. Record both agents' latest tips, save tested fallback and inspect uncommitted local changes.
2. Reconcile main changes into the existing Blackwell integration target with history preserved; Codex and Antigravity coordinate the overlapping files. Do not force push or invent a merge tree from file statistics.
3. Carry independent HLE, card and regression-test patches; audit all overlapping files against both parents. Preserve authorship.
4. Verify the exact candidate with docs/rc0-validation.md; baseline sync before opt-in async, then compare.
5. Capture candidate SHA and actual build/test results. No claim of 22 FPS for NVIDIA until measured.

No branch merge was performed by this inventory. Local Windows compilation and GPU validation remain with Antigravity. Continue incremental publication in the existing branches.
