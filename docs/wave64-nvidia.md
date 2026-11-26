# NVIDIA Wave64 translation policies

NVIDIA exposes 32-lane waves. The existing translator already has two distinct
compute lowering paths: independent halves for lane-local work, and console
control flow with group-shared exchanges when values cross the halves.
This change exposes those paths for controlled comparison; it does not
introduce a new GPU synchronization algorithm.

`VICTORY_WAVE64_MODE` (read once per process):
- `auto` / unset: existing behavior, including existing fallback.
- `strict`: try the independent route; use the existing paired route only
  when required, for at most 64 threads per group. Do not fall back to
  the legacy per-lane approximation after a failed console-flow lowering.
- `independent`: refuse shaders requiring paired exchange.
- `paired`: require the existing console-flow path, refuse independent-only
  shaders. Groups above 64 and cross-wave permutes are not admitted.

The last three modes are experimental translation/diagnostic modes, not
recommended gameplay presets: refusal may prevent a draw/dispatch.
They affect Wave64 compute shaders on Wave32-only targets, not native Wave64
targets or other stages. Separate shader-cache keys prevent reuse between
policies. No new barriers, dispatches, or GPU waits are added.

CPU-only regressions cover route selection, target applicability, and
translation of a lane-local synthetic shader. Before hardware execution,
use remote CI, semantic DXIL validation and GPU Gatekeeper. The paired
algorithm's correctness, convergence, and GPU speed require separate
isolated differential tests. No claim of a native Wave64 NVIDIA mode
or measured FPS improvement.
