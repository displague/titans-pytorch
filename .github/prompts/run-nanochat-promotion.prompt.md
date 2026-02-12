---
mode: agent
description: "Run the nanochat promotion cycle (protocol + postcheck) and update TODO/IMPLEMENTED with reproducible evidence."
---

Execute a promoted nanochat transfer run using repository policy from `AGENTS.md`.

Required workflow:

1. Confirm environment and scope
- Use `experiments/nanochat_transfer/` as the canonical harness.
- Keep external edits reproducible through patch/apply/revert scripts.
- Avoid concurrent GPU-heavy pytest/benchmark/protocol jobs.

2. Run full-cycle promotion
- Use `experiments/nanochat_transfer/run_nanochat_full_cycle.ps1`.
- Provide explicit `RunLabel`, `NumIterations`, `Seeds`, and checkpoint cadence.
- If interrupted, resume with continuation settings instead of starting over.

3. Capture machine-readable outputs
- Collect protocol summary JSON/CSV and postcheck JSON outputs.
- Record key deltas: quality (`mean_candidate_minus_control_bpb`) and throughput (`mean_candidate_speed_ratio`).

4. Update process files
- In `TODO.md`, add/update active run status and recovery state.
- In `IMPLEMENTED.md`, add a timestamped entry with commands, artifacts, and interpretation.
- Record superseded and replacement run tags if contention/interruption required reruns.

5. Report decision state
- State whether candidate meets promotion thresholds.
- List unresolved risks and the next control vs champion run recommendation.

Return:
- Commands executed.
- Artifact paths produced.
- Promotion decision summary and follow-up actions.
