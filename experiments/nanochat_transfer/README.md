# Nanochat Transfer Harness (16GB GPU)

This folder is an isolated pilot harness for testing transfer of Titans-inspired ideas into `nanochat` on a single 16GB GPU.

## Goals
- Keep external work isolated from core Titans code.
- Run reproducible control vs candidate experiments on one GPU.
- Produce comparable metrics over long runs (for example, 24h windows).

## Scope
- `setup_nanochat.ps1`: clone/update `nanochat` and print pinned commit info.
- `run_nanochat_16gb_smoke.ps1`: short single-GPU fit/smoke run recipe for 16GB cards.
- `run_nanochat_24h_protocol.ps1`: repeatable control/candidate protocol with seeds.
  - Supports candidate tuning flags: `-CandidateGateMix`, `-CandidateWeightDecay`, `-CandidateMatrixLr`, and `-RunLabel`.
- `apply_candidate_patch.ps1`: applies optional Titans-inspired patch to `nanochat`.
- `revert_candidate_patch.ps1`: reverts the optional patch.

## 16GB Guidance
- Start with `--depth=12`, `--max-seq-len=512` or `1024`, and `--device-batch-size=1` or `2`.
- Use single-process runs (`python -m scripts.base_train`) for Windows compatibility.
- Use `--total-batch-size` to recover effective batch via gradient accumulation.
- Use `--window-pattern L` on non-FA3 GPUs (default in both harness scripts).
- `run_nanochat_16gb_smoke.ps1` and `run_nanochat_24h_protocol.ps1` default to `DisableTorchCompile=true` to avoid Triton dependency failures.
- Increase only one pressure knob at a time (depth, seq len, or device batch size).

## Suggested Sequence
1. Run setup:
   - `powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/setup_nanochat.ps1`
2. Run smoke fit:
   - `powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/run_nanochat_16gb_smoke.ps1`
   - Candidate smoke: `powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/run_nanochat_16gb_smoke.ps1 -ApplyCandidatePatch -EnableCandidateGate -CandidateMix 0.15`
3. Apply optional candidate patch:
   - `powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/apply_candidate_patch.ps1`
4. If stable, run long protocol:
   - `powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/run_nanochat_24h_protocol.ps1 -ApplyCandidatePatch -NumIterations 30000 -Seeds "1337,2026"`
   - Example retune run: `powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/run_nanochat_24h_protocol.ps1 -ApplyCandidatePatch -NumIterations 64 -Seeds "1337,2026" -CandidateGateMix 0.05 -CandidateWeightDecay 0.2 -CandidateMatrixLr 0.02 -RunLabel mix005_n64 -OutputJson experiments/nanochat_transfer/results/nanochat_protocol_mix005_n64_latest.json -OutputCsv experiments/nanochat_transfer/results/nanochat_protocol_mix005_n64_history.csv`
   - Summary output:
   - `experiments/nanochat_transfer/results/nanochat_protocol_latest.json`
   - `experiments/nanochat_transfer/results/nanochat_protocol_history.csv`
   - Summary fields include `val_bpb`, `duration_sec`, `avg_tok_per_sec`, and candidate-control deltas.

## Candidate Patch
- Patch file: `experiments/nanochat_transfer/patches/nanochat_symplectic_candidate.patch`
- Adds optional CLI flags to `scripts.base_train`:
  - `--symplectic-gate-enabled`
  - `--symplectic-gate-mix`
  - `--symplectic-gate-eps`
- Implements tokenwise complexity gating in `nanochat/gpt.py` block updates.
- Defaults are neutral; behavior is unchanged unless flags are enabled.

## Notes
- This harness does not modify `nanochat` source by default.
- Candidate slots in `run_nanochat_24h_protocol.ps1` are intentionally easy to edit as transfer patches mature.
- Single-GPU protocol runs use `python -m scripts.base_train` (not distributed launch) for Windows compatibility.
- If tokenizer artifacts are missing, both run scripts auto-run dataset/tokenizer prep by default.
