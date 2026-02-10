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

## 16GB Guidance
- Start with `--depth=12`, `--max-seq-len=512` or `1024`, and `--device-batch-size=1` or `2`.
- Keep `--nproc_per_node=1`.
- Use `--total-batch-size` to recover effective batch via gradient accumulation.
- Increase only one pressure knob at a time (depth, seq len, or device batch size).

## Suggested Sequence
1. Run setup:
   - `powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/setup_nanochat.ps1`
2. Run smoke fit:
   - `powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/run_nanochat_16gb_smoke.ps1 -PrepareData`
3. If stable, run long protocol:
   - `powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/run_nanochat_24h_protocol.ps1 -PrepareData -NumIterations 30000`

## Notes
- This harness does not modify `nanochat` source by default.
- Candidate slots in `run_nanochat_24h_protocol.ps1` are intentionally easy to edit as transfer patches mature.
