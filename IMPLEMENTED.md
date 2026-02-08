# IMPLEMENTED

## Branch History (origin/main..feat/symplectic-gating)
- [2025-12-20 18:48:48] `f0de542` Added symplectic gating for adaptive memory decay.
- [2025-12-22 14:34:08] `7736ff7` Refactored symplectic gate scoring (`tanh`, max pooling behavior, learnable scale path).
- [2025-12-22 14:34:27] `ed0d149` Added symplectic verification benchmarks.
- [2025-12-23 07:45:18] `465cdd8` Fixed vmap dimensions and implemented manifold paging objective reduction.
- [2025-12-30 17:49:47] `eae41a9` Integrated symplectic reduction with moment map and per-sample paging.
- [2025-12-30 19:36:18] `2575ca4` Added DMD module and DMD-based neural memory gating.
- [2026-02-08 15:15:45] `bf7839d` Added gated symplectic options and synthetic spiral/helix benchmark extensions.
- [2026-02-08 15:35:44] `9b277c5` Guarded `torch.compile` and ignored `uv.lock`.
- [2026-02-08 16:23:00] `bf58584` Added soft/top-k gating and tracked progress docs.

## AI Changes In Progress (Not Yet Committed Before Current Step)
- [2026-02-08 16:54:24] Added phase-aware complexity in `SymplecticGating`:
- New opt-in args: `phase_mix`, `phase_pairs`.
- Added optional `return_phase_map` output.
- [2026-02-08 16:54:24] Added `symplectic_gate_kwargs` passthrough in `NeuralMemory` for configurable gate experiments without changing defaults.
- [2026-02-08 16:54:24] Added tests:
- `tests/test_symplectic.py`: `11 passed` (phase map shape, spiral preference, kwargs integration).
- `tests/test_symplectic_reduction.py`: `3 passed`.
- `tests/test_titans.py`: `5193 passed`, `5 skipped`.
- [2026-02-08 16:54:24] Added benchmark regression tracking output:
- `benchmarks/benchmark_symplectic.py` now supports `--output-json` and `--tag`.
- Added `Symplectic+Phase+Paging` benchmark variant.
- [2026-02-08 16:54:24] Benchmark run (`tag=phase_probe`, CUDA, `benchmarks/results/symplectic_latest.json`):
- Forward ms: baseline `6.04`, symplectic `6.84`, symplectic+phase+paging `7.70`.
- Train ms: baseline `13.46`, symplectic `14.82`, symplectic+phase+paging `19.77`.
- Reconstruction loss: baseline `0.154579`, symplectic `0.137844` (`+10.83%` improvement).
- Spiral loss: baseline `0.068237`, symplectic `0.068459`, symplectic+paging `0.003655`, symplectic+phase+paging `0.003359`.
- Helix+Drift loss: baseline `0.017306`, symplectic `0.013395`, symplectic+paging `0.001002`, symplectic+phase+paging `0.000518`.
- [2026-02-08 16:54:24] Added `README.md` section documenting new gate options and regression tracking command.
- [2026-02-08 16:00:46] Ran GPU benchmark `benchmarks/benchmark_symplectic.py` on CUDA:
- Forward overhead: `+6.82%` (`11.51 ms` vs `10.78 ms`)
- Train overhead: `+27.12%` (`29.80 ms` vs `23.44 ms`)
- Reconstruction loss improvement: `23.45%` (`0.125563` vs `0.164036`)
- Spiral loss: Baseline `0.068931`, Symplectic `0.073620`, Symplectic+Paging `0.003655`
- Helix+Drift loss: Baseline `0.015451`, Symplectic `0.014486`, Symplectic+Paging `0.000549`
- [2026-02-08 16:04:33] Added `gate_mode="soft"` and sparse routing toggles (`top_k`, `adaptive_topk_ratio`) to `SymplecticGating`.
- [2026-02-08 16:04:33] Added and updated tests:
- `tests/test_symplectic.py`: `8 passed`
- `tests/test_symplectic_reduction.py`: `3 passed`
- `tests/test_titans.py`: `5193 passed`, `5 skipped`
- [2026-02-08 16:04:33] Added compatibility skips in `tests/test_titans.py`:
- Skip flex-attention tests when dynamo is unavailable.
- Skip accelerated assoc-scan test when `ninja` is unavailable.

## Decisions
- [2026-02-08 16:04:33] Keep all new gating behavior opt-in by constructor toggles.
- [2026-02-08 16:04:33] Keep Python 3.14 compatibility by guarding compile-dependent paths and tests.
- [2026-02-08 16:27:44] Rebased `feat/symplectic-gating` onto `origin/main` and re-ran post-rebase validation.
- [2026-02-08 16:54:24] Keep benchmark regression snapshots as explicit JSON outputs (`--output-json`) rather than hard-coded committed artifacts.

