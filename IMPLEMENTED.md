# IMPLEMENTED

## Branch History (origin/main..feat/symplectic-gating)
- [2025-12-20 18:48:48] `f0de542` `feat: Add Symplectic Gating mechanism for adaptive memory decay`
- [2025-12-22 14:34:08] `7736ff7` `Refactor Symplectic Gate: Use Tanh, Max Pooling, and Learnable Scale`
- [2025-12-22 14:34:27] `ed0d149` `Add Symplectic verification benchmarks`
- [2025-12-23 07:45:18] `465cdd8` `Fix vmap dimensions, implement Manifold Paging for Objective Reduction`
- [2025-12-30 17:49:47] `eae41a9` `feat: integrate symplectic reduction with moment map and per-sample paging`
- [2025-12-30 19:36:18] `2575ca4` `feat: add Dynamic Mode Decomposition (DMD) module and DMD-based Neural Memory Gating`
- [2026-02-08 15:15:45] `bf7839d` `Add gated symplectic options and synthetic benchmarks`
- [2026-02-08 15:35:44] `9b277c5` `Guard torch.compile and ignore uv lockfile`
- [2026-02-08 16:23:00] `bf58584` `Add soft/top-k gating and tracked progress docs`
- [2026-02-08 16:28:43] `792d65d` `Refresh IMPLEMENTED history after rebase`
- [2026-02-08 16:55:40] `dd8ef58` `Add phase-aware gating and benchmark regression tracking`
- [2026-02-08 17:17:26] `5cd9b22` `Advance manifold gating, ablations, and regression tracking`
- [2026-02-08 17:26:30] `8e6f4ac` `Add long-horizon recovery benchmark and next research topic`

## Uncommitted Work (Current Session)
- [2026-02-08 17:40:33] Added manifold-state keyed paging prototype in `NeuralMemory`.
- New constructor toggle: `manifold_state_keyed_paging=False` (opt-in, no default behavior change).
- When enabled and manifold state is available, page routing uses circular mean phase angle to select a page bucket.
- Added regression test: `tests/test_symplectic_reduction.py::test_manifold_state_keyed_paging_routes_by_angle_bucket`.

- [2026-02-08 17:40:33] Added gate-variant ablation benchmark script: `benchmarks/benchmark_gate_variants.py`.
- Variants: `symplectic_default`, `hard_diag`, `soft_diag`, `hard_topk8`, `adaptive_topk`, `phase_mix`.
- Outputs: `benchmarks/results/gate_variants_latest.json`, `benchmarks/results/gate_variants_history.csv`.
- Run tag `gate_variants_v2` (CUDA, steps=12):
- `symplectic_default`: spiral `0.027641`, helix `0.026637`.
- `hard_diag`: spiral `0.033863`, helix `0.026360`.
- `soft_diag`: spiral `0.031234`, helix `0.026194`.
- `hard_topk8`: spiral `0.025152`, helix `0.027509`, sparse_k `8.0`.
- `adaptive_topk`: spiral `0.031652`, helix `0.025994`, sparse_k `7.46875`.
- `phase_mix`: spiral `0.029299`, helix `0.026211`.

- [2026-02-08 17:40:33] Added threshold calibration benchmark script: `benchmarks/benchmark_threshold_sweep.py`.
- Sweeps `symplectic_page_threshold` and tracks loss plus switch-rate bands.
- Outputs: `benchmarks/results/threshold_sweep_latest.json`, `benchmarks/results/threshold_sweep_history.csv`.
- Run tag `threshold_sweep_v1` (CUDA, steps=12):
- `0.05`: switch_rate `4.000`, mean loss roughly `0.027`.
- `0.10`: switch_rate `4.000`, mean loss roughly `0.029`.
- `0.20`: switch_rate `0.667`, mean loss roughly `0.032`.
- `0.35`: switch_rate `0.000`, mean loss roughly `0.026`.
- `0.50`: switch_rate `0.000`, mean loss roughly `0.029`.
- `0.70`: switch_rate `0.000`, mean loss roughly `0.026`.

- [2026-02-08 17:55:14] Added regression guard utility: `benchmarks/check_regression.py`.
- Compares baseline vs latest benchmark JSON on selected timing/loss/recovery metrics with configurable tolerance gates.
- Exits non-zero on regression so it can be used in CI.
- Validation run:
- `python benchmarks/check_regression.py --baseline benchmarks/results/symplectic_latest.json --latest benchmarks/results/symplectic_latest.json` -> `passed`.

- [2026-02-08 18:57:55] Added cross-field narrative hypothesis for next experiment set in `TODO.md`.
- Source pool spans microbiology, genetics, chemistry, taste/odor detection, taxonomy science, information theory, and economic theory.
- Narrative target: quorum-budgeted combinatorial gating for robust page switching under noise.

- [2026-02-08 18:57:55] Added quorum-budgeted gating experiment controls in `SymplecticGating` (all opt-in).
- New kwargs: `quorum_mix`, `quorum_window`, `quorum_threshold`, `quorum_temperature`, `budget_topk_ratio`.
- Added `apply_quorum_policy(...)` helper and optional `return_quorum_map` forward output.
- Integrated quorum policy into final complexity score post-processing.

- [2026-02-08 18:57:55] Added tests for quorum policy behavior in `tests/test_symplectic.py`:
- `test_symplectic_gate_quorum_policy_prefers_sustained_signal`
- `test_symplectic_gate_quorum_budget_limits_positions`
- `test_symplectic_gate_quorum_budget_requires_quorum_mix`

- [2026-02-08 18:57:55] Extended gate ablation benchmark in `benchmarks/benchmark_gate_variants.py`:
- Added `quorum_budget` variant.
- Added quorum diagnostics (`quorum_mean`, `quorum_active_frac`) to gate summaries.
- Run tag `gate_variants_v3` (CUDA, steps=12):
- `quorum_budget`: spiral `0.033196`, helix `0.023264`, quorum `0.1178`.

- [2026-02-08 19:29:49] Implemented taxonomy-inspired hierarchical page routing in `NeuralMemory` (opt-in).
- New constructor kwargs: `hierarchical_paging`, `coarse_pages`, `fine_pages`, `hierarchy_mix`.
- Routing policy:
- coarse page from complexity bucket.
- fine page from manifold phase-angle bucket when available, else sequential fallback.
- blended target via `hierarchy_mix` in `[0, 1]`.
- Added validation checks: `coarse_pages * fine_pages == num_pages` and range guards.

- [2026-02-08 19:29:49] Added hierarchical routing test:
- `tests/test_symplectic_reduction.py::test_hierarchical_paging_routes_with_coarse_and_fine_keys`.

- [2026-02-08 19:29:49] Extended gate ablation benchmark with `hierarchical_route` variant.
- Run tag `gate_variants_v4` (CUDA, steps=12):
- `hierarchical_route`: spiral `0.029726`, helix `0.027642`, complexity `0.1531`.

## Validation
- [2026-02-08 17:40:33] `python -m pytest -q tests/test_symplectic.py` -> `14 passed`.
- [2026-02-08 17:40:33] `python -m pytest -q tests/test_symplectic_reduction.py` -> `5 passed`.
- [2026-02-08 17:40:33] `python -m pytest -q tests/test_titans.py` -> `5193 passed`, `5 skipped`.
- [2026-02-08 19:03:42] `python -m pytest -q tests/test_symplectic.py` -> `17 passed`.
- [2026-02-08 19:03:42] `python -m pytest -q tests/test_symplectic_reduction.py` -> `5 passed`.
- [2026-02-08 19:03:42] `python -m pytest -q tests/test_titans.py` -> `5193 passed`, `5 skipped`.
- [2026-02-08 19:29:49] `python -m pytest -q tests/test_symplectic.py` -> `17 passed`.
- [2026-02-08 19:29:49] `python -m pytest -q tests/test_symplectic_reduction.py` -> `6 passed`.
- [2026-02-08 19:29:49] `python -m pytest -q tests/test_titans.py` -> `5193 passed`, `5 skipped`.

## Decisions
- [2026-02-08 17:40:33] Keep all new experimental behavior opt-in by constructor and kwargs toggles.
- [2026-02-08 17:40:33] Keep branch tracking explicit: `README.md` for usage, `TODO.md` for planned work, `IMPLEMENTED.md` for timestamped execution history and results.
