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

- [2026-02-08 19:54:39] Implemented genetics-inspired mutation/selection search benchmark.
- Added `benchmarks/benchmark_mutation_selection.py`.
- Search space covers `phase_mix`, `quorum_mix`, `budget_topk_ratio`, and hierarchical routing toggles.
- Uses population selection with mutation and elite carry-over; no runtime behavior changes to core modules.
- Outputs:
- `benchmarks/results/mutation_selection_latest.json`
- `benchmarks/results/mutation_selection_history.csv`
- Run tag `mutation_selection_v1` (CUDA, steps=8, population=6, generations=3):
- Best fitness `0.061281`, mean loss `0.041680`, mean step `19.60 ms`.
- Best candidate:
- `phase_mix=0.5`, `quorum_mix=0.75`, `budget_topk_ratio=0.3`,
- `hierarchical=False`, `quorum_window=3`, `quorum_threshold=0.2`, `quorum_temperature=0.15`.

- [2026-02-08 20:38:49] Implemented chemistry-inspired kinetics coupling in `NeuralMemory` (opt-in).
- New constructor kwargs: `kinetics_coupling`, `kinetics_mix`, `kinetics_eps`.
- Coupling policy:
- derives chunk-level reaction progress from adaptive-lr/decay balance.
- reduces decay under high reaction progress.
- increases adaptive lr toward a bounded target under high reaction progress.
- Baseline behavior remains unchanged by default.

- [2026-02-08 20:38:49] Added kinetics tests:
- `tests/test_symplectic_reduction.py::test_kinetics_coupling_modulates_adaptive_lr`
- `tests/test_symplectic_reduction.py::test_kinetics_coupling_zero_mix_is_noop_for_adaptive_lr`

- [2026-02-08 20:38:49] Extended gate ablation benchmark with `kinetics_coupled` variant.
- Run tag `gate_variants_v5` (CUDA, steps=12):
- `kinetics_coupled`: spiral `0.028904`, helix `0.029320`, complexity `0.0463`, quorum `0.1632`.

- [2026-02-08 22:57:50] Implemented information/economic budget controller benchmark loop.
- Added `benchmarks/benchmark_switch_budget.py`.
- Adds benchmark-time penalties:
- switch-budget penalty for exceeding target switch rate.
- routing-entropy penalty toward target entropy (rational inattention proxy).
- No runtime model behavior changes; benchmark-only objective shaping.
- Outputs:
- `benchmarks/results/switch_budget_latest.json`
- `benchmarks/results/switch_budget_history.csv`
- Run tag `switch_budget_v1` (CUDA, steps=20, page-threshold=0.05):
- Spiral: unconstrained recon `0.015061`, constrained recon `0.011531`, entropy `0.086`.
- Helix: unconstrained recon `0.010996`, constrained recon `0.009805`, entropy `0.045`.

- [2026-02-09 00:54:38] Implemented taste/odor-inspired combinatorial codebook probe in `SymplecticGating` (opt-in).
- New gate kwargs: `codebook_mix`, `codebook_size`, `codebook_temperature`, `codebook_topk`.
- Policy:
- projects tokens to a codebook assignment simplex.
- computes transition intensity from total-variation distance between consecutive assignments.
- blends codebook transition score into complexity (`codebook_mix`).
- Added optional `return_codebook_map` output path for diagnostics.

- [2026-02-09 00:54:38] Added codebook tests in `tests/test_symplectic.py`:
- `test_symplectic_gate_codebook_map_shape`
- `test_symplectic_gate_codebook_prefers_mixture_shifts`

- [2026-02-09 00:54:38] Extended gate ablation benchmark with `combinatorial_codebook` variant.
- Run tag `gate_variants_v6` (CUDA, steps=12):
- `combinatorial_codebook`: spiral `0.029904`, helix `0.030032`, codebook `0.3595`, quorum `0.5643`.

- [2026-02-09 08:03:57] Added dedicated codebook calibration benchmark for multi-motif recall.
- Added `benchmarks/benchmark_codebook_sweep.py`.
- Sweeps `codebook_mix`, `codebook_size`, and `codebook_topk` using a multi-motif sequence with boundary-focused error.
- Outputs:
- `benchmarks/results/codebook_sweep_latest.json`
- `benchmarks/results/codebook_sweep_history.csv`
- Run tag `codebook_sweep_v1` (CUDA, steps=8):
- Best config: `codebook_mix=0.5`, `codebook_size=16`, `codebook_topk=None`.
- Best metrics: boundary MSE `0.009467`, total MSE `0.009267`, step `21.97 ms`, score `0.031376`.

- [2026-02-10 06:45:13] Added transfer benchmark for champion-vs-non-codebook comparisons on long-horizon and interference tasks.
- Added `benchmarks/benchmark_codebook_transfer.py`.
- Variants compared under identical settings:
- `baseline`
- `symplectic_paging`
- `phase_quorum_paging`
- `codebook_champion_paging` (`codebook_mix=0.5`, `codebook_size=16`, `codebook_topk=None`)
- Outputs:
- `benchmarks/results/codebook_transfer_latest.json`
- `benchmarks/results/codebook_transfer_history.csv`
- Run tag `codebook_transfer_v1` (CUDA):
- Winner: `codebook_champion_paging` with transfer score `0.100389`.
- Relative to `phase_quorum_paging`, champion improved:
- long-horizon clean post-MSE from `0.004147` to `0.003696` (`+10.87%`).
- long-horizon phase error from `0.058091` to `0.051676` (`+11.04%`).
- interference post-B task-A loss from `0.008036` to `0.007104` (`+11.60%`).
- aggregate transfer score from `0.105256` to `0.100389` (`+4.62%`).

- [2026-02-10 07:50:11] Extended regression guard to cover codebook transfer champion metrics.
- Added optional `--codebook-baseline` and `--codebook-latest` inputs to `benchmarks/check_regression.py`.
- Tracks `clean_mse_post`, `phase_err_post`, `interference.post_a_loss`, and `transfer_score` for `codebook_champion_paging`.

- [2026-02-10 07:52:39] Added manifold-keyed paging transfer benchmark against threshold paging.
- Added `benchmarks/benchmark_manifold_paging_transfer.py`.
- Outputs:
- `benchmarks/results/manifold_paging_latest.json`
- `benchmarks/results/manifold_paging_history.csv`
- Run tag `manifold_paging_transfer_v1` (CUDA):
- `threshold_paging`: long clean post `0.004963`, phase post `0.049742`, interference post-A `0.007729`, score `0.119738`.
- `manifold_keyed_paging`: long clean post `0.004750`, phase post `0.056093`, interference post-A `0.013046`, score `0.110337`.
- Summary: manifold-keyed paging lowered the aggregate transfer score but increased post-A interference loss, so it is a tradeoff rather than a clear win.

- [2026-02-10 07:53:25] Documented paging threshold switch-rate bands in `README.md` based on `benchmark_threshold_sweep.py`.
- High switch-rate: `0.05` to `0.10`.
- Moderate: `0.20`.
- Low or off: `0.35` and above.

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
- [2026-02-08 20:38:49] `python -m pytest -q tests/test_symplectic.py` -> `17 passed`.
- [2026-02-08 20:38:49] `python -m pytest -q tests/test_symplectic_reduction.py` -> `8 passed`.
- [2026-02-08 20:38:49] `python -m pytest -q tests/test_titans.py` -> `5193 passed`, `5 skipped`.
- [2026-02-08 22:57:50] `python -m pytest -q tests/test_symplectic.py` -> `17 passed`.
- [2026-02-08 22:57:50] `python -m pytest -q tests/test_symplectic_reduction.py` -> `8 passed`.
- [2026-02-09 00:54:38] `python -m pytest -q tests/test_symplectic.py` -> `19 passed`.
- [2026-02-09 00:54:38] `python -m pytest -q tests/test_symplectic_reduction.py` -> `8 passed`.
- [2026-02-09 00:54:38] `python -m pytest -q tests/test_titans.py` -> `5193 passed`, `5 skipped`.
- [2026-02-09 08:03:57] `python -m pytest -q tests/test_symplectic.py` -> `19 passed`.
- [2026-02-09 08:03:57] `python -m pytest -q tests/test_symplectic_reduction.py` -> `8 passed`.
- [2026-02-10 07:50:11] `python benchmarks/check_regression.py --baseline benchmarks/results/symplectic_latest.json --latest benchmarks/results/symplectic_latest.json --codebook-baseline benchmarks/results/codebook_transfer_latest.json --codebook-latest benchmarks/results/codebook_transfer_latest.json` -> `passed`.
- [2026-02-10 06:45:13] `python -m pytest -q tests/test_symplectic.py` -> `19 passed`.
- [2026-02-10 06:45:13] `python -m pytest -q tests/test_symplectic_reduction.py` -> `8 passed`.

## Decisions
- [2026-02-08 17:40:33] Keep all new experimental behavior opt-in by constructor and kwargs toggles.
- [2026-02-08 17:40:33] Keep branch tracking explicit: `README.md` for usage, `TODO.md` for planned work, `IMPLEMENTED.md` for timestamped execution history and results.
