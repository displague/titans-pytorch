# TODO

## Active
- [2026-02-08 17:55:14] Wire `benchmarks/check_regression.py` into CI with a checked-in rolling baseline artifact policy.
- [2026-02-08 19:54:39] Compare mutation/selection champion against hand-designed variants on long-horizon and interference tasks.
- Hypothesis: evolved gate configs can outperform manually curated variants under equal step/latency budgets.
- Toggle plan: reuse mutation benchmark winner as a named preset in benchmark scripts only; runtime defaults unchanged.
- [2026-02-08 20:38:49] Calibrate kinetics-coupled configs on long-horizon recovery and interference benchmarks.
- Hypothesis: tuned `kinetics_mix` with quorum/phase options improves recovery without excessive switch suppression.
- Toggle plan: benchmark-only sweep over `kinetics_mix` and compare against mutation-selection champion; runtime defaults unchanged.
- [2026-02-08 22:57:50] Calibrate switch-budget targets (`switch_target`, `entropy_target`) for nontrivial switch-rate reduction.
- Hypothesis: target-band tuning can lower switch-rate without regressing reconstruction quality.
- Toggle plan: benchmark-only sweeps in `benchmark_switch_budget.py`; core runtime defaults unchanged.

## Research Narrative (Cross-Field Source Set)
- Microbiology -> quorum thresholding: `https://pubmed.ncbi.nlm.nih.gov/37057353/`
- Genetics -> transcriptional memory and gated expression persistence: `https://pubmed.ncbi.nlm.nih.gov/39029953/`
- Chemistry -> reaction-network memory and temporal processing: `https://pubmed.ncbi.nlm.nih.gov/39058812/`
- Taste detection -> distributed coding under task demands: `https://pubmed.ncbi.nlm.nih.gov/38631343/`
- Odor detection -> combinatorial code remapping: `https://www.nature.com/articles/s41586-025-09053-w`
- Taxonomy science -> hierarchy/network-informed structure: `https://journals.asm.org/doi/10.1128/mbio.02256-23`
- Information theory -> rate/distortion/perception/semantics tradeoff: `https://doi.org/10.1016/j.jfranklin.2024.106884`
- Economic theory -> rational inattention with bounded information budgets: `https://link.springer.com/article/10.1007/s10683-023-09803-9`

## Future Directions
- [2026-02-08 17:23:12] Advanced next-next topic: Renormalization-Group Memory Ladder (multiscale coarse-graining + refinement).
- Parallels:
- Cosmology: scale-dependent structure growth and horizon crossing.
- Quantum physics / QFT: RG flow and effective theories by scale.
- String theory: worldsheet beta flow and scale consistency constraints.
- Statistics: hierarchical shrinkage / partial pooling across resolutions.
- Topology: persistent homology across filtration scales.
- Language theory: hierarchy from local syntax to discourse-level constraints.
- Fusion research: turbulence cascades and confinement across scales.
- [2026-02-08 17:38:45] Expand cross-field inspiration set for new experimental hypotheses:
- Neuroscience: predictive coding, hippocampal replay, attractor consolidation.
- Social psychology: context framing, priming persistence, group-level interference analogs.
- Linear algebra: low-rank + sparse decompositions, orthogonality control, spectral conditioning.
- [2026-02-08 17:58:10] Next next-next inspiration pool for cross-field experimental hypotheses:
- Microbiology: colony dynamics, quorum-like signaling, and resilience under stress.
- Genetics: regulatory circuits, expression gating, mutation/selection style search over gating policies.
- Chemistry: reaction kinetics and equilibrium analogs for memory update vs decay balance.
- Taste/odor detection: combinatorial receptor coding and sparse mixture separation.
- Taxonomy science: hierarchical class structure and stable branch-specific specialization.
- Information theory: bottleneck trade-offs, coding efficiency, and surprise-driven switching.
- Economic theory: resource allocation, market equilibrium, and incentive-compatible routing.

## Process Notes
- Completed items move to `IMPLEMENTED.md` with timestamps and test evidence.
