# TODO

## Active
- [2026-02-08 17:38:45] Evaluate `manifold_state_keyed_paging` against threshold-step paging on long-horizon and interference tasks.
- [2026-02-08 17:38:45] Convert threshold sweep output into documented recommended switch-rate bands.
- [2026-02-08 17:55:14] Wire `benchmarks/check_regression.py` into CI with a checked-in rolling baseline artifact policy.
- [2026-02-08 19:29:49] Genetics-inspired mutation/selection sweep for gate hyperparameters.
- Hypothesis: lightweight population search over gate toggles can find robust settings faster than manual sweeps.
- Toggle plan: benchmark-time search over `phase_mix`, `quorum_mix`, `budget_topk_ratio`, and hierarchy settings; runtime defaults unchanged.
- [2026-02-08 19:29:49] Chemistry-inspired kinetics controller for memory update/decay coupling.
- Hypothesis: reaction-rate style coupling between `adaptive_lr` and `decay_factor` can improve perturb/recovery balance.
- Toggle plan: optional kinetics coefficient path in `NeuralMemory` (off by default).
- [2026-02-08 19:29:49] Information/economic budget controller for switch-rate targets.
- Hypothesis: bounded-information routing (rational inattention proxy) can keep page-switch entropy in a desired band.
- Toggle plan: optional entropy/switch-budget penalty in benchmark training loops; no baseline behavior change.
- [2026-02-08 19:29:49] Taste/odor combinatorial codebook gating probe.
- Hypothesis: sparse combinatorial receptor-style subspace probing can improve mixture separation in synthetic multi-motif recall.
- Toggle plan: optional codebook projection inside `SymplecticGating` with a small feature bank.

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
