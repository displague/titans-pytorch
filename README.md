<img src="./fig2.png" width="400px"></img>

<img src="./fig1.png" width="400px"></img>

## Titans - Pytorch
### Symplectic Gating and Manifold Paging (Objective Reduction)

This project explores optional, research-only extensions to Titans' Neural Memory based on symplectic "complexity" signals and Penrose-style Objective Reduction.

- Baseline remains unchanged by default. All new behavior is behind flags.
- Symplectic Gating: attenuates memory decay when local twist/complexity is high.
- Manifold Paging: when complexity exceeds a threshold, gradient updates are routed to a fresh "page" (head group) to avoid destructive interference.

Key flags on `NeuralMemory`:

- `use_symplectic_gating: bool` - default False (off). When True, decay is modulated by a learned scale and a per-chunk max of a tanh(wedge) complexity.
- `num_pages: int` - default 1. When >1, internal heads are expanded by pages (user_heads x num_pages). Writes route to the active page; reads pool across all pages.
- `symplectic_page_threshold: float` - threshold for page switch (collapse). Lower values induce more frequent paging; default ~0.5.

Minimal usage:

```python
from titans_pytorch import NeuralMemory

# Baseline
mem_base = NeuralMemory(dim=64, chunk_size=32)

# Gating only
mem_gate = NeuralMemory(dim=64, chunk_size=32, use_symplectic_gating=True)

# Paging (Objective Reduction)
mem_page = NeuralMemory(dim=64, chunk_size=32, use_symplectic_gating=True, num_pages=4)
mem_page.symplectic_page_threshold = 0.2  # optional tuning
```

Trade-offs:

- Interference: Paging can reduce catastrophic forgetting on conflicting tasks (A/B).
- Overhead: Internal ops scale with pages x heads; paging increases compute. Consider smaller `num_pages` (e.g., 2), moderate `chunk_size`, and fewer heads.

Benchmarks:

- Paging Interference Benchmark: `benchmarks/benchmark_paging.py` compares Baseline vs Symplectic+Paging for interference and step-time overhead.
- Symplectic Timing Benchmark: see `benchmarks/benchmark_symplectic.py`.

Paging threshold bands (toy sweep, `benchmark_threshold_sweep.py`):

- High switch-rate: `symplectic_page_threshold` in `0.05` to `0.10` (roughly 4 switches per 12 steps).
- Moderate switch-rate: `0.20` (roughly 0.7 switches per 12 steps).
- Low or off: `0.35` and above (zero switches observed in the sweep).

Recalibrate these bands for your sequence length, chunk size, and gate settings.

Expected outcomes (toy setting):

- Paging reduces interference (lower post-B loss on task A).
- Some training step overhead; tune pages/threshold/chunk sizes to balance performance vs stability.

Notes:

## Tiny MAC Transformer demo: Symplectic Gating + Manifold Paging

This repo includes an optional experimental path to reduce interference in the Neural Memory via Symplectic Gating and Manifold Paging ("Objective Reduction"). Baseline behavior is unchanged unless you enable the flags.

### Quick CPU-friendly benchmark

Run a tiny MAC Transformer benchmark that compares baseline vs gating+paging on synthetic data:

```bash
python benchmarks/benchmark_mac_paging.py --steps 100 --pages 2 --threshold 0.2
```

It prints training/validation loss means, wall-clock time, and page-switch counts per layer (when gating+paging is enabled).

Tips if you see zero page switches:

- Lower the threshold (e.g., `--threshold 0.05` to `0.15`) so the symplectic complexity triggers paging more readily on small CPU runs.
- Increase steps or model size slightly (e.g., `--steps 200`, `--pages 3`) to encourage dynamics that cross the threshold.
- Switches are reported per layer via `page_switch_events`; total is also summarized.

### Training script flags (enwik8)

In `train_mac.py`, the following flags toggle the experimental features:

- `USE_SYMPLECTIC_GATING` (default False)
- `NUM_PAGES` (default 1)
- `SYMPLECTIC_PAGE_THRESHOLD` (default 0.5)

The training script automatically falls back to CPU and disables Flex Attention when CUDA is not available.

### Notes

- Paging increases internal heads by `heads * num_pages` and routes writes to the active page while pooling reads.
- Symplectic Gating uses a tanh-normalized wedge product measure of complexity and modulates decay to retain high-twist segments.
- Page switches are thresholded by `SYMPLECTIC_PAGE_THRESHOLD` and counted per layer as `page_switch_events`.

- These extensions are experimental and optional. Set `use_symplectic_gating=False` and `num_pages=1` for baseline behavior.
- Reading pools across pages; writing goes to the active page only.

Unofficial implementation of [Titans](https://arxiv.org/abs/2501.00663) in Pytorch. Will also contain some explorations into architectures beyond their simple 1-4 layer MLP for the neural memory module, if it works well to any degree.

[Paper review by Yannic](https://www.youtube.com/watch?v=v67plFw1nMw)

[Quick Colab Run](https://colab.research.google.com/drive/11cGgSABykte3qbK-hjzPgLif3-9UUejm?usp=sharing)

## Appreciation

- [Eryk](https://github.com/sentialx) for sharing his early experimental results with me, positive for 2 layer MLP

## Install

```bash
$ pip install titans-pytorch
```

## Usage

```python
import torch
from titans_pytorch import NeuralMemory

mem = NeuralMemory(
    dim = 384,
    chunk_size = 64 # set to smaller chunk size for better perf on smaller sequence lengths (but more memory usage)
).cuda()

seq = torch.randn(2, 1024, 384).cuda()
retrieved, mem_state = mem(seq)

assert seq.shape == retrieved.shape
```

A transformer with the `MAC` configuration can be used as

```python
import torch
from titans_pytorch import MemoryAsContextTransformer

transformer = MemoryAsContextTransformer(
    num_tokens = 256,
    dim = 256,
    depth = 2,
    segment_len = 128,              # local attention window size
    num_persist_mem_tokens = 4,
    num_longterm_mem_tokens = 16,
)

token_ids = torch.randint(0, 256, (1, 1023))

loss = transformer(token_ids, return_loss = True) # (1, 1023, 256)
loss.backward()

# after much training

sampled = transformer.sample(token_ids[:, :4], 512)
```

## Experimental Symplectic Gate Options

`NeuralMemory` now accepts optional `symplectic_gate_kwargs` when `use_symplectic_gating=True`.
Defaults are unchanged, so baseline behavior is preserved unless you opt in.

Example:

```python
mem = NeuralMemory(
    dim = 64,
    chunk_size = 32,
    use_symplectic_gating = True,
    use_dmd_gating = True,
    combine_symplectic_and_dmd = True,  # optional blended complexity signal
    num_pages = 2,
    symplectic_gate_kwargs = dict(
        gated = True,
        diag = True,
        gate_mode = "soft",    # "hard" or "soft"
        top_k = 8,             # optional sparse routing
        phase_mix = 0.5,       # blend in periodic phase complexity
        phase_pairs = 4
    )
)
```

Key optional flags in `SymplecticGating`:

- `gated`, `diag`: gated SAE-style projection and diagonal projection proxy.
- `gate_mode`: hard straight-through gating or soft gating.
- `top_k`, `adaptive_topk_ratio`: sparse routing controls.
- `phase_mix`, `phase_pairs`: periodic latent phase-complexity signal.
- `quorum_mix`, `quorum_window`, `quorum_threshold`, `quorum_temperature`: quorum-style local consensus filter on complexity.
- `budget_topk_ratio`: optional sequence budget for quorum-selected positions.
- `codebook_mix`, `codebook_size`, `codebook_temperature`, `codebook_topk`: combinatorial codebook routing signal (taste/odor-inspired mixture coding).

Additional optional `NeuralMemory` toggles:

- `combine_symplectic_and_dmd`: blend symplectic and DMD complexity signals.
- `manifold_state_keyed_paging`: route page selection from manifold phase-angle keys instead of only incrementing page index on threshold crossings.
- `hierarchical_paging`: enable two-stage coarse/fine page routing.
- `coarse_pages`, `fine_pages`: coarse routing groups and pages per group (`coarse_pages * fine_pages = num_pages`).
- `hierarchy_mix`: blend between sequential fallback (`0`) and hierarchical target routing (`1`).
- `kinetics_coupling`: enable chemistry-inspired adaptive-lr/decay coupling.
- `kinetics_mix`, `kinetics_eps`: coupling strength and normalization stability constants.

Related research context:

- Anthropic dictionary-learning and gated sparse autoencoder work.
- Periodic closed-loop latent dynamics (spiral/helix manifold framing).

## Regression Tracking

Use benchmark JSON output to track improvements/regressions over time:

```bash
python benchmarks/benchmark_symplectic.py --tag nightly --output-json benchmarks/results/symplectic_latest.json
```

Additional experiment runners:

```bash
python benchmarks/benchmark_gate_variants.py --tag gate_variants --output-json benchmarks/results/gate_variants_latest.json
python benchmarks/benchmark_threshold_sweep.py --tag threshold_sweep --output-json benchmarks/results/threshold_sweep_latest.json
python benchmarks/benchmark_mutation_selection.py --tag mutation_selection --output-json benchmarks/results/mutation_selection_latest.json
python benchmarks/benchmark_mutation_selection.py --tag mutation_selection_transfer --use-transfer-fitness --transfer-weight 0.7 --steps 8 --population 6 --generations 3 --output-json benchmarks/results/mutation_selection_latest.json
python benchmarks/benchmark_switch_budget.py --tag switch_budget --output-json benchmarks/results/switch_budget_latest.json
python benchmarks/benchmark_codebook_sweep.py --tag codebook_sweep --output-json benchmarks/results/codebook_sweep_latest.json
python benchmarks/benchmark_codebook_transfer.py --tag codebook_transfer --output-json benchmarks/results/codebook_transfer_latest.json
python benchmarks/benchmark_manifold_paging_transfer.py --tag manifold_paging_transfer --output-json benchmarks/results/manifold_paging_latest.json
python benchmarks/benchmark_mutation_transfer.py --tag mutation_transfer --output-json benchmarks/results/mutation_transfer_latest.json
python benchmarks/benchmark_kinetics_sweep.py --tag kinetics_sweep --output-json benchmarks/results/kinetics_sweep_latest.json
python benchmarks/benchmark_switch_budget_sweep.py --tag switch_budget_sweep --output-json benchmarks/results/switch_budget_sweep_latest.json
python benchmarks/check_regression.py --baseline benchmarks/results/symplectic_baseline.json --latest benchmarks/results/symplectic_latest.json --codebook-baseline benchmarks/results/codebook_transfer_baseline.json --codebook-latest benchmarks/results/codebook_transfer_latest.json
```

## Nanochat Transfer Pilot (16GB)

The repo includes an isolated pilot harness for trying Titans-inspired ideas in `nanochat` without changing Titans defaults.
Recommended flow is short-window screening first, then full-horizon runs:
- Use small iteration windows to rank control vs candidate quickly.
- Promote only strong candidates to the full ~24h build budget.
- On non-FA3 GPUs, use `window_pattern=L` for better utilization (harness default).
- On Windows/Triton-missing setups, compile is disabled by default in the harness (`DisableTorchCompile=true`).
- Serialize GPU-heavy jobs (benchmarks/protocols/pytest) to avoid cross-run contention.

```powershell
powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/setup_nanochat.ps1
powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/run_nanochat_16gb_smoke.ps1
powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/apply_candidate_patch.ps1
powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/run_nanochat_24h_protocol.ps1 -ApplyCandidatePatch -NumIterations 30000 -Seeds "1337,2026"
# optional retune sweep example
powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/run_nanochat_24h_protocol.ps1 -ApplyCandidatePatch -NumIterations 64 -Seeds "1337,2026" -CandidateGateMix 0.05 -CandidateWeightDecay 0.2 -CandidateMatrixLr 0.02 -RunLabel mix005_n64 -OutputJson experiments/nanochat_transfer/results/nanochat_protocol_mix005_n64_latest.json -OutputCsv experiments/nanochat_transfer/results/nanochat_protocol_mix005_n64_history.csv
# structural recipe: odd layers only
powershell -ExecutionPolicy Bypass -File experiments/nanochat_transfer/run_nanochat_24h_protocol.ps1 -ApplyCandidatePatch -NumIterations 64 -Seeds "1337,2026" -CandidateGateMix 0.05 -CandidateOddLayersOnly -CandidateWeightDecay 0.2 -CandidateMatrixLr 0.02 -RunLabel odd_mix005_n64 -OutputJson experiments/nanochat_transfer/results/nanochat_protocol_odd_mix005_n64_latest.json -OutputCsv experiments/nanochat_transfer/results/nanochat_protocol_odd_mix005_n64_history.csv
```

Protocol summary artifacts:
- `experiments/nanochat_transfer/results/nanochat_protocol_latest.json`
- `experiments/nanochat_transfer/results/nanochat_protocol_history.csv`
- Includes per-run `val_bpb`, `duration_sec`, `avg_tok_per_sec`, plus candidate-vs-control deltas.

Latest short-window checkpoints (2 seeds, RTX 5080 laptop GPU):
- Full-depth candidate (`mix005_n64`): `mean_candidate_minus_control_bpb=+0.003640`, `mean_candidate_speed_ratio=0.858812`.
- Odd-layer candidate (`odd_mix005_n64`): `mean_candidate_minus_control_bpb=+0.002307`, `mean_candidate_speed_ratio=0.918117`.
- Odd-layer candidate (`odd_mix005_n128`): `mean_candidate_minus_control_bpb=+0.038428`, `mean_candidate_speed_ratio=0.911419`.
- Interpretation: odd-layer routing improves throughput and narrows quality gap, but does not yet clear promotion criteria at `n128`.

## Experiments

```bash
$ pip install uv
```

Then modify `train_mac.py` and run it to query nature

```bash
$ uv run train_mac.py
```

## Citations

```bibtex
@inproceedings{Behrouz2024TitansLT,
    title   = {Titans: Learning to Memorize at Test Time},
    author  = {Ali Behrouz and Peilin Zhong and Vahab S. Mirrokni},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:275212078}
}
```

```bibtex
@article{Sun2024LearningT,
    title   = {Learning to (Learn at Test Time): RNNs with Expressive Hidden States},
    author  = {Yu Sun and Xinhao Li and Karan Dalal and Jiarui Xu and Arjun Vikram and Genghan Zhang and Yann Dubois and Xinlei Chen and Xiaolong Wang and Oluwasanmi Koyejo and Tatsunori Hashimoto and Carlos Guestrin},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2407.04620},
    url     = {https://api.semanticscholar.org/CorpusID:271039606}
}
```

```bibtex
@inproceedings{Yang2024GatedDN,
    title   = {Gated Delta Networks: Improving Mamba2 with Delta Rule},
    author  = {Songlin Yang and Jan Kautz and Ali Hatamizadeh},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:274598177}
}
```

```bibtex
@inproceedings{Nguyen2024TurningUT,
    title   = {Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs},
    author  = {Minh Nguyen and Andrew Baker and Clement Neo and Allen Roush and Andreas Kirsch and Ravid Shwartz-Ziv},
    year    = {2024},
    url     = {https://api.semanticscholar.org/CorpusID:270870613}
}
```

```bibtex
@article{Zhu2024HyperConnections,
    title   = {Hyper-Connections},
    author  = {Defa Zhu and Hongzhi Huang and Zihao Huang and Yutao Zeng and Yunyao Mao and Banggu Wu and Qiyang Min and Xun Zhou},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2409.19606},
    url     = {https://api.semanticscholar.org/CorpusID:272987528}
}
```

```bibtex
@article{Zhou2024ValueRL,
    title   = {Value Residual Learning For Alleviating Attention Concentration In Transformers},
    author  = {Zhanchao Zhou and Tianyi Wu and Zhiyun Jiang and Zhenzhong Lan},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2410.17897},
    url     = {https://api.semanticscholar.org/CorpusID:273532030}
}
```

```bibtex
@software{Kyrylov_Accelerated_Scan_2024,
    author  = {Kyrylov, Volodymyr},
    doi     = {10.5281/zenodo.10600962},
    title   = {Accelerated Scan},
    version = {0.1.2},
    year    = {2024}
}
```

```bibtex
@misc{wang2025testtimeregressionunifyingframework,
    title   = {Test-time regression: a unifying framework for designing sequence models with associative memory},
    author  = {Ke Alexander Wang and Jiaxin Shi and Emily B. Fox},
    year    = {2025},
    eprint  = {2501.12352},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG},
    url     = {https://arxiv.org/abs/2501.12352},
}
```

```bibtex
@misc{jordan2024muon,
    author  = {Keller Jordan and Yuchen Jin and Vlado Boza and Jiacheng You and
                    Franz Cesista and Laker Newhouse and Jeremy Bernstein},
    title   = {Muon: An optimizer for hidden layers in neural networks},
    year    = {2024},
    url     = {https://kellerjordan.github.io/posts/muon/}
}
```

```bibtex
@inproceedings{Zhang2025TestTimeTD,
    title   = {Test-Time Training Done Right},
    author  = {Tianyuan Zhang and Sai Bi and Yicong Hong and Kai Zhang and Fujun Luan and Songlin Yang and Kalyan Sunkavalli and William T. Freeman and Hao Tan},
    year    = {2025},
    url     = {https://api.semanticscholar.org/CorpusID:279071244}
}
```

```bibtex
@inproceedings{Behrouz2025ATLASLT,
    title  = {ATLAS: Learning to Optimally Memorize the Context at Test Time},
    author = {Ali Behrouz and Ze-Minghui Li and Praneeth Kacham and Majid Daliri and Yuan Deng and Peilin Zhong and Meisam Razaviyayn and Vahab S. Mirrokni},
    year   = {2025},
    url    = {https://api.semanticscholar.org/CorpusID:278996373}
}
```

