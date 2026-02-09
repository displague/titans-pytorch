import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from titans_pytorch import NeuralMemory


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def make_spiral_data(batch, seq_len, dim, device):
    t = torch.linspace(0, 4 * 3.14159, seq_len, device = device)
    x = torch.zeros(batch, seq_len, dim, device = device)
    x[:, :, 0] = torch.cos(t)
    x[:, :, 1] = torch.sin(t)
    if dim > 2:
        x[:, :, 2] = t / t.max()
    if dim > 3:
        x[:, :, 3:] = 0.1 * torch.randn(batch, seq_len, dim - 3, device = device)
    return x


def make_helix_data(batch, seq_len, dim, device):
    t = torch.linspace(0, 6 * 3.14159, seq_len, device = device)
    x = torch.zeros(batch, seq_len, dim, device = device)
    x[:, :, 0] = torch.cos(t)
    x[:, :, 1] = torch.sin(t)
    if dim > 2:
        x[:, :, 2] = t / t.max()
    if dim > 3:
        drift = torch.linspace(0, 1.0, seq_len, device = device)
        x[:, :, 3] = drift
    if dim > 4:
        x[:, :, 4:] = 0.08 * torch.randn(batch, seq_len, dim - 4, device = device)
    return x


def run_one_task(model, data, steps):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    losses = []

    sync(data.device)
    start = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad()
        retrieved, _ = model(data)
        loss = F.mse_loss(retrieved, data)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    sync(data.device)
    end = time.perf_counter()

    return dict(
        final_loss = sum(losses[-10:]) / min(10, len(losses)),
        step_ms = ((end - start) / steps) * 1000
    )


def summarize_gate(model, probe):
    gate = model.symplectic_gate
    with torch.no_grad():
        complexity, quorum, sparse_k = gate(probe, return_quorum_map = True, return_sparse_k = True)
        mean_complexity = complexity.mean().item()
        max_complexity = complexity.max().item()
        mean_quorum = quorum.mean().item()
        quorum_active_frac = (quorum > 0).float().mean().item()
        if sparse_k is None:
            mean_sparse_k = None
        else:
            mean_sparse_k = sparse_k.float().mean().item()

    return dict(
        complexity_mean = mean_complexity,
        complexity_max = max_complexity,
        quorum_mean = mean_quorum,
        quorum_active_frac = quorum_active_frac,
        sparse_k_mean = mean_sparse_k
    )


def flatten_dict(d, prefix = ""):
    flat = {}
    for key, value in d.items():
        key_name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, key_name))
        else:
            flat[key_name] = value
    return flat


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type = str, default = "gate_variants")
    parser.add_argument("--dim", type = int, default = 128)
    parser.add_argument("--seq-len", type = int, default = 128)
    parser.add_argument("--batch-size", type = int, default = 4)
    parser.add_argument("--chunk-size", type = int, default = 32)
    parser.add_argument("--steps", type = int, default = 30)
    parser.add_argument("--pages", type = int, default = 2)
    parser.add_argument("--output-json", type = str, default = None)
    parser.add_argument("--output-csv", type = str, default = None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    spiral = make_spiral_data(args.batch_size, args.seq_len, args.dim, device)
    helix = make_helix_data(args.batch_size, args.seq_len, args.dim, device)
    probe = spiral[:1]
    hierarchical_pages = max(args.pages, 4)
    if hierarchical_pages % 2 != 0:
        hierarchical_pages += 1

    variants = {
        "symplectic_default": dict(gate_kwargs = {}, mem_kwargs = {}),
        "hard_diag": dict(
            gate_kwargs = dict(gated = True, diag = True, gate_mode = "hard"),
            mem_kwargs = {}
        ),
        "soft_diag": dict(
            gate_kwargs = dict(gated = True, diag = True, gate_mode = "soft"),
            mem_kwargs = {}
        ),
        "hard_topk8": dict(
            gate_kwargs = dict(gated = True, diag = True, gate_mode = "hard", top_k = 8),
            mem_kwargs = {}
        ),
        "adaptive_topk": dict(
            gate_kwargs = dict(
                gated = True,
                diag = True,
                gate_mode = "soft",
                adaptive_topk_ratio = 0.6,
                adaptive_topk_min_k = 1
            ),
            mem_kwargs = {}
        ),
        "phase_mix": dict(
            gate_kwargs = dict(
                gated = True,
                diag = True,
                gate_mode = "soft",
                phase_mix = 0.5,
                phase_pairs = max(1, args.dim // 8)
            ),
            mem_kwargs = {}
        ),
        "quorum_budget": dict(
            gate_kwargs = dict(
                gated = True,
                diag = True,
                gate_mode = "soft",
                phase_mix = 0.5,
                phase_pairs = max(1, args.dim // 8),
                quorum_mix = 0.75,
                quorum_window = 5,
                quorum_threshold = 0.25,
                quorum_temperature = 0.1,
                budget_topk_ratio = 0.2
            ),
            mem_kwargs = {}
        ),
        "kinetics_coupled": dict(
            gate_kwargs = dict(
                gated = True,
                diag = True,
                gate_mode = "soft",
                phase_mix = 0.5,
                phase_pairs = max(1, args.dim // 8),
                quorum_mix = 0.55,
                quorum_window = 5,
                quorum_threshold = 0.25,
                quorum_temperature = 0.1
            ),
            mem_kwargs = dict(
                kinetics_coupling = True,
                kinetics_mix = 0.5
            )
        ),
        "hierarchical_route": dict(
            gate_kwargs = dict(
                gated = True,
                diag = True,
                gate_mode = "soft",
                phase_mix = 0.5,
                phase_pairs = max(1, args.dim // 8)
            ),
            mem_kwargs = dict(
                num_pages = hierarchical_pages,
                hierarchical_paging = True,
                coarse_pages = 2,
                fine_pages = hierarchical_pages // 2,
                hierarchy_mix = 1.0
            )
        )
    }

    results = {}
    for name, cfg in variants.items():
        gate_kwargs = cfg["gate_kwargs"]
        mem_kwargs = cfg["mem_kwargs"]
        variant_pages = mem_kwargs.get("num_pages", args.pages)
        model_spiral = NeuralMemory(
            dim = args.dim,
            chunk_size = args.chunk_size,
            use_symplectic_gating = True,
            num_pages = variant_pages,
            symplectic_gate_kwargs = gate_kwargs,
            **{k: v for k, v in mem_kwargs.items() if k != "num_pages"}
        ).to(device)
        spiral_stats = run_one_task(model_spiral, spiral, args.steps)
        gate_stats = summarize_gate(model_spiral, probe)

        model_helix = NeuralMemory(
            dim = args.dim,
            chunk_size = args.chunk_size,
            use_symplectic_gating = True,
            num_pages = variant_pages,
            symplectic_gate_kwargs = gate_kwargs,
            **{k: v for k, v in mem_kwargs.items() if k != "num_pages"}
        ).to(device)
        helix_stats = run_one_task(model_helix, helix, args.steps)

        results[name] = dict(
            spiral = spiral_stats,
            helix = helix_stats,
            gate = gate_stats
        )
        print(
            f"{name:16s} | spiral={spiral_stats['final_loss']:.6f} ({spiral_stats['step_ms']:.2f} ms) "
            f"| helix={helix_stats['final_loss']:.6f} ({helix_stats['step_ms']:.2f} ms) "
            f"| complexity={gate_stats['complexity_mean']:.4f} | quorum={gate_stats['quorum_mean']:.4f} "
            f"| sparse_k={gate_stats['sparse_k_mean']}"
        )

    metrics = dict(
        timestamp_utc = datetime.now(timezone.utc).isoformat(),
        tag = args.tag,
        device = str(device),
        config = dict(
            dim = args.dim,
            seq_len = args.seq_len,
            batch_size = args.batch_size,
            chunk_size = args.chunk_size,
            steps = args.steps,
            pages = args.pages
        ),
        variants = results
    )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents = True, exist_ok = True)
        out_path.write_text(json.dumps(metrics, indent = 2), encoding = "utf-8")
        print(f"Saved JSON: {out_path}")

    if args.output_csv:
        csv_path = Path(args.output_csv)
        csv_path.parent.mkdir(parents = True, exist_ok = True)
        row = flatten_dict(metrics)
        fieldnames = sorted(row.keys())
        write_header = not csv_path.exists()
        with csv_path.open("a", newline = "", encoding = "utf-8") as f:
            writer = csv.DictWriter(f, fieldnames = fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print(f"Appended CSV row: {csv_path}")
