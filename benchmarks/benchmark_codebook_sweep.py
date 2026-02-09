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


def parse_float_list(text):
    return [float(v.strip()) for v in text.split(",") if v.strip()]


def parse_int_or_none_list(text):
    out = []
    for raw in text.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if token in ("none", "null"):
            out.append(None)
        else:
            out.append(int(token))
    return out


def make_multi_motif_data(batch, seq_len, dim, device):
    t = torch.linspace(0, 8 * 3.14159, seq_len, device = device)
    motif_a0 = torch.cos(t)
    motif_a1 = torch.sin(t)
    motif_b0 = torch.cos(2.7 * t + 0.4)
    motif_b1 = torch.sin(2.7 * t + 0.4)

    segment = max(4, seq_len // 16)
    motif_id = ((torch.arange(seq_len, device = device) // segment) % 2).float()
    mask_a = (motif_id == 0).float()
    mask_b = 1.0 - mask_a

    # Slight motif overlap to make mixture separation nontrivial.
    overlap = 0.15
    x0 = ((1.0 - overlap) * (mask_a * motif_a0 + mask_b * motif_b0)) + (overlap * (mask_a * motif_b0 + mask_b * motif_a0))
    x1 = ((1.0 - overlap) * (mask_a * motif_a1 + mask_b * motif_b1)) + (overlap * (mask_a * motif_b1 + mask_b * motif_a1))

    x = torch.zeros(batch, seq_len, dim, device = device)
    x[:, :, 0] = x0
    x[:, :, 1] = x1

    if dim > 2:
        x[:, :, 2] = mask_a
    if dim > 3:
        x[:, :, 3] = mask_b
    if dim > 4:
        x[:, :, 4:] = 0.05 * torch.randn(batch, seq_len, dim - 4, device = device)

    boundary = torch.zeros(seq_len, device = device, dtype = torch.bool)
    boundary[1:] = motif_id[1:] != motif_id[:-1]
    # Dilate by one step on both sides for a harder boundary metric.
    boundary_left = torch.roll(boundary, shifts = 1)
    boundary_right = torch.roll(boundary, shifts = -1)
    boundary_mask = boundary | boundary_left | boundary_right
    boundary_mask[0] = boundary[0] | boundary[1]
    boundary_mask[-1] = boundary[-1] | boundary[-2]

    return x, boundary_mask


def run_config(model, data, boundary_mask, steps):
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

    with torch.no_grad():
        retrieved, _ = model(data)
        mse_total = F.mse_loss(retrieved, data).item()
        boundary_error = (retrieved[:, boundary_mask] - data[:, boundary_mask]).pow(2).mean().item()

    step_ms = ((end - start) / steps) * 1000
    switch_events = int(model.page_switch_events.item()) if hasattr(model, "page_switch_events") else 0

    return dict(
        train_tail_loss = sum(losses[-10:]) / min(10, len(losses)),
        mse_total = mse_total,
        mse_boundary = boundary_error,
        step_ms = step_ms,
        switch_events = switch_events
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
    parser.add_argument("--tag", type = str, default = "codebook_sweep")
    parser.add_argument("--dim", type = int, default = 128)
    parser.add_argument("--seq-len", type = int, default = 160)
    parser.add_argument("--batch-size", type = int, default = 4)
    parser.add_argument("--chunk-size", type = int, default = 32)
    parser.add_argument("--steps", type = int, default = 10)
    parser.add_argument("--pages", type = int, default = 2)
    parser.add_argument("--mixes", type = str, default = "0.0,0.5,0.7")
    parser.add_argument("--sizes", type = str, default = "8,16,24")
    parser.add_argument("--topks", type = str, default = "none,2,4")
    parser.add_argument("--latency-weight", type = float, default = 1e-3)
    parser.add_argument("--boundary-weight", type = float, default = 0.7)
    parser.add_argument("--output-json", type = str, default = None)
    parser.add_argument("--output-csv", type = str, default = None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    codebook_mixes = parse_float_list(args.mixes)
    codebook_sizes = [int(v) for v in parse_float_list(args.sizes)]
    codebook_topks = parse_int_or_none_list(args.topks)
    boundary_weight = args.boundary_weight
    total_weight = 1.0 - boundary_weight

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Sweep mixes={codebook_mixes} sizes={codebook_sizes} topks={codebook_topks}")

    data, boundary_mask = make_multi_motif_data(args.batch_size, args.seq_len, args.dim, device)

    trials = []
    for mix in codebook_mixes:
        for size in codebook_sizes:
            for topk in codebook_topks:
                if topk is not None and topk > size:
                    continue

                gate_kwargs = dict(
                    gated = True,
                    diag = True,
                    gate_mode = "soft",
                    phase_mix = 0.5,
                    phase_pairs = max(1, args.dim // 8),
                    quorum_mix = 0.55,
                    quorum_window = 5,
                    quorum_threshold = 0.25,
                    quorum_temperature = 0.1,
                    codebook_mix = mix,
                    codebook_size = size,
                    codebook_temperature = 0.1,
                    codebook_topk = topk
                )

                model = NeuralMemory(
                    dim = args.dim,
                    chunk_size = args.chunk_size,
                    use_symplectic_gating = True,
                    num_pages = args.pages,
                    manifold_state_keyed_paging = True,
                    symplectic_gate_kwargs = gate_kwargs
                ).to(device)

                stats = run_config(model, data, boundary_mask, args.steps)
                score = (
                    boundary_weight * stats["mse_boundary"]
                    + total_weight * stats["mse_total"]
                    + args.latency_weight * stats["step_ms"]
                )

                trial = dict(
                    params = dict(
                        codebook_mix = mix,
                        codebook_size = size,
                        codebook_topk = topk
                    ),
                    stats = stats,
                    score = score
                )
                trials.append(trial)
                print(
                    f"mix={mix:.2f} size={size:2d} topk={str(topk):>4s} | "
                    f"boundary={stats['mse_boundary']:.6f} total={stats['mse_total']:.6f} "
                    f"step={stats['step_ms']:.2f}ms score={score:.6f}"
                )

    trials.sort(key = lambda t: t["score"])
    best = trials[0]
    top5 = trials[:5]

    print("-" * 90)
    print(
        f"BEST | mix={best['params']['codebook_mix']:.2f} size={best['params']['codebook_size']} "
        f"topk={best['params']['codebook_topk']} | "
        f"boundary={best['stats']['mse_boundary']:.6f} total={best['stats']['mse_total']:.6f} "
        f"step={best['stats']['step_ms']:.2f}ms score={best['score']:.6f}"
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
            pages = args.pages,
            mixes = codebook_mixes,
            sizes = codebook_sizes,
            topks = codebook_topks,
            boundary_weight = boundary_weight,
            latency_weight = args.latency_weight
        ),
        best = best,
        top5 = top5,
        trials = trials
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
