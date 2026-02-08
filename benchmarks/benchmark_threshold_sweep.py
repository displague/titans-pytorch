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


def run_task(model, data, steps):
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

    switch_events = int(model.page_switch_events.item()) if hasattr(model, "page_switch_events") else 0
    switch_rate = switch_events / max(steps, 1)

    return dict(
        final_loss = sum(losses[-10:]) / min(10, len(losses)),
        step_ms = ((end - start) / steps) * 1000,
        switch_events = switch_events,
        switch_rate = switch_rate
    )


def parse_thresholds(text):
    values = [v.strip() for v in text.split(",") if v.strip()]
    thresholds = [float(v) for v in values]
    thresholds = sorted(set(thresholds))
    return thresholds


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
    parser.add_argument("--tag", type = str, default = "threshold_sweep")
    parser.add_argument("--dim", type = int, default = 128)
    parser.add_argument("--seq-len", type = int, default = 128)
    parser.add_argument("--batch-size", type = int, default = 4)
    parser.add_argument("--chunk-size", type = int, default = 32)
    parser.add_argument("--pages", type = int, default = 2)
    parser.add_argument("--steps", type = int, default = 30)
    parser.add_argument("--thresholds", type = str, default = "0.05,0.1,0.2,0.35,0.5,0.7")
    parser.add_argument("--output-json", type = str, default = None)
    parser.add_argument("--output-csv", type = str, default = None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thresholds = parse_thresholds(args.thresholds)

    print(f"Using device: {device}")
    print(f"Threshold sweep: {thresholds}")

    spiral = make_spiral_data(args.batch_size, args.seq_len, args.dim, device)
    helix = make_helix_data(args.batch_size, args.seq_len, args.dim, device)

    sweep = {}
    for threshold in thresholds:
        model_spiral = NeuralMemory(
            dim = args.dim,
            chunk_size = args.chunk_size,
            use_symplectic_gating = True,
            num_pages = args.pages,
            symplectic_page_threshold = threshold
        ).to(device)
        spiral_stats = run_task(model_spiral, spiral, args.steps)

        model_helix = NeuralMemory(
            dim = args.dim,
            chunk_size = args.chunk_size,
            use_symplectic_gating = True,
            num_pages = args.pages,
            symplectic_page_threshold = threshold
        ).to(device)
        helix_stats = run_task(model_helix, helix, args.steps)

        mean_loss = 0.5 * (spiral_stats["final_loss"] + helix_stats["final_loss"])
        mean_switch_rate = 0.5 * (spiral_stats["switch_rate"] + helix_stats["switch_rate"])

        sweep[f"{threshold:.4f}"] = dict(
            spiral = spiral_stats,
            helix = helix_stats,
            aggregate = dict(
                mean_loss = mean_loss,
                mean_switch_rate = mean_switch_rate
            )
        )

        print(
            f"thr={threshold:.4f} | "
            f"spiral={spiral_stats['final_loss']:.6f}, helix={helix_stats['final_loss']:.6f} | "
            f"switch_rate={mean_switch_rate:.3f}"
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
            pages = args.pages,
            steps = args.steps,
            thresholds = thresholds
        ),
        threshold_sweep = sweep
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
