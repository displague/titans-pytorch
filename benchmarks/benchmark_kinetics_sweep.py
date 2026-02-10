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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type = str, default = "kinetics_sweep")
    parser.add_argument("--dim", type = int, default = 128)
    parser.add_argument("--seq-len", type = int, default = 160)
    parser.add_argument("--batch-size", type = int, default = 4)
    parser.add_argument("--chunk-size", type = int, default = 32)
    parser.add_argument("--pages", type = int, default = 2)
    parser.add_argument("--interference-steps", type = int, default = 12)
    parser.add_argument("--interference-eval-iters", type = int, default = 8)
    parser.add_argument("--long-seq-len", type = int, default = 512)
    parser.add_argument("--long-warmup-steps", type = int, default = 12)
    parser.add_argument("--long-perturb-steps", type = int, default = 8)
    parser.add_argument("--long-recovery-steps", type = int, default = 8)
    parser.add_argument("--latency-weight", type = float, default = 1e-3)
    parser.add_argument("--mixes", type = str, default = "0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--output-json", type = str, default = None)
    parser.add_argument("--output-csv", type = str, default = None)
    return parser.parse_args()


def flatten_dict(d, prefix = ""):
    flat = {}
    for key, value in d.items():
        key_name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, key_name))
        else:
            flat[key_name] = value
    return flat


def make_long_horizon_pair(batch, seq_len, dim, device):
    t = torch.linspace(0, 12 * 3.14159, seq_len, device = device)
    x_clean = torch.zeros(batch, seq_len, dim, device = device)
    x_clean[:, :, 0] = torch.cos(t)
    x_clean[:, :, 1] = torch.sin(t)
    if dim > 2:
        x_clean[:, :, 2] = t / t.max()
    if dim > 3:
        x_clean[:, :, 3:] = 0.05 * torch.randn(batch, seq_len, dim - 3, device = device)

    x_perturb = x_clean.clone()
    start = seq_len // 3
    end = (2 * seq_len) // 3
    phase_jump = 0.8
    t_mid = t[start:end] + phase_jump
    x_perturb[:, start:end, 0] = 1.2 * torch.cos(t_mid)
    x_perturb[:, start:end, 1] = 1.2 * torch.sin(t_mid)
    if dim > 2:
        x_perturb[:, start:end, 2] = (t_mid / t.max()).clamp(max = 1.0)
    return x_clean, x_perturb


def make_interference_pair(batch, seq_len, dim, device):
    t = torch.linspace(0, 6 * 3.14159, seq_len, device = device)
    x_base = torch.zeros(batch, seq_len, dim, device = device)
    x_base[:, :, 0] = torch.cos(t)
    x_base[:, :, 1] = torch.sin(t)
    if dim > 2:
        x_base[:, :, 2] = torch.sin(0.5 * t)
    if dim > 3:
        x_base[:, :, 3:] = 0.04 * torch.randn(batch, seq_len, dim - 3, device = device)

    v = torch.randn(1, 1, dim, device = device)
    v = F.normalize(v, dim = -1) * 2.0
    footprint = (torch.arange(seq_len, device = device) % 8) == 0

    x_a = x_base.clone()
    x_b = x_base.clone()
    x_a[:, footprint, :] += v
    x_b[:, footprint, :] -= v
    return x_a, x_b


def train_reconstruction(model, optimizer, data, steps):
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
        tail_loss = sum(losses[-10:]) / min(10, len(losses)),
        step_ms = ((end - start) / steps) * 1000
    )


def eval_reconstruction(model, data, eval_iters):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_iters):
            retrieved, _ = model(data)
            losses.append(F.mse_loss(retrieved, data).item())
    return sum(losses) / len(losses)


def benchmark_long_horizon(model, x_clean, x_perturb, warmup_steps, perturb_steps, recovery_steps):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    clean_angle = torch.atan2(x_clean[:, :, 1], x_clean[:, :, 0])
    total_steps = warmup_steps + perturb_steps + recovery_steps

    sync(x_clean.device)
    start = time.perf_counter()

    for _ in range(warmup_steps):
        optimizer.zero_grad()
        retrieved, _ = model(x_clean)
        loss = F.mse_loss(retrieved, x_clean)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        retrieved_clean, _ = model(x_clean)
        clean_mse_pre = F.mse_loss(retrieved_clean, x_clean).item()
        pred_angle = torch.atan2(retrieved_clean[:, :, 1], retrieved_clean[:, :, 0])
        wrapped = torch.atan2(torch.sin(pred_angle - clean_angle), torch.cos(pred_angle - clean_angle))
        phase_err_pre = wrapped.abs().mean().item()

    for _ in range(perturb_steps):
        optimizer.zero_grad()
        retrieved, _ = model(x_perturb)
        loss = F.mse_loss(retrieved, x_perturb)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        retrieved_perturb, _ = model(x_perturb)
        perturb_mse = F.mse_loss(retrieved_perturb, x_perturb).item()

    for _ in range(recovery_steps):
        optimizer.zero_grad()
        retrieved, _ = model(x_clean)
        loss = F.mse_loss(retrieved, x_clean)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        retrieved_clean_after, _ = model(x_clean)
        clean_mse_post = F.mse_loss(retrieved_clean_after, x_clean).item()
        pred_angle_post = torch.atan2(retrieved_clean_after[:, :, 1], retrieved_clean_after[:, :, 0])
        wrapped_post = torch.atan2(torch.sin(pred_angle_post - clean_angle), torch.cos(pred_angle_post - clean_angle))
        phase_err_post = wrapped_post.abs().mean().item()

    sync(x_clean.device)
    end = time.perf_counter()

    return dict(
        clean_mse_pre = clean_mse_pre,
        perturb_mse = perturb_mse,
        clean_mse_post = clean_mse_post,
        phase_err_pre = phase_err_pre,
        phase_err_post = phase_err_post,
        mse_recovery_gain = clean_mse_pre - clean_mse_post,
        phase_recovery_gain = phase_err_pre - phase_err_post,
        step_ms = ((end - start) / total_steps) * 1000
    )


def benchmark_interference(model, x_a, x_b, train_steps, eval_iters):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    train_a = train_reconstruction(model, optimizer, x_a, train_steps)
    pre_a_loss = eval_reconstruction(model, x_a, eval_iters)

    train_b = train_reconstruction(model, optimizer, x_b, train_steps)
    post_a_loss = eval_reconstruction(model, x_a, eval_iters)
    post_b_loss = eval_reconstruction(model, x_b, eval_iters)

    return dict(
        pre_a_loss = pre_a_loss,
        post_a_loss = post_a_loss,
        post_b_loss = post_b_loss,
        forgetting_delta = post_a_loss - pre_a_loss,
        train_a_tail = train_a["tail_loss"],
        train_b_tail = train_b["tail_loss"],
        step_ms = 0.5 * (train_a["step_ms"] + train_b["step_ms"])
    )


def transfer_score(variant_metrics, latency_weight):
    long_horizon = variant_metrics["long_horizon"]
    interference = variant_metrics["interference"]
    latency = long_horizon["step_ms"] + interference["step_ms"]
    return (
        long_horizon["clean_mse_post"]
        + 0.5 * long_horizon["phase_err_post"]
        + interference["post_a_loss"]
        + latency_weight * latency
    )


def build_models(args, kinetics_mix, device):
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
    )

    mem_kwargs = dict(
        dim = args.dim,
        chunk_size = args.chunk_size,
        use_symplectic_gating = True,
        num_pages = args.pages,
        symplectic_gate_kwargs = gate_kwargs,
        kinetics_coupling = True,
        kinetics_mix = kinetics_mix
    )

    model_long = NeuralMemory(**mem_kwargs).to(device)
    model_interference = NeuralMemory(**mem_kwargs).to(device)
    return model_long, model_interference


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    mixes = parse_float_list(args.mixes)
    x_clean, x_perturb = make_long_horizon_pair(args.batch_size, args.long_seq_len, args.dim, device)
    x_a, x_b = make_interference_pair(args.batch_size, args.seq_len, args.dim, device)

    trials = []
    for mix in mixes:
        model_long, model_interference = build_models(args, mix, device)
        long_stats = benchmark_long_horizon(
            model = model_long,
            x_clean = x_clean,
            x_perturb = x_perturb,
            warmup_steps = args.long_warmup_steps,
            perturb_steps = args.long_perturb_steps,
            recovery_steps = args.long_recovery_steps
        )
        interference_stats = benchmark_interference(
            model = model_interference,
            x_a = x_a,
            x_b = x_b,
            train_steps = args.interference_steps,
            eval_iters = args.interference_eval_iters
        )

        trial = dict(
            params = dict(kinetics_mix = mix),
            long_horizon = long_stats,
            interference = interference_stats
        )
        trial["transfer_score"] = transfer_score(trial, args.latency_weight)
        trials.append(trial)

        print(
            f"mix={mix:.2f} | "
            f"long.clean_post={long_stats['clean_mse_post']:.6f} "
            f"long.phase_post={long_stats['phase_err_post']:.6f} "
            f"interf.postA={interference_stats['post_a_loss']:.6f} "
            f"score={trial['transfer_score']:.6f}"
        )

    trials.sort(key = lambda t: t["transfer_score"])
    best = trials[0]

    print("-" * 96)
    print(
        f"BEST | mix={best['params']['kinetics_mix']:.2f} "
        f"score={best['transfer_score']:.6f}"
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
            interference_steps = args.interference_steps,
            interference_eval_iters = args.interference_eval_iters,
            long_seq_len = args.long_seq_len,
            long_warmup_steps = args.long_warmup_steps,
            long_perturb_steps = args.long_perturb_steps,
            long_recovery_steps = args.long_recovery_steps,
            latency_weight = args.latency_weight,
            mixes = mixes
        ),
        best = best,
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
