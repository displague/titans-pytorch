import argparse
import csv
import json
import math
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


def normalized_page_entropy(active_page_indices, num_pages, eps = 1e-8):
    if num_pages <= 1:
        return torch.tensor(0.0, device = active_page_indices.device)

    probs = torch.bincount(active_page_indices, minlength = num_pages).float()
    probs = probs / probs.sum().clamp(min = eps)
    entropy = -(probs * (probs + eps).log()).sum()
    return entropy / math.log(num_pages)


def train_with_budget_control(
    model,
    data,
    steps,
    switch_budget_target = None,
    switch_budget_weight = 0.0,
    entropy_target = None,
    entropy_weight = 0.0,
    carry_state = False
):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    recon_losses = []
    total_losses = []
    penalties = []
    switch_rates = []
    entropies = []

    state = None
    prev_switch = int(model.page_switch_events.item()) if hasattr(model, "page_switch_events") else 0

    sync(data.device)
    start = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad()
        if carry_state:
            retrieved, state = model(data, state = state)
        else:
            retrieved, state = model(data)

        recon = F.mse_loss(retrieved, data)
        penalty = torch.zeros((), device = data.device)

        step_switch_rate = 0.0
        if hasattr(model, "page_switch_events"):
            curr_switch = int(model.page_switch_events.item())
            delta = curr_switch - prev_switch
            prev_switch = curr_switch
            step_switch_rate = float(delta) / max(1, data.shape[0])

            if switch_budget_target is not None and switch_budget_weight > 0.0:
                over = max(0.0, step_switch_rate - switch_budget_target)
                penalty = penalty + switch_budget_weight * (over ** 2)

        entropy_value = 0.0
        if (
            entropy_target is not None and
            entropy_weight > 0.0 and
            state is not None and
            state.active_page_indices is not None and
            model.num_pages > 1
        ):
            entropy = normalized_page_entropy(state.active_page_indices, model.num_pages)
            entropy_value = entropy.item()
            penalty = penalty + entropy_weight * (entropy - entropy_target).pow(2)

        loss = recon + penalty
        loss.backward()
        optimizer.step()

        recon_losses.append(recon.item())
        total_losses.append(loss.item())
        penalties.append(penalty.item())
        switch_rates.append(step_switch_rate)
        entropies.append(entropy_value)
    sync(data.device)
    end = time.perf_counter()

    return dict(
        recon_loss = sum(recon_losses[-10:]) / min(10, len(recon_losses)),
        total_loss = sum(total_losses[-10:]) / min(10, len(total_losses)),
        penalty = sum(penalties[-10:]) / min(10, len(penalties)),
        switch_rate = sum(switch_rates) / max(1, len(switch_rates)),
        entropy = sum(entropies) / max(1, len(entropies)),
        step_ms = ((end - start) / steps) * 1000
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
    parser.add_argument("--tag", type = str, default = "switch_budget")
    parser.add_argument("--dim", type = int, default = 128)
    parser.add_argument("--seq-len", type = int, default = 128)
    parser.add_argument("--batch-size", type = int, default = 4)
    parser.add_argument("--chunk-size", type = int, default = 32)
    parser.add_argument("--steps", type = int, default = 20)
    parser.add_argument("--pages", type = int, default = 4)
    parser.add_argument("--page-threshold", type = float, default = 0.1)
    parser.add_argument("--switch-target", type = float, default = 0.5)
    parser.add_argument("--switch-weight", type = float, default = 0.25)
    parser.add_argument("--entropy-target", type = float, default = 0.5)
    parser.add_argument("--entropy-weight", type = float, default = 0.1)
    parser.add_argument("--carry-state", action = "store_true", default = False)
    parser.add_argument("--output-json", type = str, default = None)
    parser.add_argument("--output-csv", type = str, default = None)
    return parser.parse_args()


def make_model(dim, chunk_size, pages, page_threshold, device):
    model = NeuralMemory(
        dim = dim,
        chunk_size = chunk_size,
        use_symplectic_gating = True,
        num_pages = pages,
        manifold_state_keyed_paging = True,
        symplectic_page_threshold = page_threshold,
        symplectic_gate_kwargs = dict(
            gated = True,
            diag = True,
            gate_mode = "soft",
            phase_mix = 0.5,
            phase_pairs = max(1, dim // 8),
            quorum_mix = 0.75,
            quorum_window = 5,
            quorum_threshold = 0.25,
            quorum_temperature = 0.1,
            budget_topk_ratio = 0.2
        )
    )
    return model.to(device)


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    spiral = make_spiral_data(args.batch_size, args.seq_len, args.dim, device)
    helix = make_helix_data(args.batch_size, args.seq_len, args.dim, device)

    tasks = dict(spiral = spiral, helix = helix)
    results = {}

    for task_name, task_data in tasks.items():
        unconstrained = make_model(args.dim, args.chunk_size, args.pages, args.page_threshold, device)
        constrained = make_model(args.dim, args.chunk_size, args.pages, args.page_threshold, device)

        unconstrained_stats = train_with_budget_control(
            unconstrained,
            task_data,
            args.steps,
            switch_budget_target = None,
            switch_budget_weight = 0.0,
            entropy_target = None,
            entropy_weight = 0.0,
            carry_state = args.carry_state
        )
        constrained_stats = train_with_budget_control(
            constrained,
            task_data,
            args.steps,
            switch_budget_target = args.switch_target,
            switch_budget_weight = args.switch_weight,
            entropy_target = args.entropy_target,
            entropy_weight = args.entropy_weight,
            carry_state = args.carry_state
        )

        results[task_name] = dict(
            unconstrained = unconstrained_stats,
            constrained = constrained_stats
        )

        print(
            f"{task_name:6s} | "
            f"unconstrained recon={unconstrained_stats['recon_loss']:.6f} switch={unconstrained_stats['switch_rate']:.3f} "
            f"entropy={unconstrained_stats['entropy']:.3f} | "
            f"constrained recon={constrained_stats['recon_loss']:.6f} switch={constrained_stats['switch_rate']:.3f} "
            f"entropy={constrained_stats['entropy']:.3f}"
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
            page_threshold = args.page_threshold,
            switch_target = args.switch_target,
            switch_weight = args.switch_weight,
            entropy_target = args.entropy_target,
            entropy_weight = args.entropy_weight,
            carry_state = args.carry_state
        ),
        tasks = results
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
