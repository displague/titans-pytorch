import argparse
import csv
import json
import random
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
    return dict(
        final_loss = sum(losses[-10:]) / min(10, len(losses)),
        step_ms = ((end - start) / steps) * 1000,
        switch_events = switch_events
    )


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
        step_ms = ((end - start) / total_steps) * 1000
    )


def benchmark_interference(model, x_a, x_b, train_steps, eval_iters):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    train_reconstruction(model, optimizer, x_a, train_steps)
    pre_a_loss = eval_reconstruction(model, x_a, eval_iters)

    train_reconstruction(model, optimizer, x_b, train_steps)
    post_a_loss = eval_reconstruction(model, x_a, eval_iters)
    post_b_loss = eval_reconstruction(model, x_b, eval_iters)

    return dict(
        pre_a_loss = pre_a_loss,
        post_a_loss = post_a_loss,
        post_b_loss = post_b_loss,
        forgetting_delta = post_a_loss - pre_a_loss
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type = str, default = "mutation_selection")
    parser.add_argument("--dim", type = int, default = 128)
    parser.add_argument("--seq-len", type = int, default = 128)
    parser.add_argument("--batch-size", type = int, default = 4)
    parser.add_argument("--chunk-size", type = int, default = 32)
    parser.add_argument("--pages", type = int, default = 2)
    parser.add_argument("--hier-pages", type = int, default = 4)
    parser.add_argument("--steps", type = int, default = 12)
    parser.add_argument("--population", type = int, default = 8)
    parser.add_argument("--generations", type = int, default = 4)
    parser.add_argument("--elites", type = int, default = 3)
    parser.add_argument("--mutation-prob", type = float, default = 0.35)
    parser.add_argument("--latency-weight", type = float, default = 1e-3)
    parser.add_argument("--use-transfer-fitness", action = "store_true", default = False)
    parser.add_argument(
        "--transfer-weight",
        type = float,
        default = 0.5,
        help = "Weight of transfer fitness in final score when --use-transfer-fitness is enabled."
    )
    parser.add_argument("--long-seq-len", type = int, default = 256)
    parser.add_argument("--long-warmup-steps", type = int, default = 6)
    parser.add_argument("--long-perturb-steps", type = int, default = 4)
    parser.add_argument("--long-recovery-steps", type = int, default = 4)
    parser.add_argument("--interference-steps", type = int, default = 6)
    parser.add_argument("--interference-eval-iters", type = int, default = 4)
    parser.add_argument("--seed", type = int, default = 0)
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


def sample_candidate(rng):
    phase_mix_values = [0.0, 0.25, 0.5, 0.75]
    quorum_mix_values = [0.0, 0.35, 0.55, 0.75]
    budget_values = [None, 0.15, 0.2, 0.25, 0.3]
    hierarchy_values = [False, True]
    hierarchy_mix_values = [0.5, 0.75, 1.0]

    cand = dict(
        phase_mix = rng.choice(phase_mix_values),
        quorum_mix = rng.choice(quorum_mix_values),
        budget_topk_ratio = rng.choice(budget_values),
        hierarchical = rng.choice(hierarchy_values),
        hierarchy_mix = rng.choice(hierarchy_mix_values),
        quorum_window = rng.choice([3, 5, 7]),
        quorum_threshold = rng.choice([0.2, 0.25, 0.3]),
        quorum_temperature = rng.choice([0.08, 0.1, 0.15])
    )
    if cand["quorum_mix"] == 0.0:
        cand["budget_topk_ratio"] = None
    return cand


def mutate_candidate(parent, rng, mutation_prob):
    child = dict(parent)

    def maybe(values, key):
        if rng.random() < mutation_prob:
            child[key] = rng.choice(values)

    maybe([0.0, 0.25, 0.5, 0.75], "phase_mix")
    maybe([0.0, 0.35, 0.55, 0.75], "quorum_mix")
    maybe([None, 0.15, 0.2, 0.25, 0.3], "budget_topk_ratio")
    maybe([False, True], "hierarchical")
    maybe([0.5, 0.75, 1.0], "hierarchy_mix")
    maybe([3, 5, 7], "quorum_window")
    maybe([0.2, 0.25, 0.3], "quorum_threshold")
    maybe([0.08, 0.1, 0.15], "quorum_temperature")

    if child["quorum_mix"] == 0.0:
        child["budget_topk_ratio"] = None
    return child


def candidate_to_kwargs(candidate, args):
    gate_kwargs = dict(
        gated = True,
        diag = True,
        gate_mode = "soft",
        phase_mix = candidate["phase_mix"],
        phase_pairs = max(1, args.dim // 8),
        quorum_mix = candidate["quorum_mix"],
        quorum_window = candidate["quorum_window"],
        quorum_threshold = candidate["quorum_threshold"],
        quorum_temperature = candidate["quorum_temperature"]
    )
    if candidate["budget_topk_ratio"] is not None:
        gate_kwargs["budget_topk_ratio"] = candidate["budget_topk_ratio"]

    mem_kwargs = dict(num_pages = args.pages)
    if candidate["hierarchical"]:
        hier_pages = max(2, args.hier_pages)
        if hier_pages % 2 != 0:
            hier_pages += 1
        mem_kwargs = dict(
            num_pages = hier_pages,
            hierarchical_paging = True,
            coarse_pages = 2,
            fine_pages = hier_pages // 2,
            hierarchy_mix = candidate["hierarchy_mix"]
        )
    return gate_kwargs, mem_kwargs


def evaluate_candidate(candidate, args, spiral, helix, transfer_data, device):
    gate_kwargs, mem_kwargs = candidate_to_kwargs(candidate, args)

    model_spiral = NeuralMemory(
        dim = args.dim,
        chunk_size = args.chunk_size,
        use_symplectic_gating = True,
        symplectic_gate_kwargs = gate_kwargs,
        **mem_kwargs
    ).to(device)
    spiral_stats = run_task(model_spiral, spiral, args.steps)

    model_helix = NeuralMemory(
        dim = args.dim,
        chunk_size = args.chunk_size,
        use_symplectic_gating = True,
        symplectic_gate_kwargs = gate_kwargs,
        **mem_kwargs
    ).to(device)
    helix_stats = run_task(model_helix, helix, args.steps)

    mean_loss = 0.5 * (spiral_stats["final_loss"] + helix_stats["final_loss"])
    mean_step_ms = 0.5 * (spiral_stats["step_ms"] + helix_stats["step_ms"])
    mean_switch_events = 0.5 * (spiral_stats["switch_events"] + helix_stats["switch_events"])
    classic_fitness = mean_loss + args.latency_weight * mean_step_ms

    transfer = None
    transfer_fitness = classic_fitness
    if args.use_transfer_fitness:
        x_clean, x_perturb, x_a, x_b = transfer_data

        model_long = NeuralMemory(
            dim = args.dim,
            chunk_size = args.chunk_size,
            use_symplectic_gating = True,
            symplectic_gate_kwargs = gate_kwargs,
            **mem_kwargs
        ).to(device)
        long_stats = benchmark_long_horizon(
            model = model_long,
            x_clean = x_clean,
            x_perturb = x_perturb,
            warmup_steps = args.long_warmup_steps,
            perturb_steps = args.long_perturb_steps,
            recovery_steps = args.long_recovery_steps
        )

        model_interference = NeuralMemory(
            dim = args.dim,
            chunk_size = args.chunk_size,
            use_symplectic_gating = True,
            symplectic_gate_kwargs = gate_kwargs,
            **mem_kwargs
        ).to(device)
        interference_stats = benchmark_interference(
            model = model_interference,
            x_a = x_a,
            x_b = x_b,
            train_steps = args.interference_steps,
            eval_iters = args.interference_eval_iters
        )

        transfer_loss = (
            long_stats["clean_mse_post"]
            + 0.5 * long_stats["phase_err_post"]
            + interference_stats["post_a_loss"]
        )
        transfer_step_ms = long_stats["step_ms"]
        transfer_fitness = transfer_loss + args.latency_weight * transfer_step_ms
        transfer = dict(
            long_horizon = long_stats,
            interference = interference_stats,
            aggregate = dict(
                transfer_loss = transfer_loss,
                transfer_step_ms = transfer_step_ms,
                transfer_fitness = transfer_fitness
            )
        )

    if args.use_transfer_fitness:
        w = min(1.0, max(0.0, args.transfer_weight))
        fitness = (1.0 - w) * classic_fitness + w * transfer_fitness
    else:
        fitness = classic_fitness

    return dict(
        candidate = candidate,
        gate_kwargs = gate_kwargs,
        mem_kwargs = mem_kwargs,
        spiral = spiral_stats,
        helix = helix_stats,
        transfer = transfer,
        aggregate = dict(
            mean_loss = mean_loss,
            mean_step_ms = mean_step_ms,
            mean_switch_events = mean_switch_events,
            classic_fitness = classic_fitness,
            transfer_fitness = transfer_fitness if args.use_transfer_fitness else None,
            fitness = fitness
        )
    )


if __name__ == "__main__":
    args = parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    spiral = make_spiral_data(args.batch_size, args.seq_len, args.dim, device)
    helix = make_helix_data(args.batch_size, args.seq_len, args.dim, device)
    transfer_data = None
    if args.use_transfer_fitness:
        x_clean, x_perturb = make_long_horizon_pair(args.batch_size, args.long_seq_len, args.dim, device)
        x_a, x_b = make_interference_pair(args.batch_size, args.seq_len, args.dim, device)
        transfer_data = (x_clean, x_perturb, x_a, x_b)

    # Seed population with a strong known baseline candidate.
    population = [
        dict(
            phase_mix = 0.5,
            quorum_mix = 0.75,
            budget_topk_ratio = 0.2,
            hierarchical = True,
            hierarchy_mix = 1.0,
            quorum_window = 5,
            quorum_threshold = 0.25,
            quorum_temperature = 0.1
        )
    ]
    while len(population) < args.population:
        population.append(sample_candidate(rng))

    generations = []
    best_overall = None

    for gen in range(args.generations):
        evaluated = []
        for candidate in population:
            evaluated.append(evaluate_candidate(candidate, args, spiral, helix, transfer_data, device))

        evaluated.sort(key = lambda e: e["aggregate"]["fitness"])
        elites = evaluated[:max(1, min(args.elites, len(evaluated)))]
        best_gen = elites[0]
        generations.append(dict(generation = gen, best = best_gen))

        if args.use_transfer_fitness:
            print(
                f"gen={gen:02d} | "
                f"fitness={best_gen['aggregate']['fitness']:.6f} | "
                f"classic={best_gen['aggregate']['classic_fitness']:.6f} | "
                f"transfer={best_gen['aggregate']['transfer_fitness']:.6f} | "
                f"loss={best_gen['aggregate']['mean_loss']:.6f} | "
                f"cand={best_gen['candidate']}"
            )
        else:
            print(
                f"gen={gen:02d} | "
                f"fitness={best_gen['aggregate']['fitness']:.6f} | "
                f"loss={best_gen['aggregate']['mean_loss']:.6f} | "
                f"step_ms={best_gen['aggregate']['mean_step_ms']:.2f} | "
                f"switch={best_gen['aggregate']['mean_switch_events']:.2f} | "
                f"cand={best_gen['candidate']}"
            )

        if best_overall is None or best_gen["aggregate"]["fitness"] < best_overall["aggregate"]["fitness"]:
            best_overall = best_gen

        next_population = [dict(e["candidate"]) for e in elites]
        while len(next_population) < args.population:
            parent = rng.choice(elites)["candidate"]
            next_population.append(mutate_candidate(parent, rng, args.mutation_prob))
        population = next_population

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
            hier_pages = args.hier_pages,
            steps = args.steps,
            population = args.population,
            generations = args.generations,
            elites = args.elites,
            mutation_prob = args.mutation_prob,
            latency_weight = args.latency_weight,
            use_transfer_fitness = args.use_transfer_fitness,
            transfer_weight = args.transfer_weight,
            long_seq_len = args.long_seq_len,
            long_warmup_steps = args.long_warmup_steps,
            long_perturb_steps = args.long_perturb_steps,
            long_recovery_steps = args.long_recovery_steps,
            interference_steps = args.interference_steps,
            interference_eval_iters = args.interference_eval_iters,
            seed = args.seed
        ),
        best = best_overall,
        generation_history = generations
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
