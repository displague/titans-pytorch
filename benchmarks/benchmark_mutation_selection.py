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


def evaluate_candidate(candidate, args, spiral, helix, device):
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
    fitness = mean_loss + args.latency_weight * mean_step_ms

    return dict(
        candidate = candidate,
        gate_kwargs = gate_kwargs,
        mem_kwargs = mem_kwargs,
        spiral = spiral_stats,
        helix = helix_stats,
        aggregate = dict(
            mean_loss = mean_loss,
            mean_step_ms = mean_step_ms,
            mean_switch_events = mean_switch_events,
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
            evaluated.append(evaluate_candidate(candidate, args, spiral, helix, device))

        evaluated.sort(key = lambda e: e["aggregate"]["fitness"])
        elites = evaluated[:max(1, min(args.elites, len(evaluated)))]
        best_gen = elites[0]
        generations.append(dict(generation = gen, best = best_gen))

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
