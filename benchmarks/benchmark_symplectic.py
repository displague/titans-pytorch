
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

def benchmark_model(name, model, x, steps=5):
    model.eval()

    # Warmup
    for _ in range(5):
        _ = model(x)

    sync(x.device)
    start_time = time.perf_counter()
    for _ in range(steps):
        _ = model(x)
    sync(x.device)
    end_time = time.perf_counter()

    avg_time = (end_time - start_time) / steps
    print(f"{name}: Avg Forward Time = {avg_time*1000:.2f} ms")
    return avg_time

def benchmark_training(name, model, x, steps=5):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        retrieved, _ = model(x)
        loss = F.mse_loss(retrieved, x)
        loss.backward()
        optimizer.step()

    sync(x.device)
    start_time = time.perf_counter()
    for _ in range(steps):
        optimizer.zero_grad()
        retrieved, _ = model(x)
        loss = F.mse_loss(retrieved, x)
        loss.backward()
        optimizer.step()
    sync(x.device)
    end_time = time.perf_counter()

    avg_time = (end_time - start_time) / steps
    print(f"{name}: Avg Train Step Time = {avg_time*1000:.2f} ms")
    return avg_time

def benchmark_spiral_recall(name, model, device, seq_len=128, dim=128, steps=50):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    t = torch.linspace(0, 4 * 3.14159, seq_len, device=device)
    x = torch.zeros(4, seq_len, dim, device=device)
    x[:, :, 0] = torch.cos(t)
    x[:, :, 1] = torch.sin(t)
    if dim > 2:
        x[:, :, 2] = t / t.max()
    if dim > 3:
        x[:, :, 3:] = 0.1 * torch.randn(4, seq_len, dim - 3, device=device)

    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        retrieved, _ = model(x)
        loss = F.mse_loss(retrieved, x)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    final_loss = sum(losses[-10:]) / 10
    print(f"{name}: Spiral Reconstruction Loss = {final_loss:.6f}")
    return final_loss

def benchmark_helix_drift_recall(name, model, device, seq_len=128, dim=128, steps=50):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    t = torch.linspace(0, 6 * 3.14159, seq_len, device=device)
    x = torch.zeros(4, seq_len, dim, device=device)
    x[:, :, 0] = torch.cos(t)
    x[:, :, 1] = torch.sin(t)
    if dim > 2:
        x[:, :, 2] = t / t.max()
    if dim > 3:
        drift = torch.linspace(0, 1.0, seq_len, device=device)
        x[:, :, 3] = drift
    if dim > 4:
        x[:, :, 4:] = 0.1 * torch.randn(4, seq_len, dim - 4, device=device)

    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        retrieved, _ = model(x)
        loss = F.mse_loss(retrieved, x)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    final_loss = sum(losses[-10:]) / 10
    print(f"{name}: Helix+Drift Reconstruction Loss = {final_loss:.6f}")
    return final_loss

def benchmark_long_horizon_recovery(
    name,
    model,
    device,
    seq_len = 512,
    dim = 128,
    warmup_steps = 30,
    perturb_steps = 20,
    recovery_steps = 20
):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    t = torch.linspace(0, 12 * 3.14159, seq_len, device = device)
    x_clean = torch.zeros(4, seq_len, dim, device = device)
    x_clean[:, :, 0] = torch.cos(t)
    x_clean[:, :, 1] = torch.sin(t)
    if dim > 2:
        x_clean[:, :, 2] = t / t.max()
    if dim > 3:
        x_clean[:, :, 3:] = 0.05 * torch.randn(4, seq_len, dim - 3, device = device)

    x_perturb = x_clean.clone()
    start = seq_len // 3
    end = (2 * seq_len) // 3
    phase_jump = 0.8
    t_mid = t[start:end] + phase_jump
    x_perturb[:, start:end, 0] = 1.2 * torch.cos(t_mid)
    x_perturb[:, start:end, 1] = 1.2 * torch.sin(t_mid)
    if dim > 2:
        x_perturb[:, start:end, 2] = (t_mid / t.max()).clamp(max = 1.0)

    for _ in range(warmup_steps):
        optimizer.zero_grad()
        retrieved, _ = model(x_clean)
        loss = F.mse_loss(retrieved, x_clean)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        retrieved_clean, _ = model(x_clean)
        clean_mse_pre = F.mse_loss(retrieved_clean, x_clean).item()

        clean_angle = torch.atan2(x_clean[:, :, 1], x_clean[:, :, 0])
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

    metrics = dict(
        clean_mse_pre = clean_mse_pre,
        perturb_mse = perturb_mse,
        clean_mse_post = clean_mse_post,
        phase_err_pre = phase_err_pre,
        phase_err_post = phase_err_post,
        mse_recovery_gain = clean_mse_pre - clean_mse_post,
        phase_recovery_gain = phase_err_pre - phase_err_post
    )
    print(
        f"{name}: Long-Horizon Recovery "
        f"(clean_pre={clean_mse_pre:.6f}, perturb={perturb_mse:.6f}, clean_post={clean_mse_post:.6f}, "
        f"phase_pre={phase_err_pre:.6f}, phase_post={phase_err_post:.6f})"
    )
    return metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type = str, default = None, help = "Optional path to write benchmark metrics JSON.")
    parser.add_argument("--output-csv", type = str, default = None, help = "Optional path to append benchmark metrics as one CSV row.")
    parser.add_argument("--tag", type = str, default = "manual", help = "Run tag for regression tracking.")
    parser.add_argument("--dim", type = int, default = 128)
    parser.add_argument("--seq-len", type = int, default = 128)
    parser.add_argument("--batch-size", type = int, default = 4)
    parser.add_argument("--chunk-size", type = int, default = 32)
    parser.add_argument("--long-seq-len", type = int, default = 512)
    parser.add_argument("--long-warmup-steps", type = int, default = 30)
    parser.add_argument("--long-perturb-steps", type = int, default = 20)
    parser.add_argument("--long-recovery-steps", type = int, default = 20)
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

if __name__ == "__main__":
    args = parse_args()
    print("Initializing Benchmark...")
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    
    dim = args.dim
    seq_len = args.seq_len
    batch_size = args.batch_size
    chunk_size = args.chunk_size
    
    device = torch.device('cpu')
    if torch.cuda.is_available():
        try:
            # Simple check if cuda actually works
            torch.zeros(1).cuda()
            device = torch.device('cuda')
        except:
            print("CUDA available but non-functional. Using CPU.")
    
    print(f"Using device: {device}")
    
    x = torch.randn(batch_size, seq_len, dim).to(device)
    
    # Baseline
    baseline_mem = NeuralMemory(
        dim=dim,
        chunk_size=chunk_size,
        use_symplectic_gating=False
    ).to(device)
    
    # Symplectic
    symplectic_mem = NeuralMemory(
        dim=dim,
        chunk_size=chunk_size,
        use_symplectic_gating=True
    ).to(device)

    # Symplectic + Paging
    paged_mem = NeuralMemory(
        dim=dim,
        chunk_size=chunk_size,
        use_symplectic_gating=True,
        num_pages=2
    ).to(device)

    # DMD + Paging
    dmd_paged_mem = NeuralMemory(
        dim = dim,
        chunk_size = chunk_size,
        use_dmd_gating = True,
        num_pages = 2
    ).to(device)

    # Symplectic + DMD + Paging (combined complexity)
    combined_paged_mem = NeuralMemory(
        dim = dim,
        chunk_size = chunk_size,
        use_symplectic_gating = True,
        use_dmd_gating = True,
        combine_symplectic_and_dmd = True,
        num_pages = 2,
        symplectic_gate_kwargs = dict(
            gated = True,
            gate_mode = "soft",
            phase_mix = 0.25,
            phase_pairs = max(1, dim // 8)
        )
    ).to(device)

    # Symplectic + Phase + Paging
    phase_paged_mem = NeuralMemory(
        dim = dim,
        chunk_size = chunk_size,
        use_symplectic_gating = True,
        num_pages = 2,
        symplectic_gate_kwargs = dict(
            gated = True,
            gate_mode = "soft",
            phase_mix = 0.5,
            phase_pairs = max(1, dim // 8)
        )
    ).to(device)
    
    print(f"\nConfiguration: Dim={dim}, Seq={seq_len}, Batch={batch_size}")
    print("-" * 40)
    
    t_base_fwd = benchmark_model("Baseline (Fwd)", baseline_mem, x)
    t_symp_fwd = benchmark_model("Symplectic (Fwd)", symplectic_mem, x)
    t_dmd_fwd = benchmark_model("DMD+Paging (Fwd)", dmd_paged_mem, x)
    t_combined_fwd = benchmark_model("Symplectic+DMD+Paging (Fwd)", combined_paged_mem, x)
    t_phase_fwd = benchmark_model("Symplectic+Phase+Paging (Fwd)", phase_paged_mem, x)
    
    overhead_fwd = (t_symp_fwd - t_base_fwd) / t_base_fwd * 100
    print(f"Forward Pass Overhead: {overhead_fwd:.2f}%")
    phase_overhead_fwd = (t_phase_fwd - t_base_fwd) / t_base_fwd * 100
    print(f"Forward Pass Overhead (Phase+Paging): {phase_overhead_fwd:.2f}%")
    dmd_overhead_fwd = (t_dmd_fwd - t_base_fwd) / t_base_fwd * 100
    print(f"Forward Pass Overhead (DMD+Paging): {dmd_overhead_fwd:.2f}%")
    combined_overhead_fwd = (t_combined_fwd - t_base_fwd) / t_base_fwd * 100
    print(f"Forward Pass Overhead (Symplectic+DMD+Paging): {combined_overhead_fwd:.2f}%")
    
    print("-" * 40)
    
    t_base_train = benchmark_training("Baseline (Train)", baseline_mem, x)
    t_symp_train = benchmark_training("Symplectic (Train)", symplectic_mem, x)
    t_dmd_train = benchmark_training("DMD+Paging (Train)", dmd_paged_mem, x)
    t_combined_train = benchmark_training("Symplectic+DMD+Paging (Train)", combined_paged_mem, x)
    t_phase_train = benchmark_training("Symplectic+Phase+Paging (Train)", phase_paged_mem, x)
    
    overhead_train = (t_symp_train - t_base_train) / t_base_train * 100
    print(f"Training Step Overhead: {overhead_train:.2f}%")
    phase_overhead_train = (t_phase_train - t_base_train) / t_base_train * 100
    print(f"Training Step Overhead (Phase+Paging): {phase_overhead_train:.2f}%")
    dmd_overhead_train = (t_dmd_train - t_base_train) / t_base_train * 100
    print(f"Training Step Overhead (DMD+Paging): {dmd_overhead_train:.2f}%")
    combined_overhead_train = (t_combined_train - t_base_train) / t_base_train * 100
    print(f"Training Step Overhead (Symplectic+DMD+Paging): {combined_overhead_train:.2f}%")
    
    print("-" * 40)
    
    def benchmark_recall(name, model, seq_len=128, dim=128, steps=100):
        # Task: Memorize a specific pattern repeated in noise
        # This tests the "memory" aspect more directly than random noise
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Create a structured signal: A repeating sine wave pattern mixed with some data
        t = torch.linspace(0, 4*3.14159, seq_len).unsqueeze(0).unsqueeze(-1)
        signal = torch.sin(t).repeat(4, 1, dim) # [4, 128, 128]
        
        # Add noise to make it adapt
        x = signal + 0.1 * torch.randn_like(signal)
        x = x.to(device)
        
        losses = []
        for _ in range(steps):
            optimizer.zero_grad()
            # NeuralMemory aims to reconstruct/retrieve input
            retrieved, _ = model(x)
            loss = torch.nn.functional.mse_loss(retrieved, x)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        final_loss = sum(losses[-10:]) / 10
        print(f"{name}: Final Reconstruction Loss = {final_loss:.6f}")
        return final_loss

    print("\nTangible Performance: Reconstruction Quality (Lower is Better)")
    print("-" * 40)
    
    loss_base = benchmark_recall("Baseline", baseline_mem, dim=dim, seq_len=seq_len)
    loss_symp = benchmark_recall("Symplectic", symplectic_mem, dim=dim, seq_len=seq_len)
    
    improvement = (loss_base - loss_symp) / loss_base * 100
    print(f"Reconstruction Improvement: {improvement:.2f}%")

    print("\nTangible Performance: Spiral Reconstruction (Lower is Better)")
    print("-" * 40)

    loss_base_spiral = benchmark_spiral_recall("Baseline", baseline_mem, device, dim=dim, seq_len=seq_len)
    loss_symp_spiral = benchmark_spiral_recall("Symplectic", symplectic_mem, device, dim=dim, seq_len=seq_len)
    loss_paged_spiral = benchmark_spiral_recall("Symplectic+Paging", paged_mem, device, dim=dim, seq_len=seq_len)
    loss_dmd_spiral = benchmark_spiral_recall("DMD+Paging", dmd_paged_mem, device, dim=dim, seq_len=seq_len)
    loss_combined_spiral = benchmark_spiral_recall("Symplectic+DMD+Paging", combined_paged_mem, device, dim=dim, seq_len=seq_len)
    loss_phase_spiral = benchmark_spiral_recall("Symplectic+Phase+Paging", phase_paged_mem, device, dim=dim, seq_len=seq_len)

    print("\nTangible Performance: Helix+Drift Reconstruction (Lower is Better)")
    print("-" * 40)

    loss_base_helix = benchmark_helix_drift_recall("Baseline", baseline_mem, device, dim=dim, seq_len=seq_len)
    loss_symp_helix = benchmark_helix_drift_recall("Symplectic", symplectic_mem, device, dim=dim, seq_len=seq_len)
    loss_paged_helix = benchmark_helix_drift_recall("Symplectic+Paging", paged_mem, device, dim=dim, seq_len=seq_len)
    loss_dmd_helix = benchmark_helix_drift_recall("DMD+Paging", dmd_paged_mem, device, dim=dim, seq_len=seq_len)
    loss_combined_helix = benchmark_helix_drift_recall("Symplectic+DMD+Paging", combined_paged_mem, device, dim=dim, seq_len=seq_len)
    loss_phase_helix = benchmark_helix_drift_recall("Symplectic+Phase+Paging", phase_paged_mem, device, dim=dim, seq_len=seq_len)

    print("\nAblation Summary (Lower is Better for losses)")
    print("-" * 64)
    print(f"{'Variant':32s} | {'Spiral':>10s} | {'Helix+Drift':>12s}")
    print("-" * 64)
    print(f"{'Baseline':32s} | {loss_base_spiral:10.6f} | {loss_base_helix:12.6f}")
    print(f"{'Symplectic':32s} | {loss_symp_spiral:10.6f} | {loss_symp_helix:12.6f}")
    print(f"{'Symplectic+Paging':32s} | {loss_paged_spiral:10.6f} | {loss_paged_helix:12.6f}")
    print(f"{'DMD+Paging':32s} | {loss_dmd_spiral:10.6f} | {loss_dmd_helix:12.6f}")
    print(f"{'Symplectic+DMD+Paging':32s} | {loss_combined_spiral:10.6f} | {loss_combined_helix:12.6f}")
    print(f"{'Symplectic+Phase+Paging':32s} | {loss_phase_spiral:10.6f} | {loss_phase_helix:12.6f}")

    print("\nTangible Performance: Long-Horizon Perturbation Recovery")
    print("-" * 64)
    long_base = benchmark_long_horizon_recovery(
        "Baseline",
        baseline_mem,
        device,
        seq_len = args.long_seq_len,
        dim = dim,
        warmup_steps = args.long_warmup_steps,
        perturb_steps = args.long_perturb_steps,
        recovery_steps = args.long_recovery_steps
    )
    long_symp = benchmark_long_horizon_recovery(
        "Symplectic+Paging",
        paged_mem,
        device,
        seq_len = args.long_seq_len,
        dim = dim,
        warmup_steps = args.long_warmup_steps,
        perturb_steps = args.long_perturb_steps,
        recovery_steps = args.long_recovery_steps
    )
    long_phase = benchmark_long_horizon_recovery(
        "Symplectic+Phase+Paging",
        phase_paged_mem,
        device,
        seq_len = args.long_seq_len,
        dim = dim,
        warmup_steps = args.long_warmup_steps,
        perturb_steps = args.long_perturb_steps,
        recovery_steps = args.long_recovery_steps
    )


    print("\nVerifying Complexity Scores (Sanity Check)...")
    with torch.no_grad():
        gate = symplectic_mem.symplectic_gate
        complexity = gate(x)
        print(f"Mean Complexity Score: {complexity.mean().item():.4f}")
        print(f"Max Complexity Score:  {complexity.max().item():.4f}")
        print(f"Min Complexity Score:  {complexity.min().item():.4f}")

    if hasattr(paged_mem, "page_switch_events"):
        print(f"Page Switch Events: {paged_mem.page_switch_events.item()}")

    metrics = dict(
        timestamp_utc = datetime.now(timezone.utc).isoformat(),
        tag = args.tag,
        device = str(device),
        config = dict(dim = dim, seq_len = seq_len, batch_size = batch_size, chunk_size = chunk_size),
        timing = dict(
            fwd_ms = dict(
                baseline = t_base_fwd * 1000,
                symplectic = t_symp_fwd * 1000,
                dmd_paging = t_dmd_fwd * 1000,
                symplectic_dmd_paging = t_combined_fwd * 1000,
                symplectic_phase_paging = t_phase_fwd * 1000
            ),
            train_ms = dict(
                baseline = t_base_train * 1000,
                symplectic = t_symp_train * 1000,
                dmd_paging = t_dmd_train * 1000,
                symplectic_dmd_paging = t_combined_train * 1000,
                symplectic_phase_paging = t_phase_train * 1000
            ),
            overhead_pct = dict(
                fwd_symplectic = overhead_fwd,
                fwd_dmd_paging = dmd_overhead_fwd,
                fwd_symplectic_dmd_paging = combined_overhead_fwd,
                fwd_symplectic_phase_paging = phase_overhead_fwd,
                train_symplectic = overhead_train,
                train_dmd_paging = dmd_overhead_train,
                train_symplectic_dmd_paging = combined_overhead_train,
                train_symplectic_phase_paging = phase_overhead_train
            )
        ),
        reconstruction = dict(
            baseline = loss_base,
            symplectic = loss_symp,
            improvement_pct = improvement
        ),
        spiral = dict(
            baseline = loss_base_spiral,
            symplectic = loss_symp_spiral,
            symplectic_paging = loss_paged_spiral,
            dmd_paging = loss_dmd_spiral,
            symplectic_dmd_paging = loss_combined_spiral,
            symplectic_phase_paging = loss_phase_spiral
        ),
        helix_drift = dict(
            baseline = loss_base_helix,
            symplectic = loss_symp_helix,
            symplectic_paging = loss_paged_helix,
            dmd_paging = loss_dmd_helix,
            symplectic_dmd_paging = loss_combined_helix,
            symplectic_phase_paging = loss_phase_helix
        ),
        long_horizon_recovery = dict(
            baseline = long_base,
            symplectic_paging = long_symp,
            symplectic_phase_paging = long_phase
        ),
        diagnostics = dict(
            page_switch_events = int(paged_mem.page_switch_events.item()) if hasattr(paged_mem, "page_switch_events") else None
        )
    )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents = True, exist_ok = True)
        out_path.write_text(json.dumps(metrics, indent = 2), encoding = "utf-8")
        print(f"Saved metrics JSON: {out_path}")

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
        print(f"Appended metrics CSV row: {csv_path}")
