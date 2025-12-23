import time
import torch
from titans_pytorch import NeuralMemory

"""
Benchmark: Objective Reduction (Manifold Paging)

Goals
- Show paging reduces interference (catastrophic forgetting)
- Measure training step overhead vs baseline
- Keep baseline logic unchanged (paging is optional via flags)

Protocol
1) Train on Task A for N steps (pattern A)
2) Train on Task B for N steps (pattern B, conflicting)
3) Evaluate loss on Task A again (post-B). Lower is better; measures forgetting.

We compare:
- Baseline: use_symplectic_gating=False, num_pages=1
- Paging:  use_symplectic_gating=True,  num_pages=4 (page threshold lowered to ensure switch)

Loss is reconstruction MSE of retrieved vs input (auto-associative memory).
"""


def make_model(dim: int, chunk_size: int, use_symp: bool, num_pages: int):
    m = NeuralMemory(
        dim=dim,
        chunk_size=chunk_size,
        use_symplectic_gating=use_symp,
        num_pages=num_pages,
    )
    # lower threshold so page switching is likely in this toy setting
    if use_symp and num_pages > 1:
        m.symplectic_page_threshold = 0.1
    return m


def task_sequence(batch: int, seq_len: int, dim: int, device: torch.device, task: str):
    """
    Task A: Adds a strong vector pattern +v to every c-th token
    Task B: Adds a rotated/negated pattern -v to the same footprint
    Creates high twist/complexity when swapping between A and B.
    """
    x = torch.randn(batch, seq_len, dim, device=device) * 0.1

    # base concept vector
    v = torch.randn(1, 1, dim, device=device)
    v = torch.nn.functional.normalize(v, dim=-1) * 5.0

    # schedule of positions (every interval)
    interval = 8
    for i in range(0, seq_len, interval):
        if task == 'A':
            x[:, i:i+1, :] += v
        else:
            x[:, i:i+1, :] -= v
    return x


def train_on_task(name: str, model: NeuralMemory, optimizer, steps: int, batch: int, seq_len: int, dim: int, device: torch.device, task: str):
    model.train()
    losses = []
    start = time.time()
    for step in range(steps):
        optimizer.zero_grad()
        x = task_sequence(batch, seq_len, dim, device, task)
        retrieved, _ = model(x)
        loss = torch.nn.functional.mse_loss(retrieved, x)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % max(1, steps // 5) == 0:
            print(f"  {name} step {step:03d}: loss={loss.item():.6f}")
    dur = time.time() - start
    return losses, dur


def eval_task_loss(model: NeuralMemory, batch: int, seq_len: int, dim: int, device: torch.device, task: str, iters: int = 10):
    model.eval()
    with torch.no_grad():
        losses = []
        for _ in range(iters):
            x = task_sequence(batch, seq_len, dim, device, task)
            retrieved, _ = model(x)
            loss = torch.nn.functional.mse_loss(retrieved, x)
            losses.append(loss.item())
        return sum(losses) / len(losses)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Paging benchmark: interference and time overhead")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--pages", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    dim = args.dim
    seq_len = args.seq_len
    batch = args.batch
    steps = args.steps
    chunk_size = args.chunk_size

    # Baseline
    baseline = make_model(dim, chunk_size, use_symp=False, num_pages=1).to(device)
    opt_base = torch.optim.Adam(baseline.parameters(), lr=1e-3)

    print("\n[Baseline] Train on Task A")
    base_A_losses, base_A_time = train_on_task("Baseline:A", baseline, opt_base, steps, batch, seq_len, dim, device, task='A')
    print("[Baseline] Train on Task B")
    base_B_losses, base_B_time = train_on_task("Baseline:B", baseline, opt_base, steps, batch, seq_len, dim, device, task='B')

    base_A_post = eval_task_loss(baseline, batch, seq_len, dim, device, task='A')

    # Paging
    paging = make_model(dim, chunk_size, use_symp=True, num_pages=args.pages).to(device)
    paging.symplectic_page_threshold = args.threshold
    opt_page = torch.optim.Adam(paging.parameters(), lr=1e-3)

    print("\n[Paging] Train on Task A")
    page_A_losses, page_A_time = train_on_task("Paging:A", paging, opt_page, steps, batch, seq_len, dim, device, task='A')
    print("[Paging] Train on Task B")
    page_B_losses, page_B_time = train_on_task("Paging:B", paging, opt_page, steps, batch, seq_len, dim, device, task='B')

    page_A_post = eval_task_loss(paging, batch, seq_len, dim, device, task='A')

    # Forward-only timing (optional quick check)
    def time_forward(model, x, iters=50):
        model.eval()
        start = time.time()
        with torch.no_grad():
            for _ in range(iters):
                _ = model(x)
        return (time.time() - start) / iters

    x_rand = torch.randn(batch, seq_len, dim, device=device)
    base_fwd_ms = time_forward(baseline, x_rand) * 1000
    page_fwd_ms = time_forward(paging, x_rand) * 1000
    fwd_overhead = (page_fwd_ms - base_fwd_ms) / base_fwd_ms * 100 if base_fwd_ms > 0 else 0.0

    # Metrics
    base_A_final = sum(base_A_losses[-10:]) / 10
    base_B_final = sum(base_B_losses[-10:]) / 10
    page_A_final = sum(page_A_losses[-10:]) / 10
    page_B_final = sum(page_B_losses[-10:]) / 10

    interference_improv = (base_A_post - page_A_post) / base_A_post * 100 if base_A_post > 0 else 0.0
    train_time_overhead_A = (page_A_time - base_A_time) / base_A_time * 100 if base_A_time > 0 else 0.0
    train_time_overhead_B = (page_B_time - base_B_time) / base_B_time * 100 if base_B_time > 0 else 0.0

    print("\n" + "="*60)
    print("Objective Reduction (Paging) Benchmark")
    print("="*60)
    print(f"Baseline A final loss:   {base_A_final:.6f}")
    print(f"Baseline B final loss:   {base_B_final:.6f}")
    print(f"Paging   A final loss:   {page_A_final:.6f}")
    print(f"Paging   B final loss:   {page_B_final:.6f}")
    print(f"Baseline A post-B loss:  {base_A_post:.6f}  (forgetting)")
    print(f"Paging   A post-B loss:  {page_A_post:.6f}  (less forgetting is better)")
    print(f"Interference improvement: {interference_improv:+.2f}%")
    print(f"Train time overhead (A): {train_time_overhead_A:+.2f}%")
    print(f"Train time overhead (B): {train_time_overhead_B:+.2f}%")
    print(f"Forward-only overhead:   {fwd_overhead:+.2f}%  (Baseline {base_fwd_ms:.2f} ms | Paging {page_fwd_ms:.2f} ms)")
    print("="*60)

    if interference_improv > 0:
        print("Result: Paging reduced interference relative to baseline.")
    else:
        print("Result: No interference gain observed; consider tuning threshold/pages/chunk_size.")
