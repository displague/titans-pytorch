import time
import math
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from titans_pytorch import MemoryAsContextTransformer, MemoryMLP

class SyntheticTextDataset(Dataset):
    def __init__(self, vocab_size: int = 256, seq_len: int = 128, length: int = 50000, device: torch.device | None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.length = length
        self.device = device or torch.device('cpu')

    def __len__(self):
        return self.length // self.seq_len

    def __getitem__(self, idx):
        x = torch.randint(0, self.vocab_size, (self.seq_len + 1,), dtype=torch.long)
        return x.to(self.device)

@torch.no_grad()
def compute_loss(model, batch):
    model.eval()
    return model(batch, return_loss=True).item()


def run_experiment(use_gating: bool, num_pages: int, threshold: float, steps: int, device: torch.device):
    dim = 128
    depth = 4
    heads = 4
    window = 32
    seq_len = 128
    batch_size = 2

    neural_memory_model = MemoryMLP(dim=64, depth=2)

    use_accel = torch.cuda.is_available()
    if not use_accel:
        print("[INFO] CUDA not available, disabling accelerated scan for assoc_scan.")
    model = MemoryAsContextTransformer(
        num_tokens=256,
        dim=dim,
        depth=depth,
        segment_len=window,
        num_persist_mem_tokens=2,
        num_longterm_mem_tokens=2,
        neural_memory_layers=(2, 4),
        neural_memory_segment_len=4,
        neural_memory_batch_size=64,
        sliding_window_attn=True,
        use_flex_attn=use_accel,
        neural_memory_model=neural_memory_model,
        neural_memory_kwargs=dict(
            dim_head=64,
            heads=heads,
            attn_pool_chunks=True,
            qk_rmsnorm=True,
            momentum=True,
            momentum_order=1,
            default_step_transform_max_lr=1e-1,
            use_accelerated_scan=use_accel,
            per_parameter_lr_modulation=True,
            spectral_norm_surprises=True,
            use_symplectic_gating=use_gating,
            num_pages=num_pages,
            symplectic_page_threshold=threshold,
        ),
    ).to(device)

    dataset = SyntheticTextDataset(seq_len=seq_len, device=device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)

    # training loop
    t0 = time.time()
    train_losses = []
    val_losses = []

    for step, batch in enumerate(loader):
        if step >= steps:
            break
        model.train()
        opt.zero_grad()
        loss = model(batch, return_loss=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        train_losses.append(loss.item())

        if (step + 1) % max(10, steps // 5) == 0:
            # quick validation on a fresh batch
            val_batch = next(iter(loader))
            vloss = compute_loss(model, val_batch)
            val_losses.append(vloss)

    elapsed = time.time() - t0

    # gather page-switch counts across all NeuralMemory layers
    switch_counts = []
    for layer in model.layers:
        mem = layer[4]
        if isinstance(mem, nn.Module) and hasattr(mem, 'use_symplectic_gating') and mem.use_symplectic_gating:
            cnt = int(mem.page_switch_events.item())
            switch_counts.append(cnt)
    total_switches = sum(switch_counts)

    return {
        'train_loss_last': train_losses[-1] if train_losses else math.nan,
        'val_loss_last': val_losses[-1] if val_losses else math.nan,
        'train_loss_mean': sum(train_losses) / len(train_losses) if train_losses else math.nan,
        'val_loss_mean': sum(val_losses) / len(val_losses) if val_losses else math.nan,
        'steps': steps,
        'elapsed_sec': elapsed,
        'switch_counts': switch_counts,
        'total_switches': total_switches,
    }


def main():
    parser = argparse.ArgumentParser(description='Tiny MAC paging benchmark')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--pages', type=int, default=2)
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    print('Running baseline (no gating/paging) ...')
    base = run_experiment(use_gating=False, num_pages=1, threshold=1.0, steps=args.steps, device=device)

    print('Running gating+paging ...')
    exp = run_experiment(use_gating=True, num_pages=args.pages, threshold=args.threshold, steps=args.steps, device=device)

    print('\n=== Results ===')
    print(f"Steps: {args.steps}, Device: {device}")
    print(f"Baseline: train_mean={base['train_loss_mean']:.4f}, val_mean={base['val_loss_mean']:.4f}, time={base['elapsed_sec']:.2f}s")
    print(f"Gating+Paging: train_mean={exp['train_loss_mean']:.4f}, val_mean={exp['val_loss_mean']:.4f}, time={exp['elapsed_sec']:.2f}s, switches={exp['switch_counts']} (total={exp['total_switches']})")

    # simple delta indicator
    if not math.isnan(base['val_loss_mean']) and not math.isnan(exp['val_loss_mean']):
        delta = exp['val_loss_mean'] - base['val_loss_mean']
        print(f"Val loss delta (exp - base): {delta:+.4f}")


if __name__ == '__main__':
    main()
