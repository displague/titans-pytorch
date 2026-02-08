
import time
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

if __name__ == "__main__":
    print("Initializing Benchmark...")
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    
    dim = 128
    seq_len = 128
    batch_size = 4
    chunk_size = 32
    
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
    
    print(f"\nConfiguration: Dim={dim}, Seq={seq_len}, Batch={batch_size}")
    print("-" * 40)
    
    t_base_fwd = benchmark_model("Baseline (Fwd)", baseline_mem, x)
    t_symp_fwd = benchmark_model("Symplectic (Fwd)", symplectic_mem, x)
    
    overhead_fwd = (t_symp_fwd - t_base_fwd) / t_base_fwd * 100
    print(f"Forward Pass Overhead: {overhead_fwd:.2f}%")
    
    print("-" * 40)
    
    t_base_train = benchmark_training("Baseline (Train)", baseline_mem, x)
    t_symp_train = benchmark_training("Symplectic (Train)", symplectic_mem, x)
    
    overhead_train = (t_symp_train - t_base_train) / t_base_train * 100
    print(f"Training Step Overhead: {overhead_train:.2f}%")
    
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

    print("\nTangible Performance: Helix+Drift Reconstruction (Lower is Better)")
    print("-" * 40)

    loss_base_helix = benchmark_helix_drift_recall("Baseline", baseline_mem, device, dim=dim, seq_len=seq_len)
    loss_symp_helix = benchmark_helix_drift_recall("Symplectic", symplectic_mem, device, dim=dim, seq_len=seq_len)
    loss_paged_helix = benchmark_helix_drift_recall("Symplectic+Paging", paged_mem, device, dim=dim, seq_len=seq_len)


    print("\nVerifying Complexity Scores (Sanity Check)...")
    with torch.no_grad():
        gate = symplectic_mem.symplectic_gate
        complexity = gate(x)
        print(f"Mean Complexity Score: {complexity.mean().item():.4f}")
        print(f"Max Complexity Score:  {complexity.max().item():.4f}")
        print(f"Min Complexity Score:  {complexity.min().item():.4f}")

    if hasattr(paged_mem, "page_switch_events"):
        print(f"Page Switch Events: {paged_mem.page_switch_events.item()}")
