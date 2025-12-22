
import torch
import time
from titans_pytorch import NeuralMemory

def benchmark_model(name, model, x, steps=5):
    model.eval()
    
    # Warmup
    for _ in range(5):
        _ = model(x)
        
    start_time = time.time()
    for _ in range(steps):
        _ = model(x)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / steps
    print(f"{name}: Avg Forward Time = {avg_time*1000:.2f} ms")
    return avg_time

def benchmark_training(name, model, x, steps=5):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        loss, _ = model(x)
        loss = loss.sum()
        loss.backward()
        optimizer.step()
        
    start_time = time.time()
    for _ in range(steps):
        optimizer.zero_grad()
        loss, _ = model(x)
        loss = loss.sum()
        loss.backward()
        optimizer.step()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / steps
    print(f"{name}: Avg Train Step Time = {avg_time*1000:.2f} ms")
    return avg_time

if __name__ == "__main__":
    print("Initializing Benchmark...")
    
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

    print("\nVerifying Complexity Scores (Sanity Check)...")
    with torch.no_grad():
        gate = symplectic_mem.symplectic_gate
        complexity = gate(x)
        print(f"Mean Complexity Score: {complexity.mean().item():.4f}")
        print(f"Max Complexity Score:  {complexity.max().item():.4f}")
        print(f"Min Complexity Score:  {complexity.min().item():.4f}")
