
import torch
import time
import matplotlib.pyplot as plt
from titans_pytorch import NeuralMemory

def train_needle_retrieval(name, model, steps=160, seq_len=256, dim=128, device='cpu'):
    print(f"\nTraining {name} on Needle Retrieval for {steps} steps...")
    
    # Optimizer for the model's parameters (gates, projections, etc.)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    
    losses = []
    
    # Fixed Needle Pattern to learn
    needle_val = torch.ones(1, 1, dim).to(device) * 5.0
    
    start_time = time.time()
    
    for step in range(steps):
        optimizer.zero_grad()
        
        # 1. Generate Haystack (Noise)
        haystack = torch.randn(1, seq_len, dim).to(device)
        
        # 2. Insert Needle at random position
        needle_pos = torch.randint(0, seq_len, (1,)).item()
        
        input_seq = haystack.clone()
        input_seq[:, needle_pos:needle_pos+1, :] = needle_val
        
        # 3. Forward Pass
        # NeuralMemory returns 'retrieved' which is the prediction/reconstruction
        retrieved, _ = model(input_seq)
        
        # 4. Loss: Ability to reconstruct the input (Auto-Associative Memory)
        # We care specifically about reconstructing the needle, but standard training minimizes global MSE
        loss = torch.nn.functional.mse_loss(retrieved, input_seq)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 20 == 0:
            print(f"  Step {step:03d}: Loss = {loss.item():.6f}")

    end_time = time.time()
    print(f"{name} Final Loss: {losses[-1]:.6f} (Time: {end_time - start_time:.2f}s)")
    
    return losses

if __name__ == "__main__":
    print("Initializing Comparative Needle Benchmark...")
    
    dim = 64
    seq_len = 512
    steps = 160
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Baseline Model
    baseline_mem = NeuralMemory(
        dim=dim,
        chunk_size=32,
        use_symplectic_gating=False
    ).to(device)
    
    # 2. Symplectic Model
    symplectic_mem = NeuralMemory(
        dim=dim,
        chunk_size=32,
        use_symplectic_gating=True
    ).to(device)
    
    # Run Training
    loss_base = train_needle_retrieval("Baseline", baseline_mem, steps=steps, seq_len=seq_len, dim=dim, device=device)
    loss_symp = train_needle_retrieval("Symplectic", symplectic_mem, steps=steps, seq_len=seq_len, dim=dim, device=device)
    
    # Analysis
    final_improvement = (loss_base[-1] - loss_symp[-1]) / loss_base[-1] * 100
    avg_improvement = (sum(loss_base[-10:]) - sum(loss_symp[-10:])) / sum(loss_base[-10:]) * 100
    
    print("\n" + "="*50)
    print(f"RESULTS (Over {steps} Steps)")
    print("="*50)
    print(f"Baseline Final Loss:   {loss_base[-1]:.6f}")
    print(f"Symplectic Final Loss: {loss_symp[-1]:.6f}")
    print(f"Improvement (Spot):    {final_improvement:+.2f}%")
    print(f"Improvement (Last 10): {avg_improvement:+.2f}%")
    print("="*50)

    # Optional: ASCII Plot
    def ascii_plot(data, height=10):
        m_min, m_max = min(data), max(data)
        metrics = [
            [" " if val < m_min + (i/height)*(m_max-m_min) else "█" for val in data]
            for i in range(height)
        ]
        # Transpose and print would require lines
        # Simple line printer
        print(f"Plot Range: {m_min:.4f} - {m_max:.4f}")
        
    # print("\nBaseline Curve:")
    # ascii_plot(loss_base)
    # print("\nSymplectic Curve:")
    # ascii_plot(loss_symp)
