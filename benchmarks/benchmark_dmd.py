
import torch
import time
import math
from titans_pytorch.dmd import DynamicModeDecomposition
from titans_pytorch.symplectic_gate import SymplecticGating

def benchmark_dmd_vs_symplectic(dim=64, seq_len=128, batch_size=4, device='cpu'):
    print(f"\nConfiguration: Dim={dim}, Seq={seq_len}, Batch={batch_size}, Device={device}")
    
    # 1. Models
    dmd = DynamicModeDecomposition(rank=min(dim, seq_len)).to(device)
    symp = SymplecticGating(dim).to(device)
    
    # 2. Data Generation
    # Case A: Linear (Sine Waves) - Should be Low Complexity/Error
    t = torch.linspace(0, 4*3.14159, seq_len).to(device)
    # Different freq per batch/dim
    linear_data = torch.stack([
        torch.stack([torch.sin(t * (b+1) + (d*0.1)) for d in range(dim)], dim=-1)
        for b in range(batch_size)
    ]) # (B, Seq, Dim)
    
    # Case B: Non-Linear / Random
    nonlinear_data = torch.randn(batch_size, seq_len, dim).to(device)
    
    # 3. Speed Test
    steps = 20
    
    # DMD Speed
    start = time.time()
    for _ in range(steps):
        _ = dmd(linear_data)
    dmd_time = (time.time() - start) / steps
    
    # Symplectic Speed
    start = time.time()
    for _ in range(steps):
        _ = symp(nonlinear_data)
    symp_time = (time.time() - start) / steps
    
    print("-" * 40)
    print(f"DMD Avg Time:        {dmd_time*1000:.2f} ms")
    print(f"Symplectic Avg Time: {symp_time*1000:.2f} ms")
    print(f"Ratio (DMD/Symp):    {dmd_time/symp_time:.1f}x Slowdown")
    print("-" * 40)
    
    # 4. Sensitivity Test
    # Compare "Scores" for Linear vs Non-Linear
    
    print("\nSensitivity Analysis (Linear vs Random):")
    
    # DMD Error
    with torch.no_grad():
        err_lin = dmd(linear_data).mean().item()
        err_non = dmd(nonlinear_data).mean().item()
    
    print(f"DMD Error [Linear]:     {err_lin:.6f}")
    print(f"DMD Error [Random]:     {err_non:.6f}")
    print(f"DMD Discrimination:     {err_non / (err_lin+1e-9):.1f}x")
    
    # Symplectic Complexity
    with torch.no_grad():
        comp_lin = symp(linear_data).mean().item()
        comp_non = symp(nonlinear_data).mean().item()
        
    print(f"Symplectic Score [Lin]: {comp_lin:.6f}")
    print(f"Symplectic Score [Ran]: {comp_non:.6f}")
    print(f"Symplectic Discrim:     {comp_non / (comp_lin+1e-9):.1f}x")
    
    print("-" * 40)
    if (err_non / (err_lin+1e-9)) > (comp_non / (comp_lin+1e-9)):
        print("DMD is more discriminative (Better Signal-to-Noise ratio).")
    else:
        print("Symplectic Gating is more discriminative/aggressive.")

if __name__ == "__main__":
    device = torch.device('cpu')
    if torch.cuda.is_available():
        # Quick check
        try:
             torch.zeros(1).cuda()
             device = torch.device('cuda')
        except:
             pass
             
    benchmark_dmd_vs_symplectic(device=device)
    # Benchmark with larger sequence length to see scaling
    benchmark_dmd_vs_symplectic(dim=64, seq_len=512, batch_size=4, device=device)
