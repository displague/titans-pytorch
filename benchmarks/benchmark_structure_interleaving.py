
import torch
import time
import math
from titans_pytorch import NeuralMemory

def benchmark_interleaved_recall(name, model, dim=128, seq_len_per_context=64, steps=100, device='cpu'):
    """
    Task: Interleaved Context Recall.
    
    Structure:
    1. Context A (Key=A_k, Value=A_v) - 'Linear' topology
    2. Twist (Transition) - Should trigger Page Switch
    3. Context B (Key=B_k, Value=B_v) - 'Linear' topology (but conflicting with A?)
       Actually, let's use the SAME Keys but DIFFERENT Values to force conflict.
       Context A: Keys K1..K10 -> Values V_A1..V_A10
       Context B: Keys K1..K10 -> Values V_B1..V_B10
       
       If model stays on Page 0: It overwrites V_A with V_B.
       If model switches to Page 1: It writes V_B to Page 1.
       
    4. Recall A: Query K1..K10. Model should output V_A (from Page 0).
       Wait, if it reads from ALL pages, it gets V_A + V_B.
       We need the decay/gate to handle this?
       Or does the read mechanism differ?
       Retrieval is a sum. If V_A and V_B are both present, we get V_A + V_B.
       Unless one dominates.
       
       However, if we use *Orthogonal* keys for the 'Twist' detector but *Same* keys for storage, it simulates the "Context Switch".
       
       Let's try a simpler task first where Paging simply *expands capacity* and prevents decay of A.
    """
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Generate Context A (Low Complexity)
    # Just random noise is "Linear" enough for the Gate (as verified in tests, random ~0.1 < 0.5)
    # But we want to ensure it is stored.
    
    # Keys shared
    keys = torch.randn(1, seq_len_per_context, dim).to(device)
    
    # Values distinct
    values_A = torch.randn(1, seq_len_per_context, dim).to(device)
    values_B = torch.randn(1, seq_len_per_context, dim).to(device) 
    
    # To trigger a TWIST between A and B, we insert a "Twist Sequence".
    # A Twist Sequence is one where q(x) != k(x) significantly.
    # We found in tests that just varying seq is enough if projections differ.
    # But here we want a specific "High Twist" burst.
    
    # Let's generate a High Twist sequence manually if possible, or just rely on random noise being low
    # and we inject a high-noise burst?
    # Actually, let's look at the Gate logic.
    # twist = q_curr * k_prev - q_prev * k_curr.
    # If we set x such that this is high.
    # Hard to engineer x without knowing weights.
    # But we know random noise gives ~0.1.
    # The threshold is 0.5.
    
    # Let's simple rely on the fact that if we use the SAME model, 
    # the Symplectic one HAS the OPTION to page.
    # But we need to Force it.
    
    # Hack: We will MANUALLY set the threshold low enough that SOME noise triggers it,
    # OR we accept that for this benchmark we might need to lower threshold in the model.
    
    losses = []
    
    # Construct sequence: [Context A] [Twist] [Context B] [Query A]
    # We want to maximize recall of A.
    
    # Twist: 10 steps of high variance noise?
    twist_len = 10
    twist = torch.randn(1, twist_len, dim).to(device) * 5.0 # High magnitude might trigger something if norms aren't perfect? 
    # Actually Gate normalizes inputs. Magnitude doesn't matter.
    
    input_seq = torch.cat([values_A, twist, values_B], dim=1)
    target = values_A # We want to recall A at the end?
    # No, NeuralMemory is usually autoregressive or strictly local-associative.
    # It stores (K,V) and retrieves V given K.
    
    # This benchmark structure is getting complicated for a generic class.
    # Let's stick to the "Needle in Haystack" style from the paper that Symplectic improves.
    # Long sequence of Noise (Haystack), with repeated Needles.
    # If Paging works, it puts Needles on a separate page?
    # No, Paging puts *Complex* things on separate pages.
    # If Needles are Complex (Twisted), they go to Page 1. Haystack (Linear) stays on Page 0.
    # Then distinct pages prevents Haystack from overwriting Needle.
    
    # Dataset:
    # background = Linear (Sine wave or constant line)
    # needle = Twisted (High frequency / Random)
    
    t = torch.linspace(0, 8*3.14, seq_len_per_context).to(device)
    background = torch.sin(t).unsqueeze(0).unsqueeze(-1).repeat(1, 1, dim) # (1, N, D)
    
    # Needle: Random noise
    needle = torch.randn(1, 10, dim).to(device)
    
    # Sequence: Background -> Needle -> Background -> Recall Needle
    # Input: [Back(50), Needle(10), Back(50)]
    # We query with Needle(10) again? 
    # Or just check reconstruction loss on the random part?
    
    # Let's check reconstruction loss on the WHOLE sequence.
    # If Symplectic works:
    # 1. Back -> Page 0.
    # 2. Needle -> Twist -> Page 1.
    # 3. Back -> Page 0.
    # Since Back went to Page 0, it shouldn't overwrite Needle on Page 1.
    # So Needle should be better preserved.
    # In Baseline, Back overwrites Needle (or decays it).
    
    x = torch.cat([background[:, :50], needle, background[:, 50:]], dim=1) # (1, 110, D)
    
    # Training Loop
    start_time = time.time()
    for _ in range(steps):
        optimizer.zero_grad()
        retrieved, _ = model(x)
        
        # Loss: We care most about reconstructing the NEEDLE (middle part)
        # indices: 50:60
        needle_out = retrieved[:, 50:60]
        needle_target = x[:, 50:60]
        
        loss = torch.nn.functional.mse_loss(needle_out, needle_target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
    avg_loss = sum(losses[-10:]) / 10
    total_time = time.time() - start_time
    
    return avg_loss, total_time

if __name__ == "__main__":
    print("Initializing Structural Benchmark...")
    dim = 64
    chunk_size = 16
    
    device = torch.device('cpu')
    
    # We need a low threshold because our "Needle" (Random Noise) isn't THAT twisted (0.1 ~ 0.4).
    # Baseline threshold is 0.5.
    # We will set threshold to 0.2 for this benchmark to ensure paging triggers on the noise.
    # Background (Sine) is very linear (~0.0).
    
    print("-" * 50)
    print("Task: 'Needle in a Linear Stack'")
    print("Background: Sine Wave (Linear, Low complexity)")
    print("Needle: Random Noise (Higher complexity)")
    print("Goal: Reconstruction of the Needle after subsequent Background.")
    print("-" * 50)

    # Baseline
    baseline = NeuralMemory(dim=dim, chunk_size=chunk_size, use_symplectic_gating=False).to(device)
    
    # Symplectic
    # Note: We set num_pages=4 to give plenty of space.
    symplectic = NeuralMemory(
        dim=dim, 
        chunk_size=chunk_size, 
        use_symplectic_gating=True, 
        num_pages=4,
        symplectic_page_threshold=0.2 # Lower threshold to catch random noise as 'twist'
    ).to(device)
    
    loss_base, time_base = benchmark_interleaved_recall("Baseline", baseline, dim=dim, device=device)
    print(f"Baseline:   Loss = {loss_base:.6f} | Time = {time_base:.2f}s")
    
    loss_symp, time_symp = benchmark_interleaved_recall("Symplectic", symplectic, dim=dim, device=device)
    print(f"Symplectic: Loss = {loss_symp:.6f} | Time = {time_symp:.2f}s")
    
    imp = (loss_base - loss_symp) / loss_base * 100
    print("-" * 50)
    print(f"Improvement: {imp:.2f}%")
    if imp > 0:
        print("SUCCESS: Symplectic Memory preserved the Needle better!")
    else:
        print("FAIL: Symplectic Memory did not improve performance.")
