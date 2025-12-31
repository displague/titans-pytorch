
import torch
import time
from titans_pytorch import NeuralMemory

def generate_grammar_sequence(batch_size, seq_len, dim, device='cpu'):
    """
    Generates a sequence with 'Grammatical Structure' (Nested Dependencies).
    Specifically, we generate an 'A...B' structure where A implies B must appear later.
    This simulates code syntax (opening/closing braces) or natural language (subject-verb agreement).
    
    Structure:
    [Start Token A] [Random Noise C] [End Token B which depends on A]
    """
    # 1. Base Noise
    seq = torch.randn(batch_size, seq_len, dim).to(device) * 0.1
    
    # 2. Define "Concepts" (Orthogonal Vectors)
    params_a = torch.randn(1, 1, dim).to(device) # Concept 'Open Brace'
    params_a = torch.nn.functional.normalize(params_a, dim=-1) * 5.0
    
    params_b = -params_a # Concept 'Close Brace' is inverse of Open
    
    # 3. Inject Structure
    # Every 20-40 tokens, we start a new 'clause'
    interval = 32
    
    # Pre-squeeze for broadcasting: [1, 1, dim] -> [1, dim] -> broadcast to [8, dim]
    params_a_squeezed = params_a.squeeze(0).squeeze(0) # Shape: [dim]
    params_b_squeezed = params_b.squeeze(0).squeeze(0)
    
    for i in range(0, seq_len - interval, interval):
        # Open the clause
        seq[:, i, :] += params_a_squeezed
        
        # Close the clause (delayed by 'gap')
        gap = 16 # Short-to-medium range dependency
        seq[:, i + gap, :] += params_b_squeezed
        
        # Add a "Twist" (Interference) in between
        # This is what Symplectic Gate should detect: a rotation in the vector space
        # We rotate the noise in the gap
        # If the model flattens this, it loses the connection between A and B
        pass
        
    return seq

def train_grammar_task(name, model, steps=200, seq_len=512, dim=64, device='cpu'):
    print(f"\nTraining {name} on Structured Grammar Task...")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3) # Slightly aggressive LR
    model.train()
    
    losses = []
    
    for step in range(steps):
        optimizer.zero_grad()
        
        # Generate new batch of structured data
        input_seq = generate_grammar_sequence(8, seq_len, dim, device)
        
        # Task: Auto-encoding / Next-step prediction (simulated by reconstruction here)
        # We want to see if the model learned the "Params A implies Params B" rule
        retrieved, _ = model(input_seq)
        
        # Loss
        loss = torch.nn.functional.mse_loss(retrieved, input_seq)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 50 == 0:
            print(f"  Step {step:03d}: Loss = {loss.item():.6f}")
            
    return losses

if __name__ == "__main__":
    print("Initializing Structured Grammar Benchmark...")
    print("(Simulating Code Syntax / Nested Dependencies)")
    
    dim = 64
    seq_len = 256 # Medium-Long context
    steps = 300
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Baseline
    baseline_mem = NeuralMemory(
        dim=dim,
        chunk_size=16, # Smaller chunks for finer granularity
        use_symplectic_gating=False
    ).to(device)
    
    # 2. Symplectic
    symplectic_mem = NeuralMemory(
        dim=dim,
        chunk_size=16,
        use_symplectic_gating=True,
        num_pages=2,
        symplectic_page_threshold=0.4 # Slightly lower threshold to encourage switching
    ).to(device)
    
    # 3. DMD
    dmd_mem = NeuralMemory(
        dim=dim,
        chunk_size=16,
        use_dmd_gating=True,
        num_pages=2,
        symplectic_page_threshold=0.4
    ).to(device)
    
    # Run
    loss_base = train_grammar_task("Baseline", baseline_mem, steps=steps, seq_len=seq_len, dim=dim, device=device)
    loss_symp = train_grammar_task("Symplectic", symplectic_mem, steps=steps, seq_len=seq_len, dim=dim, device=device)
    loss_dmd  = train_grammar_task("DMD", dmd_mem, steps=steps, seq_len=seq_len, dim=dim, device=device)
    
    # Results
    base_final = sum(loss_base[-20:]) / 20
    symp_final = sum(loss_symp[-20:]) / 20
    dmd_final  = sum(loss_dmd[-20:]) / 20
    
    improv_symp = (base_final - symp_final) / base_final * 100
    improv_dmd  = (base_final - dmd_final) / base_final * 100
    
    print("\n" + "="*50)
    print(f"RESULTS (Structured Grammar)")
    print("="*50)
    print(f"Baseline Final Loss:   {base_final:.6f}")
    print(f"Symplectic Final Loss: {symp_final:.6f} (Imp: {improv_symp:+.2f}%)")
    print(f"DMD Final Loss:        {dmd_final:.6f} (Imp: {improv_dmd:+.2f}%)")
    print("="*50)
    
    best_strategy = "Baseline"
    if symp_final < base_final and symp_final < dmd_final:
        best_strategy = "Symplectic"
    elif dmd_final < base_final:
        best_strategy = "DMD"
        
    print(f"\nBest Strategy: {best_strategy}")
