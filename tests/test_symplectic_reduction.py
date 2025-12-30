import torch
import torch.nn.functional as F
from titans_pytorch.neural_memory import NeuralMemory
from titans_pytorch.symplectic_gate import SymplecticGating

def test_symplectic_gating_logic():
    """
    Verify that SymplecticGating correctly detects twists (Moment Map norm).
    """
    dim = 4
    gate = SymplecticGating(dim)
    
    # We force the projections to be identity for predictable behavior
    # to_twist_q(x) = x, to_twist_k(x) = x
    # This might not be possible directly since they are Linear layers, 
    # but we can set weights.
    with torch.no_grad():
        gate.to_twist_q.weight.copy_(torch.eye(dim))
        gate.to_twist_k.weight.copy_(torch.eye(dim))
    
    # CASE 1: Rational / Linear (No twist)
    # q = k. q_curr = k_curr, q_prev = k_prev
    # q_curr * k_prev - q_prev * k_curr = q_c * q_p - q_p * q_c = 0
    seq_linear = torch.randn(1, 10, dim)
    seq_linear = F.normalize(seq_linear, dim=-1) # Pre-normalize to be safe, though layer does it too
    
    print(f"Testing Linear Sequence...")
    complexity, moment_map = gate(seq_linear, return_moment_map=True)
    
    print(f"Linear Max Moment Map: {moment_map.abs().max().item()}")
    assert moment_map.abs().max().item() < 1e-5, "Linear sequence should have near-zero moment map"
    
    # CASE 2: Twist (Loop)
    # We need q_curr * k_prev != q_prev * k_curr
    # Let q = k roughly (since weights are same)
    # seq = [e1, e2, e1, e2]
    # q1=e1, q2=e2
    # q2 * k1 = e2 * e1 = 0
    # q1 * k2 = e1 * e2 = 0
    # Wait, elementwise product? 
    # The code says: (q_curr * k_prev) - (q_prev * k_curr)
    # This is elementwise. 
    # If q=k, then q_curr*k_prev = q_curr*q_prev. 
    # q_prev*k_curr = q_prev*q_curr.
    # They are equal (elementwise mult is commutative).
    # So if to_twist_q == to_twist_k, then twist is ALWAYS 0?
    # Let's check the code:
    # twist = (q_curr * k_prev) - (q_prev * k_curr)
    # Yes. If Q and K map to the same space, the symplectic form represented this way is 0???
    # Ah, standard symplectic form on R^2n is \sum dq \wedge dp.
    # If we map x -> q and x -> k differently, then it works.
    # So we must ENSURE weights are DIFFERENT.
    
    with torch.no_grad():
        gate.to_twist_q.weight.copy_(torch.eye(dim))
        # Make K a permutation or rotation
        rot = torch.zeros(dim, dim)
        # Shift: 1->2, 2->3...
        for i in range(dim):
            rot[i, (i+1)%dim] = 1.0
        gate.to_twist_k.weight.copy_(rot)
        
    # Now q(x) = x, k(x) = rot(x)
    # Try x1 = e1, x2 = e2
    # q1 = e1, k1 = e2
    # q2 = e2, k2 = e3
    # term1 = q2 * k1 = e2 * e2 = e2 (nonzero at index 1)
    # term2 = q1 * k2 = e1 * e3 = 0
    # twist = e2 - 0 != 0.
    
    print(f"Testing Twisted Sequence (with distinct Q/K projections)...")
    
    # Construct sequence e1, e2, ...
    seq_twist = torch.zeros(1, 5, dim)
    seq_twist[0, 0, 0] = 1.0 # e1
    seq_twist[0, 1, 1] = 1.0 # e2
    seq_twist[0, 2, 2] = 1.0 # e3
    
    complexity, moment_map = gate(seq_twist, return_moment_map=True)
    max_moment = moment_map.abs().max().item()
    print(f"Twist Max Moment Map: {max_moment}")
    
    assert max_moment > 0.1, "Twisted sequence should have significant moment map"


def test_per_sample_paging():
    """
    Verify that NeuralMemory switches pages ONLY for samples that exceed threshold.
    """
    dim = 4
    num_pages = 2
    heads = 4
    mem = NeuralMemory(
        dim=dim,
        chunk_size=2, # Small chunk size
        num_pages=num_pages,
        use_symplectic_gating=True,
        heads=heads,
        symplectic_page_threshold=0.5
    )
    
    # Mock the symplectic gate to return controlled complexity using a hook or just patching
    # Easier: Patch the gate
    
    original_gate = mem.symplectic_gate
    
    class MockGate(torch.nn.Module):
        def forward(self, x):
            # x is (B, N, D)
            # We want Sample 0 to be LOW complexity
            # Sample 1 to be HIGH complexity
            
            B, N, D = x.shape
            complexity = torch.zeros(B, N, 1)
            
            # Sample 1 -> High complexity > 0.5
            if B > 1:
                complexity[1, :, :] = 0.9
                
            return complexity
            
    mem.symplectic_gate = MockGate()
    
    # Create input: Batch 2
    bs = 2
    seq_len = 4
    seq = torch.randn(bs, seq_len, dim)
    
    # Check init state logic implicitly by running forward
    # State is None initially
    
    print("Running NeuralMemory forward...")
    retrieved, next_state = mem(seq)
    
    # Check active_page_indices in next_state
    active_pages = next_state.active_page_indices
    print(f"Active Pages: {active_pages}")
    
    assert active_pages is not None
    assert active_pages[0].item() == 0, "Sample 0 should stay on Page 0 (Low complexity)"
    assert active_pages[1].item() == 1, "Sample 1 should switch to Page 1 (High complexity)"
    
    # Verify State Persistence
    # Run again with same state
    print("Running Step 2 (Persistence)...")
    # Low complexity for both this time to see if they stay (page switch is mod num_pages, but here it keeps state)
    # Logic: if complexity > threshold, switch. If NOT, keep same.
    
    # Set mock to return 0 for all
    class MockGateZero(torch.nn.Module):
         def forward(self, x):
             return torch.zeros(x.shape[0], x.shape[1], 1)
    
    mem.symplectic_gate = MockGateZero()
    
    retrieved_2, next_state_2 = mem(seq, state=next_state)
    active_pages_2 = next_state_2.active_page_indices
    print(f"Active Pages Step 2: {active_pages_2}")
    
    assert active_pages_2[0].item() == 0, "Sample 0 should stay on Page 0"
    assert active_pages_2[1].item() == 1, "Sample 1 should stay on Page 1 (No new twist)"

    print("Per-Sample Paging Verified.")

if __name__ == "__main__":
    test_symplectic_gating_logic()
    test_per_sample_paging()
    print("All tests passed.")
