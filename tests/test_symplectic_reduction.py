import torch
import torch.nn as nn
import torch.nn.functional as F
from titans_pytorch.neural_memory import NeuralMemory, NeuralMemState
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


def test_inactive_pages_receive_no_updates():
    dim = 4
    mem = NeuralMemory(
        dim = dim,
        chunk_size = 2,
        num_pages = 2,
        heads = 1,
        use_symplectic_gating = True,
        symplectic_page_threshold = 0.1
    )

    class MockGateZero(nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], x.shape[1], 1)

    mem.symplectic_gate = MockGateZero()

    init_weights = mem.init_weights(batch = 1)
    state = NeuralMemState(0, init_weights, None, None, None, torch.tensor([1]))

    seq = torch.randn(1, 2, dim)
    _, next_state = mem(seq, state = state)

    assert next_state.active_page_indices.item() == 1

    updates = next_state.updates
    assert updates is not None

    for name, t in updates.items():
        t = t.reshape(1, mem.internal_heads, *t.shape[1:])
        inactive = 0
        init = init_weights[name].reshape(1, mem.internal_heads, *init_weights[name].shape[1:])
        assert torch.allclose(t[:, inactive, 0], init[:, inactive], atol = 1e-6)

def test_paging_stress_state_carry_isolates_pages():
    dim = 8
    mem = NeuralMemory(
        dim = dim,
        chunk_size = 2,
        num_pages = 3,
        heads = 1,
        momentum = False,
        use_symplectic_gating = True,
        symplectic_page_threshold = 0.5
    )

    class ScriptedGate(nn.Module):
        def __init__(self, values):
            super().__init__()
            self.values = values
            self.index = 0

        def forward(self, x):
            v = self.values[min(self.index, len(self.values) - 1)]
            self.index += 1
            return torch.full((x.shape[0], x.shape[1], 1), v, device = x.device)

    # Page transitions from initial page 0:
    # high -> 1, low -> 1, high -> 2, high -> 0, low -> 0
    scripted = ScriptedGate([0.9, 0.0, 0.9, 0.9, 0.0])
    mem.symplectic_gate = scripted

    seq = torch.randn(1, 2, dim)
    state = None
    observed_pages = []
    page1_snapshot = None

    for step in range(5):
        _, state = mem(seq, state = state)
        page = state.active_page_indices.item()
        observed_pages.append(page)

        last_update, _ = state.states
        if step == 1:
            page1_snapshot = {name: t[1].detach().clone() for name, t in last_update.items()}
        elif step > 1 and page != 1:
            for name, t in last_update.items():
                assert torch.allclose(t[1], page1_snapshot[name], atol = 1e-6), f"Page 1 drifted while inactive for {name}"

    assert observed_pages == [1, 1, 2, 0, 0]

def test_manifold_state_keyed_paging_routes_by_angle_bucket():
    dim = 8
    num_pages = 3
    mem = NeuralMemory(
        dim = dim,
        chunk_size = 2,
        num_pages = num_pages,
        heads = 1,
        momentum = False,
        use_symplectic_gating = True,
        manifold_state_keyed_paging = True,
        symplectic_page_threshold = 0.1
    )

    class MockGateWithState(nn.Module):
        def forward(self, x, return_manifold_state = False):
            batch, seq_len, _ = x.shape
            complexity = torch.ones(batch, seq_len, 1, device = x.device) * 0.9
            if not return_manifold_state:
                return complexity

            angles = torch.tensor(
                [-0.9 * torch.pi, 0.0, 0.9 * torch.pi],
                device = x.device,
                dtype = x.dtype
            )[:batch]
            angles = angles.view(batch, 1, 1).expand(batch, seq_len, 1)
            radius = torch.ones_like(angles)
            manifold_state = dict(phase_angle = angles, phase_radius = radius)
            return complexity, manifold_state

    mem.symplectic_gate = MockGateWithState()

    seq = torch.randn(3, 2, dim)
    _, state = mem(seq)

    assert state.active_page_indices.tolist() == [0, 1, 2]

def test_hierarchical_paging_routes_with_coarse_and_fine_keys():
    dim = 8
    mem = NeuralMemory(
        dim = dim,
        chunk_size = 2,
        num_pages = 4,
        heads = 1,
        momentum = False,
        use_symplectic_gating = True,
        hierarchical_paging = True,
        coarse_pages = 2,
        fine_pages = 2,
        hierarchy_mix = 1.0,
        symplectic_page_threshold = 0.1
    )

    class MockHierarchicalGate(nn.Module):
        def forward(self, x, return_manifold_state = False):
            batch, seq_len, _ = x.shape
            values = torch.tensor([0.2, 0.8], device = x.device, dtype = x.dtype)[:batch]
            complexity = values.view(batch, 1, 1).expand(batch, seq_len, 1)
            if not return_manifold_state:
                return complexity

            angles = torch.tensor(
                [0.75 * torch.pi, -0.75 * torch.pi],
                device = x.device,
                dtype = x.dtype
            )[:batch]
            angles = angles.view(batch, 1, 1).expand(batch, seq_len, 1)
            manifold_state = dict(
                phase_angle = angles,
                phase_radius = torch.ones_like(angles)
            )
            return complexity, manifold_state

    mem.symplectic_gate = MockHierarchicalGate()

    seq = torch.randn(2, 2, dim)
    _, state = mem(seq)

    # coarse=0 with fine=1 -> page 1 ; coarse=1 with fine=0 -> page 2
    assert state.active_page_indices.tolist() == [1, 2]


if __name__ == "__main__":
    test_symplectic_gating_logic()
    test_per_sample_paging()
    print("All tests passed.")
