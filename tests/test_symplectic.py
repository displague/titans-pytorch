import torch
import pytest
from titans_pytorch.symplectic_gate import SymplecticGating
from titans_pytorch.neural_memory import NeuralMemory

def test_symplectic_gate_shapes():
    dim = 64
    seq_len = 128
    batch = 2
    
    gate = SymplecticGating(dim)
    x = torch.randn(batch, seq_len, dim)
    
    complexity = gate(x)
    
    # Check output shape: (batch, seq, 1)
    assert complexity.shape == (batch, seq_len, 1)
    
    # Check range (tanh output)
    assert (complexity >= 0).all() and (complexity <= 1).all()

def test_neural_memory_integration():
    dim = 16
    mem = NeuralMemory(
        dim = dim,
        chunk_size = 2,
        use_symplectic_gating = True
    )
    
    seq_len = 16
    x = torch.randn(1, seq_len, dim)
    
    retrieved, _ = mem(x)
    loss = retrieved.sum()
    loss.backward()
    
    # If backward passed, graph is connected.
    assert True

def test_symplectic_gate_constant_sequence_zero():
    dim = 8
    seq_len = 16

    gate = SymplecticGating(dim)

    with torch.no_grad():
        gate.to_twist_q.weight.copy_(torch.eye(dim))
        rot = torch.zeros(dim, dim)
        for i in range(dim):
            rot[i, (i + 1) % dim] = 1.0
        gate.to_twist_k.weight.copy_(rot)

    seq = torch.zeros(1, seq_len, dim)
    seq[..., 0] = 1.0

    complexity, moment_map = gate(seq, return_moment_map = True)

    assert moment_map.abs().max().item() < 1e-6
    assert complexity.abs().max().item() < 1e-6

def test_symplectic_gate_twist_higher_than_linear():
    dim = 4
    seq_len = 16

    gate = SymplecticGating(dim)

    with torch.no_grad():
        gate.to_twist_q.weight.copy_(torch.eye(dim))
        rot = torch.zeros(dim, dim)
        for i in range(dim):
            rot[i, (i + 1) % dim] = 1.0
        gate.to_twist_k.weight.copy_(rot)

    linear = torch.zeros(1, seq_len, dim)
    linear[..., 0] = 1.0

    twist = torch.zeros(1, seq_len, dim)
    for i in range(seq_len):
        twist[0, i, i % dim] = 1.0

    linear_mean = gate(linear).mean().item()
    twist_mean = gate(twist).mean().item()

    assert twist_mean > linear_mean + 0.05

def test_symplectic_gate_gated_diag_zeroes_when_gate_off():
    dim = 8
    seq_len = 16

    gate = SymplecticGating(dim, gated = True, diag = True, gate_threshold = 0.0)

    with torch.no_grad():
        gate.gate_weight.zero_()
        gate.gate_bias.fill_(-10.0)
        gate.mag_weight.fill_(1.0)
        gate.mag_bias.zero_()

    x = torch.randn(1, seq_len, dim)
    complexity, moment_map = gate(x, return_moment_map = True)

    assert moment_map.abs().max().item() < 1e-6
    assert complexity.abs().max().item() < 1e-6

def test_symplectic_gate_gated_diag_detects_twist():
    dim = 4
    seq_len = 16

    gate = SymplecticGating(dim, gated = True, diag = True, gate_threshold = 0.0)

    with torch.no_grad():
        gate.gate_weight.zero_()
        gate.gate_bias.fill_(10.0)
        gate.mag_weight.fill_(1.0)
        gate.mag_bias.zero_()
        gate.to_twist_q.weight.copy_(torch.eye(dim))
        rot = torch.zeros(dim, dim)
        for i in range(dim):
            rot[i, (i + 1) % dim] = 1.0
        gate.to_twist_k.weight.copy_(rot)

    twist = torch.zeros(1, seq_len, dim)
    for i in range(seq_len):
        twist[0, i, i % dim] = 1.0

    twist_mean = gate(twist).mean().item()

    assert twist_mean > 0.05

def test_symplectic_gate_soft_mode_leaks_when_gate_off():
    dim = 4
    seq_len = 8

    gate_hard = SymplecticGating(dim, gated = True, diag = True, gate_threshold = 0.0, gate_mode = "hard")
    gate_soft = SymplecticGating(dim, gated = True, diag = True, gate_threshold = 0.0, gate_mode = "soft")

    for gate in (gate_hard, gate_soft):
        with torch.no_grad():
            gate.gate_weight.zero_()
            gate.gate_bias.fill_(-10.0)
            gate.mag_weight.fill_(1.0)
            gate.mag_bias.zero_()

    x = torch.randn(1, seq_len, dim)

    hard_mean = gate_hard(x).mean().item()
    soft_mean = gate_soft(x).mean().item()

    assert hard_mean < 1e-6
    assert soft_mean > hard_mean

def test_symplectic_gate_topk_reduces_complexity():
    dim = 4
    seq_len = 16

    gate_full = SymplecticGating(dim, gated = True, diag = True, gate_threshold = 0.0, top_k = None)
    gate_topk = SymplecticGating(dim, gated = True, diag = True, gate_threshold = 0.0, top_k = 1)

    for gate in (gate_full, gate_topk):
        with torch.no_grad():
            gate.gate_weight.zero_()
            gate.gate_bias.fill_(10.0)
            gate.mag_weight.fill_(1.0)
            gate.mag_bias.zero_()
            gate.to_twist_q.weight.copy_(torch.eye(dim))
            rot = torch.zeros(dim, dim)
            for i in range(dim):
                rot[i, (i + 1) % dim] = 1.0
            gate.to_twist_k.weight.copy_(rot)

    twist = torch.zeros(1, seq_len, dim)
    for i in range(seq_len):
        twist[0, i, i % dim] = 1.0

    full_mean = gate_full(twist).mean().item()
    topk_mean = gate_topk(twist).mean().item()

    assert topk_mean < full_mean

def test_symplectic_gate_phase_map_shape():
    gate = SymplecticGating(8, phase_mix = 1.0, phase_pairs = 2)
    x = torch.randn(2, 12, 8)

    complexity, phase = gate(x, return_phase_map = True)

    assert complexity.shape == (2, 12, 1)
    assert phase.shape == (2, 12, 1)

def test_symplectic_gate_phase_mode_prefers_spiral():
    dim = 8
    seq_len = 32
    gate = SymplecticGating(dim, phase_mix = 1.0, phase_pairs = 2)

    with torch.no_grad():
        gate.to_phase.weight.zero_()
        gate.to_phase.weight[0, 0] = 1.
        gate.to_phase.weight[1, 1] = 1.
        gate.to_phase.weight[2, 2] = 1.
        gate.to_phase.weight[3, 3] = 1.

    linear = torch.zeros(1, seq_len, dim)
    linear[..., 0] = 1.0
    linear[..., 2] = 1.0

    t = torch.linspace(0, 2 * torch.pi, seq_len)
    spiral = torch.zeros(1, seq_len, dim)
    spiral[0, :, 0] = torch.cos(t)
    spiral[0, :, 1] = torch.sin(t)
    spiral[0, :, 2] = torch.cos(2 * t)
    spiral[0, :, 3] = torch.sin(2 * t)

    linear_mean = gate(linear).mean().item()
    spiral_mean = gate(spiral).mean().item()

    assert spiral_mean > linear_mean + 0.05

def test_neural_memory_symplectic_gate_kwargs():
    dim = 16
    mem = NeuralMemory(
        dim = dim,
        chunk_size = 2,
        use_symplectic_gating = True,
        symplectic_gate_kwargs = dict(
            gated = True,
            diag = True,
            gate_mode = "soft",
            phase_mix = 0.5,
            phase_pairs = 2
        )
    )

    x = torch.randn(1, 16, dim)
    retrieved, _ = mem(x)
    loss = retrieved.sum()
    loss.backward()

    assert retrieved.shape == x.shape

def test_symplectic_gate_adaptive_topk_varies_per_token():
    dim = 8
    gate = SymplecticGating(
        dim,
        gated = True,
        diag = True,
        adaptive_topk_ratio = 0.75,
        adaptive_topk_min_k = 1
    )

    with torch.no_grad():
        gate.gate_weight.fill_(1.0)
        gate.gate_bias.zero_()

    gate_pre = torch.zeros(1, 4, dim)
    gate_pre[0, 0, 0] = 0.05
    gate_pre[0, 1, :2] = 1.0
    gate_pre[0, 2, :4] = 2.0
    gate_pre[0, 3, :] = 4.0

    _, sparse_k = gate.compute_sparse_mask(gate_pre, return_k = True)

    assert sparse_k.shape == (1, 4, 1)
    assert sparse_k[0, 0, 0] <= sparse_k[0, 1, 0] <= sparse_k[0, 2, 0] <= sparse_k[0, 3, 0]

def test_symplectic_gate_extract_manifold_state_shapes():
    gate = SymplecticGating(16, phase_mix = 0.5, phase_pairs = 3)
    x = torch.randn(2, 10, 16)

    state = gate.extract_manifold_state(x)

    assert state is not None
    assert state["phase_radius"].shape == (2, 10, 3)
    assert state["phase_angle"].shape == (2, 10, 3)

def test_neural_memory_combined_symplectic_dmd():
    dim = 16
    mem = NeuralMemory(
        dim = dim,
        chunk_size = 2,
        use_symplectic_gating = True,
        use_dmd_gating = True,
        combine_symplectic_and_dmd = True,
        symplectic_gate_kwargs = dict(
            gated = True,
            gate_mode = "soft",
            phase_mix = 0.5,
            phase_pairs = 2
        )
    )

    x = torch.randn(1, 16, dim)
    retrieved, _ = mem(x)
    loss = retrieved.sum()
    loss.backward()

    assert retrieved.shape == x.shape

def test_symplectic_gate_quorum_policy_prefers_sustained_signal():
    gate = SymplecticGating(
        8,
        quorum_mix = 1.0,
        quorum_window = 3,
        quorum_threshold = 0.5,
        quorum_temperature = 0.05
    )

    complexity = torch.tensor([[[0.0], [0.0], [0.9], [0.0], [0.9], [0.9], [0.9]]])
    adjusted, quorum = gate.apply_quorum_policy(complexity)

    # Isolated spike is down-weighted relative to sustained high-complexity segment.
    assert quorum[0, 2, 0] < quorum[0, 5, 0]
    assert adjusted[0, 2, 0] < adjusted[0, 5, 0]

def test_symplectic_gate_quorum_budget_limits_positions():
    gate = SymplecticGating(
        8,
        quorum_mix = 1.0,
        quorum_window = 1,
        quorum_threshold = 0.0,
        quorum_temperature = 1.0,
        budget_topk_ratio = 0.25
    )

    complexity = torch.tensor([[[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8]]])
    adjusted, quorum, budget_k = gate.apply_quorum_policy(complexity, return_budget_k = True)

    assert budget_k == 2
    assert (quorum > 0).sum().item() == 2
    assert (adjusted > 0).sum().item() == 2

def test_symplectic_gate_quorum_budget_requires_quorum_mix():
    with pytest.raises(ValueError):
        SymplecticGating(8, budget_topk_ratio = 0.25)

def test_symplectic_gate_codebook_map_shape():
    gate = SymplecticGating(
        8,
        codebook_mix = 0.8,
        codebook_size = 6,
        codebook_temperature = 0.2,
        codebook_topk = 2
    )
    x = torch.randn(2, 12, 8)

    complexity, codebook = gate(x, return_codebook_map = True)

    assert complexity.shape == (2, 12, 1)
    assert codebook.shape == (2, 12, 1)
    assert (codebook >= 0).all() and (codebook <= 1).all()

def test_symplectic_gate_codebook_prefers_mixture_shifts():
    dim = 4
    seq_len = 32
    gate = SymplecticGating(
        dim,
        codebook_mix = 1.0,
        codebook_size = 2,
        codebook_temperature = 0.1
    )

    with torch.no_grad():
        gate.to_codebook.weight.zero_()
        gate.to_codebook.weight[0, 0] = 5.0
        gate.to_codebook.weight[1, 1] = 5.0

    stable = torch.zeros(1, seq_len, dim)
    stable[..., 0] = 1.0

    switching = torch.zeros(1, seq_len, dim)
    switching[0, ::2, 0] = 1.0
    switching[0, 1::2, 1] = 1.0

    stable_mean = gate(stable).mean().item()
    switching_mean = gate(switching).mean().item()

    assert switching_mean > stable_mean + 0.05
