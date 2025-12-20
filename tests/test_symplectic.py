import torch
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
    
    # Check range (sigmoid output)
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
