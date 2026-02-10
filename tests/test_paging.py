import torch
from titans_pytorch.neural_memory import NeuralMemory


def test_objective_reduction_page_switch():
    dim = 32
    seq_len = 64

    mem = NeuralMemory(
        dim = dim,
        chunk_size = 8,
        use_symplectic_gating = True,
        num_pages = 3
    )

    # make it easy to trigger a page switch
    mem.symplectic_page_threshold = 0.05

    x = torch.randn(1, seq_len, dim)

    # Force a switch on every forward by lowering threshold.
    mem.symplectic_page_threshold = -1.0

    state = None
    # first forward should switch to page 1
    _, state = mem(x, state = state)
    assert state.active_page_indices.item() == 1

    # second forward should switch to page 2
    _, state = mem(x, state = state)
    assert state.active_page_indices.item() == 2

    # retrieved shape always matches input
    retrieved, _ = mem(x, state = state)
    assert retrieved.shape == x.shape
