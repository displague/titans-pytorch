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

    # initial page is 0
    assert mem.active_page_index.item() == 0

    # first forward should switch to page 1
    _ = mem(x)
    assert mem.active_page_index.item() == 1

    # second forward should switch to page 2
    _ = mem(x)
    assert mem.active_page_index.item() == 2

    # retrieved shape always matches input
    retrieved, _ = mem(x)
    assert retrieved.shape == x.shape
