
import torch
import numpy as np
from titans_pytorch.dmd import DynamicModeDecomposition

def test_dmd_linear_system():
    """
    Test DMD on a perfect linear system: x_{k+1} = A x_k
    """
    dim = 16
    seq_len = 50
    batch_size = 2
    
    dmd = DynamicModeDecomposition(rank=16)
    
    # Create a linear system matrix A
    # Rotation + Decay
    # We construct A with known eigenvalues
    
    # Simple: Just use a random matrix
    # Stable Linear System Construction
    # A = Q D Q^T where D has entries <= 1
    # Random orthogonal matrix Q
    X_rand = torch.randn(dim, dim)
    U, _, Vh = torch.linalg.svd(X_rand)
    Q = U
    
    # Eigenvalues: Random phases with magnitude <= 1.0 (Marginally stable or decaying)
    # To be real-valued A, we need conjugate pairs or just real eigenvalues.
    # Let's simple use a symmetric matrix with real eigenvalues in [-0.9, 0.9]
    D = torch.diag(torch.rand(dim) * 1.8 - 0.9)
    A = torch.matmul(torch.matmul(Q, D), Q.T)

    
    # Generate trajectories
    x = torch.zeros(batch_size, seq_len, dim)
    x0 = torch.randn(batch_size, dim)
    x[:, 0, :] = x0
    
    for t in range(1, seq_len):
        x[:, t, :] = torch.matmul(x[:, t-1, :], A.T)
        
    # DMD should reconstruct this perfectly (machine precision)
    mse = dmd(x)
    
    print(f"Linear System MSE: {mse.mean().item()}")
    assert mse.mean().item() < 1e-5, "DMD should perfectly reconstruct a linear system"

def test_dmd_nonlinear_system():
    """
    Test DMD on a non-linear system. Error should be higher.
    """
    dim = 2
    seq_len = 50
    batch_size = 1
    
    dmd = DynamicModeDecomposition(rank=2)
    
    # Non-linear: x_{k+1} = x_k^2 (Logistic map style, or component-wise square)
    # Actually just sin(t) is linear in lifted space but non-linear in state?
    # No, sin(wt) is linear (rotation).
    
    # Let's use a step discontinuity or chaos. 
    # Or just random noise.
    
    x = torch.randn(batch_size, seq_len, dim)
    
    mse = dmd(x)
    print(f"Random/Non-linear System MSE: {mse.mean().item()}")
    
    assert mse.mean().item() > 1e-2, "DMD should have non-trivial error on random noise"

def test_eigenvalues_return():
    dim = 4
    seq_len = 20
    dmd = DynamicModeDecomposition(rank=4)
    x = torch.randn(1, seq_len, dim)
    
    mse, eigvals = dmd(x, return_eigenvalues=True)
    assert eigvals.shape[-1] == 4
    assert eigvals.is_complex()

if __name__ == "__main__":
    test_dmd_linear_system()
    test_dmd_nonlinear_system()
    test_eigenvalues_return()
    print("All DMD tests passed.")
