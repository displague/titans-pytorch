import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicModeDecomposition(nn.Module):
    """
    Dynamic Mode Decomposition (DMD) Module.
    
    Extracts spectral properties (modes, eigenvalues) from a sequence of data snapshots.
    Can be used to measure how well the sequence is approximated by a linear dynamical system.
    
    High reconstruction error suggests non-linear dynamics (Twists/Surprises).
    """
    
    def __init__(self, rank: int | None = None):
        """
        Args:
            rank: Truncation rank for SVD. If None, uses full rank (min(seq_len, dim)).
        """
        super().__init__()
        self.rank = rank
        
    def forward(self, x: torch.Tensor, return_eigenvalues: bool = False):
        """
        Args:
            x: Input tensor of shape (Batch, SeqLen, Dim).
            return_eigenvalues: If True, returns the eigenvalues of the Koopman operator.
            
        Returns:
            reconstruction_error: (Batch, 1) or (Batch,) - Mean Squared Error of the linear approximation.
            eigenvalues (optional): (Batch, Rank) - Complex eigenvalues.
        """
        
        # 1. Prepare Data Matrices
        # Snapshots: x_0, x_1, ..., x_m
        # X = [x_0, ..., x_{m-1}]
        # Y = [x_1, ..., x_m]
        
        # x is (B, L, D). We usually treat 'D' as the state dimension and 'L' as time snapshots.
        # So we permute to (B, D, L) for the standard math notation X = (D, L).
        
        x_t = x.transpose(1, 2) # (B, D, L)
        
        X = x_t[..., :-1] # (B, D, L-1)
        Y = x_t[..., 1:]  # (B, D, L-1)
        
        B, D, T = X.shape
        
        # 2. SVD of X
        # X = U Sigma V*
        # torch.linalg.svd returns U, S, Vh
        # U: (B, D, D), S: (B, min(D,T)), Vh: (B, T, T)
        
        # We assume full matrices=False usually, but let's stick to default which is efficient.
        try:
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        except RuntimeError:
            # SVD convergence failure backup (rare but possible with mixed precision or NaNs)
            return torch.zeros(B, device=x.device), (torch.zeros(B, 1, device=x.device) if return_eigenvalues else None)
        
        # 3. Truncate Rank
        r = self.rank if self.rank is not None else min(D, T)
        
        U_r = U[..., :r]  # (B, D, r)
        S_r = S[..., :r]  # (B, r)
        Vh_r = Vh[..., :r, :] # (B, r, L-1) - Note: Vh is (r, L-1) for full_matrices=False
        
        # S_r is a vector, we need diagonal matrix inverse.
        # 1/S_r. Avoid div by zero.
        S_inv = torch.diag_embed(1.0 / (S_r + 1e-6)) # (B, r, r)
        
        # 4. Compute Operator A_tilde (Projection of A onto POD modes)
        # A_tilde = U* Y V Sigma^{-1}
        # Shapes: (B,r,D) @ (B,D,T) @ (B,T,r) @ (B,r,r) -> (B, r, r)
        
        # Step 1: U* Y
        # U_r.mT is (B, r, D)
        step1 = torch.matmul(U_r.transpose(-2, -1), Y) # (B, r, T)
        
        # Step 2: * V
        # V is Vh.mT -> (B, T, r)
        step2 = torch.matmul(step1, Vh_r.transpose(-2, -1)) # (B, r, r)
        
        # Step 3: * Sigma^{-1}
        A_tilde = torch.matmul(step2, S_inv) # (B, r, r)
        
        # 5. Eigendecomposition of A_tilde
        # A_tilde W = W Lambda
        eigvals, W = torch.linalg.eig(A_tilde) # eigvals: (B, r), W: (B, r, r)
        
        # 6. Reconstruct High-Dimensional Modes (Phi)
        # Phi = Y V Sigma^{-1} W  (Exact DMD)
        # or Phi = U W (Projected DMD)
        # We usually care about the reconstruction error.
        
        # Let's compute the predicted future state:
        # x_k = Phi Lambda^k b
        # But efficiently:
        # X_approx = U A_tilde U* X (Approximation on the subspace)
        
        # Wait, the error metric for "Linearity" is usually: |Y - A X| or |X_next - A_approx X_prev|
        # Ideally, we reconstruction X using the modes.
        
        # Simple Proxy for Error: 
        # Linearity Error = || Y - U A_tilde U* X || 
        # Because we projected dynamics onto U.
        
        # However, for gating, we want to know if the data *fits* the low-rank linear model.
        # So we reconstruct Y_pred = U * A_tilde * U^H * X (Project X to latent, step fwd, project back)
        
        # Latent X: x_tilde = U^H * X   (B, r, T)
        # Latent Y_pred = A_tilde * x_tilde  (B, r, T)
        # Y_pred = U * Latent Y_pred     (B, D, T)
        
        x_tilde = torch.matmul(U_r.transpose(-2, -1), X) # (B, r, T)
        y_tilde_pred = torch.matmul(A_tilde, x_tilde)    # (B, r, T)
        Y_pred = torch.matmul(U_r, y_tilde_pred)         # (B, D, T)
        
        # If input x was real, we expect Y_pred to be real (conjugate modes cancel out).
        # Any imaginary part is numerical error or imperfect truncation.
        if not x.is_complex():
            Y_pred = Y_pred.real
        
        # Compute MSE Error
        # Y is (B, D, T). Y_pred is (B, D, T).
        # We normalize by the norm of Y to get relative error?
        # Absolute error might be better for thresholding? 
        # Let's return Mean Squared Error.
        
        diff = Y - Y_pred
        mse = torch.mean(diff.abs() ** 2, dim=(-1, -2)) # (B,)
        
        if return_eigenvalues:
            return mse, eigvals
            
        return mse
