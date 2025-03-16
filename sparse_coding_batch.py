import scipy.io as sio
import time
import torch
from visualizer import display_network

def sc_l1(X, A, S, sparsity, max_norm2, max_iters, max_inner_iters):
    """
    Performs sparse coding via L1 regularization using block coordinate descent.

    The algorithm alternates between updating the sparse codes S (via coordinate descent)
    and the dictionary A (via block coordinate descent).

    Parameters
    ----------
    X : torch.Tensor
        Data matrix of shape (L, N).
    A : torch.Tensor
        Dictionary matrix of shape (L, M) that will be updated in-place.
    S : torch.Tensor
        Sparse code matrix of shape (M, N) that will be updated in-place.
    sparsity : float
        Regularization parameter for the L1 norm.
    max_norm2 : float
        (Unused; kept for compatibility).
    max_iters : int
        Number of outer iterations.
    max_inner_iters : int
        Number of inner iterations for both S and A updates.

    Returns
    -------
    A : torch.Tensor
        Learned dictionary matrix (updated in-place from initA).
    S : torch.Tensor
        Learned sparse codes (updated in-place from initS).
    E : list
        Energy (cost) history for each outer iteration.
    """
    # Use the input tensors directly for in-place updates.
    L, N = X.shape
    M = A.shape[1]
    E = []  # List to store energy at each iteration

    # Define soft-thresholding operator for sparse coding update
    def soft_threshold(u, t):
        return torch.sign(u) * torch.clamp(torch.abs(u) - t, min=0)

    for iter in range(max_iters):
        # -------------------------------
        # Update S via coordinate descent
        # -------------------------------
        AtA = A.t() @ A              # Compute A^T * A, shape (M, M)
        diagAtA = torch.diag(AtA).clone()  # Extract diagonal elements
        AtA.fill_diagonal_(0.0)      # Zero out diagonal
        
        AtX = A.t() @ X              # Compute A^T * X, shape (M, N)
        
        # Perform inner iterations for updating S
        for _ in range(max_inner_iters):
            for i in range(M):
                u = AtX[i, :] - torch.matmul(AtA[i, :], S)
                # Check if diagonal element is sufficiently nonzero
                if diagAtA[i].abs() > 1e-8:
                    new_val = soft_threshold(u, sparsity) / diagAtA[i]
                else:
                    new_val = soft_threshold(u, sparsity)
                # In-place update of the i-th row of S
                S[i, :].copy_(new_val)
        
        # -------------------------------
        # Update A via block coordinate descent
        # -------------------------------
        XSt = X @ S.t()  # Compute X * S^T, shape (L, M)
        SSt = S @ S.t()  # Compute S * S^T, shape (M, M)
        diagSSt = torch.diag(SSt).clone()  # Diagonal of SSt
        SSt = SSt - torch.diag(torch.diag(SSt))  # Zero out diagonal
        
        # Perform inner iterations for updating A
        for _ in range(max_inner_iters):
            for i in range(M):
                if diagSSt[i].abs() < 1e-8:
                    continue  # Skip update if denominator is near zero
                a = XSt[:, i] - A @ SSt[:, i]
                norm_a = torch.norm(a)
                # Avoid division by a too small number
                denom = max(norm_a.item(), 1 / diagSSt[i].item())
                # In-place update of the i-th column of A
                A[:, i].copy_(a / denom)
        
        # -------------------------------
        # Compute the energy (objective function)
        # -------------------------------
        reconstruction_error = X - A @ S
        energy = 0.5 * torch.sum(reconstruction_error ** 2) + sparsity * torch.sum(torch.abs(S))
        E.append(energy.item())
        print(f"iter={iter+1}, E={energy.item():.4f}")
    
    return A, S, E


def run_demo():
    """
    Demonstrates sparse coding by loading data, running the sc_l1 algorithm,
    and displaying the learned dictionary features.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data from .mat file and move to the selected device
    mat_contents = sio.loadmat('data.mat')
    X = torch.tensor(mat_contents['X'], dtype=torch.float32).to(device)
    L, N = X.shape
    M = 169  # Number of dictionary atoms

    # Set seed for reproducibility and generate dummy initializations
    torch.manual_seed(0)
    A = torch.randn([L, M], dtype=torch.float32, device=device)
    S = torch.randn([M, N], dtype=torch.float32, device=device)
    sparsity = 0.1
    max_norm2 = 1.0  # Unused parameter (kept for compatibility)
    max_iters = 150
    max_inner_iters = 10

    # Run the sparse coding algorithm and time its execution
    start_time = time.time()
    A, S, E = sc_l1(X, A, S, sparsity, max_norm2, max_iters, max_inner_iters)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    # Display the learned dictionary features
    display_network(A.cpu())


# Example usage:
if __name__ == "__main__":
    run_demo()
