import numpy as np

def gradient(W):
    # Compute forward finite differences along both dimensions
    grad_x = np.roll(W, -1, axis=0) - W
    grad_y = np.roll(W, -1, axis=1) - W
    return grad_x, grad_y

def divergence(p_x, p_y):
    # Compute the divergence (negative adjoint of the gradient)
    div_x = p_x - np.roll(p_x, 1, axis=0)
    div_y = p_y - np.roll(p_y, 1, axis=1)
    return div_x + div_y

def project_dual(p_x, p_y):
    # Project each gradient vector onto the L2 unit ball
    norm = np.maximum(1, np.sqrt(p_x**2 + p_y**2))
    return p_x / norm, p_y / norm

def enforce_normalization(W):
    return W / np.sum(W)

def chambolle_pock(Y, lam, tau, sigma, theta, max_iter=100):
    # Initialize
    W = Y.copy()      # initial guess: measured (corrected) data
    W_bar = W.copy()
    p_x = np.zeros_like(W)
    p_y = np.zeros_like(W)
    
    for k in range(max_iter):
        # Dual update:
        grad_W_bar_x, grad_W_bar_y = gradient(W_bar)
        p_x = p_x + sigma * grad_W_bar_x
        p_y = p_y + sigma * grad_W_bar_y
        p_x, p_y = project_dual(p_x, p_y)
        
        # Primal update:
        # Compute divergence of dual variables
        div_p = divergence(p_x, p_y)
        W_old = W.copy()
        W = W - tau * (W - Y + lam * div_p)
        # Enforce normalization constraint:
        W = enforce_normalization(W)
        
        # Extrapolation:
        W_bar = W + theta * (W - W_old)
    
    return W

# Example usage:
# Y is the affine-corrected measured Wigner function (a 2D numpy array).
# Choose parameters (tau, sigma, theta, lam) according to the problem scaling.
denoised_W = chambolle_pock(Y, lam=0.1, tau=0.25, sigma=0.25, theta=1.0, max_iter=200)
