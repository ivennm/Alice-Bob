import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.linalg import sqrtm

# -------------------------------
# Helper functions
# -------------------------------

def load_pickle(filename):
    """Load a pickle file containing a numpy array."""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def estimate_affine_params(W_measured, edge_fraction=0.1):
    """Estimate the background offset b and scale factor a.

    Parameters:
      - W_measured: 2D numpy array of the measured Wigner function.
      - edge_fraction: fraction of the edge (from each side) used to estimate b.

    Returns:
      - W_corrected: The affine-corrected Wigner function.
      - a_est: estimated scale factor.
      - b_est: estimated background offset.
    """
    h, w = W_measured.shape
    edge_h = int(np.ceil(h * edge_fraction))
    edge_w = int(np.ceil(w * edge_fraction))
    
    # Use the top, bottom, left, and right edges
    top_edge = W_measured[:edge_h, :]
    bottom_edge = W_measured[-edge_h:, :]
    left_edge = W_measured[:, :edge_w]
    right_edge = W_measured[:, -edge_w:]
    
    # Concatenate the edge pixels (flattened) and estimate background via median
    edges = np.concatenate((top_edge.flatten(), bottom_edge.flatten(),
                            left_edge.flatten(), right_edge.flatten()))
    b_est = np.median(edges)
    
    # Subtract b_est and compute scaling a_est based on normalization:
    W_sub = W_measured - b_est
    a_est = np.sum(W_sub)
    
    # Correct the measured Wigner function:
    W_corrected = W_sub / a_est
    return W_corrected, a_est, b_est

def apply_gaussian_filter(W, sigma):
    """Apply a 2D Gaussian filter."""
    return gaussian_filter(W, sigma=sigma)

def reconstruct_density_matrix(W):
    """
    Placeholder: Reconstruct the density matrix from the Wigner function.
    
    Note: In your pipeline, replace this with your reconstruction method 
          from the previous task.
    """
    # For illustration, we assume that the Wigner function is a proxy for 
    # the diagonal of the density matrix in some basis, and we simply use 
    # it to create a density matrix (this is not physically rigorous).
    # In practice, use the appropriate inversion (e.g. the inverse Radon transform).
    rho = np.outer(W.flatten(), W.flatten())
    # Normalize the density matrix:
    rho = rho / np.trace(rho)
    return rho

def quantum_fidelity(rho_ref, rho_est):
    """
    Compute the quantum fidelity between two density matrices using
    F = [Tr(sqrt(sqrt(rho_ref) * rho_est * sqrt(rho_ref)))]^2.
    """
    sqrt_rho_ref = sqrtm(rho_ref)
    product = sqrt_rho_ref @ rho_est @ sqrt_rho_ref
    sqrt_product = sqrtm(product)
    fid = np.real(np.trace(sqrt_product)) ** 2
    return fid

# -------------------------------
# Main pipeline
# -------------------------------

# Example filenames (replace these with your actual file paths)
noisy_wigner_files = ['noisy_wigner_0.pickle', 
                      'noisy_wigner_1.pickle', 
                      'noisy_wigner_2.pickle']
# For benchmarking you would have corresponding clean quantum states;
quantum_state_files = ['quantum_state_0.pickle', 
                       'quantum_state_1.pickle', 
                       'quantum_state_2.pickle']

# Store fidelity results for different sigma values
sigma_values = np.linspace(0.5, 5.0, 10)
fidelities_raw = []
fidelities_denoised = []

# Loop over test cases (here we simply demonstrate for each file pair)
for wigner_file, state_file in zip(noisy_wigner_files, quantum_state_files):
    
    # Load measured Wigner function (assumed 2D array)
    W_measured = load_pickle(wigner_file)
    
    # Load the reference (clean) quantum state; assume density matrix format
    rho_ref = load_pickle(state_file)
    
    # Affine correction:
    W_corr, a_est, b_est = estimate_affine_params(W_measured)
    
    # Reconstruction from the corrected (raw) Wigner function:
    rho_est_raw = reconstruct_density_matrix(W_corr)
    fid_raw = quantum_fidelity(rho_ref, rho_est_raw)
    
    # Now vary sigma and measure fidelity after denoising:
    fid_sigma = []
    for sigma in sigma_values:
        W_denoised = apply_gaussian_filter(W_corr, sigma=sigma)
        rho_est_denoised = reconstruct_density_matrix(W_denoised)
        fid = quantum_fidelity(rho_ref, rho_est_denoised)
        fid_sigma.append(fid)
    
    fidelities_raw.append(fid_raw)
    fidelities_denoised.append(fid_sigma)
    
    # Plotting the effect of sigma for this test case:
    plt.figure()
    plt.plot(sigma_values, fid_sigma, 'o-', label='Fidelity after denoising')
    plt.axhline(fid_raw, color='r', linestyle='--', label='Fidelity without denoising')
    plt.xlabel('Gaussian filter width (σ)')
    plt.ylabel('Quantum Fidelity')
    plt.title(f'Fidelity vs. Filter Width for {wigner_file}')
    plt.legend()
    plt.show()

# -------------------------------
# Experimental Test Case
# -------------------------------

# You would similarly load experimental Wigner data, correct (and optionally denoise),
# and then visually compare the before and after.
# Example (using a placeholder filename):
experimental_wigner_file = 'noisy_wigner_0.pickle'  # replace with your file name
W_experimental = load_pickle(experimental_wigner_file)

# Apply correction:
W_exp_corr, _, _ = estimate_affine_params(W_experimental)

# Optionally denoise with a chosen sigma based on previous benchmarking
sigma_exp = 2.0  # you may choose this based on your fidelity curves
W_exp_denoised = apply_gaussian_filter(W_exp_corr, sigma=sigma_exp)

# Plot side-by-side comparison:
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(W_exp_corr, cmap='viridis')
plt.title('Experimental Wigner (Corrected)')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(W_exp_denoised, cmap='viridis')
plt.title('Experimental Wigner (Denoised)')
plt.colorbar()
plt.tight_layout()
plt.show()
