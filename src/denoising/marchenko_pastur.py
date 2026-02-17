import numpy as np
import pandas as pd

from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize
from scipy.linalg import eigh

def get_mp_pdf(
        var : float,
        q : float,
        pts : int
    ) -> pd.Series :
    """
    Generates the theoretical Marchenko-Pastur probability density function.
    """
    e_min = var * (1 - (1./q)**0.5)**2
    e_max = var * (1 + (1./q)**0.5)**2
    e_val = np.linspace(e_min, e_max, pts)
    pdf = q / (2 * np.pi * var * e_val) * ((e_max - e_val) * (e_val - e_min))**0.5
    pdf = pd.Series(pdf.flatten(), index=e_val.flatten())
    return pdf

def fit_kde(
        observations : np.ndarray,
        b_width : float = 0.25 ,
        x_range = None
    ) -> pd.Series :
    """
    Fits a Kernel Density Estimator to the empirical eigenvalues.
    """
    if x_range is None:
        x_range = np.unique(observations).reshape(-1, 1)
    kde = KernelDensity(bandwidth=b_width, kernel='gaussian').fit(observations.reshape(-1, 1))
    log_prob = kde.score_samples(x_range)
    pdf = pd.Series(np.exp(log_prob), index=x_range.flatten())
    return pdf

def err_pdfs(
        var : float,
        observations : np.ndarray,
        q : float, b_width : float,
        pts : int =1000
    ) -> float :
    """
    Calculates the squared error between theoretical and empirical PDFs.
    """
    pdf0 = get_mp_pdf(var, q, pts)
    pdf1 = fit_kde(observations, b_width, x_range=pdf0.index.values.reshape(-1, 1))
    sse = np.sum((pdf0 - pdf1)**2)
    return sse

def denoise_vi_distance_shrinkage(
        vi_dist_df : pd.DataFrame,
        q_ratio : float,
        alpha : float =0.5,
        b_width : float =0.01
    ) -> pd.DataFrame :
    """
    Denoises a VI distance matrix using the MP-Law and Eigenvalue Shrinkage.

    Args:
        vi_dist_df: Linear VI distance matrix (pd.DataFrame).
        q_ratio: T/N ratio.
        alpha: Shrinkage factor (0.0: No change, 1.0: Full flattening to average).
               Set closer to 0.5 to preserve some structure in the noise.
        b_width: KDE bandwidth.
    """
    # Step 1: Convert Distance to Similarity (S = 1 - D)
    similarity_matrix = 1 - vi_dist_df.values

    # Step 2: Eigenvalue Decomposition
    e_val, e_vec = eigh(similarity_matrix)
    indices = e_val.argsort()[::-1]
    e_val, e_vec = e_val[indices], e_vec[:, indices]

    # Step 3: Fit MP distribution to find var (sigma^2)
    # Using the same err_pdfs and get_mp_pdf from previous steps
    out = minimize(lambda *x: err_pdfs(*x), .5, args=(e_val, q_ratio, b_width),
                   bounds=((1e-5, 1-1e-5),))
    var = out.x[0] if out.success else 1.0

    # Step 4: Determine e_max (Threshold)
    e_max = var * (1 + (1./q_ratio)**0.5)**2
    n_facts = e_val[e_val > e_max].shape[0]

    # Logic Check: Ensure we don't accidentally lose all features
    n_facts = max(n_facts, 1)

    print(f"Signal discovered: {n_facts} factors above MP-threshold ({e_max:.4f})")

    # Step 5: Apply Shrinkage to the noise eigenvalues
    # Instead of replacing with a constant, we shrink them towards the average.
    e_val_corr = e_val.copy()
    avg_noise = e_val_corr[n_facts:].mean()

    # Shrinkage formula: (alpha * average) + ((1 - alpha) * original)
    e_val_corr[n_facts:] = alpha * avg_noise + (1 - alpha) * e_val_corr[n_facts:]

    print(f"Noise eigenvalues shrunk towards {avg_noise:.4f} with alpha={alpha}")

    # Step 6: Reconstruct Denoised Similarity Matrix
    denoised_sim = np.dot(e_vec, e_val_corr[:, None] * e_vec.T)

    # Re-scale diagonal to 1
    diag = np.diag(denoised_sim)
    denoised_sim = denoised_sim / np.sqrt(np.outer(diag, diag))

    # Step 7: Convert back to Distance (D = 1 - S)
    denoised_vi_dist = np.clip(1 - denoised_sim, 0, 1)

    print("Shrinkage-based denoising complete.")
    return pd.DataFrame(denoised_vi_dist, index=vi_dist_df.index, columns=vi_dist_df.columns)

def denoise_and_soft_detone_vi_distance(
        vi_dist_df : pd.DataFrame,
        q_ratio : float,
        gamma : float = 0.7,
        alpha : float = 0.5,
        b_width : float =0.01
    ) -> pd.DataFrame :
    """
    Perform spectral denoising and soft-detoning on the VI distance matrix.

    Args:
        vi_dist_df (pd.DataFrame): Linear VI distance matrix.
        q_ratio (float): T/N ratio for Marchenko-Pastur thresholding.
        gamma (float): Detoning strength [0, 1]. 1.0 is hard-detoning.
        alpha (float): Shrinkage intensity for noise eigenvalues.
    """
    # 1. Spectral Decomposition
    S = 1 - vi_dist_df.values
    evals, evecs = np.linalg.eigh(S)
    idx = evals.argsort()[::-1]
    evals, evecs = evals[idx], evecs[:, idx]

    # 2. Denoising (MP-Law with Shrinkage)
    # Optimization to find sigma^2 is assumed to be defined (err_pdfs)
    res = minimize(err_pdfs, [0.5], args=(evals, q_ratio, b_width), bounds=[(1e-5, 1-1e-5)])
    sigma2 = res.x[0] if res.success else 1.0
    e_max = sigma2 * (1 + (1./q_ratio)**0.5)**2

    n_signals = np.sum(evals > e_max)
    avg_noise = evals[n_signals:].mean()
    evals_corr = evals.copy()
    evals_corr[n_signals:] = alpha * avg_noise + (1 - alpha) * evals_corr[n_signals:]

    # 3. Soft Detoning: Scaling the Market Mode (lambda_1)
    # Gamma controls the 'Market Gravity'
    original_lambda1 = evals_corr[0]
    evals_corr[0] = original_lambda1 * (1 - gamma)

    # 4. Reconstruct and Rescale
    # Maintaining the unit diagonal property of the similarity matrix
    S_detoned = evecs @ np.diag(evals_corr) @ evecs.T
    std = np.sqrt(np.diag(S_detoned))
    S_normalized = S_detoned / np.outer(std, std)

    # 5. Inverse Mapping to Distance Space
    D_detoned = np.clip(1 - S_normalized, 0, 1)

    print(f"[*] Gamma {gamma}: Lambda_1 suppressed from {original_lambda1:.3f} to {evals_corr[0]:.3f}")
    return pd.DataFrame(D_detoned, index=vi_dist_df.index, columns=vi_dist_df.columns)