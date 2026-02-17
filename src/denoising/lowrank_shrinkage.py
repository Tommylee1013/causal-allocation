import numpy as np
import pandas as pd

def _as_symmetric(df: pd.DataFrame, tol: float = 1e-12) -> pd.DataFrame:
    """
    Force symmetry by (A + A.T) / 2 to remove numerical asymmetry.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas.DataFrame.")

    if df.shape[0] != df.shape[1]:
        raise ValueError("Matrix must be square (n x n).")

    if list(df.index) != list(df.columns):
        common = df.index.intersection(df.columns)
        if len(common) != df.shape[0]:
            raise ValueError("Index and columns must contain the same asset labels.")
        df = df.loc[common, common]

    A = df.astype(float)
    A_sym = (A + A.T) / 2.0

    if np.nanmax(np.abs(A_sym.values - A_sym.values.T)) > tol:
        raise ValueError("Failed to symmetrize the matrix within tolerance.")

    return A_sym

def _eigh_sorted(A: np.ndarray):
    """
    Eigen-decomposition for a symmetric matrix with eigenvalues sorted in descending order.
    """
    w, V = np.linalg.eigh(A)
    idx = np.argsort(w)[::-1]
    return w[idx], V[:, idx]


def _nmi_distance_to_similarity(
    dist_df: pd.DataFrame,
    eps: float = 1e-12,
    mode: str = "linear",
) -> pd.DataFrame:
    """
    Convert an NMI-based distance matrix into a similarity matrix.

    Your construction uses:
        nmi = mi / max(H_i, H_j)
        distance = max(0, 1 - nmi)

    Therefore, the natural similarity is:
        similarity = 1 - distance  (clipped to [0, 1])

    Parameters
    ----------
    dist_df : pd.DataFrame
        NMI-based distance matrix (assets x assets), typically in [0, 1].
    eps : float
        Numerical tolerance.
    mode : str
        - "linear": similarity = 1 - distance
        - "exp":    similarity = exp(-distance / scale)

    Returns
    -------
    pd.DataFrame
        Similarity matrix S in [0, 1] with diagonal set to 1.
    """
    D = _as_symmetric(dist_df)
    A = D.values

    if mode == "linear":
        S = 1.0 - A
        S = np.clip(S, 0.0, 1.0)
        np.fill_diagonal(S, 1.0)
        return pd.DataFrame(S, index=D.index, columns=D.columns)

    if mode == "exp":
        off = A[~np.eye(A.shape[0], dtype=bool)]
        scale = np.nanmedian(off)
        scale = float(scale) if np.isfinite(scale) and scale > eps else 1.0
        S = np.exp(-A / scale)
        np.fill_diagonal(S, 1.0)
        return pd.DataFrame(S, index=D.index, columns=D.columns)

    raise ValueError("mode must be one of {'linear', 'exp'}.")


def denoise_lowrank_shrinkage_on_similarity(
    sim_df: pd.DataFrame,
    k: int | None = None,
    energy: float = 0.90,
    alpha: float = 0.50,
    psd_clip: bool = False,
) -> pd.DataFrame:
    """
    Denoise a similarity matrix using low-rank approximation with shrinkage mixing.

    S_denoise = alpha * S_k + (1 - alpha) * S

    Parameters
    ----------
    sim_df : pd.DataFrame
        Similarity matrix (assets x assets), symmetric.
    k : int | None
        Rank of the low-rank approximation. If None, choose k automatically by energy.
    energy : float
        Cumulative explained energy threshold (based on positive eigenvalues).
    alpha : float
        Shrinkage strength.
    psd_clip : bool
        If True, clip negative eigenvalues to zero before reconstructing S_k.

    Returns
    -------
    pd.DataFrame
        Denoised similarity matrix.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1].")
    if not (0.0 < energy <= 1.0):
        raise ValueError("energy must be in (0, 1].")

    S = _as_symmetric(sim_df)
    A = S.values

    w, V = _eigh_sorted(A)

    if psd_clip:
        w = np.maximum(w, 0.0)

    if k is None:
        w_pos = w[w > 0]
        if w_pos.size == 0:
            return S.copy()
        cum = np.cumsum(w_pos) / np.sum(w_pos)
        k = int(np.searchsorted(cum, energy) + 1)
        k = max(1, min(k, A.shape[0]))

    if not (1 <= k <= A.shape[0]):
        raise ValueError(f"k must be between 1 and {A.shape[0]}")

    Vk = V[:, :k]
    wk = w[:k]

    S_k = (Vk * wk) @ Vk.T
    S_denoise = alpha * S_k + (1.0 - alpha) * A

    return pd.DataFrame(S_denoise, index=S.index, columns=S.columns)


def detone_first_eigencomponent(
    sim_df: pd.DataFrame,
    gamma: float = 0.7,
    keep_diagonal: bool = True,
) -> pd.DataFrame:
    """
    Detone a similarity matrix by removing the first eigencomponent (global common mode).

    S_detone = S - lambda1 * v1 v1^T

    Parameters
    ----------
    sim_df : pd.DataFrame
        Symmetric similarity matrix (typically denoised).
    keep_diagonal : bool
        If True, restore the original diagonal after detoning.

    Returns
    -------
    pd.DataFrame
        Detoned similarity matrix.
    """
    S = _as_symmetric(sim_df)
    A = S.values
    diag_before = np.diag(A).copy()

    w, V = _eigh_sorted(A)

    lam1 = w[0]
    v1 = V[:, 0:1]

    A_detone = A - (lam1 * gamma) * (v1 @ v1.T)

    if keep_diagonal:
        np.fill_diagonal(A_detone, diag_before)

    return pd.DataFrame(A_detone, index=S.index, columns=S.columns)


def denoise_and_detone_nmi_distance(
    nmi_distance_df: pd.DataFrame,   # NMI-based distance matrix (assets x assets), values in [0, 1], output of your NMI construction
    similarity_mode: str = "linear", # How to convert distance to similarity: "linear" (S = 1 - D) or "exp" (S = exp(-D / scale))
    k: int | None = None,            # Rank of the low-rank approximation; None selects k automatically based on energy threshold
    energy: float = 0.90,            # Cumulative explained energy threshold used to choose k when k is None
    alpha: float = 0.50,             # Shrinkage strength: S_denoise = alpha * S_lowrank + (1 - alpha) * S_original
    psd_clip: bool = False,          # If True, clip negative eigenvalues to zero before low-rank reconstruction (optional stabilization)
    keep_diagonal: bool = True,      # If True, enforce unit diagonal (1.0) after denoising and detoning
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end pipeline for your NMI-distance output:
    1) Convert NMI-distance -> similarity matrix
    2) Denoise via low-rank + shrinkage
    3) Detone via first eigencomponent removal

    Parameters
    ----------
    nmi_distance_df : pd.DataFrame
        Output of your function: distance = max(0, 1 - nmi).
    similarity_mode : str
        "linear" (recommended) or "exp".
    k, energy, alpha, psd_clip : see denoise_lowrank_shrinkage_on_similarity
    keep_diagonal : bool
        Keep diagonal unchanged after detoning.

    Returns
    -------
    denoised_similarity : pd.DataFrame
    detoned_similarity : pd.DataFrame
    """
    sim = _nmi_distance_to_similarity(
        dist_df=nmi_distance_df,
        mode=similarity_mode,
    )
    denoised = denoise_lowrank_shrinkage_on_similarity(
        sim_df=sim,
        k=k,
        energy=energy,
        alpha=alpha,
        psd_clip=psd_clip,
    )
    detoned = detone_first_eigencomponent(
        sim_df=denoised,
        keep_diagonal=keep_diagonal,
    )
    return detoned

def soft_detone_and_normalize(sim_df, gamma=0.7):
    S = _as_symmetric(sim_df)
    A = S.values
    w, V = _eigh_sorted(A)

    lam1 = w[0]
    v1 = V[:, 0:1]
    A_detone = A - (lam1 * gamma) * (v1 @ v1.T)

    d_inv = 1.0 / np.sqrt(np.diag(A_detone))
    A_norm = A_detone * np.outer(d_inv, d_inv)

    A_norm = np.clip(A_norm, -1.0, 1.0)
    np.fill_diagonal(A_norm, 1.0)

    print(f"Soft Detoning & Normalization complete. (Gamma: {gamma})")
    return pd.DataFrame(A_norm, index=sim_df.index, columns=sim_df.columns)

def final_nmi_pipeline(nmi_distance_df, gamma=0.7, alpha=0.5, energy=0.9):
    # (1) NMI-distance -> similarity 변환
    sim = _nmi_distance_to_similarity(dist_df=nmi_distance_df, mode="linear")

    denoised = denoise_lowrank_shrinkage_on_similarity(
        sim_df=sim,
        energy=energy,
        alpha=alpha,
        psd_clip=True
    )

    detoned_final = soft_detone_and_normalize(denoised, gamma=gamma)

    return detoned_final
