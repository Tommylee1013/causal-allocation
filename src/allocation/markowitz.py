import numpy as np
import pandas as pd
import cvxpy as cp

def sharpe_opt_weights(
        mu: pd.Series,
        cov: pd.DataFrame,
        long_only=True,
        max_weight=1.0,
        gamma=1.0
    ) -> pd.Series :
    """
    max_w  mu^T w - (gamma/2) w^T Sigma w
    s.t.   sum(w)=1, w>=0 (옵션), w<=max_weight
    """
    n = len(mu)
    w = cp.Variable(n)

    Sigma = cov.values
    mu_vec = mu.values

    obj = cp.Maximize(mu_vec @ w - 0.5 * gamma * cp.quad_form(w, Sigma))
    cons = [cp.sum(w) == 1]

    if long_only:
        cons += [w >= 0]
    if max_weight is not None:
        cons += [w <= max_weight]

    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.OSQP, verbose=False)

    w_opt = np.array(w.value).ravel()
    w_opt = np.maximum(w_opt, 0)
    w_opt = w_opt / w_opt.sum()

    return pd.Series(w_opt, index=mu.index)