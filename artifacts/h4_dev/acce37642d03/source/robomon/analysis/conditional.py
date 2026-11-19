"""H3 estimator: pairwise conditional logistic composite likelihood within landmark strata.

For each failure/success pair inside (task, condition, landmark), the conditional probability that
the failed member is the case is sigmoid(beta @ (x_failure - x_success)). We sum those log
probabilities. This is a PAIRWISE COMPOSITE likelihood, not the exact all-members conditional
likelihood. Its dependent pairs require whole-episode cluster bootstrap, not Hessian standard errors.
No ridge penalty or fallback changes the target when a fit is separated or unidentifiable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from robomon.analysis.landmarks import _validate


def _design(frame, features, strata, cluster):
    for feature in features:
        _validate(frame, feature, strata)
    if not features or len(set(features)) != len(features):
        raise ValueError("Features must be distinct and nonempty")
    if not np.isfinite(frame[list(features)].to_numpy()).all():
        raise ValueError(
            "Supply common complete-case landmarks; missing features are not imputed"
        )
    if cluster not in frame or frame[cluster].isna().any():
        raise ValueError("Nonmissing bootstrap cluster IDs required")
    if (frame.groupby(cluster).task.nunique() > 1).any() or (
        frame.groupby("episode_id")[cluster].nunique() > 1
    ).any():
        raise ValueError(
            "Clusters cannot cross tasks, and episodes cannot cross clusters"
        )
    clusters = list(frame[cluster].unique())
    cindex = {key: i for i, key in enumerate(clusters)}
    blocks = [
        np.asarray([cindex[key] for key in group[cluster].unique()])
        for _, group in frame.groupby("task", sort=True)
    ]
    differences, left, right = [], [], []
    for _, group in frame.groupby(list(strata), observed=True, sort=True):
        failed, passed = group[group.failure == 1], group[group.failure == 0]
        if failed.empty or passed.empty:
            continue
        a, b = (
            failed[list(features)].to_numpy(float),
            passed[list(features)].to_numpy(float),
        )
        differences.append((a[:, None, :] - b[None, :, :]).reshape(-1, len(features)))
        left.extend(np.repeat([cindex[key] for key in failed[cluster]], len(passed)))
        right.extend(np.tile([cindex[key] for key in passed[cluster]], len(failed)))
    if not differences:
        raise ValueError("No comparable outcome pairs")
    return (
        np.concatenate(differences),
        np.asarray(left),
        np.asarray(right),
        blocks,
        len(clusters),
    )


def _fit(differences, weights):
    active = weights > 0
    x, w = differences[active], weights[active].astype(float)
    if len(x) < x.shape[1] or np.linalg.matrix_rank(x) < x.shape[1]:
        return None, "unidentifiable"
    w /= w.sum()

    def objective(beta):
        eta = np.einsum("ij,j->i", x, beta, optimize=False)
        loss = float(np.sum(w * np.logaddexp(0.0, -eta)))
        gradient = -np.einsum("i,ij->j", w * expit(-eta), x, optimize=False)
        return loss, gradient

    fit = minimize(
        objective,
        np.zeros(x.shape[1]),
        jac=True,
        method="BFGS",
        options={"gtol": 1e-8, "maxiter": 1000},
    )
    if not fit.success and np.max(np.abs(fit.jac)) > 1e-6:
        return None, "not_converged"
    probability = expit(np.einsum("ij,j->i", x, fit.x, optimize=False))
    hessian = np.einsum(
        "i,ij,ik->jk", w * probability * (1 - probability), x, x, optimize=False
    )
    eigenvalues = np.linalg.eigvalsh(hessian)
    if (
        not np.isfinite(fit.x).all()
        or np.max(np.abs(fit.x)) > 25
        or eigenvalues[0] < 1e-8
    ):
        return None, "separated_or_singular"
    return fit.x, "ok"


def fit_pairwise_conditional(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    *,
    strata: tuple[str, ...] = ("task", "condition", "landmark"),
    cluster: str = "pair_id",
    n_boot: int = 1000,
    seed: int = 20261111,
    scales: dict[str, float] | None = None,
    interval: str = "basic",
) -> dict:
    """Coefficients per declared feature SD; basic-bootstrap CIs cluster by paired initial state.

    If scales is omitted, use marginal feature SD over the supplied analysis rows, without labels.
    Supplied scales support freezing a development normalization. Fits with >10% invalid bootstrap
    draws are labelled inconclusive. All invalid-draw reasons are reported.
    """
    if n_boot < 1:
        raise ValueError("Positive bootstrap count required")
    if interval not in ("basic", "percentile"):
        raise ValueError("interval must be basic or percentile")
    data, left, right, blocks, n_clusters = _design(frame, features, strata, cluster)
    scales = scales or {key: float(frame[key].std(ddof=0)) for key in features}
    scale = np.asarray([scales[key] for key in features])
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Each feature needs a finite positive scale")
    data = data / scale
    beta, reason = _fit(data, np.ones(len(data)))
    if beta is None:
        return {
            "status": "inconclusive",
            "reason": reason,
            "coefficients": None,
            "pairs": len(data),
            "scales": scales,
            "method": "pairwise_conditional_composite_logistic",
        }
    rng = np.random.default_rng(seed)
    estimates, invalid = [], {}
    for _ in range(n_boot):
        counts = np.zeros(n_clusters, dtype=int)
        for block in blocks:
            counts += np.bincount(
                rng.choice(block, len(block), replace=True), minlength=n_clusters
            )
        fit, reason = _fit(data, counts[left] * counts[right])
        if fit is None:
            invalid[reason] = invalid.get(reason, 0) + 1
        else:
            estimates.append(fit)
    percentile = (
        np.quantile(estimates, [0.025, 0.975], axis=0)
        if estimates
        else np.full((2, len(features)), np.nan)
    )
    bounds = (
        np.array([2 * beta - percentile[1], 2 * beta - percentile[0]])
        if interval == "basic"
        else percentile
    )
    coefficients = {
        key: {
            "coef": float(beta[i]),
            "ci_low": float(bounds[0, i]),
            "ci_high": float(bounds[1, i]),
        }
        for i, key in enumerate(features)
    }
    return {
        "status": "ok" if len(estimates) >= 0.9 * n_boot else "inconclusive",
        "method": "pairwise_conditional_composite_logistic",
        "coefficients": coefficients,
        "scales": scales,
        "pairs": len(data),
        "n_boot": n_boot,
        "bootstrap_valid": len(estimates),
        "interval": interval,
        "invalid_reasons": invalid,
        "seed": seed,
        "cluster": cluster,
    }
