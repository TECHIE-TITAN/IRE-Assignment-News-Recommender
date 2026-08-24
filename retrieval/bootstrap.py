"""Q4.4: bootstrap 95% confidence intervals.

Two variants, because coverage isn't a per-impression scalar you can just
resample-and-average like AUC/MRR/nDCG/diversity/novelty are:

  - `bootstrap_ci_mean`: resamples a 1-D array of per-impression metric
    values with replacement and takes percentiles of the resampled means.
  - `bootstrap_ci_coverage`: resamples *impressions* and recomputes the
    catalog-coverage set-union each time. Implemented with boolean-array
    fancy indexing (not Python sets) so 1000 iterations over ~70K
    impressions stays fast -- see scripts/evaluate_ranking.py for how the
    flattened (impression_owner, item_row) arrays it takes are built.
"""

import numpy as np


def bootstrap_ci_mean(values, n_boot=1000, ci=0.95, random_state=42):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(random_state)
    n = len(values)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = values[idx].mean()
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_means, [alpha, 1 - alpha])
    return {"mean": float(values.mean()), "ci_low": float(lo), "ci_high": float(hi), "n": int(n)}


def bootstrap_ci_coverage(item_owner, item_row, n_impressions, n_docs, n_boot=1000, ci=0.95, random_state=42):
    """`item_owner[i]` / `item_row[i]`: parallel arrays, one entry per
    (impression, recommended-item) pair -- which impression it came from,
    and that item's row index in the corpus. Each bootstrap draw resamples
    impression indices [0, n_impressions) with replacement, marks which
    impressions were included at least once, and computes coverage as the
    fraction of the `n_docs`-article catalog touched by *any* included
    impression's top-K list.

    Coverage is a set-union statistic, so this is subject to a known
    bootstrap bias: sampling `n_impressions` indices *with replacement*
    only touches ~63% of the distinct impressions on average (1 - 1/e), so
    every resample's union is systematically smaller than the union over
    the full, all-distinct evaluated set. `point_estimate` (coverage on the
    actual, un-resampled evaluation run -- the number that matters) is
    reported separately from `mean`/`ci_low`/`ci_high` (the resampling
    distribution) for that reason; don't expect the point estimate to fall
    inside the CI band."""
    rng = np.random.default_rng(random_state)
    boot_cov = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.integers(0, n_impressions, size=n_impressions)
        included = np.zeros(n_impressions, dtype=bool)
        included[sampled] = True
        mask = included[item_owner]
        covered = np.zeros(n_docs, dtype=bool)
        covered[item_row[mask]] = True
        boot_cov[b] = covered.mean()
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(boot_cov, [alpha, 1 - alpha])

    covered_point = np.zeros(n_docs, dtype=bool)
    covered_point[item_row] = True
    point = float(covered_point.mean())
    return {
        "point_estimate": point,
        "mean": float(boot_cov.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": int(n_impressions),
        "note": "point_estimate is coverage on the full evaluated set (the number to report); "
                "mean/ci_low/ci_high are the with-replacement resampling distribution, which is "
                "systematically lower -- see docstring.",
    }
