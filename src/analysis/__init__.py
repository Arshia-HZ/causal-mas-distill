"""Analysis utilities: statistics, audits, and reporting."""

from .stats import (  # noqa: F401
    Interval,
    paired_bootstrap_ci,
    cluster_bootstrap_ci,
    se_of_delta,
    required_k,
    eb_shrink,
    benjamini_hochberg,
    variance_decomposition,
)
