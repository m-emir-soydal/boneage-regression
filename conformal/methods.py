import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import cps_config as C


def conformal_quantile(scores, confidence):
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return 0.0

    n = len(scores)
    q_level = np.ceil((n + 1) * confidence) / n
    q_level = min(q_level, 1.0)
    try:
        return float(np.quantile(scores, q_level, method="higher"))
    except TypeError:
        return float(np.quantile(scores, q_level, interpolation="higher"))


def _clip(lower, upper):
    return np.clip(lower, C.Y_MIN, C.Y_MAX), np.clip(upper, C.Y_MIN, C.Y_MAX)


class StandardSplitCP:
    display_name = "SCP"

    def fit(self, y_cal, y_pred_cal):
        self.scores = np.abs(np.asarray(y_cal) - np.asarray(y_pred_cal))
        return self

    def predict_interval(self, y_pred_test, confidence):
        q = conformal_quantile(self.scores, confidence)
        y_pred_test = np.asarray(y_pred_test)
        return _clip(y_pred_test - q, y_pred_test + q)


class KNNNormalizedCP:
    display_name = "KNN-NCP"

    def __init__(self, k=50, metric="cosine", eps=1e-6, standardize=True):
        self.k = k
        self.metric = metric
        self.eps = eps
        self.standardize = standardize

    def _prepare_embeddings(self, z, fit=False):
        z = np.asarray(z, dtype=float)
        if self.standardize:
            if fit:
                self.scaler = StandardScaler()
                return self.scaler.fit_transform(z)
            return self.scaler.transform(z)
        return z

    def _local_sigma(self, z):
        n_neighbors = min(self.k, len(self.r_scale))
        _, idx = self.nn.kneighbors(z, n_neighbors=n_neighbors)
        return self.r_scale[idx].mean(axis=1) + self.eps

    def fit(self, y_scale, y_pred_scale, z_scale, y_cal, y_pred_cal, z_cal):
        self.r_scale = np.abs(np.asarray(y_scale) - np.asarray(y_pred_scale))
        z_scale = self._prepare_embeddings(z_scale, fit=True)
        z_cal = self._prepare_embeddings(z_cal, fit=False)

        n_neighbors = min(self.k, len(self.r_scale))
        self.nn = NearestNeighbors(n_neighbors=n_neighbors, metric=self.metric)
        self.nn.fit(z_scale)

        sigma_cal = self._local_sigma(z_cal)
        residual_cal = np.abs(np.asarray(y_cal) - np.asarray(y_pred_cal))
        self.scores = residual_cal / sigma_cal
        return self

    def predict_interval(self, y_pred_test, z_test, confidence):
        z_test = self._prepare_embeddings(z_test, fit=False)
        sigma_test = self._local_sigma(z_test)
        q = conformal_quantile(self.scores, confidence)
        y_pred_test = np.asarray(y_pred_test)
        lower, upper = _clip(y_pred_test - q * sigma_test, y_pred_test + q * sigma_test)
        return lower, upper, sigma_test


class AgeSexQuantileMondrianCP:
    display_name = "AS-MCP"

    def __init__(self, n_bins=5, quantiles=None, min_bin_size=30):
        self.n_bins = n_bins
        self.quantiles = quantiles if quantiles is not None else C.AGE_QUANTILES
        self.min_bin_size = min_bin_size

    def _bin_indices(self, y_pred):
        y_pred = np.asarray(y_pred, dtype=float)
        inner_edges = self.age_edges[1:-1] if len(self.age_edges) > 2 else []
        return np.searchsorted(inner_edges, y_pred, side="right")

    def _labels(self, sex, y_pred):
        sex = np.asarray(sex).astype(int)
        bins = self._bin_indices(y_pred)
        return np.array([f"{s}__qbin_{b}" for s, b in zip(sex, bins)])

    def fit(self, y_cal, y_pred_cal, sex_cal):
        y_cal = np.asarray(y_cal, dtype=float)
        y_pred_cal = np.asarray(y_pred_cal, dtype=float)
        residuals = np.abs(y_cal - y_pred_cal)
        self.global_scores = residuals

        edges = np.quantile(y_pred_cal, self.quantiles)
        self.age_edges = np.unique(edges)
        if len(self.age_edges) < 2:
            self.age_edges = np.array([y_pred_cal.min(), y_pred_cal.max()])

        labels = self._labels(sex_cal, y_pred_cal)
        self.bin_scores = {}
        for label in np.unique(labels):
            self.bin_scores[label] = residuals[labels == label]
        return self

    def predict_interval(self, y_pred_test, sex_test, confidence):
        y_pred_test = np.asarray(y_pred_test, dtype=float)
        labels = self._labels(sex_test, y_pred_test)

        lower = np.empty_like(y_pred_test, dtype=float)
        upper = np.empty_like(y_pred_test, dtype=float)
        global_q = conformal_quantile(self.global_scores, confidence)

        for i, label in enumerate(labels):
            scores = self.bin_scores.get(label)
            if scores is not None and len(scores) >= self.min_bin_size:
                q = conformal_quantile(scores, confidence)
            else:
                q = global_q
            lower[i] = y_pred_test[i] - q
            upper[i] = y_pred_test[i] + q

        return _clip(lower, upper)


class BackwardFixedWidthCP:
    """
    Regression adaptation of Backward CP.

    The paper fixes a prediction-set size constraint and estimates the resulting
    marginal coverage. For scalar regression, we use a fixed interval width as
    the size constraint and estimate coverage from calibration residuals.
    """

    display_name = "BCP"

    def __init__(self, interval_width_months=24.0):
        self.interval_width_months = float(interval_width_months)
        self.half_width = self.interval_width_months / 2.0

    def fit(self, y_cal, y_pred_cal):
        residuals = np.abs(np.asarray(y_cal, dtype=float) - np.asarray(y_pred_cal, dtype=float))
        self.residuals = residuals
        self.loo_miscoverages = (residuals > self.half_width).astype(float)
        self.estimated_miscoverage = float(np.mean(self.loo_miscoverages))
        self.estimated_coverage = 1.0 - self.estimated_miscoverage
        return self

    def predict_interval(self, y_pred_test, confidence=None):
        y_pred_test = np.asarray(y_pred_test, dtype=float)
        return _clip(y_pred_test - self.half_width, y_pred_test + self.half_width)

    def diagnostics(self, confidence=None):
        trust = np.nan
        if confidence is not None:
            trust = bool(self.estimated_coverage >= confidence)
        return {
            "bcp_interval_width_months": self.interval_width_months,
            "bcp_estimated_coverage": self.estimated_coverage,
            "bcp_estimated_miscoverage": self.estimated_miscoverage,
            "bcp_trust_at_confidence": trust,
        }
