"""Binary OOD metrics (numpy only). Positive = anomalous (NEO); a higher score means more anomalous."""

import numpy as np

# non-score columns of the OOD scores.csv
META_COLUMNS = ("index", "image", "path", "center", "label", "is_anomaly")


def _as_arrays(scores, is_anomaly):
    return np.asarray(scores, dtype=np.float64), np.asarray(is_anomaly, dtype=bool)


def _pr_points(scores, is_anomaly):
    """TP, FP and thresholds at every distinct threshold (predict positive when score >= threshold),
    from the highest threshold to the lowest; tied scores are always on the same side."""
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], is_anomaly[order]
    tp, fp = np.cumsum(y), np.cumsum(~y)
    cut = np.r_[s[1:] != s[:-1], True]
    return tp[cut], fp[cut], s[cut]


def auroc(scores, is_anomaly):
    """AUROC via the Mann-Whitney U statistic (ties get average ranks)."""
    scores, is_anomaly = _as_arrays(scores, is_anomaly)
    n_pos = int(is_anomaly.sum())
    n_neg = len(is_anomaly) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    _, first_idx, counts = np.unique(scores[order], return_index=True, return_counts=True)
    ranks = np.empty(len(scores))
    ranks[order] = np.repeat(first_idx + (counts + 1) / 2.0, counts)
    return float((ranks[is_anomaly].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ppv_at_recall(scores, is_anomaly, target_recall=0.9):
    """Best PPV (precision) over all thresholds whose recall is >= target_recall.
    Returns (ppv, threshold, recall reached at that threshold)."""
    scores, is_anomaly = _as_arrays(scores, is_anomaly)
    n_pos = int(is_anomaly.sum())
    if n_pos == 0:
        return float("nan"), float("nan"), float("nan")
    tp, fp, thresholds = _pr_points(scores, is_anomaly)
    recall = tp / n_pos
    ppv = tp / (tp + fp)
    valid_ppv = np.where(recall >= target_recall - 1e-12, ppv, -1.0)
    # among equally good thresholds, report the one with the highest recall (the last one)
    i = int(np.flatnonzero(valid_ppv == valid_ppv.max())[-1])
    return float(ppv[i]), float(thresholds[i]), float(recall[i])


def auprc(scores, is_anomaly):
    """Area under the precision-recall curve as average precision: sum_n (R_n - R_{n-1}) * P_n
    (step-wise, no interpolation; same definition as sklearn's average_precision_score)."""
    scores, is_anomaly = _as_arrays(scores, is_anomaly)
    n_pos = int(is_anomaly.sum())
    if n_pos == 0:
        return float("nan")
    tp, fp, _ = _pr_points(scores, is_anomaly)
    recall = tp / n_pos
    ppv = tp / (tp + fp)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * ppv))
