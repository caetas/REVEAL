"""RARE25 evaluation protocol on the per-image OOD scores written by `iREPA_rare.py --ood` (scores.csv).

For each repetition, all non-dysplastic (NDBE) images are kept and neoplasia (NEO) images are sampled with
replacement at a ratio of 1 NEO per `--ratio` NDBE (~1% prevalence for 100). PPV@90Recall is computed per
repetition and the final score is the median over `--n_reps` repetitions. AUROC and AUPRC are reported as
secondary metrics (also medians over the same repetitions). The same NEO draws are used for every score column,
so the scores are compared on identical resamples.
"""

import argparse
import csv
import json
import os

import numpy as np
from utils.ood_metrics import META_COLUMNS, auprc, auroc, ppv_at_recall


def load_scores(path):
    if os.path.isdir(path):
        path = os.path.join(path, "scores.csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    is_anomaly = np.array([int(r["is_anomaly"]) for r in rows], dtype=bool)
    names = [k for k in rows[0] if k not in META_COLUMNS]
    scores = {k: np.array([float(r[k]) for r in rows]) for k in names}
    return path, is_anomaly, scores


def _stats(values):
    values = np.asarray(values)
    return {
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p2.5": float(np.percentile(values, 2.5)),
        "p97.5": float(np.percentile(values, 97.5)),
    }


def rare25_protocol(scores, is_anomaly, n_reps=1000, ratio=100, target_recall=0.9, seed=0, n_pos=None):
    neg_idx = np.flatnonzero(~is_anomaly)
    pos_idx = np.flatnonzero(is_anomaly)
    if len(neg_idx) == 0 or len(pos_idx) == 0:
        raise ValueError("Need both NDBE and NEO images")
    n_pos = n_pos or max(1, int(round(len(neg_idx) / ratio)))

    rng = np.random.default_rng(seed)
    draws = rng.choice(pos_idx, size=(n_reps, n_pos), replace=True)
    y = np.r_[np.zeros(len(neg_idx), dtype=bool), np.ones(n_pos, dtype=bool)]

    results = {}
    for name, s in scores.items():
        ppv, roc, prc = [], [], []
        for draw in draws:
            s_rep = s[np.r_[neg_idx, draw]]
            ppv.append(ppv_at_recall(s_rep, y, target_recall)[0])
            roc.append(auroc(s_rep, y))
            prc.append(auprc(s_rep, y))
        results[name] = {
            "ppv_at_recall": _stats(ppv),
            "auroc": _stats(roc),
            "auprc": _stats(prc),
            # without resampling, on the full set, for reference
            "full_set": {
                "ppv_at_recall": ppv_at_recall(s, is_anomaly, target_recall)[0],
                "auroc": auroc(s, is_anomaly),
                "auprc": auprc(s, is_anomaly),
            },
        }
    setup = {
        "n_ndbe": int(len(neg_idx)),
        "n_neo_available": int(len(pos_idx)),
        "n_neo_per_rep": int(n_pos),
        "prevalence": n_pos / (n_pos + len(neg_idx)),
        "n_reps": n_reps,
        "target_recall": target_recall,
        "seed": seed,
    }
    return setup, results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scores", type=str, help="scores.csv from `iREPA_rare.py --ood`, or its results folder")
    parser.add_argument("--n_reps", type=int, default=1000, help="Number of resampling repetitions")
    parser.add_argument("--ratio", type=float, default=100, help="Number of NDBE images per sampled NEO image")
    parser.add_argument("--n_pos", type=int, default=None, help="NEO images per repetition (overrides --ratio)")
    parser.add_argument("--target_recall", type=float, default=0.9, help="Recall at which PPV is computed")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the NEO resampling")
    parser.add_argument("--columns", type=str, default=None, help="Comma-separated score columns (default: all)")
    parser.add_argument("--out", type=str, default=None, help="Output JSON (default: rare25_eval.json next to CSV)")
    args = parser.parse_args()

    path, is_anomaly, scores = load_scores(args.scores)
    if args.columns:
        scores = {k: scores[k] for k in args.columns.split(",")}

    setup, results = rare25_protocol(
        scores, is_anomaly, args.n_reps, args.ratio, args.target_recall, args.seed, args.n_pos
    )

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(path)), "rare25_eval.json")
    with open(out, "w") as f:
        json.dump({"scores_csv": os.path.abspath(path), "setup": setup, "results": results}, f, indent=2)

    r = f"{args.target_recall:g}"
    print(
        f"{setup['n_ndbe']} NDBE + {setup['n_neo_per_rep']} NEO per repetition (sampled with replacement from "
        f"{setup['n_neo_available']}), prevalence {setup['prevalence']:.2%}, {setup['n_reps']} repetitions"
    )
    header = f"{'score':<22} {'PPV@' + r + 'R median [95% range]':>32} {'AUROC':>7} {'AUPRC':>7}"
    print(header)
    print("-" * len(header))
    for name, m in sorted(results.items(), key=lambda kv: -kv[1]["ppv_at_recall"]["median"]):
        p = m["ppv_at_recall"]
        interval = f"{p['median']:.4f} [{p['p2.5']:.4f}, {p['p97.5']:.4f}]"
        print(f"{name:<22} {interval:>32} {m['auroc']['median']:>7.4f} {m['auprc']['median']:>7.4f}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
