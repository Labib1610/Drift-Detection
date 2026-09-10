#!/usr/bin/env python3
"""
TASK 6 — Part 3. S6, the supervised reference detector.

Online River logistic regression over 2^18 hashed bag-of-words features, predicting
publisher (3 classes, flat prior by T2 panel design) and, as a secondary task, topic
(8 classes). Two variants:

  * frozen (S6)      — trained ONLY on the first 10% of the stream (vocab+reference
                       epochs), then frozen. Its per-window error rate rises as the
                       world drifts away from its training distribution. This is the
                       deployed-model reference the label-free signals are timed against.
  * prequential (S6p)— test-then-train, continuously updating; a self-updating model
                       masks drift, so the contrast is informative.

Per-window error rate is z-scored with the same split-epoch protocol as every other
signal and ADWIN-calibrated on the 10 shuffled null streams (same grid, same targets).

Usage:
    python src/classifier.py --params params.yaml
    python src/classifier.py --demo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from collections import Counter

import numpy as np
import pandas as pd
import yaml
from river import linear_model, multiclass, optim

# reuse the calibration primitives so there is one definition of each
from detect import adwin_alarms, far_curve, pick_delta, epoch_bounds  # noqa

OUT_DIR = "features/classifier"
RES_DIR = "results"
STREAM = "bn_panel"


def log(m):
    print(m, flush=True)


def hashed_bow(text, nf):
    d = Counter()
    for tok in str(text).split():
        d[zlib.crc32(tok.encode("utf-8")) % nf] += 1
    return dict(d)


def window_bounds(nwords_ord, target):
    b = [0]; acc = 0
    for i, w in enumerate(nwords_ord):
        acc += int(w)
        if acc >= target:
            b.append(i + 1); acc = 0
    return np.array(b)


def zscore_split(values, lo, hi):
    v = np.asarray(values, float)
    ref = v[lo:hi]; ref = ref[~np.isnan(ref)]
    if ref.size == 0 or np.std(ref) < 1e-9:
        return np.full_like(v, np.nan), float("nan"), float("nan")
    return (v - ref.mean()) / ref.std(), float(ref.mean()), float(ref.std())


def make_model(n_classes, lr):
    return multiclass.OneVsRestClassifier(
        linear_model.LogisticRegression(optimizer=optim.SGD(lr)))


def per_doc_error_frozen(feats, labels, train_idx, n_classes, lr):
    """Train on train_idx (in that order), freeze, return per-doc 0/1 error for all."""
    model = make_model(n_classes, lr)
    for i in train_idx:
        model.learn_one(feats[i], labels[i])
    err = np.ones(len(feats), dtype=np.int8)
    ref_correct = 0
    for i in range(len(feats)):
        p = model.predict_one(feats[i])
        err[i] = 0 if p == labels[i] else 1
    return err, model


def per_doc_error_preq(feats, labels, order, n_classes, lr):
    """Prequential test-then-train in `order`; return per-doc error aligned to doc id."""
    model = make_model(n_classes, lr)
    err = np.full(len(feats), np.nan)
    for i in order:
        p = model.predict_one(feats[i])
        err[i] = 0.0 if p == labels[i] else 1.0
        model.learn_one(feats[i], labels[i])
    return err


def window_series(doc_err, order, bounds, dates_ns):
    """Per-window mean error + median date for one permutation."""
    nwin = len(bounds) - 1
    ew = np.empty(nwin); md = np.empty(nwin, np.int64)
    eo = doc_err[order]; do = dates_ns[order]
    for wi in range(nwin):
        s, e = bounds[wi], bounds[wi + 1]
        seg = eo[s:e]
        ew[wi] = np.nanmean(seg) if seg.size else np.nan
        md[wi] = int(np.median(do[s:e]))
    return ew, md


def calibrate(null_zstreams, grid, targets):
    curve, tot = far_curve(null_zstreams, grid)
    picks = {}
    for tf in targets:
        d = pick_delta(curve, tf)
        picks[f"{tf:g}"] = {"delta": d, "far": curve.get(d) if d is not None else None}
    return {f"{g:g}": curve[g] for g in grid}, picks, tot


def main():
    ap = argparse.ArgumentParser(description="TASK 6 Part 3 — supervised S6")
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    P = yaml.safe_load(open(args.params))
    nf = P["classifier"]["n_features"]
    lr = P["classifier"]["learning_rate"]
    targets = P["classifier"]["targets"]
    vf = P["window"]["vocab_fraction"]; rf = P["window"]["reference_fraction"]
    tw = P["window"]["words_per_window"]
    grid = sorted(float(d) for d in P["adwin"]["delta_grid"])
    far_targets = [float(P["adwin"]["target_far"])] + [float(x) for x in P["adwin"]["far_sensitivity"]]
    demo = args.demo
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)

    df = pd.read_parquet(f"data/interim/{STREAM}.parquet",
                         columns=["doc_id", "text", "date", "publisher", "topic", "n_words"])
    if demo:
        df = df.head(6000).reset_index(drop=True)
    n = len(df)
    nwords = df["n_words"].to_numpy(np.int64)
    dates_ns = pd.to_datetime(df["date"]).values.astype("datetime64[ns]").astype(np.int64)
    years = pd.to_datetime(df["date"]).dt.year.to_numpy()
    log(f"[classifier] {n:,} docs; hashing bag-of-words ...")
    feats = [hashed_bow(t, nf) for t in df["text"].tolist()]

    # permutations (real=perm00 identity, nulls 01..10)
    perms = {0: np.arange(n)}
    if not demo:
        npz = np.load(f"data/streams/{STREAM}_perms.npz")
        for k in range(1, 11):
            perms[k] = npz[f"perm_{k:02d}"].astype(np.int64)
        perms[0] = npz["perm_00"].astype(np.int64)
    else:
        for k in range(1, 4):
            perms[k] = np.random.default_rng(40 + k).permutation(n)
    null_ks = [k for k in perms if k != 0]

    # reference epoch docs on the real stream (first 10% of windows)
    b0 = window_bounds(nwords[perms[0]], tw)
    nwin0 = len(b0) - 1
    nv0, nre0 = epoch_bounds(nwin0, vf, rf)
    train_docs = perms[0][: int(b0[nre0])]

    calibration = {}
    metrics = {}
    S6_KEYS = {"publisher": "S6", "topic": "S6_topic"}
    for target in targets:
        labels = df[target].astype(str).tolist()
        classes = sorted(set(labels))
        log(f"[classifier] target={target} ({len(classes)} classes): frozen train "
            f"on {len(train_docs):,} docs ...")
        err_frozen, model = per_doc_error_frozen(feats, labels, train_docs, len(classes), lr)
        # frozen accuracy: reference epoch + per year
        ref_acc = 1 - err_frozen[train_docs].mean()
        yr_acc = {int(y): float(1 - err_frozen[years == y].mean()) for y in sorted(set(years))}
        mono = all(yr_acc[y] >= yr_acc[y2] - 1e-9 for y, y2 in zip(sorted(yr_acc)[1:], sorted(yr_acc)[:-1]))
        # note: error rises => accuracy falls => yr_acc should be *non-increasing*
        acc_vals = [yr_acc[y] for y in sorted(yr_acc)]
        err_monotone = all(acc_vals[i] >= acc_vals[i + 1] - 1e-3 for i in range(len(acc_vals) - 1))
        metrics[target] = {"ref_accuracy": float(ref_acc), "year_accuracy": yr_acc,
                           "frozen_error_monotone_rising": bool(err_monotone)}
        log(f"  ref acc={ref_acc:.3f}; year acc={ {y: round(a,3) for y,a in yr_acc.items()} }")

        variants = {"frozen": err_frozen}
        # prequential: real + (publisher only) nulls for calibration; topic real only
        preq_real = per_doc_error_preq(feats, labels, perms[0], len(classes), lr)
        variants["preq"] = preq_real

        for variant, doc_err in variants.items():
            sig = "S6" if (target == "publisher" and variant == "frozen") else \
                  "S6p" if (target == "publisher" and variant == "preq") else \
                  f"{target}_{variant}"
            # per-perm z-series
            null_z = []
            for k in sorted(perms):
                order = perms[k]
                bounds = window_bounds(nwords[order], tw)
                nwin = len(bounds) - 1
                nv, nre = epoch_bounds(nwin, vf, rf)
                if variant == "preq" and k != 0:
                    # prequential is order-dependent — recompute per null (publisher only)
                    if target != "publisher":
                        continue
                    de = per_doc_error_preq(feats, labels, order, len(classes), lr)
                    ew, md = window_series(de, order, bounds, dates_ns)  # de is doc-indexed
                else:
                    ew, md = window_series(doc_err, order, bounds, dates_ns)
                z, mu, sd = zscore_split(ew, nv, nre)
                out = pd.DataFrame({"window_idx": np.arange(nwin, dtype=np.int32),
                                    "median_date": pd.to_datetime(md),
                                    "err": ew, "z": z})
                sfx = "_demo" if demo else ""
                out.to_parquet(f"{OUT_DIR}/{target}_{variant}__perm{k:02d}{sfx}.parquet",
                               index=False)
                if k in null_ks:
                    zz = z[nre:]; null_z.append(zz[~np.isnan(zz)])
            if null_z:
                curve, picks, tot = calibrate(null_z, grid, far_targets)
                calibration[sig] = {"target": target, "variant": variant,
                                    "far_curve": curve, "targets": picks, "null_windows": tot}
                log(f"  {sig}: delta*(1e-3)={picks[f'{float(P['adwin']['target_far']):g}']['delta']}")

    with open(f"{RES_DIR}/classifier_calibration.json", "w") as fh:
        json.dump({"grid": grid, "far_targets": far_targets, "calibration": calibration}, fh, indent=2)
    with open(f"{RES_DIR}/classifier_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    log(f"[classifier] done. Wall-clock {time.time()-t0:.1f}s")
    return 0


def _win_from_stream(doc_err_in_docorder, order, bounds, dates_ns):
    """Per-window mean where doc_err_in_docorder is indexed by doc id (not stream)."""
    nwin = len(bounds) - 1
    ew = np.empty(nwin); md = np.empty(nwin, np.int64)
    eo = doc_err_in_docorder[order]; do = dates_ns[order]
    for wi in range(nwin):
        s, e = bounds[wi], bounds[wi + 1]
        ew[wi] = np.nanmean(eo[s:e]) if e > s else np.nan
        md[wi] = int(np.median(do[s:e]))
    return ew, md


if __name__ == "__main__":
    sys.exit(main())
