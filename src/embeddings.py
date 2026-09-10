#!/usr/bin/env python3
"""
TASK 6 — Part 4. S5, the MMD embedding baseline (GPU).

Encodes every panel document once with LaBSE (Bengali coverage), then per window
computes MMD² against a fixed 2,000-document reference pool with an RBF kernel whose
bandwidth is the median-heuristic value computed ONCE on the reference pool and frozen
(recomputing per window would leak future information). S5 = MMD², z-scored with the
split-epoch protocol and ADWIN-calibrated on the 10 shuffled null streams.

Deterministic: encoding order is the parquet doc order; the reference pool sample and
all seeds are fixed.

Usage:
    python src/embeddings.py --params params.yaml
    python src/embeddings.py --demo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

from detect import far_curve, pick_delta, epoch_bounds  # noqa

EMB_DIR = "features/embeddings"
RES_DIR = "results"
STREAM = "bn_panel"


def log(m):
    print(m, flush=True)


def window_bounds(nwords_ord, target):
    b = [0]; acc = 0
    for i, w in enumerate(nwords_ord):
        acc += int(w)
        if acc >= target:
            b.append(i + 1); acc = 0
    return np.array(b)


def zscore_split(v, lo, hi):
    v = np.asarray(v, float); ref = v[lo:hi]; ref = ref[~np.isnan(ref)]
    if ref.size == 0 or np.std(ref) < 1e-9:
        return np.full_like(v, np.nan)
    return (v - ref.mean()) / ref.std()


def rbf_mmd2(X, Y, gamma, kyy_mean):
    """Biased MMD² between X (m×d) and Y (n×d) with RBF kernel, kyy_mean precomputed."""
    if X.shape[0] == 0:
        return np.nan
    xx = (X * X).sum(1)
    yy = (Y * Y).sum(1)
    Dxx = xx[:, None] + xx[None, :] - 2 * X @ X.T
    Dxy = xx[:, None] + yy[None, :] - 2 * X @ Y.T
    kxx = np.exp(-gamma * np.maximum(Dxx, 0)).mean()
    kxy = np.exp(-gamma * np.maximum(Dxy, 0)).mean()
    return float(kxx + kyy_mean - 2 * kxy)


def main():
    ap = argparse.ArgumentParser(description="TASK 6 Part 4 — MMD embedding baseline")
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    P = yaml.safe_load(open(args.params))
    E = P["embeddings"]
    vf = P["window"]["vocab_fraction"]; rf = P["window"]["reference_fraction"]
    tw = P["window"]["words_per_window"]
    grid = sorted(float(d) for d in P["adwin"]["delta_grid"])
    far_targets = [float(P["adwin"]["target_far"])] + [float(x) for x in P["adwin"]["far_sensitivity"]]
    demo = args.demo
    os.makedirs(EMB_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)

    df = pd.read_parquet(f"data/interim/{STREAM}.parquet",
                         columns=["doc_id", "text", "date", "n_words"])
    if demo:
        df = df.head(6000).reset_index(drop=True)
    n = len(df)
    nwords = df["n_words"].to_numpy(np.int64)
    dates_ns = pd.to_datetime(df["date"]).values.astype("datetime64[ns]").astype(np.int64)

    cache = f"{EMB_DIR}/{STREAM}{'_demo' if demo else ''}.npy"
    if os.path.exists(cache) and np.load(cache, mmap_mode="r").shape[0] == n:
        log(f"[embeddings] using cached {cache}")
        emb = np.load(cache).astype(np.float32)
    else:
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            msg = (f"torch/sentence-transformers not available ({type(e).__name__}: {e}). "
                   "Install for the RTX 5060 Ti (Blackwell, needs CUDA 12.8+):\n"
                   "  pip install torch --index-url https://download.pytorch.org/whl/cu128\n"
                   "  pip install sentence-transformers\n"
                   "then re-run: python src/embeddings.py --params params.yaml")
            log("[embeddings] BLOCKED: " + msg)
            with open(f"{RES_DIR}/embeddings_status.json", "w") as fh:
                json.dump({"status": "blocked", "reason": msg}, fh, indent=2)
            return 0
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"[embeddings] encoding {n:,} docs with {E['model']} on {dev} ...")
        model = SentenceTransformer(E["model"], device=dev)
        model.max_seq_length = E["max_seq_length"]
        texts = df["text"].tolist()
        budget_s = E["encode_budget_min"] * 60
        emb = model.encode(texts, batch_size=E["batch_size"], convert_to_numpy=True,
                           show_progress_bar=True, normalize_embeddings=False).astype(np.float32)
        np.save(cache, emb.astype(np.float16))
        log(f"[embeddings] encoded in {time.time()-t0:.0f}s -> {cache} "
            f"({'over' if time.time()-t0 > budget_s else 'within'} budget)")

    # reference pool: 2000 docs from the reference epoch of the real stream
    perms = {0: np.arange(n)}
    if not demo:
        npz = np.load(f"data/streams/{STREAM}_perms.npz")
        perms[0] = npz["perm_00"].astype(np.int64)
        for k in range(1, 11):
            perms[k] = npz[f"perm_{k:02d}"].astype(np.int64)
    else:
        for k in range(1, 4):
            perms[k] = np.random.default_rng(40 + k).permutation(n)
    b0 = window_bounds(nwords[perms[0]], tw)
    nv0, nre0 = epoch_bounds(len(b0) - 1, vf, rf)
    ref_docs = perms[0][int(b0[nv0]):int(b0[nre0])]  # reference epoch [5%,10%)
    rng = np.random.default_rng(P["seed"])
    pool_n = min(E["ref_pool_size"], len(ref_docs))
    pool = rng.choice(ref_docs, pool_n, replace=False)
    Y = emb[pool]
    # median-heuristic bandwidth, frozen on the pool
    yy = (Y * Y).sum(1)
    Dyy = np.maximum(yy[:, None] + yy[None, :] - 2 * Y @ Y.T, 0)
    # cast to native float: embeddings are float32, so np.median returns a np.float32
    # which json.dump cannot serialize (this is what truncated the calibration file).
    med = float(np.median(np.sqrt(Dyy[np.triu_indices(len(Y), 1)])))
    gamma = 1.0 / (2 * med * med + 1e-12)
    kyy_mean = float(np.exp(-gamma * Dyy).mean())
    log(f"[embeddings] ref pool={pool_n}, median dist={med:.3f}, gamma={gamma:.5f}")

    calib_nulls = []
    max_w = E["max_docs_per_window"]
    for k in sorted(perms):
        order = perms[k]
        bounds = window_bounds(nwords[order], tw)
        nwin = len(bounds) - 1
        nv, nre = epoch_bounds(nwin, vf, rf)
        s5 = np.empty(nwin); md = np.empty(nwin, np.int64)
        do = dates_ns[order]
        for wi in range(nwin):
            s, e = bounds[wi], bounds[wi + 1]
            docs = order[s:e]
            if len(docs) > max_w:
                docs = rng.choice(docs, max_w, replace=False)
            s5[wi] = rbf_mmd2(emb[docs], Y, gamma, kyy_mean)
            md[wi] = int(np.median(do[s:e]))
        z = zscore_split(s5, nv, nre)
        out = pd.DataFrame({"window_idx": np.arange(nwin, dtype=np.int32),
                            "median_date": pd.to_datetime(md), "S5": s5, "z_S5": z})
        out.to_parquet(f"{EMB_DIR}/s5__perm{k:02d}{'_demo' if demo else ''}.parquet", index=False)
        if k != 0:
            zz = z[nre:]; calib_nulls.append(zz[~np.isnan(zz)])
    curve, tot = far_curve(calib_nulls, grid)
    picks = {}
    for tf in far_targets:
        d = pick_delta(curve, tf)
        picks[f"{tf:g}"] = {"delta": (float(d) if d is not None else None),
                            "far": (float(curve[d]) if d is not None else None)}
    with open(f"{RES_DIR}/embeddings_calibration.json", "w") as fh:
        json.dump({"grid": grid, "far_targets": far_targets,
                   "S5": {"far_curve": {f"{g:g}": float(curve[g]) for g in grid},
                          "targets": picks, "gamma": float(gamma), "median_dist": float(med),
                          "ref_pool_size": int(pool_n), "null_windows": int(tot)}}, fh, indent=2)
    with open(f"{RES_DIR}/embeddings_status.json", "w") as fh:
        json.dump({"status": "ok", "n_docs": int(n), "gamma": float(gamma)}, fh, indent=2)
    log(f"[embeddings] done. delta*(1e-3)={picks[f'{float(P['adwin']['target_far']):g}']['delta']}. "
        f"Wall-clock {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
