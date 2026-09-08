#!/usr/bin/env python3
"""
TASK 3 — Part 1. Stream index permutations.

A "stream" is just a permutation of the (already chronological) row indices of a
prepared parquet. Stream 0 is the identity (the real chronological order); streams
1..N are full-array random shuffles that serve as the temporal null for T5's
false-alarm-rate calibration. Because fertility is invariant to stream order, these
index arrays are all we need — we never re-materialise the documents.

Usage:
    python src/streams.py --params params.yaml
    python src/streams.py --demo        # small, fast smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import yaml

STREAMS = ["bn_panel", "bn_full"]


def log(msg: str) -> None:
    print(msg, flush=True)


def n_rows(parquet_path: str) -> int:
    return pq.read_metadata(parquet_path).num_rows


def build_perms(n: int, num_shuffles: int):
    """Return (arrays dict, seeds int32 array). perm_00 is identity; perm_k uses
    seed 42+k-1. seeds[0] = -1 marks the identity (no seed)."""
    arrays = {"perm_00": np.arange(n, dtype=np.int32)}
    seeds = [-1]
    for k in range(1, num_shuffles + 1):
        seed = 41 + k  # 42, 43, ..., 51
        perm = np.random.default_rng(seed).permutation(n).astype(np.int32)
        arrays[f"perm_{k:02d}"] = perm
        seeds.append(seed)
    return arrays, np.array(seeds, dtype=np.int32)


def verify(arrays: dict, n: int):
    """Return (ok, details). Checks bijection, identity, pairwise uniqueness."""
    ref = np.arange(n, dtype=np.int32)
    details = {}
    all_ok = True
    for name, arr in arrays.items():
        bijection = arr.shape == (n,) and np.array_equal(np.sort(arr), ref)
        details[name] = {"bijection": bool(bijection), "len": int(arr.shape[0])}
        all_ok = all_ok and bijection
    identity_ok = np.array_equal(arrays["perm_00"], ref)
    details["perm_00_identity"] = bool(identity_ok)
    names = sorted(arrays)
    dup = False
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if np.array_equal(arrays[names[i]], arrays[names[j]]):
                dup = True
                details.setdefault("duplicates", []).append([names[i], names[j]])
    details["no_duplicates"] = not dup
    all_ok = all_ok and identity_ok and not dup
    return all_ok, details


def main():
    ap = argparse.ArgumentParser(description="TASK 3 Part 1 — stream permutations")
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    with open(args.params) as fh:
        P = yaml.safe_load(fh)
    num_shuffles = P["streams"]["num_shuffles"]

    out_dir = "data/streams"
    os.makedirs(out_dir, exist_ok=True)
    meta = {"num_shuffles": num_shuffles, "streams": {}}

    for stream in STREAMS:
        parquet = f"data/interim/{stream}.parquet"
        if not os.path.exists(parquet):
            log(f"  SKIP {stream}: {parquet} not found")
            meta["streams"][stream] = {"error": "parquet missing"}
            continue
        n = n_rows(parquet)
        if args.demo:
            n = min(n, 3000)
        arrays, seeds = build_perms(n, num_shuffles)
        ok, details = verify(arrays, n)
        suffix = "_demo" if args.demo else ""
        out = os.path.join(out_dir, f"{stream}_perms{suffix}.npz")
        np.savez(out, seeds=seeds, **arrays)
        meta["streams"][stream] = {
            "n": int(n), "n_perms": len(arrays), "seeds": seeds.tolist(),
            "valid": bool(ok), "details": details, "file": out,
        }
        log(f"  {stream}: n={n:,}, {len(arrays)} perms, valid={ok} -> {out}")

    if not args.demo:
        with open(os.path.join(out_dir, "streams_meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
    log("streams done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
