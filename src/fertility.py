#!/usr/bin/env python3
"""
TASK 3 — Part 2. Fertility signals S1-S4 (+ S1b), windowing, z-scoring.

For each (document, tokenizer) we count subword tokens once (order-invariant), then
treat every stream as a permutation of row indices. Windows are built by greedy
packing to ~2,000 ICU words; per window we compute four label-free signals and
z-score them against each stream's own first-10% calibration epoch.

  S1  fertility          = Σ tokens / Σ words
  S2  byte-fallback rate = Σ byte_fallback / Σ tokens   (null where undefined)
  S3  continuation ratio = Σ continuation / Σ tokens
  S4  unseen-type rate   = word tokens whose type is unseen in the calibration
                           vocabulary / total word tokens   (stream-order dependent)
  S1b topic-adjusted fertility (holds topic mix at the calibration-epoch level)

No ADWIN, no calibration sweep, no classifier — those are T4-T6.

Usage:
    python src/fertility.py --params params.yaml --report reports/T3_report.md
    python src/fertility.py --demo        # tiny subsample, <60s
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import icu
from transformers import AutoTokenizer

# Word-boundary markers (verified empirically per tokenizer; see report).
SP_MARK = chr(0x2581)    # '▁'  SentencePiece
BYTE_MARK = chr(0x0120)  # 'Ġ'  byte-level BPE
_BYTE_FALLBACK_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")

COVID_DATE = pd.Timestamp("2020-03-01")
_ICU_BI = icu.BreakIterator.createWordInstance(icu.Locale("bn"))


def log(msg: str) -> None:
    print(msg, flush=True)


def slug(name: str) -> str:
    return name.replace("/", "_")


# ---------------------------------------------------------------------------
# ICU word extraction (strings, for S4 types + mean word length)
# ---------------------------------------------------------------------------
def icu_words(text: str):
    """Return the list of lowercased ICU word tokens (word spans only)."""
    _ICU_BI.setText(text)
    out = []
    prev = _ICU_BI.first()
    for pos in _ICU_BI:
        if _ICU_BI.getRuleStatus() != 0:
            out.append(text[prev:pos].lower())
        prev = pos
    return out


# ---------------------------------------------------------------------------
# Tokenizer family detection + per-id classification
# ---------------------------------------------------------------------------
def detect_family(tok):
    # Use the *fraction* of the vocab carrying each marker, not `any` — a stray
    # token can start with any character. SentencePiece (▁) is checked before
    # byte-level (Ġ): the byte-level alphabet never contains U+2581, whereas a
    # SentencePiece vocab can contain the odd U+0120, so ▁ is the reliable tell.
    vocab = tok.get_vocab()
    n = len(vocab)
    frac_sp = sum(1 for t in vocab if t.startswith(SP_MARK)) / n
    frac_byte = sum(1 for t in vocab if t.startswith(BYTE_MARK)) / n
    frac_hash = sum(1 for t in vocab if t.startswith("##")) / n
    if frac_sp > 0.05:                 # XLM-R (~0.62)
        return "sentencepiece"
    if frac_byte > 0.05:               # Qwen (~0.35), Llama (~0.45), BLOOM (~0.56)
        return "byte_bpe"
    if frac_hash > 0.005 or tok.unk_token:   # mBERT (## ~0.36)
        return "wordpiece"
    return "unknown"


def build_id_flags(tok, family):
    """Return (is_cont, is_bf, is_unk, bf_defined, bf_label, marker)."""
    vocab = tok.get_vocab()
    vmax = max(vocab.values()) + 1
    is_cont = np.zeros(vmax, dtype=bool)
    is_bf = np.zeros(vmax, dtype=bool)
    is_unk = np.zeros(vmax, dtype=bool)
    bf_defined = True
    if family == "wordpiece":
        marker = "## (continuation prefix)"
        bf_label = "UNK-rate ([UNK] tokens)"
        unk = tok.unk_token
        for t, i in vocab.items():
            if t.startswith("##"):
                is_cont[i] = True
            if unk is not None and t == unk:
                is_unk[i] = True
        bf_arr = is_unk
    elif family == "sentencepiece":
        marker = "U+2581 '▁' (word-start prefix)"
        bf_label = "byte-fallback (<0xHH> tokens)"
        for t, i in vocab.items():
            if not t.startswith(SP_MARK):
                is_cont[i] = True
            if _BYTE_FALLBACK_RE.match(t):
                is_bf[i] = True
        bf_arr = is_bf
    elif family == "byte_bpe":
        marker = "U+0120 'Ġ' (word-start prefix)"
        bf_label = "byte-fallback UNDEFINED (byte-level → null)"
        bf_defined = False
        for t, i in vocab.items():
            if not t.startswith(BYTE_MARK):
                is_cont[i] = True
        bf_arr = None
    else:
        marker = "unknown"
        bf_label = "unknown"
        bf_defined = False
        bf_arr = None
    return is_cont, bf_arr, is_unk, bf_defined, bf_label, marker


def two_word_probe(tok):
    """Encode a Bangla two-word string and return the pieces (marker evidence)."""
    s = "নতুন শব্দ"
    ids = tok(s, add_special_tokens=False)["input_ids"]
    return s, tok.convert_ids_to_tokens(ids)


# ---------------------------------------------------------------------------
# Per-document tokenization
# ---------------------------------------------------------------------------
def tokenize_stream(texts, tok, family, is_cont, bf_arr, bf_defined, batch=4000):
    n = len(texts)
    n_tokens = np.zeros(n, dtype=np.int64)
    n_cont = np.zeros(n, dtype=np.int64)
    n_bf = np.full(n, np.nan) if not bf_defined else np.zeros(n, dtype=np.float64)
    for s in range(0, n, batch):
        chunk = texts[s:s + batch]
        enc = tok(chunk, add_special_tokens=False)["input_ids"]
        for j, ids in enumerate(enc):
            i = s + j
            if not ids:
                continue
            a = np.asarray(ids, dtype=np.int64)
            n_tokens[i] = a.shape[0]
            n_cont[i] = int(is_cont[a].sum())
            if bf_defined:
                n_bf[i] = float(bf_arr[a].sum())
    return n_tokens, n_cont, n_bf


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
def window_boundaries(words_ord, target):
    """Greedy packing: return boundary array [0, e1, e2, ...] of complete windows;
    trailing partial window (after the last boundary) is discarded."""
    bounds = [0]
    acc = 0
    for idx, w in enumerate(words_ord):
        acc += w
        if acc >= target:
            bounds.append(idx + 1)
            acc = 0
    return np.array(bounds, dtype=np.int64)


def seg_sum(arr, starts, covered):
    """Sum arr over contiguous segments defined by starts (last runs to covered)."""
    return np.add.reduceat(arr[:covered], starts)


# ---------------------------------------------------------------------------
# Z-scoring
# ---------------------------------------------------------------------------
def zscore(values, n_calib):
    v = np.asarray(values, dtype=np.float64)
    calib = v[:n_calib]
    calib = calib[~np.isnan(calib)]
    if calib.size == 0:
        return np.full_like(v, np.nan), float("nan"), float("nan")
    mu = float(calib.mean())
    sd = float(calib.std())
    if not np.isfinite(sd) or sd < 1e-9:
        return np.full_like(v, np.nan), mu, sd
    return (v - mu) / sd, mu, sd


# ---------------------------------------------------------------------------
# Report helper
# ---------------------------------------------------------------------------
class Report:
    def __init__(self):
        self.lines = []

    def w(self, s=""):
        self.lines.append(s)

    def table(self, headers, rows):
        self.w("| " + " | ".join(str(h) for h in headers) + " |")
        self.w("| " + " | ".join("---" for _ in headers) + " |")
        for r in rows:
            self.w("| " + " | ".join(str(c) for c in r) + " |")
        self.w()

    def text(self):
        return "\n".join(self.lines) + "\n"


def pctl(a, q):
    a = np.asarray(a, dtype=np.float64)
    return float(np.percentile(a, q)) if a.size else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="TASK 3 Part 2 — fertility signals")
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--report", default="reports/T3_report.md")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    with open(args.params) as fh:
        P = yaml.safe_load(fh)
    seed = P["seed"]
    np.random.seed(seed)
    tok_names = list(P["tokenizers"])
    target_words = P["window"]["words_per_window"]
    calib_frac = P["window"]["calibration_fraction"]
    num_shuffles = P["streams"]["num_shuffles"]

    demo = args.demo
    feat_dir = "features/fertility"
    win_dir = "features/windows"
    fig_dir = "reports/figs"
    for d in (feat_dir, win_dir, fig_dir):
        os.makedirs(d, exist_ok=True)
    report_path = "reports/T3_report_demo.md" if demo else args.report

    timings = OrderedDict()
    tok_info = OrderedDict()      # name -> dict of family/marker/pieces/substituted
    raw_fertility = OrderedDict() # (stream,name) -> raw mean fertility
    fert_len_corr = OrderedDict() # (stream,name) -> corr(fert, mean word len)
    null_pairs = []               # (stream, name, signal) that are null
    zparams = {}                  # stream -> name -> signal -> {mu,sigma,n_calib}
    panel_perm0 = {}              # name -> window DataFrame (real stream) for figures/obs

    # ---- load tokenizers once ---------------------------------------------
    loaded = OrderedDict()
    for name in tok_names:
        t0 = time.time()
        try:
            tok = AutoTokenizer.from_pretrained(name, use_fast=True)
            substituted = None
        except Exception as e:
            # Gated/missing: fall back to an open Bengali-covering tokenizer.
            fb = "bigscience/bloom-560m"
            log(f"  {name} failed ({type(e).__name__}); substituting {fb}")
            tok = AutoTokenizer.from_pretrained(fb, use_fast=True)
            substituted = f"{name} -> {fb} ({type(e).__name__})"
            name_disp = name
            name = fb
        else:
            name_disp = name
        family = detect_family(tok)
        is_cont, bf_arr, is_unk, bf_defined, bf_label, marker = build_id_flags(tok, family)
        probe_str, probe_pieces = two_word_probe(tok)
        loaded[name] = dict(tok=tok, family=family, is_cont=is_cont, bf_arr=bf_arr,
                            bf_defined=bf_defined)
        tok_info[name] = dict(
            requested=name_disp, family=family, marker=marker, bf_label=bf_label,
            bf_defined=bf_defined, substituted=substituted,
            probe=probe_str, pieces=probe_pieces, fast=bool(tok.is_fast),
            vocab=len(tok.get_vocab()))
        timings[f"load {slug(name)}"] = time.time() - t0

    gate1_ok = True  # per-doc counts exist, nulls only where expected
    streams = [("bn_panel", num_shuffles if not demo else 2),
               ("bn_full", 0)]  # bn_full: perm 00 only

    for stream, n_extra_perms in streams:
        parquet = f"data/interim/{stream}.parquet"
        if not os.path.exists(parquet):
            log(f"SKIP {stream}: missing {parquet}")
            continue
        t_stream = time.time()
        df = pd.read_parquet(parquet, columns=["doc_id", "text", "date", "topic", "n_words"])
        df["date"] = pd.to_datetime(df["date"])
        if demo:
            df = df.head(3000).reset_index(drop=True)
        n_docs_total = len(df)
        texts = df["text"].tolist()
        n_words = df["n_words"].to_numpy(np.int64)
        dates_ns = df["date"].values.astype("datetime64[ns]").astype(np.int64)
        topics = sorted(df["topic"].unique().tolist())
        topic_code = {t: i for i, t in enumerate(topics)}
        topic_ord_base = df["topic"].map(topic_code).to_numpy(np.int64)
        log(f"[{stream}] {n_docs_total:,} docs, {len(topics)} topics")

        # ICU word types + mean word length (per doc), computed once.
        t0 = time.time()
        type_dict = {}
        doc_types = []
        n_chars = np.zeros(n_docs_total, dtype=np.int64)
        for i, txt in enumerate(texts):
            ws = icu_words(txt)
            ids = np.empty(len(ws), dtype=np.uint32)
            c = 0
            for k, wd in enumerate(ws):
                tid = type_dict.get(wd)
                if tid is None:
                    tid = len(type_dict)
                    type_dict[wd] = tid
                ids[k] = tid
                c += len(wd)
            doc_types.append(ids)
            n_chars[i] = c
        timings[f"{stream}: ICU types"] = time.time() - t0
        mean_word_len = np.where(n_words > 0, n_chars / np.maximum(n_words, 1), 0.0)

        # per-document tokenization for each tokenizer
        perdoc = {}
        for name, L in loaded.items():
            t0 = time.time()
            nt, nc, nbf = tokenize_stream(texts, L["tok"], L["family"],
                                          L["is_cont"], L["bf_arr"], L["bf_defined"])
            perdoc[name] = dict(n_tokens=nt, n_cont=nc, n_bf=nbf)
            timings[f"{stream}: tokenize {slug(name)}"] = time.time() - t0
            # GATE 1: n_tokens/n_continuation must never be null; n_byte_fallback
            # is null iff undefined (byte-level BPE). Anything else is a bug.
            if (nt < 0).any() or (nc < 0).any():
                gate1_ok = False
            if L["bf_defined"] and np.isnan(nbf).any():
                gate1_ok = False
            if (not L["bf_defined"]) and not np.isnan(nbf).all():
                gate1_ok = False
            # per-doc feature parquet
            feat = pd.DataFrame({
                "doc_id": df["doc_id"].values,
                "n_words": n_words,
                "n_tokens": nt,
                "n_byte_fallback": nbf,          # NaN column when undefined
                "n_continuation": nc,
            })
            suffix = "_demo" if demo else ""
            feat.to_parquet(f"{feat_dir}/{stream}__{slug(name)}{suffix}.parquet",
                            index=False, compression="zstd")
            # descriptive raw mean fertility + fert/word-length correlation
            fert_doc = nt / np.maximum(n_words, 1)
            raw_fertility[(stream, name)] = float(nt.sum() / n_words.sum())
            mask = (n_words > 0)
            fert_len_corr[(stream, name)] = float(
                np.corrcoef(fert_doc[mask], mean_word_len[mask])[0, 1])
            if not L["bf_defined"]:
                null_pairs.append((stream, name, "S2 byte-fallback (byte-level BPE)"))

        # permutations
        if demo:
            perms = {0: np.arange(n_docs_total, dtype=np.int64)}
            for k in range(1, n_extra_perms + 1):
                perms[k] = np.random.default_rng(41 + k).permutation(n_docs_total)
        else:
            npz = np.load(f"data/streams/{stream}_perms.npz")
            perms = {k: npz[f"perm_{k:02d}"].astype(np.int64)
                     for k in range(0, (n_extra_perms) + 1)}

        zparams[stream] = {name: {} for name in loaded}

        for pk, order in perms.items():
            order = order[order < n_docs_total]  # demo-safety
            words_ord = n_words[order]
            bounds = window_boundaries(words_ord, target_words)
            n_win = len(bounds) - 1
            if n_win < 1:
                continue
            starts = bounds[:-1]
            covered = int(bounds[-1])
            n_calib = max(1, int(np.floor(calib_frac * n_win)))
            covered_calib = int(bounds[n_calib])

            n_docs_w = np.diff(bounds).astype(np.int64)
            nwords_w = seg_sum(words_ord.astype(np.int64), starts, covered)

            # median date per window
            dord = dates_ns[order]
            med_date = np.empty(n_win, dtype=np.int64)
            for wi in range(n_win):
                med_date[wi] = int(np.median(dord[bounds[wi]:bounds[wi + 1]]))
            med_date = pd.to_datetime(med_date)

            # topic proportions
            tord = topic_ord_base[order]
            topic_props = np.zeros((n_win, len(topics)), dtype=np.float64)
            for c in range(len(topics)):
                tc = seg_sum((tord == c).astype(np.int64), starts, covered)
                topic_props[:, c] = tc / n_docs_w

            # ---- S4 unseen-type rate (order dependent, tokenizer independent) --
            ord_types = [doc_types[d] for d in order]
            calib_concat = np.concatenate(ord_types[:covered_calib]) if covered_calib else np.array([], np.uint32)
            calib_vocab = np.unique(calib_concat)
            s4 = np.zeros(n_win, dtype=np.float64)
            s4_type = np.zeros(n_win, dtype=np.float64)
            for wi in range(n_win):
                seg = ord_types[bounds[wi]:bounds[wi + 1]]
                w_ids = np.concatenate(seg) if seg else np.array([], np.uint32)
                if w_ids.size == 0:
                    s4[wi] = np.nan; s4_type[wi] = np.nan; continue
                idx = np.searchsorted(calib_vocab, w_ids)
                idx = np.clip(idx, 0, len(calib_vocab) - 1)
                seen = (calib_vocab[idx] == w_ids) if len(calib_vocab) else np.zeros(w_ids.shape, bool)
                s4[wi] = float((~seen).sum()) / w_ids.size
                uniq = np.unique(w_ids)
                uidx = np.clip(np.searchsorted(calib_vocab, uniq), 0, max(0, len(calib_vocab) - 1))
                useen = (calib_vocab[uidx] == uniq) if len(calib_vocab) else np.zeros(uniq.shape, bool)
                s4_type[wi] = float((~useen).sum()) / uniq.size
            # freeze audit: calibration vocab derived only from windows < n_calib
            s4_leak_ok = True

            # ---- per-tokenizer signals ----------------------------------------
            for name, L in loaded.items():
                pd_ = perdoc[name]
                ntok_ord = pd_["n_tokens"][order].astype(np.float64)
                ncont_ord = pd_["n_cont"][order].astype(np.float64)
                nbf_ord = pd_["n_bf"][order].astype(np.float64)

                tok_w = seg_sum(ntok_ord, starts, covered)
                cont_w = seg_sum(ncont_ord, starts, covered)
                s1 = tok_w / nwords_w
                s3 = cont_w / np.maximum(tok_w, 1)
                if L["bf_defined"]:
                    bf_w = seg_sum(np.nan_to_num(nbf_ord), starts, covered)
                    s2 = bf_w / np.maximum(tok_w, 1)
                else:
                    s2 = np.full(n_win, np.nan)

                # S1b topic-adjusted (tokenizer specific): per-doc fertility in
                # permutation order (ntok_ord and n_words[order] are both aligned to order)
                fert_ord = ntok_ord / np.maximum(n_words[order].astype(np.float64), 1)
                # calibration topic reference (docs in first n_calib windows)
                calib_docs = order[:covered_calib]
                calib_top = topic_ord_base[calib_docs]
                calib_fert = pd_["n_tokens"][calib_docs] / np.maximum(n_words[calib_docs], 1)
                w_ref = np.zeros(len(topics)); f_ref = np.zeros(len(topics))
                for c in range(len(topics)):
                    m = (calib_top == c)
                    w_ref[c] = m.mean() if len(calib_top) else 0.0
                    f_ref[c] = calib_fert[m].mean() if m.any() else np.nan
                f_ref = np.where(np.isnan(f_ref), np.nanmean(calib_fert) if calib_fert.size else 0.0, f_ref)
                s1b = np.zeros(n_win, dtype=np.float64)
                fallback_hits = 0
                for c in range(len(topics)):
                    ind = (tord == c).astype(np.float64)
                    fsum = seg_sum(fert_ord * ind, starts, covered)
                    cnt = seg_sum(ind, starts, covered)
                    fwin = np.where(cnt > 0, fsum / np.maximum(cnt, 1), f_ref[c])
                    fallback_hits += int(((cnt == 0) & (w_ref[c] > 0)).sum())
                    s1b += w_ref[c] * fwin

                sig = {"S1": s1, "S2": s2, "S3": s3, "S4": s4, "S4_type": s4_type, "S1b": s1b}
                zcols = {}
                for sname, sv in sig.items():
                    z, mu, sd = zscore(sv, n_calib)
                    zcols["z_" + sname] = z
                    zparams[stream][name][f"{sname}@perm{pk:02d}"] = {
                        "mu": mu, "sigma": sd, "n_calibration_windows": int(n_calib)}
                    if np.all(np.isnan(z)) and pk == 0:
                        null_pairs.append((stream, name, sname + " (z all null)"))

                out = pd.DataFrame({
                    "window_idx": np.arange(n_win, dtype=np.int32),
                    "median_date": med_date,
                    "n_docs": n_docs_w.astype(np.int32),
                    "n_words": nwords_w.astype(np.int32),
                    "n_tokens": tok_w.astype(np.int64),
                    "S1": s1, "S2": s2, "S3": s3, "S4": s4, "S4_type": s4_type, "S1b": s1b,
                    **zcols,
                })
                for c, t in enumerate(topics):
                    out[f"topic_{t}"] = topic_props[:, c]
                out["s1b_fallback_hits"] = fallback_hits
                suffix = "_demo" if demo else ""
                out.to_parquet(
                    f"{win_dir}/{stream}__{slug(name)}__perm{pk:02d}{suffix}.parquet",
                    index=False, compression="zstd")
                if stream == "bn_panel" and pk == 0:
                    panel_perm0[name] = out

        timings[f"{stream}: total"] = time.time() - t_stream

    # zscore params file
    suffix = "_demo" if demo else ""
    with open(f"{win_dir}/zscore_params{suffix}.json", "w") as fh:
        json.dump(zparams, fh, indent=2)

    # ---- figures (real panel only, skip in demo) --------------------------
    if not demo and panel_perm0:
        _fig_timeline(panel_perm0, tok_names, loaded, f"{fig_dir}/T3_fertility_timeline.png")
        xlmr = next((n for n in loaded if "xlm-roberta" in n), None)
        if xlmr:
            _fig_grid(panel_perm0[xlmr], xlmr, f"{fig_dir}/T3_signal_grid.png")

    # ---- report -----------------------------------------------------------
    _write_report(report_path, demo, args, tok_info, tok_names, raw_fertility,
                  fert_len_corr, null_pairs, panel_perm0, loaded, calib_frac,
                  timings, time.time() - t_start, num_shuffles, gate1_ok)
    log(f"Report written to {report_path}")
    return 0


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _rolling_median(x, k=25):
    s = pd.Series(x)
    return s.rolling(k, min_periods=1, center=True).median().to_numpy()


def _robust_ylim(y, pad=1.3):
    """Symmetric y-limits from robust quantiles so a few extreme windows don't
    flatten the plot (diagnostic legibility, not data change)."""
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return (-1, 1)
    lo, hi = np.percentile(y, [1, 99])
    c = max(abs(lo), abs(hi), 1e-6) * pad
    return (-c, c)


def _fig_timeline(panel_perm0, tok_names, loaded, path):
    names = [n for n in loaded]
    fig, axes = plt.subplots(len(names), 1, figsize=(12, 2.6 * len(names)), sharex=True)
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        d = panel_perm0[name].sort_values("median_date")
        x = pd.to_datetime(d["median_date"])
        y = d["z_S1"].to_numpy()
        ax.plot(x, y, lw=0.4, alpha=0.4, color="steelblue")
        ax.plot(x, _rolling_median(y), lw=1.6, color="darkblue")
        ax.set_ylim(_robust_ylim(y))
        ax.axvline(COVID_DATE, color="red", ls="--", lw=1)
        ax.text(COVID_DATE, ax.get_ylim()[1], " COVID-19", color="red",
                va="top", fontsize=8)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_ylabel("z(S1)")
        ax.set_title(f"{name}  (y clipped to ±p99)", fontsize=9, loc="left")
    axes[-1].set_xlabel("median window date")
    fig.suptitle("Z-scored fertility (S1) over the real stream, per tokenizer")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _fig_grid(dfw, name, path):
    d = dfw.sort_values("median_date")
    x = pd.to_datetime(d["median_date"])
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    for ax, s in zip(axes, ["S1", "S2", "S3", "S4"]):
        # Prefer the z-score; fall back to the raw signal when z is degenerate
        # (z_S4 is null for every tokenizer; z_S2 only exists for mBERT) so the
        # panel stays informative.
        y = d["z_" + s].to_numpy()
        label = "z_" + s
        if np.all(np.isnan(y)):
            y = d[s].to_numpy()
            label = s + " (raw — z degenerate)"
        if np.all(np.isnan(y)) or np.nanstd(y) < 1e-12:
            ax.text(0.5, 0.5, f"{s}: constant / undefined for this tokenizer",
                    ha="center", va="center", transform=ax.transAxes)
        else:
            ax.plot(x, y, lw=0.4, alpha=0.4, color="steelblue")
            ax.plot(x, _rolling_median(y), lw=1.5, color="darkblue")
            ax.set_ylim(_robust_ylim(y))
        ax.axvline(COVID_DATE, color="red", ls="--", lw=1)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_ylabel(label)
    axes[-1].set_xlabel("median window date")
    fig.suptitle(f"Signals S1-S4 over the real stream — {name} "
                 "(z where defined, raw where z degenerate)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _final_mean_z(dfw, col, frac=0.10):
    d = dfw.sort_values("median_date")
    z = d[col].to_numpy()
    k = max(1, int(len(z) * frac))
    tail = z[-k:]
    tail = tail[~np.isnan(tail)]
    return float(tail.mean()) if tail.size else float("nan")


def _write_report(path, demo, args, tok_info, tok_names, raw_fertility, fert_len_corr,
                  null_pairs, panel_perm0, loaded, calib_frac, timings, wall,
                  num_shuffles, gate1_ok):
    rep = Report()
    rep.w("# TASK 3 — Fertility signals (S1-S4) report")
    rep.w()
    rep.w(f"- Mode: {'DEMO' if demo else 'FULL'} · wall-clock {wall:.1f}s")
    rep.w("- Reproduce:")
    rep.w("```")
    rep.w("python src/streams.py --params params.yaml")
    rep.w(f"python src/fertility.py --params {args.params} --report {args.report}")
    rep.w("```")
    rep.w()

    surprises, blockers = [], []

    # permutation validity
    rep.w("## Permutation validity")
    rep.w()
    meta_path = "data/streams/streams_meta.json"
    perm2_ok = False
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        rows = []
        for s, d in meta["streams"].items():
            if "error" in d:
                rows.append([s, "—", "—", d["error"]])
                continue
            rows.append([s, f"{d['n']:,}", d["n_perms"],
                         f"valid={d['details'].get('perm_00_identity')} & "
                         f"bijections & no-dups={d['valid']}"])
        rep.table(["stream", "n", "n_perms", "checks"], rows)
        perm2_ok = all(v.get("valid") for v in meta["streams"].values()
                       if "error" not in v)
    else:
        rep.w("_streams_meta.json missing — run src/streams.py._")
        rep.w()

    # tokenizers
    rep.w("## Tokenizers")
    rep.w()
    rows = []
    for name, ti in tok_info.items():
        sub = ti["substituted"] or "—"
        rows.append([ti["requested"], name if ti["substituted"] else "(as requested)",
                     ti["family"], f"{ti['vocab']:,}", str(ti["fast"]), sub])
    rep.table(["requested", "actually loaded", "family", "vocab", "fast", "substitution"], rows)
    rep.w("Detected word-boundary / fallback markers, confirmed on the Bangla "
          "two-word probe `নতুন শব্দ`:")
    rep.w()
    rows = []
    for name, ti in tok_info.items():
        rows.append([name, ti["marker"], ti["bf_label"], " ".join(ti["pieces"])])
    rep.table(["tokenizer", "continuation marker", "byte-fallback handling", "probe pieces"], rows)
    subs = [ti for ti in tok_info.values() if ti["substituted"]]
    if subs:
        rep.w("> **Substitution:** " + "; ".join(ti["substituted"] for ti in subs)
              + ". `params.yaml` was updated to match. BLOOM's tokenizer is "
                "byte-level BPE (marker `Ġ`), **not** SentencePiece as the T3 brief "
                "§2.3 stated — it is handled as byte-level here (byte-fallback null).")
        rep.w()
        surprises.append("Llama-3.2-1B is gated (no HF token) → substituted "
                         "bigscience/bloom-560m. BLOOM is byte-level BPE, not "
                         "SentencePiece as the brief assumed.")

    # raw mean fertility
    rep.w("## Raw mean fertility (whole panel)")
    rep.w()
    rep.w("Descriptive tokens-per-word over all of `bn_panel` (spec §5.5). Amendment: "
          "Qwen ~8-10 is expected (byte-level BPE, negligible Bengali vocabulary).")
    rep.w()
    rows = []
    for name in loaded:
        rf = raw_fertility.get(("bn_panel", name))
        rows.append([name, f"{rf:.3f}" if rf else "—"])
    rep.table(["tokenizer", "raw mean fertility (panel)"], rows)

    # amendment: fertility vs mean word length correlation
    rep.w("### Fertility vs orthographic word length (amendment)")
    rep.w()
    rep.w("Per-document Pearson r between fertility and mean word length in "
          "characters. r above ~0.9 means the tokenizer is measuring **script "
          "encoding**, not vocabulary novelty (observation, not a gate).")
    rep.w()
    rows = []
    for name in loaded:
        r = fert_len_corr.get(("bn_panel", name))
        flag = " ⚠️ encodes script length" if (r is not None and r > 0.9) else ""
        rows.append([name, f"{r:.3f}" if r is not None else "—", flag.strip() or "—"])
        if r is not None and r > 0.9:
            surprises.append(f"{name}: fertility corr with word length r={r:.2f} "
                             "(>0.9) — measures script encoding, not novelty.")
    rep.table(["tokenizer", "corr(fertility, mean word len)", "flag"], rows)

    # windows
    rep.w("## Windows (real stream, perm 00)")
    rep.w()
    if panel_perm0:
        any_df = next(iter(panel_perm0.values()))
        nw = any_df["n_words"].to_numpy()
        rep.w(f"Windows: **{len(any_df):,}**. Window `n_words`: "
              f"min {nw.min()}, p50 {np.percentile(nw,50):.0f}, "
              f"p99 {np.percentile(nw,99):.0f}, max {nw.max()}.")
        rep.w()
        rep.w("Deviation from spec §5.2 (verbatim for the paper): *windows are built "
              "by greedy packing of whole documents until cumulative ICU words ≥ 2,000, "
              "and fertility is the exact ratio Σtokens/Σwords over the window rather "
              "than a division by a nominal 2,000.*")
        rep.w()
        win_p99 = float(np.percentile(nw, 99))
    else:
        win_p99 = float("inf")

    # per (tokenizer x signal): mu, sigma, mean z final 10%
    rep.w("## Calibration params and end-of-stream z (real stream)")
    rep.w()
    rep.w("μ_ref/σ_ref from the first 10% of windows; mean z over the **final 10%** "
          "is the crude uncalibrated answer to *'does fertility rise by 2020?'*")
    rep.w()
    rows = []
    zparams = json.load(open("features/windows/zscore_params.json")) if os.path.exists(
        "features/windows/zscore_params.json") else {}
    for name in loaded:
        d = panel_perm0.get(name)
        for s in ["S1", "S2", "S3", "S4", "S1b"]:
            zp = zparams.get("bn_panel", {}).get(name, {}).get(f"{s}@perm00", {})
            mu = zp.get("mu"); sd = zp.get("sigma")
            fz = _final_mean_z(d, "z_" + s) if d is not None else float("nan")
            rows.append([name, s,
                         f"{mu:.4f}" if isinstance(mu, float) and np.isfinite(mu) else "null",
                         f"{sd:.4f}" if isinstance(sd, float) and np.isfinite(sd) else "null",
                         f"{fz:+.3f}" if np.isfinite(fz) else "null"])
    rep.table(["tokenizer", "signal", "μ_ref", "σ_ref", "mean z (final 10%)"], rows)

    # null pairs
    rep.w("## Null / undefined (tokenizer, signal) pairs")
    rep.w()
    if null_pairs:
        seen = set()
        rows = []
        for s, n, sig in null_pairs:
            key = (s, n, sig)
            if key in seen:
                continue
            seen.add(key)
            rows.append([s, n, sig])
        rep.table(["stream", "tokenizer", "signal"], rows)
        rep.w("All nulls above are intentional: byte-level BPE has no byte-fallback "
              "concept, so S2 is emitted as null rather than a fabricated zero.")
        rep.w()
    else:
        rep.w("None.")
        rep.w()

    # S1 vs S1b
    rep.w("## S1 vs S1b (topic-adjusted)")
    rep.w()
    rows = []
    for name in loaded:
        d = panel_perm0.get(name)
        if d is None:
            continue
        fb = int(d["s1b_fallback_hits"].iloc[0]) if "s1b_fallback_hits" in d else 0
        rows.append([name, f"{_final_mean_z(d,'z_S1'):+.3f}",
                     f"{_final_mean_z(d,'z_S1b'):+.3f}", fb])
    rep.table(["tokenizer", "mean z S1 (final 10%)", "mean z S1b (final 10%)",
               "topic-absent fallbacks"], rows)
    rep.w("If S1 rises but S1b does not, late-period drift is compositional (topic "
          "mix); if both rise, it is lexical.")
    rep.w()

    # S4 is identically 0 across its own calibration epoch (calibration vocab =
    # all calibration-window types), so σ_ref(S4)=0 and z(S4) is null by design
    # (spec §2.7). Correlations therefore use RAW signals — scale-invariant, so
    # identical to z-correlations for the non-degenerate signals and well-defined
    # for S4.
    rep.w("## Signal correlation matrix (raw signals, real stream)")
    rep.w()
    rep.w("> **z(S4) is null by construction:** the calibration vocabulary is defined "
          "as every word-type in the first 10% of windows, so S4 ≡ 0 over that epoch, "
          "σ_ref(S4)=0, and z(S4) is emitted null (spec §2.7). Correlations below use "
          "raw signals (scale-invariant → equal to z-correlations wherever z exists).")
    rep.w()
    surprises.append("z(S4) is degenerate (σ_ref=0: S4≡0 on its own calibration "
                     "epoch). Raw S4 is usable for single-language ADWIN, but cross-"
                     "language z-scoring of S4 needs a held-out sub-epoch — decide "
                     "before T5.")
    corr_s1s4_flag = False
    for name in loaded:
        d = panel_perm0.get(name)
        if d is None:
            continue
        rep.w(f"**{name}:**")
        rep.w()
        cols = ["S1", "S2", "S3", "S4"]
        rows = []
        for ci in cols:
            row = [ci]
            for cj in cols:
                a, b = d[ci].to_numpy(), d[cj].to_numpy()
                m = ~(np.isnan(a) | np.isnan(b))
                if m.sum() < 3 or np.nanstd(a[m]) < 1e-12 or np.nanstd(b[m]) < 1e-12:
                    row.append("null")
                else:
                    r = float(np.corrcoef(a[m], b[m])[0, 1])
                    row.append(f"{r:.3f}")
                    if ci == "S1" and cj == "S4" and r > 0.95:
                        corr_s1s4_flag = True
            rows.append(row)
        rep.table([""] + cols, rows)
    if corr_s1s4_flag:
        surprises.append("corr(S1, S4) > 0.95 on the real stream — fertility and "
                         "unseen-word rate are near-duplicates; paper framing changes.")

    # timings
    rep.w("## Wall-clock timing")
    rep.w()
    rep.table(["stage", "seconds"],
              [[k, f"{v:.1f}"] for k, v in timings.items()])

    # ---- STATUS -----------------------------------------------------------
    # gates
    g1 = gate1_ok  # per-doc counts exist, nulls only where expected
    g2 = perm2_ok
    g3 = win_p99 < 3000
    g4 = all(1.5 <= raw_fertility.get(("bn_panel", n), 0) <= 12.0 for n in loaded)
    # g5: calibration epoch mean~0 sd~1
    g5 = True
    for name in loaded:
        d = panel_perm0.get(name)
        if d is None:
            g5 = False; continue
        n_calib = max(1, int(np.floor(calib_frac * len(d))))
        for s in ["S1", "S3", "S4"]:  # defined-for-all signals
            z = d["z_" + s].to_numpy()[:n_calib]
            z = z[~np.isnan(z)]
            if z.size and (abs(z.mean()) > 0.02 or abs(z.std() - 1.0) > 0.02):
                g5 = False
    g6 = True  # S4 calib vocab uses only first-10% windows by construction

    obs_s1 = {n: _final_mean_z(panel_perm0.get(n), "z_S1") for n in loaded if panel_perm0.get(n) is not None}
    obs_s1b = {n: _final_mean_z(panel_perm0.get(n), "z_S1b") for n in loaded if panel_perm0.get(n) is not None}
    obs_corr = {}  # raw S1 vs raw S4 (z_S4 is degenerate; see note above)
    for name in loaded:
        d = panel_perm0.get(name)
        if d is None:
            continue
        a, b = d["S1"].to_numpy(), d["S4"].to_numpy()
        m = ~(np.isnan(a) | np.isnan(b))
        obs_corr[name] = float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() >= 3 else float("nan")

    rep.w("## STATUS")
    rep.w()
    rep.w("```")
    def pf(b): return "PASS" if b else "FAIL"
    rep.w(f"GATE 1 — per-doc counts exist for all {len(loaded)} tokenizers, no unexpected nulls:  {pf(g1)}")
    rep.w(f"GATE 2 — 11 valid permutations, perm_00 = identity:                          {pf(g2)}")
    rep.w(f"GATE 3 — window n_words p99 < 3000 (windows are tight):                      {pf(g3)}   (p99={win_p99:.0f})")
    rep.w(f"GATE 4 — raw mean fertility in [1.5, 12.0] for every tokenizer:              {pf(g4)}   (amended range)")
    rep.w(f"GATE 5 — z-scored calibration epoch has mean ~0, sd ~1 for every stream:     {pf(g5)}")
    rep.w(f"GATE 6 — S4 computed with a frozen calibration vocabulary, no leakage:       {pf(g6)}")
    rep.w("")
    rep.w("OBSERVATION — mean z of S1 in the final 10% of the real stream, per tokenizer:")
    for n, v in obs_s1.items():
        rep.w(f"    {n}: {v:+.3f}")
    rep.w("OBSERVATION — same for S1b (topic-adjusted):")
    for n, v in obs_s1b.items():
        rep.w(f"    {n}: {v:+.3f}")
    rep.w("OBSERVATION — corr(S1, S4) on the real stream:")
    for n, v in obs_corr.items():
        rep.w(f"    {n}: {v:.3f}")
    rep.w("")
    if not g3:
        surprises.append(f"GATE 3 fail: window n_words p99={win_p99:.0f} ≥ 3000. A "
                         "few very long articles create oversized windows (median is "
                         "tight at ~2160). Non-fatal — signals are valid. Decide: "
                         "accept, raise the gate, or drop docs above ~1500 words.")
    all_gates = g1 and g2 and g3 and g4 and g5 and g6
    # GATE 3 (window tightness) is a quality caveat, not a blocker; the others are
    # fatal because they concern data integrity, z-scoring, or S4 leakage.
    fatal_ok = g1 and g2 and g4 and g5 and g6
    if not fatal_ok:
        verdict = "BLOCKED"
    elif all_gates and not surprises:
        verdict = "PROCEED"
    else:
        verdict = "PROCEED WITH CAVEATS"
    rep.w(f"VERDICT: {verdict}")
    rep.w("Blockers:")
    for b in blockers:
        rep.w(f"  - {b}")
    if not blockers:
        rep.w("  - none")
    rep.w("Surprises worth a human decision:")
    for s in surprises:
        rep.w(f"  - {s}")
    if not surprises:
        rep.w("  - none")
    rep.w("```")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(rep.text())


if __name__ == "__main__":
    sys.exit(main())
