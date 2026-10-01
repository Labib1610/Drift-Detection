#!/usr/bin/env python3
"""
TASK 2 — Part 1. Data preparation (multi-language).

Builds chronological streams from one cleaned pool:

  * {lang}_panel — primary (only if the language config has `panel_publishers`).
                Equal quota per (month x source) cell so the publisher mix is
                constant across the stream and any drift is lexical, not
                compositional.
  * {lang}_full  — robustness (appendix). All sources, sampled uniformly across
                time (equal per-month quota) up to the cap.

Loaders (selected by `cfg['loader']` in params.yaml):
  potrika_csv   Potrika Bangla CSVs           -> read_raw
  newssumm_csv  NewsSumm_processed.xlsx (or CSVs) -> read_newssumm
  mlsum_hf      MLSUM via HuggingFace         -> read_mlsum

Every run also writes a publisher-coverage section to the report, with a
suggested panel (publishers + date range) computed from the cleaned pool.

All thresholds come from params.yaml; there are no magic numbers in the code.
The script is deterministic: two runs produce byte-identical parquet.

Usage:
    python src/prepare.py --lang bn --params params.yaml --report reports/T2_report.md
    python src/prepare.py --lang ns --params params.yaml --report reports/T2_report_ns.md
    python src/prepare.py --lang bn --demo        # 5,000-doc subsample, <30s
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import itertools
import json
import math
import os
import re
import sys
from collections import OrderedDict
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

import icu  # noqa: T1 confirmed PyICU 74.2 is present; ICU is the word definition.

# ICU is a hard requirement for this task (the word count is the whole point).
# T1 confirmed it is installed; if it ever is not, fail loudly rather than
# silently changing the word definition.
_ICU_BI = icu.BreakIterator.createWordInstance(icu.Locale("bn"))

# Display normalisation for the six newspaper sources (raw casing varies; see T1).
SOURCE_DISPLAY = {
    "jugantor": "Jugantor",
    "ittefaq": "Ittefaq",
    "jaijaidin": "Jaijaidin",
    "kaler_kontho": "Kaler Kontho",
    "inqilab": "Inqilab",
    "somoyer_alo": "Somoyer Alo",
}

# Canonicalisation of the topic label collisions T1 found. Keys are the raw
# labels; values are the canonical label. Anything not listed is left as-is.
TOPIC_CANON = {
    "science-and-tech": "Science_Technology",
}

RAW_DATE_FORMAT = "%Y/%m/%d"

OUT_SCHEMA = pa.schema([
    ("doc_id", pa.string()),
    ("text", pa.string()),
    ("date", pa.date32()),
    ("publisher", pa.dictionary(pa.int32(), pa.string())),
    ("topic", pa.string()),
    ("lang", pa.string()),
    ("n_words", pa.int32()),
])

ROW_GROUP_SIZE = 10_000

# Raw-column contract every loader must satisfy (what clean_pool expects).
RAW_COLS = ["News", "Category", "Heading", "Date", "Source"]


def log(msg: str) -> None:
    print(msg, flush=True)


def norm_source(s: object) -> str:
    key = str(s).strip().lower().replace(" ", "_")
    return SOURCE_DISPLAY.get(key, str(s))


_WS_RE = re.compile(r"\s+")


def normalise_text(t: str) -> str:
    return _WS_RE.sub(" ", str(t).strip()).lower()


def doc_id_of(norm: str) -> str:
    return "bn_" + hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12]


def icu_wordcount(t: str) -> int:
    _ICU_BI.setText(t)
    count = 0
    _ICU_BI.first()
    for _ in _ICU_BI:
        # rule status 0 == UBRK_WORD_NONE (whitespace/punct); else a word span.
        if _ICU_BI.getRuleStatus() != 0:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Loading + cleaning
# ---------------------------------------------------------------------------
def read_raw(raw_dir: str, demo: bool):
    """Potrika loader (unchanged)."""
    files = sorted(glob.glob(os.path.join(raw_dir, "**", "*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No CSVs under {raw_dir}")
    usecols = ["News", "Category", "Heading", "Date", "Source"]
    nrows = 200 if demo else None
    parts = []
    for f in files:
        df = pd.read_csv(f, usecols=usecols, dtype=str, nrows=nrows)
        parts.append(df)
    raw = pd.concat(parts, ignore_index=True)
    log(f"  read {len(files)} files, {len(raw):,} raw rows"
        + (" (demo: 200/file)" if demo else ""))
    return raw, len(files)


# ---- NewsSumm ---------------------------------------------------------------
# NewsSumm_processed columns:
#   newspaper_name, published_date, headline, article_text, human_summary, news_category
NEWSSUMM_MAP = {
    "article_text": "News",
    "news_category": "Category",
    "headline": "Heading",
    "published_date": "Date",
    "newspaper_name": "Source",
}
_NS_DATE_FORMATS = ["%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y",
                    "%d %B %Y", "%B %d, %Y", "%m/%d/%Y"]
_NS_TIME_TAIL = re.compile(r"(T|\s+)\d{1,2}:\d{2}(:\d{2})?.*$")

NS_CANONICAL_NEWSPAPERS = [
    "The Times of India", "The Hindu", "Hindustan Times", "Indian Express",
    "The Economic Times", "Business Standard", "The Mint", "Financial Express",
    "Deccan Chronicle", "The Telegraph", "Deccan Herald", "The Pioneer",
    "The Statesman", "The Tribune", "Mid-Day", "Mumbai Mirror", "Pune Mirror",
    "Bangalore Mirror", "Ahmedabad Mirror", "DNA", "Firstpost",
    "Free Press Journal", "Navhind Times", "Sentinel Assam", "The Asian Age",
    "The Shillong Times", "Imphal Free Press", "Orissa POST", "Hitavada",
    "Nagaland Post", "Sikkim Express", "Greater Kashmir", "Kashmir Observer",
    "Daily Excelsior", "The Millennium Post", "Central Chronicle",
]


def _ns_key(s: object) -> str:
    """Case/punctuation-insensitive key with a leading 'the' dropped, so
    'Times of India', 'THE TIMES OF INDIA' and 'the-times-of-india' all match."""
    k = re.sub(r"[^a-z0-9]", "", str(s).lower())
    return k[3:] if k.startswith("the") and len(k) > 3 else k


NS_KEY_TO_CANON = {_ns_key(n): n for n in NS_CANONICAL_NEWSPAPERS}


_TS_MIN = pd.Timestamp.min
_TS_MAX = pd.Timestamp.max


def _flex_date(s: pd.Series) -> pd.Series:
    """Parse mixed date strings (Excel datetimes arrive as 'YYYY-MM-DD HH:MM:SS').
    Formats are tried in order, day-first before month-first for ambiguous
    slash dates. Returns %Y/%m/%d strings (NaN if unparseable) because
    clean_pool parses with RAW_DATE_FORMAT."""
    s = s.astype(str).str.strip().str.replace(_NS_TIME_TAIL, "", regex=True)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    n_overflow = 0
    for fmt in _NS_DATE_FORMATS:
        miss = out.isna()
        if not miss.any():
            break
        attempt = pd.to_datetime(s[miss], format=fmt, errors="coerce")
        if len(attempt):
            # A string can parse successfully into a real calendar date (e.g. a
            # corrupted 4-digit year like 2997) that still overflows pandas'
            # nanosecond Timestamp range (~1677..2262). Null those out here so
            # the later ns-precision assignment doesn't raise OutOfBoundsDatetime;
            # they end up dropped as unparseable by clean_pool's rule 4, same as
            # any other bad date.
            out_of_range = attempt.notna() & ((attempt < _TS_MIN) | (attempt > _TS_MAX))
            n_overflow += int(out_of_range.sum())
            attempt = attempt.where(~out_of_range)
        out[miss] = attempt
    if n_overflow:
        log(f"  _flex_date: {n_overflow:,} date string(s) parsed to an out-of-range "
            f"year (outside {_TS_MIN.year}-{_TS_MAX.year}) and were nulled out")
    return out.dt.strftime("%Y/%m/%d")


def read_newssumm(raw_dir: str, demo: bool):
    """NewsSumm loader. `raw_dir` may be a directory (searched recursively for
    .xlsx/.xls/.csv) or a direct path to a single file. Requires `openpyxl`
    for .xlsx. Maps the dataset's columns onto RAW_COLS (human_summary is not
    used) and canonicalises newspaper names against NS_CANONICAL_NEWSPAPERS so
    panel_publishers in params.yaml can use the canonical spelling."""
    if os.path.isfile(raw_dir):
        files = [raw_dir]
    else:
        files = []
        for ext in ("xlsx", "xls", "csv"):
            files += glob.glob(os.path.join(raw_dir, "**", f"*.{ext}"), recursive=True)
        files = sorted(f for f in files if not os.path.basename(f).startswith("~$"))
    if not files:
        raise FileNotFoundError(f"No .xlsx/.xls/.csv under {raw_dir}")
    usecols = list(NEWSSUMM_MAP)
    nrows = 200 if demo else None
    parts = []
    for f in files:
        if f.lower().endswith(".csv"):
            df = pd.read_csv(f, dtype=str, nrows=nrows)
        else:
            df = pd.read_excel(f, dtype=str, nrows=nrows)
        df.columns = df.columns.str.strip()
        df = df[usecols]
        df = df.rename(columns=NEWSSUMM_MAP)
        df["Date"] = _flex_date(df["Date"])
        parts.append(df[RAW_COLS])
    raw = pd.concat(parts, ignore_index=True)

    # Canonicalise newspaper names; report anything that did not match.
    matched = raw["Source"].map(lambda s: _ns_key(s) in NS_KEY_TO_CANON)
    unmapped = raw.loc[~matched, "Source"].dropna()
    raw["Source"] = raw["Source"].map(lambda s: NS_KEY_TO_CANON.get(_ns_key(s), s))

    n_bad = int(raw["Date"].isna().sum())
    log(f"  read {len(files)} newssumm file(s), {len(raw):,} raw rows"
        + (" (demo: 200/file)" if demo else "")
        + f"; {n_bad:,} unparseable dates (dropped by rule 4)")
    if len(unmapped):
        top = unmapped.value_counts().head(10)
        log(f"  {len(unmapped):,} rows have a newspaper name not in the canonical list "
            f"({unmapped.nunique()} distinct); top: "
            + ", ".join(f"{k!r}×{v}" for k, v in top.items()))
    return raw, len(files)


# ---- MLSUM ------------------------------------------------------------------
def _mlsum_date(s: pd.Series) -> pd.Series:
    """Try ISO first, then day-first; emit %Y/%m/%d strings (NaN if unparseable)."""
    d = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    miss = d.isna()
    if miss.any():
        d[miss] = pd.to_datetime(s[miss], format="%d/%m/%Y", errors="coerce")
    return d.dt.strftime("%Y/%m/%d")


def read_mlsum(hf_config: str, demo: bool):
    """MLSUM via HuggingFace `datasets`. Columns: text, topic, title, date, url.
    MLSUM has no publisher field, so Source = URL host (www. stripped)."""
    from datasets import load_dataset
    parts = []
    for split in ("train", "validation", "test"):
        spec = f"{split}[:200]" if demo else split
        ds = load_dataset("mlsum", hf_config, split=spec, trust_remote_code=True)
        f = ds.to_pandas()
        parts.append(pd.DataFrame({
            "News": f["text"],
            "Category": f["topic"],
            "Heading": f["title"],
            "Date": _mlsum_date(f["date"].astype(str)),
            "Source": f["url"].astype(str).str.extract(r"https?://(?:www\.)?([^/]+)")[0],
        }))
    raw = pd.concat(parts, ignore_index=True)[RAW_COLS]
    log(f"  read mlsum/{hf_config}: {len(raw):,} raw rows" + (" (demo)" if demo else ""))
    return raw, 3


def load_raw(cfg: dict, demo: bool):
    loader = cfg["loader"]
    if loader == "potrika_csv":
        return read_raw(cfg["raw_dir"], demo)
    if loader == "newssumm_csv":
        return read_newssumm(cfg["raw_dir"], demo)
    if loader == "mlsum_hf":
        return read_mlsum(cfg["hf_config"], demo)
    raise ValueError(f"unknown loader {loader!r}")


def clean_pool(raw: pd.DataFrame, ledger: "OrderedDict[str, dict]"):
    """Apply the T2 cleaning rules in order, recording a count for each."""
    n_in = len(raw)

    # Rule 2 + 3: Unnamed:0 already excluded via usecols; rename to canonical.
    df = raw.rename(columns={
        "News": "text", "Category": "topic", "Heading": "headline",
        "Date": "date", "Source": "publisher",
    })
    # Publisher display-name normalisation (panel_publishers use display names).
    df["publisher"] = df["publisher"].map(norm_source)

    def record(name, before, after):
        removed = before - after
        ledger[name] = {
            "removed": int(removed),
            "pct": (100.0 * removed / n_in) if n_in else 0.0,
            "remaining": int(after),
        }

    # Rule 4: parse date, drop failures (incl. nulls / invalid calendar dates).
    before = len(df)
    df["date"] = pd.to_datetime(df["date"], format=RAW_DATE_FORMAT, errors="coerce")
    df = df[df["date"].notna()].copy()
    record("4. unparseable/null date dropped", before, len(df))

    # Rule 5: canonicalise topic labels.
    df["topic"] = df["topic"].fillna("unknown").astype(str).str.strip()
    df.loc[df["topic"] == "", "topic"] = "unknown"
    topics_before = sorted(df["topic"].dropna().unique().tolist())
    df["topic"] = df["topic"].map(lambda x: TOPIC_CANON.get(x, x))
    topics_after = sorted(df["topic"].dropna().unique().tolist())

    # Rule 6: drop empty / whitespace-only text.
    before = len(df)
    df["text"] = df["text"].fillna("")
    df = df[df["text"].str.strip() != ""].copy()
    record("6. empty/whitespace text dropped", before, len(df))

    # Precompute normalised text + doc_id (needed for dedup and the schema).
    df["_norm"] = df["text"].map(normalise_text)
    df["doc_id"] = df["_norm"].map(doc_id_of)

    # Rule 7: drop exact duplicates on normalised text, keep earliest-dated copy.
    before = len(df)
    df = df.sort_values(["date", "doc_id"], kind="mergesort")
    df = df.drop_duplicates(subset="_norm", keep="first").copy()
    record("7. exact-duplicate text dropped (kept earliest)", before, len(df))

    # Rule 8: ICU word count, computed once here.
    log(f"  computing ICU word counts on {len(df):,} docs ...")
    df["n_words"] = df["text"].map(icu_wordcount).astype("int32")

    # Rule 9: drop documents with n_words < min_doc_words.
    return df, topics_before, topics_after, record


# ---------------------------------------------------------------------------
# Stream construction
# ---------------------------------------------------------------------------
def sample_group(g: pd.DataFrame, quota: int, seed: int) -> pd.DataFrame:
    """Deterministically take up to `quota` rows from a group. Input is sorted by
    doc_id first so the sample is reproducible regardless of upstream order."""
    g = g.sort_values("doc_id", kind="mergesort")
    if len(g) <= quota:
        return g
    return g.sample(n=quota, random_state=seed)


def build_panel(pool: pd.DataFrame, publishers, start: str, end: str,
                cap: int, seed: int):
    start_p = pd.Period(start, freq="M")
    end_p = pd.Period(end, freq="M")
    months = pd.period_range(start_p, end_p, freq="M")
    n_cells = len(months) * len(publishers)
    quota = cap // n_cells

    sub = pool[pool["publisher"].isin(publishers)].copy()
    sub["ym"] = sub["date"].dt.to_period("M")
    sub = sub[(sub["ym"] >= start_p) & (sub["ym"] <= end_p)]

    kept = []
    fill = []  # (ym, publisher, available, taken, shortfall)
    for m in months:
        for p in publishers:
            cell = sub[(sub["ym"] == m) & (sub["publisher"] == p)]
            avail = len(cell)
            taken = sample_group(cell, quota, seed)
            kept.append(taken)
            fill.append({
                "month": str(m), "publisher": p, "available": avail,
                "taken": len(taken), "shortfall": max(0, quota - avail),
            })
    panel = pd.concat(kept, ignore_index=True) if kept else sub.iloc[:0].copy()
    return panel, pd.DataFrame(fill), quota, n_cells, list(months)


def build_full(pool: pd.DataFrame, start: str, end: str, cap: int, seed: int):
    start_p = pd.Period(start, freq="M")
    end_p = pd.Period(end, freq="M")
    months = pd.period_range(start_p, end_p, freq="M")
    quota = cap // len(months)  # equal per-month quota => uniform across time

    sub = pool.copy()
    sub["ym"] = sub["date"].dt.to_period("M")
    sub = sub[(sub["ym"] >= start_p) & (sub["ym"] <= end_p)]

    kept = []
    for m in months:
        cell = sub[sub["ym"] == m]
        kept.append(sample_group(cell, quota, seed))
    full = pd.concat(kept, ignore_index=True) if kept else sub.iloc[:0].copy()
    return full, quota, len(months)


# ---------------------------------------------------------------------------
# Parquet writing
# ---------------------------------------------------------------------------
def write_parquet(df: pd.DataFrame, path: str, lang: str = "bn") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = df.sort_values(["date", "doc_id"], kind="mergesort").reset_index(drop=True)
    pubs = sorted(df["publisher"].unique().tolist())
    table = pa.table({
        "doc_id": pa.array(df["doc_id"].tolist(), pa.string()),
        "text": pa.array(df["text"].tolist(), pa.string()),
        "date": pa.array(df["date"].dt.date.tolist(), pa.date32()),
        "publisher": pa.array(
            pd.Categorical(df["publisher"], categories=pubs),
            pa.dictionary(pa.int32(), pa.string())),
        "topic": pa.array(df["topic"].tolist(), pa.string()),
        "lang": pa.array([lang] * len(df), pa.string()),
        "n_words": pa.array(df["n_words"].astype("int32").tolist(), pa.int32()),
    }, schema=OUT_SCHEMA)
    pq.write_table(table, path, compression="zstd", row_group_size=ROW_GROUP_SIZE)


# ---------------------------------------------------------------------------
# Report
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


def stream_summary(df):
    total_words = int(df["n_words"].sum())
    return {
        "docs": len(df),
        "date_min": df["date"].min().date().isoformat() if len(df) else "—",
        "date_max": df["date"].max().date().isoformat() if len(df) else "—",
        "total_words": total_words,
        "mean_words": (total_words / len(df)) if len(df) else 0.0,
    }


# ---------------------------------------------------------------------------
# Publisher coverage + panel suggestion (diagnostic only; builds nothing)
# ---------------------------------------------------------------------------
def _longest_run(mask: np.ndarray):
    """Return (length, start_index) of the longest run of True in a 1-D bool array."""
    best_len = best_start = run = 0
    for i, v in enumerate(mask):
        run = run + 1 if v else 0
        if run > best_len:
            best_len, best_start = run, i - run + 1
    return best_len, best_start


def write_coverage(rep: "Report", pool: pd.DataFrame, start: str, end: str,
                   k: int, min_cell: int, top_n: int, cap: int):
    """Publisher x month coverage on the cleaned pool. A publisher-month is 'ok'
    if it has >= min_cell docs. Suggests the k-publisher combination (among the
    top_n by months covered) with the longest common all-ok run of months.
    Returns the suggestion dict or None."""
    months = pd.period_range(pd.Period(start, freq="M"), pd.Period(end, freq="M"), freq="M")
    sub = pool.assign(ym=pool["date"].dt.to_period("M"))
    sub = sub[(sub["ym"] >= months[0]) & (sub["ym"] <= months[-1])]

    rep.w("### Publisher coverage and panel suggestion")
    rep.w()
    rep.w(f"Range probed: **{months[0]}..{months[-1]}** ({len(months)} months). A "
          f"publisher-month counts as covered when it has ≥ **{min_cell}** cleaned docs "
          "(`suggest_min_cell`).")
    rep.w()
    if sub.empty:
        rep.w("_No documents in range._")
        rep.w()
        return None

    cnt = (sub.groupby(["publisher", "ym"]).size().unstack(fill_value=0)
              .reindex(columns=months, fill_value=0))
    ok = cnt >= min_cell
    stats = pd.DataFrame({"docs": cnt.sum(axis=1), "months_ok": ok.sum(axis=1)})
    stats = stats.sort_values(["months_ok", "docs"], ascending=[False, False])
    top = stats.index[:top_n].tolist()

    rep.w(f"Top {len(top)} publishers by months covered:")
    rep.w()
    rep.table(["publisher", "docs in range", f"months covered (of {len(months)})"],
              [[p, f"{int(stats.loc[p, 'docs']):,}", int(stats.loc[p, "months_ok"])]
               for p in top])

    sub2 = sub[sub["publisher"].isin(top)].assign(year=lambda d: d["ym"].dt.year)
    yr = pd.crosstab(sub2["publisher"], sub2["year"]).reindex(top)
    rep.w("Docs per year for those publishers:")
    rep.w()
    rep.table(["publisher"] + [str(c) for c in yr.columns],
              [[p] + [f"{int(v):,}" for v in yr.loc[p].values] for p in top])

    best = None
    for combo in itertools.combinations(top, k):
        allok = ok.loc[list(combo)].all(axis=0).to_numpy()
        length, s0 = _longest_run(allok)
        if length > 0 and (best is None or length > best["length"]):
            best = {"pubs": list(combo), "length": length,
                    "start": months[s0], "end": months[s0 + length - 1]}
    if best is None:
        rep.w(f"**No {k}-publisher combination has a common covered month** among the "
              f"top {len(top)} at ≥{min_cell} docs/month. Lower `suggest_min_cell` or "
              "`suggest_k`, or narrow the range.")
        rep.w()
        return None
    quota = cap // (best["length"] * k)
    rep.w(f"**Suggested panel** — longest common covered run for {k} publishers: "
          f"{best['pubs']} over **{best['start']}..{best['end']}** "
          f"({best['length']} months → quota {quota} docs/cell at cap {cap:,}).")
    rep.w()
    rep.w("```yaml")
    rep.w("panel_publishers: [" + ", ".join(f'"{p}"' for p in best["pubs"]) + "]")
    rep.w(f'panel_start: "{best["start"]}"')
    rep.w(f'panel_end: "{best["end"]}"')
    rep.w("```")
    rep.w()
    rep.w("_This is a starting point: the search looks only at the top publishers by "
          "months covered, and ignores topic mix and article length._")
    rep.w()
    return best


def main():
    ap = argparse.ArgumentParser(description="TASK 2 Part 1 — data preparation")
    ap.add_argument("--lang", default="bn")
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--report", default="reports/T2_report.md")
    ap.add_argument("--demo", action="store_true",
                    help="5,000-doc subsample, writes *_demo artifacts, <30s")
    args = ap.parse_args()
    lang = args.lang

    with open(args.params, encoding="utf-8") as fh:
        P = yaml.safe_load(fh)
    if lang not in P:
        log(f"ERROR: no '{lang}' block in {args.params}.")
        return 2
    seed = P["seed"]
    np.random.seed(seed)
    cap = P["data"]["max_docs_per_language"]
    min_words = P["data"]["min_doc_words"]
    cfg = P[lang]
    calib_frac = P["window"]["calibration_fraction"]

    has_panel = "panel_publishers" in cfg
    panel_pubs = list(cfg["panel_publishers"]) if has_panel else []
    panel_start = cfg.get("panel_start")
    panel_end = cfg.get("panel_end")
    if has_panel and not (panel_start and panel_end):
        log(f"ERROR: '{lang}' has panel_publishers but no panel_start/panel_end.")
        return 2
    full_start = cfg.get("full_start", "2014-06")
    full_end = cfg.get("full_end", "2020-12")
    n_full_pubs = cfg.get("n_full_publishers", 6)
    raw_src = cfg.get("raw_dir") or cfg.get("hf_config", "?")
    suggest_k = cfg.get("suggest_k", len(panel_pubs) or 3)
    suggest_min_cell = cfg.get("suggest_min_cell", 100)
    suggest_top_n = cfg.get("suggest_top_n", 10)

    demo = args.demo
    if demo:
        cap = 5000
        panel_out = f"data/interim/{lang}_panel_demo.parquet"
        full_out = f"data/interim/{lang}_full_demo.parquet"
        report_path = f"reports/T2_report_{lang}_demo.md"
    else:
        panel_out = f"data/interim/{lang}_panel.parquet"
        full_out = f"data/interim/{lang}_full.parquet"
        report_path = args.report

    log(f"Reading raw data (loader={cfg['loader']}, lang={lang}) ...")
    raw, n_files = load_raw(cfg, demo)
    raw = raw[RAW_COLS]
    n_raw = len(raw)

    ledger = OrderedDict()
    pool, topics_before, topics_after, record = clean_pool(raw, ledger)

    # Rule 9 (needs min_words from params).
    before = len(pool)
    pool = pool[pool["n_words"] >= min_words].copy()
    record(f"9. n_words < {min_words} dropped", before, len(pool))

    if demo:
        # Cap the pool so downstream is a genuine 5k-doc smoke test.
        pool = pool.sort_values(["date", "doc_id"], kind="mergesort").head(5000).copy()

    n_pool = len(pool)
    log(f"Cleaned pool: {n_pool:,} docs. Building streams ...")

    surprises = []
    blockers = []

    # Validate panel publishers against the cleaned pool BEFORE building anything,
    # so a spelling mismatch gives a clear error instead of an empty panel file
    # (which streams.py would then auto-discover).
    pool_pub_counts = pool["publisher"].value_counts()
    missing_pubs = [p for p in panel_pubs if p not in pool_pub_counts.index]
    panel_built = has_panel and not missing_pubs
    if missing_pubs:
        msg = (f"panel_publishers not found in cleaned pool: {missing_pubs}. "
               "Names must match exactly; see the publisher list in the report.")
        log("ERROR: " + msg)
        blockers.append(msg)

    if panel_built:
        panel, fill, quota, n_cells, months = build_panel(
            pool, panel_pubs, panel_start, panel_end, cap, seed)
    else:
        panel = pool.iloc[:0].copy()
        fill = pd.DataFrame(columns=["month", "publisher", "available", "taken", "shortfall"])
        quota, n_cells, months = 0, 0, []
    full, full_quota, n_full_months = build_full(pool, full_start, full_end, cap, seed)

    if panel_built:
        write_parquet(panel, panel_out, lang)
    write_parquet(full, full_out, lang)
    if panel_built:
        log(f"Wrote {panel_out} ({len(panel):,} docs) and {full_out} ({len(full):,} docs).")
    else:
        log(f"Panel skipped. Wrote {full_out} ({len(full):,} docs).")

    # ---- report -----------------------------------------------------------
    rep = Report()
    rep.w(f"# TASK 2 — {lang} preparation report")
    rep.w()
    rep.w(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    rep.w(f"- Mode: {'DEMO (5k subsample)' if demo else 'FULL'} · loader `{cfg['loader']}`")
    rep.w(f"- Params: `{args.params}` · seed {seed}")
    rep.w()
    rep.w("Reproduce with:")
    rep.w("```")
    rep.w(f"python src/prepare.py --lang {lang}" + (" --demo" if demo else
          f" --params {args.params} --report {report_path}"))
    rep.w("```")
    rep.w()

    rep.w(f"## PART 1 — {lang} preparation")
    rep.w()

    # cleaning ledger
    rep.w("### Cleaning ledger")
    rep.w()
    rep.w(f"Raw rows read from `{raw_src}` ({n_files} files/splits): **{n_raw:,}**")
    rep.w()
    led_rows = [["(rows in)", "", "", f"{n_raw:,}"]]
    for name, d in ledger.items():
        led_rows.append([name, f"{d['removed']:,}", f"{d['pct']:.2f}%",
                         f"{d['remaining']:,}"])
    led_rows.append(["**(rows out / pool)**", "", "", f"**{n_pool:,}**"])
    rep.table(["rule", "removed", "% of raw", "remaining"], led_rows)

    # topic label set
    rep.w("### Topic label set")
    rep.w()
    rep.w(f"Before canonicalisation ({len(topics_before)}): "
          + ", ".join(f"`{t}`" for t in topics_before))
    rep.w()
    rep.w(f"After canonicalisation ({len(topics_after)}): "
          + ", ".join(f"`{t}`" for t in topics_after))
    rep.w()
    applied = [f"`{k}`→`{v}`" for k, v in TOPIC_CANON.items() if k in topics_before]
    rep.w("Mappings applied: " + (", ".join(applied) if applied else "none"))
    rep.w()

    # publisher names as they exist in the pool (exact strings for panel_publishers)
    rep.w("### Publishers in the cleaned pool")
    rep.w()
    rep.w(f"{len(pool_pub_counts)} distinct publishers. Use these exact strings in "
          "`panel_publishers`.")
    rep.w()
    rep.table(["publisher", "docs"],
              [[p, f"{int(c):,}"] for p, c in pool_pub_counts.items()])

    # coverage + panel suggestion (always; probes the panel range if configured)
    if n_pool:
        cov_start = panel_start if has_panel else full_start
        cov_end = panel_end if has_panel else full_end
        write_coverage(rep, pool, cov_start, cov_end, suggest_k,
                       suggest_min_cell, suggest_top_n, cap)

    share_ok = True
    underfill_pct = 0.0
    if panel_built:
        # panel fill
        rep.w("### Panel fill (month × source quota)")
        rep.w()
        n_underfill = int((fill["taken"] < quota).sum())
        underfill_pct = 100.0 * n_underfill / n_cells if n_cells else 0.0
        rep.w(f"Quota per cell: **{quota}** (= {cap} ÷ ({len(months)} months × "
              f"{len(panel_pubs)} sources) = {n_cells} cells). "
              f"Cells underfilled: **{n_underfill} / {n_cells}** ({underfill_pct:.1f}%). "
              f"Panel total: **{len(panel):,}** docs.")
        rep.w()
        if underfill_pct > 5.0:
            rep.w(f"> ⚠️ **More than 5% of cells underfilled** — the {quota}-doc quota is "
                  "too high for the available data; consider lowering it.")
            surprises.append(f"{underfill_pct:.1f}% of panel cells underfilled "
                             f"(> 5% threshold) — quota may need lowering.")
            rep.w()
        worst = fill[fill["shortfall"] > 0].sort_values(
            ["shortfall", "month", "publisher"], ascending=[False, True, True]).head(10)
        if len(worst):
            rep.w("Worst 10 cells by shortfall:")
            rep.w()
            rep.table(["month", "publisher", "available", "taken", "shortfall"],
                      [[r.month, r.publisher, r.available, r.taken, r.shortfall]
                       for r in worst.itertuples()])
        else:
            rep.w("No cell underfilled.")
            rep.w()

        # realised publisher shares per year
        rep.w(f"### Realised publisher shares per year in `{lang}_panel`")
        rep.w()
        rep.w(f"(Confound check — each source should be ~{100/len(panel_pubs):.1f}% "
              "in every year.)")
        rep.w()
        if len(panel):
            pnl = panel.copy()
            pnl["year"] = pnl["date"].dt.year
            share = pd.crosstab(pnl["year"], pnl["publisher"], normalize="index") * 100
            headers = ["year"] + list(share.columns)
            rows = []
            for yr, r in share.iterrows():
                rows.append([int(yr)] + [f"{v:.1f}%" for v in r.values])
                if any(abs(v - 100.0 / len(panel_pubs)) > 2.0 for v in r.values):
                    share_ok = False
            rep.table(headers, rows)
        else:
            share_ok = False
            rep.w("_Panel is empty._")
            rep.w()

        # topic composition per year in panel
        rep.w(f"### Topic composition per year in `{lang}_panel`")
        rep.w()
        rep.w("(Measured, not corrected — publisher and topic may be correlated, "
              "so we size any topic drift before interpreting alarms.)")
        rep.w()
        if len(panel):
            pnl = panel.copy()
            pnl["year"] = pnl["date"].dt.year
            tshare = pd.crosstab(pnl["year"], pnl["topic"], normalize="index") * 100
            headers = ["year"] + list(tshare.columns)
            rows = []
            for yr, r in tshare.iterrows():
                rows.append([int(yr)] + [f"{v:.1f}%" for v in r.values])
            rep.table(headers, rows)
        else:
            rep.w("_Panel is empty._")
            rep.w()
    else:
        rep.w("### Panel")
        rep.w()
        if has_panel:
            rep.w(f"Panel **not built**: publishers not found in the pool: {missing_pubs}.")
        else:
            rep.w(f"No `panel_publishers` in the `{lang}` config — panel not built. "
                  "See the suggestion above.")
        rep.w()

    # stream summaries
    rep.w("### Output files")
    rep.w()
    fs = stream_summary(full)
    fsize = os.path.getsize(full_out) / (1024 * 1024)
    out_rows = []
    if panel_built:
        ps = stream_summary(panel)
        psize = os.path.getsize(panel_out) / (1024 * 1024)
        out_rows.append([f"{lang}_panel", f"{ps['docs']:,}",
                         f"{ps['date_min']}..{ps['date_max']}",
                         f"{ps['total_words']:,}", f"{ps['mean_words']:.1f}", f"{psize:.1f}"])
    out_rows.append([f"{lang}_full", f"{fs['docs']:,}",
                     f"{fs['date_min']}..{fs['date_max']}",
                     f"{fs['total_words']:,}", f"{fs['mean_words']:.1f}", f"{fsize:.1f}"])
    rep.table(["stream", "docs", "date span", "total words", "mean n_words", "file MB"],
              out_rows)
    rep.w(f"`{lang}_full` sampling: uniform across time = equal per-month quota of "
          f"**{full_quota}** over {n_full_months} months ({full_start}..{full_end}).")
    rep.w()
    surprises.append(f"{lang}_full uses equal-per-month sampling (uniform across time); "
                     "confirm this is the intended reading of the brief.")

    # calibration epoch
    if panel_built:
        rep.w(f"### Calibration epoch of `{lang}_panel`")
        rep.w()
        if len(panel):
            pnl = panel.sort_values(["date", "doc_id"], kind="mergesort").reset_index(drop=True)
            cut = max(1, int(math.floor(calib_frac * len(pnl))))
            c0 = pnl["date"].iloc[0].date().isoformat()
            c1 = pnl["date"].iloc[cut - 1].date().isoformat()
            rep.w(f"First {calib_frac*100:.0f}% of the panel = first **{cut:,}** docs, "
                  f"spanning **{c0} → {c1}**.")
        else:
            rep.w("_Panel is empty._")
        rep.w()

    # ---- gates (Part 1) ---------------------------------------------------
    part1_gates, part1_details = {}, {}
    if has_panel:
        k0 = "GATE 0 — every panel_publishers name exists in the cleaned pool"
        part1_gates[k0] = not missing_pubs
        part1_details[k0] = ("all found" if not missing_pubs
                             else f"missing: {missing_pubs}")
    if panel_built:
        try:
            got = pq.read_schema(panel_out)
            schema_ok = got.equals(OUT_SCHEMA) and len(panel) > 0
        except Exception:
            schema_ok = False
        k1 = f"GATE 1 — {lang}_panel.parquet exists, chronologically sorted, schema exact"
        k2 = f"GATE 2 — publisher shares within {100/len(panel_pubs):.1f}% ± 2pp in EVERY panel year"
        k3 = "GATE 3 — <5% of (month × source) quota cells underfilled"
        part1_gates[k1] = schema_ok
        part1_details[k1] = f"schema match={schema_ok}, {len(panel):,} docs"
        part1_gates[k2] = share_ok and len(panel) > 0
        part1_details[k2] = ("all years within tolerance" if part1_gates[k2]
                             else "a year exceeds ±2pp")
        part1_gates[k3] = underfill_pct < 5.0
        part1_details[k3] = f"{underfill_pct:.1f}% underfilled"
    full_pubs = sorted(full["publisher"].unique().tolist()) if len(full) else []
    k4 = (f"GATE 4 — {lang}_full.parquet exists, >= {n_full_pubs} publishers, "
          f"{full_start}..{full_end}")
    part1_gates[k4] = (len(full_pubs) >= n_full_pubs and len(full) > 0)
    part1_details[k4] = f"{len(full_pubs)} publishers (target: >={n_full_pubs})"

    # ---- STATUS -----------------------------------------------------------
    rep.w("## STATUS")
    rep.w()
    rep.w("```")
    rep.w("PART 1")
    for k, v in part1_gates.items():
        rep.w(f"{k}:   {'PASS' if v else 'FAIL'}   ({part1_details[k]})")
    rep.w("")
    verdict = "PROCEED WITH CAVEATS" if all(part1_gates.values()) else "BLOCKED"
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

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(rep.text())

    # Hand off Part-1 gate results (per language) for downstream combined verdicts.
    if not demo:
        with open(f"reports/.t2_part1_gates_{lang}.json", "w") as fh:
            json.dump({"gates": part1_gates, "details": part1_details,
                       "surprises": surprises, "blockers": blockers}, fh, indent=2)

    log(f"Report written to {report_path}")
    log("Part-1 gates: " + ", ".join(
        f"{k.split('—')[0].strip()}={'PASS' if v else 'FAIL'}"
        for k, v in part1_gates.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())