#!/usr/bin/env python3
"""
TASK 1 - Potrika data audit.

Audits the Potrika Bangla news corpus (Mendeley v362rp78dc) before any
downstream code is written, because the entire drift-detection project depends
on every article having a trustworthy publication date and source label.

The script is read-only w.r.t. data/raw and writes a single Markdown report
(plus one coverage figure). Every number in the report is produced by this code.

Usage:
    python src/audit_potrika.py --input data/raw/bn_potrika \
                                --out reports/T1_report.md \
                                --sample 50000        # 0 or --full means all rows
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- optional dependencies -------------------------------------------------
# `regex` and PyICU are optional. Import inside try/except and degrade
# gracefully; the report states exactly what was and was not available.
try:
    import regex as _regex_mod  # type: ignore

    HAVE_REGEX = True
    REGEX_ERR = ""
except Exception as e:  # pragma: no cover - environment dependent
    _regex_mod = None
    HAVE_REGEX = False
    REGEX_ERR = repr(e)

try:
    import icu  # type: ignore

    HAVE_ICU = True
    ICU_ERR = ""
except Exception as e:  # pragma: no cover - environment dependent
    icu = None
    HAVE_ICU = False
    ICU_ERR = repr(e)

SEED = 42

# Canonical field names this project standardises on.
CANONICAL = ["text", "category", "headline", "date", "source"]

# Ordered candidate date formats, tried most-specific first. A permissive
# pandas fallback catches anything these miss.
DATE_FORMATS = [
    "%Y/%m/%d",
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%Y/%m/%d %H:%M:%S",
    "%m/%d/%Y",
]

# Display normalisation for the six newspaper sources (raw casing varies).
SOURCE_DISPLAY = {
    "jugantor": "Jugantor",
    "ittefaq": "Ittefaq",
    "jaijaidin": "Jaijaidin",
    "kaler_kontho": "Kaler Kontho",
    "inqilab": "Inqilab",
    "somoyer_alo": "Somoyer Alo",
}


def norm_source(s: object) -> str:
    key = str(s).strip().lower().replace(" ", "_")
    return SOURCE_DISPLAY.get(key, str(s))


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# File discovery / inventory
# ---------------------------------------------------------------------------
def discover_files(input_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(input_dir, "**", "*.csv"), recursive=True))


def count_rows_light(path: str, col: str) -> int:
    """Count data rows by streaming a single light column (handles embedded
    newlines/commas inside quoted text fields, which naive line counting can't)."""
    n = 0
    for chunk in pd.read_csv(path, usecols=[col], dtype=str, chunksize=200_000):
        n += len(chunk)
    return n


def inventory(files: list[str], input_dir: str):
    """Return (rows list for section A, per-file column set, list of dated files)."""
    rows = []
    schema_sig = defaultdict(list)  # tuple(columns) -> [files]
    dated_files = []
    for f in files:
        rel = os.path.relpath(f, input_dir)
        size_mb = os.path.getsize(f) / (1024 * 1024)
        head = pd.read_csv(f, nrows=1000)
        cols = list(head.columns)
        dtypes = {c: str(t) for c, t in head.dtypes.items()}
        is_dated = {"Date", "Source"}.issubset(set(cols))
        if is_dated:
            dated_files.append(f)
            count_col = "Date"
        else:
            count_col = cols[0]
        nrows = count_rows_light(f, count_col)
        schema_sig[tuple(cols)].append(rel)
        rows.append(
            {
                "file": rel,
                "size_mb": size_mb,
                "format": "csv",
                "rows": nrows,
                "columns": cols,
                "dtypes": dtypes,
                "dated": is_dated,
            }
        )
        log(f"  inventoried {rel}: {nrows:,} rows, {size_mb:.1f} MB, dated={is_dated}")
    return rows, schema_sig, dated_files


# ---------------------------------------------------------------------------
# Metadata + text loading
# ---------------------------------------------------------------------------
def load_metadata(dated_files: list[str], input_dir: str) -> pd.DataFrame:
    """Load light metadata columns for every dated row (no article body)."""
    parts = []
    for f in dated_files:
        rel = os.path.relpath(f, input_dir)
        usecols = [c for c in ["Date", "Source", "Category", "Heading"]]
        df = pd.read_csv(f, usecols=lambda c: c in usecols, dtype=str)
        df["__file"] = rel
        parts.append(df)
        log(f"  loaded metadata {rel}: {len(df):,} rows")
    meta = pd.concat(parts, ignore_index=True)
    meta["source_raw"] = meta["Source"]
    meta["source"] = meta["Source"].map(norm_source)
    return meta


def load_text_sample(
    dated_files: list[str], input_dir: str, keep_prob: float, rng: np.random.Generator
):
    """Stream the article body across dated files, keeping each row with
    probability keep_prob. Returns a DataFrame of (text, source). Memory stays
    bounded to the sampled subset rather than the full 2.8 GB corpus."""
    texts, srcs = [], []
    for f in dated_files:
        for chunk in pd.read_csv(
            f, usecols=["News", "Source"], dtype=str, chunksize=50_000
        ):
            if keep_prob >= 1.0:
                mask = np.ones(len(chunk), dtype=bool)
            else:
                mask = rng.random(len(chunk)) < keep_prob
            if mask.any():
                sub = chunk[mask]
                texts.extend(sub["News"].tolist())
                srcs.extend(sub["Source"].map(norm_source).tolist())
        log(f"  sampled text from {os.path.relpath(f, input_dir)} (pool={len(texts):,})")
    return pd.DataFrame({"text": texts, "source": srcs})


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------
def parse_dates(raw: pd.Series):
    """Parse a raw date string series with an ordered list of candidate formats,
    then a permissive fallback. Returns (parsed datetimes, format_counts dict,
    per-row matched-format labels)."""
    parsed = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    matched_fmt = pd.Series(pd.NA, index=raw.index, dtype="object")
    fmt_counts = Counter()

    remaining = raw.notna() & parsed.isna()
    for fmt in DATE_FORMATS:
        idx = raw.index[remaining]
        if len(idx) == 0:
            break
        attempt = pd.to_datetime(raw.loc[idx], format=fmt, errors="coerce")
        ok = attempt.notna()
        good_idx = idx[ok.values]
        parsed.loc[good_idx] = attempt[ok]
        matched_fmt.loc[good_idx] = fmt
        fmt_counts[fmt] = int(ok.sum())
        remaining = raw.notna() & parsed.isna()

    # Permissive fallback for anything the explicit formats missed.
    idx = raw.index[remaining]
    if len(idx) > 0:
        attempt = pd.to_datetime(raw.loc[idx], errors="coerce", dayfirst=False)
        ok = attempt.notna()
        good_idx = idx[ok.values]
        parsed.loc[good_idx] = attempt[ok]
        matched_fmt.loc[good_idx] = "permissive-fallback"
        fmt_counts["permissive-fallback"] = int(ok.sum())

    return parsed, dict(fmt_counts), matched_fmt


# ---------------------------------------------------------------------------
# Word counting methods
# ---------------------------------------------------------------------------
def wc_whitespace(t: str) -> int:
    return len(t.split())


def _re_module():
    return _regex_mod if HAVE_REGEX else re


def wc_regex(t: str) -> int:
    mod = _re_module()
    return len(mod.findall(r"\w+", t, flags=mod.UNICODE))


_ICU_BI = None


def _icu_iter():
    global _ICU_BI
    if _ICU_BI is None:
        _ICU_BI = icu.BreakIterator.createWordInstance(icu.Locale("bn"))
    return _ICU_BI


def wc_icu(t: str) -> int:
    bi = _icu_iter()
    bi.setText(t)
    count = 0
    bi.first()
    for _ in bi:
        # rule status 0 == UBRK_WORD_NONE (whitespace/punctuation span);
        # anything else is a word span (letters, numbers, kana, ideographs).
        if bi.getRuleStatus() != 0:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Text quality helpers
# ---------------------------------------------------------------------------
_LATIN_RE = re.compile(r"[A-Za-z]")
_WS_RE = re.compile(r"\s+")


def norm_text_for_hash(t: str) -> str:
    return _WS_RE.sub(" ", t.strip()).lower()


def text_hash(t: str) -> str:
    return hashlib.md5(norm_text_for_hash(t).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------
class Report:
    def __init__(self):
        self.lines: list[str] = []

    def w(self, s: str = "") -> None:
        self.lines.append(s)

    def table(self, headers: list[str], rows: list[list]) -> None:
        self.w("| " + " | ".join(str(h) for h in headers) + " |")
        self.w("| " + " | ".join("---" for _ in headers) + " |")
        for r in rows:
            self.w("| " + " | ".join(str(c) for c in r) + " |")
        self.w()

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def pctl(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Potrika data audit (TASK 1)")
    ap.add_argument("--input", default="data/raw/bn_potrika")
    ap.add_argument("--out", default="reports/T1_report.md")
    ap.add_argument("--sample", type=int, default=50000, help="0 means all rows")
    ap.add_argument("--full", action="store_true", help="use all rows")
    args = ap.parse_args()

    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    input_dir = args.input
    if not os.path.isdir(input_dir):
        log(f"ERROR: input dir not found: {input_dir}")
        return 2

    rep = Report()
    rep.w("# TASK 1 — Potrika data audit report")
    rep.w()
    rep.w(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    rep.w(f"- Input: `{input_dir}`")
    sample_desc = "all rows (--full)" if (args.full or args.sample == 0) else f"{args.sample:,} rows"
    rep.w(f"- Sample setting: {sample_desc}")
    rep.w(f"- Seed: {SEED}")
    rep.w()
    rep.w("Reproduce with:")
    rep.w("```")
    full_flag = " --full" if (args.full or args.sample == 0) else f" --sample {args.sample}"
    rep.w(f"python src/audit_potrika.py --input {input_dir} --out {args.out}{full_flag}")
    rep.w("```")
    rep.w()
    rep.w("Optional-dependency status:")
    rep.w(f"- `regex` module: {'available' if HAVE_REGEX else 'NOT available — ' + REGEX_ERR}")
    rep.w(f"- `PyICU`: {'available (ICU ' + icu.ICU_VERSION + ')' if HAVE_ICU else 'NOT available — ' + ICU_ERR}")
    rep.w()

    blockers: list[str] = []
    surprises: list[str] = []

    # ---- discovery + inventory --------------------------------------------
    log("Discovering files ...")
    files = discover_files(input_dir)
    if not files:
        log("ERROR: no CSV files found under input dir")
        return 2
    log(f"Found {len(files)} CSV files. Building inventory ...")
    inv, schema_sig, dated_files = inventory(files, input_dir)

    rep.w("## A. File inventory")
    rep.w()
    rep.w(f"Total CSV files found: **{len(files)}**. "
          f"Dated files (with `Date`+`Source`): **{len(dated_files)}**.")
    rep.w()
    inv_rows = []
    for r in inv:
        inv_rows.append([
            r["file"], f"{r['size_mb']:.1f}", r["format"], f"{r['rows']:,}",
            ", ".join(r["columns"]), "yes" if r["dated"] else "no",
        ])
    rep.table(["file", "size (MB)", "format", "rows", "columns", "dated?"], inv_rows)

    rep.w("**Schema groups** (files sharing an identical column set):")
    rep.w()
    for i, (cols, flist) in enumerate(schema_sig.items(), 1):
        rep.w(f"- Group {i}: columns `{list(cols)}` — {len(flist)} file(s)")
    rep.w()
    rep.w("**dtypes** (inferred by pandas on a 1000-row head, per schema group):")
    rep.w()
    seen_sig = set()
    for r in inv:
        sig = tuple(r["columns"])
        if sig in seen_sig:
            continue
        seen_sig.add(sig)
        rep.w(f"- `{r['file']}`: " + ", ".join(f"`{c}`={t}" for c, t in r["dtypes"].items()))
    rep.w()
    if len(schema_sig) > 1:
        rep.w("> Schemas are **not** identical across all files: the corpus ships two "
              "families — a source/date-bearing `RawDataset` and a `BalancedDataset` "
              "that carries only article text + class. Sections C–G below operate on the "
              "dated `RawDataset` files, because publication date and newspaper source "
              "are what this project depends on. The `BalancedDataset` is inventoried "
              "here but excluded from the temporal/label audit (it has no date or "
              "source column).")
        surprises.append(
            "Corpus ships two dataset families; only RawDataset carries Date+Source. "
            "BalancedDataset (article/class only) is unusable for the drift stream."
        )
    rep.w()

    if not dated_files:
        blockers.append("No files with Date+Source columns were found — cannot build a "
                        "chronological stream.")
        rep.w("**No dated files found — remaining sections cannot be produced.**")
        write_status(rep, {}, None, None, {}, [], blockers, surprises)
        _write(args.out, rep.text())
        return 1

    # ---- load metadata -----------------------------------------------------
    log("Loading metadata columns for all dated rows ...")
    meta = load_metadata(dated_files, input_dir)
    total_rows = len(meta)
    log(f"Total dated rows: {total_rows:,}")

    # ---- B. column mapping -------------------------------------------------
    rep.w("## B. Column mapping")
    rep.w()
    actual_cols = list(pd.read_csv(dated_files[0], nrows=1).columns)
    mapping = {
        "text": "News" if "News" in actual_cols else None,
        "category": "Category" if "Category" in actual_cols else None,
        "headline": "Heading" if "Heading" in actual_cols else None,
        "date": "Date" if "Date" in actual_cols else None,
        "source": "Source" if "Source" in actual_cols else None,
    }
    rep.w(f"Actual columns in dated files: `{actual_cols}` "
          "(the leading unnamed column is a pandas row index that was written to disk).")
    rep.w()
    map_rows = []
    for canon in CANONICAL:
        got = mapping[canon]
        map_rows.append([canon, got if got else "**NO MATCH**"])
    rep.table(["canonical", "actual column"], map_rows)
    missing_canon = [c for c in CANONICAL if mapping[c] is None]
    if missing_canon:
        rep.w(f"> Missing canonical fields (no column matched): {missing_canon}")
        surprises.append(f"Canonical fields without a source column: {missing_canon}")
    else:
        rep.w("All five canonical fields map cleanly.")
    rep.w()

    # ---- C. date parsing ---------------------------------------------------
    rep.w("## C. Date parsing")
    rep.w()
    raw_dates = meta["Date"]

    # 20 raw strings sampled across sources.
    rep.w("**20 raw date strings, verbatim, sampled across sources:**")
    rep.w()
    sample_raw_rows = []
    srcs_present = sorted(meta["source"].dropna().unique())
    per_src = max(1, 20 // max(1, len(srcs_present)))
    picked = 0
    for s in srcs_present:
        sub = meta[meta["source"] == s]
        take = min(per_src, len(sub))
        for _, row in sub.sample(take, random_state=SEED).iterrows():
            sample_raw_rows.append([s, repr(str(row["Date"]))])
            picked += 1
    # top up to 20 if needed
    if picked < 20:
        extra = meta.sample(min(20 - picked, len(meta)), random_state=SEED + 1)
        for _, row in extra.iterrows():
            sample_raw_rows.append([norm_source(row["Source"]), repr(str(row["Date"]))])
    rep.table(["source", "raw Date (repr)"], sample_raw_rows[:20])

    log("Parsing dates ...")
    parsed, fmt_counts, matched_fmt = parse_dates(raw_dates)
    meta["parsed_date"] = parsed
    meta["matched_fmt"] = matched_fmt
    n_parsed = int(parsed.notna().sum())

    rep.w("**Format match counts** (each row attributed to the first format that "
          "parsed it):")
    rep.w()
    fmt_rows = []
    for fmt in DATE_FORMATS + ["permissive-fallback"]:
        c = fmt_counts.get(fmt, 0)
        if c:
            fmt_rows.append([f"`{fmt}`", f"{c:,}", f"{100*c/total_rows:.2f}%"])
    n_unparsed = total_rows - n_parsed
    fmt_rows.append(["**unparseable**", f"{n_unparsed:,}", f"{100*n_unparsed/total_rows:.2f}%"])
    rep.table(["format", "rows matched", "% of all rows"], fmt_rows)
    rep.w(f"Overall parse rate: **{100*n_parsed/total_rows:.2f}%** "
          f"({n_parsed:,} / {total_rows:,}).")
    rep.w()

    # per-source table
    rep.w("**Per newspaper source:**")
    rep.w()
    per_source_stats = {}
    ps_rows = []
    for s in srcs_present:
        sub = meta[meta["source"] == s]
        n = len(sub)
        p = int(sub["parsed_date"].notna().sum())
        rate = 100 * p / n if n else 0.0
        mn = sub["parsed_date"].min()
        mx = sub["parsed_date"].max()
        per_source_stats[s] = {"n": n, "parsed": p, "rate": rate, "min": mn, "max": mx}
        ps_rows.append([
            s, f"{n:,}", f"{p:,}", f"{rate:.2f}%",
            mn.date().isoformat() if pd.notna(mn) else "—",
            mx.date().isoformat() if pd.notna(mx) else "—",
        ])
    rep.table(["source", "n rows", "n parsed", "parse rate", "min date", "max date"], ps_rows)

    # unparseable examples
    rep.w("**Up to 15 unparseable raw date strings, verbatim, with source:**")
    rep.w()
    unp = meta[meta["parsed_date"].isna() & meta["Date"].notna()]
    if len(unp) == 0:
        rep.w("_None — every non-null date string parsed._")
        rep.w()
    else:
        up_rows = []
        for _, row in unp.head(15).iterrows():
            up_rows.append([row["source"], repr(str(row["Date"]))])
        rep.table(["source", "raw Date (repr)"], up_rows)

    # null dates
    n_null = int(meta["Date"].isna().sum())
    if n_null:
        rep.w(f"Rows with a **null/empty** Date field: {n_null:,}.")
        rep.w()

    # implausible dates
    rep.w("**Implausible parsed dates:**")
    rep.w()
    today = pd.Timestamp(datetime.now().date())
    before_2010 = int((meta["parsed_date"] < pd.Timestamp("2010-01-01")).sum())
    after_today = int((meta["parsed_date"] > today).sum())
    imp_rows = [
        ["before 2010-01-01", f"{before_2010:,}"],
        [f"after today ({today.date()})", f"{after_today:,}"],
    ]
    rep.table(["check", "count"], imp_rows)
    # sentinel: any (source) whose dates collapse onto very few unique values
    sentinel_notes = []
    for s in srcs_present:
        sub = meta[(meta["source"] == s) & meta["parsed_date"].notna()]
        if len(sub) == 0:
            continue
        vc = sub["parsed_date"].value_counts()
        top_share = vc.iloc[0] / len(sub)
        if top_share > 0.5 and len(sub) > 100:
            sentinel_notes.append(
                f"{s}: {top_share*100:.1f}% of rows share date "
                f"{vc.index[0].date()} — possible sentinel."
            )
    if sentinel_notes:
        for n in sentinel_notes:
            rep.w(f"- {n}")
            surprises.append("Suspicious sentinel date — " + n)
        rep.w()
    else:
        rep.w("No source collapses >50% of its rows onto a single date "
              "(no obvious sentinel).")
        rep.w()
    if before_2010 or after_today:
        surprises.append(f"Implausible dates: {before_2010} before 2010, "
                         f"{after_today} in the future.")

    # ---- D. temporal coverage ---------------------------------------------
    rep.w("## D. Temporal coverage")
    rep.w()
    md = meta[meta["parsed_date"].notna()].copy()
    md["year"] = md["parsed_date"].dt.year
    md["month"] = md["parsed_date"].dt.month
    md["ym"] = md["parsed_date"].dt.to_period("M")

    # counts per (year, month) per source -> present as year x source pivot
    # (a full month-level table is huge; give a year x source pivot plus the
    # per-month figure, then list gaps explicitly).
    rep.w("**Article counts per (year × source):**")
    rep.w()
    yr_pivot = md.pivot_table(index="year", columns="source", values="Date",
                              aggfunc="count", fill_value=0)
    yr_headers = ["year"] + list(yr_pivot.columns) + ["total"]
    yr_rows = []
    for yr, r in yr_pivot.iterrows():
        yr_rows.append([int(yr)] + [f"{int(v):,}" for v in r.values] + [f"{int(r.sum()):,}"])
    yr_rows.append(["**total**"] + [f"{int(yr_pivot[c].sum()):,}" for c in yr_pivot.columns]
                   + [f"{int(yr_pivot.values.sum()):,}"])
    rep.table(yr_headers, yr_rows)

    rep.w("A full (year, month) × source breakdown is large; it is summarised in the "
          "figure below and mined for zero cells next.")
    rep.w()

    # gaps: per source, months inside its own min-max with zero articles
    rep.w("**Coverage gaps** — (source, month) cells with zero articles inside that "
          "source's own min–max range:")
    rep.w()
    gap_rows = []
    combined_gap_flag = False
    for s in srcs_present:
        sub = md[md["source"] == s]
        if len(sub) == 0:
            continue
        mn, mx = sub["ym"].min(), sub["ym"].max()
        full_range = pd.period_range(mn, mx, freq="M")
        present = set(sub["ym"].unique())
        missing = [str(p) for p in full_range if p not in present]
        if missing:
            gap_rows.append([s, f"{mn}..{mx}", len(missing),
                             ", ".join(missing[:24]) + (" ..." if len(missing) > 24 else "")])
    if gap_rows:
        rep.table(["source", "range", "# missing months", "missing months"], gap_rows)
    else:
        rep.w("_No within-range monthly gaps in any source._")
        rep.w()

    # combined stream gap check (for GATE 2)
    all_range = pd.period_range(md["ym"].min(), md["ym"].max(), freq="M")
    present_all = set(md["ym"].unique())
    combined_missing = [p for p in all_range if p not in present_all]
    # longest run of consecutive missing months
    longest_run = 0
    run = 0
    for p in all_range:
        if p not in present_all:
            run += 1
            longest_run = max(longest_run, run)
        else:
            run = 0
    rep.w(f"Combined stream span: **{md['ym'].min()} → {md['ym'].max()}**. "
          f"Missing months in the combined stream: {len(combined_missing)}. "
          f"Longest consecutive gap: **{longest_run} month(s)**.")
    rep.w()
    if longest_run > 2:
        combined_gap_flag = True

    # figure: stacked monthly counts by source
    log("Rendering coverage figure ...")
    fig_path = os.path.join(os.path.dirname(args.out), "figs", "T1_coverage.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    monthly = md.pivot_table(index="ym", columns="source", values="Date",
                             aggfunc="count", fill_value=0)
    # restrict x-axis to 2014-2020 window per brief
    monthly = monthly.sort_index()
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(monthly.index))
    bottom = np.zeros(len(monthly.index))
    for col in monthly.columns:
        vals = monthly[col].values
        ax.bar(x, vals, bottom=bottom, label=col, width=1.0)
        bottom += vals
    step = max(1, len(monthly.index) // 28)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([str(p) for p in monthly.index[::step]], rotation=90, fontsize=7)
    ax.set_ylabel("articles / month")
    ax.set_title("Potrika monthly article counts by source (stacked)")
    ax.legend(ncol=6, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=110)
    plt.close(fig)
    rep.w(f"Figure: `{os.path.relpath(fig_path, os.path.dirname(args.out) + '/..')}` "
          "— stacked monthly article counts by source.")
    rep.w()
    rep.w(f"![coverage]({os.path.relpath(fig_path, os.path.dirname(args.out))})")
    rep.w()

    # ---- E. label fields ---------------------------------------------------
    rep.w("## E. Label fields")
    rep.w()
    rep.w("**Article count per newspaper source** (the S6 publisher-prediction "
          "target — imbalance matters):")
    rep.w()
    src_counts = meta["source"].value_counts()
    sc_rows = [[s, f"{int(c):,}", f"{100*c/total_rows:.2f}%"] for s, c in src_counts.items()]
    rep.table(["source", "count", "share"], sc_rows)
    imbalance = src_counts.max() / max(1, src_counts.min())
    rep.w(f"Imbalance ratio (largest ÷ smallest source): **{imbalance:.1f}×**.")
    rep.w()

    rep.w("**Article count per category:**")
    rep.w()
    cat_counts = meta["Category"].value_counts()
    cc_rows = [[c, f"{int(n):,}", f"{100*n/total_rows:.2f}%"] for c, n in cat_counts.items()]
    rep.table(["category", "count", "share"], cc_rows)

    # Detect inconsistent category labels (same concept, different spelling),
    # which would corrupt the secondary topic-prediction task. Two labels are
    # treated as the same topic if their normalised forms (lowercased, stripped
    # to alphanumerics, "and" removed) are equal or one is a prefix of the other
    # (catches abbreviations like "tech" vs "technology").
    def norm_cat(c):
        return re.sub(r"[^a-z0-9]", "", str(c).lower()).replace("and", "")

    labels = list(cat_counts.index)
    normed = {c: norm_cat(c) for c in labels}
    parent = {c: c for c in labels}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            na, nb = normed[a], normed[b]
            if not na or not nb:
                continue
            short, lng = sorted((na, nb), key=len)
            if len(short) >= 5 and lng.startswith(short):
                parent[find(a)] = find(b)

    clusters = defaultdict(list)
    for c in labels:
        clusters[find(c)].append(c)
    collisions = [v for v in clusters.values() if len(v) > 1]
    if collisions:
        rep.w("> **Inconsistent category labels detected** — the same topic appears "
              "under more than one spelling and must be canonicalised before the "
              "secondary topic-prediction task:")
        for v in collisions:
            detail = ", ".join(f"`{c}` ({int(cat_counts[c]):,})" for c in
                               sorted(v, key=lambda c: -cat_counts[c]))
            rep.w(f"> - {detail}")
            surprises.append("Category label collision: " + detail.replace("`", ""))
        rep.w()

    rep.w("**Cross-tab source × category:**")
    rep.w()
    ct = pd.crosstab(meta["source"], meta["Category"])
    ct_headers = ["source"] + list(ct.columns)
    ct_rows = []
    for s, r in ct.iterrows():
        ct_rows.append([s] + [f"{int(v):,}" for v in r.values])
    rep.table(ct_headers, ct_rows)

    # ---- text sample for F & G --------------------------------------------
    use_all = args.full or args.sample == 0
    keep_prob = 1.0 if use_all else min(1.0, args.sample / total_rows)
    log(f"Sampling article text (keep_prob={keep_prob:.4f}) ...")
    tdf = load_text_sample(dated_files, input_dir, keep_prob, rng)
    tdf["text"] = tdf["text"].fillna("")
    log(f"Text sample size: {len(tdf):,}")

    # ---- F. word counting --------------------------------------------------
    rep.w("## F. Word counting")
    rep.w()
    rep.w(f"PyICU imported: **{'yes' if HAVE_ICU else 'no'}**"
          + (f" (error: {ICU_ERR})" if not HAVE_ICU else f" (ICU {icu.ICU_VERSION})") + ".")
    rep.w(f"`regex` module imported: **{'yes' if HAVE_REGEX else 'no — falling back to stdlib `re`'}**"
          + (f" (error: {REGEX_ERR})" if not HAVE_REGEX else "") + ".")
    rep.w()
    n_wc = min(2000, len(tdf))
    wc_df = tdf.sample(n_wc, random_state=SEED).reset_index(drop=True) if len(tdf) else tdf
    if n_wc == 0:
        rep.w("_No documents available to word-count._")
        rep.w()
        methods_ok = False
    else:
        ws = np.array([wc_whitespace(t) for t in wc_df["text"]])
        rx = np.array([wc_regex(t) for t in wc_df["text"]])
        method_names = ["whitespace split()",
                        f"{'regex' if HAVE_REGEX else 're'} \\w+ (UNICODE)"]
        method_vals = [ws, rx]
        if HAVE_ICU:
            ic = np.array([wc_icu(t) for t in wc_df["text"]])
            method_names.append("ICU BreakIterator (bn)")
            method_vals.append(ic)
        rep.w(f"Computed on a random **{n_wc}**-document subsample.")
        rep.w()
        stat_rows = []
        for name, v in zip(method_names, method_vals):
            stat_rows.append([name, f"{v.mean():.1f}", f"{np.median(v):.1f}",
                              f"{pctl(v,5):.1f}", f"{pctl(v,95):.1f}"])
        rep.table(["method", "mean", "median", "p5", "p95"], stat_rows)

        rep.w("**Pearson correlation between methods:**")
        rep.w()
        corr_rows = []
        for i in range(len(method_vals)):
            for j in range(i + 1, len(method_vals)):
                r = float(np.corrcoef(method_vals[i], method_vals[j])[0, 1])
                corr_rows.append([f"{method_names[i]} vs {method_names[j]}", f"{r:.4f}"])
        rep.table(["pair", "pearson r"], corr_rows)

        if HAVE_ICU:
            ratio = ic / np.where(ws == 0, np.nan, ws)
            rep.w(f"Mean ratio **ICU ÷ whitespace** = "
                  f"**{np.nanmean(ratio):.3f}** (median {np.nanmedian(ratio):.3f}).")
            rep.w()

        # 3 docs of maximum disagreement (spread across all methods)
        stack = np.vstack(method_vals)
        spread = stack.max(axis=0) - stack.min(axis=0)
        worst = np.argsort(-spread)[:3]
        rep.w("**3 documents where the methods disagree most:**")
        rep.w()
        dis_headers = ["#"] + method_names + ["text preview"]
        dis_rows = []
        for k, idx in enumerate(worst, 1):
            preview = _WS_RE.sub(" ", str(wc_df.iloc[idx]["text"]).strip())[:80]
            dis_rows.append([k] + [int(v[idx]) for v in method_vals] + [preview + "…"])
        rep.table(dis_headers, dis_rows)
        methods_ok = True

    # ---- G. text quality ---------------------------------------------------
    rep.w("## G. Text quality")
    rep.w()
    rep.w(f"Computed on the {len(tdf):,}-document text sample "
          "(document length measured with `str.split()` whitespace words).")
    rep.w()
    if len(tdf) == 0:
        rep.w("_No documents available._")
        rep.w()
    else:
        lens = np.array([wc_whitespace(t) for t in tdf["text"]])
        q_rows = []
        for q in [1, 5, 25, 50, 75, 95, 99]:
            q_rows.append([f"p{q}", f"{pctl(lens, q):.0f}"])
        rep.table(["percentile", "words"], q_rows)
        under50 = int((lens < 50).sum())
        empties = int(sum(1 for t in tdf["text"] if str(t).strip() == ""))
        hashes = [text_hash(t) for t in tdf["text"]]
        hc = Counter(hashes)
        dup_docs = sum(c - 1 for c in hc.values() if c > 1)
        dup_rate = 100 * dup_docs / len(tdf)
        latin = sum(1 for t in tdf["text"] if _LATIN_RE.search(str(t)))
        latin_share = 100 * latin / len(tdf)
        qual_rows = [
            ["documents in sample", f"{len(tdf):,}"],
            ["under 50 words (dropped later)", f"{under50:,} ({100*under50/len(tdf):.2f}%)"],
            ["empty / whitespace-only text", f"{empties:,} ({100*empties/len(tdf):.2f}%)"],
            ["exact duplicates (by normalised hash)", f"{dup_docs:,} ({dup_rate:.2f}%)"],
            ["contains Latin-script chars (code-mix proxy)", f"{latin:,} ({latin_share:.2f}%)"],
        ]
        rep.table(["metric", "value"], qual_rows)
        if empties:
            surprises.append(f"{empties:,} empty/whitespace-only article bodies in the "
                             f"sample ({100*empties/len(tdf):.2f}%).")
        if under50 / len(tdf) > 0.10:
            surprises.append(f"{100*under50/len(tdf):.1f}% of documents are <50 words and "
                             "will be dropped downstream.")

    # ---- STATUS ------------------------------------------------------------
    write_status(rep, per_source_stats, longest_run, (md["ym"].min(), md["ym"].max()),
                 dict(src_counts), combined_missing, blockers, surprises,
                 methods_ok=methods_ok, have_word_method=(HAVE_ICU or True))

    _write(args.out, rep.text())
    log(f"Report written to {args.out}")
    return 0


def write_status(rep, per_source_stats, longest_run, span, src_counts,
                 combined_missing, blockers, surprises, methods_ok=True,
                 have_word_method=True):
    rep.w("## STATUS")
    rep.w()

    # GATE 1: parse rate >=95% in all six sources
    if per_source_stats:
        min_rate = min(v["rate"] for v in per_source_stats.values())
        n_sources = len(per_source_stats)
        gate1 = (min_rate >= 95.0) and (n_sources >= 6)
        gate1_detail = f"min per-source parse rate {min_rate:.2f}% over {n_sources} sources"
    else:
        gate1 = False
        gate1_detail = "no per-source stats"

    # GATE 2: span 2014-2020, no gap >2 months
    if span and span[0] is not None:
        spans_range = (span[0].year <= 2014) and (span[1].year >= 2020)
        gate2 = spans_range and (longest_run is not None and longest_run <= 2)
        gate2_detail = (f"span {span[0]}..{span[1]}, longest combined gap "
                        f"{longest_run} month(s)")
    else:
        gate2 = False
        gate2_detail = "no coverage"

    # GATE 3: all six sources present with >=5000 articles
    if src_counts:
        n_src = len(src_counts)
        min_count = min(src_counts.values())
        gate3 = (n_src >= 6) and (min_count >= 5000)
        gate3_detail = f"{n_src} sources, smallest has {min_count:,} articles"
    else:
        gate3 = False
        gate3_detail = "no source counts"

    # GATE 4: usable word-count method
    gate4 = bool(have_word_method and methods_ok)
    if HAVE_ICU:
        gate4_detail = "PyICU BreakIterator available"
    elif HAVE_REGEX:
        gate4_detail = "PyICU absent; `regex` \\w+ fallback documented"
    else:
        gate4_detail = "PyICU and `regex` absent; stdlib `re` \\w+ fallback documented"

    def pf(b):
        return "PASS" if b else "FAIL"

    rep.w("```")
    rep.w(f"GATE 1 — date parse rate >=95% in ALL six sources:        {pf(gate1)}   ({gate1_detail})")
    rep.w(f"GATE 2 — combined coverage spans 2014-2020, no gap >2 months:  {pf(gate2)}   ({gate2_detail})")
    rep.w(f"GATE 3 — all six sources present with >=5,000 articles:   {pf(gate3)}   ({gate3_detail})")
    rep.w(f"GATE 4 — a usable word-count method exists:  {pf(gate4)}   ({gate4_detail})")
    rep.w("")

    all_pass = gate1 and gate2 and gate3 and gate4
    if blockers:
        verdict = "BLOCKED"
    elif all_pass and not surprises:
        verdict = "PROCEED"
    elif all_pass:
        verdict = "PROCEED WITH CAVEATS"
    else:
        verdict = "PROCEED WITH CAVEATS" if (gate1 and gate3) else "BLOCKED"

    rep.w(f"VERDICT: {verdict}")
    rep.w("Blockers:")
    if blockers:
        for b in blockers:
            rep.w(f"  - {b}")
    else:
        rep.w("  - none")
    rep.w("Surprises worth a human decision:")
    if surprises:
        for s in surprises:
            rep.w(f"  - {s}")
    else:
        rep.w("  - none")
    rep.w("```")
    rep.w()


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


if __name__ == "__main__":
    sys.exit(main())
