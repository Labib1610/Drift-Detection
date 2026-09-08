#!/usr/bin/env python3
"""
TASK 2 — Part 1. Bangla (Potrika) data preparation.

Builds two chronological streams from one cleaned pool:

  * bn_panel  — primary. Inqilab / Jugantor / Kaler Kontho only, 2016-01..2020-12,
                equal quota per (month x source) cell so the publisher mix is
                constant across the stream and any fertility drift is lexical,
                not compositional.
  * bn_full   — robustness (appendix). All six sources, 2014-06..2020-12, sampled
                uniformly across time (equal per-month quota) up to the cap.

All thresholds come from params.yaml; there are no magic numbers in the code.
The script is deterministic: two runs produce byte-identical parquet.

Usage:
    python src/prepare.py --lang bn --params params.yaml --report reports/T2_report.md
    python src/prepare.py --lang bn --demo        # 5,000-doc subsample, <30s
"""

from __future__ import annotations

import argparse
import glob
import hashlib
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
def write_parquet(df: pd.DataFrame, path: str) -> None:
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
        "lang": pa.array(["bn"] * len(df), pa.string()),
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


def main():
    ap = argparse.ArgumentParser(description="TASK 2 Part 1 — Bangla preparation")
    ap.add_argument("--lang", default="bn")
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--report", default="reports/T2_report.md")
    ap.add_argument("--demo", action="store_true",
                    help="5,000-doc subsample, writes *_demo artifacts, <30s")
    args = ap.parse_args()

    if args.lang != "bn":
        log(f"ERROR: only --lang bn is implemented in T2 (got {args.lang}).")
        return 2

    with open(args.params) as fh:
        P = yaml.safe_load(fh)
    seed = P["seed"]
    np.random.seed(seed)
    cap = P["data"]["max_docs_per_language"]
    min_words = P["data"]["min_doc_words"]
    bn = P["bn"]
    raw_dir = bn["raw_dir"]
    panel_pubs = list(bn["panel_publishers"])
    panel_start, panel_end = bn["panel_start"], bn["panel_end"]
    calib_frac = P["window"]["calibration_fraction"]

    demo = args.demo
    if demo:
        cap = 5000
        panel_out = "data/interim/bn_panel_demo.parquet"
        full_out = "data/interim/bn_full_demo.parquet"
        report_path = "reports/T2_report_demo.md"
    else:
        panel_out = "data/interim/bn_panel.parquet"
        full_out = "data/interim/bn_full.parquet"
        report_path = args.report

    log("Reading raw Potrika (RawDataset only) ...")
    raw, n_files = read_raw(raw_dir, demo)
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

    panel, fill, quota, n_cells, months = build_panel(
        pool, panel_pubs, panel_start, panel_end, cap, seed)
    full, full_quota, n_full_months = build_full(
        pool, "2014-06", "2020-12", cap, seed)

    write_parquet(panel, panel_out)
    write_parquet(full, full_out)
    log(f"Wrote {panel_out} ({len(panel):,} docs) and {full_out} ({len(full):,} docs).")

    # ---- report -----------------------------------------------------------
    rep = Report()
    rep.w("# TASK 2 — Bangla preparation + CC-News probe report")
    rep.w()
    rep.w(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    rep.w(f"- Mode: {'DEMO (5k subsample)' if demo else 'FULL'}")
    rep.w(f"- Params: `{args.params}` · seed {seed}")
    rep.w()
    rep.w("Reproduce with:")
    rep.w("```")
    if demo:
        rep.w("python src/prepare.py --lang bn --demo")
    else:
        rep.w(f"python src/prepare.py --lang bn --params {args.params} --report {report_path}")
    rep.w("```")
    rep.w()

    surprises = []
    blockers = []

    rep.w("## PART 1 — Bangla preparation")
    rep.w()

    # cleaning ledger
    rep.w("### Cleaning ledger")
    rep.w()
    rep.w(f"Raw rows read from `{raw_dir}` ({n_files} files): **{n_raw:,}**")
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
        rep.w("> ⚠️ **More than 5% of cells underfilled** — the 555-doc quota is too "
              "high for the available data; consider lowering it.")
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
    rep.w("### Realised publisher shares per year in `bn_panel`")
    rep.w()
    rep.w("(Confound check — each source should be ~33.3% in every year.)")
    rep.w()
    share_ok = True
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
    rep.w("### Topic composition per year in `bn_panel`")
    rep.w()
    rep.w("(Measured, not corrected — publisher and topic are correlated in "
          "Potrika, so we size any topic drift before interpreting alarms.)")
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

    # stream summaries
    rep.w("### Output files")
    rep.w()
    ps = stream_summary(panel)
    fs = stream_summary(full)
    psize = os.path.getsize(panel_out) / (1024 * 1024)
    fsize = os.path.getsize(full_out) / (1024 * 1024)
    rep.table(
        ["stream", "docs", "date span", "total words", "mean n_words", "file MB"],
        [["bn_panel", f"{ps['docs']:,}", f"{ps['date_min']}..{ps['date_max']}",
          f"{ps['total_words']:,}", f"{ps['mean_words']:.1f}", f"{psize:.1f}"],
         ["bn_full", f"{fs['docs']:,}", f"{fs['date_min']}..{fs['date_max']}",
          f"{fs['total_words']:,}", f"{fs['mean_words']:.1f}", f"{fsize:.1f}"]])
    rep.w(f"`bn_full` sampling: uniform across time = equal per-month quota of "
          f"**{full_quota}** over {n_full_months} months (2014-06..2020-12). "
          "The T2 brief's phrase *'sample proportionally within each month'* "
          "conflicts with *'uniformly across time … do not over-represent "
          "high-volume years'*; the equal-per-month reading is used and flagged "
          "below for your decision.")
    rep.w()
    surprises.append("bn_full uses equal-per-month sampling (uniform across time). "
                     "The brief also says 'proportionally within each month' — "
                     "confirm which you meant; only affects the appendix stream.")

    # calibration epoch
    rep.w("### Calibration epoch of `bn_panel`")
    rep.w()
    if len(panel):
        pnl = panel.sort_values(["date", "doc_id"], kind="mergesort").reset_index(drop=True)
        cut = max(1, int(math.floor(calib_frac * len(pnl))))
        c0 = pnl["date"].iloc[0].date().isoformat()
        c1 = pnl["date"].iloc[cut - 1].date().isoformat()
        rep.w(f"First {calib_frac*100:.0f}% of the panel = first **{cut:,}** docs, "
              f"spanning **{c0} → {c1}**.")
    else:
        c0 = c1 = "—"
        rep.w("_Panel is empty._")
    rep.w()

    # ---- gates (Part 1) ---------------------------------------------------
    schema_ok = True
    try:
        got = pq.read_schema(panel_out)
        schema_ok = got.equals(OUT_SCHEMA) and len(panel) > 0
    except Exception:
        schema_ok = False
    gate1 = schema_ok
    gate2 = share_ok and len(panel) > 0
    gate3 = (underfill_pct < 5.0)
    full_pubs = sorted(full["publisher"].unique().tolist()) if len(full) else []
    gate4 = (len(full_pubs) == 6 and len(full) > 0)

    part1_gates = {
        "GATE 1 — bn_panel.parquet exists, chronologically sorted, schema exact": gate1,
        "GATE 2 — publisher shares within 33.3% ± 2pp in EVERY panel year": gate2,
        "GATE 3 — <5% of (month × source) quota cells underfilled": gate3,
        "GATE 4 — bn_full.parquet exists, 6 publishers, 2014-06..2020-12": gate4,
    }
    part1_details = {
        "GATE 1 — bn_panel.parquet exists, chronologically sorted, schema exact":
            f"schema match={schema_ok}, {len(panel):,} docs",
        "GATE 2 — publisher shares within 33.3% ± 2pp in EVERY panel year":
            "all years within tolerance" if gate2 else "a year exceeds ±2pp",
        "GATE 3 — <5% of (month × source) quota cells underfilled":
            f"{underfill_pct:.1f}% underfilled",
        "GATE 4 — bn_full.parquet exists, 6 publishers, 2014-06..2020-12":
            f"{len(full_pubs)} publishers: {full_pubs}",
    }

    # Placeholder for Part 2 (filled by src/probe_ccnews.py).
    rep.w("## PART 2 — CC-News volume probe")
    rep.w()
    rep.w("_Not yet run. Execute:_")
    rep.w("```")
    rep.w("python src/probe_ccnews.py --langs hi,ar,uk --years 2016-2024 "
          "--max-rows-per-year 200000")
    rep.w("```")
    rep.w("_It will replace this section and finalise the STATUS block below._")
    rep.w()

    # ---- STATUS -----------------------------------------------------------
    rep.w("## STATUS")
    rep.w()
    rep.w("```")
    rep.w("PART 1")
    for k, v in part1_gates.items():
        rep.w(f"{k}:   {'PASS' if v else 'FAIL'}   ({part1_details[k]})")
    rep.w("")
    rep.w("PART 2")
    rep.w("GATE 5 — hi: top-3 domains each >=15K over >=36 clean months:   PENDING (run probe)")
    rep.w("GATE 6 — ar: same condition:                                    PENDING (run probe)")
    rep.w("GATE 7 — uk: same condition:                                    PENDING (run probe)")
    rep.w("")
    all1 = all(part1_gates.values())
    verdict = ("PROCEED WITH CAVEATS" if all1 else "BLOCKED") + " (Part 2 pending)"
    rep.w(f"VERDICT: {verdict}")
    rep.w("Blockers:")
    for b in blockers:
        rep.w(f"  - {b}")
    if not blockers:
        rep.w("  - none (Part 1)")
    rep.w("Surprises worth a human decision:")
    for s in surprises:
        rep.w(f"  - {s}")
    if not surprises:
        rep.w("  - none")
    rep.w("```")

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(rep.text())

    # Hand off Part-1 gate results so probe_ccnews.py can build a combined verdict.
    if not demo:
        with open("reports/.t2_part1_gates.json", "w") as fh:
            json.dump({"gates": part1_gates, "details": part1_details,
                       "surprises": surprises, "blockers": blockers}, fh, indent=2)

    log(f"Report written to {report_path}")
    log("Part-1 gates: " + ", ".join(
        f"{k.split('—')[0].strip()}={'PASS' if v else 'FAIL'}"
        for k, v in part1_gates.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
