#!/usr/bin/env python3
"""
TASK 3b — CC-News re-probe with the correct schema (network, read-only).

The first probe looked for `date` and `domain`; the real `stanford-oval/ccnews`
schema uses `published_date` and `publisher` (and `plain_text`, not `text`). This
version reads field names from params.yaml (`ccnews:` block), filters on
`language_score >= 0.90`, and reports a full per-language acceptance funnel.

Writes NO corpus data — counts only. Streaming, so nothing is downloaded to disk.
Resumable from reports/ccnews_probe.json (delete it before a fresh run).

Usage:
    python src/probe_ccnews.py --langs hi,ar,uk --years 2016-2024 \
        --target-valid 150000 --max-seconds 2400
    python src/probe_ccnews.py --demo        # tiny bounded smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

try:
    import icu
    _ICU_BI = icu.BreakIterator.createWordInstance(icu.Locale.getRoot())
    HAVE_ICU = True
except Exception as e:  # pragma: no cover
    _ICU_BI = None
    HAVE_ICU = False
    _ICU_ERR = repr(e)

JSON_PATH = "reports/ccnews_probe.json"
REPORT_PATH = "reports/T3b_report.md"

MIN_DOC_WORDS = 50
MIN_DOMAIN_ARTICLES = 15_000
MIN_CLEAN_MONTHS = 36
MAX_MONTH_GAP = 2
SAVE_EVERY_SEC = 60

_ALT_DATE_FORMATS = ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                     "%Y/%m/%d", "%d/%m/%Y"]


def log(msg):
    print(msg, flush=True)


def icu_wordcount(t):
    if not HAVE_ICU:
        return len(str(t).split())
    _ICU_BI.setText(str(t))
    n = 0
    _ICU_BI.first()
    for _ in _ICU_BI:
        if _ICU_BI.getRuleStatus() != 0:
            n += 1
    return n


def parse_years(spec):
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def parse_date(s, primary_fmt):
    s = str(s)
    for fmt in [primary_fmt] + [f for f in _ALT_DATE_FORMATS if f != primary_fmt]:
        try:
            return datetime.strptime(s[:len(fmt) + 4] if "%H" in fmt else s[:10], fmt), fmt
        except Exception:
            continue
    return None, None


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------
def empty_state():
    return {
        "seen": 0, "lang_match": 0, "score_ok": 0, "date_ok": 0,
        "pub_ok": 0, "valid": 0, "score_dropped": 0,
        "date_formats": {}, "score_hist": {},
        "crawl_pub": {},              # crawl_year -> {pub_year: n}
        "publisher_counts": {},
        "pub_month": {},              # publisher -> {YYYY-MM: n}
        "cat_nonempty": 0, "tag_nonempty": 0,
        "category_values": {},
        "word_counts": [],            # ICU word counts of valid docs
        "years_done": [],
        "target_reached": False,
    }


def longest_clean_span(month_counts):
    if not month_counts:
        return 0, 999
    months = sorted(pd.Period(m, freq="M") for m in month_counts)
    full = pd.period_range(months[0], months[-1], freq="M")
    present = set(months)
    best = run = gap = maxgap = 0
    for p in full:
        if p in present:
            run += 1; gap = 0
        else:
            gap += 1; maxgap = max(maxgap, gap)
            if gap > MAX_MONTH_GAP:
                best = max(best, run); run = 0
    return max(best, run), maxgap


def evaluate_language(st):
    top = sorted(st["publisher_counts"].items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    diags, all_ok, doms, dmin, dmax = [], len(top) == 3, [], None, None
    for dom, cnt in top:
        mc = st["pub_month"].get(dom, {})
        clean, maxgap = longest_clean_span(mc)
        ok = cnt >= MIN_DOMAIN_ARTICLES and clean >= MIN_CLEAN_MONTHS
        all_ok = all_ok and ok
        diags.append({"publisher": dom, "articles": cnt, "clean_months": clean,
                      "max_gap": maxgap, "ok": ok})
        if mc:
            ms = sorted(mc)
            dmin = ms[0] if dmin is None else min(dmin, ms[0])
            dmax = ms[-1] if dmax is None else max(dmax, ms[-1])
        doms.append(dom)
    panel = {"publishers": doms, "range": f"{dmin}..{dmax}"} if all_ok else None
    return all_ok, panel, diags


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------
def run_probe(cfg, langs, years, target_valid, max_seconds, states, t0,
              per_year_rows):
    from datasets import load_dataset

    fd, fp = cfg["field_date"], cfg["field_publisher"]
    ft, fl = cfg["field_text"], cfg["field_language"]
    fs = cfg["field_language_score"]
    fcat, ftag = cfg["field_categories"], cfg["field_tags"]
    min_score = float(cfg["min_language_score"])
    primary_fmt = cfg["date_format"]
    langset = set(langs)
    last_save = time.time()

    def all_done():
        return all(states[l]["target_reached"] for l in langs)

    for year in years:
        if all_done():
            break
        remaining = [l for l in langs if year not in states[l]["years_done"]]
        if not remaining:
            continue
        log(f"  streaming {cfg['repo']} name={year} ...")
        ds = load_dataset(cfg["repo"], name=str(year), split="train", streaming=True)
        scanned_this_year = 0
        for row in ds:
            scanned_this_year += 1
            # Per-year row cap ensures every crawl config 2016-2024 is sampled, so
            # temporal (36-month) coverage can be assessed within the time budget,
            # instead of one abundant year monopolising the scan.
            if scanned_this_year > per_year_rows:
                break
            if time.time() - t0 > max_seconds:
                log("  wall-clock cap reached; stopping.")
                for l in remaining:
                    states[l]["years_done"].append(year)
                _save(states, cfg, langs, years, t0)
                return states, "cap"
            lang = row.get(fl)
            if lang not in langset:
                # still count a scanned row toward throughput for matched langs only
                continue
            st = states[lang]
            st["seen"] += 1
            st["lang_match"] += 1
            if st["target_reached"]:
                continue
            # score
            try:
                score = float(row.get(fs))
            except (TypeError, ValueError):
                score = float("nan")
            bucket = f"{np.floor(score * 50) / 50:.2f}" if np.isfinite(score) else "nan"
            st["score_hist"][bucket] = st["score_hist"].get(bucket, 0) + 1
            if not (np.isfinite(score) and score >= min_score):
                st["score_dropped"] += 1
                continue
            st["score_ok"] += 1
            # date
            dt, fmt = parse_date(row.get(fd), primary_fmt)
            if dt is None:
                continue
            st["date_ok"] += 1
            st["date_formats"][fmt] = st["date_formats"].get(fmt, 0) + 1
            # publisher
            pub = row.get(fp)
            if not pub or not str(pub).strip():
                continue
            pub = str(pub).strip().lower()
            st["pub_ok"] += 1
            # words
            nw = icu_wordcount(row.get(ft) or "")
            if nw < MIN_DOC_WORDS:
                continue
            # fully valid
            st["valid"] += 1
            st["word_counts"].append(int(nw))
            st["publisher_counts"][pub] = st["publisher_counts"].get(pub, 0) + 1
            ym = dt.strftime("%Y-%m")
            st["pub_month"].setdefault(pub, {})
            st["pub_month"][pub][ym] = st["pub_month"][pub].get(ym, 0) + 1
            cy, py = str(year), str(dt.year)
            st["crawl_pub"].setdefault(cy, {})
            st["crawl_pub"][cy][py] = st["crawl_pub"][cy].get(py, 0) + 1
            cat = row.get(fcat)
            if cat:
                st["cat_nonempty"] += 1
                for c in (cat if isinstance(cat, list) else [cat]):
                    c = str(c).strip()
                    if c:
                        st["category_values"][c] = st["category_values"].get(c, 0) + 1
            tag = row.get(ftag)
            if tag:
                st["tag_nonempty"] += 1
            if st["valid"] >= target_valid:
                st["target_reached"] = True
                log(f"    {lang}: reached target {target_valid:,} valid rows")
            if time.time() - last_save > SAVE_EVERY_SEC:
                _save(states, cfg, langs, years, t0)
                last_save = time.time()
        for l in remaining:
            if year not in states[l]["years_done"]:
                states[l]["years_done"].append(year)
        _save(states, cfg, langs, years, t0)
    return states, ("done" if all_done() else "exhausted")


def _save(states, cfg, langs, years, t0):
    os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)
    payload = {"generated": datetime.now().isoformat(timespec="seconds"),
               "elapsed_sec": time.time() - t0, "langs": langs, "years": years,
               "cfg": cfg, "states": states}
    tmp = JSON_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh)
    os.replace(tmp, JSON_PATH)


def _load_resume(langs):
    states = {l: empty_state() for l in langs}
    prev_elapsed = 0.0
    if os.path.exists(JSON_PATH):
        try:
            prev = json.load(open(JSON_PATH))
            for l in langs:
                if l in prev.get("states", {}):
                    states[l] = prev["states"][l]
            prev_elapsed = prev.get("elapsed_sec", 0.0)
            log(f"  resuming from {JSON_PATH} (prev elapsed {prev_elapsed:.0f}s)")
        except Exception as e:
            log(f"  could not resume ({e!r}); fresh start")
    return states, prev_elapsed


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _tbl(headers, rows):
    out = ["| " + " | ".join(map(str, headers)) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return out


def build_report(states, cfg, langs, years, elapsed, status, probe_error):
    L = ["# TASK 3b — CC-News re-probe report", "",
         f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
         f"- Elapsed (cumulative): {elapsed:.0f}s · scan status: **{status}**",
         f"- Repo: `{cfg['repo']}` · fields: date=`{cfg['field_date']}`, "
         f"publisher=`{cfg['field_publisher']}`, text=`{cfg['field_text']}`, "
         f"lang_score≥{cfg['min_language_score']}", ""]
    if probe_error:
        L += [f"> **Probe did not run:** {probe_error}", "",
              "> ```", "> pip install datasets",
              "> python src/probe_ccnews.py --langs hi,ar,uk --years 2016-2024",
              "> ```", ""]
    L += ["> **Methodology caveat — read before trusting the coverage numbers.** "
          "This is a bounded scan of the *head* of each crawl-config's parquet shards "
          "(per-year row cap), and those shards are **not** shuffled by publication "
          "date. So a publisher's articles cluster into whatever date window the first "
          "shards happen to cover: the **volume** figures are informative lower bounds, "
          "but the **clean-months / max-gap** figures understate true coverage and must "
          "not be read as evidence a publisher lacks a 36-month span. Confirm coverage "
          "with a full (or shard-shuffled) extraction before ruling a language in or out "
          "on coverage grounds; rule out only on volume.", ""]

    passes, obs_drop, obs_rate = {}, {}, {}
    for l in langs:
        st = states[l]
        L.append(f"## {l}")
        L.append("")
        # funnel
        L.append("**Acceptance funnel** (one row removed per stage):")
        L.append("")
        L += _tbl(["stage", "count"], [
            ["rows seen (lang-matched)", f"{st['lang_match']:,}"],
            [f"language_score ≥ {cfg['min_language_score']}", f"{st['score_ok']:,}"],
            ["published_date parses", f"{st['date_ok']:,}"],
            ["publisher present", f"{st['pub_ok']:,}"],
            [f"≥{MIN_DOC_WORDS} ICU words (VALID)", f"**{st['valid']:,}**"],
        ])
        dropped_by_score = st["score_dropped"]
        pct_score = 100 * dropped_by_score / st["lang_match"] if st["lang_match"] else 0
        obs_drop[l] = pct_score
        parse_rate = 100 * st["date_ok"] / st["score_ok"] if st["score_ok"] else 0.0
        L.append(f"published_date parse rate after language filter: **{parse_rate:.1f}%**. "
                 f"Dropped by score<{cfg['min_language_score']}: {dropped_by_score:,} "
                 f"({pct_score:.1f}% of matched).")
        L.append("")
        # date formats
        if st["date_formats"]:
            L.append("published_date formats: " + ", ".join(
                f"`{k}`×{v:,}" for k, v in sorted(st["date_formats"].items(),
                                                  key=lambda x: -x[1])))
            nonstd = {k: v for k, v in st["date_formats"].items() if k != "%Y-%m-%d"}
            if nonstd:
                L.append(f"> ⚠️ non-`YYYY-MM-DD` formats seen: {nonstd}")
            L.append("")
        # score hist
        if st["score_hist"]:
            top_h = sorted(st["score_hist"].items(), key=lambda x: x[0])
            L.append("language_score histogram (bucketed): " +
                     ", ".join(f"{k}:{v:,}" for k, v in top_h if k != "nan"))
            L.append("")
        # crawl x pub crosstab
        if st["crawl_pub"]:
            pub_years = sorted({py for d in st["crawl_pub"].values() for py in d})
            rows = []
            for cy in sorted(st["crawl_pub"]):
                rows.append([cy] + [f"{st['crawl_pub'][cy].get(py, 0):,}" for py in pub_years])
            L.append("**Crawl-year config × publication year** (valid rows):")
            L.append("")
            L += _tbl(["crawl↓ / pub→"] + pub_years, rows)
        # top publishers
        top = sorted(st["publisher_counts"].items(), key=lambda kv: (-kv[1], kv[0]))[:15]
        if top:
            L.append("Top 15 publishers by valid volume:")
            L.append("")
            L += _tbl(["publisher", "articles"], [[p, f"{c:,}"] for p, c in top])
        # panel check
        ok, panel, diags = evaluate_language(st)
        passes[l] = ok
        L.append("Top-3 publisher panel check (each ≥15K over ≥36 months, gap ≤2):")
        L.append("")
        L += _tbl(["publisher", "articles", "clean months", "max gap", "ok"],
                  [[d["publisher"], f"{d['articles']:,}", d["clean_months"],
                    d["max_gap"], "yes" if d["ok"] else "no"] for d in diags])
        if panel:
            L.append(f"**Recommended panel (lower bound):** {panel['publishers']} over "
                     f"{panel['range']}.")
        else:
            L.append("**No 3-publisher panel qualifies** (on this partial scan) — candidate "
                     "for substitution by Turkish/Russian from MLSUM (spec §4.3).")
        L.append("")
        # categories/tags
        cat_cov = 100 * st["cat_nonempty"] / st["valid"] if st["valid"] else 0
        tag_cov = 100 * st["tag_nonempty"] / st["valid"] if st["valid"] else 0
        L.append(f"Secondary-topic coverage: `categories` non-empty in "
                 f"**{cat_cov:.1f}%**, `tags` in **{tag_cov:.1f}%** of valid rows.")
        if st["category_values"]:
            topc = sorted(st["category_values"].items(), key=lambda x: -x[1])[:10]
            L.append("Top 10 category values: " + ", ".join(f"`{k}`({v:,})" for k, v in topc))
        L.append("")
        # words + throughput
        wc = np.array(st["word_counts"]) if st["word_counts"] else np.array([0])
        med = int(np.median(wc))
        rate = st["valid"] / elapsed if elapsed else 0.0
        seen_rate = st["lang_match"] / elapsed if elapsed else 0.0
        proj_hours = (100_000 / rate / 3600) if rate else float("inf")
        obs_rate[l] = (seen_rate, rate, proj_hours)
        L.append(f"Median ICU words/doc: **{med}**. Throughput: "
                 f"{seen_rate:.0f} matched rows/s, {rate:.1f} valid rows/s → projected "
                 f"**{proj_hours:.1f} h** to extract 100K valid docs.")
        L.append("")

    # STATUS
    gate0 = all(
        (100 * states[l]["date_ok"] / states[l]["score_ok"] if states[l]["score_ok"] else 0) >= 80.0
        for l in langs) and not probe_error
    g = {"hi": None, "ar": None, "uk": None}
    for l in ("hi", "ar", "uk"):
        if l in passes:
            g[l] = passes[l]
    gate8 = any((100 * states[l]["cat_nonempty"] / states[l]["valid"] if states[l]["valid"] else 0) >= 50.0
                for l in langs)

    surprises, blockers = [], []
    if probe_error:
        blockers.append(probe_error)
    for l in langs:
        if l in passes and not passes[l]:
            surprises.append(f"{l}: no CC-News 3-publisher panel qualifies (partial scan) "
                             "— consider Turkish/Russian from MLSUM.")
    partial = status in ("cap", "exhausted")
    if partial and not probe_error:
        surprises.append("Scan is PARTIAL — all PASS/FAIL below are lower bounds.")

    L.append("## STATUS")
    L.append("")
    L.append("```")
    def pf(b): return "PASS" if b else "FAIL"
    lb = " (lower bound)" if partial else ""
    L.append(f"GATE 0 — published_date parses at >=80% after language filtering:     {pf(gate0)}")
    for l, gate in (("hi", "GATE 5"), ("ar", "GATE 6"), ("uk", "GATE 7")):
        val = g.get(l)
        L.append(f"{gate} — {l}: top-3 publishers each >=15K over >=36 clean months:  "
                 f"{pf(bool(val))}{lb if val else ''}")
    L.append(f"GATE 8 — a secondary topic field exists for >=50% of rows:            {pf(gate8)}")
    L.append("")
    L.append("OBSERVATION — rows/second, and projected hours to extract 100K docs/language:")
    for l in langs:
        sr, r, ph = obs_rate.get(l, (0, 0, float('inf')))
        L.append(f"    {l}: {sr:.0f} matched/s, {r:.1f} valid/s -> {ph:.1f} h")
    L.append("OBSERVATION — % of rows dropped by the language_score >= 0.90 filter, per language:")
    for l in langs:
        L.append(f"    {l}: {obs_drop.get(l, 0):.1f}%")
    L.append("")
    fatal = bool(blockers)
    p5_7 = all(bool(g[l]) for l in ("hi", "ar", "uk") if l in g and g[l] is not None) and \
        all(g[l] is not None for l in ("hi", "ar", "uk"))
    if fatal:
        verdict = "BLOCKED"
    elif gate0 and p5_7 and not surprises:
        verdict = "PROCEED"
    else:
        verdict = "PROCEED WITH CAVEATS"
    L.append(f"VERDICT: {verdict}")
    L.append("Blockers:")
    L += [f"  - {b}" for b in blockers] if blockers else ["  - none"]
    L.append("Surprises worth a human decision:")
    L += [f"  - {s}" for s in surprises] if surprises else ["  - none"]
    L.append("```")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description="TASK 3b — CC-News re-probe")
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--langs", default="hi,ar,uk")
    ap.add_argument("--years", default="2016-2024")
    ap.add_argument("--target-valid", type=int, default=150_000)
    ap.add_argument("--max-seconds", type=int, default=2700)
    ap.add_argument("--per-year-rows", type=int, default=350_000,
                    help="max rows streamed per crawl-year config (bounds time, "
                         "guarantees every year is sampled)")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.params))["ccnews"]
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    years = parse_years(args.years)
    target = args.target_valid
    max_seconds = args.max_seconds
    per_year_rows = args.per_year_rows
    if args.demo:
        target, max_seconds, per_year_rows = 200, 120, 20_000

    states, prev_elapsed = _load_resume(langs)
    t0 = time.time() - prev_elapsed

    probe_error = ""
    try:
        import datasets  # noqa
    except Exception as e:
        probe_error = f"HuggingFace `datasets` not installed ({e!r})."

    status = "not-run"
    if not probe_error:
        try:
            states, status = run_probe(cfg, langs, years, target, max_seconds, states,
                                       t0, per_year_rows)
        except Exception as e:
            probe_error = f"probe failed at runtime: {type(e).__name__}: {str(e)[:300]}"

    elapsed = time.time() - t0
    report = build_report(states, cfg, langs, years, elapsed, status, probe_error)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write(report)
    log(f"Report written to {REPORT_PATH} (status={status})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
