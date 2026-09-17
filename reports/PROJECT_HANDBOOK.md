# PROJECT HANDBOOK — setup, DVC/Git, file-by-file reference, adding languages

**Project:** Label-Free Concept Drift Detection via Tokenizer Fertility
**Repo root:** `Drift-Detection/`
**Audience:** a collaborator who needs to clone the project onto a new machine, reproduce
every result, understand exactly what every file and function does, and extend it to new
languages (Turkish, English).

This document is the *engineering* reference. The companion document
[`PROJECT_WORKFLOW.md`](PROJECT_WORKFLOW.md) explains the *science* — what each experiment
does and why, with worked examples. Read this one to run the code; read that one to
understand the results.

---

## 0. TL;DR — reproduce everything in six commands

```bash
git clone <your-repo-url> Drift-Detection && cd Drift-Detection
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # (or the explicit list in §3)
dvc pull                                    # downloads the 4.3 GB raw corpus + interim/features from Google Drive
dvc repro                                   # rebuilds the whole DAG (prepare → streams → fertility → detect → classify → embed)
less reports/T9_report.md                   # read the final result
```

If `dvc pull` cannot reach the Drive remote, see §5.4 (auth) and §6 (getting data another way).

---

## 1. What this repository is

A fully reproducible research pipeline. It takes raw Bangla news (the Potrika corpus),
builds a chronological "stream," computes several **label-free drift signals** (tokenizer
fertility and variants) plus supervised and embedding **baselines**, calibrates every
detector to a matched false-alarm rate, and measures whether any of them detect real
concept drift earlier than watching a model's error rate. The answer, and the machinery
that establishes it honestly, live in `reports/`.

Everything is driven by:

- **`params.yaml`** — the single source of all configuration (seeds, thresholds, tokenizer
  list, event dates, etc.). No magic numbers live in code.
- **`dvc.yaml`** — the pipeline graph: which script produces which artifact from which
  inputs and parameters.
- **`src/*.py`** — eight scripts, one per pipeline stage.
- **DVC + Git** — Git tracks *code and small text* (scripts, params, reports, `.dvc`
  pointers); DVC tracks *large data and derived artifacts* (the 4.3 GB corpus, parquet
  files, feature matrices) and pushes their bytes to a Google Drive remote.

---

## 2. Prerequisites

| Requirement | Notes |
|---|---|
| Linux (tested on Ubuntu) | macOS works; commands assume bash. |
| **Python 3.12** | The `.venv` was built with 3.12.3. |
| System package for PyICU | `sudo apt-get install libicu-dev pkg-config` **before** `pip install pyicu`. |
| ~10 GB free disk | 4.3 GB raw corpus + ~0.5 GB interim/features + DVC cache. |
| NVIDIA GPU (optional) | Only Part 4 (LaBSE/MMD embeddings, `src/embeddings.py`) uses CUDA. An RTX 5060 Ti (16 GB) was used. Everything else is CPU. |
| HuggingFace account + token (optional) | Only needed to load the **gated** `meta-llama/Llama-3.2-1B` tokenizer. `huggingface-cli login` once; the token lands in `~/.cache/huggingface/token`. Without it, drop Llama from `params.yaml` `tokenizers:` or let `fertility.py` fall back to BLOOM. |
| Google account with access to the Drive remote | For `dvc pull`/`dvc push`. The remote folder id is in `.dvc/config`. |

---

## 3. Environment setup

```bash
cd Drift-Detection
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# system dep for PyICU FIRST (Debian/Ubuntu):
#   sudo apt-get install libicu-dev pkg-config

pip install \
  "numpy" "pandas" "pyarrow" "pyyaml" "matplotlib" \
  "pyicu" \
  "transformers" "tokenizers" \
  "river" "scipy" \
  "datasets" \
  "dvc" "dvc-gdrive"

# GPU stack for Part 4 only (RTX 50-series/Blackwell needs the CUDA 12.8 build):
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install sentence-transformers
```

**Exact versions this project was built and verified against** (freeze these in
`requirements.txt` for the collaborator):

```
python 3.12.3
numpy 2.5.3        pandas 3.0.5       pyarrow 25.0.1     pyyaml 6.0.3
matplotlib 3.11.1  pyicu 2.16.2
transformers 5.16.1  tokenizers 0.23.2
torch 2.14.0       sentence-transformers 6.0.1
river 0.26.1       scipy 1.18.1       datasets 5.0.1
dvc 3.67.1         dvc-gdrive 3.0.1
```

> **Determinism note.** Results are byte-identical across runs *given the same package
> versions* (parquet writers and tokenizers can change output subtly across major versions).
> If a collaborator upgrades `pyarrow`, `transformers`, or `tokenizers`, expect the hashes to
> change even though the numbers stay the same. Pin versions if byte-identity matters.

**Be gentle on a shared machine.** The long CPU jobs were always launched thread-limited and
niced so they don't starve other work — do the same:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 15 python src/<script>.py ...
```

---

## 3.5 Credentials & auth — placing the shared tokens (do this once, right after cloning)

Three separate credentials are needed. The repo owner shares two of them (HF token, Drive
credentials); GitHub write access the owner grants on GitHub itself. **None of these live in
the Git repo** — they sit in your home directory / gitignored files, so cloning does not give
them to you; you install them by hand once.

### 3.5.1 GitHub write access (so you can `git push`/`pull` any time)

1. The owner adds you as a **collaborator** on the GitHub repo (GitHub → repo → Settings →
   Collaborators → add your username). Accept the email invite.
2. Set your identity so commits are attributed to you:
   ```bash
   git config user.name  "Your Name"
   git config user.email "you@example.com"
   ```
3. Authenticate your pushes. Easiest is a **Personal Access Token (PAT)** over HTTPS:
   - GitHub → Settings → Developer settings → Fine-grained tokens → generate one with
     `Contents: Read/Write` on this repo.
   - First `git push` will prompt for username + password; paste the **PAT as the password**.
     Cache it: `git config --global credential.helper store` (or use the `gh` CLI:
     `gh auth login`). Or use an **SSH key** (`ssh-keygen`, add the public key to GitHub, and
     clone with the `git@github.com:...` URL).
4. Verify: `git pull` then `git commit --allow-empty -m "auth test" && git push` (then
   `git reset --hard origin/main` if you want to drop the empty commit).

### 3.5.2 HuggingFace token (only needed for the gated Llama tokenizer)

`params.yaml` lists `meta-llama/Llama-3.2-1B`, which is **gated** — HuggingFace only serves it
to accounts that accepted its licence and present a token. Two ways to satisfy it:

- **Use the owner's shared token** (they send you the `hf_...` string privately). Install it:
  ```bash
  huggingface-cli login          # paste the hf_... token when prompted
  #   -> writes ~/.cache/huggingface/token   (this is where the code looks for it)
  # OR, non-interactively:
  export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx      # add to ~/.bashrc to persist
  ```
- **Or make your own HF account**, click "Agree and access repository" on the Llama-3.2-1B
  model page, then `huggingface-cli login` with *your* token.
- **Or skip Llama entirely:** delete the `meta-llama/Llama-3.2-1B` line from `params.yaml`
  `tokenizers:`. Nothing else depends on it — the other four tokenizers still give the full
  cross-family comparison. (`fertility.py` also auto-falls back to BLOOM if a tokenizer fails
  to load, and prints the substitution.)

Verify: `python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B'); print('Llama OK')"`.

### 3.5.3 Google Drive credentials for DVC (so `dvc pull`/`push` work)

The DVC remote is a Google Drive folder (`.dvc/config` → `gdrive://1mOcVF9s...`). You need
Drive credentials to read/write those bytes. Pick one:

- **Owner shares their cached OAuth credentials.** DVC-gdrive stores a token file after the
  first browser login. The owner sends you their
  `.dvc/tmp/gdrive-user-credentials.json`; you place it at exactly that path in your clone:
  ```bash
  mkdir -p .dvc/tmp
  cp /path/to/received/gdrive-user-credentials.json .dvc/tmp/gdrive-user-credentials.json
  ```
  (It is gitignored, so it never gets committed.)
- **Or first-run browser OAuth (your own Google account).** Just run `dvc pull`; DVC-gdrive
  opens a browser, you approve, and it caches the token under `.dvc/tmp/` for next time. The
  owner must have **shared the Drive folder** with your Google account (Editor access) first.
- **Or a service account (best for headless servers).** The owner shares a service-account
  JSON and shares the Drive folder with the service-account email; you configure it locally
  (kept out of Git via `.dvc/config.local`):
  ```bash
  dvc remote modify gdrive gdrive_use_service_account true
  dvc remote modify --local gdrive gdrive_service_account_json_file_path /path/to/sa.json
  ```

Verify: `dvc status -c` (checks the remote) then `dvc pull` (downloads the corpus). If it
downloads `data/raw/bn_potrika/`, you're set.

> **Golden rule:** credentials are personal secrets. Never `git add` a token, a
> `*-credentials.json`, or a service-account key. They belong only in `~/.cache/huggingface/`,
> `.dvc/tmp/`, or `.dvc/config.local` (all gitignored).

---

## 4. How to copy the project onto another machine

There are two situations. Pick one.

### 4.1 From GitHub (recommended — code + pointers, data pulled from Drive)

Git holds only code, `params.yaml`, `dvc.yaml`, `dvc.lock`, the reports, and the tiny
`data/raw/bn_potrika.dvc` **pointer** file (3 lines with an MD5). It does **not** hold the
4.3 GB corpus.

```bash
git clone <your-repo-url> Drift-Detection
cd Drift-Detection
# create venv + install (see §3)
dvc pull                 # reads the MD5 pointers, downloads the real bytes from the Drive remote
```

After `dvc pull` you will have `data/raw/bn_potrika/` (the corpus), and — if they were
pushed — `data/interim/`, `features/`, and `results/calibration.json` restored from cache.
If only the raw data was pushed, run `dvc repro` to regenerate the rest.

### 4.2 From a device / USB / rsync (full byte copy, no Drive needed)

Copy the **entire** directory *including hidden folders*, most importantly `.dvc/cache`
(this is where DVC stores the actual bytes locally) and `.git`:

```bash
rsync -av --progress \
  "Drift-Detection/" /media/usb/Drift-Detection/       # source → destination
# or, onto another host:
rsync -av --progress -e ssh "Drift-Detection/" user@host:~/Drift-Detection/
```

Then on the destination:

```bash
cd Drift-Detection
# venv + install (see §3)
dvc checkout            # materialises data/raw, data/interim, features/... from the copied .dvc/cache
```

`dvc checkout` reads `dvc.lock` + `*.dvc`, finds the matching hashes in `.dvc/cache`, and
hard-links/copies them into place. No network required because the cache came along in the
rsync.

> **Common trap:** if you copy the working files but *not* `.dvc/cache`, DVC has the
> pointers but not the bytes, and `dvc checkout` fails. Either include the cache in the copy
> (§4.2) or `dvc pull` from Drive (§4.1).

---

## 5. DVC — how it is set up and what every piece does

### 5.1 The mental model

- **Git** = version control for *text*: `src/`, `params.yaml`, `dvc.yaml`, `dvc.lock`,
  `reports/`, and `.dvc` pointer files.
- **DVC** = version control for *bytes*: it hashes large files/dirs into `.dvc/cache/`
  (content-addressed by MD5), leaves a tiny pointer in Git, and can push/pull those bytes to
  a remote (here, Google Drive). Git stays small; data stays reproducible.

### 5.2 What is DVC-tracked here

Two kinds:

1. **A tracked data directory** — `data/raw/bn_potrika/`, added once with `dvc add`. Its
   pointer is `data/raw/bn_potrika.dvc`:
   ```yaml
   outs:
   - md5: d2253f905c1d17dede543f972b705995.dir
     size: 4282557755        # 4.3 GB
     nfiles: 71
     path: bn_potrika
   ```
   The real 71 CSVs live in `.dvc/cache`, not in Git.

2. **Pipeline outputs** — declared as `outs:` of stages in `dvc.yaml` (see §5.5). These are
   gitignored (see `data/.gitignore`, `features/.gitignore`, `results/.gitignore`) and
   tracked by DVC via `dvc.lock`.

### 5.3 The remote

`.dvc/config`:
```ini
[core]
    remote = gdrive
['remote "gdrive"']
    url = gdrive://1mOcVF9s3TP7bygDzYlIswoTpnIUJL22I
```
The default remote is a Google Drive folder (id `1mOcVF9...`). `dvc push` uploads cached
bytes there; `dvc pull` downloads them.

### 5.4 Google Drive auth (first `dvc push`/`dvc pull` on a new machine)

`dvc-gdrive` opens a browser OAuth flow the first time and caches credentials under
`.dvc/tmp/`. On a headless box, either:
- run it once on a machine with a browser and copy `.dvc/tmp/gdrive-user-credentials.json`, or
- use a **service account**: share the Drive folder with the service-account email, then
  ```bash
  dvc remote modify gdrive gdrive_use_service_account true
  dvc remote modify gdrive --local gdrive_service_account_json_file_path /path/to/sa.json
  ```
Credentials are personal — never commit them (they belong in `.dvc/config.local`, which is
gitignored).

### 5.5 The pipeline graph (`dvc.yaml`)

Seven stages. Each declares `cmd` (what runs), `deps` (files that trigger a rerun if
changed), `params` (keys in `params.yaml` that trigger a rerun if changed), and `outs`
(produced artifacts DVC tracks).

| stage | cmd | key deps | params | outs |
|---|---|---|---|---|
| **prepare** | `python src/prepare.py --lang bn --params params.yaml` | `data/raw/bn_potrika`, `src/prepare.py` | `seed, data, bn` | `data/interim` |
| **streams** | `python src/streams.py` | `data/interim`, `src/streams.py` | `seed, streams` | `data/streams` |
| **fertility** | `python src/fertility.py` | `data/interim`, `data/streams`, `src/fertility.py` | `tokenizers, window` | `features/fertility`, `features/windows` |
| **detect** | `python src/detect.py --params params.yaml` | `features/windows`, `src/detect.py` | `detect, window` | `results/calibration.json`; **metric** `results/detection_metrics.json` |
| **classify** | `python src/classifier.py --params params.yaml` | `data/interim`, `features/windows`, `src/classifier.py` | `classifier` | `features/classifier` |
| **embed** | `python src/embeddings.py --params params.yaml` | `data/interim`, `src/embeddings.py` | `embeddings` | `features/embeddings` |

Run the whole thing with **`dvc repro`** (runs only stages whose deps/params changed),
inspect it with **`dvc dag`**, and show metrics with **`dvc metrics show`**.

> **Important wiring caveat.** The `detect` stage produces the T5–T9 reports and reads
> `features/classifier/` and `features/embeddings/` *opportunistically at runtime*, but those
> are **not** declared as its `deps`. So after re-running `classify` or `embed`, `dvc repro`
> will **not** automatically re-run `detect`. Force it: `dvc repro -f detect`. To make it
> automatic, add `-d features/classifier -d features/embeddings` to the `detect` stage.
> Also: the reports in `reports/` are **git-committed**, not DVC `outs` — the `detect` script
> writes them as a side effect.

### 5.6 The most useful DVC commands

```bash
dvc status                 # what's out of date vs dvc.lock
dvc repro                  # rebuild changed stages (whole graph)
dvc repro fertility        # rebuild one stage (+ upstream if needed)
dvc repro -f detect        # force-rerun a stage even if DVC thinks it's cached
dvc dag                    # ASCII pipeline graph
dvc metrics show           # print results/detection_metrics.json
dvc metrics diff v1 v2     # compare metrics between two git tags  (viva demo)
dvc push -j 4              # upload cached bytes to Drive (4 parallel)
dvc pull -j 4              # download them
dvc checkout               # materialise files from local .dvc/cache (no network)
dvc gc -w                  # garbage-collect cache not referenced by the current workspace
```

---

## 6. Getting the data

| Corpus | Where it lives | How to get it |
|---|---|---|
| **Bangla — Potrika** | `data/raw/bn_potrika/` (DVC-tracked, 4.3 GB, 71 CSVs) | `dvc pull` (or `dvc checkout` from a copied cache). Original source: Mendeley `v362rp78dc`. |
| **English — NewsSumm** | `data/raw/en_newssumm/` — **currently empty (0 files)** | Download from Zenodo `https://zenodo.org/records/17670865` and unzip here. Then audit (see §9). |
| **Hindi/Arabic/Ukrainian — CC-News** | *not stored on disk* | Streamed on demand by `src/probe_ccnews.py` from `stanford-oval/ccnews` (HuggingFace). Counts only; no corpus written. |
| **Turkish/Russian — MLSUM** | *not stored yet* | To be probed/streamed from `reciTAL/mlsum` (see §11.2). |

---

## 7. Repository layout

```
Drift-Detection/
├── params.yaml                # ALL configuration (see §10)
├── dvc.yaml / dvc.lock        # pipeline graph + locked input/output hashes
├── requirements.txt           # (create from §3 version list)
├── .dvc/                       # DVC internals: config (remote), cache (bytes)
├── data/
│   ├── raw/
│   │   ├── bn_potrika/         # DVC-tracked corpus (RawDataset/ = the usable, dated one;
│   │   │                       #   BalancedDataset/ = undated, unused)
│   │   ├── bn_potrika.dvc      # 3-line MD5 pointer (in Git)
│   │   └── en_newssumm/        # EMPTY — put NewsSumm here
│   ├── interim/                # prepare.py output: bn_panel.parquet, bn_full.parquet (DVC out, gitignored)
│   └── streams/                # streams.py output: permutation .npz files (DVC out)
├── features/
│   ├── fertility/              # per-document token counts, one parquet per (stream,tokenizer)
│   ├── windows/                # per-window signal matrices, one parquet per (stream,tokenizer,perm)
│   ├── type_fertility/         # per-word-type piece counts, one parquet per tokenizer
│   ├── classifier/             # S6 per-window error series
│   └── embeddings/             # S5 MMD series + LaBSE embedding cache (bn_panel.npy)
├── results/
│   ├── calibration.json        # frozen ADWIN delta* per (signal,tokenizer,variant,FAR target)
│   ├── detection_metrics.json  # DVC metric file
│   ├── classifier_{calibration,metrics}.json
│   └── embeddings_{calibration,status}.json
├── reports/                    # ALL human-readable outputs (git-committed)
│   ├── T1_report.md … T9_report.md
│   ├── ccnews_probe.json       # resumable probe state
│   ├── figs/*.png              # 21 figures
│   ├── PROJECT_HANDBOOK.md     # this file
│   └── PROJECT_WORKFLOW.md     # the science
├── src/                        # the eight pipeline scripts (see §8)
├── md_files/                   # the task briefs (FERTILITY_DRIFT_SPEC.md + T1..T9) — the "orders"
└── logs/                       # tee'd run logs
```

---

## 8. Source files — what each does and every function in it

Every script: reads `params.yaml`, seeds everything with `42`, prints progress to stdout,
supports `--demo` (a tiny fast subsample for smoke tests), and writes its report/artifacts.

### 8.1 `src/audit_potrika.py` (T1 — data audit, 937 lines)

Read-only sanity check of the raw corpus before any processing. Produces
`reports/T1_report.md` + `reports/figs/T1_coverage.png`.

- `discover_files(input_dir)` — recursively find every CSV under `data/raw/bn_potrika`.
- `count_rows_light(path, col)` — count rows by streaming one column (handles embedded
  newlines inside quoted article bodies that would break naive line counting).
- `inventory(files, input_dir)` — per-file: size, format, row count, columns, dtypes; groups
  files by identical schema (discovers the two families: dated `RawDataset`, undated
  `BalancedDataset`).
- `load_metadata(dated_files, input_dir)` — load only the light columns (Date, Source,
  Category, Heading) for every row; normalises source names.
- `load_text_sample(...)` — stream the article body, keeping each row with probability
  `keep_prob` (memory-bounded sampling for the text-quality sections).
- `parse_dates(raw)` — parse the `Date` column with an ordered list of candidate formats,
  then a permissive fallback; returns per-row matched-format labels + counts (so you know
  *which* format each row needed).
- `wc_whitespace / wc_regex / wc_icu` — three word-count methods; `wc_icu` uses PyICU
  `BreakIterator` (UAX-#29 word boundaries) and is the project-wide word definition.
- `norm_source`, `norm_text_for_hash`, `text_hash`, `pctl` — helpers.
- `norm_cat` + union-find (`find`) inside `main` — detect category-label collisions
  (e.g. `science-and-tech` vs `Science_Technology`).
- `Report` class — accumulates markdown lines/tables; `write_status`/`_write` — emit the
  gated STATUS block.

### 8.2 `src/prepare.py` (T2 — build the Bangla streams, 602 lines)

Cleans the raw pool once and emits two parquet streams. Output:
`data/interim/bn_panel.parquet` (99,900 docs, publisher-balanced) and
`data/interim/bn_full.parquet` (95,935 docs, all six publishers). Report `reports/T2_report.md`.

- `read_raw(raw_dir, demo)` — read `RawDataset/**/*.csv` (skips undated BalancedDataset),
  keeping only needed columns.
- `norm_source` — map raw source spellings (`kaler_kontho`, `Somoyer_Alo`) to canonical
  display names (`Kaler Kontho`, `Somoyer Alo`).
- `normalise_text(t)` — whitespace-collapse + strip + lowercase (used for dedup + doc_id).
- `doc_id_of(norm)` — stable id `bn_` + first 12 hex of `sha1(normalised text)`.
- `icu_wordcount(t)` — ICU word count (the frozen word definition).
- `clean_pool(raw, ledger)` — the cleaning pipeline, one counted rule at a time: drop
  index col, rename to canonical (`News→text`, `Category→topic`, `Heading→headline`,
  `Date→date`, `Source→publisher`), parse date (`%Y/%m/%d`, drop failures), canonicalise
  topic labels, drop empty text, drop exact duplicates (keep earliest), compute ICU
  `n_words`, drop docs `< min_doc_words`.
- `build_panel(pool, publishers, start, end, cap, seed)` — the **panel design**: keep only
  Inqilab/Jugantor/Kaler Kontho over 2016-01..2020-12, sample an equal quota per
  (month × publisher) cell (555 = 100000/(60·3)) so the publisher mix is constant in every
  year → any drift is lexical, not compositional. Returns the panel + a fill table.
- `build_full(pool, start, end, cap, seed)` — the robustness stream: all six publishers,
  uniform-across-time (equal per-month quota).
- `sample_group(g, quota, seed)` — deterministic within-cell sampling (sort by doc_id,
  seeded `.sample`).
- `write_parquet(df, path)` — write with an explicit pyarrow schema (`doc_id, text,
  date=date32, publisher=dictionary, topic, lang, n_words=int32`), zstd, fixed row-group
  size, sorted by (date, doc_id) → byte-identical output.
- `stream_summary`, `Report`, `main` — reporting + orchestration.

### 8.3 `src/probe_ccnews.py` (T3b — CC-News volume probe, 511 lines)

Network, read-only, writes **counts only**. Streams `stanford-oval/ccnews` and measures how
much usable (valid date + publisher + ≥50 words) data exists for hi/ar/uk. Resumable via
`reports/ccnews_probe.json`. Report `reports/T3b_report.md`.

- `parse_years`, `parse_date` — spec parsing.
- `icu_wordcount` — ICU words (for the ≥50-word filter).
- `empty_state()` — per-language accumulator (funnel counters, publisher/month histograms,
  date formats, word-count sample, categories/tags coverage).
- `run_probe(cfg, langs, years, target, max_seconds, states, t0, per_year_rows)` — the
  streaming scan: per row apply the acceptance funnel (lang match → `language_score ≥ 0.90`
  → date parses → publisher present → ≥50 ICU words); per-year row cap keeps the scan
  temporally representative; saves JSON after each year.
- `longest_clean_span(month_counts)` — longest run of months with no gap > 2 (panel
  coverage check).
- `evaluate_language(st)` — the panel condition: top-3 publishers each ≥15k over ≥36 clean
  months.
- `_save`, `_load_resume`, `build_report`, `main` — persistence + reporting.

### 8.4 `src/streams.py` (T3 Part 1 — stream permutations, 116 lines)

Turns each parquet into 11 index permutations (identity + 10 shuffles). Output
`data/streams/{bn_panel,bn_full}_perms.npz`.

- `n_rows(parquet_path)` — row count from parquet metadata.
- `build_perms(n, num_shuffles)` — `perm_00` = identity (the real chronological order);
  `perm_01..10` = full-array shuffles with seeds 42..51 (the **temporal null**).
- `verify(arrays, n)` — assert each is a valid bijection over `[0,n)`, `perm_00` is identity,
  no two permutations identical.
- `main` — writes the `.npz` + `data/streams/streams_meta.json`.

### 8.5 `src/fertility.py` (T3+T4 — signals, windowing, z-scoring, 1329 lines)

The heart of feature generation. Tokenises every document once per tokenizer, packs the
stream into ~2000-word windows, computes signals S1–S9 per window, z-scores them against the
split calibration epoch, writes `features/fertility/*`, `features/windows/*`,
`features/type_fertility/*`, `features/windows/zscore_params.json`, and reports
`reports/T3_report.md` + `reports/T4_report.md` (+ figures).

- `icu_words(text)` — ICU word tokens (lowercased) — used for word *types* (S4) and lengths.
- `detect_family(tok)` — classify a tokenizer as **wordpiece** (mBERT), **sentencepiece**
  (XLM-R), or **byte_bpe** (Qwen/Llama/BLOOM) by the *fraction* of vocab carrying each
  marker (`▁` U+2581 checked before `Ġ` U+0120, because the byte-level alphabet never
  contains `▁`).
- `build_id_flags(tok, family)` — per-token-id boolean arrays: `is_continuation`,
  `is_byte_fallback` (`<0xHH>` for SP; undefined→null for byte-level), `is_unk` (mBERT).
- `two_word_probe(tok)` — empirically confirm the boundary marker by encoding `নতুন শব্দ`.
- `tokenize_stream(texts, tok, ...)` — batched encoding → per-doc `n_tokens`,
  `n_continuation`, `n_byte_fallback`.
- `build_type_fertility(tok, id_to_type)` — pieces-per-type for every word type, tokenised in
  **word-initial** form (space-prefixed) — needed by S1c/S7/S8/S9.
- `window_boundaries(words_ord, target)` — greedy packing: accumulate whole documents until
  cumulative ICU words ≥ 2000, then close the window (trailing partial window discarded).
- `seg_sum(arr, starts, covered)` — fast per-window sums via `np.add.reduceat`.
- `zscore(values, ref_lo, ref_hi)` — z-score against the **reference epoch** `[5%,10%)`
  (the T4 split protocol); emits null if σ_ref≈0.
- **The signals** (computed per window in `main`):
  - **S1** fertility = Σtokens/Σwords; **S1b** topic-adjusted fertility; **S1c**
    type-weighted fertility (mean f(t) over unique types); **S2** byte-fallback rate; **S3**
    continuation ratio; **S4** unseen-type rate (types not in the frozen vocab epoch);
    **S4_type** the same on unique types; **S7** novel-token fertility mass; **S8/S9** mean
    fertility of novel/seen *types* (type-weighted); **S8tok/S9tok** the token-weighted
    versions that close the dilution identity.
- `_write_report` (T3 report) and `_write_t4_report` (T4 report: split-epoch, per-type
  validation, dilution decomposition, trend tables), plus figure builders
  (`_fig_timeline`, `_fig_grid`, `_fig_decomposition`, `_fig_signal_comparison`).

### 8.6 `src/classifier.py` (T6 Part 3 — supervised reference S6, 253 lines)

River online logistic regression over 2¹⁸ hashed bag-of-words features, predicting
`publisher` (3-class) and `topic` (8-class). Output `features/classifier/*` +
`results/classifier_{calibration,metrics}.json`.

- `hashed_bow(text, nf)` — deterministic hashed bag-of-words (`zlib.crc32(token) % nf`;
  Python's built-in `hash` is *not* used because it is randomised per process).
- `make_model(n_classes, lr)` — `OneVsRestClassifier(LogisticRegression(SGD(lr)))`.
- `per_doc_error_frozen(feats, labels, train_idx, ...)` — train on the first 10% of the real
  stream, freeze, return per-doc 0/1 error (order-independent → **S6**).
- `per_doc_error_preq(feats, labels, order, ...)` — prequential test-then-train in stream
  order (order-dependent → **S6p**).
- `window_series(doc_err, order, bounds, dates_ns)` — per-window mean error + median date.
- `calibrate(null_zstreams, grid, targets)` — FAR calibration of S6/S6p on the shuffled
  nulls (imports `far_curve`/`pick_delta`/`epoch_bounds` from `detect.py`).
- `main` — trains both targets × both variants, z-scores, calibrates, writes metrics
  (frozen accuracy per year, monotonicity flag).

### 8.7 `src/embeddings.py` (T6 Part 4 — MMD baseline S5, 193 lines, GPU)

Encodes every document with LaBSE once, caches to `features/embeddings/bn_panel.npy`
(float16), then computes per-window MMD² against a fixed 2000-doc reference pool. Output
`features/embeddings/s5__perm*.parquet` + `results/embeddings_calibration.json`.

- `rbf_mmd2(X, Y, gamma, kyy_mean)` — biased MMD² with an RBF kernel; the reference-pool
  term `kyy_mean` and bandwidth `gamma` are precomputed once and **frozen** (recomputing per
  window would leak).
- `window_bounds`, `zscore_split` — local copies of the windowing + z-scoring helpers.
- `main` — cache-aware encode (skips the GPU pass if `bn_panel.npy` already matches N),
  median-heuristic bandwidth on the reference pool, per-perm MMD, calibration, status file.
  **Degrades gracefully** if `torch`/`sentence-transformers` are missing (writes a "blocked"
  status with install instructions instead of crashing).

### 8.8 `src/detect.py` (T5–T9 — the analysis engine, 3119 lines)

The largest file. `main` runs T5 (redundancy, sign anomaly, FAR-calibrated ADWIN detection)
then chains `run_t5b → run_t6 → run_t7 → run_t8 → run_t9`, each writing its own report +
figures. Shared primitives:

- `load_perm(tok_slug, pk)` — read a window parquet for a (tokenizer, permutation).
- `epoch_bounds(n, vocab_frac, ref_frac)` — window indices of the vocab/reference epoch
  boundaries.
- `detection_values(df, zcol, ...)` — the z-signal over the detection epoch `[10%, end)`,
  optionally residualised on `n_words`.
- `adwin_alarms(values, delta)` — run River ADWIN over a z-series, return alarm indices.
- `far_curve(null_streams, grid)` / `pick_delta(curve, target)` — sweep δ over the grid on
  the 10 shuffled nulls, compute false-alarm rate, pick the **largest δ with FAR ≤ target**
  (this is the matched-FAR calibration; δ* is frozen to `results/calibration.json`).
- `pearson / spearman / partial_corr` — stats helpers (spearman/partial in numpy so scipy
  isn't strictly required for T5).
- `part2_covariates(...)` — window-level covariates for the sign-anomaly analysis.
- `frozen_delta(...)`, `zscore_ref(...)`, `real_stream_delay(...)`, `_series_delay(...)`,
  `_load_signal_series("S5"/"S6"/"S6p")` — read frozen δ* and compute event detection delays.
- `build_doc_level(...)` — one ICU pass over the panel returning per-doc word-type arrays,
  per-tokenizer token counts, type-piece tables, dates, and window covariates (feeds T6/T8/T9).
- `_synth_zseries(...)` — build one **semi-synthetic** stream (early pool before W*, mixed
  after) and return per-signal z-series; `simulate_stream` derives detection from it,
  `simulate_response` derives the response R.
- `run_t5b` — extended-δ calibration audit, full 5×5 delay grid, paired Wilcoxon, synthetic
  drift injection (5 intensities × 20 replicates), real changepoints, covariate trends.
- `run_t6` — p=0 control arm + excess power, exact decile decomposition of the sign flip,
  reads S5/S6, lead-time-over-S6 table + sign tests.
- `run_t7` — alarm census, event-detection **permutation test** (the validity check), and
  replicate-level bootstrap CIs on excess power.
- `run_t8` — response-curve inversion (later deprecated in T9), pooled-alarm test, power
  analysis of the permutation test, event footprint, sensitivity-floor figure.
- `run_t9` — separates **news turnover** from **event footprint** (seed terms + pre-event
  baseline + excess), the floor-from-real-data correlations, COVID robustness at FAR 1e-4.
- `_fig_*` — one builder per figure.

---

## 9. `params.yaml` — every key explained

```yaml
seed: 42                          # global RNG seed used everywhere
data:
  max_docs_per_language: 100000   # per-language document cap
  min_doc_words: 50               # drop shorter docs (fertility too noisy)
  top_k_publishers: 3             # publisher-panel classes (amended from 6; see T2)
  languages: [bn, en, hi, ar, uk] # the five target languages
bn:
  raw_dir: data/raw/bn_potrika/RawDataset
  panel_publishers: [Inqilab, Jugantor, Kaler Kontho]  # the 3 with full 2016-2020 coverage
  panel_start: "2016-01"
  panel_end:   "2020-12"
window:
  words_per_window: 2000          # greedy packing target
  calibration_fraction: 0.10      # legacy single-epoch (T3), superseded by:
  vocab_fraction: 0.05            # [0,5%)  freeze the S4 word-type vocabulary V
  reference_fraction: 0.05        # [5%,10%) compute mu_ref/sigma_ref for every signal
tokenizers:                       # 5 tokenizers spanning 3 families
  - bert-base-multilingual-cased  #   wordpiece
  - xlm-roberta-base              #   sentencepiece
  - meta-llama/Llama-3.2-1B       #   byte-level BPE (gated → needs HF token)
  - Qwen/Qwen2.5-0.5B             #   byte-level BPE
  - bigscience/bloom-560m         #   byte-level BPE (strong Bengali coverage)
streams:
  num_shuffles: 10                # number of null permutations
adwin:
  delta_grid: [1e-6 … 0.99]       # 14-point sensitivity grid (larger δ = more sensitive)
  target_far: 0.001               # 1 false alarm per 1000 null windows
  far_sensitivity: [0.01, 0.0001] # robustness sweep at 10x looser / stricter
detect:
  changepoint: "2020-03-08"       # COVID reference date t*
  covid_window_days: 60
  signals: [S1, S1c, S3, S4, S7]  # the label-free signals under test
classifier: {n_features: 262144, learning_rate: 0.5, targets: [publisher, topic]}
embeddings: {model: sentence-transformers/LaBSE, max_seq_length: 256, batch_size: 64,
             ref_pool_size: 2000, max_docs_per_window: 200, encode_budget_min: 60,
             subsample_docs: 40000}
synthetic: {intensities: [...], replicates: 20, n_windows: 2000,
            early_years: [2016,2017], late_years: [2020], wstar_low: 0.20, wstar_high: 0.80}
changepoints: [ 7 verified event dates, each with cited source + Bangla seed terms ]
ccnews: { field names + language_score threshold for the CC-News schema }
```

Because `dvc.yaml` lists these keys under each stage's `params:`, **editing a value makes
`dvc repro` rebuild exactly the stages that depend on it** — e.g. change `top_k_publishers`
and `prepare` re-runs; add a tokenizer and `fertility` re-runs.

---

## 10. Operational gotchas (learned the hard way)

1. **`data/interim` can vanish after a bad `dvc` op.** If `dvc checkout`/`pull` removes it
   and the bytes aren't in cache, regenerate deterministically: `dvc repro prepare` (or
   `python src/prepare.py --lang bn`). Output is byte-identical.
2. **`detect` doesn't auto-rerun after `classify`/`embed`.** Use `dvc repro -f detect`
   (§5.5).
3. **`json.dump` + numpy scalars.** Always cast (`float(x)`, `int(x)`) before dumping —
   `numpy.float32` is not JSON-serializable and will truncate the file mid-write. (This bit
   `embeddings.py` once.)
4. **Gated Llama.** Needs `huggingface-cli login`. Otherwise remove it from `tokenizers:`.
5. **CPU courtesy.** Long jobs: `OMP_NUM_THREADS=2 nice -n 15 python ...`.
6. **Determinism depends on package versions** (§3).

---

## 11. COLLABORATOR PLAYBOOK — adding a new language (English, Turkish, or any other)

> **Who this is for:** the second author, who has GitHub push/pull access and will add new
> languages *from different datasets*. **You are not inventing any new method.** You reproduce
> exactly what was done for Bangla, on a new corpus. Read §11.0 twice, then follow §11.2 →
> §11.3/§11.4 → §11.5 → §11.6 in order.

### 11.0 The mental model — what you touch vs what you must never touch

The pipeline is **language-generic by design**. Everything after `prepare.py` consumes one of
two things:
- a parquet with the **canonical schema** `(doc_id, text, date, publisher, topic, lang,
  n_words)`, written to `data/interim/{lang}_panel.parquet` and/or `{lang}_full.parquet`;
- permutation index files `data/streams/{lang}_*_perms.npz`.

So adding a language is **only** (1) getting the raw data, (2) writing a small *loader* that
emits the canonical schema, (3) adding a config block, and (4) a one-time edit that lets the
existing scripts accept a `--lang` argument. **The science does not change.**

| ✅ YOU EDIT / CREATE (data + config only) | ⛔ NEVER TOUCH (the methodology) |
|---|---|
| `params.yaml`: a new `{lang}:` block; (Tier 2) per-language `changepoints` | `fertility.py`: `detect_family`, `build_id_flags`, `tokenize_stream`, `build_type_fertility`, `window_boundaries`, `seg_sum`, `zscore`, and the signal math in `main` |
| A loader: a new `elif` branch in `prepare.py` **or** a copy `src/prepare_{lang}.py` | `detect.py`: `run_t5b/6/7/8/9`, `simulate_stream`, `far_curve`, `pick_delta`, `adwin_alarms`, `epoch_bounds`, all `_fig_*` |
| The **one-time** `--lang` generalization (§11.1) | `classifier.py` model logic; `embeddings.py` MMD math |
| `dvc.yaml`: a `prepare_{lang}` stage | The window size, calibration split, FAR target, seeds — anything under `window:`, `adwin:`, `seed:` |

**Two tiers of work. Do Tier 1 first.**
- **Tier 1 — descriptive label-free analysis (≈ the core cross-language result).** Run
  `prepare → streams → fertility` for the new language and read the T3/T4 numbers (raw mean
  fertility per tokenizer, whether S1/S4/S7 move, the dilution decomposition). This is what
  the cross-language claim mainly needs and it requires **no** events, classifier, or
  embeddings. Files touched: the loader + a params block + the one-time `--lang` edit to
  `streams.py` and `fertility.py`.
- **Tier 2 — full detection (optional, more work).** Additionally provide a list of *verified
  event dates in that language* and run `classifier → embeddings → detect --lang {lang}`.
  Requires the `--lang` edit to `detect.py`/`classifier.py`/`embeddings.py` and a per-language
  `changepoints` list. Only do this if you want the T5–T9 detection/lead-time story for the
  new language too.

---

### 11.1 THE ONE-TIME GENERALIZATION (do this once; then every language is a copy-paste)

The code currently hardcodes Bangla stream names. Generalize them **once** so every script
takes `--lang`. Below is exactly what to change, in which file, in which function, and a
ready-to-paste prompt for an AI coding tool (Claude Code / Cursor) to do each edit safely.

**(a) `src/prepare.py` — dispatch `main()` on `--lang`, add per-dataset loaders.**
Current state: `main()` errors for any lang except bn (line ~304) and does `bn = P["bn"]`
(line 314); `read_raw(raw_dir, demo)` (line 113) reads Potrika CSVs. Change `main()` to read
`cfg = P[args.lang]`, choose the raw loader by a new `cfg["loader"]` key, then run the
**identical** cleaning (`clean_pool`), and build a panel only if `cfg` has `panel_publishers`
(otherwise build the `_full` uniform-across-time stream). Keep `write_parquet`, `clean_pool`,
`icu_wordcount`, `doc_id_of` byte-for-byte unchanged.

> **Prompt to paste into your AI tool:**
> *"In `src/prepare.py`, generalize `main()` to support any `--lang`, without changing any
> cleaning rule, the output schema, or `write_parquet`. Currently it errors unless lang=='bn'
> and reads `bn = P['bn']`. Change it to `cfg = P[args.lang]`. Add a loader dispatch: if
> `cfg['loader'] == 'potrika_csv'` call the existing `read_raw(cfg['raw_dir'], demo)`; if
> `'newssumm_csv'` call a new `read_newssumm(cfg['raw_dir'], demo)` I will describe; if
> `'mlsum_hf'` call a new `read_mlsum(cfg['hf_config'], demo)`. Each loader must return a
> pandas DataFrame with the same raw columns the current code expects going into `clean_pool`
> (News, Category, Heading, Date, Source) — map the dataset's real columns onto those names
> inside the loader. After cleaning, if `cfg` contains `panel_publishers`, build the panel
> exactly as now and write `data/interim/{lang}_panel.parquet`; always also write
> `data/interim/{lang}_full.parquet` via `build_full`. Set the `lang` column to `args.lang`.
> Do not modify `clean_pool`, `build_panel`, `build_full`, `write_parquet`, `icu_wordcount`,
> or `doc_id_of`."*

**(b) `src/streams.py` — auto-discover streams instead of the hardcoded list.**
Current: line 27 `STREAMS = ["bn_panel", "bn_full"]`. Make it discover every parquet in
`data/interim/`.

> **Prompt:** *"In `src/streams.py`, replace the hardcoded `STREAMS = ['bn_panel','bn_full']`
> with auto-discovery: `STREAMS = sorted(os.path.splitext(os.path.basename(p))[0] for p in
> glob.glob('data/interim/*.parquet'))`. Import `glob` if needed. Everything else stays the
> same, so it builds permutation .npz files for every prepared language automatically."*

**(c) `src/fertility.py` — add `--lang`, derive PANEL/FULL, replace `"bn_panel"` literals.**
Current: line 330 `streams = [("bn_panel", ...), ("bn_full", 0)]`; and literal `"bn_panel"`
appears in the report writers and result-dict keys (lines ~420, 600, 616, 625, 798, 815,
1101). These literals key into per-stream result dicts and pick the panel for the T3/T4
headline tables.

> **Prompt:** *"In `src/fertility.py`, add an argparse `--lang` (default 'bn'). Define
> `PANEL = f'{args.lang}_panel'` and `FULL = f'{args.lang}_full'` in `main()`. Replace the
> hardcoded `streams = [('bn_panel', ...), ('bn_full', 0)]` with the same structure using
> PANEL and FULL, but only include a stream if its parquet exists in `data/interim/`. Replace
> every remaining literal `'bn_panel'` (report-table selection and dict keys, ~lines 420, 600,
> 616, 625, 798, 815, 1101) with the `PANEL` variable, and any `'bn_full'` with `FULL`. Pass
> `args.lang` through so report filenames become `reports/T3_{lang}_report.md` and
> `reports/T4_{lang}_report.md` for non-bn languages (keep the bare names for bn to avoid
> churn). Do not change any signal computation, windowing, z-scoring, or figure logic."*

**(d) `src/detect.py`, `src/classifier.py`, `src/embeddings.py` — add `--lang` / `--stream`
(Tier 2 only).**
Current: each has a module-level `STREAM = "bn_panel"` (detect.py:48, classifier.py:44,
embeddings.py:35). `detect.py` also reads `P["changepoints"]` (bn events) in
`run_t5b/6/7/8/9`.

> **Prompt:** *"In `src/detect.py`, `src/classifier.py`, and `src/embeddings.py`, add an
> argparse `--lang` (default 'bn') and set the module/`main`-scope `STREAM = f'{lang}_panel'`
> from it instead of the hardcoded `'bn_panel'`. In `src/detect.py`, make the changepoints
> per-language: read `cps = P['changepoints'][args.lang]` if `P['changepoints']` is a dict
> keyed by language, else fall back to the current flat list; apply this in every place that
> does `P['changepoints']`. Write reports as `reports/T*_{lang}_report.md` for non-bn.
> Change no statistics, no ADWIN, no calibration, no permutation-test, no figure math."*

Commit this generalization once (`git commit -m "generalize pipeline to --lang"`), push, and
you never touch methodology again — adding each subsequent language is only §11.2–§11.6.

---

### 11.2 Getting + auditing the raw data (do this per language, before anything else)

**English — NewsSumm** (goes in `data/raw/en_newssumm/`, currently **empty**; 317k
Indian-English articles, 2000–2025, 36 newspapers, 20+ categories — the longest span, the
high-resource control):
```bash
# download Zenodo record 17670865 into data/raw/en_newssumm/ and unzip, then:
dvc add data/raw/en_newssumm          # track its bytes in DVC (like bn_potrika)
git add data/raw/en_newssumm.dvc data/raw/.gitignore && git commit -m "data: newssumm raw"
dvc push
```
**Turkish — MLSUM** (`reciTAL/mlsum`, config `tu`; ~250k articles with `date`+`topic`; streamed,
not stored on disk — nothing to `dvc add`).

**Always audit first** (reuse the T1 pattern). For a CSV corpus like NewsSumm, copy the
auditor; for a streamed corpus like MLSUM, write a 20-line probe (see §11.4):
```bash
cp src/audit_potrika.py src/audit_newssumm.py     # then point it at data/raw/en_newssumm,
                                                  # map its real columns, check date parsing + per-paper volume
python src/audit_newssumm.py --input data/raw/en_newssumm --out reports/EN1_audit.md
```
The audit answers the two questions that decide the design: **do the dates parse?** and **how
concentrated are the outlets?** (many balanced outlets → publisher panel like Bangla;
one dominant outlet → use topic or `_full` only).

---

### 11.3 English walkthrough (function by function)

1. **Add an `en:` block to `params.yaml`** mirroring `bn:` (NewsSumm has many newspapers, so a
   publisher panel *does* transfer):
   ```yaml
   en:
     loader: newssumm_csv
     raw_dir: data/raw/en_newssumm
     panel_publishers: [<top-3 or top-6 newspapers by volume from the audit>]
     panel_start: "<first month all panel papers cover>"
     panel_end:   "<last such month>"
   ```
2. **Write the loader** `read_newssumm(raw_dir, demo)` inside `prepare.py` (the AI prompt in
   §11.1a covers this). Its *only* job: read NewsSumm's CSVs and return a DataFrame whose
   columns map onto what `clean_pool` expects — `News` (article text), `Category` (topic),
   `Heading` (title), `Date` (publication date), `Source` (newspaper). Everything after that —
   dedup, ICU word count, `<50`-word drop, `doc_id`, panel sampling, parquet write — is the
   **existing, unchanged** code.
3. **Run it:**
   ```bash
   python src/prepare.py --lang en --params params.yaml --report reports/T2_en_report.md
   #   -> data/interim/en_panel.parquet  (+ en_full.parquet)
   python src/streams.py --params params.yaml          # auto-discovers en_panel/en_full
   OMP_NUM_THREADS=2 nice -n 15 python src/fertility.py --lang en --params params.yaml
   #   -> features/fertility/en_panel__*.parquet, features/windows/en_panel__*, reports/T3_en_report.md, T4_en_report.md
   ```
4. **Read `reports/T3_en_report.md`**: raw mean fertility per tokenizer (expect English ~1.2–
   1.5 — far lower than Bangla; that is the point of a high-resource control) and whether
   S1/S4/S7 move across 2000–2025. If fertility doesn't move on a 25-year English span, it
   won't move anywhere — this is the sanity-check language.

> English function-of-file map for this task: **you write** `read_newssumm` in `prepare.py`
> and an `en:` block in `params.yaml`. **You reuse unchanged**: `clean_pool`, `build_panel`,
> `build_full`, `write_parquet` (prepare.py); all of `streams.py`; all of `fertility.py`.

---

### 11.4 Turkish walkthrough (probe first — single-outlet caveat)

1. **Probe outlet concentration + date coverage** (single-outlet dominance is the risk):
   ```bash
   python - <<'PY'
   from datasets import load_dataset
   from collections import Counter
   ds = load_dataset("reciTAL/mlsum", "tu", split="train", streaming=True)
   pub, yr = Counter(), Counter()
   for i, r in zip(range(50000), ds):
       pub[(r.get("url") or "?").split("/")[2] if r.get("url") else "?"] += 1
       yr[str(r.get("date"))[:4]] += 1
   print("top outlets:", pub.most_common(8)); print("years:", sorted(yr.items()))
   PY
   ```
   Decision rule (mirrors T3b): you need ≥3 outlets each with enough volume over ≥36 clean
   months to build a publisher panel. MLSUM-tu will likely **fail** (one dominant outlet).
2. **Because the publisher panel won't transfer**, choose one:
   - **Topic as the label** — MLSUM has clean `topic`, so set `publisher = topic` conceptually
     and predict topic in the S6 baseline; note in the paper "Turkish uses topic labels." OR
   - **Label-free only** — the fertility signals (S1/S1c/S3/S4/S7) need **no** label at all, so
     just build the `_full` stream and skip S6 for Turkish (Tier 1 only).
3. **`tr:` block in `params.yaml`** (no `panel_publishers` → prepare builds only `_full`):
   ```yaml
   tr:
     loader: mlsum_hf
     hf_config: tu
     # no panel_publishers -> uniform-across-time _full stream (single-outlet safe)
   ```
4. **Loader** `read_mlsum(hf_config, demo)` in `prepare.py`: stream `reciTAL/mlsum` config
   `tu`, keep `(text→News, date→Date, topic→Category)`, set `Heading`/`Source` to placeholders
   (title / outlet). Everything downstream unchanged.
5. **Run** (`_full` stream name):
   ```bash
   python src/prepare.py --lang tr --params params.yaml --report reports/T2_tr_report.md
   python src/streams.py --params params.yaml
   OMP_NUM_THREADS=2 nice -n 15 python src/fertility.py --lang tr --params params.yaml
   ```
   Turkish is agglutinative → high baseline fertility and large dynamic range → a good stress
   test for the fertility signal even without a publisher panel.

---

### 11.5 Full run order + how to verify each step

| step | command | produces | verify |
|---|---|---|---|
| 1. prepare | `python src/prepare.py --lang {L} --params params.yaml` | `data/interim/{L}_panel.parquet` / `{L}_full.parquet` | `python -c "import pandas as pd; d=pd.read_parquet('data/interim/{L}_panel.parquet'); print(d.shape, d.columns.tolist()); print(d.publisher.value_counts())"` — schema matches, dates sorted |
| 2. streams | `python src/streams.py --params params.yaml` | `data/streams/{L}_*_perms.npz` | `reports`/stdout: "valid=True" for the new streams |
| 3. fertility (Tier 1) | `... nice -n 15 python src/fertility.py --lang {L}` | `features/fertility/{L}_*`, `features/windows/{L}_*`, `reports/T3_{L}_report.md`, `T4_{L}_report.md` | open the report: raw fertility per tokenizer sane (>1.0), GATEs PASS |
| 4. classify (Tier 2) | `... python src/classifier.py --lang {L}` | `features/classifier/*` | `results/classifier_metrics.json` accuracy per year |
| 5. embed (Tier 2, GPU) | `... python src/embeddings.py --lang {L}` | `features/embeddings/*` | `results/embeddings_status.json` = "ok" |
| 6. detect (Tier 2) | `... python src/detect.py --lang {L} --params params.yaml` | `reports/T5_{L}..T9_{L}_report.md` + figs | read the STATUS blocks |

Always run the heavy steps thread-limited and niced (`OMP_NUM_THREADS=2 nice -n 15 python ...`)
so you don't freeze a shared machine. Every script also has `--demo` for a <60s smoke test
before the real run.

---

### 11.6 The Git + DVC collaborator loop (do this every session)

```bash
# --- start of a session: get the latest code AND data ---
git pull                      # latest code, params, dvc.lock, reports
dvc pull                      # latest data/interim, features, raw (bytes from Drive)

# --- do your work (edit loader/params, run the pipeline for your language) ---

# --- end of a session: publish code (Git) and data (DVC) separately ---
git add params.yaml src/prepare.py dvc.yaml reports/T2_en_report.md reports/T3_en_report.md   # code + small text + reports
dvc add data/raw/en_newssumm          # only if you added a new raw corpus
dvc repro                              # rebuild the DAG so dvc.lock records the new hashes
git add dvc.lock data/raw/en_newssumm.dvc
git commit -m "en: NewsSumm loader + panel + fertility"
dvc push -j 4                          # upload the produced bytes (interim/features) to Drive
git push                               # upload the code + pointers
```

Rules of thumb:
- **Code, `params.yaml`, `dvc.yaml`, `dvc.lock`, `reports/*.md`, `*.dvc` → Git.**
- **Big data (`data/interim`, `features/`, raw corpora) → DVC (`dvc push`), never Git.**
- **`git pull` then `dvc pull`** at the start; **`git push` and `dvc push`** at the end.
- Work on a **branch** if you want to be safe: `git checkout -b add-english`; open a Pull
  Request; the owner merges. Data still goes via `dvc push` (DVC is branch-agnostic —
  `dvc.lock` on your branch points at the bytes you pushed).
- If two people edit at once, resolve `dvc.lock` conflicts by re-running `dvc repro` after
  merging the code, then commit the regenerated `dvc.lock`.

---

### 11.7 All the copy-paste AI prompts in one place

Use these with an AI coding tool, one at a time, checking the diff after each. They are
scoped so the tool changes *only* data/config wiring and never the methodology.

1. **Generalize prepare** — the §11.1(a) prompt.
2. **Auto-discover streams** — the §11.1(b) prompt.
3. **Parameterize fertility** — the §11.1(c) prompt.
4. **Parameterize detect/classifier/embeddings** — the §11.1(d) prompt (Tier 2).
5. **Write the NewsSumm loader:** *"Add `read_newssumm(raw_dir, demo)` to `src/prepare.py`.
   Read every CSV under `raw_dir` with pandas; NewsSumm's real columns are `<paste the column
   names from reports/EN1_audit.md>`. Return a DataFrame renamed to News (article body),
   Category (topic), Heading (title), Date (publication date, any parseable format), Source
   (newspaper name). If `demo`, read only `nrows=200` per file. Do not clean or dedup here —
   `clean_pool` does that."*
6. **Write the MLSUM loader:** *"Add `read_mlsum(hf_config, demo)` to `src/prepare.py` that
   streams `reciTAL/mlsum` config `hf_config` with `datasets.load_dataset(..., streaming=True)`
   and returns a DataFrame with News=text, Category=topic, Heading=title, Date=date,
   Source=<outlet from url domain or a constant>. Stop after `data.max_docs_per_language` rows
   (or 3000 if `demo`). No cleaning here."*
7. **Add a language audit:** *"Copy `src/audit_potrika.py` to `src/audit_newssumm.py` and point
   `discover_files` at `data/raw/en_newssumm`; adapt the column mapping section to NewsSumm's
   columns; keep every check (date parse rate per source, per-source volume, word-count
   methods, dup rate). Output to `reports/EN1_audit.md`."*

---

### 11.8 Troubleshooting for the collaborator

| symptom | cause / fix |
|---|---|
| `dvc pull` says "no remote" or asks for auth | §3.5.3 — place the Drive credentials or run the browser OAuth once. |
| Llama tokenizer 401/gated error | §3.5.2 — `huggingface-cli login`, or remove Llama from `params.yaml` `tokenizers:`. |
| `prepare.py` errors "only --lang bn implemented" | You skipped §11.1(a) — do the one-time generalization first. |
| `fertility.py` report still shows Bangla numbers | You skipped §11.1(c) — the `"bn_panel"` literals still select bn. |
| `data/interim/{L}_panel.parquet` missing publishers / empty | Your loader's column mapping is wrong, or `panel_publishers` names don't match what the loader emits — check `reports/T2_{L}_report.md`'s cleaning ledger and publisher counts. |
| `dvc repro` re-runs everything | Expected the first time a new stage/param appears; afterwards it only re-runs what changed. |
| process killed (exit 137) | Memory/CPU contention — rerun with `OMP_NUM_THREADS=2 nice -n 15`; if prepare OOMs, that's the 2.8 GB text load — it still completes single-threaded. |

**Do-not-touch reminder:** if an AI tool proposes changing anything in the ⛔ column of §11.0
(signals, windowing, calibration, ADWIN, permutation test, figures), reject it — that would
change the method and break comparability with Bangla. Your job is data + config only.

---

## 12. The viva/demo cheatsheet

```bash
dvc dag                 # show the pipeline graph
dvc repro               # everything up to date → prints "cached", instant
dvc metrics show        # detection_metrics.json
git tag -l              # versions
dvc metrics diff v1 v2  # metric deltas between two tagged runs
less reports/T9_report.md
```
Every experiment also has a `--demo` flag (`python src/detect.py --demo`) that runs on a tiny
subsample in seconds — useful for a live walkthrough.
