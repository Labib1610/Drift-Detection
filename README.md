# Label-Free Concept Drift Detection via Tokenizer Fertility

A fully reproducible research pipeline that tests whether **tokenizer fertility** (subword
pieces per word) — a cheap, label-free, model-free signal computed from raw text — can detect
temporal concept drift in a news stream *earlier* than watching a model's error rate, at a
**matched false-alarm rate**, across five tokenizer families.

**Primary corpus:** Potrika (665K Bangla news articles, 2016–2020, six newspapers).
**Pipeline:** raw data → cleaned chronological "panel" → fertility/novelty signals →
FAR-calibrated ADWIN detection → supervised (S6) & embedding (S5) baselines → validity tests.

### The honest headline result
On real Bangla news, **no label-free signal — nor a supervised error monitor (S6), nor an MMD
embedding baseline (S5) — aligns its alarms with real events better than chance** (permutation
test, T7). The reason is a quantified **sensitivity floor**: real news events change only a few
percent of the stream (only a pandemic-scale event is lexically large), far below the ~25–50%
these detectors need to fire at a deployable false-alarm rate (T8–T9). In *controlled
synthetic* drift the novelty-weighted signals (S4, S7) do work. This is a clean,
mechanism-backed **negative-with-methodology** — the contribution is the matched-FAR evaluation
+ the dilution mechanism + the sensitivity-floor quantification.

---

## Documentation map

| read this | for |
|---|---|
| **[reports/PROJECT_WORKFLOW.md](reports/PROJECT_WORKFLOW.md)** | the science from scratch — every term, every signal (S1–S9, S5, S6) with worked examples, and all nine experiments |
| **[reports/PROJECT_HANDBOOK.md](reports/PROJECT_HANDBOOK.md)** | engineering reference — setup, auth, DVC/Git, file-by-file & function-by-function, and the collaborator playbook for adding a language |
| **[reports/RESULTS_ANALYSIS.md](reports/RESULTS_ANALYSIS.md)** | what every result file and every key means, with real examples |
| **[reports/ENGLISH_CLASSIFICATION_ANOMALY.md](reports/ENGLISH_CLASSIFICATION_ANOMALY.md)** | why the English classification task behaves differently, and how to diagnose it |
| `reports/T1_report.md … T9_report.md` | the per-experiment reports (each ends with a PASS/FAIL STATUS block) |

---

## Quickstart (reproduce everything)

```bash
# 1. clone
git clone <your-repo-url> Drift-Detection && cd Drift-Detection

# 2. environment  (Python 3.12)
sudo apt-get install libicu-dev pkg-config          # system dep for PyICU (Debian/Ubuntu)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional GPU baseline (Part 4 only):
#   pip install torch --index-url https://download.pytorch.org/whl/cu128
#   pip install "sentence-transformers==6.0.1"

# 3. credentials (once) — see "Auth" below
huggingface-cli login          # for the gated Llama tokenizer (or drop it from params.yaml)
#   + place Google Drive creds for DVC (see Auth)

# 4. get the data 
dvc pull

# 5. run the whole pipeline (only re-runs what changed)
dvc repro

# 6. read the result
less reports/T9_report.md
```

Be courteous on a shared machine — run heavy steps thread-limited and niced:
`OMP_NUM_THREADS=2 nice -n 15 python src/<script>.py ...`.

---

## Auth (do this once, right after cloning)

Three credentials; **none live in Git** (they're secrets). Full detail in the handbook §3.5.

1. **GitHub** — the owner adds you as a collaborator; authenticate pushes with a Personal
   Access Token (HTTPS) or an SSH key. `git config user.name / user.email`.
2. **HuggingFace token** (only for the gated `meta-llama/Llama-3.2-1B` tokenizer) —
   `huggingface-cli login` (writes `~/.cache/huggingface/token`), or `export HF_TOKEN=hf_...`,
   or simply delete the Llama line from `params.yaml → tokenizers:`.
3. **Google Drive for DVC** — either place the owner's shared
   `.dvc/tmp/gdrive-user-credentials.json`, or let the first `dvc pull` open a browser OAuth
   (your Google account must have Editor access to the shared folder), or use a service account
   (`dvc remote modify --local gdrive gdrive_service_account_json_file_path /path/sa.json`).

---

## DVC — the full data/version-control setup

**Model:** Git tracks *code + small text* (scripts, `params.yaml`, `dvc.yaml`, `dvc.lock`,
`reports/`, and tiny `.dvc` pointers). DVC tracks *big bytes* (the 4.3 GB corpus, interim
parquet, feature matrices), stores them content-addressed in `.dvc/cache`, and pushes/pulls
them to a **Google Drive** remote.

### It's already initialised in this repo
```
.dvc/config                     # remote definition (below)
data/raw/bn_potrika.dvc         # 3-line MD5 pointer to the 4.3 GB corpus (real bytes in cache/Drive)
dvc.yaml / dvc.lock             # the pipeline graph + locked input/output hashes
```

### The remote (already configured)
```ini
# .dvc/config
[core]
    remote = gdrive
['remote "gdrive"']
    url = gdrive://1mOcVF9s3TP7bygDzYlIswoTpnIUJL22I
```

### If you ever set this up from scratch (reference)
```bash
git init && dvc init
dvc remote add -d gdrive gdrive://<your-folder-id>     # default Google Drive remote
dvc add data/raw/bn_potrika                            # hash the corpus into cache + write the .dvc pointer
git add data/raw/bn_potrika.dvc data/raw/.gitignore .dvc/config
git commit -m "data: potrika raw corpus (DVC-tracked)"
dvc push -j 4                                          # upload bytes to Drive
```

### Everyday DVC commands
```bash
dvc pull            # download data/interim, features, raw from Drive
dvc push -j 4       # upload produced bytes
dvc repro           # rebuild only the stages whose deps/params changed

dvc dag             # show the pipeline graph
dvc metrics show    # print results/detection_metrics.json
dvc checkout        # materialise files from the LOCAL cache (no network)
dvc status          # what's out of date
```

### The pipeline graph (`dvc.yaml`)
```
prepare  ─┐                                     data/raw/bn_potrika ─▶ data/interim
          ├─▶ streams   ─▶ data/streams
          └─▶ fertility ─▶ features/fertility, features/windows
                              │
            detect ──────────┘  ─▶ results/calibration.json, detection_metrics.json + reports/T5..T9
            classify ─▶ features/classifier         (S6 supervised baseline)
            embed    ─▶ features/embeddings          (S5 MMD baseline, GPU)
```
> **Note:** `detect` reads `features/classifier/` and `features/embeddings/` at runtime but
> they are not declared as its deps, so after re-running `classify`/`embed` do
> `dvc repro -f detect` to fold S5/S6 into the reports.

---

## Repository layout

```
Drift-Detection/
├── README.md               ← you are here
├── requirements.txt        ← pinned deps (Python 3.12)
├── params.yaml             ← ALL configuration (no magic numbers in code)
├── dvc.yaml / dvc.lock     ← pipeline graph + locked hashes
├── .dvc/                   ← DVC config (remote) + local cache
├── data/
│   ├── raw/bn_potrika/     ← DVC-tracked corpus (RawDataset = usable; BalancedDataset = unused)
│   ├── raw/en_newssumm/    ← (empty) put NewsSumm here for the English branch
│   ├── interim/            ← prepare.py output: bn_panel.parquet, bn_full.parquet
│   └── streams/            ← streams.py output: permutation .npz files
├── features/               ← fertility/, windows/, type_fertility/, classifier/, embeddings/
├── results/                ← calibration.json, detection_metrics.json, classifier_*/embeddings_* json
├── reports/                ← T1..T9 reports, figs/, and the 4 docs above
├── src/                    ← the 8 pipeline scripts (below)
└── md_files/               ← the task briefs (spec + T1..T9), gitignored by default
```

### The eight scripts (`src/`)
| script | stage | does |
|---|---|---|
| `audit_potrika.py` | T1 | audits the raw corpus (dates, coverage, dedup, word counts) |
| `prepare.py` | T2 | cleans + builds `bn_panel` (publisher-balanced) and `bn_full` |
| `streams.py` | T3 | builds 11 index permutations (1 real + 10 shuffled nulls) |
| `fertility.py` | T3–T4 | tokenizes, windows, computes signals S1–S9, z-scores them |
| `detect.py` | T5–T9 | FAR calibration, ADWIN detection, redundancy/validity/floor analyses |
| `classifier.py` | T6 | supervised reference S6 (River logistic, publisher/topic) |
| `embeddings.py` | T6 | MMD baseline S5 (LaBSE embeddings, GPU) |
| `probe_ccnews.py` | T3b | streams CC-News to size hi/ar/uk volume (counts only) |

---

## Key results at a glance (Bangla panel)

- **Data:** 664,884 raw rows, dates parse 100%; cleaned pool 649,691; `bn_panel` = 99,900 docs
  at exactly 33.3% per publisher every year.
- **Raw mean fertility:** mBERT 2.72, XLM-R 2.04, BLOOM 1.68, Qwen 6.99, Llama 7.71.
- **Plain fertility (S1) is flat**; novelty (S4, +0.63) and novel-token mass (S7, +0.75–0.82)
  rise — explained exactly by **dilution** (only ~6% of tokens are novel).
- **Supervised S6 degrades monotonically:** publisher accuracy 0.859 → 0.789 → 0.718 → 0.670 →
  0.668 (2016→2020).
- **Validity (T7):** no signal — including S5 and S6 — passes the event-detection permutation
  test (all p > 0.05).
- **Sensitivity floor (T8–T9):** real events touch ~a few % of the stream above baseline (only
  COVID is large); detectors need ~25–50%.

---

## Adding a language (English / Turkish / other)

The pipeline is language-generic after `prepare.py`. Adding a language is **data + config only,
no new methodology**: write a small loader, add a `{lang}:` block to `params.yaml`, and do a
one-time `--lang` generalization of the stream names. **Full step-by-step, function-level
instructions, and copy-paste AI-tool prompts are in
[reports/PROJECT_HANDBOOK.md §11](reports/PROJECT_HANDBOOK.md).** English uses NewsSumm
(Zenodo 17670865); Turkish uses MLSUM `reciTAL/mlsum` config `tu` (probe single-outlet
dominance first).

---

## License

See [LICENSE](LICENSE).
