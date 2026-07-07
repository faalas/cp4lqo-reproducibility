# Reproducibility Report: "Conformal Prediction for Verifiable Learned Query Optimization"

**Original Paper:** Conformal Prediction for Verifiable Learned Query Optimization, VLDB 2025  
**Authors:** Hanwen Liu, Shashank Giridhara, Ibrahim Sabek (University of Southern California)  
**Link to Paper:** [https://www.vldb.org/pvldb/vol18/p2653-liu.pdf](https://www.vldb.org/pvldb/vol18/p2653-liu.pdf)
**Link to Original Code:** [https://github.com/ihanwen99/Conformal-Prediction-for-Verifiable-Learned-Query-Optimization](https://github.com/ihanwen99/Conformal-Prediction-for-Verifiable-Learned-Query-Optimization)

**Report by:** Faisal Azmi Sirajuddin

---

## 1. Summary and Reproduction Goal

### Paper Summary

This paper introduces a Conformal Prediction (CP) framework for Learned Query Optimizers (LQOs) that provides **formal, probabilistic guarantees** on query execution latency. Given a user-specified risk level δ, the framework guarantees that the predicted latency upper bound U will be violated with probability at most δ. The framework is applied to three LQOs — Balsa, Lero, and RTOS — across two workloads (JOB and TPC-H), covering four components: latency bounds, runtime verification, adaptive CP, and CP-guided plan search.

### Reproduction Goal

This reproduction focuses on **Balsa on JOB** (§6.2 and §6.6), specifically:

- **§6.2 (Figure 5 & 6):** Coverage validation — empirically verifying that the unified and pattern-based upper bounds achieve ≥ 90% coverage (δ=0.1)
- **§6.6 (Figure 11–14):** CP-guided plan search — evaluating whether CP guidance improves plan quality and reduces planning time compared to vanilla Balsa

---

## 2. Environment Setup

### Hardware

- **CPU:** Intel Core i7 (20 cores)
- **GPU:** NVIDIA RTX 4050 (6GB VRAM)
- **RAM:** 32GB
- **OS:** Ubuntu 22.04 (WSL2 on Windows 11)

### Software

- **Python:** 3.7 (balsa_env via Conda)
- **PostgreSQL:** 12.5 (compiled from source)
- **CUDA:** 12.1
- **Key Dependencies:** PyTorch, Ray, wandb (offline), psycopg2

### Installation

```bash
# Step 1: Clone this fork
git clone https://github.com/faalas/cp4lqo-reproducibility.git
cd cp4lqo-reproducibility

# Step 2: Create conda environment
conda create -n balsa_env python=3.7 -y
conda activate balsa_env

# Step 3: Install dependencies
pip install -r requirements.txt
pip install -e .
pip install -e pg_executor

# Step 4: Install PostgreSQL 12.5 from source
wget https://ftp.postgresql.org/pub/source/v12.5/postgresql-12.5.tar.gz
tar xzvf postgresql-12.5.tar.gz
cd postgresql-12.5
./configure --prefix=$HOME/postgresql-builds/pg12 --without-readline
make -j$(nproc) && make install
export PATH=$HOME/postgresql-builds/pg12/bin:$PATH

# Step 5: Install pg_hint_plan (required for Balsa)
git clone https://github.com/ossc-db/pg_hint_plan.git -b REL12_1_3_7
cd pg_hint_plan
sed -i "s|PG_CONFIG = pg_config|PG_CONFIG = $HOME/postgresql-builds/pg12/bin/pg_config|" Makefile
make && make install

# Step 6: Setup PostgreSQL + IMDB dataset
initdb -D ~/imdb_data
cp conf/balsa-postgresql.conf ~/imdb_data/postgresql.conf
echo "shared_preload_libraries = 'pg_hint_plan'" >> ~/imdb_data/postgresql.conf
pg_ctl -D ~/imdb_data start -l ~/imdb_data/logfile
createdb imdbload

# Download and load IMDB dataset (~1GB, ~30-60 min)
mkdir -p ~/datasets/imdb && cd ~/datasets/imdb
wget https://event.cwi.nl/da/job/imdb.tgz && tar -xvzf imdb.tgz
cd ~/cp4lqo-reproducibility
mkdir -p datasets/imdb && cp ~/datasets/imdb/*.csv datasets/imdb/
python scripts/prepend_imdb_headers.py
bash load-postgres/load_job_postgres.sh datasets/imdb
psql imdbload -c "CREATE EXTENSION IF NOT EXISTS pg_hint_plan;"
```

### Quick Start (every new terminal session)

```bash
conda activate balsa_env
export WANDB_MODE=offline
pg_ctl -D ~/imdb_data start -l ~/imdb_data/logfile  # if not running
cd ~/cp4lqo-reproducibility
```

### Code Modifications

The following changes were made from the original repository:

1. **`balsa/optimizer.py`** — Added CP-guided beam search support:
   - Added `cp_hashmap` parameter to `Optimizer.__init__()` with fallback default values
   - Added `_beam_search_bk_cp()` — beam search variant that sorts candidates by `U = cost + C` (conformal upper bound) instead of raw cost
   - Modified `plan()` to switch between baseline and CP-guided search based on `cp_assist` flag
   - Added `[Calib]` pattern printing in `_beam_search_bk()` for calibration data collection

2. **`conf/balsa-postgresql.conf`** — Reduced memory settings to fit 32GB RAM machine (paper used a larger server):
   ```
   shared_buffers: 32GB → 128MB
   temp_buffers:   32GB → 8MB
   work_mem:       4GB  → 4MB
   ```

3. **`load-postgres/load_job_postgres.sh`** — Two fixes:
   - Added `dropdb --if-exists $DBNAME` before `createdb` to safely re-create the database
   - Fixed CSV path bug: changed `'\copy name from '$1/name.csv''` → `'\copy name from 'name.csv''` (original used absolute path after `pushd`, causing file-not-found errors)

4. **`experiments.py`** — Added `cp_hashmap_path` parameter:
   ```python
   p.Define('cp_hashmap_path', None, 'Path to JSON file containing calibrated cp_hashmap.')
   ```

5. **`run.py`** — Updated `Main()` with three switchable query split modes (uncomment as needed):
   ```python
   # MODE 1 — Training: uncomment → train_nodes = 33a
   # p.test_query_glob = ['10b.sql', '10c.sql', ...]  # b+c+d+e+f

   # MODE 2 — Eval/Test: active by default → test_nodes = 33b
   p.test_query_glob = ['1b.sql', ..., '33b.sql']

   # MODE 3 — Calibration: uncomment → calib_nodes = 47 c/d/e/f
   # CALIB_QUERIES_47 = ['10c.sql', ...]
   # p.test_query_glob = CALIB_QUERIES_47
   ```
   Also updated `agent_checkpoint` → `1-train-checkpoints/` and `cp_hashmap_path` → `2-calibration/calib/`.

---

## 3. Repository Structure

```
cp4lqo-reproducibility/  (fork of paper repo)
├── README.md
├── .gitignore
├── 2-calibration.py              ← collect R per-partial-plan, build cp_hashmap
├── 3-merge.py                    ← merge calib + test records → 80q pool
├── 3-repro-figure5-6.ipynb       ← reproduce Figure 5 & 6 (§6.2)
├── 4-evaluation.sh               ← automate 30 eval runs (3 ckpt × 2 mode × 5)
├── 5-repro-figure11-14.ipynb     ← reproduce Figure 11–14 (§6.6)
│
├── 1-train-checkpoints/
│   ├── checkpoint_{49,99,149}.pt
│   └── checkpoint_{49,99,149}_metadata.txt
│
├── 2-calibration/
│   ├── calib/
│   │   ├── cp_hashmap_{49,99,149}.json
│   │   └── cp_hashmap_{49,99,149}_raw.json
│   └── test/
│       └── raw_33b_{49,99,149}.json
│
├── 3-repro-figure5-6/
│   ├── raw_80q_{49,99,149}.json
│   ├── figure5_{all,49,99,149}.png
│   ├── figure6_{49,99,149}_natural.png
│   └── figure6_{49,99,149}_paperdefined.png
│
├── 4-evaluation/
│   ├── CP/
│   │   └── balsa{49,99,149}-True-{1..5}.log
│   └── NoCP/
│       └── balsa{49,99,149}-False-{1..5}.log
│
└── 5-repro-figure11-14/
    ├── figure11_12_{49,99,149}_{all,paper,paper_plus}.png
    └── figure13_14_{49,99,149}_{all,paper,paper_plus}.png
```

## 4. Step-by-Step Reproduction

### Step 1 — Train Balsa (checkpoint_49 → checkpoint_149)

- **Queries used:** 33 template-a queries (`query_glob = ['*.sql']`, train split)
- **Input:** `queries/join-order-benchmark/*.sql`, `1-train-checkpoints/checkpoint_49.pt`
- **Config** (edit `run.py` `Main()`):

```python
p.val_iters        = 100
p.agent_checkpoint = "1-train-checkpoints/checkpoint_49.pt"
p.eval_mode        = False
p.sim              = False
p.cp_guided        = False
```

```bash
WANDB_MODE=offline python -u run.py --local 2>&1 | tee train_49to149.log
```

- **Output:** `1-train-checkpoints/train_49to149.log` (+ checkpoint_99.pt, checkpoint_149.pt auto-saved)

---

### Step 2 — Calibration

- **Queries used:** 47 calibration queries (templates c/d/e/f)
- **Input:** `1-train-checkpoints/checkpoint_{49,99,149}.pt`, `queries/join-order-benchmark/*.sql`
- **Script:** `2-calibration.py`

```bash
WANDB_MODE=offline python 2-calibration.py
```

- **Output:**
```
2-calibration/calib/cp_hashmap_{49,99,149}.json
2-calibration/calib/cp_hashmap_{49,99,149}_raw.json
2-calibration/test/raw_33b_{49,99,149}.json
```

---

### Step 3 — Reproduce Figure 5 & 6 (§6.2)

- **Input:** `2-calibration/calib/cp_hashmap_{49,99,149}_raw.json`, `2-calibration/test/raw_33b_{49,99,149}.json`

```bash
# Step 3a — merge calibration + test records into 80-query pool
python 3-merge.py
# → 3-repro-figure5-6/raw_80q_{49,99,149}.json

# Step 3b — run notebook
jupyter notebook 3-repro-figure5-6.ipynb
```

- **Output:**
```
3-repro-figure5-6/
├── raw_80q_{49,99,149}.json
├── figure5_all.png                  ← 3 curves overlay (iter 50/100/150)
├── figure5_{49,99,149}.png          ← individual per checkpoint
├── figure6_{49,99,149}_natural.png
└── figure6_{49,99,149}_paperdefined.png
```

---

### Step 4 — Evaluation (§6.6)

- **Queries used:** 33 template-b queries
- **Input:** `1-train-checkpoints/checkpoint_{49,99,149}.pt`, `2-calibration/calib/cp_hashmap_{49,99,149}.json`

```bash
chmod +x 4-evaluation.sh
./4-evaluation.sh
```

- **Output:**
```
4-evaluation/CP/balsa{49,99,149}-True-{1..5}.log
4-evaluation/NoCP/balsa{49,99,149}-False-{1..5}.log
```

---

### Step 5 — Reproduce Figure 11–14 (§6.6)

- **Script:** `5-repro-figure11-14.ipynb`
- **Input:** `4-evaluation/CP/`, `4-evaluation/NoCP/`

```bash
jupyter notebook 5-repro-figure11-14.ipynb
```

- **Output:**
```
5-repro-figure11-14/
├── figure11_12_{49,99,149}_{all,paper,paper_plus}.png
└── figure13_14_{49,99,149}_{all,paper,paper_plus}.png
```

---

## 5. Results and Analysis

### §6.2 — Coverage Validation (Figure 5 & 6)

#### Figure 5 — Unified-based Upper Bound

| | Paper (Balsa, JOB) | Ours |
|---|---|---|
| Peak coverage | ~91% | **90.5%** |
| Target (1−δ) | 90% | 90% |
| Result | ✅ Valid | ✅ Valid |

Our reproduction achieves peak coverage of **90.5% ≥ 90%**, empirically validating Eq.4 (Lemma 1). The curve shape closely matches Figure 5(a) of the paper.

#### Figure 6 — Pattern-based Upper Bounds

**Top-3 popular patterns:** reproduced exactly — `(NL,NL,IS)`, `(NL,HJ,IS)`, `(HJ,NL,SS)` — same as paper ✅

**Least-3 popular patterns:** different from paper.

| | Paper | Ours |
|---|---|---|
| Least-3 | `(NL,NL,SS)`, `(HL,SS,SS)`, `(HL,SS,IS)` | `(HJ,HJ,IS)` K=11, `(HJ,NL,IS)` K=14, `(HJ,SS,IS)` K=24 |

**Root cause:** Balsa's beam search is non-deterministic. The plan selected for each query differs between runs, causing different operator pattern distributions. However, both datasets show the same conclusion: patterns with small K produce wider, noisier coverage distributions while still satisfying peak ≥ 90% — consistent with Lemma 1.

---

### §6.6 — CP-Guided Plan Search (Figure 11–14)

#### Plan Quality (Figure 11 & 12)

| Checkpoint | CP Better | Paper |
|---|---|---|
| iter 50  | **13/33 (39.4%)** | 11/33 (33.3%) |
| iter 100 | 7/33 (21.2%) | — |
| iter 150 | **16/33 (48.5%)** | — |

Best improvement: **15.76x** on query 27b at iter 50 (paper reports 9.84x).

**Anomaly at iter 100:** Severe degradation on 25b (0.03x) and 31b (0.02x) — caused by stale `cp_hashmap` calibrated at iter 50 applied to iter 100 model whose error distribution had shifted.

#### Planning Time Reduction (Figure 13 & 14)

| Checkpoint | Reduction |
|---|---|
| iter 50  | ~24% |
| iter 100 | ~18% |
| iter 150 | ~25% |

---

## 6. Key Challenges and Insights

- **`ray.init` double-init error:** Fixed by adding `ray.shutdown()` before each `BalsaAgent()` call in `2-calibration.py`.
- **Stale calibration (critical finding):** Applying `cp_hashmap` from iter 50 to iter 100 model causes breakdown because the model's cost prediction distribution shifts during training.
- **Non-determinism affects pattern distribution:** Balsa's beam search produces different plans across runs, causing Least-3 patterns to differ from paper — but conclusions remain consistent.

---

## 7. Conclusion

| Claim | Status | Notes |
|---|---|---|
| Coverage ≥ 1−δ = 90% (Figure 5) | ✅ Reproduced | Peak 90.5% |
| Top-3 patterns same as paper (Figure 6) | ✅ Reproduced | Exact match |
| Least-3 patterns same as paper (Figure 6) | ⚠️ Partial | Non-determinism |
| CP improves plan quality (Figure 11/12) | ✅ Reproduced | 13/33 vs paper 11/33 |
| CP reduces planning time (Figure 13/14) | ✅ Reproduced | ~24-25% reduction |

---

## Original README (Paper Authors)

> The following is the original README from the paper authors.

---

# Conformal Prediction for Verifiable Learned Query Optimization

This link is for the Additional Experiments mentioned in the revision comments: [Additional Experiments](https://anonymous.4open.science/r/Conformal-Prediction-4-Database-646E/Additional_Experiments_Conformal_Prediction_for_Query_Optimisation.pdf).

This repository contains the source code and commands used for the "Conformal Prediction for Verifiable Learned Query Optimization VLDB submission". It also includes the adapted source code of the main learned query optimizers used in our experiments, including Balsa, Lero, and RTOS.

### Balsa: CP Guided Plan Search

The pre-trained checkpoint is available in the `balsa/train-checkpoints` directory. To initiate the experiment, execute:

```sh
python run.py --local
```

By default, the experiment runs in the `without cp-guided` mode. To enable CP-guided mode, modify line 2274 by setting `p.cp_guided = True`.

Please refer to [Balsa Original Repo](https://github.com/balsa-project/balsa) to set up the whole environment or retrain Balsa.

### Lero

```sh
python lero/server.py
python lero/test_sript/train_model.py --query_folder train-query-path --test_query_folder test-query-path --algo lero --output_query_latency_file imdb.log --model_prefix imdb_model --topK 3
```

### RTOS

```sh
python CostTraining.py
python LatencyTuning.py
python Analysis/convertQueries.py
python Analysis/runAllQueries.py
```

### Scripts

The `scripts/` directory contains resources for data parsing and processing. A clear example is `RTOS/4-CP.ipynb`, where we maintain execution logs.
