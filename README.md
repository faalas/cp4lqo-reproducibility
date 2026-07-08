# Reproducibility Report: "Conformal Prediction for Verifiable Learned Query Optimization"

**Original Paper:** Conformal Prediction for Verifiable Learned Query Optimization, VLDB 2025  
**Authors:** Hanwen Liu, Shashank Giridhara, Ibrahim Sabek (University of Southern California)

**Link to Paper:** [https://www.vldb.org/pvldb/vol18/p2653-liu.pdf](https://www.vldb.org/pvldb/vol18/p2653-liu.pdf)

**Link to Original Code:** [https://github.com/ihanwen99/Conformal-Prediction-for-Verifiable-Learned-Query-Optimization](https://github.com/ihanwen99/Conformal-Prediction-for-Verifiable-Learned-Query-Optimization).

**Report by:** Faisal Azmi Sirajuddin

---

## 1. Summary and Reproducibility Goal

### Paper Summary

This paper introduces a Conformal Prediction (CP) framework for Learned Query Optimizers (LQOs) that provides **formal, probabilistic guarantees** on query execution latency. Given a user-specified risk level δ, the framework guarantees that the predicted latency upper bound U will be violated with probability at most δ. The framework is applied to three LQOs — Balsa, Lero, and RTOS — across two workloads (JOB and TPC-H), covering four components: latency bounds, runtime verification, adaptive CP, and CP-guided plan search.

### Reproducibility Goal

This reproducibility study focuses on **Balsa on JOB** (§6.2 and §6.6), specifically:

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
cd cp4lqo-reproducibility/Balsa

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
cd ~/cp4lqo-reproducibility/Balsa
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
cd ~/cp4lqo-reproducibility/Balsa
```

### Code Modifications

The following files were modified from the original repository. A step-by-step reproduction guide for Figure 5/6 and Figure 11–14 is provided in Section 4.

1. **`balsa/optimizer.py`** — Added CP-guided beam search support:
   - Added `cp_hashmap` parameter to `Optimizer.__init__()` with fallback default values
   - Added `_beam_search_bk_cp()` — beam search variant that sorts candidates by `U = cost + C` (conformal upper bound) instead of raw cost
   - Modified `plan()` to switch between baseline and CP-guided search based on `cp_assist` flag
   - Added `[Calib]` pattern printing in `_beam_search_bk()` for calibration data collection

2. **`conf/balsa-postgresql.conf`** — Reduced memory settings to fit 32GB RAM machine (paper used a larger server):
   ```
   shared_buffers: 32GB → 128MB
   temp_buffers:   32GB → 8MB
   work_mem:        4GB → 4MB
   ```

3. **`load-postgres/load_job_postgres.sh`** — Two fixes for correct IMDB loading:
   - Added `dropdb --if-exists $DBNAME` before `createdb` to safely re-create the database
   - Fixed CSV path bug: `'\copy name from '$1/name.csv''` → `'\copy name from 'name.csv''` (original used absolute path after `pushd`, causing file-not-found errors)

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
cp4lqo-reproducibility/
├── .gitignore
├── README.md
└── Balsa/                             ← fork of paper repo (Balsa only)
    ├── 2-calibration.py               ← collect R per-partial-plan, build cp_hashmap
    ├── 3-merge.py                     ← merge calib + test records → 80q pool
    ├── 3-repro-figure5-6.ipynb        ← reproduce Figure 5 & 6 (§6.2)
    ├── 4-evaluation.sh                ← automate 30 eval runs (3 ckpt × 2 mode × 5)
    ├── 5-repro-figure11-14.ipynb      ← reproduce Figure 11–14 (§6.6)
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

---

## 4. Step-by-Step Reproduction

### Step 1 — Train Balsa (checkpoint_49 → checkpoint_149)

- **Queries used:** 33 template-a queries
- **Input:** `queries/join-order-benchmark/*.sql`, `1-train-checkpoints/checkpoint_49.pt`
- **Config** (in `run.py` `Main()`, uncomment MODE 1):

```python
p.val_iters        = 100
p.agent_checkpoint = "1-train-checkpoints/checkpoint_49.pt"
p.eval_mode        = False
p.sim              = False
p.cp_guided        = False
```

```bash
WANDB_MODE=offline python -u run.py --local 2>&1 | tee 1-train-checkpoints/train_49to149.log
```

- **Output:** `1-train-checkpoints/train_49to149.log` (+ `checkpoint_99.pt`, `checkpoint_149.pt` auto-saved)

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

# Step 3b — run notebook
jupyter notebook 3-repro-figure5-6.ipynb
```

- **Output:**
```
3-repro-figure5-6/
├── raw_80q_{49,99,149}.json
├── figure5_all.png
├── figure5_{49,99,149}.png
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

All generated figures can be found in the `3-repro-figure5-6/` and `5-repro-figure11-14/` directories. This analysis focuses on the results that most closely align with the core claims of the paper — specifically §6.2 (coverage validation) and §6.6 (CP-guided plan search) — which represent the primary contributions of the CP framework applied to Balsa on JOB.

---

#### Comparison for Figure 5: Coverage distribution of unified-based upper bounds (§6.2)

**Paper's Result:**
<div align="center">
  <img height="300" src="https://github.com/user-attachments/assets/a682a724-b7b0-4798-a753-c0471f0f6a50" />
</div>

**Reproducibility Result:**
<div align="center">
  <img height="300" src="https://github.com/user-attachments/assets/716518e5-83f3-4912-9d68-24d754d560a0" />
</div>

**Analysis:**

Our reproducibility focuses on Balsa only (one of three LQOs in the paper). The reproduction achieves a peak coverage of **90.5%**, which satisfies the target of 1−δ = 90% (δ=0.1), empirically validating Lemma 1 (Eq. 4). The curve shape — asymmetric, rising gradually from ~65%, peaking near 90%, then decreasing — closely matches the Balsa curve in Figure 5(a) of the paper. The small difference (90.5% vs. ~91% in the paper) is attributable to non-determinism in Balsa's beam search, which produces slightly different plan selections and R values across runs.

---

#### Comparison for Figure 6: Coverage distribution of pattern-based upper bounds (§6.2)

**Paper's Result:**
<div align="center">
  <img height="300" src="https://github.com/user-attachments/assets/afa143b8-ad84-40a5-ad94-661af5a6ab26" />
</div>

**Reproducibility Result:**
<div align="center">
  <img height="300" src="https://github.com/user-attachments/assets/6a363310-5905-4d84-82f5-37ecb7e8c81f" />
  <img height="300" src="https://github.com/user-attachments/assets/88245159-90f8-41c9-80b0-859a4d56ceb2" />
  <img height="300" src="https://github.com/user-attachments/assets/58433ffc-2e8d-4866-8a72-1cafb02141d9" />
</div>

**Analysis:**

The **Top-3 popular patterns** are reproduced exactly — `(NL,NL,IS)`, `(NL,HJ,IS)`, `(HJ,NL,SS)` — matching the paper's Figure 6(a). The coverage curves for all three patterns peak above 90%, consistent with the paper's claim that high-K patterns produce tight, well-calibrated upper bounds.

The **Least-3 popular patterns** differ from the paper. The paper reports `(NL,NL,SS)`, `(HL,SS,SS)`, `(HL,SS,IS)` as least popular, while our data identifies `(HJ,HJ,IS)` K=11, `(HJ,NL,IS)` K=14, `(HJ,SS,IS)` K=24. This discrepancy arises from Balsa's non-deterministic beam search — the plan chosen for each query varies across runs, causing the operator pattern distribution to differ from the paper's. Patterns that are rare in the paper's model appear frequently in ours (e.g., `(NL,NL,SS)` K=47, `(HJ,SS,SS)` K=58), so different patterns end up as "least popular."

Despite the difference in which patterns appear, the key observation is consistent with the paper: patterns with small K (low sample count) produce **wider, noisier, and more spread-out coverage distributions** — visible in panel (b) where the curves are multimodal and extend from 30% to 105%. This confirms Lemma 1's requirement that K ≥ K* = 9 is necessary for stable coverage guarantees.

---

#### Comparison for Figure 11 & 12: Plan quality — CP-guided vs. Balsa (§6.6)

**Paper's Result:**
<div align="center">
  <img height="300" src="https://github.com/user-attachments/assets/a29d035d-f1b1-400b-a4f3-f596e494b5f1" />
</div>

**Reproducibility Result:**
<div align="center">
  <img height="300" src="https://github.com/user-attachments/assets/a492392f-239f-42f1-b195-2f5c0168a627" />
</div>

**Analysis:**

The paper's Figure 11 and 12 shows CP-guided search consistently selecting lower-latency plans than vanilla Balsa for a subset of queries at 50 iterations. Our reproduction confirms this trend across all three checkpoints, with each checkpoint using its own calibrated `cp_hashmap` (computed from the corresponding model checkpoint via `2-calibration.py`).

At **50 iterations**, CP-guided search improves **20/33 queries** (paper: 11/33), with improvement ratios ranging from **1.12x to 14.7x**.

At **100 iterations**, improvement is observed in **12/33 queries** (1.10x–5.53x), with a notable exception on query **6b** which achieves **12.53x** — the highest at this checkpoint.

At **150 iterations**, CP-guided search improves **21/33 queries** (1.11x–3.76x), with query **6b** again showing significant improvement at **10.85x**. Excluding the 6b outlier, the improvement ceiling shrinks steadily across checkpoints — **14.7x → 5.53x → 3.76x** — consistent with the paper's claim that improvements are greatest in early training and the scope for improvement narrows as Balsa becomes more refined.

The differences between our reproducibility results and the paper are attributable to:
- **(1)** We recompute C from scratch using `2-calibration.py` rather than using the paper's hardcoded values, producing different per-pattern bounds
- **(2)** Balsa's non-deterministic beam search produces different plan selections across runs, meaning the NoCP baseline itself varies between our runs and the paper's

---

#### Comparison for Figure 13 & 14: Planning time — CP-guided vs. Balsa (§6.6)

**Paper's Result:**
<div align="center">
  <img height="300" src="https://github.com/user-attachments/assets/3aa5b65a-16d3-492c-bc5a-948743dc138c" />
  <img height="300" src="https://github.com/user-attachments/assets/1d49fc8c-dba2-4620-9422-2a2d0f925158" />
</div>

**Reproducibility Result:**
<div align="center">
  <img height="300" src="https://github.com/user-attachments/assets/b233d777-e662-4cba-a542-cbe24a8d131d" />
  <img height="300" src="https://github.com/user-attachments/assets/9d5d0766-b586-4bc2-8002-3797c102bf7c" />
</div>

**Analysis:**

The paper's Figure 13 and 14 show CP-guided search reducing planning time relative to vanilla Balsa for a subset of queries. Our reproduction confirms this trend across all three checkpoints, using the full 33-query test set.

At **50 iterations**, CP-guided search reduces planning time on **22/33 queries**, with an overall reduction of **14.18%**, and the largest improvement on query **5b (3.50x)**.

At **100 iterations**, planning time improves for **17/33 queries**, an overall reduction of only **10.78%** — the weakest checkpoint.

At **150 iterations**, CP-guided search improves **23/33 queries**, achieving the best overall reduction of **28.68%**, with query **19b** showing the largest improvement at **7.03x**. This does not follow the paper's claim that planning time reduction narrows as the model converges; instead, the dip at 100 iterations followed by recovery at 150 iterations suggests the checkpoint-to-checkpoint variation here is dominated by the factors below rather than a training-stage effect.

The differences between our reproducibility results and the paper are attributable to:
- **(1)** We recompute C from scratch using `2-calibration.py` rather than using the paper's hardcoded values, producing different per-pattern bounds
- **(2)** Balsa's non-deterministic beam search produces different plan selections across runs, meaning the NoCP baseline itself varies between our runs and the paper's
- **(3)** checkpoint_49 was provided directly in the original repo, while checkpoint_99 and checkpoint_149 were obtained by continuing training ourselves — a different training environment than the paper's original run, which may account for checkpoint_49 tracking the paper's results more closely than the later two checkpoints

---

## 6. Key Challenges and Insights

### Technical Challenges

- **Absent Calibration Pipeline:** The most significant challenge was that the original repository provides no calibration scripts — C values are hardcoded directly into `optimizer.py`. Reproducing §6.2 required us to reverse-engineer the calibration process from the paper's description and implement `2-calibration.py`, `3-merge.py`, and `3-repro-figure5-6.ipynb` from scratch, without ground truth to validate against.

- **Reconstructing the Reproduction Pipeline:** Closely related to the above, multiple scripts had to be designed to closely mimic the paper's experimental setup without knowing the exact implementation details — including how R records were collected per partial-plan, how the 80-query pool was constructed, and how coverage was measured at the query level.

- **Memory Configuration:** The original `balsa-postgresql.conf` was configured for a large server (32GB `shared_buffers`, 32GB `temp_buffers`, 4GB `work_mem`). Running on a 32GB RAM machine required reducing these settings significantly to prevent PostgreSQL from failing to start.

- **Interrupted Training:** Training had to be resumed twice from intermediate checkpoints due to session interruptions, requiring careful log merging and checkpoint management to ensure continuity.

- **Dependency Conflicts:** Setting up the Python 3.7 environment for Balsa required careful version management — several packages had breaking changes across versions, and combining Ray, PyTorch, psycopg2, and wandb in the same environment required iterative resolution of conflicts.

### Insights Gained

- **Model Performance is Training-Sensitive:** Balsa's performance is highly dependent on its training trajectory. Checkpoints produced by our training run differ from the paper's original checkpoint, leading to different plan selections, different pattern distributions, and different CP effectiveness — even when using the same query set and configuration.

- **Non-Determinism Limits Exact Reproducibility:** Balsa's beam search is non-deterministic, meaning plan selections vary across runs. This affects both the calibration data (R values) and the NoCP baseline, making exact numerical reproduction of the paper's results fundamentally difficult without fixing the random seed at the Balsa level.

- **Calibration Freshness Matters:** Computing C per-checkpoint (rather than using a single hardcoded value) is critical for CP to function correctly. A misaligned C — either hardcoded or computed from a different checkpoint — degrades CP guidance and can produce worse results than the NoCP baseline on individual queries.

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
