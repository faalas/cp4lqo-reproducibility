#!/bin/bash
# 4-evaluation.sh
# ================
# Automates evaluation runs for all three checkpoints (49, 99, 149),
# both with and without CP guidance (5 runs each = 30 runs total).
#
# For each run, patches run.py in-place before executing, then restores it.
#
# Output:
#   4-evaluation/CP/balsa{49,99,149}-True-{1..5}.log
#   4-evaluation/NoCP/balsa{49,99,149}-False-{1..5}.log
#
# Usage (run from the Balsa/ root directory):
#   chmod +x 4-evaluation.sh
#   ./4-evaluation.sh

set -e

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINTS=(49 99 149)
N_RUNS=5
CKPT_DIR="./1-train-checkpoints"
HASHMAP_DIR="./2-calibration/calib"
OUT_DIR="./4-evaluation"
RUN_PY="run.py"
RUN_PY_BAK="run.py.bak"

mkdir -p "${OUT_DIR}/CP"
mkdir -p "${OUT_DIR}/NoCP"

# ── Patch helpers ─────────────────────────────────────────────────────────────

# Save original run.py before any modification
backup_run_py() {
    cp "${RUN_PY}" "${RUN_PY_BAK}"
}

# Restore run.py from backup
restore_run_py() {
    cp "${RUN_PY_BAK}" "${RUN_PY}"
}

# Patch run.py for a specific checkpoint + CP mode
patch_run_py() {
    local ckpt=$1       # e.g. 49
    local cp_guided=$2  # True or False

    cp "${RUN_PY_BAK}" "${RUN_PY}"   # start clean from backup each time

    sed -i \
        "s|p\.agent_checkpoint = .*|p.agent_checkpoint = \"${CKPT_DIR}/checkpoint_${ckpt}.pt\"|" \
        "${RUN_PY}"

    sed -i \
        "s|p\.eval_mode = .*|p.eval_mode = True|" \
        "${RUN_PY}"

    sed -i \
        "s|p\.cp_guided = .*|p.cp_guided = ${cp_guided}|" \
        "${RUN_PY}"

    if [ "${cp_guided}" = "True" ]; then
        # CP-guided: use the calibrated hashmap for this checkpoint
        sed -i \
            "s|p\.cp_hashmap_path = .*|p.cp_hashmap_path = \"${HASHMAP_DIR}/cp_hashmap_${ckpt}.json\"|" \
            "${RUN_PY}"
    else
        # NoCP baseline: explicitly set to None (hashmap is not used)
        sed -i \
            "s|p\.cp_hashmap_path = .*|p.cp_hashmap_path = None|" \
            "${RUN_PY}"
    fi
}

# ── Trap: always restore run.py on exit / error ───────────────────────────────
trap restore_run_py EXIT

# ── Main loop ─────────────────────────────────────────────────────────────────
backup_run_py
echo "Backed up ${RUN_PY} → ${RUN_PY_BAK}"
echo ""

for ckpt in "${CHECKPOINTS[@]}"; do

    # ── CP-guided ─────────────────────────────────────────────────────────────
    echo "============================================================"
    echo "  checkpoint_${ckpt}  |  CP-guided = True"
    echo "============================================================"
    patch_run_py "${ckpt}" "True"

    for i in $(seq 1 ${N_RUNS}); do
        LOG="${OUT_DIR}/CP/balsa${ckpt}-True-${i}.log"
        echo "  Run ${i}/${N_RUNS} → ${LOG}"
        WANDB_MODE=offline python "${RUN_PY}" --local \
            2>&1 | tee "${LOG}"
        echo "  ✓ Done run ${i}"
    done
    echo ""

    # ── No CP (baseline) ──────────────────────────────────────────────────────
    echo "============================================================"
    echo "  checkpoint_${ckpt}  |  CP-guided = False (NoCP baseline)"
    echo "============================================================"
    patch_run_py "${ckpt}" "False"

    for i in $(seq 1 ${N_RUNS}); do
        LOG="${OUT_DIR}/NoCP/balsa${ckpt}-False-${i}.log"
        echo "  Run ${i}/${N_RUNS} → ${LOG}"
        WANDB_MODE=offline python "${RUN_PY}" --local \
            2>&1 | tee "${LOG}"
        echo "  ✓ Done run ${i}"
    done
    echo ""

done

echo "============================================================"
echo "  All 30 runs complete."
echo "  CP   logs : ${OUT_DIR}/CP/"
echo "  NoCP logs : ${OUT_DIR}/NoCP/"
echo "============================================================"
