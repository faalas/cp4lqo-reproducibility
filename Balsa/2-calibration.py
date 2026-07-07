"""
2-calibration.py
================
Collect per-partial-plan non-conformity scores (R = |predicted - actual|)
from two query sets for each Balsa checkpoint:

  1. Calibration queries (47 queries, templates c/d/e/f)
     → 2-calibration/calib/cp_hashmap_{49,99,149}.json
     → 2-calibration/calib/cp_hashmap_{49,99,149}_raw.json

  2. Test queries (33 queries, template b)
     → 2-calibration/test/raw_33b_{49,99,149}.json

The two raw outputs are later merged in step 3 (Figure 5/6 reproduction)
to form an 80-query pool.

Usage (run from the Balsa/ root directory):
    python 2-calibration.py

Requirements:
    - PostgreSQL running with the IMDB database loaded
    - Checkpoints present at ./1-train-checkpoints/checkpoint_{49,99,149}.pt
    - Balsa package installed (conda activate balsa_env)
"""

import json
import os
import ray
import balsa
from balsa.util import postgres
from run import BalsaAgent

# ── Query lists ───────────────────────────────────────────────────────────────

CALIB_QUERIES = [
    "10c.sql", "11c.sql", "11d.sql", "12c.sql", "13c.sql", "13d.sql",
    "14c.sql", "15c.sql", "15d.sql", "16c.sql", "16d.sql", "17c.sql",
    "17d.sql", "17e.sql", "17f.sql", "18c.sql", "19c.sql", "19d.sql",
    "1c.sql",  "1d.sql",  "20c.sql", "21c.sql", "22c.sql", "22d.sql",
    "23c.sql", "25c.sql", "26c.sql", "27c.sql", "28c.sql", "29c.sql",
    "2c.sql",  "2d.sql",  "30c.sql", "31c.sql", "33c.sql", "3c.sql",
    "4c.sql",  "5c.sql",  "6c.sql",  "6d.sql",  "6e.sql",  "6f.sql",
    "7c.sql",  "8c.sql",  "8d.sql",  "9c.sql",  "9d.sql",
]

TEST_QUERIES = [
    "1b.sql",  "2b.sql",  "3b.sql",  "4b.sql",  "5b.sql",  "6b.sql",
    "7b.sql",  "8b.sql",  "9b.sql",  "10b.sql", "11b.sql", "12b.sql",
    "13b.sql", "14b.sql", "15b.sql", "16b.sql", "17b.sql", "18b.sql",
    "19b.sql", "20b.sql", "21b.sql", "22b.sql", "23b.sql", "24b.sql",
    "25b.sql", "26b.sql", "27b.sql", "28b.sql", "29b.sql", "30b.sql",
    "31b.sql", "32b.sql", "33b.sql",
]

CHECKPOINTS = [49, 99, 149]

CHECKPOINT_DIR = "./1-train-checkpoints"
OUTPUT_CALIB   = "./2-calibration/calib"
OUTPUT_TEST    = "./2-calibration/test"

DELTA = 0.1

# ── Operator shorthand ────────────────────────────────────────────────────────

SHORT = {
    "Nested Loop":      "NL",
    "Hash Join":        "HJ",
    "Merge Join":       "MJ",
    "Seq Scan":         "SS",
    "Index Scan":       "IS",
    "Index Only Scan":  "IOS",
}

def short_op(op):
    return SHORT.get(op, op)

# ── Pattern extraction ────────────────────────────────────────────────────────

def extract_pattern(node):
    """Return (parent, left-child, right-child) operator pattern, or None."""
    if len(node.children) != 2:
        return None
    return "({},{},{})".format(
        short_op(node.node_type),
        short_op(node.children[0].node_type),
        short_op(node.children[1].node_type),
    )

# ── Recursive R collection ────────────────────────────────────────────────────

def collect_R(pred_node, exec_node, records, qname):
    """
    Walk the predicted plan tree and the executed plan tree in parallel,
    collecting R = |predicted_cost - actual_latency| for every internal node
    that has exactly two children.
    """
    if len(pred_node.children) == 2 and len(exec_node.children) == 2:
        pattern   = extract_pattern(pred_node)
        predicted = pred_node.cost
        actual    = exec_node.actual_time_ms
        if pattern is not None and predicted is not None and actual is not None:
            records.append({
                "qname":     qname,
                "pattern":   pattern,
                "predicted": predicted,
                "actual":    actual,
                "R":         abs(actual - predicted),
            })

    if len(pred_node.children) == len(exec_node.children):
        for pc, ec in zip(pred_node.children, exec_node.children):
            collect_R(pc, ec, records, qname)
    else:
        print(
            f"  [WARNING] {qname}: plan tree structure mismatch "
            f"(pred={len(pred_node.children)} children, "
            f"exec={len(exec_node.children)} children) "
            f"at pred={pred_node.node_type}, exec={exec_node.node_type}"
        )

# ── Run planning + execution for a query list ─────────────────────────────────

def run_collection(checkpoint_path, query_glob, label):
    """
    Load the Balsa model from checkpoint_path, plan each query in query_glob,
    execute the chosen plan via EXPLAIN ANALYZE, and return all R records.

    Args:
        checkpoint_path : path to the .pt checkpoint file
        query_glob      : list of .sql filenames to process
        label           : short label for progress output ("calib" or "test")

    Returns:
        list of dicts with keys: qname, pattern, predicted, actual, R
    """
    p = balsa.params_registry.Get("Balsa_JOBRandSplit")
    p.use_local_execution = True
    p.sim_checkpoint      = None
    p.epochs              = 1
    p.val_iters           = 1
    p.query_glob          = ["*.sql"]
    p.test_query_glob     = query_glob
    p.agent_checkpoint    = checkpoint_path
    p.eval_mode           = True
    p.cp_guided           = False
    p.sim                 = False

    ray.shutdown()  # ensure no stale Ray instance before init
    agent = BalsaAgent(p)
    agent.curr_value_iter = 0
    model, dataset = agent.eval_load_model()
    planner        = agent._MakePlanner(model, dataset)

    all_records = []
    total = len(agent.test_nodes)

    for idx, node in enumerate(agent.test_nodes, 1):
        qname = node.info["query_name"]
        sql   = node.info["sql_str"]

        _, found_plan, predicted_total, _ = planner.plan(
            node, p.search_method, bushy=p.bushy,
            return_all_found=True, verbose=False,
        )

        hint = found_plan.hint_str(with_physical_hints=True)
        try:
            _, exec_node = postgres.ExecuteSql(sql, hint=hint, check_hint=False)
        except Exception as exc:
            print(f"  [SKIP] [{label}] {qname}: execution error: {exc}")
            continue

        before = len(all_records)
        collect_R(found_plan, exec_node, all_records, qname)
        n_samples = len(all_records) - before

        print(
            f"  [{label}] ({idx}/{total}) {qname}: "
            f"{n_samples} samples — "
            f"root predicted={predicted_total:.1f} ms, "
            f"root actual={exec_node.actual_time_ms:.1f} ms"
        )

    return all_records

# ── Build cp_hashmap from R records ──────────────────────────────────────────

def build_cp_hashmap(records, delta=DELTA):
    """
    Compute per-pattern upper bound C using Algorithm 1 (paper §3.3):
        C = sortedR[ ceil((K+1)(1-delta)) - 1 ]

    Patterns with fewer than K* = (1-delta)/delta samples fall back
    to the unified (pooled) upper bound.

    Returns:
        dict mapping pattern string → C value (float), plus "__default__"
    """
    import math
    K_min     = (1 - delta) / delta
    R_all     = []
    R_by_pat  = {}

    for rec in records:
        R_all.append(rec["R"])
        R_by_pat.setdefault(rec["pattern"], []).append(rec["R"])

    print(f"\n  Total K = {len(R_all)} samples across {len(R_by_pat)} patterns")
    for pat, rs in sorted(R_by_pat.items(), key=lambda kv: -len(kv[1])):
        print(f"    {pat}: K={len(rs)}")

    sorted_all  = sorted(R_all)
    n_all       = len(sorted_all)
    idx_unified = max(0, int(n_all * (1 - delta)) - 1)
    unified_C   = sorted_all[idx_unified]

    cp_hashmap = {}
    for pat, R_list in R_by_pat.items():
        if len(R_list) >= K_min:
            rs  = sorted(R_list)
            idx = max(0, int(len(rs) * (1 - delta)) - 1)
            cp_hashmap[pat] = round(rs[idx], 1)
        else:
            print(f"  [FALLBACK] {pat}: K={len(R_list)} < K*={K_min:.0f}, using unified C")
            cp_hashmap[pat] = round(unified_C, 1)

    cp_hashmap["__default__"] = round(unified_C, 1)
    return cp_hashmap

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_CALIB, exist_ok=True)
    os.makedirs(OUTPUT_TEST,  exist_ok=True)

    for ckpt in CHECKPOINTS:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_{ckpt}.pt")
        print(f"\n{'='*60}")
        print(f"  Checkpoint {ckpt}  ({checkpoint_path})")
        print(f"{'='*60}")

        # ── Calibration queries (47, templates c/d/e/f) ───────────────────
        print(f"\n[1/2] Collecting R from calibration queries (47 queries)...")
        calib_records = run_collection(checkpoint_path, CALIB_QUERIES, "calib")

        calib_raw_path = os.path.join(OUTPUT_CALIB, f"cp_hashmap_{ckpt}_raw.json")
        calib_map_path = os.path.join(OUTPUT_CALIB, f"cp_hashmap_{ckpt}.json")

        with open(calib_raw_path, "w") as f:
            json.dump(calib_records, f, indent=2)
        print(f"  Saved raw records ({len(calib_records)}) → {calib_raw_path}")

        print(f"\n  Building cp_hashmap (delta={DELTA})...")
        cp_hashmap = build_cp_hashmap(calib_records)
        with open(calib_map_path, "w") as f:
            json.dump(cp_hashmap, f, indent=2)
        print(f"  Saved cp_hashmap ({len(cp_hashmap)} patterns) → {calib_map_path}")

        # ── Test queries (33b) ────────────────────────────────────────────
        print(f"\n[2/2] Collecting R from test queries (33b queries)...")
        test_records = run_collection(checkpoint_path, TEST_QUERIES, "test")

        test_raw_path = os.path.join(OUTPUT_TEST, f"raw_33b_{ckpt}.json")
        with open(test_raw_path, "w") as f:
            json.dump(test_records, f, indent=2)
        print(f"  Saved raw records ({len(test_records)}) → {test_raw_path}")

    print(f"\n{'='*60}")
    print("  Calibration complete.")
    print(f"  Calib output : {OUTPUT_CALIB}/")
    print(f"  Test output  : {OUTPUT_TEST}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()