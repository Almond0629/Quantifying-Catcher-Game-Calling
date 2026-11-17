"""C1 freeze, step 1: reproduce C1's full scored run under the committed source and compare it with the stored run.

Outside gamecalling/ on purpose: adding a stage inside the package would change the package's code hash.

    python3.12 freeze_c1_reproduce.py selftest      # comparator checks on synthetic data; reads and writes nothing
    python3.12 freeze_c1_reproduce.py selfcompare   # read-only: the stored full run against itself (loading paths)
    python3.12 freeze_c1_reproduce.py guard         # hash the fixed inventory of existing artifacts (once)
    python3.12 freeze_c1_reproduce.py run           # the reproduction run (once); the guard is verified even on failure
    python3.12 freeze_c1_reproduce.py compare       # the comparison contract; the guard is verified even on failure

Full run: data/models/C1_repro/ with per-action rows in data/models/C1/runs/repro/. Nothing existing is written to.
S8_FULL_SMOKE=<dir> makes `run` a smoke run on one game in five, entirely under <dir>, as C1_repro_smoke / repro_smoke;
it shares no directory or run label with the full run and means nothing for decisions.

Comparison contract (fixed 2026-11-16, before the run). Compared exactly, by key, after checking row counts and key
uniqueness on both sides: scores, the four prediction files, per-action values, coverage; and from the manifests the
specification fields, inputs, concentrations and lambdas, policy concentrations, solve log, abnormal terminations (as
records with multiplicity) and fit-id hashes. Excluded, and only these: timestamps, timings, file paths, run-label
prefixes, the code block, and whole-file hashes of the per-action files (reported, not required). A manifest key that
is in neither list fails the comparison. Any mismatch stops the freeze for diagnosis; no tolerance applies.
A pass shows that the compared outputs reproduce. It does not cover the halves, the S1 runs or the validation stage.
"""

from gamecalling import research_s8_full as full  # noqa: I001 (first: sets the thread limit)

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from gamecalling import config, research
from gamecalling import research_s6_full as sf

ROOT = Path(__file__).resolve().parent
COMMIT = "60f4cd7"
CODE_HASH = "1000e503014a"                 # gamecalling_sha256 prefix at COMMIT (checked 2026-11-16)
SMOKE = os.environ.get("S8_FULL_SMOKE")
REF_NAME, REF_LABEL = "C1_full", "full"
NEW_NAME, NEW_LABEL = ("C1_repro_smoke", "repro_smoke") if SMOKE else ("C1_repro", "repro")
STORED = config.DATA_DIR / "models"        # the real stored outputs, also in smoke mode
RECORDS = Path(SMOKE) if SMOKE else config.DATA_DIR / "research"
GUARD_JSON = config.DATA_DIR / "research" / "c1_repro_guard.json"   # always the real inventory
RUN_JSON = RECORDS / "c1_repro_run.json"
COMPARE_JSON = RECORDS / "c1_repro_compare.json"

KEYS = list(research.KEYS)
TABLES = {"scores": KEYS, "heldout_outcome": KEYS, "heldout_contact": KEYS, "forward_outcome": KEYS,
          "forward_contact": KEYS}
ACTION_KEYS = KEYS + ["scoring_fold", "action"]
MANIFEST_EXACT = ("id", "description", "base", "seasons", "params", "changes", "inputs")
MANIFEST_RECORDS = ("alphas", "policy_alphas", "action_values", "solve_log", "abnormal_log")
MANIFEST_EXCLUDED = ("directory", "created", "code", "run_label", "seconds", "timings")
RECORD_EXCLUDED = {"action_values": ("path", "run_label", "sha256"),
                   "solve_log": ("seconds_probabilities", "seconds_solve")}


# --- Comparison primitives ---

def keyed_compare(new: pd.DataFrame, ref: pd.DataFrame, keys: list) -> dict:
    """Row counts, key uniqueness on both sides, then every column exactly after sorting by key. No tolerance."""
    if list(new.columns) != list(ref.columns):
        return {"identical": False, "detail": f"columns differ: {list(new.columns)} vs. {list(ref.columns)}"}
    if len(new) != len(ref):
        return {"identical": False, "detail": f"row counts differ: {len(new)} vs. {len(ref)}"}
    for name, d in (("new", new), ("stored", ref)):
        dup = int(d.duplicated(keys).sum())
        if dup:
            return {"identical": False, "detail": f"{dup} duplicate keys in the {name} table"}
    same_order = bool(new[keys].reset_index(drop=True).equals(ref[keys].reset_index(drop=True)))
    a = new.sort_values(keys, kind="stable").reset_index(drop=True)
    b = ref.sort_values(keys, kind="stable").reset_index(drop=True)
    diff = [c for c in b.columns if not a[c].equals(b[c])]
    if not diff:
        return {"identical": True, "rows": len(a), "same_row_order": same_order, "detail": "identical"}
    num = [c for c in diff if pd.api.types.is_numeric_dtype(b[c]) and not pd.api.types.is_bool_dtype(b[c])]
    largest = float(max((a[c] - b[c]).abs().max() for c in num)) if num else None
    return {"identical": False, "rows": len(a), "same_row_order": same_order, "detail": f"differ: {', '.join(diff)}",
            "largest_difference": largest}


def fit_key(fit_id: str) -> str:
    """'C1/<label>/<season>/<fold>/<hash>' without its run label."""
    model, _label, season, fold, digest = fit_id.split("/")
    return f"{model}/{season}/{fold}/{digest}"


def _canon(record: dict, drop=()) -> str:
    r = {k: (fit_key(v) if k == "fit_id" else v) for k, v in record.items() if k not in drop}
    return json.dumps(r, sort_keys=True)


def records_compare(new: list, ref: list, drop=()) -> dict:
    """Records with multiplicity: a duplicated or missing record is a difference."""
    a, b = Counter(_canon(r, drop) for r in new), Counter(_canon(r, drop) for r in ref)
    if a == b:
        return {"identical": True, "rows": len(ref), "detail": "identical"}
    only_new, only_ref = list((a - b).elements()), list((b - a).elements())
    return {"identical": False, "rows": len(ref),
            "detail": f"{len(only_new)} records only in the new run, {len(only_ref)} only in the stored run",
            "only_new": only_new[:5], "only_stored": only_ref[:5]}


def manifest_compare(new: dict, ref: dict) -> list:
    rows = []
    known = set(MANIFEST_EXACT) | set(MANIFEST_RECORDS) | set(MANIFEST_EXCLUDED) | {"fit_ids"}
    unknown = sorted((set(new) | set(ref)) - known)
    rows.append({"check": "manifest: every key is either compared or explicitly excluded", "identical": not unknown,
                 "detail": "none unlisted" if not unknown else f"unlisted keys: {unknown}"})
    for k in MANIFEST_EXACT:
        same = json.dumps(new.get(k), sort_keys=True) == json.dumps(ref.get(k), sort_keys=True)
        rows.append({"check": f"manifest: {k}", "identical": same, "detail": "identical" if same else "differs"})
    for k in MANIFEST_RECORDS:
        rows.append({"check": f"manifest: {k} (records with multiplicity)",
                     **records_compare(new.get(k, []), ref.get(k, []), RECORD_EXCLUDED.get(k, ()))})
    a = {k: fit_key(v) for k, v in new.get("fit_ids", {}).items()}
    b = {k: fit_key(v) for k, v in ref.get("fit_ids", {}).items()}
    rows.append({"check": "manifest: fit-id hashes by season and fold", "identical": a == b,
                 "detail": "identical" if a == b else f"{a} vs. {b}"})
    return rows


def compare_runs(new_dir: Path, new_label: str, ref_dir: Path, ref_label: str, models_root: Path) -> list:
    rows = []
    for what, keys in TABLES.items():
        r = keyed_compare(pd.read_parquet(new_dir / f"{what}.parquet"), pd.read_parquet(ref_dir / f"{what}.parquet"), keys)
        rows.append({"check": f"{what} (keyed, exact)", **r})
    cov_new, cov_ref = pd.read_csv(new_dir / "coverage.csv"), pd.read_csv(ref_dir / "coverage.csv")
    rows.append({"check": "coverage (exact)", **sf.exact_compare(cov_new, cov_ref)})
    man_new = json.loads((new_dir / "manifest.json").read_text())
    man_ref = json.loads((ref_dir / "manifest.json").read_text())
    rows += manifest_compare(man_new, man_ref)
    for rec in man_ref["action_values"]:
        rel = Path("action_values") / f"season={rec['season']}" / f"scoring_fold={rec['scoring_fold']}.parquet"
        p_new = models_root / "C1" / "runs" / new_label / rel
        p_ref = STORED / "C1" / "runs" / ref_label / rel
        r = keyed_compare(pd.read_parquet(p_new), pd.read_parquet(p_ref), ACTION_KEYS)
        r["file_hash_equal"] = _sha(p_new) == _sha(p_ref)     # reported, not required
        rows.append({"check": f"per-action values {rec['season']}/{rec['scoring_fold']} (keyed, exact)", **r})
    return rows


# --- Guard: a fixed inventory of existing artifacts ---

def _sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory() -> list:
    """Existing artifacts only. The reproduction's own outputs (NEW_NAME, NEW_LABEL) are outside it by construction."""
    files = [config.DATA_DIR / "phase1_scores.parquet", config.PITCHES_PARQUET, config.PAS_PARQUET,
             research.SUPPORT_PARQUET, config.DATA_DIR / "statcast_2024_2025.parquet"]
    for m in ("P0_repro", "M1c", "M3a", "M4", "C1", "C1_off", "M3a_full", "M4_full", "C1_full", "S1c_C1", "S1d_C1",
              "S1e_C1"):
        files += sorted(p for p in (STORED / m).iterdir() if p.is_file())
    for label_dir in sorted((STORED / "C1" / "runs").iterdir()):
        if label_dir.name not in ("repro", "repro_smoke"):
            files += sorted(p for p in label_dir.rglob("*") if p.is_file())
    files += sorted(p for p in (config.DATA_DIR / "research").glob("s8*") if p.is_file())
    return files


def stage_guard():
    if GUARD_JSON.exists():
        raise FileExistsError(f"{GUARD_JSON} exists: the inventory is taken once")
    hashes = {str(p): _sha(p) for p in inventory()}
    GUARD_JSON.write_text(json.dumps(hashes, indent=1))
    print(f"guarded {len(hashes)} existing files", flush=True)


def verify_guard() -> list:
    before = json.loads(GUARD_JSON.read_text())
    changed = [p for p, h in before.items() if not Path(p).exists() or _sha(p) != h]
    print(f"guard: {len(before)} files, {len(changed)} changed or missing", flush=True)
    return changed


# --- Run ---

def provenance() -> dict:
    import numpy
    import pyarrow
    import scipy
    try:
        from threadpoolctl import threadpool_info
        pools = [{k: d.get(k) for k in ("internal_api", "version", "num_threads")} for d in threadpool_info()]
    except ImportError:
        pools = None
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    return {"argv": sys.argv, "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "numpy": numpy.__version__, "scipy": scipy.__version__, "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__, "threadpools": pools,
            "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                          "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS")},
            "workers": full.WORKERS, "smoke": SMOKE, "git_head": git("rev-parse", "HEAD"),
            "git_status_gamecalling": git("status", "--porcelain", "gamecalling"), "code": research.code_state(),
            "script_sha256": _sha(Path(__file__))}


def preflight(prov: dict):
    if not prov["git_head"].startswith(COMMIT):
        raise RuntimeError(f"HEAD is {prov['git_head'][:7]}, not {COMMIT}")
    if prov["git_status_gamecalling"]:
        raise RuntimeError(f"gamecalling/ is not clean: {prov['git_status_gamecalling']}")
    if not prov["code"]["gamecalling_sha256"].startswith(CODE_HASH):
        raise RuntimeError(f"code hash {prov['code']['gamecalling_sha256'][:12]} is not {CODE_HASH}")
    stored = json.loads((STORED / REF_NAME / "manifest.json").read_text())["inputs"]
    now = research.input_state(full.C1.extra_inputs)
    if now != stored:
        raise RuntimeError("an input file differs from the stored full run's manifest")
    for dest in (research.MODELS_DIR / NEW_NAME, research.MODELS_DIR / "C1" / "runs" / NEW_LABEL, RUN_JSON):
        if dest.exists():
            raise FileExistsError(f"{dest} exists: this run happens once")


def stage_run():
    prov = provenance()
    preflight(prov)
    t = time.time()
    try:
        man = research.run_model(full.C1, out_name=NEW_NAME, run_label=NEW_LABEL, workers=full.WORKERS)
        rec = {"provenance": prov, "out_name": NEW_NAME, "run_label": NEW_LABEL, "started": t, "seconds": time.time() - t,
               "timings": man["timings"], "abnormal": len(man["abnormal_log"]),
               "scored": len(research.load(NEW_NAME, "scores"))}
        RUN_JSON.write_text(json.dumps(rec, indent=1, default=str))
        print(f"run {NEW_NAME}/{NEW_LABEL}: {rec['seconds']:.0f} s, {rec['scored']:,} scored pitches", flush=True)
    finally:
        changed = verify_guard()
    if changed:
        raise RuntimeError(f"existing artifacts changed: {changed[:5]}")


def stage_compare():
    if SMOKE:
        raise RuntimeError("the smoke run has no stored reference on its subset; it is not compared")
    if COMPARE_JSON.exists():
        raise FileExistsError(f"{COMPARE_JSON} exists: the comparison is recorded once")
    try:
        rows = compare_runs(STORED / NEW_NAME, NEW_LABEL, STORED / REF_NAME, REF_LABEL, STORED)
        COMPARE_JSON.write_text(json.dumps({"script_sha256": _sha(Path(__file__)), "rows": rows}, indent=1))
        print(pd.DataFrame(rows)[["check", "identical", "detail"]].to_string(), flush=True)
    finally:
        changed = verify_guard()
    if changed:
        raise RuntimeError(f"existing artifacts changed: {changed[:5]}")
    if not all(r["identical"] for r in rows):
        raise RuntimeError("the reproduction differs from the stored full run: stop and diagnose")
    print("all compared outputs identical", flush=True)


# --- Checks of the comparator itself ---

def stage_selfcompare():
    """Read-only: the stored full run against itself. Shows the loading paths and the manifest key lists are complete."""
    rows = compare_runs(STORED / REF_NAME, REF_LABEL, STORED / REF_NAME, REF_LABEL, STORED)
    print(pd.DataFrame(rows)[["check", "identical", "detail"]].to_string(), flush=True)
    if not all(r["identical"] for r in rows):
        raise RuntimeError("the stored run does not compare equal to itself: the comparator is wrong")


def stage_selftest():
    base = pd.DataFrame({"season": [2024, 2024, 2025], "pa_id": [1, 1, 2], "pitch_number": [1, 2, 1],
                         "D": [0.1, float("nan"), -0.2], "rec_action": ["FF", "SL", "CH"]})
    ok = lambda r: r["identical"]  # noqa: E731
    assert ok(keyed_compare(base, base.copy(), KEYS))
    shuffled = keyed_compare(base.iloc[::-1], base, KEYS)
    assert ok(shuffled) and not shuffled["same_row_order"]
    changed = base.copy()
    changed.loc[2, "D"] = -0.2 + 1e-15
    assert not ok(keyed_compare(changed, base, KEYS))
    rec = base.copy()
    rec.loc[0, "rec_action"] = "SI"
    assert not ok(keyed_compare(rec, base, KEYS))
    dup = pd.concat([base, base.iloc[[0]]])
    assert "duplicate" in keyed_compare(dup, dup.copy(), KEYS)["detail"]
    assert "row counts" in keyed_compare(dup, base, KEYS)["detail"]
    assert "columns" in keyed_compare(base.drop(columns="D"), base, KEYS)["detail"]
    ab = [{"variant": "M3a", "season": 2025, "status": 2, "max_grad": 1e-9}]
    assert ok(records_compare(ab, list(ab)))
    assert not ok(records_compare(ab + ab, ab))          # multiplicity
    assert not ok(records_compare([], ab))
    assert not ok(records_compare([{**ab[0], "max_grad": 2e-9}], ab))
    log = [{"fit_id": "C1/full/2024/1/abc", "iterations": 5, "seconds_solve": 1.0}]
    relabeled = [{"fit_id": "C1/repro/2024/1/abc", "iterations": 5, "seconds_solve": 9.0}]
    assert ok(records_compare(relabeled, log, RECORD_EXCLUDED["solve_log"]))
    assert not ok(records_compare([{**relabeled[0], "fit_id": "C1/repro/2024/1/abd"}], log, RECORD_EXCLUDED["solve_log"]))
    man = {"id": "C1", "description": "d", "base": "P0", "seasons": [2024], "params": {}, "changes": {}, "inputs": {},
           "alphas": [], "policy_alphas": [], "action_values": [], "solve_log": log, "abnormal_log": ab,
           "fit_ids": {"2024/1": "C1/full/2024/1/abc"}, "created": "x", "code": {}, "run_label": "full",
           "seconds": {}, "timings": {}, "directory": "C1_full"}
    other = {**man, "created": "y", "run_label": "repro", "directory": "C1_repro", "solve_log": relabeled,
             "fit_ids": {"2024/1": "C1/repro/2024/1/abc"}}
    assert all(ok(r) for r in manifest_compare(other, man))
    assert not all(ok(r) for r in manifest_compare({**other, "new_key": 1}, man))
    assert not all(ok(r) for r in manifest_compare({**other, "params": {"runners": False}}, man))
    print("selftest: all comparator checks passed", flush=True)


if __name__ == "__main__":
    stages = {"selftest": stage_selftest, "selfcompare": stage_selfcompare, "guard": stage_guard, "run": stage_run,
              "compare": stage_compare}
    if len(sys.argv) != 2 or sys.argv[1] not in stages:
        sys.exit(__doc__)
    stages[sys.argv[1]]()
