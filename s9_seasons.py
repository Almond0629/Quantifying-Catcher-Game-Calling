"""S9 Stage B driver: the shared control, the completeness gate and (later phases) the 2023 and 2026 seasons.

Specification: research_notes/research_program/S9_stage_B_protocol.md (LOCKED 2026-11-21). Outside gamecalling/ on purpose,
so the package's code hash stays 1000e503014a; it uses package functions only. Permitted changes (C1_frozen.json): input
paths and the versioned dugout flag (data/flags/dugout_calling_v1.json). Nothing here changes a model, tuning,
eligibility, scoring or ranking rule.

Phase 1 (this revision): the stages that need no season authorization and read no 2023 or 2026 pitch row.
Revision 7 (2026-11-22), after revision 6 (b400b28b..., commit b33c331) failed its first production control and gate:
(1) protocol amendment 1, the one permitted dtype difference in C1/C2 (PERMITTED_DTYPE: support.n stored int64 vs
float64 from the frozen package; values exact); (2) completeness mapping v2, a general rule revalidated on development
data as §8.1 allows (kept(): any no-pitch call); (3) amendment 2, the diagnosed 2024 shortfall in 746731
(EXPECTED_DEV_SHORTFALL), pinned to MLB's kept-event sequence and Statcast's descriptions. Both amendments were
approved by the user on 2026-11-22. Review round 1 (Astra, Neon): duplicate atBatIndex rejected; cast round-trip guard.
Revision 8 (2026-11-22): phase 2, the target-season stages for 2023 (below). Accepts revision 7's records.
Revision 9 (2026-11-22), after revision 8's smoke run stopped at validate (approved by the user): (1) the frozen criterion
baseline may give an observed class probability exactly 0 (no HBP at 3-1 in the smoke valuation fold); its log loss is
reported as +inf with the rows disclosed, while the §7 probability checks, every equality check and finite log
probabilities for P0, C1 and the league-pooled level stay fail-closed (heldout_metrics); (2) the comparison tables in
`checks` (coverage, environment, B2) label each sample, so the smoke stand-in never shares a label with development 2024;
(3) season child results (run, halves, validate) are strict JSON (json_safe: +inf/-inf labels, NaN as null). Review round 1 (Astra): B2 labels, the
validate summaries' NaN, and per-component wording in the zero-probability disclosure.

    python3.12 s9_seasons.py selftest    # synthetic checks only; reads no project data, writes nothing
    python3.12 s9_seasons.py guard       # once: hash the protected inventory
    python3.12 s9_seasons.py audit       # once: code and schema audit (§1); reads schemas only
    python3.12 s9_seasons.py control     # once: the shared control C1-C8b on 2024-2025 (§4)
    python3.12 s9_seasons.py gate        # once per mapping version: the completeness gate on 2024-2025 (§8.1)
    python3.12 s9_seasons.py recheck     # read-only, repeatable: the guard and every record

Phase 2 (revision 8): the target-season stages (2023; §5, §6). Each refuses unless guard, audit, control, gate_v2
verify ("ready for target-season work") and the season's authorization record verifies.

    python3.12 s9_seasons.py domains                 # once, development data only: the §5 B0 closed domains
    python3.12 s9_seasons.py authorize 2023 <date>   # once, ONLY on the user's explicit instruction (A11-3)
    python3.12 s9_seasons.py prefetch 2023           # repeatable, no record: schedule and play-by-play into the cache
    python3.12 s9_seasons.py extract 2023            # once: the 2023 rows of statcast_2015_2023.parquet
    python3.12 s9_seasons.py build 2023              # once: season tables and halves (before any score)
    python3.12 s9_seasons.py checks 2023             # once: B0-B2, pre-fit, completeness, coverage, environment
    python3.12 s9_seasons.py run 2023                # once: P0 and C1 full runs, post-run checks (child process)
    python3.12 s9_seasons.py halves 2023             # once: the independent refit in each half (child process)
    python3.12 s9_seasons.py validate 2023           # once: §6 rows 2-9 and the 2023-2024 pair (child process)
    python3.12 s9_seasons.py report 2023             # once: reports/research_s9_2023.md
Smoke runs the season stages only on the stand-in 2024 (a development season), never on a target season.

Bindings (§3): one binding per process. A context (single-season run, joint run, halves refit, replay) runs as its own
child process `s9_seasons.py _ctx <name>` with its paths in the environment variable S9_BIND (JSON), applied at import
right after gamecalling.research_s6 (the thread limit). Spawned workers inherit the environment and re-import this
script as __mp_main__, so they apply the same binding. A binding never changes inside a process.

Smoke: S9_SEASON_SMOKE=<dir> runs every stage on one game in five (game_pk % 5 == 0), entirely under <dir>. Its
references are rebuilt on the same subset through the package's joint path (context smoke_ref), so the comparisons that
matter (single season vs joint, worker bindings, restricted forward queries) are exact there too; the rest only exercise
the code. Smoke results mean nothing for decisions. Every other variable containing SMOKE is refused in every mode.
"""

import os
import sys

_FOREIGN_SMOKE = sorted(k for k in os.environ if "SMOKE" in k and k != "S9_SEASON_SMOKE")
if _FOREIGN_SMOKE:
    sys.exit(f"refused: {_FOREIGN_SMOKE} set. S8_SMOKE changes C1's input transformation; no other project smoke "
             f"setting may be inherited by this script.")
if "S9_SEASON_SMOKE" in os.environ and not os.environ["S9_SEASON_SMOKE"].strip():
    sys.exit("refused: S9_SEASON_SMOKE is set but empty. Unset it for production, or give the smoke directory.")
_ROOT_EARLY = os.path.dirname(os.path.realpath(__file__))
if os.environ.get("S9_SEASON_SMOKE"):
    _sd = os.path.realpath(os.environ["S9_SEASON_SMOKE"])
    if _sd == _ROOT_EARLY or _sd.startswith(_ROOT_EARLY + os.sep):
        sys.exit(f"refused: the smoke directory {_sd} is inside the project; smoke runs only outside it (scratchpad).")
_IS_CTX = len(sys.argv) >= 3 and sys.argv[1] == "_ctx"
if os.environ.get("S9_BIND") and __name__ == "__main__" and not _IS_CTX:
    sys.exit("refused: S9_BIND is set for a top-level stage (a stale outer binding); only child contexts take one.")

from gamecalling import research_s6  # noqa: E402,I001 (first: sets the thread limit before numpy loads)

import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import pickle  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from gamecalling import arsenal, batter, config, loader, models, research, validation  # noqa: E402
from gamecalling import batter_scoring as bs  # noqa: E402
from gamecalling import research_s6_full as sf  # noqa: E402
from gamecalling import research_s7  # noqa: E402
from gamecalling import research_s8  # noqa: E402
from gamecalling import research_s8_full as s8f  # noqa: E402
from gamecalling import solver, payoff  # noqa: E402
from gamecalling import validation_pilot as vp  # noqa: E402
import freeze_c1_reproduce as fcr  # noqa: E402  (reviewed comparators: keyed_compare, records_compare, fit_key)

ROOT = Path(__file__).resolve().parent
SCRIPT = Path(__file__).resolve()
SMOKE = os.environ.get("S9_SEASON_SMOKE")
MODE = "smoke" if SMOKE else "production"
SMOKE_DIR = str(Path(SMOKE).resolve()) if SMOKE else None
BASE = Path(SMOKE).resolve() if SMOKE else config.DATA_DIR / "s9"
RECORDS = BASE / "records"
CONTROL = BASE / "control"
MLB_CACHE = BASE / "mlb_cache"

CODE_HASH = "1000e503014a"
FULL_CODE_HASH = json.loads((ROOT / "data" / "models" / "C1_frozen.json").read_text())["source"]["gamecalling_sha256"]
PROTOCOL = ROOT / "research_notes" / "research_program" / "S9_stage_B_protocol.md"
FLAGS = ROOT / "data" / "flags" / "dugout_calling_v1.json"
WORKERS = 8
KEYS = list(research.KEYS)
DEV_SEASONS = (2024, 2025)

# The stored development inputs and references, by path (never through config, which a binding changes).
D = ROOT / "data"
ORIG = {"raw": D / "statcast_2024_2025.parquet", "pitches": D / "pitches_2024_2025.parquet",
        "pas": D / "plate_appearances_2024_2025.parquet", "support": D / "arsenal_support.parquet"}
RAW_2015_2023 = D / "statcast_2015_2023.parquet"
STORED_REF = {
    "P0_dir": D / "models" / "P0_repro", "C1_dir": D / "models" / "C1_full",
    "C1_actions": D / "models" / "C1" / "runs" / "full",
    "P0_halves": D / "validation_pilot" / "v1" / "half_runs.pkl", "C1_halves": D / "research" / "s8_full_halves_C1.pkl",
    "v0": D / "validation_pilot" / "v1" / "v0.pkl", "v2": D / "validation_pilot" / "v1" / "v2.pkl",
    "s8": D / "research" / "s8_validate.pkl",
}
FORWARD_RECORD = D / "models" / "C1_forward_2025.json"
FROZEN_RECORD = D / "models" / "C1_frozen.json"

# Manifest contracts (§4 C4): what is compared, what may differ, and P0's expected defaults.
MANIFEST_EXACT = ("description", "base", "params")
MANIFEST_RECORDS = ("alphas", "policy_alphas")
C1_RECORDS = ("action_values", "solve_log", "abnormal_log")
PERMITTED = ("directory", "created", "run_label", "seconds", "timings", "code")
P0_DEFAULTS = {"fit_ids": {}, "solve_log": [], "abnormal_log": [], "action_values": []}
ACTION_KEYS = KEYS + ["scoring_fold", "action"]

# Completeness gate (§8.1)
MAPPING_VERSION = 2          # v1 (VB, VP, AC only) failed on development data, 2026-11-22: see kept()
SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&gameType=R&season={season}"
PBP_URL = "https://statsapi.mlb.com/api/v1/game/{pk}/playByPlay"
EXPECTED_COMPLETED = {2024: 2429, 2025: 2430}
EXPECTED_ABSENT = {2025: {778563, 778564}}           # the Tokyo games missing from the development extract
# Protocol amendment 2 (approved by the user 2026-11-22): a diagnosed shortfall in the frozen 2024 development extract.
# 746731, PA 68 (bottom 8th): MLB's play-by-play has five pitches (F, B, F, *B ball in dirt, S); Statcast has four
# rows (foul, ball, foul, swinging_strike), without the ball in dirt, and records the strikeout pitch at 1-2 where MLB
# has 2-2. A data shortfall, not a mapping failure. The gate reports it as a confirmed missing pitch and lets
# development validation pass only on exactly this defect: the key difference, MLB's kept-event codes in that PA, and
# Statcast's descriptions there (Astra: a key difference alone could come from another omitted pitch).
EXPECTED_DEV_SHORTFALL = {2024: {746731: {"missing": [[68, 5]], "extra": [], "pa": 68,
                                          "mlb_kept_codes": ["F", "B", "F", "*B", "S"],
                                          "statcast_descriptions": ["foul", "ball", "foul", "swinging_strike"]}}}
CANCELLED_EXAMPLE = 746577
BACKOFF = (2, 5, 15, 45)
FETCH_THREADS = 6
USER_AGENT = "research-client"
AUTOMATIC_CALLS = frozenset({"VB", "VP", "AC"})      # mapping v1's list (history); v2 keeps any no-pitch call
# Smoke only: the one-in-five subset changes per-batter PA counts, so the position-player flag (and only what follows
# from it) may differ in a single-season build. Any other difference fails the smoke run.
SMOKE_ALLOWED_COLS = frozenset({"position_player_pitching", "exclusion_reason", "outcome_fit_eligible",
                                "contact_fit_eligible", "arsenal_reason", "in_eligible_set", "scorable"})
# Smoke only: the numerical comparisons a season with an expected position-player difference may fail. Exit status,
# worker probe and every structural or provenance check (ids, inputs, code digest, keys, defaults) must still pass.
SMOKE_WAIVABLE = ("scores (keyed", "heldout_outcome (keyed", "heldout_contact (keyed", "coverage (exact)",
                  "alphas (records", "solve_log (records", "action_values (records", "fit-id hashes",
                  "per-action values")   # abnormal_log is never waived; structural rows are never waived
RECORD_IDS = {"alphas": ("season", "fold"), "policy_alphas": ("season", "scoring_fold", "role"),
              "solve_log": ("season", "scoring_fold", "role"), "action_values": ("model", "season", "scoring_fold"),
              "abnormal_log": ("season", "component", "code", "inner", "lambda")}
# Reviewed earlier revisions whose records a later one accepts. Revision 8 (phase 2) accepts revision 7's development
# records (guard, audit, control, gate_v<MAPPING_VERSION>, domains): revision 7's final digest, set when it is committed.
APPROVED_PREDECESSOR_SCRIPTS = frozenset({"080ac1996b03c8da17a6e1cd76bed80b4df99364b428f21b7ece080c65ce846c"})
# Protocol amendment 1 (approved by the user 2026-11-22, after the first production control failed at C1 on this): the stored
# arsenal_support.parquet was written by Phase 0 on 2026-10-27 under ca9d25c, whose support_table counted rows with
# .size() (int64). Commit 483d573 (2026-10-29) added unit weights (weighting.w, float), so the frozen package's
# support_table sums floats and gives n as float64. All 36,385 rows' values are identical. (table, column, stored dtype,
# new dtype); applied only in tables_compare (C1, C2), only to integral finite values, and nothing else is relaxed.
PERMITTED_DTYPE = (("support", "n", "int64", "float64"),)

# Phase 2: target seasons (§3, §5, §6). 2026's fetch, reconcile, publish and forward stages are phase 3 (§7, §8.2).
TARGETS = (2023,)                    # production: the season stages run only on these, each after its authorization
SMOKE_STANDIN = 2024                 # smoke only: a development season stands in for a target, so no target row is read
SEASON_LABEL = {2023: "previously unmodeled, with prior data-quality inspection disclosed",
                2026: "not previously downloaded"}
AUTHORIZATION = {2023: "A11-3"}
PAIRS = {2023: ((2023, 2024),), SMOKE_STANDIN: ((2024, 2025),)}   # §6 row 6: the A4 pairs a season belongs to
SEASONS_ROOT = (Path(SMOKE).resolve() / "seasons") if SMOKE else D / "seasons"
REPORTS_OUT = (Path(SMOKE).resolve() / "reports") if SMOKE else ROOT / "reports"
SEASON_CONTEXT_STAGES = ("run", "halves", "validate")     # season stages whose work runs in a bound child process
PROTOCOL_LOCKED_ON = "2026-11-21"
CAL_MIN_N = 10                       # §6 row 10: calibration is not computed on fewer rows (Neon)
# Each child context belongs to exactly one stage (Astra: bind the dispatched context to its reserved stage).
CONTEXT_STAGE = {"smoke_ref": "control", "single": "control", "joint": "control", "halves": "control",
                 "replay": "control", "forward": "control",
                 "season_run": "run", "season_halves": "halves", "season_validate": "validate"}
EXPECTED_PREDECESSORS = {"guard": set(), "audit": {"guard"}, "control": {"guard", "audit"}}   # gate_v<n>: {"guard"}


# ===================================================================================== binding (one per process)

try:
    BIND = json.loads(os.environ["S9_BIND"]) if os.environ.get("S9_BIND") else None
except ValueError:
    sys.exit("refused: S9_BIND is not valid JSON")
if BIND is not None and not isinstance(BIND, dict):
    sys.exit("refused: S9_BIND must be a JSON object")


def apply_binding(b: dict) -> None:
    """Point the package's module-level paths at one context's files. Called once, at import, in every process."""
    config.RAW_PARQUET = Path(b["raw"])
    config.PITCHES_PARQUET = Path(b["pitches"])
    config.PAS_PARQUET = Path(b["pas"])
    research.SUPPORT_PARQUET = Path(b["support"])
    research.MODELS_DIR = Path(b["models_dir"])
    research._DATA.clear()
    vp._DATA.clear()
    if b.get("vp_seasons"):
        vp.SEASONS = tuple(b["vp_seasons"])


def validate_binding(b: dict, context=None) -> list:
    """A child's binding: its output inside this mode's control directory (control contexts) or exactly its own
    season stage's directory with the season's canonical inputs (season contexts, phase 2), never the project's guarded
    data; models under its output; every bound input present; the dispatched context belonging to the bound stage; and
    that stage reserved (contexts are not callable on their own)."""
    problems = []
    stage = b.get("stage", "control")
    if not isinstance(stage, str):
        return ["binding stage is not a string"]
    if stage == "control":
        root = (BASE / "control").resolve()
        name = "control"
    else:
        name, _, y = stage.rpartition("_")
        if name not in SEASON_CONTEXT_STAGES or not y.isdigit() or str(b.get("season")) != y:
            return [f"binding stage {stage!r} is neither control nor a season context stage of its own season"]
        root = (SEASONS_ROOT / y).resolve()
    if context is not None and CONTEXT_STAGE.get(context) != name:
        return [f"context {context!r} does not belong to stage {stage!r}"]
    out = Path(b.get("out", "")).resolve()
    if out == root or root not in out.parents:
        problems.append(f"output {out} is not inside {root}")
    if stage != "control":
        if out != root / name:
            problems.append(f"output {out} is not the stage directory {root / name}")
        for k, f in (("raw", "raw.parquet"), ("pitches", "pitches.parquet"), ("pas", "pas.parquet"),
                     ("support", "support.parquet")):
            if Path(b.get(k, "")).resolve() != root / f:
                problems.append(f"bound input {k} is not the season's {f}")
    if Path(b.get("models_dir", "")).resolve() != out / "models":
        problems.append("models_dir must be <out>/models")
    for k in ("raw", "pitches", "pas", "support"):
        if not Path(b.get(k, "")).is_file():
            problems.append(f"bound input {k} missing: {b.get(k)}")
    rec = BASE / "records" / f"s9_{stage}.json"
    try:
        reserved = rec.exists() and json.loads(rec.read_text()).get("status") == "reserved"
    except ValueError:
        reserved = False
    if not reserved:
        problems.append(f"no reserved {stage} stage: contexts run only inside one")
    return problems


if BIND is not None:
    if _IS_CTX:
        _bad = validate_binding(BIND, sys.argv[2])
        if _bad:
            sys.exit("refused: invalid binding:\n  " + "\n  ".join(_bad))
    apply_binding(BIND)


def binding_view() -> dict:
    """What this process actually sees (the worker probe compares it with the binding)."""
    return {"raw": str(config.RAW_PARQUET), "pitches": str(config.PITCHES_PARQUET), "pas": str(config.PAS_PARQUET),
            "support": str(research.SUPPORT_PARQUET), "models_dir": str(research.MODELS_DIR),
            "vp_seasons": list(vp.SEASONS), "attach_raw_reads": str(config.RAW_PARQUET), "pid": os.getpid()}


def probe(_=None) -> dict:
    """Run in a spawned worker: the paths it sees, with the hashes of its input files (a missing input is recorded as
    missing, never dropped)."""
    v = binding_view()
    v["hashes"] = {k: (sha(v[k]) if Path(v[k]).is_file() else "MISSING") for k in ("raw", "pitches", "pas", "support")}
    v["threads"] = research_s6.verify_threads()
    return v


def probe_check(b: dict) -> dict:
    """A spawned worker's view must equal this child's binding (§3 worker probe)."""
    with ProcessPoolExecutor(max_workers=1) as ex:
        w = ex.submit(probe).result()
    want = {"raw": b["raw"], "pitches": b["pitches"], "pas": b["pas"], "support": b["support"],
            "models_dir": b["models_dir"], "attach_raw_reads": b["raw"]}
    bad = {k: (w[k], v) for k, v in want.items() if w[k] != v}
    if b.get("vp_seasons") and w["vp_seasons"] != list(b["vp_seasons"]):
        bad["vp_seasons"] = (w["vp_seasons"], b["vp_seasons"])
    main = {k: (sha(b[k]) if Path(b[k]).is_file() else "MISSING") for k in ("raw", "pitches", "pas", "support")}
    if "MISSING" in main.values() or w["hashes"] != main:
        bad["hashes"] = (w["hashes"], main)
    return {"ok": not bad, "worker_pid": w["pid"], "parent_pid": os.getpid(), "differences": bad,
            "threads": w["threads"]}


# ===================================================================================== small utilities

def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def jdump(obj) -> str:
    return json.dumps(obj, indent=1, default=str, sort_keys=True)


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def code_ok() -> dict:
    cs = research.code_state()
    dirty = git("status", "--porcelain", "--", "gamecalling")
    return {"ok": cs["gamecalling_sha256"] == FULL_CODE_HASH and not dirty, "gamecalling_sha256": cs["gamecalling_sha256"],
            "gamecalling_dirty": dirty.splitlines(), "head": git("rev-parse", "HEAD")}


def script_state() -> dict:
    tracked = bool(git("ls-files", "--", SCRIPT.name))
    dirty = bool(git("status", "--porcelain", "--", SCRIPT.name)) if tracked else True
    return {"path": SCRIPT.name, "sha256": sha(SCRIPT), "committed": tracked and not dirty}


def environment() -> dict:
    import scipy
    import pyarrow
    return {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__,
            "pandas": pd.__version__, "pyarrow": pyarrow.__version__, "workers": WORKERS,
            "threads": research_s6.verify_threads()}


# ===================================================================================== comparisons

def frames_compare(new: pd.DataFrame, ref: pd.DataFrame, keys: list, bitwise: bool = False,
                   same_order: bool = True) -> dict:
    """Copied from freeze_forward_2025.py (reviewed; its import refuses any *SMOKE* setting). Columns, row counts and key
    uniqueness first; same_order: keys already in the same order; bitwise: dtypes and bytes identical; otherwise values
    equal (NaN equal to NaN) and dtypes must also match (reported as a failure here)."""
    if list(new.columns) != list(ref.columns):
        return {"identical": False, "detail": f"columns differ: {list(new.columns)} vs. {list(ref.columns)}"}
    if len(new) != len(ref):
        return {"identical": False, "detail": f"row counts differ: {len(new)} vs. {len(ref)}"}
    for name, d in (("new", new), ("reference", ref)):
        dup = int(d.duplicated(keys).sum()) if keys else 0
        if dup:
            return {"identical": False, "detail": f"{dup} duplicate keys in the {name} table"}
    order = bool(new[keys].reset_index(drop=True).equals(ref[keys].reset_index(drop=True))) if keys else True
    if same_order and not order:
        return {"identical": False, "detail": "keys are not in the same order"}
    a = new.reset_index(drop=True) if order else new.sort_values(keys, kind="stable").reset_index(drop=True)
    b = ref.reset_index(drop=True) if order else ref.sort_values(keys, kind="stable").reset_index(drop=True)
    dtypes = [c for c in b.columns if a[c].dtype != b[c].dtype]
    if dtypes:
        return {"identical": False, "detail": f"dtypes differ: {dtypes}"}
    if bitwise:
        diff = []
        for c in b.columns:
            x, y = a[c].to_numpy(), b[c].to_numpy()
            same = bool(a[c].equals(b[c])) if x.dtype == object else \
                np.ascontiguousarray(x).tobytes() == np.ascontiguousarray(y).tobytes()
            if not same:
                diff.append(c)
    else:
        diff = [c for c in b.columns if not a[c].equals(b[c])]
    out = {"identical": not diff, "rows": len(b), "same_row_order": order,
           "detail": "identical" if not diff else f"differ: {', '.join(diff)}"}
    num = [c for c in diff if pd.api.types.is_float_dtype(b[c])]
    if num:
        out["largest_difference"] = float(max(np.nanmax(np.abs(a[c].to_numpy(float) - b[c].to_numpy(float)))
                                              for c in num))
    return out


def deep_compare(new, ref, path="", out=None, projection=True) -> list:
    """Every difference between a recomputed object and a stored one. projection: keys only in `new` are reported as
    extra (not differences), so a stored result is compared on what it stores (Astra, draft 6 check)."""
    out = [] if out is None else out
    if isinstance(ref, pd.DataFrame):
        if not isinstance(new, pd.DataFrame):
            out.append(f"{path}: type {type(new).__name__} vs DataFrame")
        elif list(new.columns) != list(ref.columns) or list(new.dtypes) != list(ref.dtypes) or \
                not new.index.equals(ref.index) or not new.equals(ref):
            out.append(f"{path}: DataFrame differs ({new.shape} vs {ref.shape})")
    elif isinstance(ref, pd.Series):
        if not isinstance(new, pd.Series) or new.dtype != ref.dtype or not new.index.equals(ref.index) or not new.equals(ref):
            out.append(f"{path}: Series differs")
    elif isinstance(ref, dict):
        if not isinstance(new, dict):
            out.append(f"{path}: type {type(new).__name__} vs dict")
        else:
            for k in ref:
                if k not in new:
                    out.append(f"{path}/{k!r}: missing")
                else:
                    deep_compare(new[k], ref[k], f"{path}/{k!r}", out, projection)
            if not projection:
                for k in new:
                    if k not in ref:
                        out.append(f"{path}/{k!r}: extra")
    elif isinstance(ref, (list, tuple)):
        if not isinstance(new, (list, tuple)) or len(new) != len(ref):
            out.append(f"{path}: sequence differs")
        else:
            for i, (x, y) in enumerate(zip(new, ref)):
                deep_compare(x, y, f"{path}[{i}]", out, projection)
    elif isinstance(ref, (set, frozenset)):
        if set(new) != set(ref):
            out.append(f"{path}: set differs")
    elif isinstance(ref, (float, np.floating)):
        if not isinstance(new, (float, np.floating, int, np.integer)):
            out.append(f"{path}: {new!r} vs {ref!r}")
        elif not ((math.isnan(float(ref)) and math.isnan(float(new))) or float(new) == float(ref)):
            out.append(f"{path}: {new!r} vs {ref!r}")
    elif isinstance(ref, np.ndarray):
        if not isinstance(new, np.ndarray) or not np.array_equal(new, ref, equal_nan=ref.dtype.kind == "f"):
            out.append(f"{path}: array differs")
    else:
        try:
            same = bool(new == ref)
        except Exception:  # noqa: BLE001
            same = False
        if not same:
            out.append(f"{path}: {new!r} vs {ref!r}")
    return out


def extras(new, ref, path="") -> list:
    """Keys present in the recomputed object but not in the stored one (reported, not failures)."""
    out = []
    if isinstance(ref, dict) and isinstance(new, dict):
        out += [f"{path}/{k!r}" for k in new if k not in ref]
        for k in ref:
            if k in new:
                out += extras(new[k], ref[k], f"{path}/{k!r}")
    return out


def structure_compare(new: pd.DataFrame, ref: pd.DataFrame, keys: list) -> dict:
    """Structural, never waived in smoke: same columns, same dtypes, unique keys, and the same sorted key set."""
    if list(new.columns) != list(ref.columns):
        return {"identical": False, "detail": "columns differ"}
    dt = [c for c in ref.columns if new[c].dtype != ref[c].dtype]
    if dt:
        return {"identical": False, "detail": f"dtypes differ: {dt}"}
    if new.duplicated(keys).any() or ref.duplicated(keys).any():
        return {"identical": False, "detail": "duplicate keys"}
    a = new[keys].sort_values(keys).reset_index(drop=True)
    b = ref[keys].sort_values(keys).reset_index(drop=True)
    return {"identical": bool(a.equals(b)), "detail": "same key set" if a.equals(b) else "key sets differ",
            "rows": len(ref)}


def records_structure(new: list, ref: list, ids: tuple) -> dict:
    """Structural, never waived in smoke: the same number of records with the same identifying fields."""
    key = lambda r: json.dumps({k: r.get(k) for k in ids}, sort_keys=True, default=str)  # noqa: E731
    a, b = Counter(map(key, new)), Counter(map(key, ref))
    return {"identical": a == b, "detail": f"{len(new)} vs {len(ref)} records" + ("" if a == b else "; identities differ")}


def manifest_contract(new: dict, ref: dict, version: str, season=None, joint: bool = False, actual_inputs=None) -> list:
    """§4 C4 and C5. season: filter the reference records to that season (single-season runs). joint: inputs and
    seasons must also be equal. actual_inputs: research.input_state of the files actually bound (names, bytes, full
    digests); the new manifest's inputs must equal it exactly."""
    rows = []

    def add(check, ok, detail=""):
        rows.append({"check": f"manifest {version}: {check}", "identical": bool(ok), "detail": detail or
                     ("identical" if ok else "differs")})

    filt = (lambda recs: [r for r in recs if r.get("season") == season]) if season is not None else (lambda r: r)
    compared = set(MANIFEST_EXACT) | set(MANIFEST_RECORDS) | {"id", "changes", "seasons", "inputs", "fit_ids"} | \
        set(C1_RECORDS)
    unknown = sorted((set(new) | set(ref)) - compared - set(PERMITTED))
    add("every key compared or permitted", not unknown, f"unlisted: {unknown}" if unknown else "none unlisted")
    add("id", new.get("id") == ref.get("id") == version)
    for k in MANIFEST_EXACT:
        add(k, json.dumps(new.get(k), sort_keys=True, default=str) == json.dumps(ref.get(k), sort_keys=True, default=str))
    add("code hash is the frozen full digest", new.get("code", {}).get("gamecalling_sha256") == FULL_CODE_HASH)
    if actual_inputs is not None:
        add("inputs are the bound files (names, bytes, full digests)",
            json.dumps(new.get("inputs"), sort_keys=True) == json.dumps(actual_inputs, sort_keys=True))
    if joint:
        add("seasons", new.get("seasons") == ref.get("seasons"))
        add("inputs (same files)", json.dumps(new.get("inputs"), sort_keys=True) == json.dumps(ref.get("inputs"), sort_keys=True))
    else:
        add("seasons (the one season)", new.get("seasons") == [season])
    if version == "P0":
        rc, nc = ref.get("changes", {}), new.get("changes", {})
        add("changes on the stored keys", {k: nc.get(k) for k in rc} == rc)
        add("changes.solve is false (expected default)", nc.get("solve") is False and set(nc) == set(rc) | {"solve"})
        for k, v in P0_DEFAULTS.items():
            add(f"{k} is the empty default", new.get(k) == v and (k not in ref or ref[k] == v))
    else:
        add("changes", new.get("changes") == ref.get("changes"))
        for k in C1_RECORDS:
            st = records_structure(new.get(k, []), filt(ref.get(k, [])), RECORD_IDS[k])
            add(f"{k}: record count and identities (structural)", st["identical"], st["detail"])
            r = fcr.records_compare(new.get(k, []), filt(ref.get(k, [])), fcr.RECORD_EXCLUDED.get(k, ()))
            add(f"{k} (records with multiplicity)", r["identical"], r["detail"])
        a = {k: fcr.fit_key(v) for k, v in new.get("fit_ids", {}).items()}
        b = {k: fcr.fit_key(v) for k, v in ref.get("fit_ids", {}).items()
             if season is None or k.startswith(f"{season}/")}
        member = lambda d: {k: v.rsplit("/", 1)[0] for k, v in d.items()}  # noqa: E731  model/season/fold
        add("fit-id membership (model, season, fold)", member(a) == member(b),
            "identical" if member(a) == member(b) else f"{member(a)} vs {member(b)}")
        add("fit-id hashes (run label dropped, model id kept)", a == b, "identical" if a == b else f"{a} vs {b}")
    for k in MANIFEST_RECORDS:
        st = records_structure(new.get(k, []), filt(ref.get(k, [])), RECORD_IDS[k])
        add(f"{k}: record count and identities (structural)", st["identical"], st["detail"])
        r = fcr.records_compare(new.get(k, []), filt(ref.get(k, [])))
        add(f"{k} (records with multiplicity)", r["identical"], r["detail"])
    return rows


# ===================================================================================== tables (§3)

def load_flags(path=FLAGS) -> dict:
    d = json.loads(Path(path).read_text())
    if d.get("version") != 1:
        raise ValueError("dugout flag table: version 1 expected")
    return d


def dugout_columns(pitches: pd.DataFrame, flags: dict) -> tuple[pd.Series, pd.Series]:
    """dugout_called (documented practice only) and dugout_level, from the versioned table (§2)."""
    level = pd.Series(pd.NA, index=pitches.index, dtype="object")
    for e in flags["entries"]:
        m = pitches["season"].eq(e["season"]) & pitches["fielding_team"].eq(e["team"])
        if e.get("from"):
            m &= pitches["game_date"] >= e["from"]
        if e.get("to"):
            m &= pitches["game_date"] <= e["to"]
        level = level.mask(m, e["level"])
    return level.eq("documented").fillna(False).astype(bool), level


def build_tables(raw: pd.DataFrame, flags: dict) -> tuple:
    """phase0.main's steps on the given raw rows, with the versioned dugout flag: dugout_called keeps its column
    position (values from the table); dugout_level is appended last, after the arsenal marks."""
    pitches = loader.build_pitches(raw)
    called, _ = dugout_columns(pitches, flags)
    pitches["dugout_called"] = called.to_numpy()
    pitches = pitches.join(arsenal.assign_folds(pitches), on="game_pk")
    support = arsenal.support_table(pitches)
    pitches = arsenal.mark_arsenal(pitches, arsenal.eligible_sets(support))
    pas = loader.build_plate_appearances(pitches)
    pitches["dugout_level"] = dugout_columns(pitches, flags)[1].to_numpy()
    return pitches, pas, support


def smoke_subset(raw: pd.DataFrame) -> pd.DataFrame:
    return raw[raw["game_pk"] % 5 == 0].reset_index(drop=True) if SMOKE else raw


def read_raw_rows(path: Path, season=None) -> pd.DataFrame:
    """The build's columns (loader.read_raw with an explicit path), optionally one game_year."""
    raw = loader.read_raw(path)
    if season is not None:
        raw = raw[raw["game_year"] == season].reset_index(drop=True)
    return smoke_subset(raw)


def write_raw_extract(src: Path, dst: Path, season: int) -> dict:
    """All columns of one game_year (every game type), as the season's raw file for attach_raw."""
    if dst.exists():
        raise FileExistsError(f"{dst} exists")
    t = pq.read_table(src, filters=[("game_year", "=", season)])
    if SMOKE:
        df = t.to_pandas()
        t = t.from_pandas(df[df["game_pk"] % 5 == 0].reset_index(drop=True), schema=t.schema, preserve_index=False)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(t, dst)
    return {"path": str(dst), "rows": t.num_rows, "sha256": sha(dst)}


def save_tables(d: Path, pitches, pas, support) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, df in (("pitches", pitches), ("pas", pas), ("support", support)):
        p = d / f"{name}.parquet"
        if p.exists():
            raise FileExistsError(f"{p} exists")
        df.to_parquet(p, index=False)
        out[name] = str(p)
    return out


def roundtrip(df: pd.DataFrame) -> pd.DataFrame:
    """The frame as the pipeline reads it back: written to Parquet and read again (pd.NA in object columns becomes
    None, as in the stored tables, which phase0 wrote with to_parquet)."""
    import io
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    return pd.read_parquet(buf)


def permitted_dtype(table: str, a: pd.DataFrame, b: pd.DataFrame) -> tuple:
    """Protocol amendment 1 (§4 C1/C2; PERMITTED_DTYPE): the one listed dtype difference. Returns (frame to compare,
    note). When the reference's column has the listed reference dtype and the new column the listed new dtype, the new
    column is cast to the reference dtype only if every value is finite and integral; the cast frame is then compared
    exactly like any other (every value, every other dtype). Anything else is returned unchanged and fails as before."""
    for t, col, ref_dt, new_dt in PERMITTED_DTYPE:
        if t != table or col not in a or col not in b or str(b[col].dtype) != ref_dt or str(a[col].dtype) != new_dt:
            continue
        x = a[col].to_numpy(float)
        if not (np.isfinite(x).all() and (x == np.round(x)).all()) or \
                not a[col].astype(ref_dt).astype(new_dt).equals(a[col]):     # Neon: no overflow on the cast
            return a, f"{table}.{col}: {new_dt} values are not all finite integers; the permitted cast is not applied"
        return a.assign(**{col: a[col].astype(ref_dt)}), (
            f"{table}.{col}: permitted dtype difference ({new_dt} new, {ref_dt} stored; amendment 1); every value "
            f"finite and integral, cast, then compared exactly")
    return a, None


def tables_compare(new: tuple, ref: tuple, label: str) -> list:
    """Legacy columns only (the reference's columns, in order), after a Parquet round trip of the new build;
    dugout_level is checked separately. The single permitted dtype difference (amendment 1) is applied first."""
    new = tuple(roundtrip(x) for x in new)
    rows = []
    for name, keys, a, b in (("pitches", KEYS, new[0], ref[0]), ("PAs", ["pa_id"], new[1], ref[1]),
                             ("support", arsenal.SUPPORT_KEY, new[2], ref[2])):
        a = a[list(b.columns)].reset_index(drop=True)
        a, note = permitted_dtype(name, a, b)
        r = frames_compare(a, b.reset_index(drop=True), keys)
        rows.append({"check": f"{label}: {name} (legacy columns, exact)", **r, **({"permitted": note} if note else {})})
    return rows


def level_check(pitches: pd.DataFrame, flags: dict) -> dict:
    """dugout_level against the flag table, computed independently of build_tables' order of operations."""
    want = dugout_columns(pitches, flags)[1]
    ok = bool(pitches["dugout_level"].reset_index(drop=True).equals(want.reset_index(drop=True)))
    counts = pitches["dugout_level"].value_counts(dropna=False).to_dict()
    return {"check": "dugout_level against the flag table", "identical": ok,
            "detail": json.dumps({str(k): int(v) for k, v in counts.items()})}


# ===================================================================================== halves and groups (§3, C3)

def season_halves(pitches: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Random halves (validation protocol §3, seed [20261103, season]) and calendar halves (median game date)."""
    halves = arsenal.assign_folds(pitches, n_folds=2, seed=vp.SEED).rename("half")
    games = pitches.drop_duplicates("game_pk")[["game_pk", "season", "game_date"]].set_index("game_pk")
    med = games.groupby("season")["game_date"].transform(lambda d: d.sort_values().iloc[(len(d) - 1) // 2])
    calendar = (games["game_date"] >= med).astype("int8").rename("calendar_half")
    return halves, calendar


def halves_structure(pitches: pd.DataFrame, halves: pd.Series) -> dict:
    games = set(pitches["game_pk"].unique())
    h0, h1 = set(halves.index[halves == 0]), set(halves.index[halves == 1])
    folds_ok = {}
    for h in (0, 1):
        hp = pitches[pitches["game_pk"].isin(h0 if h == 0 else h1)]
        f = arsenal.assign_folds(hp, seed=vp.SEED + 10 + h)
        folds_ok[h] = sorted(int(x) for x in set(f.values))
    ok = not (h0 & h1) and (h0 | h1) == games and all(v == [0, 1, 2] for v in folds_ok.values())
    return {"ok": ok, "disjoint": not (h0 & h1), "union_is_all_games": (h0 | h1) == games, "folds_present": folds_ok,
            "games": len(games), "half_sizes": [len(h0), len(h1)]}


def primary_and_movers(scores: pd.DataFrame, pair: tuple) -> tuple[dict, pd.DataFrame]:
    """Primary teams per season from scorable-pitch counts (no score column read) and the pair's mover groups."""
    prim = {}
    for season in pair:
        c = scores[scores["season"] == season].groupby(["catcher", "fielding_team"]).agg(
            pitches=("game_pk", "size"), games=("game_pk", "nunique")).reset_index()
        prim[season] = vp.primary_teams(c)
    old = vp.SEASONS
    try:
        vp.SEASONS = tuple(pair)
        movers = vp.mover_groups(prim)
    finally:
        vp.SEASONS = old
    return prim, movers


# ===================================================================================== analyses (shared by C7 and the seasons)

def with_ab(scores: pd.DataFrame, pitches_path: Path) -> pd.DataFrame:
    ab = pd.read_parquet(pitches_path, columns=["season", "pa_id", "pitch_number", "at_bat_number"])
    return scores.merge(ab, on=["season", "pa_id", "pitch_number"], how="left", validate="one_to_one")


def pair_analyses(scores: pd.DataFrame, pair: tuple, halves, calendar, runs, movers, prim) -> dict:
    """The validation protocol's analyses for a season pair, through validation_pilot with SEASONS = pair for the
    duration of the call (one binding per process; restored after)."""
    old_seasons, old_scores = vp.SEASONS, vp._DATA.get("scores")
    try:
        vp.SEASONS = tuple(pair)
        vp._DATA["scores"] = scores
        res = {}
        if halves is not None:
            res["split_random"] = vp.within_split(scores, halves, "split of existing scores (random halves)")
            res["split_calendar"] = vp.within_split(scores, calendar, "split of existing scores (calendar halves; secondary)")
        if runs is not None:
            res["refit"] = vp.within_refit_compare(runs)
        fits, sets = vp.season_fits(), vp.season_sets()
        res["sets"] = {k: {"qual": len(v["qual"]), "main": len(v["main"])} for k, v in sets.items()}
        res["repeatability"] = vp.repeatability(fits, sets, movers)
        res["sensitivity"] = vp.sensitivity(fits, sets)
        res["team_staff"] = vp.team_staff(fits, sets, prim)
        res["season_sd"] = pd.DataFrame([{"season": se, "score": sc, "model": "fixed" if fx else "random", "sd": f["sd"],
                                          "zero": f["zero"], "converged": f["converged"]}
                                         for se, ff in fits.items() for (sc, fx), f in ff.items()])
        res["spread"] = spread_table(fits, sets, scores)
        res["_fits"], res["_sets"] = fits, sets
        return res
    finally:
        vp.SEASONS = old_seasons
        if old_scores is None:
            vp._DATA.pop("scores", None)
        else:
            vp._DATA["scores"] = old_scores


def spread_table(fits: dict, sets: dict, scores: pd.DataFrame) -> pd.DataFrame:
    """A3 (§6 row 4): (a) REML catcher SD with flags; (b) empirical SD (ddof 1) of shrunken estimates over qualifying
    main-group catchers; (c) pitch-level SD of the score x 100 (runs per 100 pitches, ddof 1) over scored pitches."""
    rows = []
    for season, ff in fits.items():
        ids = sorted(sets[season]["qual"] & sets[season]["main"])
        s = scores[scores["season"] == season]
        for (score, fixed), f in ff.items():
            est = f["saved"].reindex(ids)
            rows.append({"season": season, "score": score, "model": "fixed" if fixed else "random",
                         "reml_sd": f["sd"], "converged": f["converged"], "zero": f["zero"],
                         "n_catchers": len(ids), "shrunken_sd": empirical_sd(est.to_numpy(float)),
                         "pitch_sd_per100": empirical_sd(100.0 * s[score].to_numpy(float)), "n_pitches": len(s)})
    return pd.DataFrame(rows)


def empirical_sd(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(np.std(x, ddof=1)) if len(x) >= 2 else float("nan")


def r3b_sensitivity(scores: pd.DataFrame, pitches_path: Path, fits: dict, sets: dict, pair: tuple) -> pd.DataFrame:
    """R3b's table only (C7 compares it with R3 on the stored inputs); r3b_full keeps the excluded list as well."""
    return r3b_full(scores, pitches_path, fits, sets, pair)["table"]


def r3b_full(scores: pd.DataFrame, pitches_path: Path, fits: dict, sets: dict, pair: tuple) -> dict:
    """R3b (§2): vp.sensitivity on the scores with dugout_called replaced by 'any flag level', keeping only its R3 rows,
    relabeled, and its excluded-catcher list for those rows (Neon's note: R3b's exclusions are reported, not dropped).
    The level comes from the season's pitch table (scores.parquet carries only dugout_called)."""
    if "dugout_level" in pq.read_schema(pitches_path).names:
        lv = pd.read_parquet(pitches_path, columns=["season", "pa_id", "pitch_number", "dugout_level"])
    else:   # the stored 2024-2025 tables predate the column: the flag table on the same keyed pitch rows
        lv = pd.read_parquet(pitches_path, columns=["season", "pa_id", "pitch_number", "fielding_team", "game_date"])
        lv = lv.assign(dugout_level=dugout_columns(lv, load_flags())[1].to_numpy())[
            ["season", "pa_id", "pitch_number", "dugout_level"]]
    s = scores.merge(lv, on=["season", "pa_id", "pitch_number"], how="left", validate="one_to_one", indicator=True)
    if not (s["_merge"] == "both").all():
        raise ValueError(f"R3b: {int((s['_merge'] != 'both').sum())} scored pitches have no pitch-table row")
    s = s.assign(dugout_called=s["dugout_level"].notna()).drop(columns=["dugout_level", "_merge"])
    old_seasons, old_scores = vp.SEASONS, vp._DATA.get("scores")
    try:
        vp.SEASONS = tuple(pair)
        vp._DATA["scores"] = s
        sens = vp.sensitivity(fits, sets)
    finally:
        vp.SEASONS = old_seasons
        if old_scores is None:
            vp._DATA.pop("scores", None)
        else:
            vp._DATA["scores"] = old_scores
    relabel = {"R3_D": "R3b_D", "R3_A": "R3b_A"}
    t = sens["table"][sens["table"]["check"].isin(list(relabel))].copy()
    t["check"] = t["check"].map(relabel)
    ex = sens["excluded"]
    ex = ex[ex["check"].isin(list(relabel))].copy() if len(ex) else ex.copy()
    if len(ex):
        ex["check"] = ex["check"].map(relabel)
    return {"table": t.reset_index(drop=True), "excluded": ex.reset_index(drop=True)}


def forward_matched_table(fb: pd.DataFrame, fc: pd.DataFrame) -> pd.DataFrame:
    """research_s8_full.prediction_tables' forward part on explicit frames (P0 fb, C1 fc), S9 outcome-based seen mask."""
    rows = []
    for kind in ("outcome", "contact"):
        b, c = fb[kind], fc[kind]
        m = research.matched(b, c)
        seen = m.merge(c[KEYS + ["seen_cell"]], on=KEYS, how="left", validate="one_to_one")["seen_cell"].to_numpy(bool)
        seen_p0 = m.merge(b[KEYS + ["seen_cell"]], on=KEYS, how="left", validate="one_to_one")["seen_cell"].to_numpy(bool)
        for name, g in (("all", m), ("seen cells", m[seen]), ("unseen cells", m[~seen])):
            rows.append({"model": kind, "sample": name, "n": len(g), "P0": -g["lp_base"].mean(),
                         "C1": -g["lp_cand"].mean(), "gain": (g["lp_cand"] - g["lp_base"]).mean(),
                         "seen_flags_equal": bool(np.array_equal(seen, seen_p0))})
        rows[-3]["max_cal_P0"] = research.max_calibration(b, m[KEYS])
        rows[-3]["max_cal_C1"] = research.max_calibration(c, m[KEYS])
    return pd.DataFrame(rows)


def forward_baseline_table(models_: dict, train: pd.DataFrame, test: pd.DataFrame, mask: str) -> pd.DataFrame:
    """validation_pilot.forward_check's loop on given fitted models. mask: 'component' (each component's own training
    pairs, the historical definition) or 'outcome' (outcome-training pairs for both, the S9 definition)."""
    rows = []
    outcome_pairs = set(map(tuple, models.outcome_training(train)[["pitcher", "action"]].drop_duplicates().to_numpy()))
    for kind in ("outcome", "contact"):
        if kind == "outcome":
            tr, obs = models.outcome_training(train), models.outcome_training(test)
            K = models.N_OUTCOMES
            base = validation._league_baseline(tr, ["cidx", "action"], ["cidx"], K)
            keys, names = ["cidx", "action"], list(solver.OUTCOMES)
        else:
            tr, obs = models.contact_training(train), models.contact_training(test)
            K = models.N_BINS
            n = models._counts(tr, ["action"], K)
            base = (n + n.sum(axis=0) / n.to_numpy().sum()).div(n.sum(axis=1) + 1, axis=0)
            keys, names = ["action"], [f"bin_{b + 1}" for b in range(K)]
        model = models_[kind]
        pred, league = model.predict(obs), model.league_mean(obs)
        bp = models._lookup(base, obs, keys, fill=np.nan)
        overall = models._counts(tr.assign(_all=0), ["_all"], K).to_numpy()[0]
        bp = np.where(np.isnan(bp), overall / overall.sum(), bp)
        pairs = set(map(tuple, tr[["pitcher", "action"]].drop_duplicates().to_numpy())) if mask == "component" \
            else outcome_pairs
        seen = np.array([(a, b) in pairs for a, b in obs[["pitcher", "action"]].to_numpy()])
        y = obs["y"].to_numpy()
        for subset, m in (("all", np.ones(len(y), bool)), ("seen_cells", seen), ("unseen_cells", ~seen)):
            if m.sum() == 0:
                continue
            i = np.arange(m.sum())
            yy = y[m]
            row = {"model": kind, "subset": subset, "n": int(m.sum()), "share": float(m.mean()),
                   "log_loss": -np.log(pred[m][i, yy]).mean(), "baseline_log_loss": -np.log(bp[m][i, yy]).mean(),
                   "league_pooled_log_loss": -np.log(league[m][i, yy]).mean()}
            for j, nm in enumerate(names):
                row[f"cal_{nm}"] = validation.calibration_error(pred[m][:, j], yy == j)
            row["max_cal"] = max(row[f"cal_{nm}"] for nm in names)
            rows.append(row)
    return pd.DataFrame(rows)


def query_frame(name, kind, model, rows, spec) -> tuple:
    """Copied from freeze_forward_2025.py: the stored query frame and its input hash."""
    obs = research._training(kind, rows, spec)
    f = research._pred_frame(rows, obs, model.predict(obs), fold=-1)
    f.insert(0, "component", kind)
    f.insert(0, "model", name)
    return f, query_hash(rows, obs)


def query_hash(rows: pd.DataFrame, obs: pd.DataFrame) -> str:
    h = hashlib.sha256()
    keys = rows.loc[obs.index, KEYS].reset_index(drop=True)
    for frame in (keys, obs.reset_index(drop=True)):
        h.update(json.dumps([[c, str(t)] for c, t in frame.dtypes.items()]).encode())
        h.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return h.hexdigest()


def reconstruction_check(rows: pd.DataFrame, scores: pd.DataFrame) -> dict:
    """§6 post-run check, coverage first (Astra's correction): exactly one finite action-value row for every scored
    (pitch, actual action) and (pitch, recommended action); then the recommendation (first minimum of Q_select in slot
    order), Q_actual and D compared exactly."""
    K = bs.PITCH_KEYS
    dup = int(rows.duplicated(K + ["action"]).sum())
    r = rows[K + ["action", "Q_value"]]
    a = scores[K + ["action", "rec_action", "Q_actual", "D"]].reset_index(drop=True)
    act = a.merge(r, on=K + ["action"], how="left", validate="many_to_one" if not dup else None)
    rec = a.merge(r.rename(columns={"action": "rec_action", "Q_value": "Q_rec"}), on=K + ["rec_action"], how="left",
                  validate="many_to_one" if not dup else None)
    missing_act = int(act["Q_value"].isna().sum()) if len(act) == len(a) else -1
    missing_rec = int(rec["Q_rec"].isna().sum()) if len(rec) == len(a) else -1
    finite = bool(np.isfinite(act["Q_value"]).all() and np.isfinite(rec["Q_rec"]).all()) if missing_act == 0 and \
        missing_rec == 0 else False
    out = {"pitches": len(a), "duplicate_rows": dup, "missing_actual_rows": missing_act, "missing_rec_rows": missing_rec,
           "all_finite": finite}
    if dup or missing_act or missing_rec or not finite:
        out["ok"] = False
        return out
    first = rows.loc[rows.groupby(K, sort=False)["Q_select"].idxmin(), K + ["action"]].rename(columns={"action": "rec_rows"})
    m = a.merge(first, on=K, how="left", validate="one_to_one")
    out["rec_differs"] = int((m["rec_action"] != m["rec_rows"]).sum())
    out["Q_actual_differs"] = int((a["Q_actual"].to_numpy() != act["Q_value"].to_numpy()).sum())
    out["D_differs"] = int((a["D"].to_numpy() != (act["Q_value"].to_numpy() - rec["Q_rec"].to_numpy())).sum())
    out["ok"] = out["rec_differs"] == 0 and out["Q_actual_differs"] == 0 and out["D_differs"] == 0
    return out


# ===================================================================================== records and the guard

def record_path(stage: str) -> Path:
    return RECORDS / f"s9_{stage}.json"


def reserve(stage: str) -> Path:
    path = record_path(stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w") as f:
        f.write(jdump({"stage": stage, "status": "reserved", "started": now(), "mode": MODE, "smoke_dir": SMOKE_DIR}))
    return path


def finalize(path: Path, rec: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(jdump(rec))
    os.replace(tmp, path)


def load_record(stage: str) -> dict:
    return json.loads(record_path(stage).read_text())


def read_record(path: Path):
    """(record, problem): a missing, unreadable or non-object record is a problem, never an exception."""
    try:
        r = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None, "no record"
    except (ValueError, OSError) as exc:
        return None, f"unreadable record: {type(exc).__name__}"
    return (r, None) if isinstance(r, dict) else (None, "record is not a JSON object")


def identity() -> dict:
    """What every record is bound to: this script revision, the locked protocol, the flag table and the package."""
    return {"script_sha256": sha(SCRIPT), "protocol_sha256": sha(PROTOCOL), "flags_sha256": sha(FLAGS),
            "gamecalling_sha256": research.code_state()["gamecalling_sha256"], "mode": MODE, "smoke_dir": SMOKE_DIR}


def expected_predecessors(stage: str):
    """The predecessor records each stage must name (and only those). Phase 2 season stages are <name>_<year>; each
    names the season's authorization record and its immediate predecessor (extract names the whole development chain)."""
    if stage in EXPECTED_PREDECESSORS:
        return EXPECTED_PREDECESSORS[stage]
    if stage.startswith("gate_v") or stage == "domains":
        return {"guard"}
    name, _, y = stage.rpartition("_")
    if not y.isdigit():
        return None
    a = f"authorize_{y}"
    return {"authorize": {"guard"},
            "extract": {"guard", "audit", "control", f"gate_v{MAPPING_VERSION}", "domains", a},
            "build": {"guard", a, f"extract_{y}"},
            "checks": {"guard", a, f"build_{y}", "domains", f"gate_v{MAPPING_VERSION}"},
            "run": {"guard", a, f"checks_{y}"},
            "halves": {"guard", a, f"run_{y}"},
            "validate": {"guard", a, f"run_{y}", f"halves_{y}"},
            "report": {"guard", a, f"checks_{y}", f"run_{y}", f"validate_{y}"}}.get(name)


def record_problems(stage: str, rec: dict, ident: dict) -> list:
    """A predecessor record must be this stage's, finished and passed, in this mode, under this identity (or an
    explicitly approved earlier script revision), and its own predecessors must still match their recorded hashes."""
    p = []
    if rec.get("stage") != stage:
        p.append(f"record names stage {rec.get('stage')!r}")
    if rec.get("status") != "finished" or rec.get("passed") is not True:
        p.append("not a finished, passed record")
    rid = rec.get("identity")
    if not isinstance(rid, dict):
        return p + ["identity missing or not an object"]
    if rid.get("mode") != ident["mode"] or rid.get("smoke_dir") != ident["smoke_dir"]:
        p.append("other mode or smoke directory")
    if rid.get("script_sha256") != ident["script_sha256"] and rid.get("script_sha256") not in APPROVED_PREDECESSOR_SCRIPTS:
        p.append("written by another script revision")
    for k in ("protocol_sha256", "flags_sha256", "gamecalling_sha256"):
        if rid.get(k) != ident[k]:
            p.append(f"{k} differs")
    want = expected_predecessors(stage)
    preds = rec.get("predecessors")
    if not isinstance(preds, dict):
        p.append("predecessors missing")
        preds = {}
    elif want is not None and set(preds) != want:
        p.append(f"predecessors {sorted(preds)} are not the expected {sorted(want)}")
    for name, h in preds.items():
        q = record_path(name)
        if not q.exists() or sha(q) != h:
            p.append(f"its predecessor {name} changed")
    if stage == "guard" and not isinstance(rec.get("inventory"), dict):
        p.append("guard record has no inventory")
    return p


def inventory() -> list:
    """The protected inventory: both freeze records and every file they list, both archives, the raw Parquets, the
    stored 2024-2025 tables, scores, validation outputs, the forward-model files, the locked protocol and the flag table."""
    files = {FROZEN_RECORD, FORWARD_RECORD, RAW_2015_2023, *ORIG.values(), D / "phase1_scores.parquet", PROTOCOL, FLAGS}
    fz = json.loads(FROZEN_RECORD.read_text())
    for v in fz["artifacts"].values():
        if isinstance(v, list):
            files |= {ROOT / e["path"] for e in v if isinstance(e, dict) and "path" in e}
    fw = json.loads(FORWARD_RECORD.read_text())
    for a in fw["artifacts"]:
        for k in ("pickle", "queries"):
            if isinstance(a.get(k), dict):
                files.add(ROOT / a[k]["path"])
    files.add(ROOT / fw["manifest"]["path"])
    files |= {ROOT / r["path"] for r in fw["stage_records"].values()}
    files |= {ROOT / v["copy"] for v in fw["archive"].values()}
    for arch in ("C1_freeze_2026-11-18", "C1_forward_2025_2026-11-19"):
        files |= {p for p in (ROOT / "Sessions" / "Archive" / arch).rglob("*") if p.is_file()}
    files |= {p for p in (D / "validation_pilot" / "v1").rglob("*") if p.is_file()} | {STORED_REF["s8"], STORED_REF["C1_halves"]}
    files.add(ROOT / "reports" / "phase1_calibration.csv")
    return sorted(str(p.relative_to(ROOT)) for p in files)


def take_inventory() -> dict:
    return {p: sha(ROOT / p) for p in inventory()}


def guard_compare() -> dict:
    g, err = read_record(record_path("guard"))
    if err:
        return {"ok": False, "changed": [], "added_to_inventory": [], "files": 0, "detail": f"guard: {err}"}
    if g.get("status") != "finished" or g.get("passed") is not True or not isinstance(g.get("inventory"), dict):
        return {"ok": False, "changed": [], "added_to_inventory": [], "files": 0,
                "detail": f"guard record unusable: status={g.get('status')} passed={g.get('passed')}"}
    cur = take_inventory()
    changed = sorted(p for p in g["inventory"] if cur.get(p) != g["inventory"][p])
    added = sorted(set(cur) - set(g["inventory"]))
    return {"ok": not changed and not added, "changed": changed, "added_to_inventory": added, "files": len(cur)}


def verify_records() -> dict:
    out = {}
    for script in ("freeze_c1_record.py", "freeze_forward_record.py"):
        env = {k: v for k, v in os.environ.items() if "SMOKE" not in k and k != "S9_BIND"}
        env["PYTHONPATH"] = str(ROOT)
        r = subprocess.run([sys.executable, script, "verify"], cwd=ROOT, capture_output=True, text=True, env=env)
        out[script] = {"ok": r.returncode == 0, "tail": r.stdout.strip().splitlines()[-1:] + r.stderr.strip().splitlines()[-2:]}
    return out


def chain_problems(stage: str, path_of=None, seen=None) -> list:
    """Every predecessor digest recorded anywhere up the chain from `stage` must equal the current record file (Astra:
    a direct prerequisite's current digest alone cannot show it is the record an earlier stage pinned)."""
    path_of = path_of or record_path
    seen = set() if seen is None else seen
    rec, err = read_record(path_of(stage))
    if err:
        return [f"{stage}: {err}"]
    out = []
    preds = rec.get("predecessors")
    for name, h in (preds.items() if isinstance(preds, dict) else []):
        q = path_of(name)
        if not q.exists() or sha(q) != h:
            out.append(f"{stage} pinned {name} at another digest")
        elif name not in seen:
            seen.add(name)
            out += chain_problems(name, path_of, seen)
    return out


def preconditions(stage: str, needs=()) -> dict:
    ident = identity()
    pre = {"code": code_ok(), "script": script_state(), "records": verify_records(), "identity": ident}
    problems = []
    if not pre["code"]["ok"]:
        problems.append(f"gamecalling/ is not clean at {FULL_CODE_HASH}: {pre['code']}")
    for k, v in pre["records"].items():
        if not v["ok"]:
            problems.append(f"{k} verify failed: {v['tail']}")
    preds = {}
    for s in needs:
        p = record_path(s)
        r, err = read_record(p)
        if err:
            problems.append(f"stage {s}: {err}")
            continue
        problems += [f"stage {s}: {x}" for x in record_problems(s, r, ident)]
        problems += [f"stage {s}: {x}" for x in chain_problems(s)]
        preds[s] = sha(p)
    if problems:
        raise SystemExit("refused:\n  " + "\n  ".join(problems))
    pre["predecessors"] = preds
    return pre


def run_stage(stage: str, body, needs=("guard",)):
    """preconditions -> selftest -> guard compared -> record reserved (O_EXCL) -> body -> guard and both freeze records
    again (even on failure) -> record finalized. A failed stage exits nonzero after its record is written. Only a hard
    kill leaves the reservation."""
    if record_path(stage).exists():
        raise SystemExit(f"refused: {record_path(stage)} exists; stages run once")
    pre = preconditions(stage, needs)
    st = selftest_results()
    if not all(r["ok"] for r in st):
        raise SystemExit(f"refused: selftest failed: {[r for r in st if not r['ok']]}")
    g0 = guard_compare()
    if not g0["ok"]:
        raise SystemExit(f"refused: guard differs before {stage}: {g0}")
    path = reserve(stage)
    t0 = time.time()
    rec = {"stage": stage, "identity": pre["identity"], "predecessors": pre["predecessors"], "started": now(),
           "preconditions": pre, "selftest": {"passed": True, "cases": len(st)}, "guard_before": g0}
    try:
        rec["environment"] = environment()
        res = body()
        rec.update(result=res, passed=bool(res.get("passed")))
    except Exception as exc:  # noqa: BLE001
        import traceback
        rec.update(passed=False, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-4000:])
    except BaseException as exc:  # an interrupt: recorded, finalized below, then re-raised
        rec.update(passed=False, error=f"interrupted: {type(exc).__name__}: {exc}")
        raise
    finally:
        post_ok = True
        try:
            g1 = guard_compare()
            rec["guard_after"] = g1
            post_ok &= g1["ok"]
        except BaseException as exc:  # noqa: BLE001
            rec["post_guard_error"] = f"{type(exc).__name__}: {exc}"
            post_ok = False
        try:
            post = verify_records()
            rec["records_after"] = post
            post_ok &= all(v["ok"] for v in post.values())
        except BaseException as exc:  # noqa: BLE001
            rec["post_records_error"] = f"{type(exc).__name__}: {exc}"
            post_ok = False
        rec["passed"] = bool(rec.get("passed")) and post_ok
        rec.update(status="finished", finished=now(), seconds=time.time() - t0)
        finalize(path, rec)
        print(f"{stage}: {'PASSED' if rec['passed'] else 'FAILED'} ({rec['seconds']:.0f} s) -> {path}", flush=True)
    if not rec["passed"]:
        sys.exit(1)
    return rec


def ready_for_targets() -> dict:
    """Phase 2's gate: guard, audit, control and the gate must each have a verified record under the current identity
    (or an approved earlier script revision)."""
    ident = identity()
    out = {}
    for s in ("guard", "audit", "control", f"gate_v{MAPPING_VERSION}"):
        r, err = read_record(record_path(s))
        out[s] = [err] if err else record_problems(s, r, ident)
    if out.get("guard") == []:
        g = guard_compare()
        if not g["ok"]:
            out["guard"].append(f"inventory changed: {g['changed'][:5]} added: {g['added_to_inventory'][:5]}"
                                f" {g.get('detail', '')}")
    c = code_ok()
    out["code"] = [] if c["ok"] else [f"gamecalling not clean at the frozen digest: {c}"]
    out["freeze_records"] = [f"{k}: {v['tail']}" for k, v in verify_records().items() if not v["ok"]]
    return {"ready": all(not v for v in out.values()), "problems": out}


# ===================================================================================== stages: guard, audit

def stage_guard():
    path = record_path("guard")
    if path.exists():
        raise SystemExit(f"refused: {path} exists (the inventory is taken once)")
    pre = preconditions("guard")
    st = selftest_results()
    if not all(r["ok"] for r in st):
        raise SystemExit(f"refused: selftest failed: {[r for r in st if not r['ok']]}")
    p = reserve("guard")
    rec = {"stage": "guard", "identity": pre["identity"], "predecessors": {}, "started": now(), "preconditions": pre}
    try:
        rec["inventory"] = take_inventory()
        rec["passed"] = True
    except Exception as exc:  # noqa: BLE001
        rec.update(passed=False, error=f"{type(exc).__name__}: {exc}")
    except BaseException as exc:  # an interrupt: finalized as failed, then re-raised
        rec.update(passed=False, error=f"interrupted: {type(exc).__name__}: {exc}")
        raise
    finally:
        rec.update(status="finished", finished=now())
        rec.setdefault("passed", False)
        finalize(p, rec)
        print(f"guard: {'PASSED' if rec['passed'] else 'FAILED'}, {len(rec.get('inventory', {}))} files -> {p}")
    if not rec["passed"]:
        sys.exit(1)


RAW_BAT_TRACKING = ("bat_speed", "swing_length", "attack_angle")


def stage_audit():
    def body():
        res = {"checks": []}
        a, b = pq.read_schema(ORIG["raw"]), pq.read_schema(RAW_2015_2023)
        need = list(dict.fromkeys(loader.RAW_COLUMNS + research_s7.RAW_COLUMNS))
        types = {c: (str(a.field(c).type) if c in a.names else None, str(b.field(c).type) if c in b.names else None)
                 for c in need}
        bad = {c: t for c, t in types.items() if t[0] is None or t[0] != t[1]}
        res["checks"].append({"check": f"{len(need)} required raw columns present with equal types", "ok": not bad,
                              "detail": bad or len(need)})
        res["checks"].append({"check": "both raw files have the same column set", "ok": set(a.names) == set(b.names),
                              "detail": f"{len(a.names)} / {len(b.names)}"})
        src = "".join(p.read_text() for p in sorted((ROOT / "gamecalling").glob("*.py")))
        used = [c for c in RAW_BAT_TRACKING + ("miss_distance",) if c in src]
        res["checks"].append({"check": "no bat-tracking or miss_distance column used in gamecalling/", "ok": not used,
                              "detail": used})
        res["checks"].append({"check": "config.SEASONS unchanged", "ok": tuple(config.SEASONS) == DEV_SEASONS})
        flags = load_flags()
        res["checks"].append({"check": "dugout flag table v1 readable with sources", "ok":
                              all(e.get("source") for e in flags["entries"]), "detail": flags["entries"]})
        res["checks"].append({"check": "description domain = OUTCOME | AUTOMATIC (15 labels)",
                              "ok": len(set(loader.OUTCOME) | loader.AUTOMATIC) == 15})
        res["passed"] = all(c["ok"] for c in res["checks"])
        return res
    run_stage("audit", body)


# ===================================================================================== the control (§4)

def smoke_excused(v: dict) -> tuple:
    """Smoke only. A season's single-season context is excused when it exited 0, its worker probe passed, it has checks,
    and every failure is waivable by name AND by kind: a value difference ("differ: ..."), records differing in content,
    or changed fit-id digests. Never columns, row counts, duplicate keys or fit-id membership."""
    fails = [c for c in v.get("checks", []) if not c.get("identical")]
    failing = [c["check"] for c in fails]
    structural = [c["check"] for c in fails
                  if not any(w in c["check"] for w in SMOKE_WAIVABLE)
                  or not (str(c.get("detail", "")).startswith("differ: ")
                          or "records only in the new run" in str(c.get("detail", ""))
                          or c["check"].endswith("fit-id hashes (run label dropped, model id kept)"))]
    ok = v.get("returncode") == 0 and v.get("probe", {}).get("ok") is True and bool(v.get("checks")) and not structural
    return ok, failing, structural


def ctl_dir(*parts) -> Path:
    return CONTROL.joinpath(*map(str, parts))


def refs() -> dict:
    """Production: the stored references. Smoke: the subset references rebuilt by the smoke_ref context."""
    if not SMOKE:
        return {k: str(v) for k, v in STORED_REF.items()}
    r = ctl_dir("smoke_ref")
    return {"P0_dir": str(r / "models" / "P0_ref"), "C1_dir": str(r / "models" / "C1_ref"),
            "C1_actions": str(r / "models" / "C1" / "runs" / "s9ref"), "P0_halves": str(r / "halves_P0.pkl"),
            "C1_halves": str(r / "halves_C1.pkl"), "v0": str(r / "v0.pkl"), "v2": None, "s8": None}


def run_child(context: str, binding: dict) -> dict:
    """One context in its own process with its own S9_BIND; it writes result.json in its output directory."""
    out = Path(binding["out"])
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["S9_BIND"] = json.dumps(binding)
    env["PYTHONPATH"] = str(ROOT)
    t = time.time()
    r = subprocess.run([sys.executable, str(SCRIPT), "_ctx", context], cwd=ROOT, env=env, capture_output=True, text=True)
    (out / "stdout.txt").write_text(r.stdout)
    (out / "stderr.txt").write_text(r.stderr)
    res_path = out / "result.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {"passed": False}
    res.update(context=context, returncode=r.returncode, seconds=time.time() - t)
    if r.returncode != 0:
        res["passed"] = False
        res["stderr_tail"] = r.stderr.strip().splitlines()[-15:]
    print(f"  context {context}: {'passed' if res.get('passed') else 'FAILED'} ({res['seconds']:.0f} s)", flush=True)
    return res


def binding_for(tables: dict, raw: str, out: Path, **extra) -> dict:
    return {"raw": raw, "pitches": tables["pitches"], "pas": tables["pas"], "support": tables["support"],
            "models_dir": str(out / "models"), "out": str(out), "refs": refs(), **extra}


def stage_control():
    def body():
        flags = load_flags()
        checks, ctx = [], {}
        CONTROL.mkdir(parents=True, exist_ok=True)

        # C1: joint build
        t = time.time()
        joint_new = build_tables(read_raw_rows(ORIG["raw"]), flags)
        if SMOKE:
            # an independent reference: phase0.main's steps on the same subset, without the flag override
            raw_s = read_raw_rows(ORIG["raw"])
            rp = loader.build_pitches(raw_s)
            rp = rp.join(arsenal.assign_folds(rp), on="game_pk")
            rs = arsenal.support_table(rp)
            rp = arsenal.mark_arsenal(rp, arsenal.eligible_sets(rs))
            joint_ref_paths = save_tables(ctl_dir("tables", "joint_ref"), rp, loader.build_plate_appearances(rp), rs)
            joint_ref = tuple(pd.read_parquet(joint_ref_paths[k]) for k in ("pitches", "pas", "support"))
            jr = ctl_dir("tables", "joint_raw.parquet")
            t_raw = pq.read_table(ORIG["raw"])
            df_raw = t_raw.to_pandas()
            pq.write_table(t_raw.from_pandas(df_raw[df_raw["game_pk"] % 5 == 0].reset_index(drop=True),
                                             schema=t_raw.schema, preserve_index=False), jr)
            del t_raw, df_raw
        else:
            joint_ref = tuple(pd.read_parquet(ORIG[k]) for k in ("pitches", "pas", "support"))
            joint_ref_paths = {k: str(ORIG[k]) for k in ("pitches", "pas", "support")}
        checks += tables_compare(joint_new, joint_ref, "C1 joint build")
        checks.append(level_check(joint_new[0], flags))
        c1_ok = all(c["identical"] for c in checks)
        if not c1_ok:
            return {"passed": False, "checks": checks, "contexts": ctx, "stopped": "C1 differs: diagnosis needed"}
        print(f"  C1 done ({time.time() - t:.0f} s)", flush=True)

        # C2: single-season builds (exact gate) and raw extracts
        single = {}
        for season in DEV_SEASONS:
            raw_path = ctl_dir("tables", season, "raw.parquet")
            ext = write_raw_extract(ORIG["raw"], raw_path, season)
            new = build_tables(read_raw_rows(ORIG["raw"], season), flags)
            ref = tuple(x[x["season"] == season] for x in joint_ref)
            checks += tables_compare(new, ref, f"C2 single-season build {season}")
            checks.append({**level_check(new[0], flags), "check": f"C2 {season}: dugout_level against the flag table"})
            single[season] = {"tables": save_tables(ctl_dir("tables", season), *new), "raw": ext}
        print(f"  C2 done ({time.time() - t:.0f} s)", flush=True)
        smoke_expected = set()
        if SMOKE:
            for c in checks:
                if c["identical"] or not c["check"].startswith("C2 single-season build") or ": pitches" not in c["check"]:
                    continue
                cols = set(c["detail"].removeprefix("differ: ").split(", ")) if c["detail"].startswith("differ: ") else {"?"}
                if cols <= SMOKE_ALLOWED_COLS and "position_player_pitching" in cols:
                    season = int(c["check"].split()[3].rstrip(":"))
                    smoke_expected.add(season)
                    c.update(identical=True, smoke_expected_difference=sorted(cols),
                             smoke_note="the subset changes the position-player flag; only its consequences differ")
        c2_ok = all(c["identical"] for c in checks if c["check"].startswith("C2"))

        # C3: assignments and groups
        v0_ref = None
        halves_new, calendar_new = {}, {}
        for season in DEV_SEASONS:
            p = pd.read_parquet(single[season]["tables"]["pitches"])
            h, c = season_halves(p)
            halves_new[season], calendar_new[season] = h, c
            st = halves_structure(p, h)
            checks.append({"check": f"C3 {season}: halves disjoint, union all games, within-half folds populated",
                           "identical": st["ok"], "detail": json.dumps(st)})
        halves_all = pd.concat([halves_new[s] for s in DEV_SEASONS]).sort_index()
        calendar_all = pd.concat([calendar_new[s] for s in DEV_SEASONS]).sort_index()
        with open(ctl_dir("halves_dev.pkl"), "wb") as f:
            pickle.dump({"halves": halves_all, "calendar": calendar_all}, f)
        if SMOKE:
            jp = joint_ref[0]
            v0_ref = {"halves": arsenal.assign_folds(jp, n_folds=2, seed=vp.SEED).rename("half"),
                      "calendar": season_halves(jp)[1]}
        else:
            with open(STORED_REF["v0"], "rb") as f:
                v0_ref = pickle.load(f)
        hv = v0_ref["halves"].reindex(halves_all.index)
        checks.append({"check": "C3 random halves equal v0's, game by game", "identical":
                       bool(hv.notna().all() and (hv.astype(int).to_numpy() == halves_all.astype(int).to_numpy()).all())})
        cv = v0_ref["calendar"].reindex(calendar_all.index)
        checks.append({"check": "C3 calendar halves equal v0's, game by game", "identical":
                       bool(cv.notna().all() and (cv.astype(int).to_numpy() == calendar_all.astype(int).to_numpy()).all())})

        # Smoke only: the subset references through the package's joint path
        if SMOKE:
            ctx["smoke_ref"] = run_child("smoke_ref", binding_for(joint_ref_paths, str(ctl_dir("tables", "joint_raw.parquet")),
                                                                  ctl_dir("smoke_ref")))
            if not ctx["smoke_ref"].get("passed"):
                return {"passed": False, "checks": checks, "contexts": ctx}

        # C3 groups (needs P0 reference scores)
        R = refs()
        p0_scores = pd.read_parquet(Path(R["P0_dir"]) / "scores.parquet",
                                    columns=["season", "catcher", "fielding_team", "game_pk"])
        prim, movers = primary_and_movers(p0_scores, DEV_SEASONS)
        if SMOKE:
            with open(R["v0"], "rb") as f:
                v0_groups = pickle.load(f)
        else:
            v0_groups = v0_ref
        diffs = deep_compare({"primary": prim, "movers": movers}, {"primary": v0_groups["primary"],
                                                                   "movers": v0_groups["movers"]}, projection=False)
        checks.append({"check": "C3 primary teams and mover groups (explicit pair) equal v0's", "identical": not diffs,
                       "detail": diffs[:5]})

        if not c2_ok:
            return {"passed": False, "checks": checks, "contexts": ctx, "stopped": "C2 differs: user decision (D3)"}

        # C4: single-season end to end
        for season in DEV_SEASONS:
            ctx[f"single_{season}"] = run_child("single", binding_for(single[season]["tables"], single[season]["raw"]["path"],
                                                                      ctl_dir(f"single_{season}"), season=season))
        # C5: joint historical path (stored tables; smoke: the subset joint tables)
        joint_raw = str(ctl_dir("tables", "joint_raw.parquet")) if SMOKE else str(ORIG["raw"])
        ctx["joint"] = run_child("joint", binding_for(joint_ref_paths, joint_raw, ctl_dir("joint")))
        # C6: real halves refit, 2024, through the single-season files
        ctx["halves_2024"] = run_child("halves", binding_for(single[2024]["tables"], single[2024]["raw"]["path"],
                                                             ctl_dir("halves_2024"), season=2024,
                                                             halves=str(ctl_dir("halves_dev.pkl"))))
        # C7: analysis replays on the stored inputs
        for version in ("P0", "C1"):
            ctx[f"replay_{version}"] = run_child("replay", binding_for(joint_ref_paths, joint_raw, ctl_dir(f"replay_{version}"),
                                                                       version=version, vp_seasons=list(DEV_SEASONS),
                                                                       halves=str(ctl_dir("halves_dev.pkl"))))
        # C8 and C8b: the forward route on development data
        ctx["forward_route"] = run_child("forward", binding_for(joint_ref_paths, joint_raw, ctl_dir("forward_route")))
        if not SMOKE:
            same = sha(D / "phase1_scores.parquet") == sha(STORED_REF["P0_dir"] / "scores.parquet")
            checks.append({"check": "phase1_scores.parquet and P0_repro/scores.parquet are the same bytes",
                           "identical": same})
        def ctx_ok(k, v):
            if v.get("passed"):
                return True
            if not (SMOKE and k.startswith("single_") and int(k.split("_")[1]) in smoke_expected):
                return False
            ok, failing, structural = smoke_excused(v)
            v["smoke_note"] = ("expected in smoke: only numerical comparisons differ through the position-player flag"
                               if ok else f"not excused: structural failures {structural}")
            v["smoke_waived"] = failing if ok else []
            return ok
        passed = all(c["identical"] for c in checks) and all(ctx_ok(k, v) for k, v in ctx.items())
        return {"passed": passed, "checks": checks, "contexts": ctx, "single": single,
                "smoke_expected_seasons": sorted(smoke_expected),
                "note": "smoke: a serialization and plumbing check, not evidence" if SMOKE else None}
    run_stage("control", body, needs=("guard", "audit"))


# ===================================================================================== contexts (child processes)

def ctx_result(res: dict, strict: bool = False) -> None:
    """A child's result. strict (the season contexts, revision 9): non-finite floats labeled (json_safe) and a bare
    NaN/Infinity refused, so the record parses with a strict reader. The phase-1 control contexts keep revision 7's
    serialization unchanged."""
    text = (json.dumps(json_safe(res), indent=1, default=str, sort_keys=True, allow_nan=False) if strict
            else jdump(res))
    Path(BIND["out"], "result.json").write_text(text)


def c1_spec():
    """research_s8_full.C1 with extra_inputs bound to this process's raw file (the imported spec captured the original
    path at import, before the binding)."""
    return replace(s8f.C1, extra_inputs=(config.RAW_PARQUET,))


def run_version(version: str, seasons: tuple, out_name: str, label: str, extra_inputs=None):
    if version == "P0":
        spec = replace(research.P0, seasons=tuple(seasons))
    else:
        spec = replace(s8f.C1, seasons=tuple(seasons), extra_inputs=tuple(extra_inputs or (config.RAW_PARQUET,)))
    if (research.MODELS_DIR / out_name).exists():
        raise FileExistsError(f"{research.MODELS_DIR / out_name} exists")
    return research.run_model(spec, out_name=out_name, run_label=label, workers=WORKERS), spec


def compare_run(version: str, new_dir: Path, ref_dir: Path, new_actions: Path, ref_actions: Path, season=None,
                joint=False, spec=None) -> list:
    rows = []
    sel = (lambda f: f[f["season"] == season].reset_index(drop=True)) if season is not None else (lambda f: f)
    kinds = ["scores", "heldout_outcome", "heldout_contact"] + (["forward_outcome", "forward_contact"] if joint else [])
    for what in kinds:
        a, b = pd.read_parquet(new_dir / f"{what}.parquet"), sel(pd.read_parquet(ref_dir / f"{what}.parquet"))
        rows.append({"check": f"{version} {what}: key set and dtypes (structural)", **structure_compare(a, b, KEYS)})
        rows.append({"check": f"{version} {what} (keyed, exact)", **fcr.keyed_compare(a, b, KEYS)})
    if not joint:
        rows.append({"check": f"{version}: no forward file for one season (outcome and contact)", "identical":
                     not (new_dir / "forward_outcome.parquet").exists() and not (new_dir / "forward_contact.parquet").exists()})
    ca, cb = pd.read_csv(new_dir / "coverage.csv"), sel(pd.read_csv(ref_dir / "coverage.csv"))
    rows.append({"check": f"{version} coverage statuses and dtypes (structural)",
                 **structure_compare(ca, cb, ["season", "status"])})
    rows.append({"check": f"{version} coverage (exact)", **sf.exact_compare(ca, cb)})
    man_new = json.loads((new_dir / "manifest.json").read_text())
    man_ref = json.loads((ref_dir / "manifest.json").read_text())
    actual = research.input_state(spec.extra_inputs) if spec is not None else None
    if spec is not None and BIND is not None:
        want = {Path(BIND[k]).name for k in ("pitches", "pas", "support")} | {Path(x).name for x in spec.extra_inputs}
        rows.append({"check": f"{version}: manifest input names are the bound files", "identical":
                     set(json.loads((new_dir / "manifest.json").read_text())["inputs"]) == want and
                     {Path(x).resolve() for x in spec.extra_inputs} <= {Path(BIND["raw"]).resolve()}})
    rows += manifest_contract(man_new, man_ref, version, season=season, joint=joint, actual_inputs=actual)
    if version == "C1":
        recs = [r for r in man_ref["action_values"] if season is None or r["season"] == season]
        for rec in recs:
            rel = Path("action_values") / f"season={rec['season']}" / f"scoring_fold={rec['scoring_fold']}.parquet"
            an, ar = pd.read_parquet(new_actions / rel), pd.read_parquet(ref_actions / rel)
            rows.append({"check": f"C1 action rows {rec['season']}/{rec['scoring_fold']}: key set and dtypes (structural)",
                         **structure_compare(an, ar, ACTION_KEYS)})
            r = fcr.keyed_compare(an, ar, ACTION_KEYS)
            rows.append({"check": f"C1 per-action values {rec['season']}/{rec['scoring_fold']} (keyed, exact)", **r})
    return rows


def ctx_smoke_ref():
    """Smoke only: references on the subset through the package's joint path (run_model on the joint tables)."""
    if not SMOKE:
        raise SystemExit("smoke_ref is a smoke-only context")
    res = {"probe": probe_check(BIND)}
    out = Path(BIND["out"])
    for version, name in (("P0", "P0_ref"), ("C1", "C1_ref")):
        run_version(version, DEV_SEASONS, name, "s9ref")
    p, _, _ = research.base_tables()
    halves = arsenal.assign_folds(p, n_folds=2, seed=vp.SEED).rename("half")
    jobs = [(s, h, np.sort(halves.index[(halves == h).to_numpy() & halves.index.isin(p.loc[p["season"] == s, "game_pk"])]))
            for s in DEV_SEASONS for h in (0, 1)]
    with ProcessPoolExecutor(max_workers=4) as ex:
        p0 = {(r["meta"]["season"], r["meta"]["half"]): r for r in ex.map(vp._half_pipeline, jobs)}
        c1 = {(r["season"], r["half"]): r for r in ex.map(sf._half_job, [(c1_spec(), s, h, g, f"s9ref_half_{s}_{h}")
                                                                         for s, h, g in jobs])}
    with open(out / "halves_P0.pkl", "wb") as f:
        pickle.dump(p0, f)
    with open(out / "halves_C1.pkl", "wb") as f:
        pickle.dump({"runs": c1}, f)
    sc = pd.read_parquet(research.MODELS_DIR / "P0_ref" / "scores.parquet",
                         columns=["season", "catcher", "fielding_team", "game_pk"])
    prim, movers = primary_and_movers(sc, DEV_SEASONS)
    with open(out / "v0.pkl", "wb") as f:
        pickle.dump({"primary": prim, "movers": movers}, f)
    res["passed"] = res["probe"]["ok"]
    ctx_result(res)


def ctx_single():
    """C4: one season through the real single-season path, against the stored references restricted to it."""
    season, R = int(BIND["season"]), BIND["refs"]
    res = {"probe": probe_check(BIND), "checks": []}
    for version in ("P0", "C1"):
        label = f"s9ctl_single_{season}"
        _, spec = run_version(version, (season,), f"{version}_ctl_{season}", label, extra_inputs=(config.RAW_PARQUET,))
        res["checks"] += compare_run(version, research.MODELS_DIR / f"{version}_ctl_{season}", Path(R[f"{version}_dir"]),
                                     research.MODELS_DIR / "C1" / "runs" / label, Path(R["C1_actions"]), season=season,
                                     spec=spec)
    res["passed"] = res["probe"]["ok"] and all(c["identical"] for c in res["checks"])
    ctx_result(res)


def ctx_joint():
    """C5: the joint 2024-2025 runs through the driver, against the stored references under the same contracts."""
    R = BIND["refs"]
    res = {"probe": probe_check(BIND), "checks": []}
    for version in ("P0", "C1"):
        _, spec = run_version(version, DEV_SEASONS, f"{version}_ctl_joint", "s9ctl_joint", extra_inputs=(config.RAW_PARQUET,))
        res["checks"] += compare_run(version, research.MODELS_DIR / f"{version}_ctl_joint", Path(R[f"{version}_dir"]),
                                     research.MODELS_DIR / "C1" / "runs" / "s9ctl_joint", Path(R["C1_actions"]), joint=True,
                                     spec=spec)
    res["passed"] = res["probe"]["ok"] and all(c["identical"] for c in res["checks"])
    ctx_result(res)


def ctx_halves():
    """C6: the driver's 2024 assignments fed to the real half jobs (P0: vp._half_pipeline; C1: research_s6_full._half_job)."""
    season, R = int(BIND["season"]), BIND["refs"]
    res = {"probe": probe_check(BIND), "checks": []}
    with open(BIND["halves"], "rb") as f:
        halves = pickle.load(f)["halves"]
    p, _, _ = research.base_tables()
    games = set(p.loc[p["season"] == season, "game_pk"])
    jobs = [(season, h, np.sort(halves.index[(halves == h).to_numpy() & halves.index.isin(games)])) for h in (0, 1)]
    with ProcessPoolExecutor(max_workers=4) as ex:
        f0 = [ex.submit(vp._half_pipeline, j) for j in jobs]
        f1 = [ex.submit(sf._half_job, (c1_spec(), s, h, g, f"s9ctl_half_{s}_{h}")) for s, h, g in jobs]
        p0 = {(j[0], j[1]): f.result() for j, f in zip(jobs, f0)}
        c1 = {(j[0], j[1]): f.result() for j, f in zip(jobs, f1)}
    with open(R["P0_halves"], "rb") as f:
        ref_p0 = pickle.load(f)
    with open(R["C1_halves"], "rb") as f:
        ref_c1 = pickle.load(f)["runs"]
    for key in sorted(p0):
        r = sf.exact_compare(p0[key]["scores"], ref_p0[key]["scores"])
        res["checks"].append({"check": f"P0 half {key} scores", **r})
        m_new = {k: v for k, v in p0[key]["meta"].items() if k != "seconds"}
        m_ref = {k: v for k, v in ref_p0[key]["meta"].items() if k != "seconds"}
        d = deep_compare(m_new, m_ref, projection=False)
        res["checks"].append({"check": f"P0 half {key} meta (without seconds)", "identical": not d, "detail": d[:5]})
        r = sf.exact_compare(c1[key]["scores"], ref_c1[key]["scores"])
        res["checks"].append({"check": f"C1 half {key} scores", **r})
    res["passed"] = res["probe"]["ok"] and all(c["identical"] for c in res["checks"])
    ctx_result(res)


def ctx_replay():
    """C7: the driver's analysis code (validation_pilot with an explicit pair) on the stored inputs, against the stored
    v2 (P0) or s8_validate (C1) projections."""
    version, R = BIND["version"], BIND["refs"]
    res = {"probe": probe_check(BIND), "checks": []}
    scores = pd.read_parquet(Path(R[f"{version}_dir"]) / "scores.parquet")
    scores = with_ab(scores, config.PITCHES_PARQUET)
    with open(BIND["halves"], "rb") as f:
        hv = pickle.load(f)
    with open(R[f"{version}_halves"], "rb") as f:
        runs = pickle.load(f)
    runs = runs["runs"] if version == "C1" else runs
    with open(R["v0"], "rb") as f:
        v0 = pickle.load(f)
    prim, movers = primary_and_movers(scores, DEV_SEASONS)
    new = pair_analyses(scores, DEV_SEASONS, hv["halves"], hv["calendar"], runs, movers, prim)
    stored_path = R["v2"] if version == "P0" else R["s8"]
    if stored_path:
        with open(stored_path, "rb") as f:
            stored = pickle.load(f)
        for k in ("split_random", "split_calendar", "refit", "sets", "repeatability", "sensitivity", "team_staff",
                  "season_sd"):
            d = deep_compare(new[k], stored[k])
            res["checks"].append({"check": f"C7 {version} {k} reproduces the stored projection", "identical": not d,
                                  "detail": d[:8], "extra_keys": extras(new[k], stored[k])[:8]})
    else:
        # smoke: the explicit-pair path against validation_pilot's default path on the same inputs
        vp._DATA["scores"] = scores
        fits, sets = vp.season_fits(), vp.season_sets()
        d = deep_compare(new["repeatability"], vp.repeatability(fits, sets, v0["movers"]))
        res["checks"].append({"check": f"C7 {version} (smoke) explicit pair equals the default path", "identical": not d,
                              "detail": d[:8]})
    # R3b with no possible-exposure rows equals R3 (2024-2025 have none); the new spread table is computed and shown
    t = r3b_sensitivity(scores, config.PITCHES_PARQUET, new["_fits"], new["_sets"], DEV_SEASONS)
    r3 = new["sensitivity"]["table"]
    r3 = r3[r3["check"].isin(["R3_D", "R3_A"])].reset_index(drop=True)
    same = t.drop(columns="check").equals(r3.drop(columns="check"))
    res["checks"].append({"check": f"C7 {version} R3b equals R3 when no possible-exposure rows exist", "identical": same})
    res["spread"] = new["spread"].to_dict("records")
    res["passed"] = res["probe"]["ok"] and all(c["identical"] for c in res["checks"])
    ctx_result(res)


class HeldoutRole:
    """What validation._held_out reads from a role: its outcome and contact models and its fold."""
    def __init__(self, outcome, contact, fold):
        self.outcome, self.contact, self.fold = outcome, contact, fold


def ctx_forward():
    """C8 and C8b: the forward route on development data."""
    R = BIND["refs"]
    res = {"probe": probe_check(BIND), "checks": []}
    fw = json.loads(FORWARD_RECORD.read_text())
    pitches, _, _ = research.prepared(research.P0)
    rows25 = pitches[pitches["season"] == 2025]
    fitted = {}
    for art in fw["artifacts"]:
        name, kind = art["model"], art["component"]
        blob = (ROOT / art["pickle"]["path"]).read_bytes()
        ok = hashlib.sha256(blob).hexdigest() == art["pickle"]["sha256"]
        res["checks"].append({"check": f"C8 {name} {kind} pickle hash before loading", "identical": ok})
        if not ok:
            continue
        model = pickle.loads(blob)     # the same bytes that were hash-checked
        fitted[name, kind] = model
        spec = research.P0 if name == "P0" else replace(research_s8.SPECS["C1"], extra_inputs=(config.RAW_PARQUET,))
        rows = research.prepared(spec)[0]
        rows = rows[rows["season"] == 2025]
        frame, qh = query_frame(name, kind, model, rows, spec)
        qpath = ROOT / art["queries"]["path"]
        res["checks"].append({"check": f"C8 {name} {kind} stored query file hash", "identical":
                              sha(qpath) == art["queries"]["sha256"]})
        ref = pd.read_parquet(qpath)
        if SMOKE:   # the subset's position-player flag can admit rows the full table excludes: compare common keys
            common = frame[KEYS].merge(ref[KEYS], on=KEYS, how="inner")
            ref = ref.merge(common, on=KEYS, how="inner")
            frame = frame.merge(common, on=KEYS, how="inner")
        r = frames_compare(frame, ref, KEYS, bitwise=True, same_order=True)
        res["checks"].append({"check": f"C8 {name} {kind} 2025 queries reproduce bitwise", **r})
        if not SMOKE:
            res["checks"].append({"check": f"C8 {name} {kind} query-input hash equals the record's", "identical":
                                  qh == art["query_inputs_sha256"]})
    # seen-cell source (§7): 2025 outcome-training pairs vs P0/outcome.pkl's cell_n pairs
    if ("P0", "outcome") in fitted:
        pairs_rows = set(map(tuple, models.outcome_training(rows25)[["pitcher", "action"]].drop_duplicates().to_numpy()))
        idx = fitted["P0", "outcome"].cell_n.index.to_frame(index=False)
        pairs_pkl = set(map(tuple, idx[["pitcher", "action"]].drop_duplicates().to_numpy()))
        ok = pairs_rows == pairs_pkl if not SMOKE else True   # smoke: subset pairs reported only
        res["checks"].append({"check": "C8 seen-cell source: 2025 outcome-training pairs equal the pickle's pairs",
                              "identical": ok, "detail": f"{len(pairs_rows)} / {len(pairs_pkl)}"})
    # C7: the S9 mask (2024 outcome-training pairs, both components) against the stored research.forward files, key by
    # key; and the historical component mask's contact differences
    p24 = pitches[pitches["season"] == 2024]
    pairs_out = set(map(tuple, models.outcome_training(p24)[["pitcher", "action"]].drop_duplicates().to_numpy()))
    pairs_con = set(map(tuple, models.contact_training(p24)[["pitcher", "action"]].drop_duplicates().to_numpy()))
    for kind in ("outcome", "contact"):
        obs = research._training(kind, rows25, research.P0)
        mine = rows25.loc[obs.index, KEYS].reset_index(drop=True).assign(
            seen_mine=[(a, b) in pairs_out for a, b in obs[["pitcher", "action"]].to_numpy()])
        for name in ("P0", "C1"):
            f = pd.read_parquet(Path(R[f"{name}_dir"]) / f"forward_{kind}.parquet")[KEYS + ["seen_cell"]]
            m = mine.merge(f, on=KEYS, how="outer", indicator=True, validate="one_to_one")
            ok = bool((m["_merge"] == "both").all() and (m["seen_mine"] == m["seen_cell"]).all())
            res["checks"].append({"check": f"C7 S9 seen mask (2024 outcome pairs) equals stored {name} forward_{kind}"
                                           f".seen_cell key by key", "identical": ok, "detail": f"{len(m)} keys"})
        if kind == "contact":
            comp = np.array([(a, b) in pairs_con for a, b in obs[["pitcher", "action"]].to_numpy()])
            n_diff = int((comp != mine["seen_mine"].to_numpy()).sum())
            res["mask_differences_contact"] = n_diff
            if not SMOKE:
                res["checks"].append({"check": "C7 contact rows whose stratum differs between the masks = 1,206",
                                      "identical": n_diff == 1206, "detail": n_diff})
    # C7: held-out baselines (validation._held_out through calibration_check), P0 roles refitted per fold, against
    # reports/phase1_calibration.csv (outcome)
    roles = {(s_, k): HeldoutRole(models.fit_outcome_model(pitches[(pitches["season"] == s_) & (pitches["fold"] == k)]),
                                  models.fit_contact_model(pitches[(pitches["season"] == s_) & (pitches["fold"] == k)]), k)
             for s_ in DEV_SEASONS for k in range(3)}
    cal = validation.calibration_check(pitches, roles, seasons=DEV_SEASONS, kind="outcome")
    if not SMOKE:
        ref_cal = pd.read_csv(ROOT / "reports" / "phase1_calibration.csv", float_precision="round_trip")
        d = deep_compare(cal.reset_index(drop=True), ref_cal)
        res["checks"].append({"check": "C7 held-out outcome log loss, both baselines and calibration reproduce "
                                       "reports/phase1_calibration.csv", "identical": not d, "detail": d[:5]})
    # C8b: forward reporting reproduces the stored forward tables (historical projections, each under its own mask)
    fb = {k: pd.read_parquet(Path(R["P0_dir"]) / f"forward_{k}.parquet") for k in ("outcome", "contact")}
    fc = {k: pd.read_parquet(Path(R["C1_dir"]) / f"forward_{k}.parquet") for k in ("outcome", "contact")}
    t_s8 = forward_matched_table(fb, fc)
    p0_24 = {"outcome": models.fit_outcome_model(p24), "contact": models.fit_contact_model(p24)}
    t_vp = forward_baseline_table(p0_24, p24, rows25, mask="component")
    if R.get("s8"):
        with open(R["s8"], "rb") as f:
            s8 = pickle.load(f)
        d = deep_compare(t_s8, s8["prediction"]["forward"])
        res["checks"].append({"check": "C8b forward matched table (S9 mask) reproduces s8_validate", "identical": not d,
                              "detail": d[:5]})
    if R.get("v2"):
        with open(R["v2"], "rb") as f:
            v2 = pickle.load(f)
        d = deep_compare(t_vp, v2["forward"])
        res["checks"].append({"check": "C8b forward baseline table (component mask) reproduces v2", "identical": not d,
                              "detail": d[:5]})
    res["passed"] = res["probe"]["ok"] and all(c["identical"] for c in res["checks"])
    ctx_result(res)


CONTEXTS = {"smoke_ref": ctx_smoke_ref, "single": ctx_single, "joint": ctx_joint, "halves": ctx_halves,
            "replay": ctx_replay, "forward": ctx_forward}


# ===================================================================================== completeness gate (§8.1)

def valid_schedule(obj) -> bool:
    """Total: False on any malformed input. Every consumed field has its type."""
    try:
        if not isinstance(obj, dict) or not isinstance(obj.get("dates"), list) or not obj["dates"]:
            return False
        for d in obj["dates"]:
            if not isinstance(d, dict) or not isinstance(d.get("games"), list):
                return False
            for g in d["games"]:
                if not (isinstance(g, dict) and isinstance(g.get("gamePk"), int) and not isinstance(g["gamePk"], bool)
                        and isinstance(g.get("status"), dict) and isinstance(g.get("teams"), dict)
                        and isinstance(g.get("gameType"), str)):
                    return False
                st = g["status"]
                if not all(isinstance(st.get(k), (str, type(None))) for k in ("abstractGameState", "detailedState")):
                    return False
                for side in ("home", "away"):
                    t = g["teams"].get(side, {})
                    if not isinstance(t, dict):
                        return False
                    sc = t.get("score")       # absent or null for unfinished or cancelled games
                    if sc is not None and (not isinstance(sc, int) or isinstance(sc, bool)):
                        return False
        return True
    except Exception:  # noqa: BLE001
        return False


def valid_pbp(obj) -> bool:
    """Total: False on any malformed input, and on an empty play list (a completed game has plays). Every event field
    the mapping consumes (isPitch, type, details.code, details.call.code, details.eventType) is checked for type. A
    repeated atBatIndex is malformed (Astra: it would merge two plays' events into one expected PA; none of the 4,859
    development feeds has one)."""
    try:
        if not isinstance(obj, dict) or not isinstance(obj.get("allPlays"), list) or not obj["allPlays"]:
            return False
        seen_ab = set()
        for p in obj["allPlays"]:
            ab = p.get("about", {}).get("atBatIndex") if isinstance(p, dict) and isinstance(p.get("about"), dict) else None
            if not isinstance(ab, int) or isinstance(ab, bool) or ab < 0 or not isinstance(p.get("playEvents"), list) \
                    or ab in seen_ab:
                return False
            seen_ab.add(ab)
            for e in p["playEvents"]:
                if not isinstance(e, dict) or not isinstance(e.get("type"), str):
                    return False
                if "isPitch" in e and not isinstance(e["isPitch"], bool):
                    return False
                d = e.get("details")
                if d is not None and not isinstance(d, dict):
                    return False
                d = d or {}
                if not all(isinstance(d.get(k), (str, type(None))) for k in ("code", "eventType")):
                    return False
                c = d.get("call")
                if c is not None and (not isinstance(c, dict) or not isinstance(c.get("code"), (str, type(None)))):
                    return False
        return True
    except Exception:  # noqa: BLE001
        return False


def fetch(url: str, path: Path, valid=lambda o: True) -> bytes:
    """Cached on disk; resumable; retries with backoff. A cached file that does not parse or has the wrong shape is
    fetched again; a response still malformed after the last try raises (the game is then "reference unavailable")."""
    if path.exists():
        try:
            data = path.read_bytes()
            if valid(json.loads(data)):
                return data
        except (ValueError, OSError):
            pass
        path.unlink(missing_ok=True)
    last = None
    for attempt in range(len(BACKOFF) + 1):
        if attempt:
            time.sleep(BACKOFF[attempt - 1])
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if not valid(json.loads(data)):
                raise ValueError("response has the wrong shape")
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, path)
            return data
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RuntimeError(f"{url}: {last}")


def completed_games(schedule: dict) -> pd.DataFrame:
    """Completed regular-season games: final, played (not cancelled or postponed), both scores present; matched by
    game_pk across dates (a suspended game appears on more than one date)."""
    rows = []
    for d in schedule.get("dates", []):
        for g in d.get("games", []):
            st = g.get("status", {})
            home, away = g.get("teams", {}).get("home", {}), g.get("teams", {}).get("away", {})
            rows.append({"game_pk": int(g["gamePk"]), "date": d.get("date"), "game_type": g.get("gameType"),
                         "abstract": st.get("abstractGameState"), "detailed": st.get("detailedState"),
                         "home_score": home.get("score"), "away_score": away.get("score")})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    ok = df["game_type"].eq("R") & df["abstract"].eq("Final") & ~df["detailed"].isin(["Cancelled", "Postponed"]) \
        & df["home_score"].notna() & df["away_score"].notna()
    fin = df[ok]
    agree = fin.groupby("game_pk")[["home_score", "away_score"]].nunique().le(1).all(axis=1)
    if not agree.all():
        raise ValueError(f"completed entries disagree on scores: {sorted(agree.index[~agree])}")
    return fin.drop_duplicates("game_pk", keep="last").set_index("game_pk").sort_index()


def call_code(e: dict):
    return ((e.get("details", {}) or {}).get("call") or {}).get("code")


def kept(e: dict) -> bool:
    """Mapping v2 (general rule): an MLB play event is a Statcast row if it is a pitch, or a no-pitch event that carries
    a call code (MLB's automatic ball and strike calls). v1 kept only VB, VP and AC and failed on development data: 28
    games had a Statcast automatic_ball or automatic_strike row at an AB, V, VC or VS call (all "Automatic Ball/Strike -
    ..." in MLB's description). Under v2, 4,856 of 4,859 development games match key for key; the other three are the
    two Tokyo games (absent) and 746731 (EXPECTED_DEV_SHORTFALL). A no-pitch event without a call (code N, 487 in
    development) is not a row."""
    if e.get("isPitch"):
        return True
    return e.get("type") == "no_pitch" and call_code(e) is not None


def event_signature(e: dict) -> tuple:
    """(type, isPitch, code, call code, eventType): MLB's event subtype is kept, so a new action kind is distinguishable
    from a familiar one."""
    d = e.get("details", {}) or {}
    return (e.get("type"), bool(e.get("isPitch")), d.get("code"), call_code(e), d.get("eventType"))


def unrecognized(e: dict) -> bool:
    """A no-pitch event v2 keeps whose MLB description is not an automatic ball or strike (its row status is unverified;
    the gate fails on it for review)."""
    if e.get("type") != "no_pitch" or call_code(e) is None:
        return False
    desc = (e.get("details") or {}).get("description")
    return not (isinstance(desc, str) and desc.startswith(("Automatic Ball", "Automatic Strike")))


def pa_evidence(pbp: dict, rows: pd.DataFrame, ab: int) -> tuple:
    """(MLB's kept-event codes, Statcast's descriptions in pitch order) for one plate appearance."""
    codes = [(e.get("details") or {}).get("code") for p in pbp.get("allPlays", []) if p["about"]["atBatIndex"] + 1 == ab
             for e in p["playEvents"] if kept(e)]
    descs = rows[rows["at_bat_number"] == ab].sort_values("pitch_number")["description"].tolist()
    return codes, descs


def expected_shortfall(season: int, game_pk: int, kc: dict, pbp=None, rows=None) -> bool:
    """A development key mismatch equal to a listed, diagnosed shortfall (EXPECTED_DEV_SHORTFALL, amendment 2): the
    exact key difference, and MLB's kept-event codes and Statcast's descriptions in the listed PA. Nothing else."""
    want = EXPECTED_DEV_SHORTFALL.get(season, {}).get(game_pk)
    if want is None or pbp is None or rows is None:
        return False
    if not (kc["n_missing"] == len(want["missing"]) and kc["n_extra"] == len(want["extra"]) and
            [list(x) for x in kc["missing"]] == want["missing"] and [list(x) for x in kc["extra"]] == want["extra"]):
        return False
    codes, descs = pa_evidence(pbp, rows, want["pa"])
    return codes == want["mlb_kept_codes"] and descs == want["statcast_descriptions"]


def expected_keys(pbp: dict) -> set:
    """(at_bat_number, pitch_number): at_bat_number = atBatIndex + 1 (no renumbering); kept events numbered from 1."""
    out = set()
    for play in pbp.get("allPlays", []):
        ab = int(play["about"]["atBatIndex"]) + 1
        n = 0
        for e in play.get("playEvents", []):
            if kept(e):
                n += 1
                out.add((ab, n))
    return out


def vocabulary(pbp: dict) -> Counter:
    return Counter(event_signature(e) for p in pbp.get("allPlays", []) for e in p.get("playEvents", []))


def key_check(expected: set, actual: set) -> dict:
    return {"ok": expected == actual, "missing": sorted(expected - actual)[:10], "extra": sorted(actual - expected)[:10],
            "n_missing": len(expected - actual), "n_extra": len(actual - expected)}


def gate_body(seasons=DEV_SEASONS, raw_path=ORIG["raw"], expected=EXPECTED_COMPLETED, absent=EXPECTED_ABSENT):
    res = {"mapping_version": MAPPING_VERSION, "seasons": {}, "tests": []}
    raw = pd.read_parquet(raw_path, columns=["game_pk", "game_type", "game_year", "at_bat_number", "pitch_number",
                                             "post_home_score", "post_away_score", "description"])
    raw = raw[raw["game_type"] == "R"]
    vocab_all = Counter()
    for season in seasons:
        sch = json.loads(fetch(SCHEDULE_URL.format(season=season), MLB_CACHE / "schedule" / f"{season}.json",
                               valid_schedule))
        comp = completed_games(sch)
        r = raw[raw["game_year"] == season]
        games = sorted(comp.index)
        if SMOKE:
            games = [g for g in games if g % 5 == 0][:100]
        dup = int(r.duplicated(["game_pk", "at_bat_number", "pitch_number"]).sum())
        statcast = {g: set(map(tuple, x[["at_bat_number", "pitch_number"]].to_numpy()))
                    for g, x in r.groupby("game_pk")}
        smax = r.groupby("game_pk")[["post_home_score", "post_away_score"]].max()

        def one(g):
            try:
                return g, json.loads(fetch(PBP_URL.format(pk=g), MLB_CACHE / "pbp" / f"{g}.json", valid_pbp)), None
            except Exception as exc:  # noqa: BLE001
                return g, None, str(exc)
        unavailable, mismatch, score_bad, missing, unrec, shortfall = [], [], [], [], [], []
        with ThreadPoolExecutor(max_workers=FETCH_THREADS) as ex:
            for g, pbp, err in ex.map(one, games):
                if err:
                    unavailable.append({"game_pk": g, "error": err})
                    continue
                vocab_all.update(vocabulary(pbp))
                u = [event_signature(e) for p in pbp["allPlays"] for e in p["playEvents"] if unrecognized(e)]
                if u:
                    unrec.append({"game_pk": g, "events": sorted(set(map(str, u)))})
                if g not in statcast:
                    missing.append(g)
                    continue
                kc = key_check(expected_keys(pbp), statcast[g])
                if not kc["ok"]:
                    listed = expected_shortfall(season, g, kc, pbp, r[r["game_pk"] == g])
                    (shortfall if listed else mismatch).append(
                        {"game_pk": g, **kc, **({"diagnosis": "confirmed missing pitch (amendment 2)"} if listed else {})})
                c = comp.loc[g]
                if not (smax.loc[g, "post_home_score"] == c["home_score"] and smax.loc[g, "post_away_score"] == c["away_score"]):
                    score_bad.append({"game_pk": g, "statcast": smax.loc[g].tolist(),
                                      "official": [c["home_score"], c["away_score"]]})
        unexpected_ids = sorted(set(r["game_pk"]) - set(comp.index))
        exp_absent = absent.get(season, set())
        s = {"completed": len(comp), "completed_expected": expected.get(season), "checked": len(games),
             "cancelled_example_excluded": CANCELLED_EXAMPLE not in comp.index if season == 2024 else None,
             "duplicate_statcast_keys": dup, "unavailable": unavailable, "key_mismatch": mismatch[:50],
             "n_key_mismatch": len(mismatch), "score_mismatch": score_bad[:50], "n_score_mismatch": len(score_bad),
             "missing_games": missing, "unexpected_statcast_games": unexpected_ids, "unrecognized_events": unrec[:50],
             "n_unrecognized": len(unrec), "expected_shortfall": shortfall,
             "expected_shortfall_listed": sorted(EXPECTED_DEV_SHORTFALL.get(season, {}))}
        s["ok"] = ((SMOKE or len(comp) == expected.get(season)) and s["cancelled_example_excluded"] is not False
                   and not dup and not unavailable and not mismatch and not score_bad and not unexpected_ids and not unrec
                   and set(missing) == (exp_absent if not SMOKE else set(missing) & exp_absent)
                   and (SMOKE or {x["game_pk"] for x in shortfall} == set(EXPECTED_DEV_SHORTFALL.get(season, {}))))
        res["seasons"][season] = s
    res["vocabulary"] = [{"type": k[0], "isPitch": k[1], "code": k[2], "call": k[3], "eventType": k[4], "kept": kept(
        {"type": k[0], "isPitch": k[1], "details": {"code": k[2], "call": {"code": k[3]} if k[3] else None}}), "n": v}
        for k, v in sorted(vocab_all.items(), key=lambda kv: (-kv[1], str(kv[0])))]
    res["tests"] = gate_tests(raw)
    res["passed"] = all(s["ok"] for s in res["seasons"].values()) and all(t["ok"] for t in res["tests"])
    return res


def gate_tests(raw: pd.DataFrame) -> list:
    """In-memory tests on development games: a zero-pitch pickoff play passes; deletions and a shifted key fail."""
    tests = []

    def statcast(g):
        x = raw[raw["game_pk"] == g]
        return x, set(map(tuple, x[["at_bat_number", "pitch_number"]].to_numpy()))
    g = 744797
    pbp = json.loads(fetch(PBP_URL.format(pk=g), MLB_CACHE / "pbp" / f"{g}.json", valid_pbp))
    play = [p for p in pbp["allPlays"] if p["about"]["atBatIndex"] == 3][0]
    x, keys = statcast(g)
    tests.append({"test": "744797 PA 4 (pickoff only): no kept event, no Statcast row, game passes",
                  "ok": not any(kept(e) for e in play["playEvents"]) and not (x["at_bat_number"] == 4).any()
                  and key_check(expected_keys(pbp), keys)["ok"]})
    g = 746043
    pbp = json.loads(fetch(PBP_URL.format(pk=g), MLB_CACHE / "pbp" / f"{g}.json", valid_pbp))
    x, keys = statcast(g)
    exp = expected_keys(pbp)
    tests.append({"test": "746043 passes as stored (four intentional balls numbered 1-4)",
                  "ok": key_check(exp, keys)["ok"] and {(67, i) for i in (1, 2, 3, 4)} <= keys})
    x = x.sort_values(["at_bat_number", "pitch_number"])
    pa_sizes = x.groupby("at_bat_number").size()
    big = int(pa_sizes[pa_sizes >= 3].index[0])
    pa = x[x["at_bat_number"] == big]
    pre = x[(x["at_bat_number"] == big)]
    scoreless = pre[(pre["post_home_score"] == pre["post_home_score"].min())].iloc[0]
    cases = {
        "a scoreless pitch removed": keys - {(int(scoreless["at_bat_number"]), int(scoreless["pitch_number"]))},
        "a PA's last pitch removed": keys - {(big, int(pa["pitch_number"].max()))},
        "a whole PA removed": {k for k in keys if k[0] != big},
        "a key shifted with the count unchanged": {k for k in keys if k[0] != big} | {(big, p + 1) for (_, p) in
                                                                                      [k for k in keys if k[0] == big]},
    }
    for name, ks in cases.items():
        tests.append({"test": f"negative: {name} is flagged", "ok": not key_check(exp, ks)["ok"]})
    return tests


def stage_gate():
    run_stage(f"gate_v{MAPPING_VERSION}", gate_body, needs=("guard",))


# ===================================================================================== phase 2: target seasons (§3, §5, §6)
#
# Order (§10): authorize -> extract -> build -> checks -> run -> halves -> validate -> report, each recorded once.
# `domains` (development data only) records the closed domains before any target access. `prefetch` (repeatable, no
# record) only fills the play-by-play cache. Every season stage refuses unless the development chain is ready for
# target work (guard, audit, control, gate_v<MAPPING_VERSION>), the domains record verifies, and the season's
# authorization record is valid.

def season_info(year: int) -> dict:
    """The season's source, pairs and label. Production: TARGETS only. Smoke: the development stand-in only, so a
    smoke run never reads a target-season row."""
    if SMOKE:
        if year != SMOKE_STANDIN:
            raise SystemExit(f"refused: smoke runs the season stages only on the stand-in {SMOKE_STANDIN} (a development "
                             f"season), never on a target season")
        return {"year": year, "source": ORIG["raw"], "pairs": PAIRS[year], "authorization": "smoke",
                "label": "smoke stand-in (a development season; means nothing)"}
    if year not in TARGETS:
        raise SystemExit(f"refused: season stages are implemented for {TARGETS} only (2026 needs phase 3: §7, §8.2)")
    return {"year": year, "source": RAW_2015_2023, "pairs": PAIRS[year], "authorization": AUTHORIZATION[year],
            "label": SEASON_LABEL[year]}


def season_dir(year: int) -> Path:
    return SEASONS_ROOT / str(year)


def other_reference(version: str) -> tuple:
    """(scores, pitch table) of the stored scores of the same version, for the other season of a pair (§6 row 6)."""
    R = refs()
    pitches = ctl_dir("tables", "joint_ref", "pitches.parquet") if SMOKE else ORIG["pitches"]
    return Path(R[f"{version}_dir"]) / "scores.parquet", Path(pitches)


def file_digests(paths) -> dict:
    return {str(Path(p)): sha(p) for p in paths}


def verify_inputs(*stages) -> list:
    """Every file a predecessor stage recorded as an output must still have its recorded digest."""
    bad = []
    for st in stages:
        rec, err = read_record(record_path(st))
        if err:
            bad.append(f"{st}: {err}")
            continue
        outs = (rec.get("result") or {}).get("outputs")
        if not isinstance(outs, dict) or not outs:
            bad.append(f"{st}: no recorded outputs")
            continue
        for p, h in outs.items():
            if not Path(p).is_file():
                bad.append(f"{st}: {p} missing")
            elif sha(p) != h:
                bad.append(f"{st}: {p} changed")
    return bad


def season_gate(year: int, stage: str) -> dict:
    """Before a season stage reserves its record (refusals leave no record)."""
    info = season_info(year)
    rd = ready_for_targets()
    if not rd["ready"]:
        raise SystemExit(f"refused: the development chain is not ready for target-season work: "
                         f"{ {k: v for k, v in rd['problems'].items() if v} }")
    rec, err = read_record(record_path(f"authorize_{year}"))
    if err:
        raise SystemExit(f"refused: no valid authorization record for {year} ({err}); it is written only on the user's "
                         f"explicit instruction: s9_seasons.py authorize {year} <YYYY-MM-DD>")
    bad = record_problems(f"authorize_{year}", rec, identity())
    if bad or (rec.get("result") or {}).get("season") != year:
        raise SystemExit(f"refused: the authorization record for {year} does not verify: {bad}")
    drec, derr = read_record(record_path("domains"))
    dbad = [derr] if derr else record_problems("domains", drec, identity())
    if dbad:      # §5 B0: the closed domains are recorded before any target access, prefetch included (Astra)
        raise SystemExit(f"refused: the domains record does not verify: {dbad}")
    return info


def run_season_stage(year: int, name: str, body):
    stage = f"{name}_{year}"
    want = expected_predecessors(stage)
    return run_stage(stage, body, needs=tuple(sorted(want)))


# ------------------------------------------------------------------------------------- domains and authorization

def dev_domains() -> dict:
    """The closed domains of §5 B0, from the 2024-2025 regular-season raw rows and the frozen code."""
    raw = pd.read_parquet(ORIG["raw"], columns=["game_type", "description", "events", "bb_type", "stand", "p_throws",
                                                "inning_topbot"])
    r = raw[raw["game_type"] == "R"]
    dom = {"description": sorted(set(loader.OUTCOME) | set(loader.AUTOMATIC)),
           "description_dev": sorted(r["description"].dropna().unique()),
           "events_dev": sorted(r["events"].dropna().unique()),
           "events_code": sorted(set(loader.PA_END) | set(loader.OUTS_ON_PLAY))}
    for c in ("bb_type", "stand", "p_throws", "inning_topbot"):
        dom[c] = sorted(r[c].dropna().unique())
    return dom


def stage_domains():
    def body():
        dom = dev_domains()
        checks = [{"check": "description domain has 15 labels and equals the 2024-2025 regular-season set",
                   "ok": len(dom["description"]) == 15 and dom["description"] == dom["description_dev"]},
                  {"check": "bb_type domain is batter.BB_TYPES", "ok": set(dom["bb_type"]) == set(batter.BB_TYPES)},
                  {"check": "stand and p_throws are {L, R}", "ok": dom["stand"] == ["L", "R"] == dom["p_throws"]}]
        return {"domains": dom, "checks": checks, "passed": all(c["ok"] for c in checks),
                "note": "recorded before any target-season access (§5 B0)"}
    run_stage("domains", body, needs=("guard",))


def valid_authorization_date(date: str) -> bool:
    import datetime as dt
    if not isinstance(date, str) or len(date) != 10 or date[4] != "-" or date[7] != "-":
        return False
    try:
        d = dt.date.fromisoformat(date)
    except ValueError:
        return False
    return dt.date.fromisoformat(PROTOCOL_LOCKED_ON) <= d <= dt.date.today()


def stage_authorize(year: int, date: str):
    """Records the user's authorization of a season (2023: A11-3). Run only on the user's explicit instruction; the
    script cannot know who authorized, so this record is the auditable trace the season stages require."""
    info = season_info(year)
    if not valid_authorization_date(date):
        raise SystemExit(f"refused: authorization date must be YYYY-MM-DD between {PROTOCOL_LOCKED_ON} and today")

    def body():
        return {"passed": True, "season": year, "authorization": info["authorization"], "authorized_on": date,
                "label": info["label"], "note": "recorded on the user's explicit instruction"}
    run_stage(f"authorize_{year}", body, needs=("guard",))


def stage_prefetch(year: int):
    """Repeatable, no record: the season's schedule and play-by-play into the cache (resumable), so `checks` does not
    depend on the network. Needs the same readiness and authorization as any season stage."""
    season_gate(year, "prefetch")
    sch = json.loads(fetch(SCHEDULE_URL.format(season=year), MLB_CACHE / "schedule" / f"{year}.json", valid_schedule))
    games = completeness_games(completed_games(sch))
    failed = []

    def one(g):
        try:
            fetch(PBP_URL.format(pk=g), MLB_CACHE / "pbp" / f"{g}.json", valid_pbp)
            return None
        except Exception as exc:  # noqa: BLE001
            return {"game_pk": g, "error": str(exc)}
    with ThreadPoolExecutor(max_workers=FETCH_THREADS) as ex:
        failed = [r for r in ex.map(one, games) if r]
    log = {"season": year, "games": len(games), "failed": failed, "at": now(), "mode": MODE}
    (MLB_CACHE / f"prefetch_{year}.json").write_text(jdump(log))
    print(f"prefetch {year}: {len(games)} games, {len(failed)} failed after retries")
    sys.exit(0 if not failed else 1)


# ------------------------------------------------------------------------------------- extract and build

EXTRACT_COLUMNS = list(dict.fromkeys(loader.RAW_COLUMNS + research_s7.RAW_COLUMNS + ["post_home_score", "post_away_score"]))


def stage_extract(year: int):
    info = season_gate(year, "extract")

    def body():
        dst = season_dir(year) / "raw.parquet"
        ext = write_raw_extract(info["source"], dst, year)
        sch, ref = pq.read_schema(dst), pq.read_schema(ORIG["raw"])
        bad = {c: (str(sch.field(c).type) if c in sch.names else None, str(ref.field(c).type))
               for c in EXTRACT_COLUMNS if c not in sch.names or str(sch.field(c).type) != str(ref.field(c).type)}
        t = pd.read_parquet(dst, columns=["game_type", "game_year"])
        checks = [{"check": f"{len(EXTRACT_COLUMNS)} required columns present with the development types", "ok": not bad,
                   "detail": bad or len(EXTRACT_COLUMNS)},
                  {"check": "every row has this game_year", "ok": bool(t["game_year"].eq(year).all())},
                  {"check": "regular-season rows present", "ok": bool(t["game_type"].eq("R").any())}]
        return {"extract": ext, "source": str(info["source"]), "label": info["label"],
                "columns": {"extra": sorted(set(sch.names) - set(ref.names)), "missing": sorted(set(ref.names) - set(sch.names))},
                "game_types": {str(k): int(v) for k, v in t["game_type"].value_counts(dropna=False).items()},
                "checks": checks, "outputs": file_digests([dst]), "passed": all(c["ok"] for c in checks)}
    run_season_stage(year, "extract", body)


def stage_build(year: int):
    season_gate(year, "build")

    def body():
        bad = verify_inputs(f"extract_{year}")
        if bad:
            return {"passed": False, "stopped": f"inputs changed: {bad}"}
        sd = season_dir(year)
        flags = load_flags()
        pitches, pas, support = build_tables(read_raw_rows(sd / "raw.parquet"), flags)
        paths = save_tables(sd, pitches, pas, support)
        halves, calendar = season_halves(pitches)
        st = halves_structure(pitches, halves)
        hv = pd.DataFrame({"season": year, "half": halves, "calendar_half": calendar.reindex(halves.index)})
        hpath = sd / "halves.parquet"
        if hpath.exists():
            raise FileExistsError(f"{hpath} exists")
        hv.to_parquet(hpath)                      # index game_pk; written before any score exists (§3)
        checks = [{"check": "pitch, PA and support rows are this season only", "ok":
                   all(set(x["season"].unique()) == {year} for x in (pitches, pas, support))},
                  {"check": "halves disjoint, union all games, within-half folds populated", "ok": st["ok"], "detail": st},
                  {"check": "every game has a calendar half", "ok": bool(hv["calendar_half"].notna().all())},
                  {**level_check(pitches, flags), "ok": level_check(pitches, flags)["identical"]}]
        pp = pitches.loc[pitches["position_player_pitching"].astype(bool), "pitcher"].nunique()
        return {"rows": {"pitches": len(pitches), "pas": len(pas), "support": len(support),
                         "games": int(pitches["game_pk"].nunique())},
                "position_player_pitchers": int(pp), "halves": st, "checks": checks,
                "dugout_level": {str(k): int(v) for k, v in pitches["dugout_level"].value_counts(dropna=False).items()},
                "outputs": file_digests([*paths.values(), hpath]), "passed": all(c["ok"] for c in checks)}
    run_season_stage(year, "build", body)


# ------------------------------------------------------------------------------------- checks (§5)

CHECK_RAW_COLUMNS = ["game_type", "game_year", "game_pk", "at_bat_number", "pitch_number", "description", "events",
                     "bb_type", "stand", "p_throws", "inning_topbot", "des", "launch_speed_angle", "fielder_2", "batter",
                     "pitcher", "bat_score", "fld_score", "post_bat_score", "post_fld_score", "post_home_score",
                     "post_away_score", "balls", "strikes", "outs_when_up", "pitch_type"]
SCORE_COLUMNS = ("bat_score", "fld_score", "post_bat_score", "post_fld_score", "post_home_score", "post_away_score")


def terminal_mask(r: pd.DataFrame) -> pd.Series:
    """The last row of each plate appearance (largest pitch_number within game_pk and at_bat_number)."""
    return r.groupby(["game_pk", "at_bat_number"])["pitch_number"].transform("max").eq(r["pitch_number"])


def b0_inventory(raw: pd.DataFrame, dom: dict) -> dict:
    """§5 B0 on the whole raw extract (game types; attach_bb_type's preconditions) and on its regular-season rows
    (closed domains; null counts by position, reported without a rule)."""
    out = {"game_type": {str(k): int(v) for k, v in raw["game_type"].value_counts(dropna=False).items()}}
    out["duplicate_keys_whole_file"] = int(raw.duplicated(batter.BB_KEYS).sum())
    out["bb_type_outside_labels_whole_file"] = sorted(map(str, set(raw["bb_type"].dropna().unique()) - set(batter.BB_TYPES)))
    r = raw[raw["game_type"] == "R"]
    term = terminal_mask(r)
    viol = {}
    for col in ("description", "bb_type", "stand", "p_throws", "inning_topbot"):
        extra = sorted(set(r[col].dropna().unique()) - set(dom[col]))
        if extra:
            viol[col] = {str(v): int(r[col].eq(v).sum()) for v in extra}
    ev = set(r["events"].dropna().unique())
    code_only = sorted((ev - set(dom["events_dev"])) & set(dom["events_code"]))
    ev_bad = sorted(ev - set(dom["events_dev"]) - set(dom["events_code"]))
    if ev_bad:
        viol["events"] = {str(v): int(r["events"].eq(v).sum()) for v in ev_bad}
    hip = r["description"].eq("hit_into_play")
    out.update(domain_violations=viol,
               events_accepted_through_code_only={str(v): int(r["events"].eq(v).sum()) for v in code_only},
               pitch_types=sorted(map(str, set(r["pitch_type"].dropna().unique()))),
               nulls={c: {"terminal": int(r.loc[term, c].isna().sum()), "non_terminal": int(r.loc[~term, c].isna().sum())}
                      for c in ("description", "events", "stand", "p_throws", "inning_topbot", "pitch_type")},
               bb_type_nulls={"hit_into_play": int(r.loc[hip, "bb_type"].isna().sum()),
                              "other_rows_non_null": int(r.loc[~hip, "bb_type"].notna().sum())})
    out["ok"] = out["duplicate_keys_whole_file"] == 0 and not out["bb_type_outside_labels_whole_file"] and not viol
    return out


def b1_counts(r: pd.DataFrame, pitches: pd.DataFrame) -> dict:
    """§5 B1 (report): raw regular-season hit_into_play rows that are (i) not terminal, (ii) have a null, empty or
    whitespace-only des; and how the frozen code bins each (contact_bin in the built table)."""
    term = terminal_mask(r)
    hip = r[r["description"].eq("hit_into_play")]
    sets = {"not_terminal": hip[~term.loc[hip.index]],
            "des_blank": hip[hip["des"].isna() | hip["des"].astype("string").str.strip().fillna("").eq("")]}
    bins = pitches[["game_pk", "at_bat_number", "pitch_number", "contact_bin"]]
    out = {"hit_into_play_rows": len(hip)}
    for k, d in sets.items():
        m = d[["game_pk", "at_bat_number", "pitch_number"]].merge(bins, how="left", on=["game_pk", "at_bat_number",
                                                                                      "pitch_number"])
        out[k] = {"rows": len(d), "contact_bin": {str(b): int(n) for b, n in
                                                  m["contact_bin"].astype("object").value_counts(dropna=False).items()}}
    return out


def pitch_type_shares(p: pd.DataFrame, season) -> pd.DataFrame:
    t = p[p["thrown"].astype(bool)]
    s = t["pitch_type"].astype("object").where(t["pitch_type"].notna(), "<null>")
    c = s.value_counts()
    return pd.DataFrame({"season": season, "pitch_type": c.index.astype(str), "pitches": c.to_numpy(),
                         "share": c.to_numpy() / len(t)})


def b2_table(target: pd.DataFrame, year: int, dev: pd.DataFrame, label=None) -> dict:
    """Pitch-type shares by sample. Every row carries a `sample` label (the target's from target_sample_label), so the
    smoke stand-in never shares a label with development 2024 (Astra, revision 9)."""
    label = str(year) if label is None else label
    if label in {str(s) for s in DEV_SEASONS}:
        raise ValueError(f"the target sample label {label!r} is a development season's label")
    parts = [pitch_type_shares(target, year)] + [pitch_type_shares(dev[dev["season"] == s], s) for s in DEV_SEASONS]
    labels = [label] + [str(s) for s in DEV_SEASONS]
    tab = pd.concat([p_.assign(sample=lb)[["sample"] + list(p_.columns)] for p_, lb in zip(parts, labels)],
                    ignore_index=True)
    new = sorted(set(parts[0]["pitch_type"]) - set(parts[1]["pitch_type"]) - set(parts[2]["pitch_type"]))
    return {"table": tab, "new_in_target": new}


def prefit_checks(r: pd.DataFrame, pitches: pd.DataFrame, raw_all: pd.DataFrame) -> list:
    """§5 pre-fit checks: conditions the frozen code would raise on later or silently mishandle."""
    rows = []

    def add(check, n_bad, detail=None):
        rows.append({"check": check, "ok": int(n_bad) == 0, "bad": int(n_bad), "detail": detail})
    use = pitches[pitches["outcome_fit_eligible"].astype(bool)]
    tracked = use["contact_fit_eligible"].astype(bool) & use["contact_bin"].isin(batter.TRACKED_BINS).astype(bool)
    tr = use[tracked]
    add("xwoba non-null on every tracked, non-bunt, contact-fit-eligible ball in play", tr["xwoba"].isna().sum(),
        f"{len(tr)} rows")
    try:
        bb = batter.attach_bb_type(tr[batter.BB_KEYS], raw_all[batter.BB_KEYS + ["bb_type"]])
        add("bb_type non-null on every tracked, non-bunt, contact-fit-eligible ball in play", bb["bb_type"].isna().sum())
    except ValueError as exc:
        add("bb_type join on the whole raw file (batter.attach_bb_type)", 1, str(exc))
    for c in ("fielder_2", "batter", "pitcher", "game_pk", *SCORE_COLUMNS):
        add(f"{c} non-null on regular-season rows", r[c].isna().sum())
    for c, hi in (("balls", 3), ("strikes", 2), ("outs_when_up", 2)):
        add(f"{c} in 0-{hi}", (~r[c].between(0, hi)).sum())
    for c in ("stand", "p_throws"):
        add(f"{c} in {{L, R}}", (~r[c].isin(["L", "R"])).sum())
    return rows


def environment_tables(p: pd.DataFrame, a: pd.DataFrame, season) -> dict:
    """§5 environment checks (reported; the specification does not change in response)."""
    t = p[p["thrown"].astype(bool) & p["exclusion_reason"].isna()]
    called = t["description"].eq("called_strike")
    den = called | t["description"].isin(["ball", "blocked_ball"])
    zone = t["zone"].astype("float").map(lambda z: str(int(z)) if z in set(range(1, 10)) | {11, 12, 13, 14} else "missing or other")
    cs = pd.DataFrame({"zone": zone[den], "called": called[den]}).groupby("zone")["called"].agg(["size", "sum"])
    cs = cs.rename(columns={"size": "denominator", "sum": "called_strikes"}).reset_index()
    cs["rate"] = cs["called_strikes"] / cs["denominator"]
    cs.insert(0, "season", season)
    total = {"season": season, "zone": "all", "denominator": int(den.sum()), "called_strikes": int(called.sum()),
             "rate": float(called.sum() / den.sum()) if den.sum() else float("nan")}
    cs = pd.concat([cs, pd.DataFrame([total])], ignore_index=True)
    walks = {"season": season, "pas": len(a), "walks": int(a["events"].eq("walk").sum()),
             "intentional_walks": int(a["events"].eq("intent_walk").sum())}
    walks.update(walk_rate=walks["walks"] / len(a), intentional_walk_rate=walks["intentional_walks"] / len(a))
    last = p.sort_values(["pa_id", "pitch_number"]).groupby("pa_id").tail(1)
    w = last[last["events"].eq("walk")]
    k = last[last["events"].isin(["strikeout", "strikeout_double_play"])]
    terminal = {"season": season, "walk_pas": len(w),
                "walk_final_not_taken_at_three_balls": int((~(w["balls"].eq(3) &
                                                             w["description"].isin(loader.BALL_DESCRIPTIONS))).sum()),
                "strikeout_pas": len(k), "strikeout_final_not_at_two_strikes": int((~k["strikes"].eq(2)).sum())}
    seq = {"season": season, "row_share": float(p["pa_sequence_error"].astype(bool).mean()),
           "pa_share": float(a["pa_sequence_error"].astype(bool).mean())}
    return {"called_strike": cs, "walks": walks, "pitch_mix": pitch_type_shares(p, season), "sequence_errors": seq,
            "terminal": terminal}


def target_sample_label(year: int) -> str:
    """The target sample's label in the comparison tables. Smoke: the stand-in is a development season, so it gets its
    own label and never shares one with the full-season reference (Astra, revision 9)."""
    return f"{year} (smoke stand-in)" if SMOKE else str(year)


def comparison_tables(year: int, label: str, pitches, pas, dev_p, dev_a) -> dict:
    """Coverage and the environment tables for the target sample and each development season. Every row carries a
    `sample` label; coverage shares are computed within each sample. Labels must be distinct (checked)."""
    samples = [(label, pitches, pas, year)] + [(str(s_), dev_p[dev_p["season"] == s_], dev_a[dev_a["season"] == s_], s_)
                                               for s_ in DEV_SEASONS]
    labels = [s_[0] for s_ in samples]
    if len(set(labels)) != len(labels):
        raise ValueError(f"comparison samples share a label: {labels}")
    cov = pd.concat([research.coverage(p, [se]).assign(sample=lb) for lb, p, _, se in samples], ignore_index=True)
    cov["share"] = cov["pitches"] / cov.groupby("sample")["pitches"].transform("sum")
    env = {lb: environment_tables(p, a, se) for lb, p, a, se in samples}
    front = lambda df, lb: df.assign(sample=lb)[["sample"] + list(df.columns)]  # noqa: E731
    return {"coverage": cov[["sample"] + [c for c in cov.columns if c != "sample"]],
            "called_strike": pd.concat([front(env[k]["called_strike"], k) for k in env], ignore_index=True),
            "walks": pd.DataFrame([{"sample": k, **env[k]["walks"]} for k in env]),
            "pitch_mix": pd.concat([front(env[k]["pitch_mix"], k) for k in env], ignore_index=True),
            "sequence_errors": pd.DataFrame([{"sample": k, **env[k]["sequence_errors"]} for k in env]),
            "terminal": pd.DataFrame([{"sample": k, **env[k]["terminal"]} for k in env])}


def gate_vocabulary() -> set:
    """The development play-by-play vocabulary recorded by gate_v<MAPPING_VERSION> (§8.1)."""
    rec = load_record(f"gate_v{MAPPING_VERSION}")
    return {(v["type"], v["isPitch"], v["code"], v["call"], v["eventType"]) for v in rec["result"]["vocabulary"]}


def nonpitch_runs_after_last_pitch(pbp: dict):
    """(home, away) runs MLB's play-by-play records on events that are not Statcast rows and come after the game's
    last kept event (for example a walk-off balk or an error on a pickoff); None when the feed cannot be read so."""
    try:
        plays, last = pbp["allPlays"], None
        for i, p in enumerate(plays):
            for j, e in enumerate(p["playEvents"]):
                if kept(e):
                    last = (i, j)
        if last is None:
            return None
        home = away = 0
        for i, p in enumerate(plays):
            half = p["about"].get("halfInning")
            for r in p.get("runners") or []:
                if not isinstance(r, dict) or (r.get("movement") or {}).get("end") != "score" \
                        or (r.get("movement") or {}).get("isOut") is True \
                        or (r.get("details") or {}).get("isScoringEvent") is not True:
                    continue          # Neon: all 42,957 development scoring records are isOut False, isScoringEvent True
                j = (r.get("details") or {}).get("playIndex")
                if not isinstance(j, int) or isinstance(j, bool) or not 0 <= j < len(p["playEvents"]) \
                        or half not in ("top", "bottom"):
                    return None
                if (i, j) > last and not kept(p["playEvents"][j]):
                    if half == "top":
                        away += 1
                    else:
                        home += 1
        return home, away
    except Exception:  # noqa: BLE001
        return None


def completeness_games(comp: pd.DataFrame) -> list:
    games = sorted(int(g) for g in comp.index)
    return [g for g in games if g % 5 == 0][:100] if SMOKE else games


def classify_completeness(s: dict) -> str:
    """§8.2 step 5's precedence, applied to one season's findings (for 2023 every outcome but a pass stops the path for
    the user's decision, §5)."""
    if s["duplicate_statcast_keys"] or s["unexpected_statcast_games"]:
        return "stop: data integrity"
    confirmed = [m for m in s["key_mismatch"] if m["n_missing"] > 0]
    if s["feed_stop"] or s["score_unexplained"] or s["unavailable"] or len(confirmed) < len(s["key_mismatch"]):
        return "blocked for review"          # the last: Statcast rows the mapping does not expect, none missing
    if s["missing_games"] or confirmed:
        return "incomplete: confirmed missing event or game"
    return "pass"


def season_completeness(year: int, r: pd.DataFrame, vocab_dev: set) -> dict:
    """§5 / §8.1 on a target season: completed-game definition, score check (with the non-pitch explanation), key-level
    event coverage under the current mapping, and the feed-mapping stop: an event signature outside the gate record's
    development vocabulary, or a kept no-pitch event the development gate would flag as unrecognized (the same
    classifier, Astra)."""
    sch = json.loads(fetch(SCHEDULE_URL.format(season=year), MLB_CACHE / "schedule" / f"{year}.json", valid_schedule))
    comp = completed_games(sch)
    games = completeness_games(comp)
    statcast = {g: set(map(tuple, x[["at_bat_number", "pitch_number"]].to_numpy())) for g, x in r.groupby("game_pk")}
    smax = r.groupby("game_pk")[["post_home_score", "post_away_score"]].max()
    s = {"completed": len(comp), "checked": len(games), "duplicate_statcast_keys":
         int(r.duplicated(["game_pk", "at_bat_number", "pitch_number"]).sum()),
         "unexpected_statcast_games": sorted(int(g) for g in set(r["game_pk"]) - set(comp.index)),
         "unavailable": [], "feed_stop": [], "missing_games": [], "key_mismatch": [], "score_explained": [],
         "score_unexplained": [], "new_signatures": {}}

    def one(g):
        try:
            return g, json.loads(fetch(PBP_URL.format(pk=g), MLB_CACHE / "pbp" / f"{g}.json", valid_pbp)), None
        except Exception as exc:  # noqa: BLE001
            return g, None, str(exc)
    with ThreadPoolExecutor(max_workers=FETCH_THREADS) as ex:
        for g, pbp, err in ex.map(one, games):
            if err:
                s["unavailable"].append({"game_pk": g, "error": err})
                continue
            new = sorted(set(vocabulary(pbp)) - vocab_dev, key=str)
            unrec = sorted({event_signature(e) for p in pbp["allPlays"] for e in p["playEvents"] if unrecognized(e)}, key=str)
            if new or unrec:
                s["feed_stop"].append({"game_pk": g, "signatures_outside_vocabulary": [list(x) for x in new],
                                       "unrecognized_kept_no_pitch": [list(x) for x in unrec]})
                for x in new:
                    s["new_signatures"][str(x)] = s["new_signatures"].get(str(x), 0) + 1
            if g not in statcast:
                s["missing_games"].append(g)
                continue
            kc = key_check(expected_keys(pbp), statcast[g])
            if not kc["ok"]:
                s["key_mismatch"].append({"game_pk": g, **kc})
                continue
            c = comp.loc[g]
            sh, sa = smax.loc[g, "post_home_score"], smax.loc[g, "post_away_score"]
            if sh == c["home_score"] and sa == c["away_score"]:
                continue
            extra = nonpitch_runs_after_last_pitch(pbp)
            num = lambda x: None if pd.isna(x) else int(x)  # noqa: E731
            row = {"game_pk": g, "statcast": [num(sh), num(sa)], "official": [num(c["home_score"]), num(c["away_score"])],
                   "nonpitch_runs_after_last_pitch": list(extra) if extra else None}
            ok = extra is not None and sum(extra) > 0 and (sh + extra[0], sa + extra[1]) == (c["home_score"], c["away_score"])
            s["score_explained" if ok else "score_unexplained"].append(row)
    s["outcome"] = classify_completeness(s)
    s["ok"] = s["outcome"] == "pass"
    return s


def stage_checks(year: int):
    season_gate(year, "checks")

    def body():
        bad = verify_inputs(f"extract_{year}", f"build_{year}")
        if bad:
            return {"passed": False, "stopped": f"inputs changed: {bad}"}
        sd = season_dir(year)
        raw_all = pd.read_parquet(sd / "raw.parquet", columns=CHECK_RAW_COLUMNS)
        raw_all = smoke_subset(raw_all)
        r = raw_all[raw_all["game_type"] == "R"].reset_index(drop=True)
        pitches, pas = pd.read_parquet(sd / "pitches.parquet"), pd.read_parquet(sd / "pas.parquet")
        dev_p, dev_a = pd.read_parquet(ORIG["pitches"]), pd.read_parquet(ORIG["pas"])
        dom = load_record("domains")["result"]["domains"]
        res = {"label": season_info(year)["label"]}
        res["B0"] = b0_inventory(raw_all, dom)
        res["B0"]["new_pitch_types"] = sorted(set(res["B0"]["pitch_types"]) - set(map(str, dev_p["pitch_type"].dropna().unique())))
        res["B1"] = b1_counts(r, pitches)
        b2 = b2_table(pitches, year, dev_p, target_sample_label(year))
        res["prefit"] = prefit_checks(r, pitches, raw_all)
        t = time.time()
        res["completeness"] = season_completeness(year, r, gate_vocabulary())
        res["completeness"]["seconds"] = time.time() - t
        tables = {"B2": b2["table"], **comparison_tables(year, target_sample_label(year), pitches, pas, dev_p, dev_a)}
        tpath = sd / "checks_tables.pkl"
        if tpath.exists():
            raise FileExistsError(f"{tpath} exists")
        with open(tpath, "wb") as f:
            pickle.dump(tables, f)
        res["B2_new_in_target"] = b2["new_in_target"]
        res["zone_note"] = "2026 zones are labeled possibly not comparable (§2)" if year == 2026 else None
        res["outputs"] = file_digests([tpath])
        stops = []
        if not res["B0"]["ok"]:
            stops.append("B0")
        if not all(c["ok"] for c in res["prefit"]):
            stops.append("pre-fit")
        if not res["completeness"]["ok"]:
            stops.append(f"completeness ({res['completeness']['outcome']})")
        res["stops"] = stops
        res["passed"] = not stops
        return res
    run_season_stage(year, "checks", body)


# ------------------------------------------------------------------------------------- run, halves, validate (children)

def season_binding(year: int, stage: str, **extra) -> dict:
    sd = season_dir(year)
    out = sd / stage
    return {"raw": str(sd / "raw.parquet"), "pitches": str(sd / "pitches.parquet"), "pas": str(sd / "pas.parquet"),
            "support": str(sd / "support.parquet"), "models_dir": str(out / "models"), "out": str(out),
            "refs": refs(), "season": year, "stage": f"{stage}_{year}", **extra}


def stage_run(year: int):
    season_gate(year, "run")

    def body():
        bad = verify_inputs(f"extract_{year}", f"build_{year}")
        if bad:
            return {"passed": False, "stopped": f"inputs changed: {bad}"}
        c = run_child("season_run", season_binding(year, "run"))
        return {"context": c, "outputs": c.get("outputs", {}), "passed": bool(c.get("passed")) and bool(c.get("outputs"))}
    run_season_stage(year, "run", body)


def stage_halves(year: int):
    season_gate(year, "halves")

    def body():
        bad = verify_inputs(f"extract_{year}", f"build_{year}", f"run_{year}")
        if bad:
            return {"passed": False, "stopped": f"inputs changed: {bad}"}
        c = run_child("season_halves", season_binding(year, "halves"))
        return {"context": c, "outputs": c.get("outputs", {}), "passed": bool(c.get("passed")) and bool(c.get("outputs"))}
    run_season_stage(year, "halves", body)


def stage_validate(year: int):
    season_gate(year, "validate")

    def body():
        bad = verify_inputs(f"extract_{year}", f"build_{year}", f"run_{year}", f"halves_{year}")
        if bad:
            return {"passed": False, "stopped": f"inputs changed: {bad}"}
        c = run_child("season_validate", season_binding(year, "validate"))
        return {"context": c, "outputs": c.get("outputs", {}), "passed": bool(c.get("passed")) and bool(c.get("outputs"))}
    run_season_stage(year, "validate", body)


def season_manifest_checks(version: str, man: dict, spec, year: int) -> list:
    """A season run keeps the frozen specification: id, description, base, params and changes equal the stored
    reference's; the one season; the frozen code digest; the bound inputs; every key compared or permitted."""
    ref = json.loads((Path(refs()[f"{version}_dir"]) / "manifest.json").read_text())
    rows = []

    def add(check, ok, detail=""):
        rows.append({"check": f"{version} {year} manifest: {check}", "identical": bool(ok), "detail": detail})
    j = lambda x: json.dumps(x, sort_keys=True, default=str)  # noqa: E731
    known = set(MANIFEST_EXACT) | set(MANIFEST_RECORDS) | {"id", "changes", "seasons", "inputs", "fit_ids"} | \
        set(C1_RECORDS) | set(PERMITTED)
    add("every key known", not (set(man) - known), sorted(set(man) - known))
    add("id kept", man.get("id") == version == ref.get("id"))
    for k in MANIFEST_EXACT:
        add(f"{k} equals the stored reference's", j(man.get(k)) == j(ref.get(k)))
    add("seasons is the one season", man.get("seasons") == [year])
    add("code hash is the frozen full digest", man.get("code", {}).get("gamecalling_sha256") == FULL_CODE_HASH)
    add("inputs are the bound files (names, bytes, full digests)",
        j(man.get("inputs")) == j(research.input_state(spec.extra_inputs)))
    want = {Path(BIND[k]).name for k in ("pitches", "pas", "support")} | {Path(x).name for x in spec.extra_inputs}
    add("input names are the bound files", set(man.get("inputs", {})) == want and
        {Path(x).resolve() for x in spec.extra_inputs} <= {Path(BIND["raw"]).resolve()})
    if version == "P0":
        rc, nc = ref.get("changes", {}), man.get("changes", {})
        add("changes on the stored keys", {k: nc.get(k) for k in rc} == rc)
        add("changes.solve is false", nc.get("solve") is False)
        for k, v in P0_DEFAULTS.items():
            add(f"{k} is the empty default", man.get(k) == v)
    else:
        add("changes equal the stored reference's", man.get("changes") == ref.get("changes"))
        fids = man.get("fit_ids", {})
        add("fit ids: this season only, model id kept", bool(fids) and all(k.startswith(f"{year}/") and
                                                                          str(v).startswith("C1/") for k, v in fids.items()))
        av = man.get("action_values", [])
        add("action values: one file per scoring fold, this season", sorted((r["season"], r["scoring_fold"]) for r in av)
            == [(year, k) for k in range(config.N_FOLDS)])
    for k in MANIFEST_RECORDS:
        recs = man.get(k, [])
        add(f"{k}: present, this season only", bool(recs) and all(r.get("season") == year for r in recs))
    return rows


def probability_checks(version: str, d: Path) -> list:
    """§7's probability checks, applied to the held-out files: finite, in [0, 1], rows summing to 1 within 1e-12."""
    rows = []
    for kind in ("outcome", "contact"):
        f = pd.read_parquet(d / f"heldout_{kind}.parquet")
        P = f.filter(regex=r"^p\d+$").to_numpy(float)
        ok = bool(len(P) and np.isfinite(P).all() and (P >= 0).all() and (P <= 1).all()
                  and np.abs(P.sum(axis=1) - 1).max() <= 1e-12)
        rows.append({"check": f"{version} heldout_{kind}: probabilities finite, in [0, 1], rows sum to 1 (1e-12)",
                     "identical": ok, "detail": f"{len(P)} rows"})
    return rows


def model_outputs(d: Path) -> list:
    return [d / f for f in ("scores.parquet", "heldout_outcome.parquet", "heldout_contact.parquet", "coverage.csv",
                            "manifest.json")]


def ctx_season_run():
    """§6 row 1: the full scored run of P0 and C1 within the season, then the post-run checks: C1's coverage-first
    reconstruction, P0 and C1 on the same keys, the manifests, and the probability checks."""
    year = int(BIND["season"])
    res = {"probe": probe_check(BIND), "checks": [], "versions": {}}
    mans, outs = {}, []
    for version in ("P0", "C1"):
        man, spec = run_version(version, (year,), f"{version}_{year}", f"s9_{year}", extra_inputs=(config.RAW_PARQUET,))
        mans[version] = man
        d = research.MODELS_DIR / f"{version}_{year}"
        res["checks"] += season_manifest_checks(version, man, spec, year)
        res["checks"] += probability_checks(version, d)
        res["versions"][version] = {"scored": len(pd.read_parquet(d / "scores.parquet", columns=["season"])),
                                    "abnormal_log": man.get("abnormal_log", []),
                                    "coverage": pd.read_csv(d / "coverage.csv").to_dict("records")}
        outs += model_outputs(d)
    sc = {v: pd.read_parquet(research.MODELS_DIR / f"{v}_{year}" / "scores.parquet") for v in ("P0", "C1")}
    rows = bs.read_action_values(mans["C1"]["action_values"])
    rc = reconstruction_check(rows, sc["C1"])
    res["reconstruction_C1"] = rc
    res["checks"].append({"check": "C1 post-run reconstruction (coverage first, then exact)", "identical": rc["ok"],
                          "detail": rc})
    key = lambda s: s[bs.PITCH_KEYS].sort_values(bs.PITCH_KEYS).reset_index(drop=True)  # noqa: E731
    res["checks"].append({"check": "P0 and C1 scored the same keys", "identical": bool(key(sc["P0"]).equals(key(sc["C1"])))})
    outs += [Path(r["path"]) for r in mans["C1"]["action_values"]]
    res["outputs"] = file_digests(outs)
    res["passed"] = res["probe"]["ok"] and all(c["identical"] for c in res["checks"])
    ctx_result(res, strict=True)


def read_halves(path) -> tuple:
    hv = pd.read_parquet(path)
    return hv["half"].astype("int8").rename("half"), hv["calendar_half"].astype("int8").rename("calendar_half")


def ctx_season_halves():
    """§6 row 5's independent refit: the whole pipeline inside each saved half (P0: vp._half_pipeline; C1:
    research_s6_full._half_job), as the control's C6 runs them."""
    year = int(BIND["season"])
    out = Path(BIND["out"])
    res = {"probe": probe_check(BIND), "checks": []}
    halves, _ = read_halves(season_dir(year) / "halves.parquet")     # derived, not bound (Astra)
    p, _, _ = research.base_tables()
    games = set(p.loc[p["season"] == year, "game_pk"])
    jobs = [(year, h, np.sort(halves.index[(halves == h).to_numpy() & halves.index.isin(games)])) for h in (0, 1)]
    with ProcessPoolExecutor(max_workers=4) as ex:
        f0 = [ex.submit(vp._half_pipeline, j) for j in jobs]
        f1 = [ex.submit(sf._half_job, (c1_spec(), s, h, g, f"s9_half_{s}_{h}")) for s, h, g in jobs]
        p0 = {(j[0], j[1]): f.result() for j, f in zip(jobs, f0)}
        c1 = {(j[0], j[1]): f.result() for j, f in zip(jobs, f1)}
    paths = {"P0": out / "halves_P0.pkl", "C1": out / "halves_C1.pkl"}
    for k, obj in (("P0", p0), ("C1", {"runs": c1})):
        if paths[k].exists():
            raise FileExistsError(f"{paths[k]} exists")
        with open(paths[k], "wb") as f:
            pickle.dump(obj, f)
    for key in sorted(p0):
        res["checks"].append({"check": f"P0 half {key} scored pitches", "identical": len(p0[key]["scores"]) > 0,
                              "detail": len(p0[key]["scores"])})
        res["checks"].append({"check": f"C1 half {key} scored pitches", "identical": len(c1[key]["scores"]) > 0,
                              "detail": len(c1[key]["scores"])})
    res["jobs"] = {"P0": [p0[k]["meta"] for k in sorted(p0)], "C1": [c1[k]["meta"] for k in sorted(c1)]}
    res["outputs"] = file_digests(paths.values())
    res["passed"] = res["probe"]["ok"] and all(c["identical"] for c in res["checks"])
    ctx_result(res, strict=True)


def calibration_or_note(pred: np.ndarray, hit: np.ndarray):
    """Calibration error (10 equal-count bins), or the §6 note when there are fewer than CAL_MIN_N rows."""
    return validation.calibration_error(pred, hit) if len(pred) >= CAL_MIN_N else f"not computed (n < {CAL_MIN_N})"


def heldout_baselines(pitches: pd.DataFrame, year: int, kind: str) -> tuple:
    """validation._held_out's baselines with their keys, per scoring fold: the criterion baseline (outcome: league count
    x action with the count-marginal prior; contact: league action smoothed toward overall frequencies; an action absent
    from the baseline's rows gets the overall frequencies) and P0's league-pooled level, from P0's valuation models
    refitted on the valuation fold; plus those models' own predictions (which must equal P0's held-out file)."""
    crit, league, p0 = [], [], []
    for k in range(config.N_FOLDS):
        val = arsenal.fold_roles(k)[1]
        fold = pitches[(pitches["season"] == year) & (pitches["fold"] == k) & pitches["scorable"]]
        train = pitches[(pitches["season"] == year) & (pitches["fold"] == val)]
        if kind == "outcome":
            obs, tr, model = models.outcome_training(fold), models.outcome_training(train), models.fit_outcome_model(train)
            K, keys = models.N_OUTCOMES, ["cidx", "action"]
            base = validation._league_baseline(tr, keys, ["cidx"], K)
        else:
            obs, tr, model = models.contact_training(fold), models.contact_training(train), models.fit_contact_model(train)
            K, keys = models.N_BINS, ["action"]
            n = models._counts(tr, keys, K)
            base = (n + n.sum(axis=0) / n.to_numpy().sum()).div(n.sum(axis=1) + 1, axis=0)
        bp = models._lookup(base, obs, keys, fill=np.nan)
        overall = models._counts(tr.assign(_all=0), ["_all"], K).to_numpy()[0]
        bp = np.where(np.isnan(bp), overall / overall.sum(), bp)
        crit.append(research._pred_frame(fold, obs, bp, fold=k))
        league.append(research._pred_frame(fold, obs, model.league_mean(obs), fold=k))
        p0.append(research._pred_frame(fold, obs, model.predict(obs), fold=k))
    cat = lambda x: pd.concat(x, ignore_index=True)  # noqa: E731
    return cat(crit), cat(league), cat(p0)


PROB_SUM_TOL = 1e-12                 # §7 probability checks: rows sum to 1 within 1e-12


def prob_matrix_problems(f: pd.DataFrame) -> list:
    """The §7 probability checks on one prediction frame: finite, in [0, 1], rows summing to 1 within 1e-12."""
    P = f.filter(regex=r"^p\d+$").to_numpy(float)
    bad = []
    if P.size == 0:
        return ["no probability columns"]
    if not np.isfinite(P).all():
        bad.append("non-finite probability")
    elif (P < 0).any() or (P > 1).any():
        bad.append("probability outside [0, 1]")
    elif np.abs(P.sum(axis=1) - 1.0).max() > PROB_SUM_TOL:
        bad.append(f"row sum differs from 1 by more than {PROB_SUM_TOL}")
    return bad


def heldout_metrics(kind: str, b, c, crit, league, p0r, counts=None) -> tuple:
    """One component of §6 row 2. Every equality check comes first; if any fails, no metric is computed (Astra: a
    detected mismatch must not yield log loss or calibration on the remaining rows). Returns (row or None, checks).

    Revision 9 (the 2024 smoke run exposed it): the frozen criterion baseline (`validation._league_baseline`) can give
    an observed class probability exactly 0 when the training fold has no such outcome at that count. Its log loss is
    then +inf, reported as such with every zero-probability row disclosed; nothing is floored and the analyses go on.
    Only an exact 0 from a probability matrix that passes the §7 checks is allowed, and only for the criterion
    baseline; P0, C1 and the league-pooled level must have finite log probabilities. `counts` (keys + cidx) labels the
    disclosed rows by count."""
    with np.errstate(divide="ignore"):       # log(0) is reported, not warned about
        return _heldout_metrics(kind, b, c, crit, league, p0r, counts)


def _heldout_metrics(kind, b, c, crit, league, p0r, counts) -> tuple:
    checks = []
    r = frames_compare(p0r, b, KEYS, bitwise=True, same_order=False)
    checks.append({"check": f"P0's valuation models refitted per fold reproduce P0's heldout_{kind} (bitwise)",
                   "identical": r["identical"], "detail": r["detail"]})
    cols = KEYS + ["fold", "y"]
    srt = lambda f: f[cols].sort_values(KEYS).reset_index(drop=True)  # noqa: E731
    same4 = all(srt(x).equals(srt(b)) for x in (c, crit, league))
    checks.append({"check": f"{kind}: P0, C1 and both baselines have the same keys, folds and observed classes",
                   "identical": same4, "detail": {"P0": len(b), "C1": len(c), "criterion": len(crit), "league": len(league)}})
    if not (r["identical"] and same4):
        return None, checks
    m = research.matched(b, c)
    checks.append({"check": f"{kind}: the matched set is every held-out row (no silent drop)",
                   "identical": len(m) == len(b) == len(c), "detail": len(m)})
    probs = {nm: prob_matrix_problems(f) for nm, f in (("P0", b), ("C1", c), ("criterion", crit), ("league", league))}
    checks.append({"check": f"{kind}: every probability matrix finite, in [0, 1], rows summing to 1 within "
                            f"{PROB_SUM_TOL} (P0, C1, both baselines)",
                   "identical": not any(probs.values()), "detail": {k: v for k, v in probs.items() if v} or "all valid"})
    if not all(x["identical"] for x in checks):
        return None, checks
    lp = lambda f, name: f[KEYS].assign(**{name: research._logp(f)})  # noqa: E731
    m = m.merge(lp(crit, "lp_crit"), on=KEYS, how="left", validate="one_to_one").merge(
        lp(league, "lp_league"), on=KEYS, how="left", validate="one_to_one")
    finite3 = bool(np.isfinite(m[["lp_base", "lp_cand", "lp_league"]].to_numpy(float)).all())
    checks.append({"check": f"{kind}: P0, C1 and league-pooled log probabilities all finite", "identical": finite3,
                   "detail": len(m)})
    cr = crit.merge(m[KEYS], on=KEYS, how="inner", validate="one_to_one")
    p_obs = cr.filter(regex=r"^p\d+$").to_numpy(float)[np.arange(len(cr)), cr["y"].to_numpy()]
    zero = p_obs == 0.0
    lpc = cr[KEYS].assign(lp=np.log(p_obs)).merge(m[KEYS + ["lp_crit"]], on=KEYS, validate="one_to_one")
    crit_ok = bool(len(cr) == len(m) and (lpc["lp"].to_numpy() == lpc["lp_crit"].to_numpy()).all()
                   and (np.isfinite(np.log(p_obs)) | zero).all())
    zrows = cr.loc[zero, KEYS + ["fold", "y"]].reset_index(drop=True)
    if counts is not None and len(zrows):
        zrows = zrows.merge(counts[KEYS + ["cidx"]], on=KEYS, how="left", validate="one_to_one")
    names = list(solver.OUTCOMES) if kind == "outcome" else [f"bin_{x}" for x in payoff.CONTACT_BINS]
    zrows["class"] = [names[int(v)] for v in zrows["y"]]
    by = ["fold"] + (["cidx"] if "cidx" in zrows else []) + ["y", "class"]
    zero_detail = {"component": kind, "zero_probability_rows": int(zero.sum()), "n": len(m),
                   "breakdown": zrows.groupby(by).size().rename("rows").reset_index().to_dict("records")
                   if len(zrows) else [],
                   "rows": zrows.head(50).to_dict("records"), "rows_listed": min(len(zrows), 50)}
    checks.append({"check": f"{kind}: criterion baseline log probabilities finite except exact zero probabilities on "
                            f"observed rows (log loss reported as +inf, rows disclosed; revision 9)",
                   "identical": crit_ok, "detail": zero_detail, "criterion_zero": zero_detail})
    if not all(x["identical"] for x in checks):
        return None, checks
    row = {"component": kind, "n": len(m), "P0": -m["lp_base"].mean(), "C1": -m["lp_cand"].mean(),
           "gain_C1_minus_P0": (m["lp_cand"] - m["lp_base"]).mean(), "criterion_baseline": -m["lp_crit"].mean(),
           "criterion_zero_probability_rows": int(zero.sum()), "league_pooled_P0": -m["lp_league"].mean()}
    for ver, f in (("P0", b), ("C1", c), ("criterion_baseline", crit), ("league_pooled_P0", league)):
        g = f.merge(m[KEYS], on=KEYS, how="inner")
        P, y = g.filter(regex=r"^p\d+$").to_numpy(), g["y"].to_numpy()
        cals = [calibration_or_note(P[:, jx], y == jx) for jx in range(P.shape[1])]
        for nm, v in zip(names, cals):
            row[f"cal_{ver}_{nm}"] = v
        num = [v for v in cals if not isinstance(v, str)]
        row[f"max_cal_{ver}"] = max(num) if num else f"not computed (n < {CAL_MIN_N})"
    return row, checks


def heldout_table(pitches: pd.DataFrame, year: int, p0_dir: Path, c1_dir: Path) -> tuple:
    """§6 row 2: P0 and C1 held-out prediction on one matched set per component, with both baselines; log loss and
    calibration by component. Returns (table, checks, zero): a component whose checks fail has no row; `zero` lists
    each component's criterion-baseline zero-probability disclosure (revision 9)."""
    rows, checks, zero = [], [], []
    counts = pitches[KEYS + ["balls", "strikes"]].assign(
        cidx=lambda d: d["balls"].astype(int) * 3 + d["strikes"].astype(int))[KEYS + ["cidx"]]
    for kind in ("outcome", "contact"):
        b, c = pd.read_parquet(p0_dir / f"heldout_{kind}.parquet"), pd.read_parquet(c1_dir / f"heldout_{kind}.parquet")
        crit, league, p0r = heldout_baselines(pitches, year, kind)
        row, ch = heldout_metrics(kind, b, c, crit, league, p0r, counts)
        checks += ch
        zero += [x["criterion_zero"] for x in ch if "criterion_zero" in x]
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows), checks, zero


def json_safe(obj):
    """Recursively labels non-finite floats (+inf, -inf; NaN as null), so a JSON record parses with a strict reader
    (Astra, revision 9). Finite values and every other type are unchanged."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj):
            return None
        if np.isinf(obj):
            return "+inf" if obj > 0 else "-inf"
        return float(obj)
    return obj


def json_records(df: pd.DataFrame) -> list:
    """A frame's records through json_safe."""
    return json_safe(df.to_dict("records"))


def season_analyses(scores: pd.DataFrame, year: int, halves, calendar, runs) -> dict:
    """§6 rows 3-5, 8, 9 within one season (validation_pilot with SEASONS = (year,) for the duration of the call)."""
    old_seasons, old_scores = vp.SEASONS, vp._DATA.get("scores")
    try:
        vp.SEASONS = (year,)
        vp._DATA["scores"] = scores
        res = {"split_random": vp.within_split(scores, halves, "split of existing scores (random halves)"),
               "split_calendar": vp.within_split(scores, calendar, "split of existing scores (calendar halves; secondary)"),
               "refit": vp.within_refit_compare(runs)}
        fits, sets = vp.season_fits(), vp.season_sets()
        res["sets"] = {k: {"qual": len(v["qual"]), "main": len(v["main"])} for k, v in sets.items()}
        res["sensitivity"] = vp.sensitivity(fits, sets)
        c = scores.groupby(["catcher", "fielding_team"]).agg(pitches=("game_pk", "size"),
                                                             games=("game_pk", "nunique")).reset_index()
        prim = {year: vp.primary_teams(c)}
        res["team_staff"] = vp.team_staff(fits, sets, prim)
        res["season_sd"] = pd.DataFrame([{"season": se, "score": sc, "model": "fixed" if fx else "random", "sd": f["sd"],
                                          "zero": f["zero"], "converged": f["converged"]}
                                         for se, ff in fits.items() for (sc, fx), f in ff.items()])
        res["spread"] = spread_table(fits, sets, scores)
        res["effects"] = {f"{sc}_{'fixed' if fx else 'random'}": f["saved"] for (sc, fx), f in fits[year].items()}
        return res
    finally:
        vp.SEASONS = old_seasons
        if old_scores is None:
            vp._DATA.pop("scores", None)
        else:
            vp._DATA["scores"] = old_scores


def pair_repeatability(combined: pd.DataFrame, pair: tuple) -> dict:
    """§6 rows 6-7 for one pair: random and fixed fits, unweighted Pearson and Spearman, weighted Pearson, median
    absolute change, disconnected catchers listed, and repeatability by mover group from the pair's primary teams."""
    prim, movers = primary_and_movers(combined, pair)
    old_seasons, old_scores = vp.SEASONS, vp._DATA.get("scores")
    try:
        vp.SEASONS = tuple(pair)
        vp._DATA["scores"] = combined
        fits, sets = vp.season_fits(), vp.season_sets()
        rep = vp.repeatability(fits, sets, movers)
    finally:
        vp.SEASONS = old_seasons
        if old_scores is None:
            vp._DATA.pop("scores", None)
        else:
            vp._DATA["scores"] = old_scores
    elig = movers.reindex(rep["eligible"])
    return {"pair": list(pair), "sets": {k: {"qual": len(v["qual"]), "main": len(v["main"])} for k, v in sets.items()},
            "repeatability": rep, "movers": movers, "primary": prim,
            "mover_counts_eligible": elig["group"].value_counts(dropna=False).to_dict(),
            "small_groups": small_group_listing(rep, movers)}


def small_group_listing(rep: dict, movers: pd.DataFrame) -> pd.DataFrame:
    """§6 row 7 / protocol §6.1: a mover group under MIN_SET catchers is listed, not correlated: each catcher with the
    pair's two estimates and the change, for D and A (random fits), over the pair's eligible catchers."""
    rows = []
    elig = sorted(rep["eligible"])
    for grp in ("mover", "non_mover", "neither"):
        ids = sorted(set(movers.index[movers["group"] == grp]) & set(elig))
        if len(ids) >= vp.MIN_SET:
            continue
        for sc in ("D", "A"):
            e = rep["effects"][sc, "random"].reindex(ids)
            for c_ in ids:
                rows.append({"group": grp, "score": sc, "catcher": c_, "saved_0": e.loc[c_, "y0"], "saved_1": e.loc[c_, "y1"],
                             "change": e.loc[c_, "y1"] - e.loc[c_, "y0"]})
    return pd.DataFrame(rows, columns=["group", "score", "catcher", "saved_0", "saved_1", "change"])


def ctx_season_validate():
    """§6 rows 2-9 for both versions: held-out prediction, the season analyses, and every pair the season belongs to
    (the other season's scores are the stored ones of the same version)."""
    year = int(BIND["season"])
    out = Path(BIND["out"])
    sd = season_dir(year)          # every input derived from the season, never from the binding (Astra)
    run_dir, halves_dir, pairs = sd / "run" / "models", sd / "halves", [tuple(p_) for p_ in PAIRS[year]]
    res = {"probe": probe_check(BIND), "checks": [], "summary": {}, "inputs": {"run_dir": str(run_dir),
                                                                             "halves_dir": str(halves_dir),
                                                                             "pairs": [list(p_) for p_ in pairs]}}
    pitches = research.base_tables()[0]
    halves, calendar = read_halves(sd / "halves.parquet")
    full = {}
    tab, ch, zero = heldout_table(pitches, year, run_dir / f"P0_{year}", run_dir / f"C1_{year}")
    full["heldout"], full["heldout_zero"] = tab, zero
    res["checks"] += ch
    res["heldout_zero"] = zero
    if not res["probe"]["ok"] or not all(c_["identical"] for c_ in ch) or not pairs:
        res.update(passed=False, stopped="held-out checks, probe or prescribed pairs failed; no analysis was run "
                                         "and nothing was written")
        ctx_result(res, strict=True)
        return
    for version in ("P0", "C1"):
        scores = with_ab(pd.read_parquet(run_dir / f"{version}_{year}" / "scores.parquet"), config.PITCHES_PARQUET)
        res["checks"].append({"check": f"{version}: every scored pitch has at_bat_number", "identical":
                              bool(scores["at_bat_number"].notna().all())})
        with open(halves_dir / f"halves_{version}.pkl", "rb") as f:
            runs = pickle.load(f)
        runs = runs["runs"] if version == "C1" else runs
        v = {"season": season_analyses(scores, year, halves, calendar, runs), "pairs": {}}
        for pair in pairs:
            other = [s_ for s_ in pair if s_ != year][0]
            spath, ppath = other_reference(version)
            o = with_ab(pd.read_parquet(spath), ppath)
            o = o[o["season"] == other].reset_index(drop=True)
            same_cols = list(o.columns) == list(scores.columns) and list(o.dtypes) == list(scores.dtypes)
            res["checks"].append({"check": f"{version} pair {pair}: stored {other} scores have the same columns and dtypes",
                                  "identical": same_cols})
            if not same_cols:
                continue
            combined = pd.concat([scores, o], ignore_index=True).sort_values(KEYS, kind="stable").reset_index(drop=True)
            v["pairs"][tuple(pair)] = pair_repeatability(combined, tuple(pair))
        res["checks"].append({"check": f"{version}: every prescribed pair computed", "identical":
                              set(v["pairs"]) == set(pairs), "detail": [list(k) for k in v["pairs"]]})
        full[version] = v
        sp = v["season"]["spread"]
        res["summary"][version] = {"spread": json_records(sp), "sets": v["season"]["sets"],
                                   "pairs": {str(k): json_records(p["repeatability"]["table"])
                                             for k, p in v["pairs"].items()}}
    vpath = out / "validate.pkl"
    if vpath.exists():
        raise FileExistsError(f"{vpath} exists")
    with open(vpath, "wb") as f:
        pickle.dump(full, f)
    res["heldout"] = json_records(tab)
    res["outputs"] = file_digests([vpath])
    res["passed"] = res["probe"]["ok"] and all(c["identical"] for c in res["checks"])
    ctx_result(res, strict=True)


# ------------------------------------------------------------------------------------- report

RULE_NOTES = {2023: [
    "Runners-on pitch timer 20 s (18 s from 2024); five mound visits (four from 2024); a narrower runner's lane.",
    "Pitch timer, shift limits, larger bases and disengagement limits since 2023; pitchers may wear PitchCom "
    "transmitters and call their own pitches since 2023."]}
PAIR_LABEL = {(2023, 2024): "Combines one previously unmodeled season (2023) with one development season (2024).",
              (2024, 2025): "Smoke stand-in pair (development seasons)."}
LEFT_OUT = ("the S1 (a)-(e) reruns", "the interpretation examples", "the D-vs.-A evidence-table update",
            "bridge removal", "the recovery simulation", "forward prediction for 2023 (A4)")


def md_table(df: pd.DataFrame, digits: int = 4) -> str:
    """A plain Markdown table (no tabulate dependency)."""
    if df is None or len(df) == 0:
        return "_(none)_\n"

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return ""
            if np.isinf(v):
                return "+inf" if v > 0 else "-inf"
            return f"{v:.{digits}f}"
        return str(v)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def zero_probability_note(zero: list) -> list:
    """Revision 9 disclosure: the frozen criterion baseline gave an observed class probability exactly 0."""
    hit = [z for z in zero if z["zero_probability_rows"]]
    if not hit:
        return []
    why = {"outcome": "the outcome baseline (league count x action with the count-marginal prior, "
                      "`validation._league_baseline`) gives a class probability exactly 0 when the valuation fold has no "
                      "such outcome at that count",
           "contact": "the contact baseline (league action smoothed toward overall contact-class frequencies) gives a "
                      "class probability exactly 0 only when the valuation fold has no ball in play in that class at all"}
    L = ["**Criterion baseline: zero probabilities (disclosure, revision 9).** The frozen criterion baseline is unchanged. "
         "Its log loss is +inf in a component where it gave an observed class probability exactly 0, reported as such; "
         "nothing is floored, and its calibration is computed as usual. Exposed by the 2024 smoke run on 2026-11-22.", ""]
    for z in hit:
        L += [f"- {z['component']}: {z['zero_probability_rows']} of {z['n']} rows; {why[z['component']]}.", "",
              md_table(pd.DataFrame(z["breakdown"])), md_table(pd.DataFrame(z["rows"]))]
    return L


def report_text(year: int, checks: dict, tables: dict, run: dict, full: dict) -> str:
    info = season_info(year)
    L = [f"# S9 season {year}: P0 and C1 within-season results and pairs", "",
         f"**Season label:** {info['label']}. **Status:** exploratory point estimates; descriptive only, no intervals or "
         f"significance. Catcher-associated differences; who called each pitch is unknown. D primary, A at equal "
         f"prominence, B diagnostic (D = A + B; B does not depend on the pitch thrown once the situation is fixed).",
         "", f"Written by `s9_seasons.py report {year}` on {now()} (mode {MODE}). Protocol: "
         f"`research_notes/research_program/S9_stage_B_protocol.md`.", ""]
    if SMOKE:
        L += ["**SMOKE RUN: a development season standing in for a target, one game in five. Means nothing.**", ""]
    if year in RULE_NOTES:
        L += ["## Compatibility notes (§2; reported with the results, not exclusions)", ""] + \
             [f"- {x}" for x in RULE_NOTES[year]] + [""]
    c = checks["completeness"]
    L += ["## Row checks (§5)", "",
          f"- Completeness: **{c['outcome']}**; {c['completed']} completed games on the schedule, {c['checked']} checked; "
          f"score discrepancies explained by a verified non-pitch scoring event: {len(c['score_explained'])}.",
          f"- B0: duplicate keys in the whole raw file {checks['B0']['duplicate_keys_whole_file']}; domain violations "
          f"{checks['B0']['domain_violations'] or 'none'}; events accepted through the frozen code list only "
          f"{checks['B0']['events_accepted_through_code_only'] or 'none'}; null counts {checks['B0']['nulls']}; "
          f"bb_type nulls {checks['B0']['bb_type_nulls']}.",
          f"- B1 (report): {checks['B1']}.", f"- B2: pitch types not seen in 2024-2025: {checks['B2_new_in_target'] or 'none'}.",
          "- Pre-fit checks: " + ("all passed" if all(x["ok"] for x in checks["prefit"]) else
                                  str([x for x in checks["prefit"] if not x["ok"]])), ""]
    L += ["### Coverage", "", md_table(tables["coverage"]), "### Environment (reported against 2024 and 2025)", "",
          md_table(tables["called_strike"]), md_table(tables["walks"]), md_table(tables["sequence_errors"]),
          md_table(tables["terminal"]), "### Pitch types (B2)", "", md_table(tables["B2"]), ""]
    L += ["## Full run (§6 row 1)", ""]
    for v, d in run["versions"].items():
        L.append(f"- {v}: {d['scored']} scored pitches; abnormal-termination records {len(d['abnormal_log'])}.")
    L += [f"- C1 post-run reconstruction (coverage first): {run['reconstruction_C1']}.", "",
          "## Held-out prediction (§6 row 2; one matched set per component; nats)", "", md_table(full["heldout"], 5),
          "A passing comparison shows better probabilities, not better comparisons between pitch choices.", ""]
    L += zero_probability_note(full.get("heldout_zero", []))
    for v in ("P0", "C1"):
        s = full[v]["season"]
        L += [f"## {v}", "", "### Headline spread (A3): a descriptive model estimate, never a skill distribution", "",
              "`reml_sd`: the REML between-catcher SD (runs per 100 pitches) with its `converged` and `zero` flags. "
              "`shrunken_sd`: the empirical SD (ddof 1) of the shrunken estimates over qualifying catchers (at least "
              "1,000 scorable pitches) in the main group, `n_catchers` of them. `pitch_sd_per100`: A3's third quantity, "
              "kept distinct from both, the pitch-level SD of the score times 100 over the season's scored pitches "
              "(ddof 1).", "",
              md_table(s["spread"], 5), "### Split of existing scores (random halves)", "",
              md_table(pd.DataFrame(s["split_random"][year]["compare"])),
              md_table(pd.DataFrame(s["split_random"][year]["gap"])),
              "### Split of existing scores (calendar halves; secondary)", "",
              md_table(pd.DataFrame(s["split_calendar"][year]["compare"])),
              "### Independent refit in each half", "", md_table(pd.DataFrame(s["refit"][year]["compare"])),
              md_table(pd.DataFrame(s["refit"][year]["gap"])),
              "### Between-team share, staff diagnostic", "", md_table(s["team_staff"]["share"]),
              md_table(s["team_staff"]["staff"]), "### Same-team pairs (protocol §6.4)", "",
              md_table(s["team_staff"]["pairs"]), "### Sensitivity (R1, R2, R3, R4)", "",
              md_table(s["sensitivity"]["table"])]
        if year == 2023:
            L.append("R3 in 2023: no club is flagged, so R3 removes no pitches.\n")
        for pair, p in full[v]["pairs"].items():
            L += [f"### Pair {pair[0]}-{pair[1]}", "", PAIR_LABEL.get(tuple(pair), "") + " The other season's scores are "
                  "the stored ones of the same version; mover groups come from this pair's primary teams. The pairs are "
                  "not independent.", "", md_table(p["repeatability"]["table"]),
                  f"Mover groups among the pair's eligible catchers: {p['mover_counts_eligible']}.", "",
                  "By mover group (random fits; groups under 10 are listed, not correlated):", "",
                  md_table(p["repeatability"]["by_group"]), "Catchers in groups under 10:", "",
                  md_table(p["small_groups"]), "Disconnected catchers:", "",
                  md_table(p["repeatability"]["disconnected"])]
    L += ["## Left out (D2, approved)", "", "; ".join(LEFT_OUT) + ".", ""]
    return "\n".join(L)


def stage_report(year: int):
    season_gate(year, "report")

    def body():
        bad = verify_inputs(f"checks_{year}", f"run_{year}", f"validate_{year}")
        if bad:
            return {"passed": False, "stopped": f"inputs changed: {bad}"}
        checks = load_record(f"checks_{year}")["result"]
        run = load_record(f"run_{year}")["result"]["context"]
        with open(season_dir(year) / "checks_tables.pkl", "rb") as f:
            tables = pickle.load(f)
        with open(season_dir(year) / "validate" / "validate.pkl", "rb") as f:
            full = pickle.load(f)
        path = REPORTS_OUT / f"research_s9_{year}.md"
        if path.exists():
            raise FileExistsError(f"{path} exists")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report_text(year, checks, tables, run, full))
        return {"report": str(path), "outputs": file_digests([path]), "passed": True}
    run_season_stage(year, "report", body)


SEASON_STAGE_FUNCS = {"extract": stage_extract, "build": stage_build, "checks": stage_checks, "run": stage_run,
                      "halves": stage_halves, "validate": stage_validate, "report": stage_report,
                      "prefetch": stage_prefetch}
CONTEXTS.update({"season_run": ctx_season_run, "season_halves": ctx_season_halves,
                 "season_validate": ctx_season_validate})


# ===================================================================================== selftest (synthetic only)

def selftest_results() -> list:
    out = []

    def case(name, ok):
        out.append({"case": name, "ok": bool(ok)})
    # comparator
    a = pd.DataFrame({"k": [1, 2], "x": [0.0, 1.0]})
    b = pd.DataFrame({"k": [1, 2], "x": [-0.0, 1.0]})
    case("frames_compare: -0.0 vs 0.0 differs bitwise", not frames_compare(a, b, ["k"], bitwise=True)["identical"])
    case("frames_compare: -0.0 equals 0.0 by value", frames_compare(a, b, ["k"])["identical"])
    case("frames_compare: dtype change fails", not frames_compare(a, a.astype({"x": "float32"}), ["k"])["identical"])
    case("frames_compare: duplicate keys fail", not frames_compare(pd.concat([a, a]), pd.concat([a, a]), ["k"])["identical"])
    # amendment 1: the one permitted dtype difference (support.n, int64 stored vs float64 new), values exact
    sk = list(arsenal.SUPPORT_KEY)
    sref = pd.DataFrame({**{k: [0, 1] for k in sk}, "n": np.array([3, 5], "int64"), "share": [0.5, 0.5]})
    snew = sref.assign(n=sref["n"].astype("float64"))
    pa_ = lambda new, ref, t="support": frames_compare(permitted_dtype(t, new, ref)[0], ref, sk)["identical"]  # noqa: E731
    case("amendment 1: support.n float64 with equal integral values passes", pa_(snew, sref))
    case("amendment 1: a changed n value still fails", not pa_(snew.assign(n=[3.0, 6.0]), sref))
    case("amendment 1: a non-integral n fails (no cast)", not pa_(snew.assign(n=[3.0, 5.5]), sref))
    case("amendment 1: a NaN n fails (no cast)", not pa_(snew.assign(n=[3.0, np.nan]), sref))
    case("amendment 1: the reverse direction (int64 new, float64 stored) fails", not pa_(sref, snew))
    case("amendment 1: n in another table is not permitted", not pa_(snew, sref, t="pitches"))
    case("amendment 1: another column's dtype difference still fails",
         not pa_(snew.assign(share=snew["share"].astype("float32")), sref))
    # deep_compare
    s1 = {"t": a, "f": float("nan"), "l": [1, {"z": 2}], "s": {1, 2}}
    s2 = {"t": a.copy(), "f": float("nan"), "l": [1, {"z": 2}], "s": {2, 1}, "new": 1}
    case("deep_compare: equal with NaN, projection ignores extra keys", deep_compare(s2, s1) == [])
    case("deep_compare: extra keys reported without projection", deep_compare(s2, s1, projection=False) != [])
    case("deep_compare: a changed float is found", deep_compare({"x": 1.0}, {"x": 1.0 + 1e-15}) != [])
    case("deep_compare: a changed frame is found", deep_compare({"t": b}, {"t": a.assign(x=[0.0, 2.0])}) != [])
    # manifest contracts
    ref_p0 = {"id": "P0", "description": "d", "base": None, "params": {}, "seasons": [2024, 2025],
              "changes": {"evaluated_situations": False}, "alphas": [{"season": 2024, "fold": 0, "a": 1.0}],
              "policy_alphas": [{"season": 2024, "scoring_fold": 0, "role": "select", "alpha": 2.0}],
              "code": {"gamecalling_sha256": "06cd5ff9c5dd"}, "inputs": {}, "directory": "x", "created": "t",
              "seconds": {}}
    new_p0 = {**ref_p0, "seasons": [2024], "changes": {"evaluated_situations": False, "solve": False},
              "code": {"gamecalling_sha256": FULL_CODE_HASH}, "inputs": {"other": 1}, "run_label": "r", "timings": {},
              **P0_DEFAULTS}
    rows = manifest_contract(new_p0, ref_p0, "P0", season=2024)
    case("manifest P0: single season passes with defaults", all(r["identical"] for r in rows))
    rows = manifest_contract({**new_p0, "solve_log": [1]}, ref_p0, "P0", season=2024)
    case("manifest P0: non-empty default fails", not all(r["identical"] for r in rows))
    rows = manifest_contract({**new_p0, "code": {"gamecalling_sha256": "2ffaed8e90b1"}}, ref_p0, "P0", season=2024)
    case("manifest P0: other code hash fails", not all(r["identical"] for r in rows))
    rows = manifest_contract({**new_p0, "surprise": 1}, ref_p0, "P0", season=2024)
    case("manifest P0: unlisted key fails", not all(r["identical"] for r in rows))
    ref_c1 = {"id": "C1", "description": "d", "base": "P0", "params": {}, "seasons": [2024, 2025], "changes": {"a": True},
              "alphas": [{"season": 2024, "fold": 0}, {"season": 2025, "fold": 0}], "policy_alphas": [],
              "fit_ids": {"2024/0": "C1/full/2024/0/abc", "2025/0": "C1/full/2025/0/def"},
              "action_values": [{"path": "p", "model": "C1", "run_label": "full", "season": 2024, "scoring_fold": 0,
                                 "rows": 3, "sha256": "h"}], "solve_log": [{"season": 2024, "fit_id": "C1/full/2024/0/abc",
                                                                            "seconds_solve": 1.0}],
              "abnormal_log": [{"season": 2025, "lambda": 1.0}], "code": {}, "inputs": {}}
    new_c1 = {**ref_c1, "seasons": [2024], "alphas": [{"season": 2024, "fold": 0}],
              "fit_ids": {"2024/0": "C1/s9ctl/2024/0/abc"},
              "action_values": [{"path": "q", "model": "C1", "run_label": "s9ctl", "season": 2024, "scoring_fold": 0,
                                 "rows": 3, "sha256": "g"}], "solve_log": [{"season": 2024, "fit_id": "C1/s9ctl/2024/0/abc",
                                                                            "seconds_solve": 9.0}],
              "abnormal_log": [], "code": {"gamecalling_sha256": FULL_CODE_HASH}}
    rows = manifest_contract(new_c1, ref_c1, "C1", season=2024)
    case("manifest C1: season filtering, run label dropped, model id kept", all(r["identical"] for r in rows))
    bad = {**new_c1, "fit_ids": {"2024/0": "C1_2024/s9ctl/2024/0/abc"}}
    case("manifest C1: a renamed model id fails", not all(r["identical"] for r in manifest_contract(bad, ref_c1, "C1", 2024)))
    # mapping and keys
    play = lambda i, evs: {"about": {"atBatIndex": i}, "playEvents": evs}  # noqa: E731
    P = {"isPitch": True, "type": "pitch", "details": {"code": "B", "call": {"code": "B"}}}
    IBB = {"isPitch": False, "type": "no_pitch", "pitchNumber": 0,
           "details": {"code": "VB", "call": {"code": "VB"}, "description": "Automatic Ball - Intentional"}}
    PK = {"isPitch": False, "type": "pickoff", "details": {"code": "1"}}
    NP = {"isPitch": False, "type": "no_pitch", "details": {"code": "N"}}
    pbp = {"allPlays": [play(0, [P, P]), play(1, [PK]), play(2, [IBB, IBB, IBB, IBB]), play(3, [PK, NP, P])]}
    exp = expected_keys(pbp)
    case("mapping: pitches and automatic calls kept, pickoff and call-less no-pitch dropped",
         exp == {(1, 1), (1, 2), (3, 1), (3, 2), (3, 3), (3, 4), (4, 1)})
    case("mapping: empty play leaves a gap (no PA 2)", not any(k[0] == 2 for k in exp))
    case("keys: equal sets pass", key_check(exp, set(exp))["ok"])
    case("keys: a dropped pitch fails", not key_check(exp, exp - {(1, 2)})["ok"])
    case("keys: a shifted key with the same count fails", not key_check(exp, (exp - {(4, 1)}) | {(4, 2)})["ok"])
    # completed games
    sch = {"dates": [
        {"date": "2024-09-29", "games": [{"gamePk": 1, "gameType": "R", "status": {"abstractGameState": "Final",
                                          "detailedState": "Cancelled"}, "teams": {"home": {}, "away": {}}}]},
        {"date": "2024-06-01", "games": [{"gamePk": 2, "gameType": "R", "status": {"abstractGameState": "Live",
                                          "detailedState": "Suspended"}, "teams": {"home": {"score": 1}, "away": {"score": 0}}}]},
        {"date": "2024-06-02", "games": [{"gamePk": 2, "gameType": "R", "status": {"abstractGameState": "Final",
                                          "detailedState": "Final"}, "teams": {"home": {"score": 3}, "away": {"score": 2}}},
                                         {"gamePk": 3, "gameType": "S", "status": {"abstractGameState": "Final",
                                          "detailedState": "Final"}, "teams": {"home": {"score": 3}, "away": {"score": 2}}}]}]}
    cg = completed_games(sch)
    case("completed games: cancelled-without-scores excluded, suspended matched once, spring excluded",
         list(cg.index) == [2] and cg.loc[2, "home_score"] == 3)
    # reconstruction: Astra's two-pitch false pass must fail here
    rows_ok = pd.DataFrame({"season": [2024] * 4, "pa_id": [1, 1, 2, 2], "pitch_number": [1, 1, 1, 1],
                            "action": ["FF", "SL", "FF", "SL"], "Q_select": [0.1, 0.2, 0.1, 0.3],
                            "Q_value": [0.5, 0.7, 0.4, 0.6]})
    sc = pd.DataFrame({"season": [2024, 2024], "pa_id": [1, 2], "pitch_number": [1, 1], "action": ["SL", "FF"],
                       "rec_action": ["FF", "FF"], "Q_actual": [0.7, 0.4], "D": [0.7 - 0.5, 0.0]})
    case("reconstruction: complete rows pass", reconstruction_check(rows_ok, sc)["ok"])
    rows_bad = rows_ok.drop(index=1).reset_index(drop=True)          # the first pitch's actual slider row omitted
    case("reconstruction: a missing actual-action row fails", not reconstruction_check(rows_bad, sc)["ok"])
    old = sf.reconstruct_check(rows_bad.assign(scoring_fold=0), sc)
    case("reconstruction: the frozen check alone misses it (documents the false pass)",
         old["unmatched"] == 0 and old["rec_differs"] == 0 and old["max_D_diff"] == 0)
    sc2 = sc.assign(D=[0.2000001, 0.0])
    case("reconstruction: a D mismatch fails", not reconstruction_check(rows_ok, sc2)["ok"])
    # flags
    flags = {"version": 1, "entries": [
        {"season": 2025, "team": "MIA", "level": "documented", "from": "2025-09-19", "source": "s"},
        {"season": 2026, "team": "MIA", "level": "documented", "source": "s"},
        {"season": 2026, "team": "COL", "level": "possible_exposure", "source": "s"}]}
    p = pd.DataFrame({"season": [2025, 2025, 2026, 2026, 2026, 2024], "fielding_team": ["MIA", "MIA", "MIA", "COL", "NYM", "MIA"],
                      "game_date": ["2025-09-18", "2025-09-19", "2026-04-01", "2026-04-01", "2026-04-01", "2024-09-20"]})
    called, level = dugout_columns(p, flags)
    case("flags: dugout_called is documented practice only", called.tolist() == [False, True, True, False, False, False])
    case("flags: dugout_level marks possible exposure separately",
         level.fillna("-").tolist() == ["-", "documented", "documented", "possible_exposure", "-", "-"])
    # spread
    case("spread: empirical SD uses ddof 1", abs(empirical_sd(np.array([1.0, 2.0, 3.0])) - 1.0) < 1e-15)
    case("spread: one value gives NaN", math.isnan(empirical_sd(np.array([1.0]))))
    # predecessor binding
    ident = {"script_sha256": "s", "protocol_sha256": "p", "flags_sha256": "f", "gamecalling_sha256": "g",
             "mode": "production", "smoke_dir": None}
    gguard = {"stage": "guard", "status": "finished", "passed": True, "identity": dict(ident), "predecessors": {},
              "inventory": {}}
    case("record binding: a matching guard record passes", record_problems("guard", gguard, ident) == [])
    good = {"stage": "audit", "status": "finished", "passed": True, "identity": dict(ident), "predecessors": {}}
    case("record binding: an audit record without its guard predecessor fails",
         any("expected" in x for x in record_problems("audit", good, ident)))
    case("record binding: a control record naming only guard fails",
         any("expected" in x for x in record_problems("control", {**good, "stage": "control",
                                                                  "predecessors": {"guard": "h"}}, ident)))
    case("record binding: a guard record without inventory fails",
         record_problems("guard", {**gguard, "inventory": None}, ident) != [])
    case("record binding: another stage's record fails", record_problems("audit", {**good, "stage": "other"}, ident) != [])
    case("record binding: another script revision fails",
         record_problems("audit", {**good, "identity": {**ident, "script_sha256": "wrong"}}, ident) != [])
    case("record binding: a failed record fails", record_problems("audit", {**good, "passed": False}, ident) != [])
    case("record binding: another flag table fails",
         record_problems("audit", {**good, "identity": {**ident, "flags_sha256": "x"}}, ident) != [])
    case("record binding: identity null is a problem, not an exception",
         record_problems("audit", {**good, "identity": None}, ident) != [])
    # smoke excusal: by name and by kind
    base_ctx = {"returncode": 0, "probe": {"ok": True}}
    val = {"check": "P0 scores (keyed, exact)", "identical": False, "detail": "differ: D, A"}
    case("smoke excusal: a value difference in a waivable comparison is excused",
         smoke_excused({**base_ctx, "checks": [val]})[0])
    for det in ("1 duplicate keys in the new table", "columns differ: [a] vs. [b]", "row counts differ: 1 vs. 2"):
        case(f"smoke excusal: '{det.split(':')[0]}' is never excused",
             not smoke_excused({**base_ctx, "checks": [{**val, "detail": det}]})[0])
    mem = {"check": "manifest C1: fit-id membership (model, season, fold)", "identical": False, "detail": "x"}
    dig = {"check": "manifest C1: fit-id hashes (run label dropped, model id kept)", "identical": False, "detail": "x"}
    case("smoke excusal: changed fit-id digests are excused, changed membership is not",
         smoke_excused({**base_ctx, "checks": [dig]})[0] and not smoke_excused({**base_ctx, "checks": [dig, mem]})[0])
    k1 = pd.DataFrame({"season": [2025, 2025], "pa_id": [1, 2], "pitch_number": [1, 1], "D": [0.1, 0.2]})
    case("structure: equal-length substituted keys fail", not structure_compare(k1.assign(pa_id=[1, 3]), k1, KEYS)["identical"])
    case("structure: dtype-only drift fails", not structure_compare(k1.astype({"D": "float32"}), k1, KEYS)["identical"])
    case("structure: identical structure passes", structure_compare(k1.assign(D=[9.0, 9.0]), k1, KEYS)["identical"])
    recs = [{"season": 2025, "fold": 0, "a": 1.0}, {"season": 2025, "fold": 1, "a": 2.0}]
    case("structure: a missing manifest record fails", not records_structure(recs[:1], recs, RECORD_IDS["alphas"])["identical"])
    case("structure: an extra abnormal_log record fails",
         not records_structure([{"season": 2025, "component": "outcome", "code": 1, "inner": 0, "lambda": 1.0}], [],
                               RECORD_IDS["abnormal_log"])["identical"])
    case("structure: changed values with the same identities pass",
         records_structure([{**recs[0], "a": 5.0}, recs[1]], recs, RECORD_IDS["alphas"])["identical"])
    ab_row = {"check": "manifest C1: abnormal_log (records with multiplicity)", "identical": False,
              "detail": "1 records only in the new run, 0 only in the stored run"}
    case("smoke excusal: an abnormal_log difference is never excused", not smoke_excused({**base_ctx, "checks": [ab_row]})[0])
    st_row = {"check": "P0 scores: key set and dtypes (structural)", "identical": False, "detail": "key sets differ"}
    case("smoke excusal: a structural row is never excused, even beside a waivable value row",
         not smoke_excused({**base_ctx, "checks": [val, st_row]})[0])
    case("fetch validation: a boolean or negative atBatIndex is rejected",
         not valid_pbp({"allPlays": [{"about": {"atBatIndex": False}, "playEvents": [P]}]})
         and not valid_pbp({"allPlays": [{"about": {"atBatIndex": -1}, "playEvents": [P]}]}))
    case("smoke excusal: a failed probe or nonzero exit is never excused",
         not smoke_excused({**base_ctx, "probe": {"ok": False}, "checks": [val]})[0]
         and not smoke_excused({**base_ctx, "returncode": 1, "checks": [val]})[0])
    # vocabulary: MLB's subtype is kept; only listed automatic calls are kept as rows
    sub = {"type": "action", "isPitch": False, "details": {"eventType": "pitching_substitution"}}
    chal = {"type": "action", "isPitch": False, "details": {"eventType": "abs_challenge"}}
    case("vocabulary: an unseen action subtype is distinguishable", event_signature(sub) != event_signature(chal))
    odd = {"type": "no_pitch", "isPitch": False, "details": {"code": "N", "call": {"code": "N"}}}
    case("mapping v2: a no-pitch call without an automatic description is kept but unrecognized (gate fails)",
         kept(odd) and unrecognized(odd) and kept(IBB) and not unrecognized(IBB))
    tmo = {"type": "no_pitch", "isPitch": False,
           "details": {"code": "AB", "call": {"code": "AB"}, "description": "Automatic Strike - Batter Timeout Violation"}}
    case("mapping v2: an automatic call outside v1's list (AB) is kept and recognized", kept(tmo) and not unrecognized(tmo))
    case("mapping v2: a no-pitch event with no call is not kept", not kept(NP) and not unrecognized(NP))
    kc0 = {"n_missing": 1, "n_extra": 0, "missing": [(68, 5)], "extra": []}
    pit = lambda c: {"isPitch": True, "type": "pitch", "details": {"code": c, "call": {"code": c}}}  # noqa: E731
    pbp68 = {"allPlays": [{"about": {"atBatIndex": 67}, "playEvents": [pit(c) for c in ("F", "B", "F", "*B", "S")]}]}
    rows68 = pd.DataFrame({"at_bat_number": [68] * 4, "pitch_number": [1, 2, 3, 4],
                           "description": ["foul", "ball", "foul", "swinging_strike"]})
    case("expected shortfall: only the listed game, exact key difference and both event sequences qualify",
         expected_shortfall(2024, 746731, kc0, pbp68, rows68) and not expected_shortfall(2025, 746731, kc0, pbp68, rows68)
         and not expected_shortfall(2024, 746731, {**kc0, "missing": [(68, 4)]}, pbp68, rows68)
         and not expected_shortfall(2024, 746731, {**kc0, "n_extra": 1, "extra": [(1, 1)]}, pbp68, rows68)
         and not expected_shortfall(2024, 745093, kc0, pbp68, rows68) and not expected_shortfall(2024, 746731, kc0))
    case("expected shortfall: another omitted pitch with the same missing key fails (MLB sequence differs)",
         not expected_shortfall(2024, 746731, kc0, {"allPlays": [{"about": {"atBatIndex": 67}, "playEvents":
                                                                  [pit(c) for c in ("F", "B", "C", "*B", "S")]}]}, rows68))
    case("expected shortfall: a different Statcast sequence fails",
         not expected_shortfall(2024, 746731, kc0, pbp68, rows68.assign(description=["foul", "ball", "ball", "swinging_strike"])))
    case("fetch validation: a repeated atBatIndex is rejected (Astra)",
         not valid_pbp({"allPlays": [play(0, [P]), play(0, [P])]}) and valid_pbp({"allPlays": [play(0, [P]), play(1, [P])]}))
    big = pd.DataFrame({**{k: [0] for k in sk}, "n": np.array([2.0 ** 70]), "share": [1.0]})
    case("amendment 1: a value too large for int64 is not cast (Neon)",
         permitted_dtype("support", big, sref.iloc[:1].reset_index(drop=True))[1].endswith("not applied"))
    case("fetch validation: malformed play-by-play is rejected",
         not valid_pbp({"allPlays": [{"about": {}}]}) and not valid_pbp([]) and valid_pbp(pbp))
    case("fetch validation: total on empty plays, null events and null about",
         not valid_pbp({"allPlays": []}) and not valid_pbp({"allPlays": [play(0, [None])]})
         and not valid_pbp({"allPlays": [{"about": None, "playEvents": []}]}))
    case("fetch validation: a string isPitch is rejected (no false pitch key)",
         not valid_pbp({"allPlays": [play(0, [{"type": "pitch", "isPitch": "false"}])]}))
    case("fetch validation: an unhashable leaf is rejected (no crash in vocabulary)",
         not valid_pbp({"allPlays": [play(0, [{"type": "action", "details": {"eventType": []}}])]}))
    case("fetch validation: a string score is rejected; a missing score is allowed",
         not valid_schedule({"dates": [{"games": [{"gamePk": 1, "gameType": "R", "status": {}, "teams": {"home": {"score": "3"}}}]}]})
         and valid_schedule({"dates": [{"games": [{"gamePk": 1, "gameType": "R", "status": {}, "teams": {"home": {}}}]}]}))
    case("fetch validation: schedule validator is total",
         not valid_schedule({"dates": [None]}) and not valid_schedule({"dates": []}) and valid_schedule(sch))
    # baseline and calibration fixture (independent hand values)
    pred = np.array([0.1, 0.4, 0.35, 0.8])
    hit = np.array([False, True, False, True])
    exp_cal = (2 * abs((0.1 + 0.35) / 2 - 0.0) + 2 * abs((0.4 + 0.8) / 2 - 1.0)) / 4    # bins by rank: {0.1, 0.35}, {0.4, 0.8}
    got = validation.calibration_error(pred, hit, n_bins=2)
    case("calibration error equals the hand value (2 equal-count bins)", abs(got - exp_cal) < 1e-15)
    tr = pd.DataFrame({"cidx": [0, 0, 0, 1], "action": ["FF", "FF", "SL", "FF"], "y": [0, 1, 0, 1]})
    base = validation._league_baseline(tr, ["cidx", "action"], ["cidx"], 2)
    # cell (0, FF): counts (1, 1), marginal at cidx 0 = (2/3, 1/3) -> ((1 + 2/3) / 3, (1 + 1/3) / 3)
    case("criterion baseline equals the hand value", np.allclose(base.loc[(0, "FF")].to_numpy(), [5 / 9, 4 / 9], atol=0, rtol=1e-15))
    # R3a / R3b ranking-sample fixture: vp.sensitivity with season_fit stubbed (mean score per catcher), so the sample
    # selection, the 500-pitch and connectivity rules and the relabeling are exercised without a REML fit
    rng = np.random.default_rng(1)
    n = 14 * 800
    syn = pd.DataFrame({"season": 2026, "catcher": np.repeat(np.arange(14), 800), "pitcher": rng.integers(0, 6, n),
                        "game_pk": rng.integers(0, 200, n), "base_out": rng.integers(0, 24, n),
                        "D": rng.normal(size=n), "A": rng.normal(size=n), "D_opt": rng.normal(size=n),
                        "D_1p": rng.normal(size=n), "fielding_team": "AAA", "pa_id": np.arange(n), "pitch_number": 1})
    syn["dugout_called"] = syn["catcher"].isin([0, 1]) & (syn["game_pk"] < 100)          # documented practice
    lvl = np.where(syn["dugout_called"], "documented", np.where(syn["catcher"].isin([2]), "possible_exposure", None))
    old = (vp.season_fit, vp.SEASONS, vp._DATA.get("scores"))
    try:
        def stub_fit(df, score, fixed=False):
            m = df.groupby("catcher")[score].mean()
            return {"saved": -m, "n": df.groupby("catcher").size(), "sd": float(m.std()), "zero": False,
                    "converged": True, "pitchers": None}
        vp.season_fit = stub_fit
        vp.SEASONS = (2026,)
        vp._DATA["scores"] = syn
        qual = set(range(14))
        sets = {2026: {"qual": qual, "main": vp.main_group(syn, qual), "n": syn.groupby("catcher").size()}}
        fits = {2026: {(sc, fx): stub_fit(syn, sc) for sc in ("D", "A", "B") if sc in syn for fx in (False, True)}}
        fits[2026].update({("B", False): fits[2026][("D", False)], ("B", True): fits[2026][("D", False)]})
        r3a = vp.sensitivity(fits, sets)["table"]
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "levels.parquet"
            syn.assign(dugout_level=lvl)[["season", "pa_id", "pitch_number", "dugout_level"]].to_parquet(tmp, index=False)
            r3b = r3b_sensitivity(syn, tmp, fits, sets, (2026,))
            r3b_ex = r3b_full(syn, tmp, fits, sets, (2026,))["excluded"]
    finally:
        vp.season_fit, vp.SEASONS = old[0], old[1]
        if old[2] is None:
            vp._DATA.pop("scores", None)
        else:
            vp._DATA["scores"] = old[2]
    a_rm = int(r3a.loc[r3a["check"] == "R3_D", "removed_pitches"].iloc[0])
    b_rm = int(r3b.loc[r3b["check"] == "R3b_D", "removed_pitches"].iloc[0])
    case("R3a removes exactly the documented pitches", a_rm == int(syn["dugout_called"].sum()))
    case("R3b removes documented and possible-exposure pitches", b_rm == int(pd.notna(lvl).sum()) and b_rm > a_rm)
    case("R3b's excluded list is kept and relabeled (Neon)", isinstance(r3b_ex, pd.DataFrame) and
         (len(r3b_ex) == 0 or set(r3b_ex["check"]) <= {"R3b_D", "R3b_A"}))
    case("R3b rows are relabeled and keep the common-set rules",
         set(r3b["check"]) == {"R3b_D", "R3b_A"} and int(r3b.loc[r3b["check"] == "R3b_D", "n"].iloc[0]) <= 14)
    out.extend(selftest_phase2())
    return out


def selftest_phase2() -> list:
    """Phase 2 (synthetic only): stage chaining, bindings, season gating, B0, completeness, calibration guard."""
    out = []

    def case(name, ok):
        out.append({"case": name, "ok": bool(ok)})

    def refuses(fn, *a):
        try:
            fn(*a)
        except SystemExit:
            return True
        return False
    # chaining
    case("predecessors: extract names the whole development chain and the authorization",
         expected_predecessors("extract_2023") == {"guard", "audit", "control", f"gate_v{MAPPING_VERSION}", "domains",
                                                    "authorize_2023"})
    case("predecessors: each later season stage names its authorization and its predecessor",
         expected_predecessors("run_2023") == {"guard", "authorize_2023", "checks_2023"}
         and expected_predecessors("validate_2023") == {"guard", "authorize_2023", "run_2023", "halves_2023"}
         and expected_predecessors("report_2023") == {"guard", "authorize_2023", "checks_2023", "run_2023", "validate_2023"}
         and expected_predecessors("checks_2023") == {"guard", "authorize_2023", "build_2023", "domains",
                                                      f"gate_v{MAPPING_VERSION}"})
    case("predecessors: development stages unchanged; unknown stage gives None",
         expected_predecessors("control") == {"guard", "audit"} and expected_predecessors("gate_v1") == {"guard"}
         and expected_predecessors("domains") == {"guard"} and expected_predecessors("nonsense_x") is None)
    ident = {"script_sha256": "s", "protocol_sha256": "p", "flags_sha256": "f", "gamecalling_sha256": "g",
             "mode": "production", "smoke_dir": None}
    rec = {"stage": "build_2023", "status": "finished", "passed": True, "identity": dict(ident),
           "predecessors": {"guard": "h", "extract_2023": "h"}}
    case("record binding: a season stage missing its authorization predecessor fails",
         any("expected" in x for x in record_problems("build_2023", rec, ident)))
    # bindings
    b = {"stage": "run_2023", "season": 2023, "out": str(SEASONS_ROOT / "2024" / "run"),
         "models_dir": str(SEASONS_ROOT / "2024" / "run" / "models")}
    case("binding: a season context writing outside its own season directory is refused",
         any("is not inside" in x for x in validate_binding(b)))
    case("binding: a season context with another stage name or season is refused",
         validate_binding({**b, "stage": "build_2023"})[0].startswith("binding stage")
         and validate_binding({**b, "season": 2024})[0].startswith("binding stage"))
    canon = {k: str(SEASONS_ROOT / "2023" / f"{k}.parquet") for k in ("raw", "pitches", "pas", "support")}
    good = {**b, **canon, "out": str(SEASONS_ROOT / "2023" / "run"), "models_dir": str(SEASONS_ROOT / "2023" / "run" / "models")}
    case("binding: a context dispatched under another stage is refused (Astra)",
         validate_binding(good, "season_halves")[0].startswith("context") and
         validate_binding({**good, "stage": "control"}, "season_run")[0].startswith("context"))
    case("binding: a season context must use the season's canonical inputs and its exact stage directory",
         any("is not the season's pitches.parquet" in x for x in validate_binding({**good, "pitches": "/tmp/x.parquet"},
                                                                                     "season_run"))
         and any("is not the stage directory" in x for x in validate_binding(
             {**good, "out": str(SEASONS_ROOT / "2023" / "run" / "x"),
              "models_dir": str(SEASONS_ROOT / "2023" / "run" / "x" / "models")}, "season_run")))
    case("binding: a season context needs its own reserved stage record",
         any("no reserved run_2023 stage" in x for x in validate_binding(
             {**b, "out": str(SEASONS_ROOT / "2023" / "run"), "models_dir": str(SEASONS_ROOT / "2023" / "run" / "models")})))
    # season gating
    if SMOKE:
        case("season gating (smoke): only the development stand-in is allowed",
             refuses(season_info, 2023) and refuses(season_info, 2026) and season_info(SMOKE_STANDIN)["source"] == ORIG["raw"])
    else:
        case("season gating: 2026 and development seasons are refused in phase 2",
             refuses(season_info, 2026) and refuses(season_info, 2024) and season_info(2023)["source"] == RAW_2015_2023)
    case("authorization date: strict YYYY-MM-DD, not before the lock, not in the future",
         valid_authorization_date(PROTOCOL_LOCKED_ON) and not valid_authorization_date("2026-11-20")
         and not valid_authorization_date("2099-01-01") and not valid_authorization_date("2026-11-2x")
         and not valid_authorization_date("../2026-11-22") and not valid_authorization_date(None))
    # B0
    dom = {"description": ["ball", "called_strike", "hit_into_play"], "events_dev": ["single", "walk"],
           "events_code": ["sac_bunt_double_play", "walk"], "bb_type": list(batter.BB_TYPES), "stand": ["L", "R"],
           "p_throws": ["L", "R"], "inning_topbot": ["Bot", "Top"]}
    raw = pd.DataFrame({"game_type": ["R", "R", "R", "S"], "game_pk": [1, 1, 1, 2], "at_bat_number": [1, 1, 2, 1],
                        "pitch_number": [1, 2, 1, 1], "description": ["ball", "hit_into_play", "hit_into_play", "ball"],
                        "events": [None, "single", "sac_bunt_double_play", None],
                        "bb_type": [None, "line_drive", "ground_ball", None], "stand": ["L", "L", "R", "R"],
                        "p_throws": ["R"] * 4, "inning_topbot": ["Top"] * 4, "pitch_type": ["FF"] * 4})
    r0 = b0_inventory(raw, dom)
    case("B0: a clean file passes; a code-only event is counted separately",
         r0["ok"] and r0["events_accepted_through_code_only"] == {"sac_bunt_double_play": 1})
    case("B0: an unknown description stops", not b0_inventory(raw.assign(description=["ball", "hit_into_play",
                                                                                      "new_call", "ball"]), dom)["ok"])
    case("B0: an event in neither list stops", not b0_inventory(raw.assign(events=[None, "single", "abs_out", None]), dom)["ok"])
    case("B0: a duplicate key anywhere in the raw file stops (spring rows included)",
         not b0_inventory(raw.assign(game_pk=[1, 1, 1, 1], at_bat_number=[1, 1, 2, 2]).assign(
             pitch_number=[1, 2, 1, 1]), dom)["ok"])
    case("B0: a bb_type outside the four labels anywhere in the file stops",
         not b0_inventory(raw.assign(bb_type=[None, "line_drive", "ground_ball", "bunt_pop"]), dom)["ok"])
    case("B0: terminal rows are the last pitch of each plate appearance",
         terminal_mask(raw[raw["game_type"] == "R"]).tolist() == [False, True, True])
    # completeness: non-pitch runs after the last pitch, and the precedence
    P_ = {"isPitch": True, "type": "pitch", "details": {"code": "B", "call": {"code": "B"}}}
    BALK = {"isPitch": False, "type": "action", "details": {"eventType": "balk"}}
    sc_ = lambda j, **mv: {"movement": {"end": "score", "isOut": False, **mv},  # noqa: E731
                           "details": {"playIndex": j, "isScoringEvent": True}}
    pbp_w = {"allPlays": [{"about": {"atBatIndex": 0, "halfInning": "top"}, "playEvents": [P_],
                           "runners": [sc_(0)]},
                          {"about": {"atBatIndex": 1, "halfInning": "bottom"}, "playEvents": [P_, BALK],
                           "runners": [sc_(1), {"movement": {"end": "2B"}, "details": {"playIndex": 1}}]}]}
    case("completeness: a walk-off balk after the last pitch is one home run on a non-pitch event",
         nonpitch_runs_after_last_pitch(pbp_w) == (1, 0))
    pbp_p = {"allPlays": [{**pbp_w["allPlays"][1], "runners": [sc_(0)]}]}
    case("completeness: a run scored on the last pitch itself is not a non-pitch run",
         nonpitch_runs_after_last_pitch(pbp_p) == (0, 0))
    pbp_bad = {"allPlays": [{**pbp_w["allPlays"][1], "runners": [sc_(9)]}]}
    case("completeness: an unreadable runner index gives None (unexplained)", nonpitch_runs_after_last_pitch(pbp_bad) is None)
    out_ = {"allPlays": [{**pbp_w["allPlays"][1], "runners": [sc_(1, isOut=True)]}]}
    noflag = {"allPlays": [{**pbp_w["allPlays"][1], "runners": [{"movement": {"end": "score"}, "details": {"playIndex": 1}}]}]}
    case("completeness: a scoring record marked out, or without isScoringEvent, is not credited (Neon)",
         nonpitch_runs_after_last_pitch(out_) == (0, 0) and nonpitch_runs_after_last_pitch(noflag) == (0, 0))
    # small mover groups are listed with both estimates and the change
    eff = pd.DataFrame({"y0": [1.0, 2.0, 3.0], "y1": [1.5, 2.0, 2.0], "w": 1.0}, index=[10, 11, 12])
    rep_ = {"eligible": [10, 11, 12], "effects": {("D", "random"): eff, ("A", "random"): eff}}
    mv_ = pd.DataFrame({"group": ["mover", "non_mover", "non_mover"]}, index=[10, 11, 12])
    sg = small_group_listing(rep_, mv_)
    case("pairs: groups under 10 list every catcher with both estimates and the change",
         len(sg) == 6 and set(sg["group"]) == {"mover", "non_mover"}
         and float(sg[(sg["catcher"] == 10) & (sg["score"] == "D")]["change"].iloc[0]) == 0.5)
    base = {"duplicate_statcast_keys": 0, "unexpected_statcast_games": [], "feed_stop": [], "score_unexplained": [],
            "unavailable": [], "missing_games": [], "key_mismatch": []}
    miss = {"n_missing": 1, "n_extra": 0}
    extra = {"n_missing": 0, "n_extra": 1}
    case("completeness precedence: integrity > review > confirmed missing > pass",
         classify_completeness(base) == "pass"
         and classify_completeness({**base, "missing_games": [1]}).startswith("incomplete")
         and classify_completeness({**base, "key_mismatch": [miss]}).startswith("incomplete")
         and classify_completeness({**base, "key_mismatch": [extra]}) == "blocked for review"
         and classify_completeness({**base, "key_mismatch": [miss], "unavailable": [1]}) == "blocked for review"
         and classify_completeness({**base, "feed_stop": [1], "unexpected_statcast_games": [9]}) == "stop: data integrity")
    # held-out: a detected mismatch yields no metric (Astra)
    ho = pd.DataFrame({"season": 2023, "pa_id": np.arange(12), "pitch_number": 1, "fold": np.arange(12) % 3,
                       "y": np.arange(12) % 2, "p0": 0.4, "p1": 0.6})
    row_ok, ch_ok = heldout_metrics("contact", ho, ho.copy(), ho.copy(), ho.copy(), ho.copy())
    row_bad, ch_bad = heldout_metrics("contact", ho, ho.iloc[1:].reset_index(drop=True), ho.copy(), ho.copy(), ho.copy())
    with np.errstate(divide="ignore"):          # the zero probability is the point of this case
        row_inf, _ = heldout_metrics("contact", ho, ho.assign(p0=np.where(ho["y"] == 0, 0.0, 0.4)), ho.copy(), ho.copy(),
                                     ho.copy())
    case("held-out: matched frames give one row; a missing C1 row or a zero probability gives no metric",
         row_ok is not None and row_ok["n"] == 12 and all(x["identical"] for x in ch_ok)
         and row_bad is None and not all(x["identical"] for x in ch_bad) and row_inf is None)
    # revision 9: an exact zero in the criterion baseline is reported as +inf and disclosed; nothing else is relaxed
    cnt = ho[KEYS].assign(cidx=10)
    cz = ho.copy()
    cz.loc[0, ["p0", "p1"]] = [0.0, 1.0]                  # row 0 has y == 0: an observed-class zero
    row_cz, ch_cz = heldout_metrics("contact", ho, ho.copy(), cz, ho.copy(), ho.copy(), cnt)
    zd = [x["criterion_zero"] for x in ch_cz if "criterion_zero" in x]
    case("held-out (rev 9): a criterion-baseline exact zero gives +inf, a row, and a disclosure by fold, count, class",
         row_cz is not None and math.isinf(row_cz["criterion_baseline"]) and row_cz["criterion_baseline"] > 0
         and row_cz["criterion_zero_probability_rows"] == 1 and all(x["identical"] for x in ch_cz)
         and math.isfinite(row_cz["P0"]) and not isinstance(row_cz["max_cal_criterion_baseline"], str)
         and zd and zd[0]["breakdown"] == [{"fold": 0, "cidx": 10, "y": 0, "class": f"bin_{payoff.CONTACT_BINS[0]}",
                                            "rows": 1}]
         and zd[0]["rows"][0]["pa_id"] == 0)
    cu = ho.copy()
    cu.loc[1, ["p0", "p1"]] = [0.0, 1.0]                  # row 1 has y == 1: a zero on an unobserved class only
    row_cu, ch_cu = heldout_metrics("contact", ho, ho.copy(), cu, ho.copy(), ho.copy(), cnt)
    case("held-out (rev 9): a zero on an unobserved class is finite and not disclosed",
         row_cu is not None and math.isfinite(row_cu["criterion_baseline"]) and row_cu["criterion_zero_probability_rows"] == 0)
    bad_crit = {"NaN": cz.assign(p0=np.where(cz.index == 2, np.nan, cz["p0"])),
                "negative": cz.assign(p0=np.where(cz.index == 2, -0.1, cz["p0"]), p1=np.where(cz.index == 2, 1.1, cz["p1"])),
                "row sum": cz.assign(p1=np.where(cz.index == 2, 0.6 + 1e-9, cz["p1"]))}
    fails = {k: heldout_metrics("contact", ho, ho.copy(), v, ho.copy(), ho.copy(), cnt)[0] is None for k, v in bad_crit.items()}
    case("held-out (rev 9): a criterion baseline with NaN, out-of-range or unnormalized rows still fails closed",
         all(fails.values()))
    hz = ho.copy()
    hz.loc[0, ["p0", "p1"]] = [0.0, 1.0]
    case("held-out (rev 9): an exact zero in P0, C1 or the league-pooled level still fails closed",
         heldout_metrics("contact", hz, ho.copy(), ho.copy(), ho.copy(), hz.copy(), cnt)[0] is None
         and heldout_metrics("contact", ho, hz.copy(), ho.copy(), ho.copy(), ho.copy(), cnt)[0] is None
         and heldout_metrics("contact", ho, ho.copy(), ho.copy(), hz.copy(), ho.copy(), cnt)[0] is None)
    jr = json_records(pd.DataFrame({"a": [float("inf"), float("-inf"), float("nan"), 1.5], "b": ["x", "y", "z", "w"]}))
    case("held-out (rev 9): JSON labels +inf/-inf, NaN as null, and parses with a strict reader",
         [r["a"] for r in jr] == ["+inf", "-inf", None, 1.5]
         and json.loads(json.dumps(jr, allow_nan=False)) == jr)
    full_res = {"probe": {"ok": True}, "checks": [{"check": "x", "identical": True, "detail": float("nan")}],
                "heldout": json_records(pd.DataFrame({"criterion_baseline": [float("inf")]})),
                "summary": {"P0": {"spread": json_records(pd.DataFrame({"sd": [np.float64("nan")]})),
                                   "pairs": {"(2023, 2024)": json_records(pd.DataFrame({"group": ["movers"], "n": [4],
                                                                                         "median_abs_diff": [np.nan]}))}}},
                "nested": [np.float64("inf"), {"deep": float("-inf")}]}
    strict = json.loads(json.dumps(json_safe(full_res), allow_nan=False, default=str),
                        parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    case("validate (rev 9): a complete result with an undefined pair statistic serializes as strict JSON",
         strict["summary"]["P0"]["pairs"]["(2023, 2024)"][0]["median_abs_diff"] is None
         and strict["heldout"][0]["criterion_baseline"] == "+inf" and strict["nested"] == ["+inf", {"deep": "-inf"}]
         and strict["checks"][0]["detail"] is None)
    case("report (rev 9): Markdown renders +inf and -inf",
         "| +inf | -inf |" in md_table(pd.DataFrame({"a": [float("inf")], "b": [float("-inf")]})))
    zn = zero_probability_note([{"component": "outcome", "zero_probability_rows": 1, "n": 9, "breakdown": [{"rows": 1}],
                                 "rows": [{"pa_id": 7}]}, {"component": "contact", "zero_probability_rows": 0, "n": 9,
                                                           "breakdown": [], "rows": []}])
    case("report (rev 9): the zero-probability disclosure lists only components with zeros",
         any("outcome: 1 of 9 rows; the outcome baseline" in x for x in zn) and not any("- contact:" in x for x in zn)
         and zero_probability_note([]) == [])
    # revision 9: comparison samples are labeled; the smoke stand-in never shares development 2024's label (Astra)
    def _env_frames(season, n, pa0):
        pp = pd.DataFrame({"season": season, "pa_id": pa0 + np.arange(n), "pitch_number": 1, "thrown": True,
                           "exclusion_reason": [None] * (n - 1) + ["x"], "arsenal_reason": None,
                           "description": "ball", "zone": 11.0, "events": "walk", "balls": 3, "strikes": 0,
                           "pa_sequence_error": False, "pitch_type": "FF"})
        return pp, pp[["season", "pa_id", "events", "pa_sequence_error"]].copy()
    tp, ta = _env_frames(2024, 4, 0)
    dp = pd.concat([_env_frames(2024, 10, 100)[0], _env_frames(2025, 6, 200)[0]], ignore_index=True)
    da = pd.concat([_env_frames(2024, 10, 100)[1], _env_frames(2025, 6, 200)[1]], ignore_index=True)
    ct = comparison_tables(2024, "2024 (smoke stand-in)", tp, ta, dp, da)
    wk = ct["walks"].set_index("sample")["pas"].to_dict()
    sh = ct["coverage"].groupby("sample")["share"].sum()
    try:
        comparison_tables(2024, "2024", tp, ta, dp, da)
        dup_refused = False
    except ValueError:
        dup_refused = True
    b2 = b2_table(tp, 2024, dp, "2024 (smoke stand-in)")
    try:
        b2_table(tp, 2024, dp, "2024")
        b2_dup = False
    except ValueError:
        b2_dup = True
    case("checks (rev 9): B2 rows carry distinct sample labels; a development label for the target is refused",
         list(b2["table"]["sample"]) == ["2024 (smoke stand-in)", "2024", "2025"] and b2_dup
         and b2["table"].groupby("sample")["pitches"].sum().to_dict() == {"2024 (smoke stand-in)": 4, "2024": 10, "2025": 6})
    case("checks (rev 9): the stand-in keeps its own label and counts; shares sum to 1 within each sample",
         wk == {"2024 (smoke stand-in)": 4, "2024": 10, "2025": 6} and bool(np.allclose(sh.to_numpy(), 1.0))
         and set(ct["called_strike"]["sample"]) == set(wk) and dup_refused
         and target_sample_label(2023) == ("2023 (smoke stand-in)" if SMOKE else "2023"))
    # recursive predecessor verification (Astra): a record replaced after a later stage pinned it is caught
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        pth = lambda n: Path(td) / f"s9_{n}.json"  # noqa: E731
        pth("domains").write_text(jdump({"stage": "domains", "predecessors": {}}))
        pth("extract_2023").write_text(jdump({"stage": "extract_2023", "predecessors": {"domains": sha(pth("domains"))}}))
        pth("build_2023").write_text(jdump({"stage": "build_2023", "predecessors": {"extract_2023": sha(pth("extract_2023"))}}))
        clean = chain_problems("build_2023", pth) == []
        pth("domains").write_text(jdump({"stage": "domains", "predecessors": {}, "replaced": True}))
        caught = any("pinned domains" in x for x in chain_problems("build_2023", pth))
    case("chain: a predecessor replaced after a later stage pinned it is caught two levels down", clean and caught)
    # calibration guard and the report table
    case("calibration: fewer than 10 rows is not computed (n < 10)",
         calibration_or_note(np.full(9, 0.5), np.zeros(9, bool)) == f"not computed (n < {CAL_MIN_N})"
         and isinstance(calibration_or_note(np.full(10, 0.5), np.zeros(10, bool)), float))
    case("report: Markdown table renders NaN as blank and floats to 4 places",
         md_table(pd.DataFrame({"a": [1.0, np.nan], "b": ["x", "y"]})).splitlines()[2:] == ["| 1.0000 | x |", "|  | y |"])
    return out


def stage_selftest():
    r = selftest_results()
    for c in r:
        print(f"{'ok  ' if c['ok'] else 'FAIL'} {c['case']}")
    print(f"selftest: {sum(c['ok'] for c in r)} of {len(r)} passed")
    sys.exit(0 if all(c["ok"] for c in r) else 1)


def stage_recheck():
    """Read-only: the package and both freeze records; the guard; every record's status, pass, stage name, identity and
    expected predecessors; and whether the chain is ready for target-season work. An honest partial chain exits 0; an
    unclean package, failing freeze record, changed inventory, or failed, reserved, unreadable or mis-bound record
    exits nonzero."""
    problems = []
    ident = identity()
    c = code_ok()
    print(f"gamecalling: {'clean at the frozen digest' if c['ok'] else 'NOT CLEAN ' + str(c)}")
    if not c["ok"]:
        problems.append("gamecalling not clean at the frozen digest")
    for k, v in verify_records().items():
        print(f"{k} verify: {'passed' if v['ok'] else 'FAILED ' + str(v['tail'])}")
        if not v["ok"]:
            problems.append(f"{k} verify failed")
    if record_path("guard").exists():
        g = guard_compare()
        print("guard inventory: " + ("unchanged" if g["ok"] else
              f"CHANGED {g['changed'][:5]} added {g['added_to_inventory'][:5]} {g.get('detail', '')}"))
        if not g["ok"]:
            problems.append("guard inventory changed or guard record unusable")
    for p in sorted(RECORDS.glob("s9_*.json")):
        r, err = read_record(p)
        stage = p.stem.removeprefix("s9_")
        if err:
            bad = [err]
        else:
            bad = record_problems(stage, r, ident) if r.get("status") == "finished" else [f"status {r.get('status')}"]
        print(f"{p.name}: status={r.get('status') if r else '?'} passed={r.get('passed') if r else '?'} "
              f"{'ok' if not bad else 'PROBLEMS ' + str(bad)}")
        problems += [f"{p.name}: {x}" for x in bad]
    rd = ready_for_targets()
    print(f"ready for target-season work: {'yes' if rd['ready'] else 'no'}")
    for k, v in rd["problems"].items():
        if v:
            print(f"  {k}: {v}")
    print("recheck: " + ("passed" if not problems else "FAILED\n  " + "\n  ".join(problems)))
    sys.exit(0 if not problems else 1)


STAGES = {"selftest": stage_selftest, "guard": stage_guard, "audit": stage_audit, "control": stage_control,
          "gate": stage_gate, "domains": stage_domains, "recheck": stage_recheck}

if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "_ctx":
        if BIND is None:
            sys.exit("a context needs S9_BIND")
        CONTEXTS[sys.argv[2]]()
    elif len(sys.argv) == 2 and sys.argv[1] in STAGES:
        STAGES[sys.argv[1]]()
    elif len(sys.argv) == 3 and sys.argv[1] in SEASON_STAGE_FUNCS and sys.argv[2].isdigit():
        SEASON_STAGE_FUNCS[sys.argv[1]](int(sys.argv[2]))
    elif len(sys.argv) == 4 and sys.argv[1] == "authorize" and sys.argv[2].isdigit():
        stage_authorize(int(sys.argv[2]), sys.argv[3])
    else:
        sys.exit(f"usage: python3.12 s9_seasons.py {{{'|'.join(STAGES)}}}\n"
                 f"       python3.12 s9_seasons.py {{{'|'.join(SEASON_STAGE_FUNCS)}}} <year>\n"
                 f"       python3.12 s9_seasons.py authorize <year> <YYYY-MM-DD>   (only on the user's instruction)")
