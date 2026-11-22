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
Revision 10 (2026-11-22, after the user authorized A11-4 and the 2026 run): phase 3, 2026 (§7, §8.2; below). Review
rounds: Neon (forward binding, refetch precedence, §7 fallback tests, postponed dates) and Astra (the reconciled snapshot
pinned and built from; the step-1 schedule pinned; recoverable days against the window; the C8b regression through the
production forward route; coverage-first forward table; zero rule in the weighted Pearson). Official dates map games
to Savant's days (a suspended game keeps its date). Also the
small-group listing centers each season on the listed group's mean (Astra's 2023 review), and the mover-group summary
gains the secondary weighted Pearson. Accepts revision 9's records (domains, 2023).

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

Phase 3 (revision 10): 2026. After A11-4 (`authorize 2026`): the committed downloader fetches the days outside this
script (§8.2 step 2: `statcast_download_seasons.py fetch 2026 --start <first> --end <last>`, dates from the schedule);
then
    python3.12 s9_seasons.py prefetch 2026        # schedule and play-by-play into the cache
    python3.12 s9_seasons.py reconcile 2026       # once: §8.2 step 3 on the cached days; lists any refetch days
    (one documented refetch: the command the reconcile record prints), then
    python3.12 s9_seasons.py reconcile2 2026      # once, only after a refetch: step 5's decision
    or, if the first reconciliation is blocked for review only by feed-vocabulary events (revision 10d):
    python3.12 s9_seasons.py approve-exception 2026 <date>   # only on the user's instruction: EXCEPTION_RULE
    python3.12 s9_seasons.py reconcile_reviewed 2026         # once: the same snapshot, every event verified by the rule
    python3.12 s9_seasons.py publish 2026         # once: the downloader's build, the reference schema -> raw.parquet
    python3.12 s9_seasons.py build 2026; checks 2026
After the separate Stage C authorization (`authorize-run 2026 <date>`, only on the user's instruction):
    python3.12 s9_seasons.py forward 2026         # once: §7 forward queries and §6 row 10 (child process)
    python3.12 s9_seasons.py run 2026; halves 2026; validate 2026; report 2026
Smoke runs this whole path on the stand-in 2024, from a day cache seeded with the subset's development rows.

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
APPROVED_PREDECESSOR_SCRIPTS = frozenset({"080ac1996b03c8da17a6e1cd76bed80b4df99364b428f21b7ece080c65ce846c",
                                         # revision 9 (d07c31f): domains and the 2023 records
                                         "b7df0294683c373cd73c82eb93f7ecbf8ad150d52ca7c565af37b819fb065a10",
                                         # revision 10c (e9aa410): authorize_2026 (and the blocked reconcile_2026)
                                         "536bdc4337c712c29754df88bcb75dbd89e69ffd2511eb857c32fb30de3f1153"})
# The user's feed-exception rule (2026-11-22). reconcile_2026 was blocked for review on one event outside the
# development vocabulary: a swinging pitchout (MLB call code Q) for strike three with a passed ball, game 824606. The
# user approved accepting every such pitch once verified, as a known baseball situation. The rule is applied, never a
# list: each event is verified in both sources by exception_evidence, in reconcile_reviewed and again in checks; any
# other new signature, any failed condition and any unrecognized kept no-pitch event still stops. It is active only on a
# season with a verified exception_approval record (written on the user's instruction).
EXCEPTION_RULE = {"id": "swinging-strikeout-passed-ball-v1", "approved_by": "the user", "approved_on": "2026-11-22",
                  "text": "A pitch whose play-by-play signature is outside the development vocabulary is accepted only "
                          "if MLB's play-by-play and Statcast both show strike three on a swing that ends the plate "
                          "appearance in a strikeout, with a passed ball charged to the catcher on the same pitch. Its "
                          "Statcast row is kept unchanged (§5 B0: its description is accepted at that key only)."}
# Protocol amendment 1 (approved by the user 2026-11-22, after the first production control failed at C1 on this): the stored
# arsenal_support.parquet was written by Phase 0 on 2026-10-27 under ca9d25c, whose support_table counted rows with
# .size() (int64). Commit 483d573 (2026-10-29) added unit weights (weighting.w, float), so the frozen package's
# support_table sums floats and gives n as float64. All 36,385 rows' values are identical. (table, column, stored dtype,
# new dtype); applied only in tables_compare (C1, C2), only to integral finite values, and nothing else is relaxed.
PERMITTED_DTYPE = (("support", "n", "int64", "float64"),)

# Phase 2: target seasons (§3, §5, §6); phase 3 adds 2026's acquisition and forward stages (§7, §8.2).
TARGETS = (2023, 2026)               # production: the season stages run only on these, each after its authorization
SMOKE_STANDIN = 2024                 # smoke only: a development season stands in for a target, so no target row is read
SEASON_LABEL = {2023: "previously unmodeled, with prior data-quality inspection disclosed",
                2026: "not previously downloaded"}
AUTHORIZATION = {2023: "A11-3", 2026: "A11-4"}
RUN_AUTHORIZATION = {2026: "Stage C (A5)"}   # §10: 2026's forward and run stages need this second authorization
PAIRS = {2023: ((2023, 2024),), 2026: ((2025, 2026), (2024, 2026)),
         SMOKE_STANDIN: ((2024, 2025),)}     # §6 row 6: the A4 pairs a season belongs to
ACQ_DIR = (Path(SMOKE).resolve() / "statcast_seasons") if SMOKE else D / "statcast_seasons"   # the downloader's tree
SEASONS_ROOT = (Path(SMOKE).resolve() / "seasons") if SMOKE else D / "seasons"
REPORTS_OUT = (Path(SMOKE).resolve() / "reports") if SMOKE else ROOT / "reports"
SEASON_CONTEXT_STAGES = ("run", "halves", "validate", "forward")     # season stages whose work runs in a bound child process
PROTOCOL_LOCKED_ON = "2026-11-21"
CAL_MIN_N = 10                       # §6 row 10: calibration is not computed on fewer rows (Neon)
# Each child context belongs to exactly one stage (Astra: bind the dispatched context to its reserved stage).
CONTEXT_STAGE = {"smoke_ref": "control", "single": "control", "joint": "control", "halves": "control",
                 "replay": "control", "forward": "control",
                 "season_run": "run", "season_halves": "halves", "season_validate": "validate",
                 "season_forward": "forward"}
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
    gate = f"gate_v{MAPPING_VERSION}"
    if acquired(int(y)):
        r = f"authorize_run_{y}"
        return {"authorize": {"guard"}, "authorize_run": {"guard", a},
                "reconcile": {"guard", "audit", "control", gate, "domains", a},
                "reconcile2": {"guard", a, f"reconcile_{y}"},
                # the blocked first reconciliation failed, so it is pinned by the approval's digest, not as a predecessor
                "exception_approval": {"guard", a},
                "reconcile_reviewed": {"guard", "audit", "control", gate, "domains", a, f"exception_approval_{y}"},
                "publish": {"guard", a, last_reconcile(int(y))},
                "build": {"guard", a, f"publish_{y}"},
                "checks": {"guard", a, f"build_{y}", "domains", gate},
                "forward": {"guard", a, r, f"checks_{y}"},
                "run": {"guard", a, r, f"forward_{y}"},
                "halves": {"guard", a, r, f"run_{y}"},
                "validate": {"guard", a, r, f"run_{y}", f"halves_{y}"},
                "report": {"guard", a, r, f"checks_{y}", f"forward_{y}", f"run_{y}", f"validate_{y}"}}.get(name)
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
    return p + answer_problems(stage, rec)


ANSWER_STAGES = ("exception_approval", "reconcile_reviewed")


def answer_problems(stage: str, rec: dict) -> list:
    """exception_approval_<y> and reconcile_reviewed_<y> answer a FAILED record, which cannot be a predecessor; its
    digest is a required link checked wherever these records are read (record_problems: preconditions, publish via
    last_reconcile, recheck; chain_problems: every stage downstream). The answered record must exist at the pinned
    digest and still be a block the rule can answer (Astra)."""
    name, _, y = stage.rpartition("_")
    if name not in ANSWER_STAGES or not y.isdigit():
        return []
    a = (rec.get("result") or {}).get("answers")
    if not isinstance(a, dict) or a.get("record") != f"reconcile_{y}":
        return [f"answers does not name reconcile_{y}"]
    q = record_path(a["record"])
    if not q.exists() or sha(q) != a.get("sha256"):
        return [f"the answered record {a['record']} is missing or not at the pinned digest"]
    first, bad = blocked_reconcile(int(y))
    out = [f"the answered record: {x}" for x in bad]
    if first is not None:       # both answer stages carry the approval's per-game evidence (Astra)
        out += [f"evidence: {x}" for x in feed_signature_problems(
            a.get("games"), ((first.get("result") or {}).get("completeness") or {}).get("feed_stop"))]
    return out


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
    ans = (rec.get("result") or {}).get("answers") if isinstance(rec.get("result"), dict) else None
    if isinstance(ans, dict) and "record" in ans:      # a pinned failed record (revision 10d): the link must hold
        q = path_of(ans["record"])
        if not q.exists() or sha(q) != ans.get("sha256"):
            out.append(f"{stage} answers {ans['record']} at another digest (or it is missing)")
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
                "run_authorization": "smoke", "label": "smoke stand-in (a development season; means nothing)"}
    if year not in TARGETS:
        raise SystemExit(f"refused: season stages are implemented for {TARGETS} only")
    if year == 2026:
        return {"year": year, "source": SEASONS_ROOT / "2026" / "raw.parquet", "pairs": PAIRS[year],
                "authorization": AUTHORIZATION[year], "run_authorization": RUN_AUTHORIZATION[year],
                "label": SEASON_LABEL[year]}
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
    if acquired(year) and stage in RUN_STAGES:      # §10: 2026's run stages need the separate Stage C authorization
        rr, rerr = read_record(record_path(f"authorize_run_{year}"))
        rbad = [rerr] if rerr else record_problems(f"authorize_run_{year}", rr, identity())
        if rbad or (rr.get("result") or {}).get("season") != year:
            raise SystemExit(f"refused: no valid run authorization for {year} ({rbad}); it is written only on the "
                             f"user's explicit instruction: s9_seasons.py authorize-run {year} <YYYY-MM-DD>")
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
    if acquired(year):
        raise SystemExit(f"refused: {year}'s raw file is published from the reconciled download (§8.2), not extracted")
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
        bad = verify_inputs(f"{raw_stage(year)}_{year}")
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


def b0_inventory(raw: pd.DataFrame, dom: dict, excepted_keys=frozenset()) -> dict:
    """§5 B0 on the whole raw extract (game types; attach_bb_type's preconditions) and on its regular-season rows
    (closed domains; null counts by position, reported without a rule). `excepted_keys`: the (game_pk, at_bat_number,
    pitch_number) keys EXCEPTION_RULE verified in this stage's completeness check; a description outside the domain at
    one of them is kept unchanged and listed under description_exceptions, never anywhere else (revision 10d)."""
    out = {"game_type": {str(k): int(v) for k, v in raw["game_type"].value_counts(dropna=False).items()}}
    out["duplicate_keys_whole_file"] = int(raw.duplicated(batter.BB_KEYS).sum())
    out["bb_type_outside_labels_whole_file"] = sorted(map(str, set(raw["bb_type"].dropna().unique()) - set(batter.BB_TYPES)))
    r = raw[raw["game_type"] == "R"]
    term = terminal_mask(r)
    exc = pd.MultiIndex.from_frame(r[["game_pk", "at_bat_number", "pitch_number"]].astype("int64")).isin(
        [tuple(map(int, k)) for k in excepted_keys]) if len(excepted_keys) and len(r) else np.zeros(len(r), bool)
    viol = {}
    for col in ("description", "bb_type", "stand", "p_throws", "inning_topbot"):
        rc = r.loc[~exc, col] if col == "description" else r[col]
        extra = sorted(set(rc.dropna().unique()) - set(dom[col]))
        if extra:
            viol[col] = {str(v): int(rc.eq(v).sum()) for v in extra}
    de = r.loc[exc & ~r["description"].isin(dom["description"]).to_numpy()]
    out["description_exceptions"] = de[["game_pk", "at_bat_number", "pitch_number", "description"]].astype(
        {"game_pk": "int64", "at_bat_number": "int64", "pitch_number": "int64", "description": "object"}).to_dict("records")
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
    cs = cs.reindex([str(z) for z in (*range(1, 10), 11, 12, 13, 14)] + ["missing or other"], fill_value=0)
    cs.index.name = "zone"
    cs = cs.rename(columns={"size": "denominator", "sum": "called_strikes"}).reset_index()
    cs["rate"] = (cs["called_strikes"] / cs["denominator"].where(cs["denominator"] > 0)).astype(float)
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
    seq = {"season": season, "rows": int(p["pa_sequence_error"].astype(bool).sum()), "rows_total": len(p),
           "row_share": float(p["pa_sequence_error"].astype(bool).mean()),
           "pas": int(a["pa_sequence_error"].astype(bool).sum()), "pas_total": len(a),
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


def _num_is(v, k: int) -> bool:
    """v is a real number exactly equal to k (no bool, no string, no NaN; 2.5 is not 2)."""
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)) \
        and bool(pd.notna(v)) and float(v) == float(k)


def exception_evidence(play: dict, j: int, rows) -> tuple:
    """EXCEPTION_RULE on one play event (list position j of the play) and the game's Statcast rows: (ok, evidence).
    Every condition must hold in both sources, and conflicting evidence fails closed (Astra): the event's explicit MLB
    index must equal j and be unique in the play, and the passed ball must name that index; Statcast's description
    must be a swinging label (its first token "swinging") and its strikes exactly 2. The evidence keeps every value
    read and lists each failed condition."""
    evs = play.get("playEvents") or []
    e = evs[j] if 0 <= j < len(evs) else {}
    d = e.get("details") or {}
    call = d.get("call") or {}
    res = play.get("result") or {}
    idx = e.get("index")
    explicit = isinstance(idx, int) and not isinstance(idx, bool) and idx == j and \
        [i for i, x in enumerate(evs) if x.get("index") == idx] == [j]
    kidx = [i for i, x in enumerate(evs) if kept(x)]
    ab = int(play["about"]["atBatIndex"]) + 1
    n = kidx.index(j) + 1 if j in kidx else None
    pb_idx = [(x.get("details") or {}).get("playIndex") for x in play.get("runners") or []
              if isinstance(x, dict) and (x.get("details") or {}).get("eventType") == "passed_ball"]
    pa = rows[rows["at_bat_number"] == ab] if rows is not None else None
    row = pa[pa["pitch_number"] == n] if pa is not None and n is not None else None
    one = row is not None and len(row) == 1
    g = (lambda c: row[c].iloc[0] if one and c in row else None)  # noqa: E731
    sdesc, sev, sstr, sdes, spt = g("description"), g("events"), g("strikes"), g("des"), g("pitch_type")
    mstr = (e.get("count") or {}).get("strikes")
    cond = {"MLB: a pitch": e.get("type") == "pitch" and e.get("isPitch") is True,
            "MLB: the event's explicit index equals its position and is unique in the play": explicit,
            "MLB: a swing (call 'Swinging ...')": isinstance(call.get("description"), str)
            and call["description"].startswith("Swinging "),
            "MLB: a strike": d.get("isStrike") is True,
            "MLB: strike three (the count reaches exactly three strikes)": _num_is(mstr, 3),
            "MLB: the last kept event of the plate appearance": bool(kidx) and kidx[-1] == j,
            "MLB: the plate appearance ends in a strikeout": res.get("eventType") == "strikeout",
            "MLB: a passed ball on this pitch (a runner record's playIndex equals the event's index)":
                explicit and pb_idx.count(idx) >= 1,
            "MLB: 'Passed ball by catcher' in the play description": "Passed ball by catcher" in str(res.get("description")),
            "Statcast: exactly one row at the key": one,
            "Statcast: a swinging description (first token 'swinging')": isinstance(sdesc, str)
            and sdesc.split("_")[0] == "swinging",
            "Statcast: events strikeout": sev == "strikeout",
            "Statcast: exactly two strikes before the pitch": _num_is(sstr, 2),
            "Statcast: the last row of the plate appearance": one and int(pa["pitch_number"].max()) == n,
            "Statcast: 'passed ball by catcher' in des": isinstance(sdes, str) and "passed ball by catcher" in sdes.lower()}
    py = (lambda v: v.item() if hasattr(v, "item") else v)  # noqa: E731
    ev = {"at_bat_number": ab, "pitch_number": n, "signature": list(event_signature(e)),
          "mlb_event_index": idx, "list_position": j, "passed_ball_play_indices": pb_idx, "mlb_count_strikes": mstr,
          "mlb_call": call.get("description"), "mlb_play": res.get("description"), "mlb_result": res.get("eventType"),
          "statcast_description": py(sdesc), "statcast_events": py(sev), "statcast_strikes": py(sstr),
          "statcast_pitch_type": py(spt), "statcast_des": py(sdes),
          "failed": [k for k, v in cond.items() if not v]}
    return not ev["failed"], ev


def signature_key(sig) -> str:
    return json.dumps(list(sig))


def game_feed_evidence(g: int, vocab_dev: set) -> dict:
    """An affected game's play-by-play as the approval saw it: the cache file's digest and each signature outside the
    development vocabulary with its occurrence count."""
    path = MLB_CACHE / "pbp" / f"{int(g)}.json"
    pbp = json.loads(path.read_bytes())
    voc = vocabulary(pbp)
    return {"game_pk": int(g), "pbp_sha256": sha(path),
            "signatures": {signature_key(k): int(v) for k, v in sorted(voc.items(), key=str) if k not in vocab_dev}}


def exception_binding_problems(comp: dict, answers: dict) -> list:
    """The rule's verified events must be exactly the approved evidence (Astra): the same games, each game's cached
    play-by-play unchanged since the approval, and per game the same signatures with the same occurrence counts. A
    now-missing original event, an extra one or a changed feed is a problem, never silently accepted."""
    approved = {int(x["game_pk"]): x for x in (answers or {}).get("games") or []}
    if not approved:
        return ["the approval records no affected-game evidence"]
    got = {}
    for v in comp.get("verified_exceptions") or []:
        c = got.setdefault(int(v["game_pk"]), {})
        c[signature_key(v["signature"])] = c.get(signature_key(v["signature"]), 0) + 1
    p = [] if set(got) == set(approved) else [f"verified games {sorted(got)} are not the approved {sorted(approved)}"]
    for g, a in sorted(approved.items()):
        path = MLB_CACHE / "pbp" / f"{g}.json"
        if not path.exists() or sha(path) != a.get("pbp_sha256"):
            p.append(f"game {g}: the cached play-by-play is missing or changed since the approval")
        if got.get(g, {}) != a.get("signatures"):
            p.append(f"game {g}: verified signatures {got.get(g, {})} are not the approved {a.get('signatures')}")
    return p


def feed_signature_problems(games: list, feed_stop: list) -> list:
    """The approval's per-game evidence (signatures read from the current cached feed) against the blocked record's
    feed_stop: the same games, and per game the same normalized set of signatures outside the vocabulary (Astra: a
    feed changed after the block, e.g. Q to Q2, must not be approved). The blocked record lists signatures, not
    occurrence counts; the counts and feed digests bind the approval to reconcile_reviewed and checks."""
    want = {int(f["game_pk"]): sorted(signature_key(x) for x in f.get("signatures_outside_vocabulary") or [])
            for f in feed_stop or []}
    have = {int(g["game_pk"]): sorted(g.get("signatures") or {}) for g in games or []}
    if not want:
        return ["the blocked record lists no feed-stop game"]
    p = [] if set(have) == set(want) else [f"approved games {sorted(have)} are not the blocked games {sorted(want)}"]
    p += [f"game {g}: signatures {have.get(g)} are not the blocked record's {w}" for g, w in sorted(want.items())
          if have.get(g) != w]
    return p


def feed_check(pbp: dict, rows, vocab_dev: set, rule: bool) -> dict:
    """One game's feed-mapping stop (§8.1), with EXCEPTION_RULE when `rule`: a game with signatures outside the
    development vocabulary clears only if every such event satisfies the rule and no kept no-pitch event is
    unrecognized."""
    new = sorted(set(vocabulary(pbp)) - vocab_dev, key=str)
    unrec = sorted({event_signature(e) for p in pbp["allPlays"] for e in p["playEvents"] if unrecognized(e)}, key=str)
    verified, failed = [], []
    if rule and new and not unrec:
        newset = set(new)
        for p in pbp["allPlays"]:
            for j, e in enumerate(p["playEvents"]):
                if event_signature(e) in newset:
                    ok, ev = exception_evidence(p, j, rows)
                    (verified if ok else failed).append(ev)
    stop = bool(unrec) or (bool(new) and (not rule or bool(failed) or not verified))
    return {"new": new, "unrec": unrec, "verified": verified, "failed": failed, "stop": stop}


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


def season_completeness(year: int, r: pd.DataFrame, vocab_dev: set, rule: bool = False) -> dict:
    """§5 / §8.1 on a target season: completed-game definition, score check (with the non-pitch explanation), key-level
    event coverage under the current mapping, and the feed-mapping stop: an event signature outside the gate record's
    development vocabulary, or a kept no-pitch event the development gate would flag as unrecognized (the same
    classifier, Astra). With `rule` (the user's EXCEPTION_RULE, revision 10d), new-signature events that the rule
    verifies in both sources are listed under verified_exceptions instead of stopping."""
    sch = json.loads(fetch(SCHEDULE_URL.format(season=year), MLB_CACHE / "schedule" / f"{year}.json", valid_schedule))
    comp = completed_games(sch)
    games = completeness_games(comp)
    statcast = {g: set(map(tuple, x[["at_bat_number", "pitch_number"]].to_numpy())) for g, x in r.groupby("game_pk")}
    smax = r.groupby("game_pk")[["post_home_score", "post_away_score"]].max()
    s = {"completed": len(comp), "checked": len(games), "duplicate_statcast_keys":
         int(r.duplicated(["game_pk", "at_bat_number", "pitch_number"]).sum()),
         "unexpected_statcast_games": sorted(int(g) for g in set(r["game_pk"]) - set(comp.index)),
         "unavailable": [], "feed_stop": [], "missing_games": [], "key_mismatch": [], "score_explained": [],
         "score_unexplained": [], "new_signatures": {},
         "exception_rule": EXCEPTION_RULE["id"] if rule else None, "verified_exceptions": []}
    rows_g = {int(g): x for g, x in r.groupby("game_pk")} if rule else {}

    def one(g):
        try:
            return g, json.loads(fetch(PBP_URL.format(pk=g), MLB_CACHE / "pbp" / f"{g}.json", valid_pbp)), None
        except Exception as exc:  # noqa: BLE001
            return g, None, str(exc)
    with ThreadPoolExecutor(max_workers=FETCH_THREADS) as ex:
        for g, pbp, err in ex.map(one, games):
            if g not in statcast:            # absent from Statcast, independent of the reference fetch (Astra)
                s["missing_games"].append(g)
            if err:
                s["unavailable"].append({"game_pk": g, "error": err})
                continue
            fc = feed_check(pbp, rows_g.get(int(g)), vocab_dev, rule)
            new, unrec = fc["new"], fc["unrec"]
            for x in new:
                s["new_signatures"][str(x)] = s["new_signatures"].get(str(x), 0) + 1
            if fc["stop"]:
                s["feed_stop"].append({"game_pk": g, "signatures_outside_vocabulary": [list(x) for x in new],
                                       "unrecognized_kept_no_pitch": [list(x) for x in unrec],
                                       **({"exception_rule_failed": fc["failed"]} if rule else {})})
            else:
                s["verified_exceptions"] += [{"game_pk": int(g), **v} for v in fc["verified"]]
            if g not in statcast:
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
        bad = verify_inputs(f"{raw_stage(year)}_{year}", f"build_{year}")
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
        t = time.time()     # first: B0 accepts an out-of-domain description only at keys the rule verified here
        rule = exception_rule_active(year)
        res["completeness"] = season_completeness(year, r, gate_vocabulary(), rule=rule)
        res["completeness"]["seconds"] = time.time() - t
        vx = res["completeness"]["verified_exceptions"]
        if rule:      # the same evidence the approval saw (Astra): a missing, extra or changed event stops
            ap = load_record(f"exception_approval_{year}")["result"]["answers"]
            res["exception_binding"] = exception_binding_problems(res["completeness"], ap)
        bk = pitches.set_index(["game_pk", "at_bat_number", "pitch_number"])
        for v in vx:          # the disclosure is read from the built row, not assumed (Astra)
            k = (v["game_pk"], v["at_bat_number"], v["pitch_number"])
            b = bk.loc[k] if k in bk.index else None
            v["built"] = None if b is None else {
                "pitch_type": None if pd.isna(b["pitch_type"]) else str(b["pitch_type"]),
                "outcome": None if pd.isna(b["outcome"]) else str(b["outcome"]),
                "is_pitchout": bool(b["is_pitchout"]), "outcome_fit_eligible": bool(b["outcome_fit_eligible"]),
                "exclusion_reason": None if pd.isna(b["exclusion_reason"]) else str(b["exclusion_reason"])}
        ek = {(v["game_pk"], v["at_bat_number"], v["pitch_number"]) for v in vx}
        res["B0"] = b0_inventory(raw_all, dom, frozenset(ek))
        res["B0"]["new_pitch_types"] = sorted(set(res["B0"]["pitch_types"]) - set(map(str, dev_p["pitch_type"].dropna().unique())))
        res["B1"] = b1_counts(r, pitches)
        b2 = b2_table(pitches, year, dev_p, target_sample_label(year))
        res["prefit"] = prefit_checks(r, pitches, raw_all)
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
        if res.get("exception_binding"):
            stops.append("exception evidence differs from the approval")
        if any(v.get("built") is None for v in vx):
            stops.append("a verified exception has no built pitch row")
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
        bad = verify_inputs(f"{raw_stage(year)}_{year}", f"build_{year}")
        if bad:
            return {"passed": False, "stopped": f"inputs changed: {bad}"}
        c = run_child("season_run", season_binding(year, "run"))
        return {"context": c, "outputs": c.get("outputs", {}), "passed": bool(c.get("passed")) and bool(c.get("outputs"))}
    run_season_stage(year, "run", body)


def stage_halves(year: int):
    season_gate(year, "halves")

    def body():
        bad = verify_inputs(f"{raw_stage(year)}_{year}", f"build_{year}", f"run_{year}")
        if bad:
            return {"passed": False, "stopped": f"inputs changed: {bad}"}
        c = run_child("season_halves", season_binding(year, "halves"))
        return {"context": c, "outputs": c.get("outputs", {}), "passed": bool(c.get("passed")) and bool(c.get("outputs"))}
    run_season_stage(year, "halves", body)


def stage_validate(year: int):
    season_gate(year, "validate")

    def body():
        bad = verify_inputs(f"{raw_stage(year)}_{year}", f"build_{year}", f"run_{year}", f"halves_{year}")
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
        if acquired(year):     # §2: R3a (documented practice) is vp's R3 relabeled; R3b also drops possible exposure
            t = res["sensitivity"]["table"].copy()
            t["check"] = t["check"].replace({"R3_D": "R3a_D", "R3_A": "R3a_A"})
            ex = res["sensitivity"]["excluded"].copy()
            if len(ex):
                ex["check"] = ex["check"].replace({"R3_D": "R3a_D", "R3_A": "R3a_A"})
            res["sensitivity"] = {"table": t, "excluded": ex}
            res["r3b"] = r3b_full(scores, config.PITCHES_PARQUET, fits, sets, (year,))
            res["flagged_catchers"] = flagged_catchers(scores, config.PITCHES_PARQUET, year)
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
            "small_groups": small_group_listing(rep, movers), "by_group_weighted": group_weighted_pearson(rep, movers)}


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
            e_all = rep["effects"][sc, "random"].reindex(elig)
            e = e_all.reindex(ids)
            c0, c1_ = e["y0"] - e["y0"].mean(), e["y1"] - e["y1"].mean()
            a0, a1 = e_all["y0"] - e_all["y0"].mean(), e_all["y1"] - e_all["y1"].mean()
            for c_ in ids:
                rows.append({"group": grp, "score": sc, "catcher": c_, "saved_0": e.loc[c_, "y0"], "saved_1": e.loc[c_, "y1"],
                             "raw_difference": e.loc[c_, "y1"] - e.loc[c_, "y0"],
                             "centered_on_pair": a1.loc[c_] - a0.loc[c_], "change_centered": c1_.loc[c_] - c0.loc[c_]})
    return pd.DataFrame(rows, columns=["group", "score", "catcher", "saved_0", "saved_1", "raw_difference",
                                       "centered_on_pair", "change_centered"])


def group_weighted_pearson(rep: dict, movers: pd.DataFrame) -> pd.DataFrame:
    """§4's secondary weighted Pearson by mover group (random fits), with the pair's weights; a group under MIN_SET is
    listed, not correlated (revision 10, Astra's 2023 review)."""
    rows = []
    for grp in ("mover", "non_mover", "neither"):
        ids = sorted(set(movers.index[movers["group"] == grp]) & set(rep["eligible"]))
        for sc in ("D", "A"):
            e = rep["effects"][sc, "random"].reindex(ids)
            t = rep["table"]
            zero = (t[(t["score"] == sc) & (t["model"] == "random")]["pearson"] == vp.UNDEFINED).any()
            wp = (vp.UNDEFINED if zero else f"listed (n < {vp.MIN_SET})" if len(ids) < vp.MIN_SET else
                  vp.weighted_pearson(e["y0"].to_numpy(), e["y1"].to_numpy(), e["w"].to_numpy()))
            rows.append({"group": grp, "score": sc, "n": len(ids), "weighted_pearson": wp})
    return pd.DataFrame(rows)


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
    "transmitters and call their own pitches since 2023."],
    2026: [
    "ABS challenge system: two challenges per team; the pitcher, catcher or batter challenges; an overturned call "
    "changes the count. Assumption, kept explicit: `description` records the final call after any challenge.",
    "Base coaches confined to the box.",
    "Savant documents changed measurements (plate_x and plate_z at mid-plate; ABS-defined sz_top and sz_bot) and gives "
    "no cross-year algorithm for `zone`: zone comparability across years may be affected (the called-strike-by-zone "
    "check and C1's chase feature, zones 11-14). The frozen forward predictor does not read 2026 zones; under ABS, "
    "2026's outcomes, counts and eligible sample, and so forward performance, can still change (context).",
    "Pitch timer, shift limits, larger bases and disengagement limits since 2023; pitchers may wear PitchCom "
    "transmitters and call their own pitches since 2023.",
    "Dugout pitch calling (§2): MIA documented for all 2026 regular-season games (R3a, R3b); COL possible exposure "
    "from Opening Day with no supported last date (R3b only); NYM: no regular-season documentation found (not a "
    "verified negative). Primary analyses keep every pitch (D1)."]}
PAIR_LABEL = {(2023, 2024): "Combines one previously unmodeled season (2023) with one development season (2024).",
              (2025, 2026): "Combines the last development season (2025) with the not previously downloaded season (2026).",
              (2024, 2026): "A separately labeled two-year comparison (2024 and 2026), not consecutive seasons.",
              (2024, 2025): "Smoke stand-in pair (development seasons)."}
REFERENCE_DISCLOSURE = ("Reference seasons: 2025 data are drawn from the frozen 2025 extract, which covers 2,428 of 2,430 "
                        "completed games (the two Tokyo games are absent); the frozen 2024 extract omits one pitch "
                        "recorded by MLB (game 746731, plate appearance 68; protocol amendment 2).")
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


def split_listing(d: dict) -> list:
    """Protocol §3: a split whose compared set is under MIN_SET is listed, not correlated: each catcher's two half
    estimates (D and A, both fits), and the half-specific drop reasons."""
    L = []
    small = [c for c in d.get("compare", []) if isinstance(c.get("n"), (int, np.integer)) and c["n"] < vp.MIN_SET]
    if small:
        for k, e in d.get("effects", {}).items():
            L += [f"Listed (n < {vp.MIN_SET}), {k[0]} {k[1]}:", "", md_table(e.reset_index())]
    if len(d.get("dropped", [])):
        L += ["Dropped catchers and half-specific reasons:", "", md_table(d["dropped"])]
        dropped = list(d["dropped"]["catcher"])
        for k in (("D", "random"), ("A", "random")):
            e = (d.get("effects_all") or {}).get(k)
            if e is not None:
                L += [f"Dropped catchers' half estimates where a half fit exists, {k[0]} {k[1]}:", "",
                      md_table(e.reindex([c for c in dropped if c in e.index]).reset_index())]
    return L


def refit_diagnostics(d: dict) -> list:
    """The half refit's logged diagnostics (P0: games, candidates, scored, shares and each role fold's pooling
    strengths; C1 logs scored pitches and resources only, which is said)."""
    meta = d.get("meta") or {}
    rows = []
    for h, m in meta.items():
        r = {"half": h, **{k: m[k] for k in ("games", "candidates", "scored", "scored_share", "out_of_arsenal_share")
                           if k in m}}
        for role, a in (m.get("alphas") or {}).items():
            r.update({f"alpha_{c}_role{role}": v for c, v in a.items()})
        rows.append(r)
    if not rows:
        return []
    note = [] if any("alphas" in m for m in meta.values()) else \
        ["This version's half refit logs scored pitches only; its pooling strengths were not logged.", ""]
    return ["Half-refit diagnostics:", "", md_table(pd.DataFrame(rows))] + note


def forward_section(fwd: dict) -> list:
    """§6 row 10 (2026 only): forward prediction with the frozen 2025 P0 and C1 and both baselines fitted on 2025."""
    t = fwd["table"]
    cols = [c for c in ("component", "stratum", "n", "share", "P0", "C1", "gain_C1_minus_P0", "criterion_baseline",
                        "criterion_zero_probability_rows", "league_pooled_P0", "max_cal_P0", "max_cal_C1",
                        "max_cal_criterion", "max_cal_league") if c in t.columns]
    L = ["## Forward prediction (§6 row 10; frozen 2025 models; nats)", "",
         "All 2026 outcome-fit-eligible pitches (contact: contact-fit-eligible), with no arsenal restriction. Seen cell, "
         "unseen pitcher and action unseen in 2025 are defined on the 2025 outcome-training rows (both components); "
         "unseen batter on the 2025 C1 model's batters. A stratum under 10 rows gets log loss and n only. Descriptive; "
         "no gate.", "", md_table(t[cols], 5),
         "Stratum counts: " + "; ".join(f"{k}: {v}" for k, v in fwd["strata_counts"].items()), ""]
    hit = [z for z in fwd["zero"] if z["zero_probability_rows"]]
    if hit:
        L += ["**Criterion baseline: zero probabilities (disclosure).** The criterion baseline fitted on 2025 (unchanged "
              "definition) gave an observed 2026 class probability exactly 0; its log loss is +inf where so, reported "
              "as such, nothing floored.", ""]
        for z in hit:
            L += [f"- {z['component']}: {z['zero_probability_rows']} of {z['n']} rows.", "",
                  md_table(pd.DataFrame(z["breakdown"])), md_table(pd.DataFrame(z["rows"]))]
            if z["zero_probability_rows"] > z["rows_listed"]:
                L += [f"(The first {z['rows_listed']} of {z['zero_probability_rows']} rows are listed; all are in the "
                      f"breakdown above.)", ""]
    full_cal = [c for c in t.columns if c.startswith("cal_")]
    L += ["Calibration by category (10 equal-count bins), every stratum:", "",
          md_table(t[["component", "stratum"] + full_cal], 5)]
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
    L += ["## Disclosures", "", f"- {REFERENCE_DISCLOSURE}",
          f"- Season label: {info['label']}." + (" Prior 2023 data-quality inspection: completeness, label-share and "
                                                 "column-availability checks and a schema comparison, before any model "
                                                 "touched 2023 (protocol, Disclosure)." if year == 2023 else ""), ""]
    c = checks["completeness"]
    L += ["## Row checks (§5)", "",
          f"- Completeness: **{c['outcome']}**; {c['completed']} completed games on the schedule, {c['checked']} checked; "
          f"score discrepancies explained by a verified non-pitch scoring event: {len(c['score_explained'])}"
          + (" (" + "; ".join(f"game {e['game_pk']}: official {e['official'][0]}-{e['official'][1]}, Statcast "
                               f"{e['statcast'][0]}-{e['statcast'][1]}, non-pitch runs after the last pitch "
                               f"{e['nonpitch_runs_after_last_pitch']}" for e in c["score_explained"]) + ")"
             if c["score_explained"] else "") + ".",
          *([f"- Feed-vocabulary exceptions accepted under the user's rule `{c.get('exception_rule')}` (approved "
             f"{EXCEPTION_RULE['approved_on']}; {EXCEPTION_RULE['text']}): {len(c['verified_exceptions'])} event(s): "
             + "; ".join(f"game {v['game_pk']}, plate appearance {v['at_bat_number']}, pitch {v['pitch_number']} "
                         f"(MLB call {v['mlb_call']!r}, Statcast description {v['statcast_description']!r}, events "
                         f"{v['statcast_events']!r}): {v['mlb_play']} Built row as the frozen code made it: "
                         + (f"pitch type {(v.get('built') or {}).get('pitch_type')!r}, outcome "
                            f"{(v.get('built') or {}).get('outcome') or 'none (the description has no mapping)'}, "
                            f"pitchout flag {(v.get('built') or {}).get('is_pitchout')}, exclusion reason "
                            f"{(v.get('built') or {}).get('exclusion_reason')!r}, outcome fit-eligible "
                            f"{(v.get('built') or {}).get('outcome_fit_eligible')}" if v.get("built") else "not found")
                         for v in c["verified_exceptions"])
             + ". Each was verified in both sources and matches the approval's evidence; its row is kept unchanged."]
            if c.get("verified_exceptions") else []),
          f"- B0: duplicate keys in the whole raw file {checks['B0']['duplicate_keys_whole_file']}; domain violations "
          f"{checks['B0']['domain_violations'] or 'none'}; descriptions outside the domain accepted at rule-verified "
          f"keys only {checks['B0'].get('description_exceptions') or 'none'}; events accepted through the frozen code list only "
          f"{checks['B0']['events_accepted_through_code_only'] or 'none'}; null counts {checks['B0']['nulls']}; "
          f"bb_type nulls {checks['B0']['bb_type_nulls']}.",
          f"- B1 (report): {checks['B1']}.", f"- B2: pitch types not seen in 2024-2025: {checks['B2_new_in_target'] or 'none'}.",
          "- Pre-fit checks: " + ("all passed" if all(x["ok"] for x in checks["prefit"]) else
                                  str([x for x in checks["prefit"] if not x["ok"]])), ""]
    env_head = "### Environment (reported against 2024 and 2025" + (
        "; 2026 zones are labeled possibly not comparable, §2)" if acquired(year) else ")")
    L += ["### Coverage", "", md_table(tables["coverage"]), env_head, "",
          md_table(tables["called_strike"]), md_table(tables["walks"]), md_table(tables["sequence_errors"]),
          md_table(tables["terminal"]), "### Pitch types (B2)", "", md_table(tables["B2"]), ""]
    L += ["## Full run (§6 row 1)", ""]
    for v, d in run["versions"].items():
        L.append(f"- {v}: {d['scored']} scored pitches; abnormal-termination records {len(d['abnormal_log'])}.")
    L += [f"- C1 post-run reconstruction (coverage first): {run['reconstruction_C1']}.", "",
          "## Held-out prediction (§6 row 2; one matched set per component; nats)", "", md_table(full["heldout"], 5),
          "Descriptive; no gate. A lower held-out log loss shows better probabilities, not better comparisons between "
          "pitch choices.", ""]
    L += zero_probability_note(full.get("heldout_zero", []))
    if acquired(year) and "forward" in run:
        L += forward_section(run["forward"])
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
              md_table(pd.DataFrame(s["split_random"][year]["gap"])), *split_listing(s["split_random"][year]),
              "### Split of existing scores (calendar halves; secondary)", "",
              md_table(pd.DataFrame(s["split_calendar"][year]["compare"])),
              md_table(pd.DataFrame(s["split_calendar"][year]["gap"])), *split_listing(s["split_calendar"][year]),
              "### Independent refit in each half", "", md_table(pd.DataFrame(s["refit"][year]["compare"])),
              md_table(pd.DataFrame(s["refit"][year]["gap"])), *split_listing(s["refit"][year]),
              *refit_diagnostics(s["refit"][year]),
              "### Between-team share, staff diagnostic", "",
              "Between-team share: the descriptive share of fitted-score variation between primary-team groups, not "
              "the fraction caused by teams; unrelated scores would give about (T - 1)/(N - 1).", "",
              md_table(s["team_staff"]["share"].assign(unrelated_benchmark=lambda t: (t["teams"] - 1) / (t["catchers"] - 1))),
              "Staff diagnostic: the correlation with the pitch-weighted mean of the same fit's pitcher effects; a "
              "diagnostic of how the fitted model divides variation between catchers and pitchers, not independent "
              "evidence about staff quality.", "", md_table(s["team_staff"]["staff"]), "### Same-team pairs (protocol §6.4)", "",
              "Same team; pitcher assignments may differ. No test.", "",
              md_table(s["team_staff"]["pairs"]),
              "### Sensitivity (R1, R2, " + ("R3a, R3b" if acquired(year) else "R3") + ", R4)", "",
              md_table(s["sensitivity"]["table"]), "Sensitivity exclusions:", "", md_table(s["sensitivity"]["excluded"])]
        L.append("R4 (bases empty only) changes only the ranking sample: the outcome, contact and policy models and "
                 "the payoffs are still learned from the runners-on pitches it excludes.\n")
        if year == 2023:
            L.append("R3 in 2023: no club is flagged, so R3 removes no pitches.\n")
        if acquired(year) and "r3b" in s:
            L += ["R3a drops the documented club's fielding-team pitches (MIA); R3b also drops the possible-exposure "
                  "club's (COL), as a broad bound. Both keep the opponents' pitches and change only the ranking sample.",
                  "", md_table(s["r3b"]["table"]), "R3b excluded catchers:", "", md_table(s["r3b"]["excluded"]),
                  "Flagged clubs' catchers (§2):", "", md_table(s["flagged_catchers"])]
        for pair, p in full[v]["pairs"].items():
            L += [f"### Pair {pair[0]}-{pair[1]}", "", PAIR_LABEL.get(tuple(pair), "") + " The other season's scores are "
                  "the stored ones of the same version; mover groups come from this pair's primary teams. The pairs are "
                  "not independent.", "", md_table(p["repeatability"]["table"]),
                  f"Mover groups among the pair's eligible catchers: {p['mover_counts_eligible']}. Mover comparisons, "
                  f"not natural experiments. Within-season movers (at least 300 scorable pitches for two or more "
                  f"teams): " + ", ".join(f"{se}: {int(pr['within_mover'].sum())}" for se, pr in p["primary"].items())
                  + ".", "",
                  "By mover group (random fits; groups under 10 are listed, not correlated):", "",
                  md_table(p["repeatability"]["by_group"]),
                  "Weighted Pearson by mover group (secondary; random fits):", "",
                  md_table(p["by_group_weighted"] if "by_group_weighted" in p else
                           group_weighted_pearson(p["repeatability"], p["movers"])),
                  "Catchers in groups under 10 (change centered on the listed group's means, as score differences are):",
                  "", md_table(p["small_groups"] if "change_centered" in p["small_groups"] else
                               small_group_listing(p["repeatability"], p["movers"])), "Disconnected catchers:", "",
                  md_table(p["repeatability"]["disconnected"])]
        if year == 2026 and not SMOKE:
            ref = STORED_REF["v2"] if v == "P0" else STORED_REF["s8"]
            with open(ref, "rb") as f:
                stored = pickle.load(f)["repeatability"]["table"]
            L += ["### Pair 2024-2025 (stored results, examined; not recomputed)", "",
                  f"From `{Path(ref).relative_to(ROOT)}`. " + REFERENCE_DISCLOSURE, "", md_table(stored)]
    L += ["## Left out (D2, approved)", "", "; ".join(LEFT_OUT) + ".", ""]
    return "\n".join(L)


def stage_report(year: int):
    season_gate(year, "report")

    def body():
        bad = verify_inputs(f"checks_{year}", f"run_{year}", f"validate_{year}",
                            *([f"forward_{year}"] if acquired(year) else []))
        if bad:
            return {"passed": False, "stopped": f"inputs changed: {bad}"}
        checks = load_record(f"checks_{year}")["result"]
        run = load_record(f"run_{year}")["result"]["context"]
        if acquired(year):
            with open(season_dir(year) / "forward" / "forward.pkl", "rb") as f:
                run = {**run, "forward": pickle.load(f)}
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


# ===================================================================================== phase 3: 2026 (§7, §8.2)
#
# Order (§10): after A11-4: authorize -> prefetch -> reconcile -> (one documented refetch -> reconcile2) -> publish ->
# build -> checks; after the separate Stage C authorization (authorize_run): forward -> run -> halves -> validate ->
# report. The day files come from the committed downloader (statcast_download_seasons.py fetch), run outside this
# script as §8.2 step 2 prescribes; reconcile reads its cache, and publish calls its build and converts the season file
# with statcast_combine_seasons.target_schema. Smoke: the stand-in runs this whole path on development rows (a day
# cache seeded from the subset), so no target row is read.

RUN_STAGES = ("forward", "run", "halves", "validate", "report")


def acquired(year: int) -> bool:
    """2026's path: acquisition (§8.2), forward prediction (§7), R3a/R3b, two authorizations. Smoke: the stand-in."""
    return year == 2026 or (bool(SMOKE) and year == SMOKE_STANDIN)


def raw_stage(year: int) -> str:
    """The stage that wrote the season's raw.parquet: publish (2026, smoke stand-in) or extract (2023)."""
    return "publish" if acquired(year) else "extract"


def last_reconcile(year: int) -> str:
    """The reconciliation publish follows: the reviewed one (the user's exception rule), else the one after the
    refetch, else the first. The reviewed and refetch paths exclude each other (each refuses if the other exists)."""
    for name in (f"reconcile_reviewed_{year}", f"reconcile2_{year}"):
        if record_path(name).exists():
            return name
    return f"reconcile_{year}"


def feed_block_problems(rec) -> list:
    """Why a first reconciliation record is not a block the user's exception rule can answer (empty: it is one). It
    must be finished and blocked for review by feed-vocabulary events alone: no unrecognized kept no-pitch event, no
    other completeness finding (missing game, key mismatch, score, unavailable feed, duplicate or unexpected keys),
    no integrity finding, no refetch day and no capped day."""
    if not isinstance(rec, dict) or rec.get("status") != "finished":
        return ["not a finished record"]
    r = rec.get("result") or {}
    c = r.get("completeness") or {}
    p = [] if r.get("outcome") == "blocked for review" else [f"outcome {r.get('outcome')!r}"]
    # positive proof that only the outcome failed (Astra): selftest, guard before and after, both freeze checks after
    ra = rec.get("records_after")
    if (rec.get("selftest") or {}).get("passed") is not True:
        p.append("selftest not recorded as passed")
    for k in ("guard_before", "guard_after"):
        if (rec.get(k) or {}).get("ok") is not True:
            p.append(f"{k} not recorded as ok")
    if not isinstance(ra, dict) or set(ra) != {"freeze_c1_record.py", "freeze_forward_record.py"} or \
            not all(isinstance(v, dict) and v.get("ok") is True for v in ra.values()):
        p.append("the freeze checks after the run are not recorded as ok")
    for k in ("error", "traceback", "post_guard_error", "post_records_error"):
        if k in rec:
            p.append(f"the record carries {k}")
    if not c.get("feed_stop"):
        p.append("no feed-vocabulary stop")
    if any(f.get("unrecognized_kept_no_pitch") for f in c.get("feed_stop") or []):
        p.append("an unrecognized kept no-pitch event")
    for k in ("missing_games", "key_mismatch", "score_unexplained", "unavailable", "duplicate_statcast_keys",
              "unexpected_statcast_games"):
        if c.get(k):
            p.append(f"completeness: {k}")
    for k in ("integrity", "refetch_days", "days_at_row_cap"):
        if r.get(k):
            p.append(k)
    if not r.get("snapshot") or not isinstance(r.get("snapshot_manifest"), dict):
        p.append("no snapshot recorded")
    return p


def blocked_reconcile(year: int) -> tuple:
    """(record, problems) for the season's first reconciliation as a block the exception rule can answer: bound like
    any predecessor (stage, mode, identity, predecessors) except that it did not pass."""
    rec, err = read_record(record_path(f"reconcile_{year}"))
    if err:
        return None, [err]
    bad = [x for x in record_problems(f"reconcile_{year}", rec, identity()) if x != "not a finished, passed record"]
    if rec.get("passed") is not False:
        bad.append("the record passed")
    return rec, bad + feed_block_problems(rec)


def exception_rule_active(year: int) -> bool:
    """EXCEPTION_RULE applies to a season only with a verified exception_approval record carrying this rule."""
    rec, err = read_record(record_path(f"exception_approval_{year}"))
    return not err and not record_problems(f"exception_approval_{year}", rec, identity()) and \
        (rec.get("result") or {}).get("rule") == EXCEPTION_RULE and (rec.get("result") or {}).get("season") == year


def block_resolved(year: int) -> list:
    """Problems with reading the blocked first reconciliation as answered (empty: answered): the block qualifies, the
    approval pins its digest and verifies, and reconcile_reviewed finished, passed and verifies."""
    _, bad = blocked_reconcile(year)
    ap, err = read_record(record_path(f"exception_approval_{year}"))
    if err:
        return bad + [f"exception_approval_{year}: {err}"]
    if not exception_rule_active(year):
        bad.append(f"exception_approval_{year} does not verify")
    if ((ap.get("result") or {}).get("answers") or {}).get("sha256") != sha(record_path(f"reconcile_{year}")):
        bad.append(f"exception_approval_{year} pins another reconcile_{year}")
    rv, rerr = read_record(record_path(f"reconcile_reviewed_{year}"))
    bad += [f"reconcile_reviewed_{year}: {x}" for x in ([rerr] if rerr else
                                                         record_problems(f"reconcile_reviewed_{year}", rv, identity()))]
    if not rerr and ((rv.get("result") or {}).get("answers") or {}).get("sha256") != \
            sha(record_path(f"reconcile_{year}")):
        bad.append(f"reconcile_reviewed_{year} pins another reconcile_{year}")
    return bad


def downloader():
    """The committed downloader (§8.2 steps 2 and 6), imported only by the acquisition stages."""
    import statcast_download_seasons as dl
    return dl


def game_dates(schedule: dict) -> dict:
    """game_pk -> the official date(s) it was played on: Savant files a game's pitches under its official date (a
    suspended game finished the next day keeps its original official date, e.g. 745180 in 2024), so that is the day
    to check and refetch. A postponed or cancelled entry is not a played date."""
    out = {}
    for d in schedule.get("dates", []):
        for g in d.get("games", []):
            if (g.get("status") or {}).get("detailedState") in ("Postponed", "Cancelled"):
                continue            # the game was not played that day (Neon: 36 such entries in 2024, 30 in 2025)
            out.setdefault(int(g["gamePk"]), set()).add(g.get("officialDate") or d["date"])
    return out


def step1_schedule_paths(year: int) -> tuple:
    """§8.2 step 1's recorded schedule and its query record (written when the download began)."""
    d = ACQ_DIR / f"acquisition_{year}"
    return d / f"schedule_{year}.json", d / "schedule_query.json"


def smoke_seed_cache(year: int) -> dict:
    """Smoke only: the stand-in's day files in the downloader's layout (one CSV per game_date, the reference columns),
    written from the development raw rows of the subset, with one empty day either side of the season; and a step-1
    schedule record copied from the gate's cached schedule."""
    if not SMOKE:
        raise RuntimeError("smoke_seed_cache runs only in smoke mode")
    dl = downloader()
    meta_path = ACQ_DIR / f"days_{year}.json"
    if meta_path.exists():
        return {"seeded": False}
    raw = smoke_subset(pd.read_parquet(ORIG["raw"], filters=[("game_year", "=", year)]))
    ref = dl.reference_columns()
    gd = pd.to_datetime(raw["game_date"]).dt.strftime("%Y-%m-%d")
    days = sorted(gd.unique())
    first, last = dl.shift(days[0], -1), dl.shift(days[-1], 1)
    cache = ACQ_DIR / "cache" / str(year)
    cache.mkdir(parents=True, exist_ok=True)
    meta = {}
    for day in dl.days_between(first, last):
        g = raw[gd == day].assign(game_date=day)[ref]
        g.to_csv(cache / f"{day}.csv", index=False)
        meta[day] = {"rows": len(g), "tries": 1, "games": int(g["game_pk"].nunique()) if len(g) else 0,
                     "r_rows": int((g["game_type"] == "R").sum()) if len(g) else 0, "capped": False}
    meta_path.write_text(json.dumps(meta, indent=1, sort_keys=True))
    (ACQ_DIR / f"window_{year}.json").write_text(json.dumps({"first": first, "last": last}))
    raw_s = fetch(SCHEDULE_URL.format(season=year), MLB_CACHE / "schedule" / f"{year}.json", valid_schedule)
    sp, qp = step1_schedule_paths(year)
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_bytes(raw_s)
    qp.write_text(json.dumps({"query": SCHEDULE_URL.format(season=year), "sha256": hashlib.sha256(raw_s).hexdigest(),
                              "note": "smoke: copied from the gate's cached schedule"}))
    return {"seeded": True, "days": len(meta), "rows": int(len(raw))}


def take_snapshot(year: int, attempt: int) -> tuple:
    """Copies the downloader's window, day metadata and every day file of the window, plus the step-1 schedule, into
    the season's acquisition snapshot, and returns (snapshot directory, manifest of sha256 by relative path). The
    reconciliation reads only the snapshot, and publish builds from it after verifying the manifest (Astra)."""
    import shutil
    dl = downloader()
    snap = season_dir(year) / "acquisition" / f"attempt{attempt}"
    if snap.exists():
        raise FileExistsError(f"{snap} exists")
    window = json.loads((ACQ_DIR / f"window_{year}.json").read_text())
    days = dl.days_between(window["first"], window["last"])
    (snap / "cache" / str(year)).mkdir(parents=True)
    files = [Path(f"window_{year}.json"), Path(f"days_{year}.json")] + \
        [Path("cache") / str(year) / f"{d}.csv" for d in days if (ACQ_DIR / "cache" / str(year) / f"{d}.csv").exists()]
    for rel in files:
        shutil.copy2(ACQ_DIR / rel, snap / rel)
    sp, qp = step1_schedule_paths(year)
    for p in (sp, qp):
        if p.exists():
            shutil.copy2(p, snap / p.name)
    manifest = {str(p.relative_to(snap)): sha(p) for p in sorted(snap.rglob("*")) if p.is_file()}
    return snap, manifest


def verify_snapshot(snap: Path, manifest: dict) -> list:
    have = {str(p.relative_to(snap)) for p in snap.rglob("*") if p.is_file()}
    bad = [f"missing {k}" for k in manifest if k not in have]
    bad += [f"changed {k}" for k, h in manifest.items() if k in have and sha(snap / k) != h]
    return bad


def cache_rows(year: int, src: Path) -> tuple:
    """The window's day files under `src` (the snapshot), as the downloader's build reads them: trimmed to the regular
    season's first and last dates, projected on the reference columns, identical duplicate rows dropped (counted)."""
    dl = downloader()
    window = json.loads((src / f"window_{year}.json").read_text())
    meta = json.loads((src / f"days_{year}.json").read_text())
    cache = src / "cache" / str(year)
    days = dl.days_between(window["first"], window["last"])
    info = {"window": window, "days_in_window": len(days),
            "days_not_fetched": [d for d in days if d not in meta or not (cache / f"{d}.csv").exists()]}
    if info["days_not_fetched"]:
        return info, None, meta
    frames = [df for df in (dl.parse((cache / f"{d}.csv").read_bytes()) for d in days) if not df.empty]
    if not frames:
        info["no_rows"] = True
        return info, None, meta
    df = pd.concat(frames, ignore_index=True)
    ref = dl.reference_columns()
    reg = df.loc[(df["game_type"] == "R") & (df["game_year"] == year), "game_date"].astype(str)
    if reg.empty:
        info["no_regular_season_rows"] = True
        return info, None, meta
    first_r, last_r = reg.min(), reg.max()
    gd = df["game_date"].astype(str)
    df = df[(gd >= first_r) & (gd <= last_r)]
    info.update(first_regular_season_date=first_r, last_regular_season_date=last_r,
                boundary=bool(first_r <= window["first"] or last_r >= window["last"]),
                missing_reference_columns=[c for c in ref if c not in df.columns],
                extra_columns=sorted(c for c in df.columns if c not in ref), rows_in_season_window=int(len(df)))
    proj = df[[c for c in ref if c in df.columns]]
    before = len(proj)
    proj = proj.drop_duplicates()
    info["identical_duplicate_rows_dropped"] = int(before - len(proj))
    dup = proj.duplicated(["game_pk", "at_bat_number", "pitch_number"], keep=False)
    info["conflicting_keys"] = int(proj.loc[dup, ["game_pk", "at_bat_number", "pitch_number"]].drop_duplicates().shape[0])
    info["conflicting_key_examples"] = proj.loc[dup, ["game_pk", "at_bat_number", "pitch_number"]].drop_duplicates() \
        .head(10).to_dict("records")
    return info, proj.reset_index(drop=True), meta


def key_digest(df: pd.DataFrame) -> str:
    """Order-free digest of the rows' (game_type, game_pk, at_bat_number, pitch_number), all game types."""
    k = df[["game_type", "game_pk", "at_bat_number", "pitch_number"]].astype(str).sort_values(
        ["game_pk", "at_bat_number", "pitch_number", "game_type"]).reset_index(drop=True)
    return hashlib.sha256(pd.util.hash_pandas_object(k, index=False).to_numpy().tobytes()).hexdigest()


def schedule_completed(schedule: dict) -> dict:
    c = completed_games(schedule)
    return {int(g): (None if pd.isna(r["home_score"]) else int(r["home_score"]),
                     None if pd.isna(r["away_score"]) else int(r["away_score"])) for g, r in c.iterrows()}


def acquisition_outcome(attempt: int, integrity: list, refetch: list, comp: dict, capped: list) -> str:
    """§8.2 step 5's precedence after reconciliation: data integrity > blocked for review > defer > pass. On the first
    attempt, a recoverable problem (a missing or incomplete game, an empty response, a capped day) calls for the one
    documented refetch (step 4) unless the evidence is already blocked for review; the recoverable days are recorded
    either way."""
    if integrity:
        return "stop: data integrity"
    base = classify_completeness(comp)
    if base.startswith("stop"):
        return "stop: data integrity"
    if base == "blocked for review":       # not recoverable by a refetch: the user diagnoses first (Neon)
        return "blocked for review"
    if attempt == 1 and refetch:
        return "refetch required"
    if base.startswith("incomplete") or capped:
        return "defer 2026: " + ("confirmed missing game or event" if base.startswith("incomplete") else
                                 "capped day persists after the refetch")
    return "pass"


def recoverable_days(window: dict, meta: dict, played: list, missing_or_incomplete_dates: list) -> dict:
    """§8.2 step 4's recoverable days, against the explicit window bounds (Astra): every scheduled played date of a
    missing or incomplete game inside the window, every scheduled played date inside the window whose day returned
    no rows, and every fetched day of the window at the row cap. Played dates outside the window are listed apart
    (a data-integrity stop, not a refetch)."""
    dl = downloader()
    inside = lambda d: window["first"] <= d <= window["last"]  # noqa: E731
    days = dl.days_between(window["first"], window["last"])
    capped = [d for d in days if (meta.get(d) or {}).get("capped")]
    empty = [d for d in played if inside(d) and not (meta.get(d) or {}).get("rows")]
    outside = sorted({d for d in played if not inside(d)})
    refetch = sorted({d for d in missing_or_incomplete_dates if inside(d)} | set(empty) | set(capped))
    return {"capped": capped, "empty": empty, "outside": outside, "refetch": refetch}


def reconcile_body(year: int, attempt: int, reviewed=None, answers=None) -> dict:
    """`reviewed`: the blocked first reconciliation's result (reconcile_reviewed, revision 10d). Then no new snapshot
    is taken: the first one is verified against its recorded manifest and read again, with EXCEPTION_RULE; its rows,
    rows by game type and key digest must equal the blocked record's, the verified events must be exactly the
    approval's evidence (`answers`: games, feed digests, signature counts), and a recoverable problem goes to the user
    (blocked for review), never to a refetch."""
    if reviewed is None:
        seeded = smoke_seed_cache(year) if SMOKE and attempt == 1 else None
        snap, manifest = take_snapshot(year, attempt)
    else:
        seeded, snap, manifest = None, Path(reviewed["snapshot"]), reviewed["snapshot_manifest"]
        changed = verify_snapshot(snap, manifest)
        if changed:
            return {"attempt": attempt, "reviewed": True, "snapshot": str(snap), "snapshot_manifest": manifest,
                    "integrity": [f"the first reconciliation's snapshot changed: {changed[:10]}"],
                    "outcome": "stop: data integrity", "passed": False}
    res = {"attempt": attempt, "acq_dir": str(ACQ_DIR), "smoke_seed": seeded, "snapshot": str(snap),
           "snapshot_manifest": manifest, "reviewed": reviewed is not None,
           "exception_rule": EXCEPTION_RULE if reviewed is not None else None}
    sch_bytes = fetch(SCHEDULE_URL.format(season=year), MLB_CACHE / "schedule" / f"{year}.json", valid_schedule)
    sch = json.loads(sch_bytes)
    gdates = game_dates(sch)
    checked = completeness_games(completed_games(sch))
    played = sorted({d for g in checked for d in gdates.get(int(g), ())})
    window = json.loads((snap / f"window_{year}.json").read_text())
    meta = json.loads((snap / f"days_{year}.json").read_text())
    # the step-1 schedule (recorded when the download began) against the gate's cached schedule
    sp = snap / f"schedule_{year}.json"
    q = json.loads((snap / "schedule_query.json").read_text()) if (snap / "schedule_query.json").exists() else {}
    step1_ok = sp.exists() and sha(sp) == q.get("sha256")
    diff = {}
    if step1_ok:
        a, b = schedule_completed(json.loads(sp.read_bytes())), schedule_completed(sch)
        diff = {"only_step1": sorted(set(a) - set(b))[:20], "only_gate": sorted(set(b) - set(a))[:20],
                "score_differs": sorted(g for g in set(a) & set(b) if a[g] != b[g])[:20]}
    res["schedules"] = {"step1": {"path": str(sp), "sha256": sha(sp) if sp.exists() else None,
                                  "recorded_sha256": q.get("sha256"), "query": q.get("query")},
                        "gate": {"path": str(MLB_CACHE / "schedule" / f"{year}.json"),
                                 "sha256": hashlib.sha256(sch_bytes).hexdigest()}, "completed_game_differences": diff}
    info, proj, _ = cache_rows(year, snap)
    res["cache"] = info
    if proj is None:      # a diagnostic that keeps the scheduled refetch list (Astra)
        rv = recoverable_days(window, meta, played, played)
        res.update(recoverable=rv, refetch_days=rv["refetch"], outcome="stop: days not fetched" if
                   info.get("days_not_fetched") else ("refetch required" if attempt == 1 and rv["refetch"] else
                                                      "stop: no regular-season rows"))
        if reviewed is not None:     # the blocked record had rows from this very snapshot
            res["outcome"] = "stop: data integrity"
        res["passed"] = res["outcome"] == "refetch required"
        if res["passed"]:
            res["refetch_command"] = refetch_command(year, window, rv["refetch"])
        return res
    res["published_key_digest"] = key_digest(proj)
    res["rows_by_game_type"] = {str(k): int(v) for k, v in proj["game_type"].value_counts().items()}
    r = proj[(proj["game_type"] == "R") & (proj["game_year"] == year)].reset_index(drop=True)
    r = smoke_subset(r)
    comp_s = season_completeness(year, r, gate_vocabulary(), rule=reviewed is not None)
    fed = {int(f["game_pk"]) for f in comp_s["feed_stop"]}
    gone = comp_s["missing_games"] + [m["game_pk"] for m in comp_s["key_mismatch"] if m["n_missing"] > 0]
    mi_dates = sorted({d for g in gone if int(g) not in fed for d in gdates.get(int(g), ())})
    rv = recoverable_days(window, meta, played, mi_dates)
    integrity = [x for x, bad in (("conflicting pitch keys", info["conflicting_keys"]),
                                  ("regular-season game ids not on the schedule", comp_s["unexpected_statcast_games"]),
                                  ("schedule dates outside the fetched window", rv["outside"]),
                                  ("regular season touches the fetched window", info["boundary"]),
                                  ("reference columns missing", info["missing_reference_columns"]),
                                  ("the step-1 schedule record is missing or does not match its recorded digest",
                                   not step1_ok),
                                  ("the step-1 and gate schedules disagree on completed games or scores",
                                   any(diff.values()))) if bad]
    if reviewed is not None:
        same = {"regular_season_rows": int(len(r)) == reviewed.get("regular_season_rows"),
                "rows_by_game_type": res["rows_by_game_type"] == reviewed.get("rows_by_game_type"),
                "published_key_digest": res["published_key_digest"] == reviewed.get("published_key_digest"),
                "the approved feed-stop games are the blocked record's":
                    sorted(int(x["game_pk"]) for x in (answers or {}).get("games") or []) ==
                    sorted(int(f["game_pk"]) for f in reviewed["completeness"]["feed_stop"])}
        res["same_as_blocked_record"] = same
        integrity += [f"{k} differs from the blocked record's" for k, ok in same.items() if not ok]
        res["exception_binding"] = feed_signature_problems((answers or {}).get("games"),
                                                           reviewed["completeness"]["feed_stop"]) + \
            exception_binding_problems(comp_s, answers)
        integrity += [f"exception evidence: {x}" for x in res["exception_binding"]]
    outcome = acquisition_outcome(attempt, integrity, rv["refetch"], comp_s, rv["capped"])
    if reviewed is not None and outcome == "refetch required":
        outcome = "blocked for review"       # no refetch on the reviewed path: the user decides
    res.update(completeness=comp_s, regular_season_rows=int(len(r)), recoverable=rv, days_at_row_cap=rv["capped"],
               empty_days_with_completed_games=rv["empty"], schedule_dates_outside_window=rv["outside"],
               refetch_days=rv["refetch"], integrity=integrity, outcome=outcome,
               passed=outcome in ("pass", "refetch required"))
    if outcome == "refetch required":
        res["refetch_command"] = refetch_command(year, window, rv["refetch"])
    return res


def refetch_command(year: int, window: dict, days: list) -> str:
    """The one documented refetch, into this mode's downloader tree (--out explicit: smoke never writes the real one)."""
    return (f"python3.12 statcast_download_seasons.py fetch {year} --start {window['first']} --end {window['last']} "
            f"--pad-before 0 --pad-after 0 --max-extend 0 --out {ACQ_DIR} --refetch {','.join(days)}")


def stage_reconcile(year: int):
    if not acquired(year):
        raise SystemExit(f"refused: reconcile is 2026's acquisition step (§8.2); {year} is extracted")
    season_gate(year, "reconcile")
    run_season_stage(year, "reconcile", lambda: reconcile_body(year, 1))


def stage_reconcile2(year: int):
    """After the one documented refetch (§8.2 step 4): every day the first reconciliation listed must have been
    requested again after that record was written, and the downloader's day metadata rewritten with them."""
    if not acquired(year):
        raise SystemExit(f"refused: reconcile2 is 2026's acquisition step (§8.2); {year} is extracted")
    season_gate(year, "reconcile2")
    first, err = read_record(record_path(f"reconcile_{year}"))
    if err or (first.get("result") or {}).get("outcome") != "refetch required":
        raise SystemExit("refused: reconcile2 runs only after a first reconciliation that called for the refetch")
    t1 = record_path(f"reconcile_{year}").stat().st_mtime
    cache = ACQ_DIR / "cache" / str(year)
    days = first["result"]["refetch_days"]
    meta_p = ACQ_DIR / f"days_{year}.json"
    meta = json.loads(meta_p.read_text())
    stale = [d for d in days if not (cache / f"{d}.csv").exists() or (cache / f"{d}.csv").stat().st_mtime <= t1
             or d not in meta]
    if stale or meta_p.stat().st_mtime <= t1:     # before the record is reserved (Neon)
        raise SystemExit(f"refused: the refetch is not complete (days {stale}; metadata rewritten: "
                         f"{meta_p.stat().st_mtime > t1})")

    def body():
        res = reconcile_body(year, 2)
        res["refetch"] = {"requested_days": days, "command": first["result"].get("refetch_command"),
                          "metadata_after": {d: meta[d] for d in days},
                          "files_after": {d: sha(cache / f"{d}.csv") for d in days}}
        return res
    run_season_stage(year, "reconcile2", body)


def dev_feed_replay(vocab: set) -> dict:
    """§8.2 step 5 for the changed acceptance policy (Astra): feed_check with the rule off and on over every cached
    completed development game (the gate's games and cache; the smoke subset in smoke), with the development rows.
    Passes only if each season covers the gate's checked games, all replayed, with no classification difference, no
    stop and no verified event (replay_ok)."""
    raw = pd.read_parquet(ORIG["raw"], columns=["game_pk", "game_type", "at_bat_number", "pitch_number", "description",
                                                "events", "strikes", "des", "pitch_type"])
    rows_g = {int(g): x for g, x in raw[raw["game_type"] == "R"].groupby("game_pk")}
    gate = load_record(f"gate_v{MAPPING_VERSION}")["result"]["seasons"]
    out, ok = {}, True
    for season in DEV_SEASONS:
        sch = json.loads(fetch(SCHEDULE_URL.format(season=season), MLB_CACHE / "schedule" / f"{season}.json",
                               valid_schedule))
        games = completeness_games(completed_games(sch))
        diff, unavailable, stops, verified = [], [], 0, 0
        for g in games:
            try:
                pbp = json.loads((MLB_CACHE / "pbp" / f"{int(g)}.json").read_bytes())
            except (OSError, ValueError):
                unavailable.append(int(g))
                continue
            rg = rows_g.get(int(g))
            off, on = feed_check(pbp, rg, vocab, False), feed_check(pbp, rg, vocab, True)
            if off["stop"] != on["stop"] or on["verified"]:
                diff.append(int(g))
            stops += bool(on["stop"])
            verified += len(on["verified"])
        want = (gate.get(str(season)) or {}).get("checked")
        out[str(season)] = {"games": len(games), "gate_checked": want, "replayed": len(games) - len(unavailable),
                            "not_cached": unavailable[:20], "n_not_cached": len(unavailable),
                            "classification_differences": diff, "stops_rule_on": stops, "verified_rule_on": verified}
        ok &= replay_ok(out[str(season)])
    return {"seasons": out, "ok": bool(ok)}


def replay_ok(r: dict) -> bool:
    """One development season's replay passes only if it covers exactly the frozen gate's checked games (the smoke
    subset in smoke), every one replayed, with no classification difference, no stop and no verified event (Astra)."""
    return isinstance(r.get("gate_checked"), int) and r["games"] == r["gate_checked"] == r["replayed"] \
        and not r["n_not_cached"] and not r["classification_differences"] and r["stops_rule_on"] == 0 \
        and r["verified_rule_on"] == 0


def target_fixtures(year: int, blocked: dict, vocab: set) -> dict:
    """The rule on the blocked games' real evidence (the blocked record's snapshot, verified first; the cached
    play-by-play): every event outside the vocabulary must verify, and conflicting variants of the first must fail
    (Astra): the preceding pitch claims the event's index, Statcast description 'not_swinging', strikes 2.5, the
    passed-ball record removed, the play's result changed, des without the passed ball."""
    import copy
    snap = Path(blocked["snapshot"])
    changed = verify_snapshot(snap, blocked["snapshot_manifest"])
    if changed:
        return {"ok": False, "stopped": f"the blocked record's snapshot changed: {changed[:10]}"}
    _, proj, _ = cache_rows(year, snap)
    games = [int(f["game_pk"]) for f in blocked["completeness"]["feed_stop"]]
    rows_g = {int(g): x for g, x in proj[proj["game_pk"].isin(games)].groupby("game_pk")}
    positives, negatives, first = [], [], None
    for g in games:
        pbp = json.loads((MLB_CACHE / "pbp" / f"{g}.json").read_bytes())
        for p in pbp["allPlays"]:
            for j, e in enumerate(p["playEvents"]):
                if event_signature(e) not in vocab:
                    ok, ev = exception_evidence(p, j, rows_g.get(g))
                    positives.append({"game_pk": g, "ok": ok, **ev})
                    first = first or (g, p, j)
    if first is not None:
        g, p0, j = first
        r0 = rows_g.get(g)
        ab = int(p0["about"]["atBatIndex"]) + 1

        def shift(p):     # Astra's conflict: the preceding event claims the index, the pitch gets the next one
            p["playEvents"][j - 1]["index"], p["playEvents"][j]["index"] = j, j + 1

        def statcast(col, val):
            def f(r):
                n = r.loc[r["at_bat_number"] == ab, "pitch_number"].max()
                r.loc[(r["at_bat_number"] == ab) & (r["pitch_number"] == n), col] = val
            return f
        variants = {"the preceding pitch claims the event's index": (shift, None),
                    "Statcast description 'not_swinging'": (None, statcast("description", "not_swinging")),
                    "Statcast strikes 2.5": (None, statcast("strikes", 2.5)),
                    "no passed-ball runner record": (lambda p: p.__setitem__("runners", [
                        x for x in p["runners"] if (x.get("details") or {}).get("eventType") != "passed_ball"]), None),
                    "the play's result is not a strikeout": (lambda p: p["result"].update(eventType="walk"), None),
                    "des without the passed ball": (None, statcast("des", "called out on a swinging pitchout."))}
        for name, (mp, mr) in variants.items():
            p_ = copy.deepcopy(p0)
            r_ = r0.astype({"strikes": "float64"}) if r0 is not None else None
            if mp:
                mp(p_)
            if mr and r_ is not None:
                mr(r_)
            ok, ev = exception_evidence(p_, j, r_)
            negatives.append({"variant": name, "verified": ok, "failed": ev["failed"]})
    return {"positives": positives, "negatives": negatives,
            "ok": bool(positives) and all(x["ok"] for x in positives) and len(negatives) == 6
            and not any(x["verified"] for x in negatives)}


def stage_approve_exception(year: int, date: str):
    """2026 only, and only on the user's explicit instruction: records the user's approval of EXCEPTION_RULE as the
    answer to a first reconciliation blocked for review by feed-vocabulary events alone, pinning that record's digest.
    It approves the rule, not a list: reconcile_reviewed and checks verify every event again."""
    if not acquired(year):
        raise SystemExit(f"refused: the exception approval answers 2026's acquisition (§8.2); {year} is extracted")
    if not valid_authorization_date(date):
        raise SystemExit(f"refused: approval date must be YYYY-MM-DD between {PROTOCOL_LOCKED_ON} and today")
    if record_path(f"reconcile2_{year}").exists():
        raise SystemExit(f"refused: reconcile2_{year} exists (the refetch path)")
    first, bad = blocked_reconcile(year)
    if bad:
        raise SystemExit(f"refused: reconcile_{year} is not a block the exception rule can answer: {bad}")
    h = sha(record_path(f"reconcile_{year}"))

    def body():
        c = first["result"]["completeness"]
        vocab = gate_vocabulary()
        games = [game_feed_evidence(int(f["game_pk"]), vocab) for f in c["feed_stop"]]
        replay = dev_feed_replay(vocab)
        fixtures = target_fixtures(year, first["result"], vocab)
        checks = [{"check": "development replay: rule off and on classify every cached completed game alike, and the "
                            "rule verifies nothing on development data", "ok": replay["ok"]},
                  {"check": "target fixtures: every event outside the vocabulary in the blocked games verifies; each "
                            "conflicting variant of the first fails", "ok": fixtures["ok"]},
                  {"check": "every affected game has signatures outside the vocabulary",
                   "ok": all(gm["signatures"] for gm in games)},
                  {"check": "each affected game's signatures in the current feed are the blocked record's (Astra)",
                   "ok": not feed_signature_problems(games, c["feed_stop"]),
                   "detail": feed_signature_problems(games, c["feed_stop"])}]
        return {"passed": all(x["ok"] for x in checks), "season": year, "rule": EXCEPTION_RULE, "authorized_on": date,
                "answers": {"record": f"reconcile_{year}", "sha256": h, "outcome": first["result"]["outcome"],
                            "feed_stop": c["feed_stop"], "new_signatures": c.get("new_signatures"), "games": games},
                "development_replay": replay, "target_fixtures": fixtures, "checks": checks,
                "note": "recorded on the user's explicit instruction; the rule is approved, not a list of events"}
    run_stage(f"exception_approval_{year}", body, needs=("guard", f"authorize_{year}"))


def stage_reconcile_reviewed(year: int):
    """After the user's exception approval: the first reconciliation's snapshot read again with EXCEPTION_RULE."""
    if not acquired(year):
        raise SystemExit(f"refused: reconcile_reviewed is 2026's acquisition step (§8.2); {year} is extracted")
    season_gate(year, "reconcile_reviewed")
    if record_path(f"reconcile2_{year}").exists():
        raise SystemExit(f"refused: reconcile2_{year} exists (the refetch path)")
    first, bad = blocked_reconcile(year)
    if bad:
        raise SystemExit(f"refused: reconcile_{year} is not a block the exception rule can answer: {bad}")
    ap, err = read_record(record_path(f"exception_approval_{year}"))
    if err or not exception_rule_active(year) or \
            ((ap.get("result") or {}).get("answers") or {}).get("sha256") != sha(record_path(f"reconcile_{year}")):
        raise SystemExit(f"refused: no verified exception approval pinning reconcile_{year} ({err or 'see record'}); "
                         f"it is written only on the user's instruction: s9_seasons.py approve-exception {year} <date>")

    def body():
        res = reconcile_body(year, 1, reviewed=first["result"], answers=ap["result"]["answers"])
        res["answers"] = {"record": f"reconcile_{year}", "sha256": sha(record_path(f"reconcile_{year}")),
                          "games": ap["result"]["answers"]["games"]}
        return res
    run_season_stage(year, "reconcile_reviewed", body)


def stage_publish(year: int):
    """§8.2 step 6: the downloader's build on the reconciled snapshot (its manifest verified first), then the reference
    schema (statcast_combine_seasons.target_schema) into data/seasons/<year>/raw.parquet; headers, extra and missing
    columns, cast failures and widened columns recorded; the output's rows by game type and its key digest compared
    with the reconciliation before it is accepted. Season outputs are hashed only from this file."""
    if not acquired(year):
        raise SystemExit(f"refused: publish is 2026's acquisition step (§8.2); {year} is extracted")
    season_gate(year, "publish")
    rname = last_reconcile(year)
    rec, err = read_record(record_path(rname))
    if err or (rec.get("result") or {}).get("outcome") != "pass":
        raise SystemExit(f"refused: the last reconciliation ({rname}) did not pass")

    def body():
        import pyarrow.csv as pacsv
        import statcast_combine_seasons as cs
        dl = downloader()
        rr = rec["result"]
        snap = Path(rr["snapshot"])
        bad_snap = verify_snapshot(snap, rr["snapshot_manifest"])
        if bad_snap:
            return {"passed": False, "stopped": f"the reconciled snapshot changed: {bad_snap[:10]}"}
        code = dl.stage_build(year, snap, False)
        if code != 0:
            return {"passed": False, "stopped": f"the downloader's build stopped (exit {code})"}
        summary = json.loads((snap / f"summary_{year}.json").read_text())
        src = snap / f"statcast_all_pitches_{year}.csv"
        schema, notes = cs.target_schema(snap, (year,))
        convert = pacsv.ConvertOptions(column_types={f.name: f.type for f in schema}, strings_can_be_null=True)
        table = pacsv.read_csv(src, convert_options=convert)
        names_ok = table.schema.names == schema.names
        dst = season_dir(year) / "raw.parquet"
        if dst.exists():
            raise FileExistsError(f"{dst} exists")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if names_ok:
            pq.write_table(table.cast(schema), dst, compression="zstd")
        sch, refs_ = (pq.read_schema(dst), pq.read_schema(ORIG["raw"])) if names_ok else (None, None)
        bad = ({c: (str(sch.field(c).type) if c in sch.names else None, str(refs_.field(c).type))
                for c in EXTRACT_COLUMNS if c not in sch.names or str(sch.field(c).type) != str(refs_.field(c).type)}
               if names_ok else {"columns": "differ from the reference"})
        t = pd.read_parquet(dst, columns=["game_type", "game_year", "game_pk", "at_bat_number", "pitch_number"]) \
            if names_ok else None
        n_r = int(len(smoke_subset(t[(t["game_type"] == "R") & (t["game_year"] == year)]))) if names_ok else -1
        by_type = {str(k): int(v) for k, v in t["game_type"].value_counts().items()} if names_ok else {}
        kd = key_digest(t) if names_ok else None
        checks = [{"check": "the reconciled snapshot is unchanged (manifest verified before the build)", "ok": True},
                  {"check": "the season file's columns are the reference columns, in order", "ok": names_ok},
                  {"check": "downloader header matches the reference", "ok": bool(summary.get("header_matches_reference"))},
                  {"check": "no duplicate pitch keys after the build", "ok": summary.get("duplicate_pitch_keys") == 0},
                  {"check": "no day at the row cap", "ok": summary.get("days_at_row_cap") == []},
                  {"check": "no unparseable game_date", "ok": summary.get("bad_game_dates") == 0},
                  {"check": f"{len(EXTRACT_COLUMNS)} required columns present with the development types", "ok": not bad,
                   "detail": bad or len(EXTRACT_COLUMNS)},
                  {"check": "every row has this game_year", "ok": bool(names_ok and t["game_year"].eq(year).all())},
                  {"check": "regular-season rows equal the reconciled count", "ok": n_r == rr["regular_season_rows"],
                   "detail": {"published": n_r, "reconciled": rr["regular_season_rows"]}},
                  {"check": "rows by game type equal the reconciliation's (all game types)",
                   "ok": by_type == rr["rows_by_game_type"], "detail": {"published": by_type,
                                                                        "reconciled": rr["rows_by_game_type"]}},
                  {"check": "the published key digest equals the reconciliation's (all game types)",
                   "ok": kd == rr["published_key_digest"]}]
        acq = [src, snap / f"summary_{year}.json"]
        return {"reconcile_record": rname, "snapshot": str(snap), "summary": summary, "schema_notes": notes,
                "checks": checks,
                "columns": {"extra_dropped_by_build": summary.get("extra_columns_dropped"),
                            "cast_failed": summary.get("cast_failed"),
                            "reference_empty_but_populated": summary.get("reference_empty_but_populated")},
                "acquisition_files": file_digests([p for p in acq if p.exists()]),
                "outputs": file_digests([dst]) if dst.exists() else {}, "passed": all(c["ok"] for c in checks)}
    run_season_stage(year, "publish", body)


def stage_authorize_run(year: int, date: str):
    """2026 only: the separate Stage C run authorization (§10), required by forward, run, halves, validate, report."""
    if not acquired(year):
        raise SystemExit(f"refused: the run authorization is 2026's (§10); {year} needs only its authorization")
    info = season_info(year)
    if not valid_authorization_date(date):
        raise SystemExit(f"refused: authorization date must be YYYY-MM-DD between {PROTOCOL_LOCKED_ON} and today")
    rec, err = read_record(record_path(f"authorize_{year}"))
    if err:
        raise SystemExit(f"refused: the {year} download authorization is recorded first ({err})")

    def body():
        return {"passed": True, "season": year, "authorization": info["run_authorization"], "authorized_on": date,
                "label": info["label"], "note": "recorded on the user's explicit instruction"}
    run_stage(f"authorize_run_{year}", body, needs=("guard", f"authorize_{year}"))


# ------------------------------------------------------------------------------------- forward (§7, §6 row 10)

FORWARD_STRATA = ("seen_cell", "unseen_cell", "unseen_pitcher", "unseen_batter", "action_unseen_2025", "runners_on",
                  "runners_off", "out_of_arsenal")


def criterion_baseline(train: pd.DataFrame, obs: pd.DataFrame, kind: str) -> np.ndarray:
    """§4 C7's criterion baseline fitted on `train` (validation._held_out's definition), for the query rows `obs`."""
    if kind == "outcome":
        tr, K, keys = models.outcome_training(train), models.N_OUTCOMES, ["cidx", "action"]
        base = validation._league_baseline(tr, keys, ["cidx"], K)
    else:
        tr, K, keys = models.contact_training(train), models.N_BINS, ["action"]
        n = models._counts(tr, keys, K)
        base = (n + n.sum(axis=0) / n.to_numpy().sum()).div(n.sum(axis=1) + 1, axis=0)
    bp = models._lookup(base, obs, keys, fill=np.nan)
    overall = models._counts(tr.assign(_all=0), ["_all"], K).to_numpy()[0]
    return np.where(np.isnan(bp), overall / overall.sum(), bp)


def eligible_keys(p: pd.DataFrame, year: int, kind: str) -> pd.DataFrame:
    """§7: the eligible key set and each key's observed class, from the pitch table's flags (not from a query)."""
    s = p[p["season"] == year]
    if kind == "outcome":
        s = s[s["outcome_fit_eligible"].astype(bool)]
        y = s["outcome"].map(models.OUTCOME_CODE)
    else:
        s = s[s["contact_fit_eligible"].astype(bool)]
        y = s["contact_bin"].astype(int) - 1
    return s[KEYS].assign(y=y.to_numpy().astype(int)).sort_values(KEYS).reset_index(drop=True)


def forward_strata(test: pd.DataFrame, obs: pd.DataFrame, train: pd.DataFrame, c1_model=None, kind: str = "outcome",
                   mask: str = "outcome") -> pd.DataFrame:
    """Per query key: the §7 strata. Seen cell, unseen pitcher and unseen action are defined on the training season's
    outcome-training rows (mask "outcome", the S9 definition, both components) or, for the C8b regression only, on
    the component's own training rows (mask "component", the historical definition); unseen batter on the C1 model's
    batters (False without a model)."""
    tr = models.outcome_training(train) if mask == "outcome" or kind == "outcome" else models.contact_training(train)
    pairs = set(map(tuple, tr[["pitcher", "action"]].drop_duplicates().to_numpy()))
    pitchers, actions = set(tr["pitcher"]), set(tr["action"])
    rows = test.loc[obs.index]
    seen = np.array([(a, b) in pairs for a, b in obs[["pitcher", "action"]].to_numpy()])
    batters = c1_model.features.batters if c1_model is not None else None
    return rows[KEYS].reset_index(drop=True).assign(
        action=obs["action"].to_numpy(), **({"cidx": obs["cidx"].to_numpy()} if "cidx" in obs else {}),
        seen_cell=seen, unseen_cell=~seen, unseen_pitcher=~obs["pitcher"].isin(pitchers).to_numpy(),
        unseen_batter=(~rows["batter"].isin(batters).to_numpy()) if batters is not None else np.zeros(len(obs), bool),
        action_unseen_2025=~obs["action"].isin(actions).to_numpy(),
        runners_on=rows["runners_on"].astype(bool).to_numpy(), runners_off=~rows["runners_on"].astype(bool).to_numpy(),
        out_of_arsenal=~rows["in_eligible_set"].astype(bool).to_numpy())


def forward_table(kind: str, frames: dict, strata: pd.DataFrame) -> tuple:
    """§6 row 10 for one component: log loss of P0, C1 and both baselines and calibration (10 equal-count bins per
    category), overall and in each stratum. Coverage first (Astra): the strata keys are unique and every frame covers
    exactly those keys once; P0 and C1 are then matched by research.matched (same key, fold and class) and the
    baselines joined one to one, so no row can enter a mean with a missing log. A stratum under CAL_MIN_N rows
    (empty included) gets log loss and n only, its calibration labeled "not computed (n < 10)". The criterion
    baseline's exact zeros are reported as +inf, the rows disclosed (revision 9's rule)."""
    names = list(solver.OUTCOMES) if kind == "outcome" else [f"bin_{x}" for x in payoff.CONTACT_BINS]
    if strata.duplicated(KEYS).any():
        raise ValueError(f"{kind}: strata keys are not unique")
    sk = strata[KEYS].sort_values(KEYS).reset_index(drop=True)
    for v, f in frames.items():
        fk = f[KEYS].sort_values(KEYS).reset_index(drop=True)
        if f.duplicated(KEYS).any() or not fk.equals(sk):
            raise ValueError(f"{kind}: frame {v} does not cover the strata keys exactly once")
    with np.errstate(divide="ignore"):
        mm = research.matched(frames["P0"], frames["C1"]).rename(columns={"lp_base": "P0", "lp_cand": "C1"})
        lp = {v: frames[v][KEYS + ["y"]].assign(**{v: research._logp(frames[v])}) for v in ("criterion", "league")}
    m = strata.merge(mm, on=KEYS, how="inner", validate="one_to_one")
    for v in ("criterion", "league"):
        m = m.merge(lp[v].rename(columns={"y": f"y_{v}"}), on=KEYS, how="inner", validate="one_to_one")
        if not (m[f"y_{v}"] == m["y"]).all():
            raise ValueError(f"{kind}: {v} disagrees on the observed class")
    if len(m) != len(strata) or m[["P0", "C1", "league"]].isna().any().any() or m["criterion"].isna().any():
        raise ValueError(f"{kind}: incomplete coverage after matching ({len(m)} of {len(strata)})")
    rows = []
    for name in ("all",) + FORWARD_STRATA:
        mask = np.ones(len(m), bool) if name == "all" else m[name].to_numpy(bool)
        g = m[mask]
        row = {"component": kind, "stratum": name, "n": int(mask.sum()), "share": float(mask.mean()) if len(m) else 0.0,
               "P0": np.nan, "C1": np.nan, "gain_C1_minus_P0": np.nan, "criterion_baseline": np.nan,
               "league_pooled_P0": np.nan, "criterion_zero_probability_rows": int(np.isneginf(g["criterion"]).sum())}
        if len(g):
            row.update({"P0": -g["P0"].mean(), "C1": -g["C1"].mean(), "gain_C1_minus_P0": (g["C1"] - g["P0"]).mean(),
                        "criterion_baseline": -g["criterion"].mean(), "league_pooled_P0": -g["league"].mean()})
        for v, f in frames.items():
            h = f.merge(g[KEYS], on=KEYS, how="inner")
            P, y = h.filter(regex=r"^p\d+$").to_numpy(), h["y"].to_numpy()
            k_ = f.filter(regex=r"^p\d+$").shape[1]
            cals = [calibration_or_note(P[:, j], y == j) for j in range(k_)]
            for nm, cv in zip(names, cals):
                row[f"cal_{v}_{nm}"] = cv
            num = [cv for cv in cals if not isinstance(cv, str)]
            row[f"max_cal_{v}"] = max(num) if num else f"not computed (n < {CAL_MIN_N})"
        rows.append(row)
    zero = m.loc[np.isneginf(m["criterion"].to_numpy()), KEYS + [c for c in ("cidx", "action") if c in m] + ["y"]].copy()
    zero["class"] = [names[int(y)] for y in zero["y"]]
    by = [c for c in ("cidx", "action") if c in zero] + ["y", "class"]
    return pd.DataFrame(rows), {"component": kind, "zero_probability_rows": int(len(zero)), "n": int(len(m)),
                                "breakdown": zero.groupby(by).size().rename("rows").reset_index().to_dict("records")
                                if len(zero) else [],
                                "rows": zero.head(50).to_dict("records"), "rows_listed": min(len(zero), 50)}


FWD_REG_TOL = 1e-12        # the production route against the stored 2024->2025 forward tables


def forward_regression() -> list:
    """C8b through the production route (Astra): the 2024->2025 development forward predictions stored by S8
    (P0_repro, C1_full) and P0's 2024 refit, run through eligible_keys, criterion_baseline, forward_strata and
    forward_table exactly as the 2026 stage runs them; the S9-mask table must reproduce s8_validate's forward table
    (all, seen, unseen: n, P0, C1, gain; overall max calibration) and the component-mask table must reproduce v2's
    forward_check rows (n, P0 log loss, both baselines, calibration by category; all, seen, unseen)."""
    R = {k: str(v) for k, v in STORED_REF.items()}
    dev = pd.read_parquet(ORIG["pitches"])
    p24, p25 = dev[dev["season"] == 2024], dev[dev["season"] == 2025]
    with open(R["s8"], "rb") as f:
        s8 = pickle.load(f)["prediction"]["forward"]
    with open(R["v2"], "rb") as f:
        v2 = pickle.load(f)["forward"]
    rows = []

    def close(a, b):
        if isinstance(a, str) or isinstance(b, str):
            return a == b
        return bool(np.isclose(float(a), float(b), rtol=0, atol=FWD_REG_TOL))
    for kind in ("outcome", "contact"):
        obs = models.outcome_training(p25) if kind == "outcome" else models.contact_training(p25)
        fit = models.fit_outcome_model(p24) if kind == "outcome" else models.fit_contact_model(p24)
        frames = {"P0": pd.read_parquet(Path(R["P0_dir"]) / f"forward_{kind}.parquet"),
                  "C1": pd.read_parquet(Path(R["C1_dir"]) / f"forward_{kind}.parquet")}
        frames = {v: f[KEYS + ["fold", "y"] + [c for c in f.columns if c.startswith("p") and c[1:].isdigit()]]
                  for v, f in frames.items()}
        frames["criterion"] = research._pred_frame(p25, obs, criterion_baseline(p24, obs, kind), fold=-1)
        frames["league"] = research._pred_frame(p25, obs, fit.league_mean(obs), fold=-1)
        want = eligible_keys(dev, 2025, kind)
        for v, f in frames.items():
            got = f[KEYS + ["y"]].sort_values(KEYS).reset_index(drop=True).astype({"y": int})
            rows.append({"check": f"C8b {kind} {v}: stored forward keys and classes equal the eligible set",
                         "identical": bool(got.equals(want))})
        for mask, ref in (("outcome", "s8"), ("component", "v2")):
            st = forward_strata(p25, obs, p24, None, kind, mask)
            t = forward_table(kind, frames, st)[0].set_index("stratum")
            if ref == "s8":
                r8 = s8[s8["model"] == kind].set_index("sample")
                for sname, lab in (("all", "all"), ("seen_cell", "seen cells"), ("unseen_cell", "unseen cells")):
                    a, b = t.loc[sname], r8.loc[lab]
                    ok = int(a["n"]) == int(b["n"]) and close(a["P0"], b["P0"]) and close(a["C1"], b["C1"]) and \
                        close(a["gain_C1_minus_P0"], b["gain"])
                    if sname == "all":
                        ok = ok and close(a["max_cal_P0"], b["max_cal_P0"]) and close(a["max_cal_C1"], b["max_cal_C1"])
                    rows.append({"check": f"C8b {kind} S9 mask {sname}: forward_table reproduces s8_validate",
                                 "identical": bool(ok), "detail": {"new": [a["n"], a["P0"], a["C1"]],
                                                                   "stored": [b["n"], b["P0"], b["C1"]]}})
            else:
                rv = v2[v2["model"] == kind].set_index("subset")
                cats = list(solver.OUTCOMES) if kind == "outcome" else [f"bin_{x}" for x in payoff.CONTACT_BINS]
                for sname, lab in (("all", "all"), ("seen_cell", "seen_cells"), ("unseen_cell", "unseen_cells")):
                    if lab not in rv.index:
                        continue
                    a, b = t.loc[sname], rv.loc[lab]
                    ok = int(a["n"]) == int(b["n"]) and close(a["P0"], b["log_loss"]) and \
                        close(a["criterion_baseline"], b["baseline_log_loss"]) and \
                        close(a["league_pooled_P0"], b["league_pooled_log_loss"]) and \
                        all(close(a[f"cal_P0_{c}"], b[f"cal_{c}"]) for c in cats)
                    rows.append({"check": f"C8b {kind} component mask {sname}: forward_table reproduces v2 (P0, both "
                                          f"baselines, calibration by category)", "identical": bool(ok),
                                 "detail": {"new": [a["n"], a["P0"], a["criterion_baseline"], a["league_pooled_P0"]],
                                            "stored": [b["n"], b["log_loss"], b["baseline_log_loss"],
                                                       b["league_pooled_log_loss"]]}})
    return rows


def stage_forward(year: int):
    if not acquired(year):
        raise SystemExit("refused: forward prediction is 2026's (§6 row 10); D2 leaves out 2023's")
    season_gate(year, "forward")

    def body():
        bad = verify_inputs(f"{raw_stage(year)}_{year}", f"build_{year}", f"checks_{year}")
        if bad:
            return {"passed": False, "stopped": f"inputs changed: {bad}"}
        c = run_child("season_forward", season_binding(year, "forward"))
        return {"context": c, "outputs": c.get("outputs", {}), "passed": bool(c.get("passed")) and bool(c.get("outputs"))}
    run_season_stage(year, "forward", body)


def ctx_season_forward():
    """§7 and §6 row 10: the four frozen 2025 models (hash-checked before loading) predict every outcome- and
    contact-fit-eligible pitch of the season, as research.forward selects them (no arsenal restriction); both baselines
    are fitted on the 2025 rows. Before matching, every query set must equal the key set and classes enumerated from
    the pitch table; then the §7 probability checks; then the strata table."""
    year = int(BIND["season"])
    out = Path(BIND["out"])
    res = {"probe": probe_check(BIND), "checks": []}
    fw = json.loads(FORWARD_RECORD.read_text())
    fitted = {}
    for art in fw["artifacts"]:
        blob = (ROOT / art["pickle"]["path"]).read_bytes()
        ok = hashlib.sha256(blob).hexdigest() == art["pickle"]["sha256"]
        res["checks"].append({"check": f"{art['model']} {art['component']} frozen pickle hash before loading",
                              "identical": ok})
        if ok:
            fitted[art["model"], art["component"]] = pickle.loads(blob)
    if len(fitted) != 4:
        res.update(passed=False, stopped="a frozen pickle failed its hash; nothing was predicted")
        ctx_result(res, strict=True)
        return
    dev = pd.read_parquet(ORIG["pitches"])
    train25 = dev[dev["season"] == 2025]
    pairs_rows = set(map(tuple, models.outcome_training(train25)[["pitcher", "action"]].drop_duplicates().to_numpy()))
    pairs_pkl = set(map(tuple, fitted["P0", "outcome"].cell_n.index.to_frame(index=False)[["pitcher", "action"]]
                        .drop_duplicates().to_numpy()))
    res["checks"].append({"check": "seen-cell source: 2025 outcome-training pairs equal P0/outcome.pkl's pairs",
                          "identical": pairs_rows == pairs_pkl, "detail": f"{len(pairs_rows)} / {len(pairs_pkl)}"})
    reg = forward_regression()       # the production route on the stored 2024->2025 forward tables first (Astra)
    res["checks"] += reg
    if not all(c["identical"] for c in res["checks"]):
        res.update(passed=False, stopped="the C8b regression through the production route failed; nothing predicted")
        ctx_result(res, strict=True)
        return
    specs = {"P0": research.P0, "C1": replace(research_s8.SPECS["C1"], extra_inputs=(config.RAW_PARQUET,))}
    pitches = research.base_tables()[0]
    tables, zero, frames_out, counts = [], [], {}, {}
    for kind in ("outcome", "contact"):
        frames, obs_of, test_of = {}, {}, {}
        for name, spec in specs.items():
            rows = research.prepared(spec)[0]
            test = rows[rows["season"] == year]
            test = test[research._mask(spec.eval_mask, test)]
            obs = research._training(kind, test, spec)
            frames[name] = research._pred_frame(test, obs, fitted[name, kind].predict(obs), fold=-1)
            obs_of[name], test_of[name] = obs, test
        obs = obs_of["P0"]
        frames["criterion"] = research._pred_frame(test_of["P0"], obs, criterion_baseline(train25, obs, kind), fold=-1)
        frames["league"] = research._pred_frame(test_of["P0"], obs, fitted["P0", kind].league_mean(obs), fold=-1)
        want = eligible_keys(pitches, year, kind)
        for v, f in frames.items():
            got = f[KEYS + ["y"]].sort_values(KEYS).reset_index(drop=True).astype({"y": int})
            res["checks"].append({"check": f"{kind} {v}: query keys and classes equal the pitch table's eligible set",
                                  "identical": bool(got.equals(want)), "detail": {"query": len(got), "eligible": len(want)}})
        probs = {v: prob_matrix_problems(f) for v, f in frames.items()}
        res["checks"].append({"check": f"{kind}: every probability matrix finite, in [0, 1], rows summing to 1 within "
                                       f"{PROB_SUM_TOL}", "identical": not any(probs.values()),
                              "detail": {k: v for k, v in probs.items() if v} or "all valid"})
        with np.errstate(divide="ignore"):
            fin = {v: bool(np.isfinite(research._logp(frames[v])).all()) for v in ("P0", "C1", "league")}
            pc = frames["criterion"]
            p_obs = pc.filter(regex=r"^p\d+$").to_numpy(float)[np.arange(len(pc)), pc["y"].to_numpy()]
        res["checks"].append({"check": f"{kind}: P0, C1 and league-pooled log probabilities all finite",
                              "identical": all(fin.values()), "detail": fin})
        res["checks"].append({"check": f"{kind}: criterion baseline finite except exact zero probabilities on observed "
                                       f"rows (+inf, disclosed)", "identical": bool(((p_obs > 0) | (p_obs == 0.0)).all()),
                              "detail": int((p_obs == 0.0).sum())})
        if not all(c["identical"] for c in res["checks"]):
            res.update(passed=False, stopped=f"{kind}: a query-set, probability or finiteness check failed; no table")
            ctx_result(res, strict=True)
            return
        strata = forward_strata(test_of["P0"], obs, train25, fitted["C1", kind], kind, "outcome")
        counts[kind] = {s_: int(strata[s_].sum()) for s_ in FORWARD_STRATA} | {"all": int(len(strata))}
        t, z = forward_table(kind, frames, strata)
        tables.append(t)
        zero.append(z)
        for v, f in frames.items():
            frames_out[f"{v}_{kind}"] = f
    table = pd.concat(tables, ignore_index=True)
    paths = []
    for k, f in frames_out.items():
        p = out / f"queries_{k}.parquet"
        if p.exists():
            raise FileExistsError(f"{p} exists")
        f.to_parquet(p, index=False)
        paths.append(p)
    fpath = out / "forward.pkl"
    if fpath.exists():
        raise FileExistsError(f"{fpath} exists")
    with open(fpath, "wb") as fh:
        pickle.dump({"table": table, "zero": zero, "strata_counts": counts}, fh)
    res.update(table=json_records(table), zero=zero, strata_counts=counts, outputs=file_digests([fpath, *paths]))
    res["passed"] = res["probe"]["ok"] and all(c["identical"] for c in res["checks"])
    ctx_result(res, strict=True)


def flagged_catchers(scores: pd.DataFrame, pitches_path: Path, year: int) -> pd.DataFrame:
    """§2: the catchers of the flagged clubs (documented practice or possible exposure) in the season, labeled."""
    lv = pd.read_parquet(pitches_path, columns=["season", "pa_id", "pitch_number", "dugout_level"])
    s = scores.merge(lv, on=KEYS, how="left", validate="one_to_one")
    s = s[(s["season"] == year) & s["dugout_level"].notna()]
    if not len(s):
        return pd.DataFrame(columns=["catcher", "fielding_team", "dugout_level", "pitches", "games"])
    return s.groupby(["catcher", "fielding_team", "dugout_level"]).agg(
        pitches=("game_pk", "size"), games=("game_pk", "nunique")).reset_index().sort_values(
        ["fielding_team", "pitches"], ascending=[True, False]).reset_index(drop=True)


SEASON_STAGE_FUNCS = {"extract": stage_extract, "build": stage_build, "checks": stage_checks, "run": stage_run,
                      "halves": stage_halves, "validate": stage_validate, "report": stage_report,
                      "prefetch": stage_prefetch, "reconcile": stage_reconcile, "reconcile2": stage_reconcile2,
                      "reconcile_reviewed": stage_reconcile_reviewed, "publish": stage_publish, "forward": stage_forward}
CONTEXTS.update({"season_run": ctx_season_run, "season_halves": ctx_season_halves,
                 "season_validate": ctx_season_validate, "season_forward": ctx_season_forward})


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
    out.extend(selftest_phase3())
    return out


def selftest_phase3() -> list:
    """Phase 3 (synthetic only): §7's fallbacks against values computed from the formulas (not with predict), the
    acquisition outcome's precedence, 2026's stage chain, the forward table and the eligible key set."""
    out = []

    def case(name, ok):
        out.append({"case": name, "ok": bool(ok)})
    from gamecalling import batter as bt
    from gamecalling import runners as rn
    rng = np.random.default_rng(20261122)
    K, KC = models.N_OUTCOMES, models.N_BINS

    def counts(df, keys, k):          # independent of models._counts: plain dictionary counting, unit weights
        d = {}
        for row in df[list(keys) + ["y"]].itertuples(index=False):
            key = tuple(row[:-1])
            d.setdefault(key, np.zeros(k))[int(row[-1])] += 1.0
        return d

    def post(n, m, a):                # the posterior mean, written out
        return (n + a * m) / (n.sum() + a)
    n = 900
    tr = pd.DataFrame({"pitcher": rng.choice([1, 2], n), "action": rng.choice(["FF", "SL"], n),
                       "stand": rng.choice(["L", "R"], n), "cidx": rng.choice([0, 1, 2], n), "y": rng.integers(0, K, n)})
    drop = (tr["pitcher"] == 1) & (tr["action"] == "SL") & (tr["stand"] == "R") & (tr["cidx"] == 2)
    cu = pd.DataFrame({"pitcher": 2, "action": "CU", "stand": "R", "cidx": rng.choice([0, 1, 2], 60),
                       "y": rng.integers(0, K, 60)})
    tr = pd.concat([tr[~drop], cu], ignore_index=True).assign(w=1.0, w_hyper=1.0).reset_index(drop=True)
    po = models.fit_pooled(tr, models.OUTCOME_LEVELS, K)
    c0, c1, c2 = counts(tr, ("stand", "cidx"), K), counts(tr, ("action", "stand", "cidx"), K), \
        counts(tr, ("pitcher", "action", "stand", "cidx"), K)

    def p0_out(p, a, s_, c):
        base = (c0[s_, c] + models.BASE_PSEUDO) / (c0[s_, c].sum() + K * models.BASE_PSEUDO)
        lg = post(c1.get((a, s_, c), np.zeros(K)), base, po.alpha_league)
        return base, lg, post(c2.get((p, a, s_, c), np.zeros(K)), lg, po.alpha_cell)
    q = pd.DataFrame({"pitcher": [1, 1, 99, 2, 2], "action": ["SL", "FF", "FF", "KN", "CU"],
                      "stand": ["R", "L", "L", "L", "L"], "cidx": [2, 1, 0, 1, 1]})
    pr = po.predict(q)
    e = [p0_out(*r) for r in q.itertuples(index=False)]
    case("§7 P0 outcome: seen pair with the finer cell missing gives the league mean (formula)",
         np.allclose(pr[0], e[0][1], rtol=0, atol=1e-12) and not (1, "SL", "R", 2) in c2)
    case("§7 P0 outcome: a seen cell gives the cell posterior (formula)", np.allclose(pr[1], e[1][2], rtol=0, atol=1e-12))
    case("§7 P0 outcome: an unseen pitcher gives the league mean (formula)", np.allclose(pr[2], e[2][1], rtol=0, atol=1e-12))
    case("§7 P0 outcome: an action never seen gives the base (hand x count) (formula)",
         np.allclose(pr[3], e[3][0], rtol=0, atol=1e-12))
    case("§7 P0 outcome: a seen action whose league cell is empty falls back to the base (formula; protocol note)",
         ("CU", "L", 1) not in c1 and any(k[0] == "CU" for k in c1) and np.allclose(pr[4], e[4][0], rtol=0, atol=1e-12))

    def raises(fn, *a):
        try:
            fn(*a)
            return False
        except ValueError:
            return True
    case("§7 P0 outcome: a hand or a count outside the base table raises",
         raises(po.predict, q.iloc[:1].assign(stand="S")) and raises(po.predict, q.iloc[:1].assign(cidx=11)))
    # contact: P0 and the C1 runner offset
    ct = pd.DataFrame({"pitcher": rng.choice([1, 2], n), "action": rng.choice(["FF", "SL"], n),
                       "stand": rng.choice(["L", "R"], n), "runners_on": rng.choice([0, 1], n), "y": rng.integers(0, KC, n)})
    keep = ~((ct["pitcher"] == 1) & (ct["action"] == "SL") & (ct["stand"] == "R"))       # seen pair, hand R has no BIP
    keep &= ~((ct["pitcher"] == 1) & (ct["action"] == "FF") & (ct["stand"] == "L") & (ct["runners_on"] == 1))
    cuc = pd.DataFrame({"pitcher": 2, "action": "CU", "stand": "R", "runners_on": rng.choice([0, 1], 60),
                        "y": rng.integers(0, KC, 60)})
    ct = pd.concat([ct[keep], cuc], ignore_index=True).assign(w=1.0, w_hyper=1.0).reset_index(drop=True)
    pc = models.fit_pooled(ct, models.CONTACT_LEVELS, KC)
    rm = rn.fit_runner_pooled(ct, models.CONTACT_LEVELS, KC)
    k0, k1, k2 = counts(ct, ("stand",), KC), counts(ct, ("action", "stand"), KC), counts(ct, ("pitcher", "action", "stand"), KC)
    r1, r2 = counts(ct, ("action", "stand", "runners_on"), KC), counts(ct, ("pitcher", "action", "stand", "runners_on"), KC)

    def p0_con(model, p, a, s_):
        base = (k0[(s_,)] + models.BASE_PSEUDO) / (k0[(s_,)].sum() + KC * models.BASE_PSEUDO)
        lg = post(k1.get((a, s_), np.zeros(KC)), base, model.alpha_league)
        return base, lg, post(k2.get((p, a, s_), np.zeros(KC)), lg, model.alpha_cell)

    def c1_con(p, a, s_, r):
        base, L, P = p0_con(rm.p0, p, a, s_)
        Lr = post(r1.get((a, s_, r), np.zeros(KC)), L, rm.alpha_league_r)
        m = P * Lr / L
        m = m / m.sum()
        return base, L, P, Lr, m, post(r2.get((p, a, s_, r), np.zeros(KC)), m, rm.alpha_r)
    qc = pd.DataFrame({"pitcher": [1, 99, 1, 2, 2], "action": ["SL", "FF", "FF", "KN", "CU"],
                       "stand": ["R", "L", "L", "R", "L"], "runners_on": [0, 1, 1, 0, 1]})
    pp, pr_ = pc.predict(qc), rm.predict(qc)
    ec = [p0_con(pc, p, a, s_) for p, a, s_, _ in qc.itertuples(index=False)]
    er = [c1_con(*r) for r in qc.itertuples(index=False)]
    case("§7 P0 contact: a seen pair whose hand has no 2025 balls in play gives the league mean (formula)",
         np.allclose(pp[0], ec[0][1], rtol=0, atol=1e-12))
    case("§7 P0 contact: unseen pitcher -> league mean; action never seen -> base; seen action, empty league cell -> "
         "base; a hand outside the base table raises (formula)",
         np.allclose(pp[1], ec[1][1], rtol=0, atol=1e-12) and np.allclose(pp[3], ec[3][0], rtol=0, atol=1e-12)
         and ("CU", "L") not in k1 and np.allclose(pp[4], ec[4][0], rtol=0, atol=1e-12)
         and raises(pc.predict, qc.iloc[:1].assign(stand="S")))
    case("§7 C1 contact: missing hand -> P0 cell = L, offset = normalized L_r (formula)",
         np.allclose(er[0][2], er[0][1], rtol=0, atol=1e-12) and np.allclose(pr_[0], er[0][3] / er[0][3].sum(), rtol=0,
                                                                            atol=1e-12))
    case("§7 C1 contact: unseen pitcher -> offset = normalized L_r for that runner state (formula)",
         np.allclose(pr_[1], er[1][3] / er[1][3].sum(), rtol=0, atol=1e-12))
    case("§7 C1 contact: hand present, runner state missing -> offset = normalized P0_cell x L_r / L (formula)",
         (1, "FF", "L", 1) not in r2 and np.allclose(pr_[2], er[2][4], rtol=0, atol=1e-12))
    case("§7 C1 contact: an action never seen -> L = base, L_r = L, offset = base (formula)",
         np.allclose(pr_[3], er[3][0], rtol=0, atol=1e-12))
    case("§7 C1 contact: a seen action whose league cell is empty -> L_r = L = base, offset = base (formula)",
         ("CU", "L", 1) not in r1 and np.allclose(er[4][3], er[4][0], rtol=0, atol=1e-12)
         and np.allclose(pr_[4], er[4][0], rtol=0, atol=1e-12))
    # C1 outcome: the P0 offset, then the batter adjustment; an unseen batter gets the standardized league values
    F = bt.Features(batters=pd.Index([1, 2]), totals=np.ones((2, len(bt.STAT_COLUMNS))),
                    game_index=pd.MultiIndex.from_tuples([(1, 1)]), game_stats=np.zeros((1, len(bt.STAT_COLUMNS))),
                    whiff=bt.BetaBinomial(0.25, 50.0), chase=bt.BetaBinomial(0.3, 40.0),
                    xw=bt.NormalNormal(0.37, 0.01, 0.002), mean=np.array([0.2, 0.25, 0.35]), sd=np.array([0.05, 0.06, 0.04]))
    z_ind = np.array([(0.25 - 0.2) / 0.05, (0.3 - 0.25) / 0.06, (0.37 - 0.35) / 0.04])
    case("§7 C1: an unseen batter gets the standardized league feature values (nonzero; formula)",
         np.allclose(F.z_full([999])[0], z_ind, rtol=0, atol=1e-12) and np.abs(z_ind).min() > 0)
    theta = rng.normal(0, 0.1, (3, K - 1))
    bm = bt.BatterModel("M3a", "outcome", po, F, theta, 1.0, {}, 0, 0, 0, [])
    qb = q.iloc[:3].assign(batter=[999, 999, 999])
    eta = np.log(np.vstack([e[i][2] for i in range(3)])) + np.tile(z_ind, (3, 1)) @ theta @ bt.helmert(K).T
    sm = np.exp(eta - eta.max(axis=1, keepdims=True))
    case("§7 C1 outcome: the P0 offset (league mean for an unseen pitcher), then the batter adjustment",
         np.allclose(bm.predict(qb), sm / sm.sum(axis=1, keepdims=True), rtol=0, atol=1e-12))
    Fg = bt.Features(batters=pd.Index([1, 2]), totals=np.ones((2, len(bt.STAT_COLUMNS) + 1)),
                     game_index=pd.MultiIndex.from_tuples([(1, 1)]), game_stats=np.zeros((1, len(bt.STAT_COLUMNS) + 1)),
                     whiff=bt.BetaBinomial(0.25, 50.0), chase=bt.BetaBinomial(0.3, 40.0),
                     xw=bt.NormalNormal(0.37, 0.01, 0.002), mean=np.array([0.2, 0.25, 0.35, 0.4]),
                     sd=np.array([0.05, 0.06, 0.04, 0.08]), gb=bt.BetaBinomial(0.44, 30.0))
    zg = np.append(z_ind, (0.44 - 0.4) / 0.08)
    thg = rng.normal(0, 0.1, (4, KC - 1))
    bc = bt.BatterModel("C1", "contact", rm, Fg, thg, 1.0, {}, 0, 0, 0, [])
    qcb = qc.assign(batter=999)
    etag = np.log(np.vstack([er[i][5] for i in range(len(qc))])) + np.tile(zg, (len(qc), 1)) @ thg @ bt.helmert(KC).T
    smg = np.exp(etag - etag.max(axis=1, keepdims=True))
    case("§7 C1 contact: runner offset (formula) then the batter adjustment with the ground-ball feature; unseen batter",
         np.allclose(Fg.z_full([999])[0], zg, rtol=0, atol=1e-12)
         and np.allclose(bc.predict(qcb), smg / smg.sum(axis=1, keepdims=True), rtol=0, atol=1e-12))
    # acquisition precedence (§8.2 step 5)
    ok = {"duplicate_statcast_keys": 0, "unexpected_statcast_games": [], "feed_stop": [], "score_unexplained": [],
          "unavailable": [], "missing_games": [], "key_mismatch": []}
    case("§8.2: integrity stops first; the first attempt asks for the refetch; then blocked > defer > pass",
         acquisition_outcome(1, ["conflicting pitch keys"], ["2026-04-01"], ok, []) == "stop: data integrity"
         and acquisition_outcome(1, [], ["2026-04-01"], {**ok, "missing_games": [5]}, []) == "refetch required"
         and acquisition_outcome(2, [], [], {**ok, "missing_games": [5]}, []).startswith("defer")
         and acquisition_outcome(2, [], [], {**ok, "missing_games": [5], "feed_stop": [1]}, []) == "blocked for review"
         and acquisition_outcome(2, [], [], ok, ["2026-05-01"]).startswith("defer")
         and acquisition_outcome(1, [], [], ok, []) == "pass" and acquisition_outcome(2, [], [], ok, []) == "pass"
         and acquisition_outcome(2, [], [], {**ok, "unexpected_statcast_games": [9]}, []) == "stop: data integrity"
         and acquisition_outcome(1, [], ["2026-04-01"], {**ok, "feed_stop": [{"game_pk": 5}],
                                                         "missing_games": [6]}, []) == "blocked for review")
    # EXCEPTION_RULE (revision 10d): a play modeled on 824606's (a swinging pitchout for strike three, passed ball)
    import copy

    def pitch(code, call, k, i, strike=True, out=False):
        return {"type": "pitch", "isPitch": True, "index": i, "count": {"strikes": k},
                "details": {"code": code, "call": {"code": code, "description": call}, "isStrike": strike, "isOut": out}}
    xplay = {"about": {"atBatIndex": 13},
             "playEvents": [{"type": "pickoff", "isPitch": False, "index": 0, "details": {"code": "1"}},
                            pitch("B", "Ball", 0, 1, strike=False), pitch("S", "Swinging Strike", 1, 2),
                            pitch("F", "Foul", 2, 3), pitch("Q", "Swinging Pitchout", 3, 4, out=True)],
             "result": {"eventType": "strikeout", "description": "X called out on a swinging pitchout. Y to 2nd. Passed "
                                                                "ball by catcher Z."},
             "runners": [{"details": {"eventType": "strikeout", "playIndex": 4}},
                         {"details": {"eventType": "passed_ball", "playIndex": 4}}]}
    xrows = pd.DataFrame({"game_pk": 5, "at_bat_number": 14, "pitch_number": [1, 2, 3, 4],
                          "description": ["ball", "swinging_strike", "foul", "swinging_pitchout"],
                          "events": [None, None, None, "strikeout"], "strikes": [0, 0, 1, 2],
                          "des": [None, None, None, "X called out on a swinging pitchout. Y to 2nd. Passed ball by "
                                                  "catcher Z."]})
    xpbp = {"allPlays": [xplay]}
    xvocab = set(vocabulary(xpbp)) - {event_signature(xplay["playEvents"][4])}
    okx, evx = exception_evidence(xplay, 4, xrows)
    case("rev 10d rule: the 824606 pattern verifies in both sources (key: plate appearance 14, pitch 4)",
         okx and evx["at_bat_number"] == 14 and evx["pitch_number"] == 4)

    def fails(mut_play=None, mut_rows=None):
        p_, r_ = copy.deepcopy(xplay), xrows.copy()
        if mut_play:
            mut_play(p_)
        if mut_rows:
            mut_rows(r_)
        return not exception_evidence(p_, 4, r_)[0]
    case("rev 10d rule: each condition is required (no passed ball; passed ball on another pitch; called, not "
         "swinging; not strike three; not a strikeout; no 'Passed ball by catcher'; Statcast events, strikes, "
         "description, des; a duplicate or missing Statcast row; not the last pitch)",
         fails(lambda p: p["runners"].pop(1))
         and fails(lambda p: p["runners"][1]["details"].update(playIndex=3))
         and fails(lambda p: p["playEvents"][4]["details"]["call"].update(description="Called Strike"))
         and fails(lambda p: p["playEvents"][4]["count"].update(strikes=2))
         and fails(lambda p: p["result"].update(eventType="strikeout_double_play"))
         and fails(lambda p: p["result"].update(description="X called out on a swinging pitchout."))
         and fails(mut_rows=lambda r: r.__setitem__("events", [None, None, None, "field_out"]))
         and fails(mut_rows=lambda r: r.__setitem__("strikes", [0, 0, 1, 1]))
         and fails(mut_rows=lambda r: r.__setitem__("description", ["ball", "swinging_strike", "foul", "pitchout"]))
         and fails(mut_rows=lambda r: r.__setitem__("des", [None] * 4))
         and not exception_evidence(xplay, 4, pd.concat([xrows, xrows.iloc[[3]]]))[0]
         and not exception_evidence(xplay, 4, xrows.iloc[:3])[0]
         and not exception_evidence(xplay, 4, None)[0]
         and fails(lambda p: p["playEvents"].append(pitch("B", "Ball", 3, 5, strike=False))))

    def shift(p):
        p["playEvents"][3]["index"], p["playEvents"][4]["index"] = 4, 5
    case("rev 10d (Astra): conflicting evidence fails closed: the preceding pitch claims the index (passed ball still "
         "at 4); no explicit index; a duplicated index; Statcast 'not_swinging'; strikes 2.5 or '2'; MLB strikes 3.5; "
         "the evidence keeps both indices and the values read",
         fails(shift) and fails(lambda p: p["playEvents"][4].pop("index"))
         and fails(lambda p: p["playEvents"][3].update(index=4))
         and fails(mut_rows=lambda r: r.__setitem__("description", ["ball", "swinging_strike", "foul", "not_swinging"]))
         and fails(mut_rows=lambda r: r.__setitem__("strikes", [0.0, 0.0, 1.0, 2.5]))
         and fails(mut_rows=lambda r: r.__setitem__("strikes", ["0", "0", "1", "2"]))
         and fails(lambda p: p["playEvents"][4]["count"].update(strikes=3.5))
         and evx["mlb_event_index"] == 4 and evx["list_position"] == 4 and evx["passed_ball_play_indices"] == [4]
         and evx["statcast_strikes"] == 2 and evx["statcast_description"] == "swinging_pitchout"
         and not exception_evidence(xplay, 4, xrows.assign(strikes=[0.0, 0.0, 1.0, 2.0]))[1]["failed"])
    fc_off, fc_on = feed_check(xpbp, xrows, xvocab, False), feed_check(xpbp, xrows, xvocab, True)
    bad_pbp = copy.deepcopy(xpbp)
    bad_pbp["allPlays"].append({"about": {"atBatIndex": 14}, "playEvents": [pitch("Q", "Swinging Pitchout", 1, 0)],
                                "result": {"eventType": "single", "description": ""}, "runners": []})
    unrec_pbp = copy.deepcopy(xpbp)
    unrec_pbp["allPlays"][0]["playEvents"].insert(0, {"type": "no_pitch", "isPitch": False,
                                                      "details": {"code": "VZ", "call": {"code": "VZ"},
                                                                  "description": "Something new"}})
    case("rev 10d feed_check: without the rule a new signature stops; with it a verified event clears, any failed "
         "event of the same signature stops the game, an unrecognized kept no-pitch event stops, an in-vocabulary "
         "game is untouched",
         fc_off["stop"] and not fc_on["stop"] and len(fc_on["verified"]) == 1
         and feed_check(bad_pbp, xrows, xvocab, True)["stop"]
         and feed_check(unrec_pbp, xrows, xvocab | {event_signature(unrec_pbp["allPlays"][0]["playEvents"][0])},
                        True)["stop"]
         and not feed_check(xpbp, xrows, set(vocabulary(xpbp)), True)["stop"])
    blocked = {"status": "finished", "passed": False, "selftest": {"passed": True}, "guard_before": {"ok": True},
               "guard_after": {"ok": True}, "records_after": {"freeze_c1_record.py": {"ok": True},
                                                              "freeze_forward_record.py": {"ok": True}}, "result": {
        "outcome": "blocked for review", "integrity": [], "refetch_days": [], "days_at_row_cap": [],
        "snapshot": "/x", "snapshot_manifest": {"a": "b"},
        "completeness": {**ok, "feed_stop": [{"game_pk": 5, "signatures_outside_vocabulary": [["pitch"]],
                                              "unrecognized_kept_no_pitch": []}]}}}

    def fb(**kw):
        b = copy.deepcopy(blocked)
        for k, v in kw.items():
            (b["result"]["completeness"] if k in ok else b["result"])[k] = v
        return feed_block_problems(b)
    case("rev 10d: only a block by feed-vocabulary events alone can be answered by the rule (not a missing game, key "
         "mismatch, score, unavailable feed, unrecognized no-pitch, integrity, refetch or capped day, other outcome)",
         fb() == [] and fb(missing_games=[6]) and fb(key_mismatch=[{"n_missing": 1}]) and fb(score_unexplained=[1])
         and fb(unavailable=[1]) and fb(integrity=["x"]) and fb(refetch_days=["2026-04-01"])
         and fb(days_at_row_cap=["2026-04-01"]) and fb(outcome="pass") and fb(feed_stop=[])
         and fb(feed_stop=[{"game_pk": 5, "unrecognized_kept_no_pitch": [["no_pitch"]]}])
         and feed_block_problems({**blocked, "status": "reserved"}))
    case("rev 10d (Astra): the block needs positive proof that only the outcome failed (selftest, guard before and "
         "after, both freeze checks after, no error)",
         feed_block_problems({**blocked, "guard_after": {"ok": False}})
         and feed_block_problems({**blocked, "guard_before": {}})
         and feed_block_problems({**blocked, "records_after": {"freeze_c1_record.py": {"ok": True}}})
         and feed_block_problems({**blocked, "records_after": {"freeze_c1_record.py": {"ok": True},
                                                               "freeze_forward_record.py": {"ok": False}}})
         and feed_block_problems({**blocked, "error": "ValueError: x"})
         and feed_block_problems({**blocked, "post_records_error": "x"})
         and feed_block_problems({**blocked, "selftest": {"passed": False}}))
    import tempfile
    global MLB_CACHE, RECORDS
    keep = (MLB_CACHE, RECORDS)
    try:
        with tempfile.TemporaryDirectory() as td:
            MLB_CACHE = RECORDS = Path(td)
            (Path(td) / "pbp").mkdir()
            pf = Path(td) / "pbp" / "5.json"
            pf.write_text(json.dumps(xpbp))
            gev = game_feed_evidence(5, xvocab)
            comp_ok = {"verified_exceptions": [{"game_pk": 5, **evx}]}
            bind_ok = exception_binding_problems(comp_ok, {"games": [gev]})
            bind_missing = exception_binding_problems({"verified_exceptions": []}, {"games": [gev]})
            bind_extra = exception_binding_problems({"verified_exceptions": comp_ok["verified_exceptions"] * 2},
                                                    {"games": [gev]})
            bind_other = exception_binding_problems({"verified_exceptions": [{**comp_ok["verified_exceptions"][0],
                                                                              "game_pk": 6}]}, {"games": [gev]})
            bind_none = exception_binding_problems(comp_ok, {})
            # Astra's Q -> Q2 regression: the feed changed after the block, baseball evidence intact
            q2 = copy.deepcopy(xpbp)
            q2["allPlays"][0]["playEvents"][4]["details"]["code"] = "Q2"
            q2["allPlays"][0]["playEvents"][4]["details"]["call"]["code"] = "Q2"
            (Path(td) / "pbp" / "7.json").write_text(json.dumps(q2))
            stop_q = [{"game_pk": 7, "signatures_outside_vocabulary": [list(event_signature(
                xplay["playEvents"][4]))], "unrecognized_kept_no_pitch": []}]
            q2_vocab = set(vocabulary(q2)) - {event_signature(q2["allPlays"][0]["playEvents"][4])}
            gev_q2 = game_feed_evidence(7, q2_vocab)
            sig_q2 = feed_signature_problems([gev_q2], stop_q)
            sig_q = feed_signature_problems([{**gev_q2, "signatures": {signature_key(event_signature(
                xplay["playEvents"][4])): 1}}], stop_q)
            sig_other = feed_signature_problems([{**gev, "game_pk": 8}], stop_q)
            sig_none = feed_signature_problems([gev_q2], [])
            q2_rule_ok = exception_evidence(q2["allPlays"][0], 4, xrows)[0]
            pf.write_text(json.dumps(xpbp) + " ")
            bind_changed = exception_binding_problems(comp_ok, {"games": [gev]})
            # the answer link (a pinned failed record): checked by chain_problems and record_problems
            rp = lambda n: Path(td) / f"s9_{n}.json"  # noqa: E731
            rp("reconcile_2026").write_text(json.dumps(blocked))
            ans = {"record": "reconcile_2026", "sha256": sha(rp("reconcile_2026"))}
            rp("exception_approval_2026").write_text(json.dumps({"stage": "exception_approval_2026",
                                                                 "predecessors": {}, "result": {"answers": ans}}))
            chain_ok = chain_problems("exception_approval_2026", rp)
            ans_ok = answer_problems("exception_approval_2026", {"result": {"answers": ans}})
            ans_other = answer_problems("exception_approval_2026", {"result": {"answers": {**ans, "sha256": "0"}}})
            ans_unnamed = answer_problems("reconcile_reviewed_2026", {"result": {}})
            rp("reconcile_2026").unlink()
            chain_gone = chain_problems("exception_approval_2026", rp)
            ans_gone = answer_problems("exception_approval_2026", {"result": {"answers": ans}})
    finally:
        MLB_CACHE, RECORDS = keep
    case("rev 10d (Astra): verified events must be exactly the approval's evidence: same games, unchanged cached feed, "
         "same signature counts (a missing, extra, other-game or changed-feed event is a problem; no evidence fails)",
         bind_ok == [] and gev["signatures"] == {signature_key(event_signature(xplay["playEvents"][4])): 1}
         and bind_missing and bind_extra and bind_other and bind_none and bind_changed)
    case("rev 10d (Astra, Q->Q2): the approval's per-game signatures from the current feed must be the blocked "
         "record's: a feed changed to Q2 with the baseball evidence intact is refused although the rule verifies it; "
         "the same signature passes; another game or no blocked game is refused",
         q2_rule_ok and gev_q2["signatures"] == {signature_key(event_signature(q2["allPlays"][0]["playEvents"][4])): 1}
         and sig_q2 and sig_q == [] and sig_other and sig_none)
    rclean = {"games": 1, "gate_checked": 1, "replayed": 1, "n_not_cached": 0, "classification_differences": [],
              "stops_rule_on": 0, "verified_rule_on": 0}
    case("rev 10d (Astra): the development replay passes only with zero stops (equal stops off and on included), the "
         "gate's checked-game count, every game replayed, no difference and nothing verified",
         replay_ok(rclean) and not replay_ok({**rclean, "stops_rule_on": 1})
         and not replay_ok({**rclean, "gate_checked": 2}) and not replay_ok({**rclean, "gate_checked": None})
         and not replay_ok({**rclean, "replayed": 0, "n_not_cached": 1})
         and not replay_ok({**rclean, "classification_differences": [5]}) and not replay_ok({**rclean, "verified_rule_on": 1}))
    case("rev 10d (Astra): the answered failed record is a required digest link: chain_problems and answer_problems "
         "flag it missing or at another digest; only the answer stages carry it",
         chain_ok == [] and chain_gone and ans_gone and ans_other and ans_unnamed
         and not any("is missing or not at the pinned digest" in x for x in ans_ok)
         and any("is missing or not at the pinned digest" in x for x in ans_gone + ans_other)
         and answer_problems("publish_2026", {"result": {}}) == [])
    braw = pd.DataFrame({"game_type": "R", "game_pk": [5, 5, 6], "at_bat_number": [14, 14, 1], "pitch_number": [3, 4, 1],
                         "description": ["foul", "swinging_pitchout", "ball"], "events": [None, "strikeout", None],
                         "bb_type": None, "stand": "R", "p_throws": "R", "inning_topbot": "Top", "pitch_type": "FF"})
    bdom = {"description": ["ball", "foul"], "bb_type": list(batter.BB_TYPES), "stand": ["L", "R"],
            "p_throws": ["L", "R"], "inning_topbot": ["Top", "Bot"], "events_dev": ["strikeout"], "events_code": []}
    b_no, b_yes = b0_inventory(braw, bdom), b0_inventory(braw, bdom, frozenset({(5, 14, 4)}))
    b_other = b0_inventory(braw.assign(description=["foul", "swinging_pitchout", "swinging_pitchout"]), bdom,
                           frozenset({(5, 14, 4)}))
    case("rev 10d B0: an out-of-domain description stops unless at a rule-verified key, where it is listed; the same "
         "label at any other key still stops",
         not b_no["ok"] and b_yes["ok"] and b_yes["description_exceptions"] ==
         [{"game_pk": 5, "at_bat_number": 14, "pitch_number": 4, "description": "swinging_pitchout"}]
         and not b_other["ok"] and b_other["domain_violations"] == {"description": {"swinging_pitchout": 1}})
    case("rev 10d: the reviewed and approval stages chain to the exception approval; publish follows the last "
         "reconciliation",
         expected_predecessors("reconcile_reviewed_2026") == {"guard", "audit", "control", f"gate_v{MAPPING_VERSION}",
                                                              "domains", "authorize_2026", "exception_approval_2026"}
         and expected_predecessors("exception_approval_2026") == {"guard", "authorize_2026"}
         and "reconcile_reviewed" in SEASON_STAGE_FUNCS)
    fin, pst = {"detailedState": "Final"}, {"detailedState": "Postponed"}
    sch = {"dates": [{"date": "2026-04-01", "games": [{"gamePk": 7, "status": fin}, {"gamePk": 8, "status": fin}]},
                     {"date": "2026-04-02", "games": [{"gamePk": 7, "status": fin}, {"gamePk": 9, "status": pst}]},
                     {"date": "2026-04-03", "games": [{"gamePk": 9, "status": fin}]}]}
    case("§8.2: a suspended game maps to every date it was played; a postponed entry is not a played date (Neon)",
         game_dates(sch) == {7: {"2026-04-01", "2026-04-02"}, 8: {"2026-04-01"}, 9: {"2026-04-03"}})
    sus = {"dates": [{"date": "2026-05-21", "games": [{"gamePk": 10, "status": fin, "officialDate": "2026-05-21"}]},
                     {"date": "2026-05-22", "games": [{"gamePk": 10, "status": fin, "officialDate": "2026-05-21"}]}]}
    case("§8.2 (rev 10c): a suspended game finished the next day maps to its official date only (Savant's day)",
         game_dates(sus) == {10: {"2026-05-21"}})
    win = {"first": "2026-04-01", "last": "2026-04-05"}
    meta_ = {"2026-04-01": {"rows": 10}, "2026-04-02": {"rows": 0}, "2026-04-03": {"rows": 25000, "capped": True},
             "2026-04-04": {"rows": 5}, "2026-04-05": {"rows": 0}}
    rv = recoverable_days(win, meta_, ["2026-04-01", "2026-04-02", "2026-04-04", "2026-04-07"], ["2026-04-04", "2026-04-09"])
    case("§8.2 (rev 10c, Astra): recoverable days against the explicit window: empty played days, every capped day of "
         "the window, missing-game dates inside it; played dates outside it listed apart",
         rv["refetch"] == ["2026-04-02", "2026-04-03", "2026-04-04"] and rv["outside"] == ["2026-04-07"]
         and rv["capped"] == ["2026-04-03"] and rv["empty"] == ["2026-04-02"])
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d_ = Path(td)
        (d_ / "a.csv").write_text("x\n1\n")
        man = {"a.csv": sha(d_ / "a.csv")}
        ok_snap = verify_snapshot(d_, man) == []
        (d_ / "a.csv").write_text("x\n2\n")
        case("§8.2 (rev 10c, Astra): a changed snapshot file is caught before the build",
             ok_snap and verify_snapshot(d_, man) == ["changed a.csv"]
             and "missing b.csv" in verify_snapshot(d_, {**man, "b.csv": "0"}))
    if not SMOKE:
        fp, bp_ = expected_predecessors("forward_2026"), expected_predecessors("build_2026")
        case("phase 3 chain: 2026 publishes (not extracts); its run stages name the Stage C authorization",
             "authorize_run_2026" in fp and "checks_2026" in fp and "publish_2026" in bp_
             and expected_predecessors("build_2023") == {"guard", "authorize_2023", "extract_2023"}
             and "authorize_run_2026" in expected_predecessors("report_2026") and raw_stage(2026) == "publish"
             and raw_stage(2023) == "extract" and acquired(2026) and not acquired(2023) and not acquired(2024))
    # forward table: +inf for an exact criterion zero; strata under 10 rows get log loss and n only
    keys = pd.DataFrame({"season": 2026, "pa_id": np.arange(12), "pitch_number": 1})
    base_f = keys.assign(fold=-1, y=np.arange(12) % 2, p0=0.4, p1=0.6)
    crit = base_f.copy()
    crit.loc[0, ["p0", "p1"]] = [0.0, 1.0]
    st = keys.assign(**{s_: False for s_ in FORWARD_STRATA})
    st.loc[:2, "unseen_pitcher"] = True
    tab, z = forward_table("contact", {"P0": base_f, "C1": base_f.copy(), "criterion": crit, "league": base_f.copy()}, st)
    allrow, up = tab[tab["stratum"] == "all"].iloc[0], tab[tab["stratum"] == "unseen_pitcher"].iloc[0]
    emp = tab[tab["stratum"] == "seen_cell"].iloc[0]
    case("§6 row 10: an exact criterion zero gives +inf and is disclosed; a stratum under 10 (empty included) gets no "
         "calibration, labeled",
         math.isinf(allrow["criterion_baseline"]) and z["zero_probability_rows"] == 1 and up["n"] == 3
         and str(up["max_cal_P0"]).startswith("not computed") and not isinstance(allrow["max_cal_P0"], str)
         and int(emp["n"]) == 0 and str(emp["max_cal_criterion"]).startswith("not computed") and math.isnan(emp["P0"])
         and z["breakdown"] == [{"y": 0, "class": f"bin_{payoff.CONTACT_BINS[0]}", "rows": 1}])
    stray = pd.concat([st, st.iloc[:1].assign(pa_id=999)], ignore_index=True)
    try:
        forward_table("contact", {"P0": base_f, "C1": base_f.copy(), "criterion": crit, "league": base_f.copy()}, stray)
        stray_refused = False
    except ValueError:
        stray_refused = True
    case("§6 row 10 (rev 10c, Astra): a stratum key no frame covers is refused, not silently skipped", stray_refused)
    pt = pd.DataFrame({"season": 2026, "pa_id": [1, 2, 3], "pitch_number": 1, "outcome_fit_eligible": [True, False, True],
                       "contact_fit_eligible": [True, False, False], "outcome": ["ball", "ball", "in_play"],
                       "contact_bin": [3, 1, 1], "in_eligible_set": [True, True, False]})   # 3: out of arsenal, kept
    eo, ec_ = eligible_keys(pt, 2026, "outcome"), eligible_keys(pt, 2026, "contact")
    case("§7: the eligible key set and classes come from the pitch table's flags",
         list(eo["pa_id"]) == [1, 3] and list(eo["y"]) == [models.OUTCOME_CODE["ball"], models.OUTCOME_CODE["in_play"]]
         and list(ec_["pa_id"]) == [1] and list(ec_["y"]) == [2])
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
    fgood = {**good, "out": str(SEASONS_ROOT / "2023" / "forward"), "models_dir": str(SEASONS_ROOT / "2023" / "forward" / "models"),
             "stage": "forward_2023"}
    case("binding (rev 10, Neon): season_forward binds to its own forward stage; under run it is refused",
         not any(x.startswith("binding stage") or x.startswith("context") for x in validate_binding(fgood, "season_forward"))
         and validate_binding(good, "season_forward")[0].startswith("context"))
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
        case("season gating (rev 10): development seasons are refused; 2023 extracts, 2026 publishes, with two authorizations",
             refuses(season_info, 2024) and refuses(season_info, 2025) and season_info(2023)["source"] == RAW_2015_2023
             and season_info(2026)["source"] == SEASONS_ROOT / "2026" / "raw.parquet"
             and season_info(2026)["run_authorization"] == "Stage C (A5)" and season_info(2026)["authorization"] == "A11-4")
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
    rep_ = {"eligible": [10, 11, 12], "effects": {("D", "random"): eff, ("A", "random"): eff},
            "table": pd.DataFrame({"score": ["D", "A"], "model": ["random", "random"], "pearson": [0.1, 0.2]})}
    mv_ = pd.DataFrame({"group": ["mover", "non_mover", "non_mover"]}, index=[10, 11, 12])
    sg = small_group_listing(rep_, mv_)
    chg = lambda c_: float(sg[(sg["catcher"] == c_) & (sg["score"] == "D")]["change_centered"].iloc[0])  # noqa: E731
    case("pairs (rev 10): groups under 10 list every catcher, the change centered on the listed group's means (Astra)",
         len(sg) == 6 and set(sg["group"]) == {"mover", "non_mover"}
         and chg(10) == 0.0 and chg(11) == 0.5 and chg(12) == -0.5)
    gw = group_weighted_pearson(rep_, mv_)
    case("pairs (rev 10): the weighted Pearson by mover group lists groups under 10",
         len(gw) == 6 and all(str(v).startswith("listed") for v in gw["weighted_pearson"]))
    big = pd.DataFrame({"y0": np.arange(12.0), "y1": np.arange(12.0)[::-1], "w": 1.0}, index=range(100, 112))
    rz = {"eligible": list(big.index), "effects": {("D", "random"): big, ("A", "random"): big},
          "table": pd.DataFrame({"score": ["D", "A"], "model": ["random", "random"], "pearson": [vp.UNDEFINED, -1.0]})}
    gz = group_weighted_pearson(rz, pd.DataFrame({"group": "mover"}, index=big.index))
    gzm = gz[gz["group"] == "mover"].set_index("score")["weighted_pearson"]
    case("pairs (rev 10c, Astra): a zero-variance fit leaves the group's weighted Pearson undefined even at n >= 10",
         gzm["D"] == vp.UNDEFINED and abs(float(gzm["A"]) + 1.0) < 1e-12)
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
            y = stage.removeprefix("reconcile_")
            if bad and y.isdigit() and r.get("status") == "finished" and r.get("passed") is False \
                    and record_path(f"reconcile_reviewed_{y}").exists():
                left = block_resolved(int(y))      # the block the user answered (EXCEPTION_RULE): not a problem
                if not left:
                    print(f"{p.name}: status=finished passed=False blocked for review; answered by "
                          f"exception_approval_{y} and reconcile_reviewed_{y}")
                    continue
                bad = bad + [f"block not answered: {left}"]
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
    elif len(sys.argv) == 4 and sys.argv[1] == "authorize-run" and sys.argv[2].isdigit():
        stage_authorize_run(int(sys.argv[2]), sys.argv[3])
    elif len(sys.argv) == 4 and sys.argv[1] == "approve-exception" and sys.argv[2].isdigit():
        stage_approve_exception(int(sys.argv[2]), sys.argv[3])
    else:
        sys.exit(f"usage: python3.12 s9_seasons.py {{{'|'.join(STAGES)}}}\n"
                 f"       python3.12 s9_seasons.py {{{'|'.join(SEASON_STAGE_FUNCS)}}} <year>\n"
                 f"       python3.12 s9_seasons.py authorize <year> <YYYY-MM-DD>   (only on the user's instruction)\n"
                 f"       python3.12 s9_seasons.py authorize-run 2026 <YYYY-MM-DD>   (Stage C; only on the user's "
                 f"instruction)\n"
                 f"       python3.12 s9_seasons.py approve-exception 2026 <YYYY-MM-DD>   (EXCEPTION_RULE; only on the "
                 f"user's instruction)")
