"""S9 forward models: the all-2025 fits of P0 and C1.

Specification: research_notes/research_program/S9_forward_models_2025.md (revision 2). Outside gamecalling/ on purpose,
so the package's code hash stays 1000e503014a. Script revision 2 (2026-11-18): stage gating corrected after Astra's
review (empty smoke setting refused; predecessor records checked for stage, mode, script and self-test; the guard
compared before each stage; the fit manifest verified against the fit record before use; recheck exits nonzero on any
failure; stage records reserved atomically; commit IDs recorded; the self-test runs inside every writing stage).

    python3.12 freeze_forward_2025.py selftest   # comparator and check functions on synthetic data; reads and writes nothing
    python3.12 freeze_forward_2025.py guard      # once: record the protected inventory (C1_frozen.json and its artifacts)
    python3.12 freeze_forward_2025.py control    # once: all-2024 refit through this script vs. the stored forward outputs
    python3.12 freeze_forward_2025.py fit        # once, only after a passing control: the all-2025 fits, pickled
    python3.12 freeze_forward_2025.py reload     # once, in a fresh process: reload the pickles, compare bitwise
    python3.12 freeze_forward_2025.py compare    # once: the summary record
    python3.12 freeze_forward_2025.py recheck    # read-only, repeatable: guard, pickles and query files vs. records

S9_FWD_SMOKE=<dir> runs every stage on one game in five, entirely under <dir> (records in <dir>, models in
<dir>/forward_2025_smoke). In smoke mode the control's reference is the package's own research.forward on the same
subset, so the comparison is exact there too. Smoke results mean nothing for decisions.

Refused in every mode, before any project module is imported: S8_SMOKE (it changes C1's input transformation when
research_s8 is imported), any other environment variable whose name contains SMOKE except S9_FWD_SMOKE, and an
S9_FWD_SMOKE that is set but empty (an unset shell variable must not select production).

Every writing stage: preconditions -> every earlier stage record checked (passed, finished, same stage name, same mode
and smoke directory, same script, self-test passed, hash chain intact) -> the self-test, in process -> the guard
compared -> the stage record reserved atomically (O_EXCL) -> the stage -> the guard compared again, even on failure ->
the record finalized atomically (os.replace). A reservation left by an interrupted run stays for diagnosis and blocks
the stage.
"""

import os
import sys

_FOREIGN_SMOKE = sorted(k for k in os.environ if "SMOKE" in k and k != "S9_FWD_SMOKE")
if _FOREIGN_SMOKE:
    sys.exit(f"refused: {_FOREIGN_SMOKE} set. S8_SMOKE changes C1's input transformation; no other project smoke "
             f"setting may be inherited by this script.")
if "S9_FWD_SMOKE" in os.environ and not os.environ["S9_FWD_SMOKE"].strip():
    sys.exit("refused: S9_FWD_SMOKE is set but empty. Unset it for production, or give the smoke directory.")

from gamecalling import research_s6  # noqa: E402,I001 (first: sets the thread limit before numpy loads)

import hashlib  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gamecalling import config, models, payoff, research, research_s8, solver  # noqa: E402
import freeze_c1_record as frozen  # noqa: E402

ROOT = Path(__file__).resolve().parent
SMOKE = os.environ.get("S9_FWD_SMOKE")
MODE = "smoke" if SMOKE else "production"
SMOKE_DIR = str(Path(SMOKE).resolve()) if SMOKE else None
STORED = config.DATA_DIR / "models"
OUT = Path(SMOKE) / "forward_2025_smoke" if SMOKE else STORED / "forward_2025"
RECORDS = Path(SMOKE) if SMOKE else config.DATA_DIR / "research"
ORDER = ("guard", "control", "fit", "reload", "compare")
REC = {s: RECORDS / f"s9_fwd_{s}.json" for s in ORDER}
CONTROL_SEASON, FIT_SEASON = config.SEASONS          # (2024, 2025): train 2024 -> predict 2025; train 2025
QUERY_SEASON = FIT_SEASON
VERSIONS = {"P0": research.P0, "C1": research_s8.SPECS["C1"]}
REF_FORWARD = {"P0": STORED / "P0_repro", "C1": STORED / "C1_full"}   # stored 2024 -> 2025 forward predictions
REF_INPUTS = {"P0": STORED / "P0_repro" / "manifest.json", "C1": STORED / "C1_full" / "manifest.json"}
GATE_MANIFEST = STORED / "C1" / "manifest.json"                        # C1's forward_alphas (80 fields)
KINDS = ("outcome", "contact")
CATEGORIES = {"outcome": [str(o) for o in solver.OUTCOMES], "contact": [int(b) for b in payoff.CONTACT_BINS]}
KEYS = list(research.KEYS)
SUM_TOL = 1e-12
WORKERS = 2                                          # one job per version; within the 8-worker standing limit


def sha(path) -> str:
    return frozen.sha(path)


def file_rec(path) -> dict:
    """Like freeze_c1_record.file_rec, but a path outside the project (smoke mode) is kept absolute."""
    path = Path(path).resolve()
    shown = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    return {"path": shown, "bytes": path.stat().st_size, "sha256": sha(path)}


def script_sha() -> str:
    return sha(Path(__file__))


def json_bytes(obj) -> bytes:
    return json.dumps(obj, indent=1, default=str).encode()


def dump(path: Path, obj):
    path.write_bytes(json_bytes(obj))


def load_json(path: Path):
    return json.loads(Path(path).read_text())


def load_record(path: Path):
    """The record and the sha256 of the exact bytes parsed."""
    data = Path(path).read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


# --- Checks (pure functions; exercised by selftest) ---

def env_problems(env: dict, smoke: bool) -> list:
    """Smoke settings that make a stage refuse. Production refuses its own smoke variable too."""
    bad = sorted(k for k in env if "SMOKE" in k and k != "S9_FWD_SMOKE")
    if "S9_FWD_SMOKE" in env:
        if not env["S9_FWD_SMOKE"].strip():
            bad.append("S9_FWD_SMOKE (set but empty)")
        elif not smoke:
            bad.append("S9_FWD_SMOKE")
    return bad


def record_problems(r, stage: str, script: str, mode: str, smoke_dir) -> list:
    """Why a stored stage record cannot serve as a passed predecessor here (empty: it can)."""
    if r is None:
        return [f"{stage}: no record"]
    prov = r.get("provenance") or {}
    p = []
    if r.get("stage") != stage:
        p.append(f"{stage}: the record names stage {r.get('stage')!r}")
    if r.get("status") != "finished":
        p.append(f"{stage}: status {r.get('status')!r} (a reservation: interrupted, or still running)")
    if r.get("passed") is not True:
        p.append(f"{stage}: did not pass")
    if prov.get("mode") != mode or prov.get("smoke") != smoke_dir:
        p.append(f"{stage}: ran in {prov.get('mode')!r} mode ({prov.get('smoke')}), not {mode!r} ({smoke_dir})")
    if prov.get("script_sha256") != script:
        p.append(f"{stage}: ran under a different script ({str(prov.get('script_sha256'))[:12]})")
    if (r.get("selftest") or {}).get("passed") is not True:
        p.append(f"{stage}: no passing self-test recorded")
    return p


def chain_problems(recs: dict, shas: dict) -> list:
    """Each record's saved predecessor hashes must name exactly the earlier stages and match their current bytes."""
    p = []
    for s, r in recs.items():
        want = ORDER[:ORDER.index(s)]
        got = r.get("predecessors")
        if not isinstance(got, dict) or sorted(got) != sorted(want):
            p.append(f"{s}: predecessors recorded {sorted(got) if isinstance(got, dict) else got}, expected "
                     f"{list(want)}")
            continue
        for q in want:
            if q not in shas:
                p.append(f"{s}: its predecessor {q} has no record now")
            elif got[q] != shas[q]:
                p.append(f"{s}: recorded {q} as {got[q][:12]}, now {shas[q][:12]}")
    return p


def manifest_from_bytes(data: bytes, expected_sha: str) -> dict:
    """The manifest, parsed from the same bytes whose hash matched the fit record."""
    got = hashlib.sha256(data).hexdigest()
    if got != expected_sha:
        raise RuntimeError(f"the fit manifest hashes to {got[:12]}, not the fit record's {str(expected_sha)[:12]}")
    return json.loads(data)


def probability_check(frame: pd.DataFrame, kind: str) -> dict:
    """(c): finite, within [0, 1], rows sum to 1 within 1e-12, one column per category."""
    cols = [c for c in frame.columns if c.startswith("p") and c[1:].isdigit()]
    p = frame[cols].to_numpy(dtype=float)
    finite = bool(np.isfinite(p).all())
    in_range = bool(((p >= 0) & (p <= 1)).all()) if finite else False
    worst = float(np.abs(p.sum(axis=1) - 1).max()) if finite and len(p) else (0.0 if finite else float("nan"))
    ok_cols = len(cols) == len(CATEGORIES[kind])
    ok = finite and in_range and worst <= SUM_TOL and ok_cols
    return {"ok": ok, "rows": len(p), "columns": len(cols), "expected_columns": len(CATEGORIES[kind]),
            "finite": finite, "in_unit_interval": in_range, "max_row_sum_error": worst}


def frames_compare(new: pd.DataFrame, ref: pd.DataFrame, keys: list, bitwise: bool, same_order: bool) -> dict:
    """Exact comparison. Columns, row counts and key uniqueness first. same_order: keys must already be in the same
    order (no sorting). bitwise: dtypes must match and every column's bytes must be identical (so -0.0 vs 0.0 and NaN
    payloads differ); otherwise values must be equal (NaN equal to NaN), dtypes reported."""
    if list(new.columns) != list(ref.columns):
        return {"identical": False, "detail": f"columns differ: {list(new.columns)} vs. {list(ref.columns)}"}
    if len(new) != len(ref):
        return {"identical": False, "detail": f"row counts differ: {len(new)} vs. {len(ref)}"}
    for name, d in (("new", new), ("reference", ref)):
        dup = int(d.duplicated(keys).sum())
        if dup:
            return {"identical": False, "detail": f"{dup} duplicate keys in the {name} table"}
    order = bool(new[keys].reset_index(drop=True).equals(ref[keys].reset_index(drop=True)))
    if same_order and not order:
        return {"identical": False, "detail": "keys are not in the same order"}
    a = new.reset_index(drop=True) if order else new.sort_values(keys, kind="stable").reset_index(drop=True)
    b = ref.reset_index(drop=True) if order else ref.sort_values(keys, kind="stable").reset_index(drop=True)
    dtypes = [c for c in b.columns if a[c].dtype != b[c].dtype]
    if bitwise and dtypes:
        return {"identical": False, "detail": f"dtypes differ: {dtypes}"}
    if bitwise:
        diff = []
        for c in b.columns:
            x, y = a[c].to_numpy(), b[c].to_numpy()
            if x.dtype == object:
                same = bool(a[c].equals(b[c]))
            else:
                same = np.ascontiguousarray(x).tobytes() == np.ascontiguousarray(y).tobytes()
            if not same:
                diff.append(c)
    else:
        diff = [c for c in b.columns if not a[c].equals(b[c])]
    out = {"identical": not diff, "rows": len(b), "same_row_order": order, "dtypes_differ": dtypes,
           "detail": "identical" if not diff else f"differ: {', '.join(diff)}"}
    num = [c for c in diff if pd.api.types.is_float_dtype(b[c])]
    if num:
        out["largest_difference"] = float(max(np.nanmax(np.abs(a[c].to_numpy(float) - b[c].to_numpy(float)))
                                              for c in num))
    return out


def _norm(obj):
    """JSON-normalized form, as the manifests were written (json.dumps(default=str))."""
    return json.loads(json.dumps(obj, default=str))


def diagnostics_compare(new: dict, ref: dict) -> dict:
    """Every field, exactly, after JSON normalization; missing or extra fields count."""
    a, b = _norm(new), _norm(ref)
    missing, extra = sorted(set(b) - set(a)), sorted(set(a) - set(b))
    differ = sorted(k for k in set(a) & set(b) if a[k] != b[k])
    ok = not (missing or extra or differ)
    return {"identical": ok, "fields": len(b), "missing": missing, "extra": extra, "differ": differ[:20],
            "detail": "identical" if ok else f"{len(missing)} missing, {len(extra)} extra, {len(differ)} differ"}


def query_hash(rows: pd.DataFrame, obs: pd.DataFrame) -> str:
    """The query inputs and their keys: column names, dtypes and row-wise hashes (pandas' fixed hash key)."""
    h = hashlib.sha256()
    keys = rows.loc[obs.index, KEYS].reset_index(drop=True)
    for frame in (keys, obs.reset_index(drop=True)):
        h.update(json.dumps([[c, str(t)] for c, t in frame.dtypes.items()]).encode())
        h.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return h.hexdigest()


# --- Data and fitting (the stored forward path, season as a parameter) ---

def data(spec):
    pitches, _, _ = research.prepared(spec)
    return pitches[pitches["game_pk"] % 5 == 0].copy() if SMOKE else pitches


def train_rows(pitches, spec, season):
    p = pitches[pitches["season"] == season]
    return p[research._mask(spec.train_mask, p)]


def query_rows(pitches, spec, season):
    p = pitches[pitches["season"] == season]
    return p[research._mask(spec.eval_mask, p)]


def predict(model, rows, obs):
    return research._pred_frame(rows, obs, model.predict(obs), fold=-1)


def forward_frames(models_, train, rows, spec) -> dict:
    """research.forward's output frames, for models fitted on `train`, predicting `rows`."""
    seen = set(map(tuple, models.outcome_training(train)[["pitcher", "action"]].drop_duplicates().to_numpy()))
    out = {}
    for kind, model in zip(KINDS, models_):
        obs = research._training(kind, rows, spec)
        f = predict(model, rows, obs)
        f["seen_cell"] = [(a, b) in seen for a, b in obs[["pitcher", "action"]].to_numpy()]
        out[kind] = f
    return out


def diagnostics(models_) -> dict:
    """As run_predictions writes forward_alphas: outcome first, then contact."""
    return {f"{kind}_{k}": v for kind, m in zip(KINDS, models_) for k, v in research.model_alphas(m).items()}


def query_frame(name, kind, model, rows, spec):
    obs = research._training(kind, rows, spec)
    f = predict(model, rows, obs)
    f.insert(0, "component", kind)
    f.insert(0, "model", name)
    return f, query_hash(rows, obs)


def job(args) -> dict:
    """One version in a worker: the frozen spec.fit_models(train) call, then the stage's predictions."""
    name, mode = args
    threads = research_s6.verify_threads()
    spec = VERSIONS[name]
    pitches = data(spec)
    season = CONTROL_SEASON if mode == "control" else FIT_SEASON
    train = train_rows(pitches, spec, season)
    t = time.perf_counter()
    fitted = spec.fit_models(train)
    seconds = time.perf_counter() - t
    rows = query_rows(pitches, spec, QUERY_SEASON)
    res = {"name": name, "mode": mode, "threads": threads, "fit_seconds": seconds, "train_rows": len(train),
           "train_games": int(train["game_pk"].nunique()), "diagnostics": _norm(diagnostics(fitted))}
    if mode == "control":
        res["frames"] = forward_frames(fitted, train, rows, spec)
        if SMOKE:   # the package's own forward path on the same subset is the reference
            ref_models = {}
            res["ref_frames"] = research.forward(pitches, spec, ref_models)
            res["ref_diagnostics"] = _norm(diagnostics((ref_models["outcome"], ref_models["contact"])))
        return res
    d = OUT / name
    d.mkdir()
    res["files"], res["queries"], res["checks"] = {}, {}, {}
    for kind, model in zip(KINDS, fitted):
        pkl = d / f"{kind}.pkl"
        with open(pkl, "wb") as fh:
            pickle.dump(model, fh, protocol=5)
        f, qh = query_frame(name, kind, model, rows, spec)
        qpath = d / f"queries_{QUERY_SEASON}_{kind}.parquet"
        f.to_parquet(qpath, index=False)
        res["files"][kind] = {"pickle": file_rec(pkl), "queries": file_rec(qpath),
                              "model_class": type(model).__name__}
        res["queries"][kind] = {"rows": len(f), "query_sha256": qh}
        res["checks"][kind] = probability_check(f, kind)
    return res


# --- Provenance, preconditions, guard ---

def provenance() -> dict:
    import pyarrow
    import scipy
    from threadpoolctl import threadpool_info
    pools = [{k: d.get(k) for k in ("internal_api", "version", "num_threads")} for d in threadpool_info()]
    git = lambda *a: frozen.git(*a).stdout.strip()  # noqa: E731
    src = load_json(frozen.RECORD)["source"]
    return {"script_revision": 2, "mode": MODE, "argv": sys.argv, "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__, "pickle_protocol": 5, "threadpools": pools,
            "thread_env": {k: os.environ.get(k) for k in research_s6.THREAD_VARS}, "workers": WORKERS,
            "smoke": SMOKE_DIR, "git_head": git("rev-parse", "HEAD"), "code": research.code_state(),
            "reproduction_commit": src["reproduction_commit"], "freeze_source_commit": src["freeze_source_commit"],
            "script_sha256": script_sha(), "c1_frozen_sha256": sha(frozen.RECORD)}


def verify_frozen() -> dict:
    r = subprocess.run([sys.executable, str(ROOT / "freeze_c1_record.py"), "verify"], cwd=ROOT, capture_output=True,
                       text=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
    return {"passed": r.returncode == 0, "output": (r.stdout + r.stderr).strip().splitlines()[-3:]}


def preconditions(dests=()) -> dict:
    bad = env_problems(dict(os.environ), bool(SMOKE))
    if bad:
        raise RuntimeError(f"refused: {bad} set")
    if SMOKE and Path(SMOKE).resolve().is_relative_to(ROOT):
        raise RuntimeError("the smoke directory must be outside the project (use the scratchpad)")
    frozen.check_source(load_json(config.DATA_DIR / "research" / "c1_repro_run.json"))
    for name, spec in VERSIONS.items():
        if research.input_state(spec.extra_inputs) != load_json(REF_INPUTS[name])["inputs"]:
            raise RuntimeError(f"{name}: an input file differs from {REF_INPUTS[name].parent.name}'s manifest")
    v = verify_frozen()
    if not v["passed"]:
        raise RuntimeError(f"freeze_c1_record.py verify failed: {v['output']}")
    for d in dests:
        if Path(d).exists():
            raise FileExistsError(f"{d} exists: this stage writes once")
    return v


def guard_check(g: dict, g_sha: str, v=None) -> dict:
    """C1_frozen.json verifies, is the record the guard saw, and the guard record itself is unchanged."""
    v = v or verify_frozen()
    same = sha(frozen.RECORD) == g.get("c1_frozen_sha256")
    guard_same = REC["guard"].exists() and sha(REC["guard"]) == g_sha
    return {"passed": v["passed"] and same and guard_same, "verify": v, "record_unchanged": same,
            "guard_record_unchanged": guard_same}


def require_before(stage: str):
    """Every earlier stage: present, finished, passed, same stage name, mode, smoke directory and script, self-test
    passed; and the hash chain between them intact. Returns the records and the hashes of their bytes."""
    recs, shas, problems = {}, {}, []
    for s in ORDER[:ORDER.index(stage)]:
        if not REC[s].exists():
            raise RuntimeError(f"the {s} stage has not passed: nothing after it runs")
        recs[s], shas[s] = load_record(REC[s])
        problems += record_problems(recs[s], s, script_sha(), MODE, SMOKE_DIR)
    problems += chain_problems(recs, shas)
    if problems:
        raise RuntimeError("an earlier stage record cannot be used: " + "; ".join(problems))
    return recs, shas


def selftest_results() -> list:
    return _selftest()


def reserve(path: Path, rec: dict):
    """Atomic: exactly one run can create the record. A second run, or a rerun, gets FileExistsError."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "wb") as fh:
        fh.write(json_bytes(rec))
        fh.flush()
        os.fsync(fh.fileno())


def finalize(path: Path, rec: dict):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        fh.write(json_bytes(rec))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def run_stage(stage: str, body, dests=(), pre=None):
    """Refusals before the reservation consume nothing. After it, the record is always finalized (failed or interrupted
    runs included); only a killed process leaves the reservation, which then blocks the stage for diagnosis."""
    v = preconditions([REC[stage], *dests])
    recs, shas = require_before(stage)
    tests = selftest_results()
    failed = [label for label, ok in tests if not ok]
    if failed:
        raise RuntimeError(f"the self-test failed: {failed}")
    before = None
    if stage != "guard":
        before = guard_check(recs["guard"], shas["guard"], v)
        if not before["passed"]:
            raise RuntimeError(f"the guard check failed before the stage: {before}")
    if pre:
        pre(recs)
    RECORDS.mkdir(parents=True, exist_ok=True)
    rec = {"stage": stage, "status": "reserved", "passed": False, "pid": os.getpid(),
           "started": time.strftime("%Y-%m-%d %H:%M:%S %z"), "provenance": provenance(),
           "selftest": {"passed": True, "checks": len(tests)}, "predecessors": shas, "guard_before": before}
    try:
        reserve(REC[stage], rec)
    except FileExistsError:
        raise RuntimeError(f"{REC[stage]} exists: another run reserved this stage, or it already ran") from None
    t = time.perf_counter()
    try:
        body(rec, recs)
    except BaseException as e:
        rec["passed"] = False
        rec["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        rec["seconds"] = time.perf_counter() - t
        if stage != "guard":
            try:
                rec["guard_after"] = guard_check(recs["guard"], shas["guard"])
            except BaseException as e:
                rec["guard_after"] = {"passed": False, "error": f"{type(e).__name__}: {e}"}
            if not rec["guard_after"]["passed"]:
                rec["passed"] = False
        rec["status"] = "finished"
        rec["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
        finalize(REC[stage], rec)
        print(f"{stage}: {'PASSED' if rec['passed'] else 'FAILED'} in {rec['seconds']:.0f} s; record {REC[stage]}",
              flush=True)
    if not rec["passed"]:
        raise SystemExit(f"{stage} failed: stop and diagnose")


def verified_manifest(fit_rec: dict) -> dict:
    return manifest_from_bytes((OUT / "manifest.json").read_bytes(), fit_rec["manifest"]["sha256"])


# --- Stages ---

def stage_guard():
    def body(rec, recs):
        rec["c1_frozen_sha256"] = sha(frozen.RECORD)
        rec["verify"] = verify_frozen()
        rec["inventory"] = ("every artifact and archived source listed in C1_frozen.json (including the gate manifest "
                            "and the P0_repro and C1_full forward files), plus the record itself")
        rec["passed"] = rec["verify"]["passed"] and sha(frozen.RECORD) == rec["c1_frozen_sha256"]

    run_stage("guard", body)


def stage_control():
    def body(rec, recs):
        with ProcessPoolExecutor(WORKERS) as ex:
            results = list(ex.map(job, [(n, "control") for n in VERSIONS]))
        rows = []
        for r in results:
            name = r["name"]
            for kind in KINDS:
                ref = (r["ref_frames"][kind] if SMOKE
                       else pd.read_parquet(REF_FORWARD[name] / f"forward_{kind}.parquet"))
                cmp_ = frames_compare(r["frames"][kind], ref, KEYS, bitwise=False, same_order=False)
                bits = frames_compare(r["frames"][kind], ref, KEYS, bitwise=True, same_order=False)
                rows.append({"check": f"{name} {kind}: 2024 -> 2025 predictions vs. "
                             f"{'research.forward (smoke)' if SMOKE else REF_FORWARD[name].name}",
                             **cmp_, "bitwise_identical": bits["identical"]})
            if SMOKE:
                ref_d, label = r["ref_diagnostics"], "research.forward's models (smoke)"
            elif name == "C1":
                ref_d, label = load_json(GATE_MANIFEST)["forward_alphas"], "gate manifest forward_alphas"
            else:
                ref_d = None
            if ref_d is not None:
                rows.append({"check": f"{name}: forward diagnostics vs. {label}",
                             **diagnostics_compare(r["diagnostics"], ref_d)})
        rec["checks"] = rows
        rec["jobs"] = [{k: v for k, v in r.items() if k not in ("frames", "ref_frames", "ref_diagnostics")}
                       for r in results]
        rec["passed"] = all(x["identical"] for x in rows)
        print(pd.DataFrame(rows)[["check", "identical", "detail"]].to_string(), flush=True)

    run_stage("control", body)


def stage_fit():
    def body(rec, recs):
        OUT.mkdir(parents=True)
        with ProcessPoolExecutor(WORKERS) as ex:
            results = list(ex.map(job, [(n, "fit") for n in VERSIONS]))
        manifest = {"stage": "S9 forward models, all-2025 fits", "spec": "S9_forward_models_2025.md revision 2",
                    "created": time.strftime("%Y-%m-%d %H:%M:%S %z"), "provenance": rec["provenance"],
                    "fit_season": FIT_SEASON, "query_season": QUERY_SEASON, "categories": CATEGORIES,
                    "inputs": {n: research.input_state(s.extra_inputs) for n, s in VERSIONS.items()},
                    "guard_record_sha256": rec["predecessors"]["guard"],
                    "control_record_sha256": rec["predecessors"]["control"], "models": {}}
        for r in results:
            manifest["models"][r["name"]] = {k: r[k] for k in ("train_rows", "train_games", "fit_seconds", "threads",
                                                              "diagnostics", "files", "queries", "checks")}
        dump(OUT / "manifest.json", manifest)
        rec["manifest"] = file_rec(OUT / "manifest.json")
        rec["checks"] = [{"check": f"{n} {k}: probabilities", **manifest["models"][n]["checks"][k]}
                         for n in VERSIONS for k in KINDS]
        rec["abnormal"] = {n: {k: manifest["models"][n]["diagnostics"].get(f"{k}_abnormal_log") for k in KINDS}
                           for n in VERSIONS}
        rec["passed"] = all(c["ok"] for c in rec["checks"])
        print(pd.DataFrame(rec["checks"])[["check", "ok", "rows", "max_row_sum_error"]].to_string(), flush=True)

    run_stage("fit", body, dests=[OUT])


def stage_reload():
    def body(rec, recs):
        man = verified_manifest(recs["fit"])      # parsed from the bytes that matched the fit record
        rec["manifest_sha256"] = recs["fit"]["manifest"]["sha256"]
        rows = []
        for name, spec in VERSIONS.items():
            pitches = data(spec)
            qrows = query_rows(pitches, spec, QUERY_SEASON)
            for kind in KINDS:
                files = man["models"][name]["files"][kind]
                blob = (OUT / name / f"{kind}.pkl").read_bytes()
                if hashlib.sha256(blob).hexdigest() != files["pickle"]["sha256"]:
                    rows.append({"check": f"{name} {kind}: pickle hash", "identical": False,
                                 "detail": "changed: not loaded"})
                    continue
                model = pickle.loads(blob)                 # the bytes just hashed, not a second read
                f, qh = query_frame(name, kind, model, qrows, spec)
                want = man["models"][name]["queries"][kind]["query_sha256"]
                rows.append({"check": f"{name} {kind}: query inputs hash", "identical": qh == want,
                             "detail": "identical" if qh == want else "differs"})
                stored = pd.read_parquet(OUT / name / f"queries_{QUERY_SEASON}_{kind}.parquet")
                rows.append({"check": f"{name} {kind}: reloaded predictions vs. in-memory (bitwise, same order)",
                             **frames_compare(f, stored, KEYS, bitwise=True, same_order=True)})
                pc = probability_check(f, kind)
                rows.append({"check": f"{name} {kind}: probabilities after reload", "identical": pc["ok"],
                             "detail": json.dumps(pc)})
        rec["checks"] = rows
        rec["passed"] = all(x["identical"] for x in rows)
        print(pd.DataFrame(rows)[["check", "identical", "detail"]].to_string(), flush=True)

    run_stage("reload", body, pre=lambda recs: verified_manifest(recs["fit"]))


def stage_compare():
    def body(rec, recs):
        man = verified_manifest(recs["fit"])
        rec["manifest_sha256"] = recs["fit"]["manifest"]["sha256"]
        rec["stages"] = {s: {"passed": r["passed"], "sha256": rec["predecessors"][s]} for s, r in recs.items()}
        rec["files"] = recheck_files(man)
        rec["passed"] = all(v["passed"] for v in rec["stages"].values()) and all(f["ok"] for f in rec["files"])
        print(json.dumps({"stages": rec["stages"], "files_ok": all(f["ok"] for f in rec["files"])}, indent=1),
              flush=True)

    run_stage("compare", body, pre=lambda recs: verified_manifest(recs["fit"]))


def recheck_files(man: dict) -> list:
    out = []
    for name in VERSIONS:
        for kind in KINDS:
            for what in ("pickle", "queries"):
                rec = man["models"][name]["files"][kind][what]
                p = OUT / name / Path(rec["path"]).name
                out.append({"file": str(p), "ok": p.exists() and sha(p) == rec["sha256"]})
    return out


def stage_recheck():
    """Read-only and repeatable. Exits nonzero if any applicable check fails."""
    problems = []
    present, shas = {}, {}
    for s in ORDER:
        if REC[s].exists():
            present[s], shas[s] = load_record(REC[s])
    if "guard" in present:
        g = guard_check(present["guard"], shas["guard"])
        print(f"guard: {'passed' if g['passed'] else 'FAILED'} (record unchanged: {g['record_unchanged']}; guard "
              f"record unchanged: {g['guard_record_unchanged']}) {g['verify']['output']}", flush=True)
        if not g["passed"]:
            problems.append("the guard check failed")
    else:
        problems.append("no guard record")
    for i, s in enumerate(ORDER):
        if s not in present:
            continue
        problems += [f"{s} exists but {q} does not" for q in ORDER[:i] if q not in present]
        rp = record_problems(present[s], s, script_sha(), MODE, SMOKE_DIR)
        print(f"{s}: passed={present[s].get('passed')} status={present[s].get('status')} "
              f"sha256={shas[s][:12]}{'' if not rp else ' PROBLEMS'}", flush=True)
        problems += rp
    problems += chain_problems(present, shas)
    if (OUT / "manifest.json").exists() or "fit" in present:
        try:
            if "fit" not in present:
                raise RuntimeError("a fit manifest exists without a fit record")
            files = recheck_files(verified_manifest(present["fit"]))
            n_ok = sum(f["ok"] for f in files)
            print(f"stored fit files: {n_ok} of {len(files)} match the verified manifest", flush=True)
            problems += [f"{f['file']}: missing or changed" for f in files if not f["ok"]]
        except (RuntimeError, OSError, KeyError, ValueError) as e:
            problems.append(f"fit files not checkable: {e}")
    if problems:
        for p in problems:
            print(f"FAIL {p}", flush=True)
        sys.exit(f"recheck failed: {len(problems)} problem(s)")
    print("recheck passed", flush=True)


# --- Self-test ---

def _selftest() -> list:
    """Synthetic cases for every pure check; reads and writes nothing. Returns (label, ok) pairs."""
    rng = np.random.default_rng(1)
    n = 50
    base = pd.DataFrame({"season": np.int16(2025), "pa_id": np.arange(n, dtype=np.int64),
                         "pitch_number": np.ones(n, np.int16), "fold": -1, "y": rng.integers(0, 6, n)})
    p = rng.dirichlet(np.ones(6), n)
    p = p / p.sum(axis=1, keepdims=True)
    good = pd.concat([base, pd.DataFrame(p, columns=[f"p{j}" for j in range(6)])], axis=1)
    results = []

    def expect(label, got, want):
        results.append((label, got == want))

    cmp_ = lambda a, b, bw=True, so=True: frames_compare(a, b, KEYS, bw, so)["identical"]  # noqa: E731
    expect("identical frames pass (bitwise)", cmp_(good, good.copy()), True)
    x = good.copy(); x.loc[3, "p2"] = np.nextafter(x.loc[3, "p2"], 1)
    expect("one ulp change fails", cmp_(x, good), False)
    x = good.copy(); y = good.copy(); x.loc[0, "p0"] = 0.0; y.loc[0, "p0"] = -0.0
    expect("-0.0 vs 0.0 fails bitwise", cmp_(x, y), False)
    expect("-0.0 vs 0.0 passes value comparison", cmp_(x, y, bw=False), True)
    x = good.copy(); x["y"] = x["y"].astype(np.int32)
    expect("dtype change fails bitwise", cmp_(x, good), False)
    x = good.iloc[::-1].reset_index(drop=True)
    expect("reordered keys fail when order is required", cmp_(x, good), False)
    expect("reordered keys pass when sorted (value comparison)", cmp_(x, good, bw=False, so=False), True)
    x = pd.concat([good, good.iloc[[0]]], ignore_index=True)
    expect("row count change fails", cmp_(x, good), False)
    x = good.copy(); x.loc[1, "pa_id"] = 0
    expect("duplicate key fails", cmp_(x, good), False)
    x = good.drop(columns=["p5"])
    expect("missing column fails", cmp_(x, good), False)
    nan_a, nan_b = good.copy(), good.copy()   # built through numpy: pandas' .loc canonicalizes NaN payloads
    col_a, col_b = good["p1"].to_numpy().copy(), good["p1"].to_numpy().copy()
    col_a[2] = np.frombuffer(np.uint64(0x7FF8000000000000).tobytes(), np.float64)[0]
    col_b[2] = np.frombuffer(np.uint64(0x7FF8000000000001).tobytes(), np.float64)[0]
    nan_a["p1"], nan_b["p1"] = col_a, col_b
    expect("NaN payloads survive the test's construction",
           (int(nan_a["p1"].to_numpy().view(np.uint64)[2]), int(nan_b["p1"].to_numpy().view(np.uint64)[2])),
           (0x7FF8000000000000, 0x7FF8000000000001))
    expect("same NaN payload passes bitwise", cmp_(nan_a, nan_a.copy()), True)
    expect("different NaN payloads fail bitwise", cmp_(nan_a, nan_b), False)
    expect("probability check passes a valid frame", probability_check(good, "outcome")["ok"], True)
    x = good.copy(); x.loc[0, [f"p{j}" for j in range(6)]] = [0.5, 0.6, -0.1, 0.0, 0.0, 0.0]
    pc = probability_check(x, "outcome")
    expect("negative probability in a row summing to one fails", (pc["ok"], pc["max_row_sum_error"] <= SUM_TOL),
           (False, True))
    x = good.copy(); x.loc[0, [f"p{j}" for j in range(6)]] = [1.25, -0.25, 0.0, 0.0, 0.0, 0.0]
    pc = probability_check(x, "outcome")
    expect("probability above one in a row summing to one fails", (pc["ok"], pc["max_row_sum_error"] <= SUM_TOL),
           (False, True))
    x = good.copy(); x.loc[0, "p0"] += 1e-10
    expect("row sum off by 1e-10 fails", probability_check(x, "outcome")["ok"], False)
    x = good.copy(); x.loc[0, "p0"] = np.nan
    expect("NaN probability fails", probability_check(x, "outcome")["ok"], False)
    expect("wrong category count fails", probability_check(good, "contact")["ok"], False)
    d = {"outcome_lambda": 0.1, "outcome_cv_1": 0.5, "outcome_abnormal_log": [], "contact_league": 12.5}
    expect("identical diagnostics pass", diagnostics_compare(d, dict(d))["identical"], True)
    expect("changed diagnostic fails", diagnostics_compare({**d, "outcome_cv_1": 0.5000000000000001}, d)["identical"],
           False)
    expect("missing diagnostic fails", diagnostics_compare({k: v for k, v in d.items() if k != "contact_league"},
                                                           d)["identical"], False)
    expect("extra diagnostic fails", diagnostics_compare({**d, "new": 1}, d)["identical"], False)
    expect("abnormal-log record fails", diagnostics_compare({**d, "outcome_abnormal_log": [{"lambda": 1}]},
                                                            d)["identical"], False)
    expect("S8_SMOKE refused", env_problems({"S8_SMOKE": "/x"}, smoke=True), ["S8_SMOKE"])
    expect("S8_FULL_SMOKE refused", env_problems({"S8_FULL_SMOKE": "/x"}, smoke=False), ["S8_FULL_SMOKE"])
    expect("production refuses S9_FWD_SMOKE", env_problems({"S9_FWD_SMOKE": "/x"}, smoke=False), ["S9_FWD_SMOKE"])
    expect("smoke allows S9_FWD_SMOKE only", env_problems({"S9_FWD_SMOKE": "/x"}, smoke=True), [])
    rows = good[KEYS].copy()
    obs = pd.DataFrame({"pitcher": rng.integers(0, 9, n), "action": rng.integers(0, 4, n)})
    h1 = query_hash(rows, obs)
    expect("query hash is stable", query_hash(rows.copy(), obs.copy()), h1)
    o2 = obs.copy(); o2.loc[4, "action"] += 1
    expect("query hash sees a changed input", query_hash(rows, o2) != h1, True)
    o3 = obs.copy(); o3["action"] = o3["action"].astype(np.int32)
    expect("query hash sees a dtype change", query_hash(rows, o3) != h1, True)
    expect("empty S9_FWD_SMOKE refused in production", env_problems({"S9_FWD_SMOKE": ""}, smoke=False),
           ["S9_FWD_SMOKE (set but empty)"])
    expect("blank S9_FWD_SMOKE refused", env_problems({"S9_FWD_SMOKE": "  "}, smoke=True),
           ["S9_FWD_SMOKE (set but empty)"])
    expect("production with no smoke variable passes", env_problems({"HOME": "/x"}, smoke=False), [])

    good_rec = {"stage": "control", "status": "finished", "passed": True, "selftest": {"passed": True},
                "provenance": {"mode": "production", "smoke": None, "script_sha256": "s1"}}
    rp = lambda r, stage="control": record_problems(r, stage, "s1", "production", None)  # noqa: E731
    expect("a good predecessor record passes", rp(good_rec), [])
    expect("a missing record fails", len(rp(None)), 1)
    expect("a smoke record fails in production",
           len(rp({**good_rec, "provenance": {**good_rec["provenance"], "mode": "smoke", "smoke": "/tmp/x"}})), 1)
    expect("a record from another smoke directory fails in smoke mode",
           len(record_problems({**good_rec, "provenance": {**good_rec["provenance"], "mode": "smoke",
                                                           "smoke": "/a"}}, "control", "s1", "smoke", "/b")), 1)
    expect("a record with no mode fails", len(rp({**good_rec, "provenance": {"script_sha256": "s1"}})), 1)
    expect("a record of another stage fails", len(rp(good_rec, stage="fit")), 1)
    expect("a reservation fails", len(rp({**good_rec, "status": "reserved"})), 1)
    expect("a failed record fails", len(rp({**good_rec, "passed": False})), 1)
    expect("passed must be literally true", len(rp({**good_rec, "passed": "yes"})), 1)
    expect("another script's record fails",
           len(rp({**good_rec, "provenance": {**good_rec["provenance"], "script_sha256": "s2"}})), 1)
    expect("a record without a passing self-test fails", len(rp({k: v for k, v in good_rec.items()
                                                                  if k != "selftest"})), 1)
    chain = {"guard": {"predecessors": {}}, "control": {"predecessors": {"guard": "g1"}},
             "fit": {"predecessors": {"guard": "g1", "control": "c1"}}}
    expect("an intact chain passes", chain_problems(chain, {"guard": "g1", "control": "c1", "fit": "f1"}), [])
    expect("a changed predecessor record fails",
           len(chain_problems(chain, {"guard": "g1", "control": "c2", "fit": "f1"})), 1)
    expect("a missing predecessor hash fails",
           len(chain_problems({**chain, "fit": {"predecessors": {"guard": "g1"}}},
                              {"guard": "g1", "control": "c1", "fit": "f1"})), 1)
    expect("a record without predecessors fails", len(chain_problems({"control": {}}, {"guard": "g1"})), 1)
    expect("a removed predecessor record fails", len(chain_problems({"control": chain["control"]}, {})), 1)
    mb = json.dumps({"models": {"C1": 1}}).encode()
    expect("a manifest matching the fit record parses",
           manifest_from_bytes(mb, hashlib.sha256(mb).hexdigest()), {"models": {"C1": 1}})
    try:
        manifest_from_bytes(mb.replace(b"1", b"2"), hashlib.sha256(mb).hexdigest())
        refused = False
    except RuntimeError:
        refused = True
    expect("a substituted manifest is refused", refused, True)
    return results


def stage_selftest():
    results = _selftest()
    failed = [label for label, ok in results if not ok]
    for label, ok in results:
        print(f"{'ok  ' if ok else 'FAIL'} {label}", flush=True)
    if failed:
        sys.exit(f"selftest failed: {failed}")
    print(f"selftest passed: {len(results)} checks; nothing read or written", flush=True)


STAGES = {"selftest": stage_selftest, "guard": stage_guard, "control": stage_control, "fit": stage_fit,
          "reload": stage_reload, "compare": stage_compare, "recheck": stage_recheck}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in STAGES:
        sys.exit(__doc__)
    STAGES[sys.argv[1]]()
