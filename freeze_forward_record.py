"""S9 forward-artifact record: data/models/C1_forward_2025.json (spec S9_forward_models_2025.md revision 2, section 8).

    python3.12 freeze_forward_record.py preview <dir>        # proposed (preview), not frozen; writes only <dir>
    python3.12 freeze_forward_record.py write <YYYY-MM-DD>   # once, on the user's authorization of that date
    python3.12 freeze_forward_record.py verify               # read-only, repeatable

Before building, the production stage chain must pass `freeze_forward_2025.py recheck` (exit 0) in a subprocess. The
record lists the five stage records, the fit manifest, the eight artifacts with their query hashes, and the sources it
depends on. The writer and the S9 specification are archived beside it (Sessions/Archive/C1_forward_2025_<date>/),
each copy verified by hash, because neither is tracked by git at the time of writing.
"""

import os
import sys

if any("SMOKE" in k for k in os.environ):
    sys.exit("refused: a *SMOKE* environment variable is set; the forward-artifact record is production only")

import datetime  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "data" / "models"
RESEARCH = ROOT / "data" / "research"
RECORD = MODELS / "C1_forward_2025.json"
FORWARD = MODELS / "forward_2025"
MANIFEST = FORWARD / "manifest.json"
C1_FROZEN = MODELS / "C1_frozen.json"
SPEC = ROOT / "research_notes" / "research_program" / "S9_forward_models_2025.md"
SCRIPT = ROOT / "freeze_forward_2025.py"
ORDER = ("guard", "control", "fit", "reload", "compare")
STAGE_REC = {s: RESEARCH / f"s9_fwd_{s}.json" for s in ORDER}
SCRIPT_SHA = "c4feea7a78f84dac0a967e643e3937b135efb0b8fe76838fdba98038fb229e04"   # the reviewed script
SCRIPT_COMMIT = "14b974a"
COMPARE_DATE = datetime.date(2026, 11, 19)                                          # the chain completed
VERSIONS, KINDS = ("P0", "C1"), ("outcome", "contact")
PAIRS = [(n, k) for n in VERSIONS for k in KINDS]
ARCHIVED = ("freeze_forward_record.py", "S9_forward_models_2025.md")
QUERY_SEASON = 2025


def load_verified(path, expected_sha: str):
    """(object, None) when the file exists, its bytes hash to expected_sha and parse as JSON from those same bytes;
    otherwise (None, what is wrong). Never raises for a missing, changed or malformed file."""
    p = Path(path)
    if not p.is_file():
        return None, "missing"
    data = p.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha:
        return None, "changed (sha256 differs)"
    try:
        obj = json.loads(data)
    except ValueError as e:
        return None, f"malformed JSON ({e})"
    if not isinstance(obj, dict):
        return None, f"not a JSON object (a {type(obj).__name__})"
    return obj, None


def valid_date(value) -> str | None:
    """The authorization date as written by `write`: exactly YYYY-MM-DD, from the compare stage to today. Anything
    else (a path, a timestamp, an out-of-range date) is None and is never used to build a path."""
    if not isinstance(value, str) or len(value) != 10 or value[4] != "-" or value[7] != "-":
        return None
    try:
        d = datetime.date.fromisoformat(value)
    except ValueError:
        return None
    return value if d.isoformat() == value and COMPARE_DATE <= d <= datetime.date.today() else None


def package_hash() -> str:
    """research.code_state's hash over gamecalling/*.py (the pickled classes load from this code)."""
    h = hashlib.sha256()
    for p in sorted((ROOT / "gamecalling").glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rel(p) -> str:
    """Project-relative; a path outside the project (scratch tests) stays absolute."""
    p = Path(p).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def file_rec(p) -> dict:
    p = Path(p)
    return {"path": rel(p), "bytes": p.stat().st_size, "sha256": sha(p)}


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def load(p):
    return json.loads(Path(p).read_text())


def check_date(text: str) -> str:
    d = datetime.date.fromisoformat(text)
    if not COMPARE_DATE <= d <= datetime.date.today():
        raise ValueError(f"authorization date {d} is before the compare stage ({COMPARE_DATE}) or after today")
    return d.isoformat()


def preconditions() -> dict:
    """The reviewed, committed script; the stage chain passing its own read-only recheck."""
    if sha(SCRIPT) != SCRIPT_SHA:
        raise RuntimeError("freeze_forward_2025.py is not the reviewed script")
    committed = subprocess.run(["git", "show", f"{SCRIPT_COMMIT}:freeze_forward_2025.py"], cwd=ROOT,
                               capture_output=True).stdout
    if hashlib.sha256(committed).hexdigest() != SCRIPT_SHA:
        raise RuntimeError(f"freeze_forward_2025.py at {SCRIPT_COMMIT} is not the reviewed script")
    r = subprocess.run([sys.executable, str(SCRIPT), "recheck"], cwd=ROOT, capture_output=True, text=True,
                       env={k: v for k, v in {**os.environ, "PYTHONPATH": str(ROOT)}.items() if "SMOKE" not in k})
    if r.returncode != 0:
        raise RuntimeError(f"freeze_forward_2025.py recheck failed:\n{r.stdout}{r.stderr}")
    return {"command": "python3.12 freeze_forward_2025.py recheck", "exit": 0,
            "output": r.stdout.strip().splitlines()[-3:]}


def build(authorized: str | None = None) -> dict:
    """authorized=None builds a preview: proposed, not frozen, no approval for this record."""
    recheck = preconditions()
    recs = {s: load(STAGE_REC[s]) for s in ORDER}
    stage_files = {s: file_rec(STAGE_REC[s]) for s in ORDER}
    for s in ORDER:
        r = recs[s]
        if not (r.get("stage") == s and r.get("status") == "finished" and r.get("passed") is True
                and r["provenance"]["mode"] == "production" and r["provenance"]["script_sha256"] == SCRIPT_SHA):
            raise RuntimeError(f"the {s} record is not a passed, finished production record of the reviewed script")
    for s in ORDER[1:]:
        want = {q: stage_files[q]["sha256"] for q in ORDER[:ORDER.index(s)]}
        if recs[s]["predecessors"] != want:
            raise RuntimeError(f"the {s} record's predecessor hashes do not match the stored records")
    man_rec = file_rec(MANIFEST)
    man, bad = load_verified(MANIFEST, recs["fit"]["manifest"]["sha256"])
    if bad or man_rec["sha256"] != recs["fit"]["manifest"]["sha256"]:
        raise RuntimeError(f"the fit manifest does not match the fit record ({bad})")
    prov = man["provenance"]
    code = package_hash()
    if code != prov["code"]["gamecalling_sha256"]:
        raise RuntimeError(f"code drift: gamecalling/ hashes to {code[:12]}, the fit manifest's code is "
                           f"{prov['code']['gamecalling_sha256'][:12]}; nothing is written")
    frozen = load(C1_FROZEN)
    if sha(C1_FROZEN) != recs["guard"]["c1_frozen_sha256"]:
        raise RuntimeError("C1_frozen.json is not the record the guard saw")

    artifacts = []
    for name in VERSIONS:
        m = man["models"][name]
        for kind in KINDS:
            f = m["files"][kind]
            for what in ("pickle", "queries"):
                live = file_rec(ROOT / f[what]["path"])
                if live != f[what]:
                    raise RuntimeError(f"{f[what]['path']} does not match the fit manifest")
            artifacts.append({"model": name, "component": kind, "model_class": f["model_class"],
                              "pickle": f["pickle"], "queries": f["queries"],
                              "query_rows": m["queries"][kind]["rows"],
                              "query_inputs_sha256": m["queries"][kind]["query_sha256"],
                              "probability_check": m["checks"][kind]})
    for s in ORDER[1:]:
        if not (recs[s]["guard_before"]["passed"] and recs[s]["guard_after"]["passed"]):
            raise RuntimeError(f"the {s} record does not show the guard passing before and after it")
    if man["models"]["P0"]["diagnostics"].get("outcome_lambda") is not None:
        raise RuntimeError("P0 records a lambda; the record's description of P0 would be wrong")
    ctl = recs["control"]
    rel_checks = [c for c in recs["reload"]["checks"]]
    c1d = man["models"]["C1"]["diagnostics"]

    status = ({"frozen": True, "authorized": authorized, "authorized_by": "the user (\"I authorize\")",
               "statement": "The all-2025 P0 and C1 forward models listed here are frozen. These exact files are "
                            "the models used for forward (2026) prediction; they are not refitted or replaced."}
              if authorized else
              {"frozen": False, "statement": "proposed (preview): not frozen; no authorization for this record"})
    return {
        "record": "S9 forward-artifact record: all-2025 P0 and C1 models",
        "record_version": 1,
        "built": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "status": status,
        "specification": {"note": rel(SPEC), "revision": 2, "section": "8 (afterwards)", "sha256": sha(SPEC),
                          "requirement": "Written once, separate from C1_frozen.json, only after checks 5a-5e "
                                         "pass and on the user's authorization. Marks the forward models frozen "
                                         "and lists the pickles, query hashes and manifest."},
        "procedure": {"c1_frozen": file_rec(C1_FROZEN),
                      "reproduction_commit": frozen["source"]["reproduction_commit"],
                      "freeze_source_commit": frozen["source"]["freeze_source_commit"],
                      "gamecalling_sha256": prov["code"]["gamecalling_sha256"],
                      "note": "The fitting procedure is C1's frozen procedure (C1_frozen.json); P0 is the frozen "
                              "count-based baseline. Training: every eligible 2025 pitch (fit season "
                              f"{man['fit_season']}); queries: 2025 (query season {man['query_season']})."},
        "script": {"path": rel(SCRIPT), "sha256": SCRIPT_SHA, "commit": git("rev-parse", SCRIPT_COMMIT),
                   "revision": 2},
        "execution": {"git_head": prov["git_head"], "python": prov["python"], "platform": prov["platform"],
                      "numpy": prov["numpy"], "scipy": prov["scipy"], "pandas": prov["pandas"],
                      "pyarrow": prov["pyarrow"], "pickle_protocol": prov["pickle_protocol"],
                      "workers": prov["workers"], "thread_env": prov["thread_env"]},
        "stage_records": {s: {**stage_files[s], "passed": recs[s]["passed"], "started": recs[s]["started"],
                              "finished": recs[s]["finished"]} for s in ORDER},
        "acceptance": {
            "5a_control": {"result": "passed",
                           "predictions": [{"check": c["check"], "rows": c["rows"], "identical": c["identical"],
                                            "bitwise_identical": c["bitwise_identical"],
                                            "same_row_order": c["same_row_order"]}
                                           for c in ctl["checks"] if "rows" in c],
                           "diagnostics": [{"check": c["check"], "fields": c["fields"], "identical": c["identical"]}
                                           for c in ctl["checks"] if "fields" in c]},
            "5b_numerical": {"result": "passed (batter.accept_fit raises on failure; the fit passed)",
                             "c1_selected_lambda": {"outcome": c1d.get("outcome_lambda"),
                                                    "contact": c1d.get("contact_lambda")},
                             "c1_abnormal_terminations": {"outcome": c1d.get("outcome_abnormal_log"),
                                                          "contact": c1d.get("contact_abnormal_log")},
                             "p0": "records no selected lambda and no optimizer log"},
            "5c_probabilities": {"result": "passed",
                                 "checks": [{"model": a["model"], "component": a["component"],
                                             "rows": a["probability_check"]["rows"],
                                             "max_row_sum_error": a["probability_check"]["max_row_sum_error"],
                                             "ok": a["probability_check"]["ok"]} for a in artifacts]},
            "5d_reload": {"result": "passed", "checks": len(rel_checks),
                          "all_identical": all(c["identical"] for c in rel_checks),
                          "note": "Fresh process; each pickle's bytes matched the manifest before loading; query "
                                  "input hashes matched; regenerated predictions bitwise identical to the saved "
                                  "Parquet files (columns, keys, row order, dtypes, values)."},
            "5e_guard": {"result": "the guard stage passed; the guard check passed before and after control, "
                                   "fit, reload and compare",
                         "c1_frozen_sha256": recs["guard"]["c1_frozen_sha256"]},
            "chain_recheck": recheck,
        },
        "manifest": man_rec,
        "training": {n: {"season": man["fit_season"], "rows": man["models"][n]["train_rows"],
                         "games": man["models"][n]["train_games"]} for n in VERSIONS},
        "artifacts": artifacts,
        "reviews": [
            {"reviewer": "Astra", "summary": "Reviewed the script (revisions 1 and 2) and accepted each production "
                                             "stage after independent checks: recheck, the 80 C1 diagnostics "
                                             "against the gate manifest, recomputed probability checks from the "
                                             "saved Parquet files, and the record and manifest hashes. "
                                             "Recommended writing this record."},
            {"reviewer": "Google Gemini 4 Neon (independent; relayed by Astra)",
             "summary": "No material blocker. Reloaded all four models in a fresh process and regenerated the 2025 "
                        "predictions bit for bit (keys, row order, dtypes); C1's stored feature index holds all "
                        "2,428 games from 2025 and none from 2024; all 80 C1 control diagnostics match; the "
                        "five-record hash chain and all eight artifacts match the fit manifest. Recommended "
                        "writing this record."},
        ],
        "limits": [
            "This record freezes these exact model files. It does not establish predictive performance on 2026 "
            "data, which remains open.",
            "Stage B (A5) must still settle and test how 2026 queries are built: batters missing from 2025 (the "
            "shrinkage prior), pitchers and pitch-type cells missing from 2025 (the pooled levels), the arsenal and "
            "action set, the runner flag, and reporting of the seen-cell flag.",
            "No 2026 data has been downloaded; 2023 modeling and the 2026 download are separately unauthorized.",
        ],
        "permitted_changes": "None to the listed files. A defect found later is reported and handled by a new, "
                             "separately authorized record; this record and its files are not overwritten.",
        "archive": {},
    }


def archive_dir(date: str) -> Path:
    return ROOT / "Sessions" / "Archive" / f"C1_forward_2025_{date}"


def stage_preview(directory: str):
    d = Path(directory).resolve()
    if d.is_relative_to(ROOT):
        raise RuntimeError("the preview directory must be outside the project")
    d.mkdir(parents=True, exist_ok=True)
    rec = build()
    out = d / "C1_forward_2025.preview.json"
    out.write_text(json.dumps(rec, indent=1, default=str))
    print(f"preview written: {out} (proposed, not frozen)")


def write_state(date: str) -> dict:
    """Every destination the write touches, and its lock. Any one existing means a write is running, ran, or was
    interrupted."""
    adir = archive_dir(date)
    return {"lock": RECORD.with_name(RECORD.name + ".lock"), "record": RECORD,
            "record.partial": RECORD.with_name(RECORD.name + ".partial"),
            "archive": adir, "archive.partial": adir.with_name(adir.name + ".partial")}


def describe_state(state: dict) -> str:
    present = [k for k, v in state.items() if v.exists()]
    if not present:
        return "nothing exists: no write has started"
    if present == ["record", "archive"]:
        return "the record and its archive are published: it is written once"
    if "record" in present and "archive" in present:
        return (f"the record and its archive are published, but leftovers remain (present: {present}). Run "
                "`verify`, then inspect the leftovers")
    if "record" in present:
        return (f"a record is published without its finalized archive (present: {present}): inconsistent; "
                "diagnose before doing anything")
    if present == ["lock"]:
        return ("a write lock exists with no destination: another write is running, or one stopped before creating "
                "anything. Check that no write is running, then remove the lock")
    if "archive" in present:
        return ("an earlier write was interrupted after finalizing the archive but before publishing the record "
                f"(present: {present}). Nothing is published. Inspect, then move these aside before a new "
                "authorized write")
    return (f"an earlier write was interrupted before publication (present: {present}). Nothing is published. "
            "Inspect, then move these aside before a new authorized write")


def _published(tmp: Path, body: bytes) -> bool:
    """Whether the final record is the one this write produced (same inode as the partial, or the same bytes)."""
    if not RECORD.exists():
        return False
    if tmp.exists():
        return os.path.samefile(RECORD, tmp)
    return RECORD.read_bytes() == body


def stage_write(date_text: str):
    """Order: reserve (an O_EXCL lock) -> refuse if any destination exists -> build (refuses code drift) -> archive
    copies into archive.partial, each checked against the digest captured before copying -> record.partial written
    and read back -> the full verification on the unpublished record and the partial archive -> publish
    (archive.partial renamed to archive, then record.partial hard-linked to the record, never overwriting). If
    publication fails, the archive is renamed back unless the record was in fact published; the lock stays with any
    partial files so the state is unambiguous."""
    date = check_date(date_text)
    state = write_state(date)
    lock = state["lock"]
    if RECORD.exists():               # whatever date it was authorized under
        raise FileExistsError(f"{rel(RECORD)} is already published: it is written once")
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        raise FileExistsError(describe_state(state)) from None
    created = False                   # any destination made: from then on the lock stays on failure
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps({"pid": os.getpid(), "started": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                                 "date": date}))
        others = {k: v for k, v in state.items() if k != "lock"}      # this run's own lock is not a leftover
        if any(v.exists() for v in others.values()):
            raise FileExistsError(describe_state(others))
        writer = Path(__file__).resolve()
        digests = {writer.name: sha(writer)}          # the writer as it runs, captured before the build
        rec = build(authorized=date)
        digests[SPEC.name] = rec["specification"]["sha256"]    # the specification as the build recorded it
        adir, part, tmp = state["archive"], state["archive.partial"], state["record.partial"]
        part.mkdir(parents=True)
        created = True
        for src in (writer, SPEC):
            dst = part / src.name
            shutil.copy2(src, dst)
            got = sha(dst)
            if got != digests[src.name]:
                raise RuntimeError(f"the archived copy of {src.name} hashes to {got[:12]}, not the digest captured "
                                   f"during the build ({digests[src.name][:12]}); nothing is published")
            rec["archive"][src.name] = {"source": rel(src), "copy": rel(adir / src.name),
                                        "sha256": digests[src.name]}
        problems, _, _ = verify_record(rec, archive_root=part)
        if problems:
            raise RuntimeError(f"the unpublished record fails verification; nothing is published: {problems}")
        body = json.dumps(rec, indent=1, default=str).encode()
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "wb") as fh:
            fh.write(body)
            fh.flush()
            os.fsync(fh.fileno())
        if tmp.read_bytes() != body:
            raise RuntimeError("the partial record did not read back identically; nothing is published")
        part.rename(adir)
        try:
            os.link(tmp, RECORD)          # fails if the record exists; never overwrites
            os.unlink(tmp)
        except BaseException as e:
            try:
                if _published(tmp, body):
                    if tmp.exists():
                        os.unlink(tmp)
                    raise RuntimeError(f"interrupted ({type(e).__name__}: {e}) after the record was published. The "
                                       f"archive is kept with it. Run `verify` to confirm the published state; "
                                       f"{rel(lock)} remains until then") from e
                adir.rename(part)         # nothing published: back to all-partial
            except RuntimeError:
                raise
            except BaseException as e2:
                raise RuntimeError(f"publication failed ({type(e).__name__}: {e}) and recovery also failed "
                                   f"({type(e2).__name__}: {e2}). Actual state: {describe_state(state)}") from e
            raise RuntimeError(f"publication failed ({type(e).__name__}: {e}). Rolled back: nothing is published. "
                               f"State: {describe_state(state)}") from e
    except BaseException:
        if not created and lock.exists():
            os.unlink(lock)               # nothing was made: release the reservation
        raise
    print(f"written: {rel(RECORD)} sha256 {sha(RECORD)} (authorized {date}); archive {rel(state['archive'])}")
    stage_verify()
    os.unlink(lock)


def _same(desc: dict, want: dict) -> bool:
    return {k: desc.get(k) for k in ("path", "bytes", "sha256")} == {k: want.get(k) for k in ("path", "bytes",
                                                                                                "sha256")}


def _live(desc: dict) -> str | None:
    """None if the file at desc['path'] has desc's bytes and sha256; otherwise what is wrong."""
    p = ROOT / desc["path"]
    if not p.is_file():
        return "missing"
    if p.stat().st_size != desc["bytes"]:
        return f"{p.stat().st_size} bytes, recorded {desc['bytes']}"
    if sha(p) != desc["sha256"]:
        return "changed (sha256 differs)"
    return None


def expected_paths(rec: dict) -> dict:
    """Where each listed file must be. A record naming any other path, even with matching hashes, fails."""
    date = valid_date((rec.get("status") or {}).get("authorized"))
    out = {"manifest": rel(MANIFEST), "c1_frozen": rel(C1_FROZEN), "script": rel(SCRIPT),
           **{f"stage:{s}": rel(STAGE_REC[s]) for s in ORDER},
           **{f"{n} {k} pickle": rel(FORWARD / n / f"{k}.pkl") for n, k in PAIRS},
           **{f"{n} {k} queries": rel(FORWARD / n / f"queries_{QUERY_SEASON}_{k}.parquet") for n, k in PAIRS},
           f"archive:{ARCHIVED[0]}:source": rel(Path(__file__).resolve()),
           f"archive:{ARCHIVED[1]}:source": rel(SPEC)}
    for name in ARCHIVED:
        out[f"archive:{name}:copy"] = rel(archive_dir(date) / name) if date else None
    return out


def verify_record(rec: dict, archive_root=None) -> tuple:
    """(problems, notes, files_checked). Problems fail verification; notes are informational. archive_root: check the
    archived copies there instead of at their recorded location (the write checks its partial archive this way)."""
    problems, notes, n = [], [], 0
    st = rec.get("status") or {}
    if st.get("frozen") is not True:
        problems.append("the record is not marked frozen")
    date = valid_date(st.get("authorized"))
    if date is None:
        problems.append(f"the authorization date {st.get('authorized')!r} is not a YYYY-MM-DD date from "
                        f"{COMPARE_DATE} to today; the archive location cannot be derived")
    want = expected_paths(rec)

    def bound(label, key, desc):
        if desc.get("path") != want[key]:
            problems.append(f"{label}: recorded at {desc.get('path')}, expected {want[key]}")

    # The script: the reviewed digest, at its path, and the live file must be it.
    bound("forward script", "script", rec["script"])
    if rec["script"].get("sha256") != SCRIPT_SHA:
        problems.append("the record names a script other than the reviewed one")
    n += 1
    if not SCRIPT.is_file() or sha(SCRIPT) != SCRIPT_SHA:
        problems.append(f"the live {rel(SCRIPT)} is not the reviewed script; its recheck is not run")

    # Stage records: exactly the five, each at its path, parsed from the bytes whose hash is recorded.
    srecs = rec.get("stage_records") or {}
    if sorted(srecs) != sorted(ORDER):
        problems.append(f"stage records listed: {sorted(srecs)}, expected {sorted(ORDER)}")
    parsed = {}
    for s_ in ORDER:
        if s_ not in srecs:
            continue
        d = srecs[s_]
        bound(f"{s_} stage record", f"stage:{s_}", d)
        n += 1
        bad = _live(d)
        obj, bad2 = load_verified(ROOT / d["path"], d.get("sha256"))
        if bad or bad2:
            problems.append(f"{s_} stage record {d['path']}: {bad or bad2}")
            continue
        if not isinstance(obj, dict) or obj.get("stage") != s_ or obj.get("passed") is not True:
            problems.append(f"{s_} stage record: does not record a passed {s_} stage")
            continue
        parsed[s_] = obj
    for s_, obj in parsed.items():
        summary = srecs[s_]
        for field in ("passed", "started", "finished"):
            if summary.get(field) != obj.get(field):
                problems.append(f"{s_} stage record: the record's {field} ({summary.get(field)!r}) differs from the "
                                f"stage record's ({obj.get(field)!r})")
    for s_ in ORDER[1:]:
        if s_ in parsed and all(q in srecs for q in ORDER[:ORDER.index(s_)]):
            chain = {q: srecs[q].get("sha256") for q in ORDER[:ORDER.index(s_)]}
            if parsed[s_].get("predecessors") != chain:
                problems.append(f"{s_} stage record: its predecessor hashes do not match the listed records")

    # The manifest: at its path, parsed from verified bytes, and the one the fit record names.
    bound("fit manifest", "manifest", rec["manifest"])
    n += 1
    bad = _live(rec["manifest"])
    man, bad2 = load_verified(ROOT / rec["manifest"]["path"], rec["manifest"].get("sha256"))
    if bad or bad2:
        problems.append(f"fit manifest {rec['manifest']['path']}: {bad or bad2}")
        man = None
    if "fit" in parsed and (parsed["fit"].get("manifest") or {}).get("sha256") != rec["manifest"].get("sha256"):
        problems.append("the fit stage record names a different manifest")

    # The procedure: C1_frozen.json at its path, verified, and the one the guard saw; the commits and code hash
    # agree with C1_frozen.json and the verified manifest.
    proc = rec["procedure"]
    bound("C1_frozen.json", "c1_frozen", proc["c1_frozen"])
    n += 1
    bad = _live(proc["c1_frozen"])
    frozen, bad2 = load_verified(ROOT / proc["c1_frozen"]["path"], proc["c1_frozen"].get("sha256"))
    if bad or bad2:
        problems.append(f"C1_frozen.json {proc['c1_frozen']['path']}: {bad or bad2}")
    else:
        src = frozen.get("source") or {}
        if (proc.get("reproduction_commit"), proc.get("freeze_source_commit")) != (
                src.get("reproduction_commit"), src.get("freeze_source_commit")):
            problems.append("the record's commits differ from C1_frozen.json's")
    if "guard" in parsed and parsed["guard"].get("c1_frozen_sha256") != proc["c1_frozen"].get("sha256"):
        problems.append("the guard stage record saw a different C1_frozen.json")
    if man is not None:
        mprov = man.get("provenance") or {}
        if proc.get("gamecalling_sha256") != (mprov.get("code") or {}).get("gamecalling_sha256"):
            problems.append("the record's code hash differs from the verified manifest's")
        if (mprov.get("reproduction_commit"), mprov.get("freeze_source_commit")) != (
                proc.get("reproduction_commit"), proc.get("freeze_source_commit")):
            problems.append("the record's commits differ from the verified manifest's")
        if mprov.get("script_sha256") != SCRIPT_SHA:
            problems.append("the verified manifest was made by a different script")
        if rec["script"].get("commit") != mprov.get("git_head"):
            problems.append(f"the record's script commit ({rec['script'].get('commit')}) differs from the verified "
                            f"manifest's production HEAD ({mprov.get('git_head')})")

    # Artifacts: exactly the four pairs, each at its path and cross-checked against the verified manifest.
    arts = rec.get("artifacts") or []
    pairs = [(a.get("model"), a.get("component")) for a in arts]
    if sorted(pairs) != sorted(PAIRS) or len(pairs) != len(set(pairs)):
        problems.append(f"artifact entries {pairs}: expected exactly {PAIRS}")
    if man is None:
        problems.append("the fit manifest cannot be verified, so the artifact entries cannot be cross-checked")
    for a in arts:
        label = f"{a.get('model')} {a.get('component')}"
        if (a.get("model"), a.get("component")) not in PAIRS:
            continue
        for what in ("pickle", "queries"):
            bound(f"{label} {what}", f"{label} {what}", a[what])
        if man is not None:
            m = (man.get("models") or {}).get(a["model"]) or {}
            f = (m.get("files") or {}).get(a["component"])
            q = (m.get("queries") or {}).get(a["component"])
            if f is None or q is None:
                problems.append(f"{label}: not in the fit manifest")
                continue
            for what in ("pickle", "queries"):
                if not _same(a[what], f[what]):
                    problems.append(f"{label} {what}: descriptor differs from the fit manifest")
            if a.get("model_class") != f.get("model_class"):
                problems.append(f"{label}: model class differs from the fit manifest")
            if a.get("query_rows") != q.get("rows") or a.get("query_inputs_sha256") != q.get("query_sha256"):
                problems.append(f"{label}: query rows or query-input hash differs from the fit manifest")
            if a.get("probability_check") != (m.get("checks") or {}).get(a["component"]):
                problems.append(f"{label}: probability check differs from the fit manifest")
        for what in ("pickle", "queries"):
            n += 1
            bad = _live(a[what])
            if bad:
                problems.append(f"{label} {what} {a[what]['path']}: {bad}")

    # Archived sources: exactly the two, at their paths; the frozen copy must match; the live source may move on.
    arch = rec.get("archive") or {}
    if sorted(arch) != sorted(ARCHIVED):
        problems.append(f"archive entries {sorted(arch)}: expected exactly {sorted(ARCHIVED)}")
    if (arch.get(SPEC.name) or {}).get("sha256") != rec["specification"].get("sha256"):
        problems.append("the archived specification's digest differs from the record's specification digest")
    for name, a in arch.items():
        if name not in ARCHIVED:
            continue
        n += 1
        if a.get("copy") != want[f"archive:{name}:copy"]:
            problems.append(f"archived {name}: copy recorded at {a.get('copy')}, expected "
                            f"{want[f'archive:{name}:copy']}")
        if a.get("source") != want[f"archive:{name}:source"]:
            problems.append(f"archived {name}: source recorded as {a.get('source')}, expected "
                            f"{want[f'archive:{name}:source']}")
        copy = Path(archive_root) / name if archive_root else ROOT / a["copy"]
        home = Path(archive_root) if archive_root else (archive_dir(date) if date else None)
        if home is None or not copy.resolve().is_relative_to(home.resolve()):
            problems.append(f"archived {name}: {rel(copy)} is not inside the intended archive directory "
                            f"{rel(home) if home else '(none: no valid authorization date)'}")
        elif not copy.is_file():
            problems.append(f"frozen archive copy of {name} is missing: {rel(copy)}")
        elif sha(copy) != a.get("sha256"):
            problems.append(f"frozen archive copy of {name} has changed: {rel(copy)}")
        live = ROOT / a.get("source", "")
        if not live.is_file():
            notes.append(f"live source {a.get('source')} is missing (informational: the archived copy is the "
                         "frozen version)")
        elif sha(live) != a.get("sha256"):
            notes.append(f"live source {a.get('source')} has changed since the record (informational)")

    # Code drift: the pickled model classes load from gamecalling/.
    code = package_hash()
    if code != proc.get("gamecalling_sha256"):
        problems.append(f"code drift: gamecalling/ hashes to {code[:12]}, the record's procedure is "
                        f"{str(proc.get('gamecalling_sha256'))[:12]}; the pickles may not load or behave as frozen")
    return problems, notes, n


def stage_verify(path=None):
    path = Path(path) if path else RECORD
    problems, notes, n = [], [], 0
    if not path.is_file():
        sys.exit(f"verify failed: {rel(path)} is missing")
    data = path.read_bytes()
    print(f"record {rel(path)} sha256 {hashlib.sha256(data).hexdigest()}")
    try:
        rec = json.loads(data)
        problems, notes, n = verify_record(rec)
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        problems.append(f"the record is malformed ({type(e).__name__}: {e})")
    ran = False
    if SCRIPT.is_file() and sha(SCRIPT) == SCRIPT_SHA:     # never run a script other than the reviewed one
        ran = True
        r = subprocess.run([sys.executable, str(SCRIPT), "recheck"], cwd=ROOT, capture_output=True, text=True,
                           env={k: v for k, v in {**os.environ, "PYTHONPATH": str(ROOT)}.items()
                                if "SMOKE" not in k})
        if r.returncode != 0:
            problems.append("freeze_forward_2025.py recheck failed; its output follows")
    elif not any("is not the reviewed script" in p for p in problems):
        problems.append(f"the live {rel(SCRIPT)} is not the reviewed script; its recheck is not run")
    for line in notes:
        print(f"note {line}", flush=True)
    if problems:
        for line in problems:
            print(f"FAIL {line}", flush=True)
        if ran and r.returncode != 0:
            print("--- freeze_forward_2025.py recheck ---\n" + (r.stdout + r.stderr).rstrip(), flush=True)
        sys.exit(f"verify failed: {len(problems)} problem(s)")
    print(f"verify passed: {n} files at their expected paths (4 model/component entries cross-checked against the "
          f"verified manifest; procedure, commits and code hash cross-checked; both archived sources); no code drift; "
          f"the stage chain recheck passed")


if __name__ == "__main__":
    cmds = {"preview": stage_preview, "write": stage_write, "verify": stage_verify}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds or (sys.argv[1] != "verify") != (len(sys.argv) == 3):
        sys.exit(__doc__)
    if sys.argv[1] == "verify":
        stage_verify()
    else:
        cmds[sys.argv[1]](*sys.argv[2:])
