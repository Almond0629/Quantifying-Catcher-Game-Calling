"""C1 freeze, step 2: the procedure freeze record (revision 2, 2026-11-18).

Outside gamecalling/ on purpose, like freeze_c1_reproduce.py (the package's code hash must not change).

    python3.12 freeze_c1_record.py preview <dir>       # build the PROPOSED record into <dir>/C1_frozen.json only
    python3.12 freeze_c1_record.py write <YYYY-MM-DD>  # once: verified source archive, then data/models/C1_frozen.json
    python3.12 freeze_c1_record.py verify              # read-only: recheck the written record, archive and artifacts

<YYYY-MM-DD> is the date the user authorized writing the record. It is recorded as given and names the archive
(Sessions/Archive/C1_freeze_<date>/). It may not be before the reproduction (2026-11-16) or after today.

The record freezes the fitting PROCEDURE. The all-2025 forward models of P0 and C1 do not exist yet and are recorded as
pending, not frozen. Writing the record does not lock the S9 protocol and does not authorize any new-season work.

Revision 2 makes the four corrections from Astra's review of revision 1 (kept as
Sessions/Archive/freeze_c1_record_2026-11-16.py):
1. The reproduction commit (60f4cd7) and the freeze-source commit (HEAD when the record is built) are recorded
   separately. Committing these scripts does not break the writer: it requires gamecalling/ to be clean, identical to
   the reproduction commit and at code hash 1000e503014a, and freeze_c1_reproduce.py to match the hash the
   comparison stored.
2. A preview is "proposed", not frozen, and carries no approval for this record. `write` records the date given.
3. The source-history disclosure claims only what the hashes and the reproduction establish.
4. Archived copies are checked against the hashes in the record before the record is written, and `verify` rechecks
   them. The S8 note's stage 6 entry is a separate, explicit step after the record (the record's next_steps).
"""

import datetime
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from gamecalling import config, research

ROOT = Path(__file__).resolve().parent
MODELS = config.DATA_DIR / "models"
RESEARCH = config.DATA_DIR / "research"
RECORD = MODELS / "C1_frozen.json"
ARCHIVE_PARENT = ROOT / "Sessions" / "Archive"
ARCHIVE_PREFIX = "C1_freeze_"
REPRO_COMMIT, CODE_HASH = "60f4cd7", "1000e503014a"
HISTORICAL_HASH, CRITERION2_HASH = "2ffaed8e90b1", "bd08e28622d4"
HISTORICAL_DRIVER = "research_s8_full.py"
CRITERION2_RECORD = RESEARCH / "s8_simulation_16M_C1.json"
REPRO_DATE = datetime.date(2026, 11, 16)
SCRIPTS = ("freeze_c1_reproduce.py", "freeze_c1_record.py")
SPEC_FILES = ("project_plan.md", "research_notes/research_program/S4_runners_on.md",
              "research_notes/research_program/S6_batter_index.md",
              "research_notes/research_program/S6_solver_addendum.md",
              "research_notes/research_program/S7_ground_ball.md",
              "research_notes/research_program/S8_combined_model.md",
              "research_notes/validation/01_validation_protocol.md", "requirements.txt")


def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p) -> str:
    return str(Path(p).resolve().relative_to(ROOT))


def file_rec(p) -> dict:
    return {"path": rel(p), "bytes": Path(p).stat().st_size, "sha256": sha(p)}


def git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def package_hash(exclude=()) -> str:
    """research.code_state's hash over gamecalling/*.py, optionally leaving files out."""
    h = hashlib.sha256()
    for p in sorted((ROOT / "gamecalling").glob("*.py")):
        if p.name not in exclude:
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def script_status(name) -> str:
    if git("ls-files", "--error-unmatch", name).returncode != 0:
        return "untracked: not committed"
    if git("status", "--porcelain", "--", name).stdout.strip():
        return "tracked, modified since HEAD"
    return "committed: unchanged since HEAD"


def check_source(run: dict) -> dict:
    """Correction 1: the freeze source need not be the reproduction commit, only identical to it in gamecalling/."""
    code = research.code_state()
    if [s for s in code["git_dirty_files"] if "gamecalling/" in s]:
        raise RuntimeError("gamecalling/ is not clean")
    if not code["gamecalling_sha256"].startswith(CODE_HASH):
        raise RuntimeError(f"gamecalling/ hashes to {code['gamecalling_sha256'][:12]}, not {CODE_HASH}")
    repro = git("rev-parse", "--verify", REPRO_COMMIT + "^{commit}").stdout.strip()
    if not repro or run["provenance"]["git_head"] != repro:
        raise RuntimeError("the reproduction record does not name the reproduction commit")
    if not run["provenance"]["code"]["gamecalling_sha256"].startswith(CODE_HASH):
        raise RuntimeError("the reproduction record's code hash is not the frozen code hash")
    if git("merge-base", "--is-ancestor", repro, "HEAD").returncode != 0:
        raise RuntimeError("HEAD does not descend from the reproduction commit")
    if git("diff", "--quiet", repro, "HEAD", "--", "gamecalling/").returncode != 0:
        raise RuntimeError("gamecalling/ at HEAD differs from the reproduction commit")
    return {"code": code, "repro_commit": repro}


def check_history(full: dict) -> dict:
    """Correction 3: compute, don't assume, the hashes the disclosure relies on."""
    historical = full["code"]["gamecalling_sha256"]
    criterion2 = json.loads(CRITERION2_RECORD.read_text())["code"]["gamecalling_sha256"]
    without_driver = package_hash(exclude=(HISTORICAL_DRIVER,))
    if not historical.startswith(HISTORICAL_HASH) or not criterion2.startswith(CRITERION2_HASH):
        raise RuntimeError("a stored historical code hash is not the one this record describes")
    if without_driver != criterion2:
        raise RuntimeError("the package without the driver no longer matches criterion 2's code hash")
    return {"historical": historical, "criterion2": criterion2, "without_driver": without_driver}


def check_date(text: str) -> str:
    d = datetime.date.fromisoformat(text)
    if not REPRO_DATE <= d <= datetime.date.today():
        raise ValueError(f"authorization date {d} is before the reproduction ({REPRO_DATE}) or after today")
    return d.isoformat()


def build(authorized: str | None = None) -> dict:
    """authorized=None builds a preview: proposed, not frozen, no approval for this record."""
    run = json.loads((RESEARCH / "c1_repro_run.json").read_text())
    cmp_ = json.loads((RESEARCH / "c1_repro_compare.json").read_text())
    if not all(r["identical"] for r in cmp_["rows"]):
        raise RuntimeError("the reproduction comparison did not pass")
    if sha(ROOT / "freeze_c1_reproduce.py") != cmp_["script_sha256"]:
        raise RuntimeError("freeze_c1_reproduce.py changed since the comparison")
    src = check_source(run)
    code = src["code"]
    guard = json.loads((RESEARCH / "c1_repro_guard.json").read_text())
    changed = [p for p, h in guard.items() if not Path(p).exists() or sha(p) != h]
    if changed:
        raise RuntimeError(f"guarded artifacts changed: {changed[:5]}")
    full = json.loads((MODELS / "C1_full" / "manifest.json").read_text())
    hist = check_history(full)
    n_checks = len(cmp_["rows"])
    repro_files = sorted(p for p in (MODELS / "C1_repro").iterdir() if p.is_file())
    repro_files += sorted(p for p in (MODELS / "C1" / "runs" / "repro").rglob("*") if p.is_file())
    preview = authorized is None
    archive = (f"{rel(ARCHIVE_PARENT)}/{ARCHIVE_PREFIX}<authorization date> (named at writing)" if preview
               else rel(ARCHIVE_PARENT / f"{ARCHIVE_PREFIX}{authorized}"))
    return {
        "record": ("PROPOSED C1 procedure freeze record (preview): not written, not authorized, not frozen"
                   if preview else "C1 procedure freeze (S8 stage 6)"),
        "record_version": 2,
        "built": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "status": {
            "fitting_procedure": "proposed (preview): not frozen" if preview else "frozen",
            "forward_models_2025": {"P0": "pending: not yet fitted", "C1": "pending: not yet fitted"},
            "s9_protocol": "not locked",
            "new_season_work": "not authorized by this record",
            "approvals": {"S8 full run and validation": "user, 2026-11-16",
                          "reproduction run": "user, 2026-11-16",
                          "this record": "not authorized (preview)" if preview else f"user, {authorized}"},
        },
        "specification": {
            "model": full["description"], "id": full["id"], "base": full["base"], "params": full["params"],
            "seasons_developed_on": full["seasons"],
            "documents": [file_rec(ROOT / f) for f in SPEC_FILES],
            "note": ("These documents are git-ignored. Copies, checked against the hashes above, are kept in the "
                     "source archive named below. Later edits to the live files (the S8 note's stage 6 entry; the "
                     "amendment, if adopted) change their hashes; the archived copies are the frozen versions."),
        },
        "source": {
            "reproduction_commit": src["repro_commit"],
            "reproduction_commit_note": "The commit C1's full run was reproduced under on 2026-11-16.",
            "freeze_source_commit": code["git_head"],
            "freeze_source_note": ("HEAD when this record was built. gamecalling/ is clean, identical to the "
                                   "reproduction commit (empty git diff) and at the same code hash."),
            "gamecalling_sha256": code["gamecalling_sha256"],
            "gamecalling_files": [file_rec(p) for p in sorted((ROOT / "gamecalling").glob("*.py"))],
            "scripts_outside_package": [{**file_rec(ROOT / s), "git": script_status(s)} for s in SCRIPTS],
            "source_archive": archive,
            "history": {
                "historical_full_run_code_sha256": hist["historical"],
                "historical_hash_scope": "one hash over all gamecalling/*.py files; no per-file hashes were stored",
                "criterion_2_code_sha256": hist["criterion2"],
                "criterion_2_record": rel(CRITERION2_RECORD),
                f"freeze_source_without_{HISTORICAL_DRIVER}_sha256": hist["without_driver"],
            },
            "disclosure": (
                f"The historical full run (data/models/C1_full, 2026-11-08) stored one code hash, "
                f"{hist['historical'][:12]}, over all gamecalling/*.py files; no per-file hashes were stored. Its "
                f"{HISTORICAL_DRIVER} was never committed and is not recoverable, so the historical source cannot be "
                f"reconstructed or compared file by file. With {HISTORICAL_DRIVER} left out, the freeze source "
                f"hashes to {hist['criterion2'][:12]}, the code hash stored by criterion 2's run: this establishes "
                f"agreement with criterion 2's source, not with the historical full run's source. The reproduction "
                f"under commit {REPRO_COMMIT} reproduced the compared outputs of the historical full run exactly "
                f"({n_checks} of {n_checks} checks, no tolerance): this establishes the full-run result under the "
                f"committed source. It does not recover the historical driver."),
        },
        "execution": {
            "commands": ["python3.12 freeze_c1_reproduce.py guard",
                         "S8_FULL_SMOKE=<scratchpad dir> python3.12 freeze_c1_reproduce.py run",
                         "python3.12 freeze_c1_reproduce.py run", "python3.12 freeze_c1_reproduce.py compare",
                         "python3.12 freeze_c1_record.py write <authorization date>",
                         "python3.12 freeze_c1_record.py verify"],
            "reproduction_provenance": run["provenance"],
            "workers": run["provenance"]["workers"], "math_threads_per_process": 1,
        },
        "inputs": full["inputs"],
        "fits": {"fit_ids": full["fit_ids"], "concentrations_and_lambdas": full["alphas"],
                 "policy_concentrations": full["policy_alphas"], "lambda_grid": full["params"]["lambdas"]},
        "seeds_and_rules": {
            "fold_seed": config.FOLD_SEED, "n_folds": config.N_FOLDS,
            "inner_fold_seed": "[20261030, season, code]; role fold 0-2, forward 3 (S6 note section 4.3)",
            "criterion_2_seeds": "[20261102, season, role_index, context_index, base_out] (S8 note section 12)",
            "action_unit": config.ACTION_UNIT, "arsenal_rules": config.ARSENAL_RULES,
            "sp_min_start_share": config.SP_MIN_START_SHARE, "solver_tolerance": config.SOLVER_TOL,
            "position_player_min_bat_pa": config.POSITION_PLAYER_MIN_BAT_PA,
            "dugout_flag": {"team": config.DUGOUT_CALLED_TEAM, "start": config.DUGOUT_CALLED_START, "season": 2025,
                            "note": "hard-coded to 2025 in loader.py; a year-specific flag is a permitted change"},
            "eligibility": "as in project_plan.md, 'Cross-fitting' and 'Locked pilot choices'",
        },
        "numerical_acceptance": {
            "rule": ("S6 note section 4.3: recompute objective and gradient; require finite values and "
                     "max|gradient| <= 1e-7; accept success or the specific L-BFGS-B ABNORMAL line-search "
                     "termination; stop otherwise; log every accepted abnormal termination."),
            "accepted_in_this_run": full["abnormal_log"],
            "scope": ("The rule is frozen. The terminations listed were accepted under it in this run. They are not "
                      "permission to accept a future failure that does not meet the rule."),
        },
        "reporting": {
            "primary": "D", "comparison": "A, with equal prominence", "diagnostic": "B",
            "headline_spread": "reported for both D and A",
            "decomposition": ("D = A + B. B = V_value(s) - Q_value(s, recommended) does not depend on the pitch "
                              "actually thrown once the situation and fitted models are fixed."),
            "approved": "user, 2026-11-16", "d_vs_a": "inconclusive unless the user decides otherwise",
        },
        "artifacts": {
            "historical": [{"path": rel(p), "sha256": h} for p, h in sorted(guard.items())],
            "historical_note": ("Existing artifacts as hashed before the reproduction and unchanged after it. Their "
                                "original manifests are preserved untouched."),
            "reproduced": [file_rec(p) for p in repro_files],
            "reproduced_note": f"Written by the reproduction run on 2026-11-16 under commit {REPRO_COMMIT}.",
        },
        "reproduction": {
            "record": file_rec(RESEARCH / "c1_repro_run.json"), "comparison": file_rec(RESEARCH / "c1_repro_compare.json"),
            "guard": file_rec(RESEARCH / "c1_repro_guard.json"),
            "result": f"{n_checks} of {n_checks} checks identical; no tolerance",
            "covers": "scores, the four prediction files, coverage, per-action values and the manifest fields compared",
            "does_not_cover": "the independent halves, the S1 runs and the validation stage, which were not rerun",
            "rerun_needed": ("No, while gamecalling/ stays at code hash 1000e503014a and freeze_c1_reproduce.py "
                             "matches the hash stored by the comparison; both are checked when the record is built."),
        },
        "permitted_changes": {
            "allowed": ["input paths", "documented season flags (for example a year-specific dugout flag)"],
            "condition": ("Each change must reproduce the 2024-2025 outputs under the same comparison contract "
                          "before it is applied to a new season."),
            "not_allowed": "any change to the model, tuning, eligibility, scoring or ranking specification",
        },
        "failure_handling": ("Data or numerical failures are reported and handled by rules written before the run. A "
                             "repair is never chosen by its effect on results. A rule that stops a run stays until "
                             "the user changes it."),
        "next_steps": [
            *(["The user's authorization to write this record."] if preview else []),
            ("Add the stage 6 entry to the S8 note, as a separate step after this record. It changes the live "
             "note's hash from the one recorded here; the archived copy is the frozen version, and the entry says so."),
            "Enter A1-A11 in project_plan.md only if the user adopts them; the archived copy keeps the version frozen here.",
            "A specification note for the all-2025 P0 and C1 fits, then the fits, then the forward-artifact record.",
            "Lock the S9 protocol. Locking it does not authorize new-season work.",
        ],
    }


def archived_files(rec: dict) -> dict:
    return {d["path"]: d["sha256"]
            for d in (*rec["specification"]["documents"], *rec["source"]["scripts_outside_package"])}


def stage_preview(directory: str):
    out = (Path(directory) / "C1_frozen.json").resolve()
    if out == RECORD.resolve():
        raise ValueError("a preview may not be written to the record's own path")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(), indent=1))
    print(f"preview written to {out}", flush=True)


def stage_write(date_text: str):
    authorized = check_date(date_text)
    existing = sorted(ARCHIVE_PARENT.glob(ARCHIVE_PREFIX + "*"))
    partial_record = RECORD.with_name(RECORD.name + ".partial")
    if RECORD.exists() or partial_record.exists() or existing:
        raise FileExistsError(f"a freeze record or archive exists ({RECORD.name}, {existing}): written once only")
    rec = build(authorized=authorized)
    archive = ARCHIVE_PARENT / f"{ARCHIVE_PREFIX}{authorized}"
    partial = archive.with_name(archive.name + ".partial")
    expected = archived_files(rec)
    partial.mkdir(parents=True)
    try:
        for path, digest in expected.items():
            dest = partial / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / path, dest)
            if sha(dest) != digest:
                raise RuntimeError(f"archived copy of {path} does not match the hash in the record")
    except BaseException:
        shutil.rmtree(partial)
        raise
    partial.rename(archive)
    rec["source"]["archive_verification"] = (f"{len(expected)} archived copies matched the hashes in this record "
                                             f"when written")
    text = json.dumps(rec, indent=1)
    partial_record.write_text(text)
    if json.loads(partial_record.read_text()) != rec:
        raise RuntimeError(f"the record did not read back intact; the archive {rel(archive)} exists without it")
    partial_record.rename(RECORD)
    print(f"freeze record written to {RECORD} (sha256 {sha(RECORD)})", flush=True)
    print(f"{len(expected)} sources archived and verified in {archive}", flush=True)
    print("next, as a separate step: the S8 note's stage 6 entry", flush=True)


def stage_verify():
    rec = json.loads(RECORD.read_text())
    archive = ROOT / rec["source"]["source_archive"]
    problems, drifted = [], []
    for path, digest in archived_files(rec).items():
        copy = archive / path
        if not copy.exists() or sha(copy) != digest:
            problems.append(f"archive: {path}")
        if not (ROOT / path).exists() or sha(ROOT / path) != digest:
            drifted.append(path)
    for a in rec["artifacts"]["historical"] + rec["artifacts"]["reproduced"]:
        if not (ROOT / a["path"]).exists() or sha(ROOT / a["path"]) != a["sha256"]:
            problems.append(f"artifact: {a['path']}")
    if package_hash() != rec["source"]["gamecalling_sha256"]:
        drifted.append("gamecalling/ (code hash)")
    print(f"record {RECORD} sha256 {sha(RECORD)}", flush=True)
    print(f"live files changed since the record (informational): {drifted or 'none'}", flush=True)
    if problems:
        sys.exit(f"VERIFY FAILED: {len(problems)} problems: {problems[:10]}")
    print("verify passed: every archived copy and every recorded artifact matches the record", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "preview":
        stage_preview(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == "write":
        stage_write(sys.argv[2])
    elif len(sys.argv) == 2 and sys.argv[1] == "verify":
        stage_verify()
    else:
        sys.exit(__doc__)
