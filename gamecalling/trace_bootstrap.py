"""Trace the bootstrap concentration collapse on fixed replicates (diagnostic).

For a few fixed replicates of one season, compare the original data, the plain
bootstrap and the experimental first-copy bootstrap at each stage: cell counts,
pooling concentrations, prior (league) means, fitted pitcher-cell probabilities,
scores and ranking variance components.

Run with: python3.12 -m gamecalling.trace_bootstrap   (writes reports/bootstrap_trace.md)
"""

import numpy as np
import pandas as pd

from gamecalling import arsenal, bootstrap, config, models, pooling, ranking, scoring
from gamecalling.phase1 import _md

SEASON = 2024
REPS = (0, 1, 2)
REPORT = config.REPORTS_DIR / "bootstrap_trace.md"


def stage_summary(pitches, label) -> list:
    rows = []
    for f in range(config.N_FOLDS):
        p = pitches[(pitches["season"] == SEASON) & (pitches["fold"] == f)]
        tr = models.outcome_training(p)
        m = models.fit_pooled(tr, models.OUTCOME_LEVELS, models.N_OUTCOMES)
        c = models.fit_contact_model(p)
        cells = m.cell_n
        N = cells.sum(axis=1)
        q = cells.index.to_frame(index=False)
        post = m.predict(q)
        league = m.league_mean(q)
        raw = cells.to_numpy() / N.to_numpy()[:, None]
        wN = N.to_numpy()
        rows.append(dict(
            stage=label, fold=f, pitches_weighted=float(N.sum()), cells=len(cells),
            median_cell_n=float(np.median(wN)),
            outcome_alpha_league=m.alpha_league, outcome_alpha_pitcher=m.alpha_cell,
            contact_alpha_league=c.alpha_league, contact_alpha_pitcher=c.alpha_cell,
            # Spread of pitcher cells around their league mean (count-weighted RMS, all outcomes).
            raw_spread=float(np.sqrt(np.average(((raw - league) ** 2).sum(1), weights=wN))),
            fitted_spread=float(np.sqrt(np.average(((post - league) ** 2).sum(1), weights=wN))),
            league_mean_ss=float(league[:, 2].mean()),
        ))
    return rows


def scores_and_ranking(pitches, pas):
    support = arsenal.support_table(pitches)
    marked = arsenal.mark_arsenal(pitches, arsenal.eligible_sets(support))
    run = scoring.run_pilot(marked, pas, support, seasons=(SEASON,))
    s = run.scores
    out = {}
    for col in ("D", "A"):
        f = ranking.fit(s, col)
        out[f"sd_catcher_{col}"] = float(np.sqrt(f.variances["catcher"]))
    w = s["w"].to_numpy() if "w" in s else np.ones(len(s))
    for col in ("D", "A"):
        mu = np.average(s[col], weights=w)
        out[f"pitch_sd_{col}"] = float(np.sqrt(np.average((s[col] - mu) ** 2, weights=w)))
        out[f"mean_{col}_per100"] = float(mu * 100)
    return out


def synthetic_demo(seed=0, n_cells=3000, per_cell=4, reps=3) -> pd.DataFrame:
    """Cells with identical true probabilities (no heterogeneity), each pitch from a different game.

    Resampling games multiplies each pitch by a Poisson(1)-like count. The plain
    replicate's cells then deviate from the league mix far more than multinomial
    noise at their counts allows, so the marginal likelihood reads heterogeneity.
    """
    rng = np.random.default_rng(seed)
    p = np.array([0.5, 0.3, 0.2])
    y = rng.choice(3, size=(n_cells, per_cell), p=p)

    def counts(mult):
        return np.stack([((y == k) * mult).sum(1) for k in range(3)], 1).astype(float)

    rows = [dict(sample="original", alpha=pooling.fit_alpha(counts(np.ones_like(y)), p))]
    for r in range(reps):
        m = rng.poisson(1.0, size=y.shape)
        b, fc = counts(m), counts(m > 0)
        keep = b.sum(1) > 0
        pm = b.sum(0) / b.sum()
        rows.append(dict(sample=f"plain rep {r}", alpha=pooling.fit_alpha(b[keep], pm)))
        rows.append(dict(sample=f"firstcopy rep {r}", alpha=pooling.fit_alpha(fc[keep], pm)))
    return pd.DataFrame(rows)


def main():
    pitches, pas = bootstrap._load()
    stage_rows, score_rows = [], []
    stage_rows += stage_summary(pitches, "original")
    score_rows.append(dict(stage="original", rep=-1, **scores_and_ranking(
        pitches[pitches["season"] == SEASON], pas[pas["season"] == SEASON])))
    for rep in REPS:
        for method in bootstrap.METHODS:
            rng = np.random.default_rng([bootstrap.BOOT_SEED, rep])
            # Same draw as the real replicate: draw both seasons in order, keep this season.
            draws = {s: bootstrap.resample_season_weights(pitches, pas, s, rng, method) for s in config.SEASONS}
            pw, aw = draws[SEASON]
            stage_rows += stage_summary(pw, f"{method} rep {rep}")
            score_rows.append(dict(stage=method, rep=rep, **scores_and_ranking(pw, aw)))
    st = pd.DataFrame(stage_rows)
    sc = pd.DataFrame(score_rows)
    demo = synthetic_demo()
    REPORT.write_text("\n".join([
        "# Bootstrap concentration trace", "",
        "## Synthetic cells with no true heterogeneity", "",
        "3,000 cells with identical true probabilities (0.5, 0.3, 0.2), 4 pitches each, every pitch from a different "
        "game. The correct concentration is infinite (the upper bound is 1e6).", "", _md(demo, "{:.1f}"), "",
        "In the plain replicate the cells' compositions are the observed compositions resampled, so they scatter "
        "around the league mix by the original sampling noise plus the resampling noise, while their counts are "
        "about the same. The marginal likelihood reads the excess as between-cell heterogeneity. This is the known "
        "difficulty of the nonparametric bootstrap for empirical-Bayes hyperparameters (the replicate's 'truth' is the "
        "unshrunk empirical distribution); a parametric bootstrap from the fitted prior (Laird and Louis, 1987) is the "
        "standard alternative.", "",
        "## Real data", "",
        f"Season {SEASON}; bootstrap replicates {REPS} (seed {bootstrap.BOOT_SEED}, the same draws as the stored "
        "replicates), each with the plain (reference) and first-copy (experimental) methods. Outcome model per role "
        "fold. `raw_spread` / `fitted_spread`: count-weighted RMS distance of pitcher-cell raw / fitted outcome "
        "probabilities from their league means. `league_mean_ss`: mean league swinging-strike probability over cells.", "",
        _md(st, "{:.4f}"), "",
        "Scores and ranking variance components (catcher SD in runs per 100 pitches; per-pitch SD and mean of the "
        "score, weighted):", "", _md(sc, "{:.4f}"), ""]))
    print(st.groupby(st["stage"].str.split(" ").str[0])[["outcome_alpha_pitcher", "contact_alpha_pitcher",
                                                         "raw_spread", "fitted_spread"]].mean())
    print(sc)


if __name__ == "__main__":
    main()
