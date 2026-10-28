"""Connection map and bridge-removal sensitivity (project_plan.md, Phase 3: Connection map).

Per season, on the pilot scores:
1. Link qualifying catchers (>= 1,000 pitches) through shared pitchers: a
   catcher-pitcher pair counts if it has >= 100 pitches together. Find connected
   groups and flag catchers outside the main group.
2. Bridging pitchers: >= 100 pitches for 2+ teams. Each bridge's size is the
   harmonic mean 2ab/(a+b) of his pitches for his two largest teams (descriptive).
3. Bridge removal: leave out each bridging pitcher in turn and refit the ranking
   models (random and pitcher-fixed effects) for D and A. Flag removals that
   disconnect the network. When it stays connected, both fits are centered on the
   average of the main group's catchers; when it splits, each piece is centered on
   its own average and only within-piece differences are compared. Catchers left
   outside the main group are flagged: in the random-effects model their
   placement rests only on the model's assumptions.
4. Figure: catchers as nodes colored by team, edges weighted by shared pitches.

Run with: python3.12 -m gamecalling.network   (writes reports/network_pilot.md, CSVs, PNGs)
"""

import networkx as nx
import numpy as np
import pandas as pd

from gamecalling import config, ranking
from gamecalling.phase1 import SCORES_PARQUET, _md

QUALIFY = 1000
MIN_PAIR = 100
REPORT = config.REPORTS_DIR / "network_pilot.md"


def catcher_graph(scores: pd.DataFrame, exclude_pitcher=None, qualifying=None) -> nx.Graph:
    """`qualifying` fixes the catcher set (default: >= QUALIFY pitches in `scores`)."""
    s = scores if exclude_pitcher is None else scores[scores["pitcher"] != exclude_pitcher]
    n = s.groupby("catcher").size()
    qual = set(n[n >= QUALIFY].index) if qualifying is None else set(qualifying)
    pairs = s.groupby(["catcher", "pitcher"]).size()
    pairs = pairs[pairs >= MIN_PAIR].reset_index(name="n")
    pairs = pairs[pairs["catcher"].isin(qual)]
    G = nx.Graph()
    G.add_nodes_from(qual)
    for _, g in pairs.groupby("pitcher"):
        cs = g.set_index("catcher")["n"]
        for i, a in enumerate(cs.index):
            for b in cs.index[i + 1:]:
                w = 2 * cs[a] * cs[b] / (cs[a] + cs[b])
                if G.has_edge(a, b):
                    G[a][b]["weight"] += w
                    G[a][b]["pitchers"] += 1
                else:
                    G.add_edge(a, b, weight=w, pitchers=1)
    return G


def components(G: nx.Graph) -> list[set]:
    return sorted(nx.connected_components(G), key=len, reverse=True)


def bridging_pitchers(scores: pd.DataFrame) -> pd.DataFrame:
    by = scores.groupby(["pitcher", "fielding_team"]).size().rename("n").reset_index()
    by = by[by["n"] >= MIN_PAIR]
    rows = []
    for p, g in by.groupby("pitcher"):
        if len(g) < 2:
            continue
        a, b = g["n"].nlargest(2).to_numpy()
        rows.append(dict(pitcher=p, teams=len(g), team_list=",".join(sorted(g["fielding_team"])),
                         pitches=int(g["n"].sum()), harmonic_size=2 * a * b / (a + b)))
    return pd.DataFrame(rows).sort_values("harmonic_size", ascending=False).reset_index(drop=True)


def _centered(effects: pd.Series, groups: list[set]) -> pd.Series:
    out = pd.Series(np.nan, index=effects.index)
    for g in groups:
        idx = [c for c in g if c in effects.index]
        out[idx] = effects[idx] - effects[idx].mean()
    return out


def bridge_removal(scores: pd.DataFrame, bridges: pd.DataFrame, score_cols=("D", "A")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per removal: connectivity and the largest identifiable change; per catcher: his largest change."""
    G0 = catcher_graph(scores)
    main0 = components(G0)[0]
    base = {}
    for col in score_cols:
        for fixed in (False, True):
            f = ranking.fit(scores, col, pitcher_fixed=fixed)
            base[col, fixed] = f.catchers.set_index("catcher")["effect"]
    removal_rows, per_catcher = [], []
    for p in bridges["pitcher"]:
        G = catcher_graph(scores, exclude_pitcher=p)
        comps = components(G)
        connected = len(comps[0]) == len(main0) and comps[0] == main0
        groups = [main0] if connected else [c for c in comps if len(c & main0) > 0 and len(c) > 1]
        sub = scores[scores["pitcher"] != p]
        row = dict(pitcher=p, connected=connected, pieces=len([c for c in comps if c & main0]),
                   main_group_size=len(comps[0] & main0), left_main=len(main0 - comps[0]))
        for col in score_cols:
            for fixed in (False, True):
                f = ranking.fit(sub, col, pitcher_fixed=fixed)
                new = f.catchers.set_index("catcher")["effect"]
                a = _centered(base[col, fixed].reindex(list(main0)), groups)
                b = _centered(new.reindex(list(main0)), groups)
                d = (b - a).abs().dropna()
                label = f"{col}_{'fixed' if fixed else 'random'}"
                row[f"max_change_{label}"] = d.max()
                row[f"catcher_{label}"] = d.idxmax()
                per_catcher.append(pd.DataFrame({"catcher": d.index, "pitcher_removed": p, "model": label,
                                                 "change": d.to_numpy()}))
        removal_rows.append(row)
    pc = pd.concat(per_catcher, ignore_index=True)
    largest = pc.groupby(["catcher", "model"])["change"].max().unstack()
    largest.columns = [f"largest_change_{c}" for c in largest.columns]
    return pd.DataFrame(removal_rows), largest.reset_index()


def draw(G: nx.Graph, teams: pd.Series, path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pos = nx.spring_layout(G, weight="weight", seed=20261028, k=0.6)
    team_list = sorted(set(teams.reindex(list(G.nodes)).dropna()))
    cmap = plt.get_cmap("tab20", max(len(team_list), 1))
    colors = [cmap(team_list.index(teams.get(n)) % 20) if teams.get(n) in team_list else "grey" for n in G.nodes]
    widths = [0.3 + 2.5 * G[a][b]["weight"] / max(d["weight"] for *_, d in G.edges(data=True)) for a, b in G.edges]
    fig, ax = plt.subplots(figsize=(11, 9))
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.35, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=90, ax=ax, edgecolors="black", linewidths=0.3)
    # 30 teams outrun any qualitative palette, so each node also carries its team code.
    nx.draw_networkx_labels(G, {n: (x, y + 0.025) for n, (x, y) in pos.items()},
                            labels={n: teams.get(n, "?") for n in G.nodes}, font_size=5, ax=ax)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    scores = pd.read_parquet(SCORES_PARQUET)
    out = ["# Connection map and bridge removal (pilot)", "",
           f"Qualifying catchers: >= {QUALIFY} pitches. A catcher-pitcher pair links catchers if it has >= {MIN_PAIR} "
           f"pitches. Bridging pitchers: >= {MIN_PAIR} pitches for 2+ teams; size = harmonic mean 2ab/(a+b) of his two "
           "largest teams' pitches (descriptive only). Changes are in runs per 100 pitches (cost scale; the sign "
           "flip does not change absolute changes), centered on the main group's average or, after a split, on each "
           "piece's average.", ""]
    validation = []
    for season, s in scores.groupby("season"):
        G = catcher_graph(s)
        comps = components(G)
        teams = s.groupby(["catcher", "fielding_team"]).size().reset_index().sort_values(0).groupby("catcher").tail(1)
        teams = teams.set_index("catcher")["fielding_team"]
        draw(G, teams, config.REPORTS_DIR / f"network_{season}.png", f"Catcher connection map {season}")
        outside = [c for comp in comps[1:] for c in comp]
        bridges = bridging_pitchers(s)
        rem, largest = bridge_removal(s, bridges)
        bridges.merge(rem, on="pitcher").to_csv(config.REPORTS_DIR / f"bridge_removal_{season}.csv", index=False)
        fits = {(c, f): ranking.fit(s, c, pitcher_fixed=f) for c in ("D", "A") for f in (False, True)}
        v = pd.DataFrame({"catcher": sorted(G.nodes)})
        v["season"] = season
        v["team"] = v["catcher"].map(teams)
        v["in_main_group"] = v["catcher"].isin(comps[0])
        for (c, f), fit in fits.items():
            v[f"{c}_{'fixed' if f else 'random'}_saved"] = -v["catcher"].map(fit.catchers.set_index("catcher")["effect"])
        v = v.merge(largest, on="catcher", how="left")
        validation.append(v)
        out += [f"## {season}", "",
                f"- Qualifying catchers: {G.number_of_nodes()}; connected groups: {len(comps)} "
                f"(sizes {', '.join(str(len(c)) for c in comps)}). Outside the main group: "
                f"{', '.join(map(str, outside)) if outside else 'none'} ({', '.join(teams.reindex(outside).fillna('?')) if outside else ''}).",
                f"- Main group share: {len(comps[0]) / G.number_of_nodes():.1%} (go/no-go criterion 6 needs >= 90%).",
                f"- Bridging pitchers: {len(bridges)}. Removals that disconnect the network: {int((~rem['connected']).sum())}.",
                f"- Largest identifiable change from one removal (D random / D fixed / A random / A fixed): "
                f"{rem['max_change_D_random'].max():.4f} / {rem['max_change_D_fixed'].max():.4f} / "
                f"{rem['max_change_A_random'].max():.4f} / {rem['max_change_A_fixed'].max():.4f}; median over removals "
                f"{rem['max_change_D_random'].median():.4f} / {rem['max_change_D_fixed'].median():.4f} / "
                f"{rem['max_change_A_random'].median():.4f} / {rem['max_change_A_fixed'].median():.4f}.",
                f"- Figure: `reports/network_{season}.png`. Per-removal table: `reports/bridge_removal_{season}.csv`.", "",
                "Ten largest bridges:", "", _md(bridges.merge(rem, on="pitcher").head(10)[
                    ["pitcher", "team_list", "pitches", "harmonic_size", "connected", "max_change_D_random",
                     "max_change_D_fixed", "max_change_A_random"]], "{:.4f}"), ""]
        if (~rem["connected"]).any():
            out += ["Disconnecting removals:", "", _md(rem[~rem["connected"]][["pitcher", "pieces", "left_main",
                                                                               "max_change_D_random", "max_change_D_fixed"]], "{:.4f}"), ""]
    val = pd.concat(validation, ignore_index=True)
    val.to_csv(config.REPORTS_DIR / "validation_table_pilot.csv", index=False)
    out += ["## Validation table", "",
            "Saved to `reports/validation_table_pilot.csv`: per catcher and season, random- and fixed-effects scores "
            "(runs saved per 100 pitches) for D and A and the largest identifiable change from removing one bridge. "
            "The interval column is left out until the uncertainty procedure is validated.", ""]
    REPORT.write_text("\n".join(out))
    print(f"report in {REPORT}")


if __name__ == "__main__":
    main()
