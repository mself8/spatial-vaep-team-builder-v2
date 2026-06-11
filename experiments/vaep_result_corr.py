"""Re-check the VAEP ↔ match-result correlation (basis for tab:vaep-result in the paper).

Prints the Pearson r between team total VAEP (reusing train_e2e_vaep._build_yvaep_map)
and that match's team result (match points / goal difference / goals), plus the mean team VAEP by win/draw/loss.
→ Reproduces the r=0.69/0.83/0.57 numbers of Table 1 in the paper (tab:vaep-result).

Run (from the repository root):
  python -m experiments.vaep_result_corr
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from squadhan.config import VAEP_OUTPUT_DIR, VALID_COMPETITION_IDS
from squadhan.train_e2e_vaep import _build_yvaep_map as build_yvaep_map


def main():
    ymap = build_yvaep_map()
    g = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv")
    g = g[g["competition_id"].isin(VALID_COMPETITION_IDS) & g["season"].between(2021, 2025)].copy()
    g["game_id"] = g["game_id"].astype(int)

    vaep, points, goaldiff, goals = [], [], [], []
    for _, r in g.iterrows():
        gid = int(r["game_id"])
        hs, as_ = int(r["home_score"]), int(r["away_score"])
        for is_home, (gf, ga) in ((1, (hs, as_)), (0, (as_, hs))):
            if (gid, is_home) not in ymap:
                continue
            vaep.append(ymap[(gid, is_home)])
            points.append(3 if gf > ga else (1 if gf == ga else 0))
            goaldiff.append(gf - ga)
            goals.append(gf)

    vaep = np.asarray(vaep, float)
    points = np.asarray(points, float)
    goaldiff = np.asarray(goaldiff, float)
    goals = np.asarray(goals, float)
    n = len(vaep)

    print(f"team-matches: {n}")
    for name, arr in (("points", points), ("goal_diff", goaldiff), ("goals", goals)):
        r, p = pearsonr(vaep, arr)
        print(f"  Pearson r (team_VAEP, {name:9s}) = {r:+.3f}   (p={p:.1e})")

    print("avg team VAEP by result:")
    for label, pts in (("win ", 3), ("draw", 1), ("loss", 0)):
        m = vaep[points == pts]
        print(f"  {label} (pts={pts}): mean={m.mean():+.3f}  n={len(m)}  ({len(m)/n*100:.1f}%)")


if __name__ == "__main__":
    main()
