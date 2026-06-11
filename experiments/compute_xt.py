"""
Compute xT per (game_id, team_id) using socceraction ExpectedThreat.
Fits on the full SPADL corpus (no LOSO needed — xT is a pitch value surface).
Output: outputs/metrics/xt_team_match.csv  (game_id, team_id, is_home, xt)
        outputs/metrics/xt_grid.json        (fitted xT surface)
"""

import numpy as np
import pandas as pd
from socceraction.xthreat import ExpectedThreat
import socceraction.spadl.config as spadlconfig

from squadhan.config import VAEP_OUTPUT_DIR, METRICS_DIR

GAMES_CSV = VAEP_OUTPUT_DIR / "games.csv"
SPADL_PATH = VAEP_OUTPUT_DIR / "spadl_all.parquet"
OUT_CSV = METRICS_DIR / "xt_team_match.csv"
OUT_GRID = METRICS_DIR / "xt_grid.json"


def _add_names(df: pd.DataFrame) -> pd.DataFrame:
    at = spadlconfig.actiontypes
    res = spadlconfig.results
    bp = spadlconfig.bodyparts
    df = df.copy()
    df["type_name"] = df["type_id"].apply(lambda i: at[i] if i < len(at) else "unknown")
    df["result_name"] = df["result_id"].apply(lambda i: res[i] if i < len(res) else "fail")
    df["bodypart_name"] = df["bodypart_id"].apply(lambda i: bp[i] if i < len(bp) else "foot")
    return df


def main():
    if OUT_CSV.exists():
        print(f"[skip] {OUT_CSV} already exists")
        return

    print("Loading SPADL parquet...")
    actions = pd.read_parquet(SPADL_PATH)
    actions = _add_names(actions)

    # xT fit requires finite coordinates — drop rows with NaN in spatial columns
    coord_cols = ["start_x", "start_y", "end_x", "end_y"]
    actions_fit = actions.dropna(subset=coord_cols).copy()
    print(f"Fitting xT on {len(actions_fit):,} actions (dropped {len(actions)-len(actions_fit):,} NaN rows)...")
    xt_model = ExpectedThreat(l=16, w=12)
    xt_model.fit(actions_fit)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    xt_model.save_model(str(OUT_GRID))
    print(f"Saved xT grid → {OUT_GRID}")

    print("Rating actions...")
    # socceraction rate() uses np.NaN (removed in NumPy 2.0) — replicate manually
    from socceraction.xthreat import get_successful_move_actions, _get_cell_indexes
    l_xt, w_xt = xt_model.l, xt_model.w
    grid = xt_model.xT
    ratings = np.full(len(actions), np.nan)
    move_actions = get_successful_move_actions(actions.reset_index())
    startxc, startyc = _get_cell_indexes(move_actions.start_x, move_actions.start_y, l_xt, w_xt)
    endxc, endyc = _get_cell_indexes(move_actions.end_x, move_actions.end_y, l_xt, w_xt)
    ratings[move_actions.index] = (
        grid[endyc.rsub(w_xt - 1), endxc] - grid[startyc.rsub(w_xt - 1), startxc]
    )
    actions = actions.copy()
    actions["xt"] = ratings

    # Sum xT per (game_id, team_id) — only keep positive contributions
    grp = actions.groupby(["game_id", "team_id"])["xt"].sum().reset_index()
    grp.rename(columns={"xt": "xt"}, inplace=True)

    games = pd.read_csv(GAMES_CSV)[["game_id", "home_team_id", "away_team_id"]]
    grp = grp.merge(games, on="game_id", how="left")
    grp["is_home"] = grp.apply(
        lambda r: 1 if r.team_id == r.home_team_id else (0 if r.team_id == r.away_team_id else -1),
        axis=1,
    )
    grp = grp[grp.is_home >= 0][["game_id", "team_id", "is_home", "xt"]]

    grp.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(grp)} rows → {OUT_CSV}")
    print(grp.describe())


if __name__ == "__main__":
    main()
