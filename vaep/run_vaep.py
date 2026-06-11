"""
End-to-end K-League VAEP pipeline script

Running this script performs, in order:
  1. Load the match list for all seasons from raw-data/
  2. Convert Bepro events → SPADL actions (~15 min)
  3. Save player/team metadata (teams.csv, players.csv, games.csv)
  4. Leave-One-Season-Out OOF VAEP training (~10 min)
  5. Save results (output/vaep_oof.parquet, vaep_oof_metrics.json)

Usage (from the repository root):
  pip install -r requirements.txt
  python -m vaep.run_vaep

Prerequisites:
  - The Bepro raw JSON data must be present in the raw-data/ folder.
  - Folder layout: raw-data/{KLEAGUE1|KLEAGUE2}/{season}/match/{game_id}/
"""

from pathlib import Path

from vaep.core import (
    build_loader,
    load_all_games,
    convert_games_to_spadl,
    load_players_and_teams,
    run_oof_vaep,
)

# ---------------------------------------------------------------------------
# Path configuration
#
# HERE: folder containing this script (vaep/)
# RAW_DATA: K-League raw JSON data folder (raw-data/)
# OUTPUT: results folder (vaep/output/)
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
RAW_DATA = HERE.parent / "raw-data"
OUTPUT = HERE / "output"

print("=== K-League VAEP OOF training ===")
print(f"Raw data: {RAW_DATA}")
print(f"Output:   {OUTPUT}")

# ---------------------------------------------------------------------------
# Step 1: load match metadata
#
# Fetch the match list for all KLEAGUE1 and KLEAGUE2 seasons via BeproLoader.
# The returned DataFrame includes game_id, home/away_team_id, season, competition_name.
# ---------------------------------------------------------------------------
loader = build_loader(RAW_DATA)
games = load_all_games(loader, competition_names=["KLEAGUE1", "KLEAGUE2"])
print(f"\nTotal matches: {len(games):,}")
print(games.groupby(["season"]).size().to_string())

# ---------------------------------------------------------------------------
# Step 2: SPADL conversion
#
# Convert each match's Bepro events (carries, passes, shots, ...) to the
# standard SPADL format.
# Matches that fail to convert are dropped from the games DataFrame automatically.
# Takes 15-20 minutes for ~2,300 matches.
# ---------------------------------------------------------------------------
print("\n=== SPADL conversion ===")
games, actions_dict = convert_games_to_spadl(loader, games, verbose=True)
print(f"Converted: {len(actions_dict):,} matches")

import pandas as pd
pd.concat(
    [df.assign(game_id=gid) for gid, df in actions_dict.items()],
    ignore_index=True,
).to_parquet(OUTPUT / "spadl_all.parquet", index=False)
print(f"SPADL cache saved: {OUTPUT / 'spadl_all.parquet'}")

# ---------------------------------------------------------------------------
# Step 3: save player/team info
#
# Used in notebook analyses to map player_id → player name.
# Saving CSVs to the output/ folder avoids recomputing every time.
# ---------------------------------------------------------------------------
print("\n=== Loading player/team info ===")
teams_df, players_df = load_players_and_teams(loader, games)
teams_df.to_csv(OUTPUT / "teams.csv", index=False)
players_df.to_csv(OUTPUT / "players.csv", index=False)
games.to_csv(OUTPUT / "games.csv", index=False)
print(f"Teams: {len(teams_df):,}, players: {len(players_df):,}")

# ---------------------------------------------------------------------------
# Step 4: OOF VAEP training and saving
#
# Cycles through 5 folds in Leave-One-Season-Out fashion.
# Each fold trains two XGBoost models (scoring/conceding) and predicts
# VAEP values for every action of the held-out season.
#
# Output files:
#   output/vaep_oof.parquet — unbiased VAEP values for all seasons (~3.5M rows)
#   output/vaep_oof_metrics.json — per-fold AUC metrics
# ---------------------------------------------------------------------------
print("\n=== OOF VAEP training ===")
run_oof_vaep(games, actions_dict, OUTPUT)
