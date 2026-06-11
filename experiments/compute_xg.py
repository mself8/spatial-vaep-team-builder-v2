"""
Aggregate provider xG from raw event_data.json files per (game_id, team_id).
Output: outputs/metrics/xg_team_match.csv  (game_id, team_id, is_home, xg)
"""
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from squadhan.config import VAEP_OUTPUT_DIR, METRICS_DIR

COMP_DIR = {587: "KLEAGUE1", 588: "KLEAGUE2"}
RAW_DATA = ROOT / "raw-data"
GAMES_CSV = VAEP_OUTPUT_DIR / "games.csv"
OUT = METRICS_DIR / "xg_team_match.csv"


def main():
    if OUT.exists():
        print(f"[skip] {OUT} already exists")
        return

    games = pd.read_csv(GAMES_CSV)
    records = []

    for _, row in games.iterrows():
        gid = int(row.game_id)
        comp_dir = COMP_DIR.get(int(row.competition_id))
        if comp_dir is None:
            continue
        season = int(row.season)
        path = RAW_DATA / comp_dir / str(season) / "match" / str(gid) / "event_data.json"
        if not path.exists():
            continue

        with open(path) as f:
            raw = json.load(f)
        events = raw.get("result", raw) if isinstance(raw, dict) else raw

        xg_by_team: dict[int, float] = {}
        for e in events:
            if "xg" in e:
                tid = int(e["team_id"])
                xg_by_team[tid] = xg_by_team.get(tid, 0.0) + float(e["xg"])

        home_id = int(row.home_team_id)
        away_id = int(row.away_team_id)
        for tid, xg_val in xg_by_team.items():
            is_home = 1 if tid == home_id else (0 if tid == away_id else -1)
            records.append({"game_id": gid, "team_id": tid, "is_home": is_home, "xg": xg_val})

    df = pd.DataFrame(records)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} rows → {OUT}")
    print(df.describe())


if __name__ == "__main__":
    main()
