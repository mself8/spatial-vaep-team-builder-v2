"""Shot-level AUC of the provider's xG (basis for "shot-level AUC 0.80" in the paper's §Implementation Details).

Collects shot events carrying xg from the raw event_data.json of every K League 1 & 2
match 2021--2025 and computes the AUC for the binary outcome=='Goal'. (Run 2026-06-07: 2,283 matches / 48,778 shots /
5,227 goals (10.7%) → AUC 0.7975. Identical when excluding the one match (26228) without a squad graph.)

Run: python -m experiments.xg_shot_auc  (from the repository root)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]

from squadhan.config import VAEP_OUTPUT_DIR, VALID_COMPETITION_IDS

COMP_DIR = {587: "KLEAGUE1", 588: "KLEAGUE2"}


def main():
    g = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv")
    g = g[g.competition_id.isin(VALID_COMPETITION_IDS) & g.season.between(2021, 2025)]
    xs, ys, miss = [], [], 0
    for _, r in g.iterrows():
        p = (ROOT / "raw-data" / COMP_DIR.get(int(r.competition_id), "?") /
             str(int(r.season)) / "match" / str(int(r.game_id)) / "event_data.json")
        if not p.exists():
            miss += 1
            continue
        raw = json.load(open(p))
        events = raw.get("result", raw) if isinstance(raw, dict) else raw
        for e in events:
            if "xg" not in e:
                continue
            shot = [t for t in e.get("event_types", []) if t.get("event_type") == "Shot"]
            if not shot:
                continue
            xs.append(float(e["xg"]))
            ys.append(1 if shot[0].get("outcome") == "Goal" else 0)
    xs, ys = np.array(xs), np.array(ys)
    print(f"matches {len(g)} (missing {miss}) | shots {len(xs)} | goals {int(ys.sum())} ({ys.mean():.3f})")
    print(f"shot-level AUC = {roc_auc_score(ys, xs):.4f}")


if __name__ == "__main__":
    main()
