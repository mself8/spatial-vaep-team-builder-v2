"""Minimum-minutes eligibility filter — inference-time mask sweep (free check, no retraining).

Hypothesis: the argmax selector exploits the profile variance of low-minutes (small-sample)
players (winner's curse) → SelAcc drops vs the coach. At inference, grant top-k eligibility
only to candidates with "minutes ≥ thr over the feature window (all of 21-25)" and measure how SelAcc/Δv̂ move per threshold.

- Uses the existing stage2 checkpoints as-is (no retraining/rebuilding)
- thr=0 disables the mask → must match train_e2e_vaep's test_cv rows (sanity)
- When the pool falls short (eligible GK<1 / OF<10), relax to top minutes and report the rate

Usage:
  python -m experiments.min_minutes_eval --fold 0                 # default thr sweep
  python -m experiments.min_minutes_eval --fold 0 --thr 0,450,900,1350
"""
import argparse
import os
import random

import numpy as np
import pandas as pd
import torch

# Pre-set env so module globals initialize with the same config as the evaluation target (main run: _gksel_sc_lc1_diff)
os.environ.setdefault("EDGE_SCALAR", "1")
os.environ.setdefault("GK_SELECT", "1")
os.environ.setdefault("VAEP_DIFF", "1")

from squadhan import train_e2e_vaep as T  # noqa: E402  (env must be set before import)
from squadhan.build_squad_dataset import SQUAD_GRAPHS_DIR  # noqa: E402
from squadhan.config import (CHECKPOINTS_DIR, DROPOUT, HIDDEN_CHANNELS, METRICS_DIR,  # noqa: E402
                             NODE_DIM, NUM_HEADS, NUM_LAYERS, VAEP_OUTPUT_DIR,
                             VALID_COMPETITION_IDS)
from squadhan.e2e_model_vaep import E2ELineupOptimizerVAEP  # noqa: E402


def build_diff_ymap() -> dict:
    """Same as the VAEP_DIFF transform in train_e2e_vaep.main()."""
    ymap = T._build_yvaep_map()
    dmap = {}
    for (gid, ih), val in ymap.items():
        opp = ymap.get((gid, 1 - ih))
        if opp is not None:
            dmap[(gid, ih)] = val - opp
    return dmap


def fold_split(test_season: int, seed: int, ymap: dict):
    """Replicates _load_samples_fold's triple split logic (without loading files) → test list + train mu/sd."""
    games = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv")
    games = games[games["competition_id"].isin(VALID_COMPETITION_IDS)].copy()
    games["season"] = games["season"].astype(int)
    season_map = dict(zip(games["game_id"].astype(int), games["season"]))

    test_t, rest_t = [], []
    for gid, season in season_map.items():
        for side in ("home", "away"):
            p = SQUAD_GRAPHS_DIR / f"match_{gid}_{side}.pt"
            if not p.exists():
                continue
            triple = (p, 1 if side == "home" else 0, gid)
            (test_t if season == test_season else rest_t).append(triple)

    rng = random.Random(seed)
    rng.shuffle(rest_t)
    n_val = int(len(rest_t) * T.VAL_RATIO_LOSO)
    train_t = rest_t[n_val:]
    ys = np.array([ymap[(g, ih)] for _, ih, g in train_t if (g, ih) in ymap],
                  dtype=np.float64)
    return test_t, float(ys.mean()), float(ys.std() + 1e-8)


def load_test(test_t, gkids: set, ymap: dict):
    """Same preprocessing as _load_samples_fold._load (EDGE_SCALAR reduction + is_gk attachment)."""
    out = []
    for p, is_home, gid in test_t:
        if (gid, is_home) not in ymap:
            continue
        d = torch.load(p, weights_only=False)
        for et in d.edge_types:
            d[et].edge_attr = d[et].edge_attr.sum(1, keepdim=True)
        pid = d["our_squad"].player_ids
        d["our_squad"].is_gk = torch.tensor(
            [int(x) in gkids for x in pid.tolist()], dtype=torch.bool)
        out.append(d)
    return out


def elig_masks(data, get_mins, thr: float):
    """(gk_elig, of_elig, relaxed) in pool order. Relaxed to top minutes when eligibility falls short.
    get_mins: player_id -> reference minutes (total or that season)."""
    pid = data["our_squad"].player_ids
    mins = torch.tensor([get_mins(int(p)) for p in pid.tolist()])
    is_gk = data["our_squad"].is_gk.view(-1).bool()
    elig = mins >= thr
    gk_e, of_e = elig[is_gk].clone(), elig[~is_gk].clone()
    relaxed = 0
    if int(gk_e.sum()) < 1:
        gk_e[mins[is_gk].argmax()] = True
        relaxed = 1
    if int(of_e.sum()) < 10:
        om = mins[~is_gk]
        of_e[om.topk(min(10, om.numel())).indices] = True
        relaxed = 1
    return gk_e, of_e, relaxed


@torch.no_grad()
def selected_ids(model, data, gk_elig, of_elig):
    """Deterministic selection under masks: set of 10 OF ids + selected GK id (extends train._selected_of_ids)."""
    our_emb, opp_emb = model.encoder(data)
    opp_ctx = opp_emb.mean(dim=0)
    pid = data["our_squad"].player_ids
    is_gk = data["our_squad"].is_gk.view(-1).bool()
    _, _, gidx = model.gk_selector(our_emb[is_gk], opp_ctx, k=1, training=False, elig=gk_elig)
    _, _, oidx = model.of_selector(our_emb[~is_gk], opp_ctx, k=10, training=False, elig=of_elig)
    gk_id = int(pid[is_gk][gidx].item())
    of_ids = set(pid[~is_gk].cpu().numpy()[oidx.cpu().numpy()].tolist())
    return gk_id, of_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--thr", type=str, default="0,450,900,1350")
    ap.add_argument("--ckpt-tag", type=str, default="_gksel_sc_lc1_diff")
    ap.add_argument("--window", choices=["total", "season"], default="total",
                    help="minutes aggregation window: 21-25 total vs the match's season")
    ap.add_argument("--device", type=str, default="cpu")  # avoid occupying the ablation GPU
    args = ap.parse_args()
    thrs = [float(x) for x in args.thr.split(",")]
    device = torch.device(args.device)

    print("building ymap(diff) …")
    ymap = build_diff_ymap()
    gkids = T._build_gkids()
    from squadhan.build_dataset import _ID_TO_ALL_IDS  # merge duplicate player IDs (consistent with features)
    players = pd.read_csv(VAEP_OUTPUT_DIR / "players.csv")
    gm = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv")[["game_id", "season"]]
    season_map = {int(g): int(s) for g, s in zip(gm["game_id"], gm["season"])}
    if args.window == "total":
        raw = {int(k): float(v) for k, v in
               players.groupby("player_id")["minutes_played"].sum().items()}
        minutes = {p: sum(raw.get(i, 0.0) for i in _ID_TO_ALL_IDS.get(p, frozenset([p])))
                   for p in raw}
    else:  # season: keyed by (player_id, season) — summed over merged IDs
        players = players.merge(gm, on="game_id", how="left")
        raw = {(int(p), int(s)): float(v) for (p, s), v in
               players.groupby(["player_id", "season"])["minutes_played"].sum().items()}
        minutes = {(p, s): sum(raw.get((i, s), 0.0)
                               for i in _ID_TO_ALL_IDS.get(p, frozenset([p])))
                   for (p, s) in raw}

    k = args.fold
    ts = T.SEASONS[k]
    test_t, mu, sd = fold_split(ts, T.SEED + k, ymap)
    print(f"fold {k} (test {ts}): mu={mu:.3f} sd={sd:.3f}")
    test_s = load_test(test_t, gkids, ymap)
    print(f"test samples: {len(test_s)}")

    ckpt = CHECKPOINTS_DIR / f"e2e_vaep_scalar{args.ckpt_tag}_stage2_fold{k}.pt"
    model = E2ELineupOptimizerVAEP(
        node_dim=NODE_DIM, edge_dim=1, hidden=HIDDEN_CHANNELS,
        n_heads=NUM_HEADS, n_layers=NUM_LAYERS, dropout=DROPOUT,
        gk_select=True, no_gnn=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
    model.eval()

    acc = {t: {"m": [], "c": [], "smatch": 0, "stot": 0, "gk_match": 0, "relax": 0}
           for t in thrs}
    with torch.no_grad():
        for data in test_s:
            data = data.to(device)
            pred_c, _, _ = model(data, teacher_forcing=True)
            coach = float(pred_c.item()) * sd + mu
            pid = data["our_squad"].player_ids
            coach_gk = int(pid[0].item())                       # node0 = coach's starting GK
            of_ids_all = pid[1:].cpu().numpy()
            coach_pool = data.our_starter_of_pool_idx.view(-1).cpu().numpy()
            coach_of = set(of_ids_all[coach_pool].tolist())
            if args.window == "total":
                get_mins = lambda p: minutes.get(p, 0.0)  # noqa: E731
            else:
                season = season_map.get(int(data.game_id), -1)
                get_mins = lambda p: minutes.get((p, season), 0.0)  # noqa: E731
            for t in thrs:
                gk_e, of_e, rx = elig_masks(data, get_mins, t)
                gk_e, of_e = gk_e.to(device), of_e.to(device)
                pred_m, _, _ = model(data, gk_elig=gk_e, of_elig=of_e)
                gk_id, of_sel = selected_ids(model, data, gk_e, of_e)
                a = acc[t]
                a["m"].append(float(pred_m.item()) * sd + mu)
                a["c"].append(coach)
                a["smatch"] += len(of_sel & coach_of)
                a["stot"] += 10
                a["gk_match"] += int(gk_id == coach_gk)
                a["relax"] += rx

    n = len(test_s)
    rows = []
    print(f"\n{'thr':>6} {'model_v':>8} {'coach_v':>8} {'delta':>7} "
          f"{'SelAcc':>7} {'GKmatch':>8} {'relax%':>7}")
    for t in thrs:
        a = acc[t]
        row = {"fold": k, "thr": t,
               "model_vaep": float(np.mean(a["m"])),
               "coach_vaep": float(np.mean(a["c"])),
               "delta_vaep": float(np.mean(a["m"]) - np.mean(a["c"])),
               "selection_acc": a["smatch"] / max(a["stot"], 1),
               "gk_match": a["gk_match"] / n,
               "relax_rate": a["relax"] / n, "n": n}
        rows.append(row)
        print(f"{t:6.0f} {row['model_vaep']:8.3f} {row['coach_vaep']:8.3f} "
              f"{row['delta_vaep']:7.3f} {row['selection_acc']:7.4f} "
              f"{row['gk_match']:8.4f} {100*row['relax_rate']:6.1f}%")

    suffix = "" if args.window == "total" else "_seasonwin"
    out = METRICS_DIR / f"min_minutes_mask{args.ckpt_tag}_fold{k}{suffix}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
