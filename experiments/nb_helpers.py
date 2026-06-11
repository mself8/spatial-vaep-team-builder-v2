"""Notebook helpers (standalone module; does not modify existing files).

Imported and used by paper_results.ipynb.
  - list_games(team_ko, season, opp_ko)  → DataFrame of matches
  - load_model(ckpt_path, objective, hidden, edge_scalar, gk_select, no_gnn) → model
  - recommend(model, game_id, side, mu, sd, gkids) → dict (coach/model XI, coords, delta …)
  - plot_pitch(rec, title_extra)  → mplsoccer Pitch figure (coach vs model)
  - get_mu_sd(train_samples)  → (mu, sd) from train set
  - player_meta() / team_names() → cached metadata
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from squadhan.config import (
    CHECKPOINTS_DIR, VAEP_OUTPUT_DIR, NODE_DIM, EDGE_DIM,
    HIDDEN_CHANNELS, NUM_LAYERS, NUM_HEADS, DROPOUT,
)
from squadhan.e2e_model_vaep import E2ELineupOptimizerVAEP

PITCH_L, PITCH_W = 105.0, 68.0

POS_BROAD = {
    "GK": "GK",
    "CB": "DF", "LB": "DF", "RB": "DF", "RWB": "DF", "LWB": "DF",
    "CM": "MF", "CAM": "MF", "CDM": "MF", "LM": "MF", "RM": "MF",
    "CF": "FW", "RW": "FW", "LW": "FW", "RF": "FW", "LF": "FW",
}

_player_meta_cache: Optional[tuple] = None
_team_names_cache: Optional[tuple] = None


def player_meta() -> tuple[dict, dict]:
    """player_id → (Korean name, detailed position). Cached once."""
    global _player_meta_cache
    if _player_meta_cache is None:
        p = pd.read_csv(VAEP_OUTPUT_DIR / "players.csv",
                        usecols=["player_id", "player_name", "nickname", "starting_position_name"])
        p["player_id"] = p["player_id"].astype(int)
        # Korean name: prefer nickname, fall back to player_name
        name = {}
        for pid, grp in p.groupby("player_id"):
            for col in ("nickname", "player_name"):
                vals = grp[col].dropna()
                if len(vals) > 0 and str(vals.iloc[0]).strip():
                    name[int(pid)] = str(vals.iloc[0]).strip()
                    break
        pos = p.dropna(subset=["starting_position_name"])
        primary = (pos.groupby("player_id")["starting_position_name"]
                   .agg(lambda s: s.value_counts().idxmax()).to_dict())
        primary = {int(k): v for k, v in primary.items()}
        _player_meta_cache = (name, primary)
    return _player_meta_cache


def team_names() -> tuple[dict, dict]:
    """team_id → (team_name_ko, team_name_en). Cached once."""
    global _team_names_cache
    if _team_names_cache is None:
        t = pd.read_csv(VAEP_OUTPUT_DIR / "teams.csv")
        ko = dict(zip(t["team_id"].astype(int), t["team_name_ko"]))
        en = dict(zip(t["team_id"].astype(int), t["team_name"]))
        _team_names_cache = (ko, en)
    return _team_names_cache


def list_games(team_ko: str = "", season: Optional[int] = None,
               opp_ko: str = "") -> pd.DataFrame:
    """Return matches where team_ko (substring match) is the home or away team.

    Columns: game_id, date, season, team (home), opp (away), score, side
    """
    games = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv")
    ko, _ = team_names()

    games["home_name"] = games["home_team_id"].astype(int).map(ko)
    games["away_name"] = games["away_team_id"].astype(int).map(ko)

    rows = []
    for r in games.itertuples(index=False):
        h, a = str(r.home_name or ""), str(r.away_name or "")
        if team_ko and team_ko not in h and team_ko not in a:
            continue
        if season and int(r.season) != season:
            continue
        if opp_ko:
            if team_ko in h and opp_ko not in a:
                continue
            if team_ko in a and opp_ko not in h:
                continue
        # home side
        if not team_ko or team_ko in h:
            rows.append({"game_id": int(r.game_id), "date": r.game_date,
                         "season": int(r.season), "team": h, "opp": a,
                         "score": f"{r.home_score}-{r.away_score}", "side": "home"})
        # away side (distinct)
        if team_ko and team_ko in a:
            rows.append({"game_id": int(r.game_id), "date": r.game_date,
                         "season": int(r.season), "team": a, "opp": h,
                         "score": f"{r.away_score}-{r.home_score}", "side": "away"})

    df = pd.DataFrame(rows).drop_duplicates(["game_id", "side"])
    df = df.sort_values("date", ascending=False).reset_index(drop=True)
    return df


def load_model(ckpt_path: str | Path, objective: str = "vaep",
               hidden: int = 64, edge_scalar: bool = True,
               gk_select: bool = True, no_gnn: bool = False,
               coord_skip: bool = False, value_skip: bool = False,
               device: Optional[torch.device] = None) -> E2ELineupOptimizerVAEP:
    """Load a checkpoint and return the model in eval mode."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_dim = 1 if edge_scalar else EDGE_DIM
    model = E2ELineupOptimizerVAEP(
        node_dim=NODE_DIM, edge_dim=edge_dim, hidden=hidden,
        n_heads=NUM_HEADS, n_layers=NUM_LAYERS, dropout=DROPOUT,
        gk_select=gk_select, no_gnn=no_gnn, objective=objective,
        coord_skip=coord_skip, value_skip=value_skip,
    ).to(device)
    model.load_state_dict(torch.load(str(ckpt_path), map_location=device, weights_only=False))
    model.eval()
    return model


def _prep_data(data, gkids: set, edge_scalar: bool = True):
    """Apply the EDGE_SCALAR reduction and attach the is_gk mask to the graph data (same as training)."""
    if edge_scalar:
        for et in data.edge_types:
            data[et].edge_attr = data[et].edge_attr.sum(1, keepdim=True)
    pid = data["our_squad"].player_ids
    data["our_squad"].is_gk = torch.tensor(
        [int(x) in gkids for x in pid.tolist()], dtype=torch.bool)
    return data


_minutes_merged_cache: Optional[dict] = None


def _merged_minutes() -> dict:
    """player_id → total minutes over 21-25 (duplicate player IDs (same player) merged — same basis as the node features)."""
    global _minutes_merged_cache
    if _minutes_merged_cache is None:
        from squadhan.build_dataset import _ID_TO_ALL_IDS
        mp = (pd.read_csv(VAEP_OUTPUT_DIR / "players.csv")
              .groupby("player_id")["minutes_played"].sum())
        raw = {int(k): float(v) for k, v in mp.items()}
        _minutes_merged_cache = {
            p: sum(raw.get(i, 0.0) for i in _ID_TO_ALL_IDS.get(p, frozenset([p])))
            for p in raw}
    return _minutes_merged_cache


def _elig_masks(data, thr: float):
    """Minimum-minutes eligibility masks (relaxed to top minutes when the pool is short) — same rule as train._attach_elig."""
    mins = torch.tensor([_merged_minutes().get(int(p), 0.0)
                         for p in data["our_squad"].player_ids.tolist()])
    is_gk = data["our_squad"].is_gk.view(-1).bool()
    e = mins >= thr
    gk_e, of_e = e[is_gk].clone(), e[~is_gk].clone()
    if int(gk_e.sum()) < 1:
        gk_e[mins[is_gk].argmax()] = True
    if int(of_e.sum()) < 10:
        of_e[mins[~is_gk].topk(min(10, int((~is_gk).sum().item()))).indices] = True
    return gk_e, of_e


@torch.no_grad()
def recommend(model: E2ELineupOptimizerVAEP, game_id: int, side: str,
              mu: float, sd: float, gkids: set,
              edge_scalar: bool = True, min_elig: float = 0.0) -> dict:
    """For one match, return the coach XI (actual positions) / model-recommended XI (predicted positions) + Δ.

    Returns dict:
      coach_ids, coach_coords (11×2 m, actual), coach_vhat,
      model_ids, model_coords (11×2 m, predicted), model_vhat,
      delta_vhat (model−coach), selection_acc,
      coach_formation, model_formation,
      in_only (model only), out_only (coach only)
    """
    from squadhan.build_squad_dataset import SQUAD_GRAPHS_DIR
    import torch.nn.functional as F

    p = SQUAD_GRAPHS_DIR / f"match_{game_id}_{side}.pt"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found")

    device = next(model.parameters()).device
    data = torch.load(str(p), weights_only=False)
    data = _prep_data(data, gkids, edge_scalar).to(device)

    name_map, primary_map = player_meta()

    def _broad(pid):
        return POS_BROAD.get(primary_map.get(int(pid), ""), "UNK")

    def _formation_str(ids, gk_id):
        broads = [_broad(i) for i in ids if int(i) != int(gk_id)]
        c = defaultdict(int)
        for b in broads:
            c[b] += 1
        s = f"{c['DF']}-{c['MF']}-{c['FW']}"
        if c["UNK"]:
            s += f"(+UNK{c['UNK']})"
        return s

    # Coach XI
    pid_all = data["our_squad"].player_ids
    of_pool_idx = data.our_starter_of_pool_idx.view(-1).long()
    gk_id_c = int(pid_all[0].item())
    of_ids_c = pid_all[1:][of_pool_idx]
    starter_c = torch.cat([pid_all[0:1], of_ids_c])
    order_c = torch.argsort(starter_c)
    coach_ids = starter_c[order_c].tolist()

    v_coach, _, _ = model(data, teacher_forcing=True)
    # Paper Fig. 3(a) convention: the coach panel shows actual (GT) starter positions.
    # data.our_positions rows are player_id-sorted — same order as coach_ids.
    coach_coords_norm = data.our_positions.detach().cpu().numpy()  # (11,2) [0,1]

    # Model XI (assumes gk_select) — if min_elig>0, apply the eligibility filter (paper setting: 900)
    gk_e = of_e = None
    if min_elig > 0:
        gk_e, of_e = _elig_masks(data, min_elig)
        gk_e, of_e = gk_e.to(device), of_e.to(device)
    our_emb, opp_emb = model.encoder(data)
    opp_ctx = opp_emb.mean(dim=0)
    is_gk = data["our_squad"].is_gk.view(-1).bool()
    gk_pool_ids = pid_all[is_gk]
    of_pool_ids = pid_all[~is_gk]
    gk_pool_emb = our_emb[is_gk]
    of_pool_emb = our_emb[~is_gk]

    _, _, gk_idx = model.gk_selector(gk_pool_emb, opp_ctx, k=1, training=False, elig=gk_e)
    _, _, of_idx = model.of_selector(of_pool_emb, opp_ctx, k=10, training=False, elig=of_e)
    gk_id_m = int(gk_pool_ids[gk_idx].item())
    starter_m = torch.cat([gk_pool_ids[gk_idx], of_pool_ids[of_idx]])
    order_m = torch.argsort(starter_m)
    model_ids = starter_m[order_m].tolist()

    v_model, coords_m, _ = model(data, teacher_forcing=False,
                                 gk_elig=gk_e, of_elig=of_e)
    model_coords_norm = coords_m.detach().cpu().numpy()  # (11,2) [0,1]

    def _scale_coords(coords_norm):
        # our_positions = (width_norm, depth_norm) → x_m=depth*L, y_m=width*W
        # make_fig3_case axis convention (swap): op[k,1]*L, op[k,0]*W
        xy_m = np.empty_like(coords_norm)
        xy_m[:, 0] = coords_norm[:, 1] * PITCH_L   # x_m = depth
        xy_m[:, 1] = coords_norm[:, 0] * PITCH_W   # y_m = width
        return xy_m

    coach_xy = _scale_coords(coach_coords_norm)
    model_xy = _scale_coords(model_coords_norm)

    # De-standardize the objective (for points, convert to expected match points)
    if model.objective == "points":
        v_c = float(3.0 * F.softmax(v_coach, dim=-1)[2] + F.softmax(v_coach, dim=-1)[1])
        v_m = float(3.0 * F.softmax(v_model, dim=-1)[2] + F.softmax(v_model, dim=-1)[1])
    else:
        v_c = float(v_coach.item()) * sd + mu
        v_m = float(v_model.item()) * sd + mu

    # sel_acc (over the 10 OF)
    coach_of_set = set(of_ids_c.tolist())
    model_of_set = set(of_pool_ids[of_idx].tolist())
    sel_acc = len(coach_of_set & model_of_set) / 10.0

    in_only = [i for i in model_ids if i not in set(coach_ids)]
    out_only = [i for i in coach_ids if i not in set(model_ids)]

    return {
        "coach_ids": [int(x) for x in coach_ids],
        "coach_coords": coach_xy,
        "coach_vhat": v_c,
        "coach_formation": _formation_str(coach_ids, gk_id_c),
        "model_ids": [int(x) for x in model_ids],
        "model_coords": model_xy,
        "model_vhat": v_m,
        "model_formation": _formation_str(model_ids, gk_id_m),
        "delta_vhat": v_m - v_c,
        "selection_acc": sel_acc,
        "in_only": [int(x) for x in in_only],
        "out_only": [int(x) for x in out_only],
        "objective": model.objective,
    }


def plot_pitch(rec: dict, game_id: int = 0, side: str = "home",
               figsize: tuple = (14, 8)) -> "matplotlib.figure.Figure":
    """Plot the coach vs model-recommended lineups side by side on mplsoccer Pitches.

    rec: return value of recommend()
    """
    import matplotlib.pyplot as plt
    from mplsoccer import Pitch

    name_map, primary_map = player_meta()

    def short(pid):
        n = name_map.get(int(pid), str(pid))
        return n[:6] if len(n) > 6 else n

    obj_label = "exp. points" if rec["objective"] == "points" else "team VAEP"

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, ids, xy, vhat, form, title_tag in [
        (axes[0], rec["coach_ids"], rec["coach_coords"],
         rec["coach_vhat"], rec["coach_formation"], "Coach XI — actual positions"),
        (axes[1], rec["model_ids"], rec["model_coords"],
         rec["model_vhat"], rec["model_formation"], "Model XI (recommended) — predicted positions"),
    ]:
        pitch = Pitch(pitch_type="custom", pitch_length=PITCH_L, pitch_width=PITCH_W,
                      pitch_color="#1a7a1a", line_color="white", linewidth=1.5)
        pitch.draw(ax=ax)

        in_set = set(rec["in_only"])
        out_set = set(rec["out_only"])

        for j, pid in enumerate(ids):
            px, py = float(xy[j, 0]), float(xy[j, 1])
            is_new = pid in in_set
            is_drop = pid in out_set
            c = "#FF6B35" if is_new else ("#4ECDC4" if not is_drop else "#95A5A6")
            edge_c = "white" if is_new else "white"
            ax.scatter(px, py, s=220, c=c, edgecolors=edge_c, linewidths=1.5, zorder=5)
            lbl = short(pid)
            if is_new:
                lbl = "+" + lbl
            ax.text(px, py - 3.5, lbl, ha="center", va="top", fontsize=6.5,
                    color="white", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.45, lw=0))

        delta_str = f"Δ{obj_label}={rec['delta_vhat']:+.3f}" if title_tag.startswith("Model") else ""
        ax.set_title(f"{title_tag}  [{form}]\n{obj_label}={vhat:.3f}  {delta_str}",
                     fontsize=9, pad=6)

    fig.suptitle(f"game {game_id} ({side})  sel_acc={rec['selection_acc']:.1%}",
                 fontsize=10, y=1.01)
    handles = [
        plt.scatter([], [], c="#FF6B35", s=80, label="added by model (in)"),
        plt.scatter([], [], c="#4ECDC4", s=80, label="common"),
        plt.scatter([], [], c="#95A5A6", s=80, label="coach only (out)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, framealpha=0.7)
    fig.tight_layout()
    return fig


def build_comparison_row(tag: str, objective: str, hidden: int,
                         test_metrics: dict) -> dict:
    """evaluate_test result dict → one-row dict for the comparison DataFrame."""
    row = {"tag": tag, "objective": objective, "hidden": hidden}
    for k, v in test_metrics.items():
        row[k] = round(v, 4) if isinstance(v, float) else v
    return row
