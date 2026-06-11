"""Two-stage E2E lineup recommendation training + 5-fold LOSO CV — VAEP-objective version.

Supported objectives (switched via environment variables):
  VAEP_DIFF=0 (default) : y = our team's total VAEP.        Stage2 loss = −v̂ (maximize)
  VAEP_DIFF=1           : y = our − opponent VAEP diff.     Stage2 loss = −v̂ (maximize)
  OBJECTIVE=points      : y = win/draw/loss label (0/1/2).  Stage2 loss = −log(3·P_win+P_draw)
                          (not VAEP regression; a port of the train_e2e.py approach)

Differences from the original train_e2e.py:
  - Objective: win/draw/loss (3-class CE) → VAEP regression (MSE). (OBJECTIVE=points switches back)
  - Stage 1 loss: LAMBDA_COORD*MSE(coords) + MSE(pred_vaep, y_std)
                  (y_std = per-fold standardization, leakage-safe)
  - Stage 2 loss: −pred_vaep  (maximize expected VAEP or the VAEP difference)
  - Metrics: Stage1 = R²/Pearson/RMSE/NRMSE + pos_r2/pos_pearson/pos_rmse/pos_nrmse
             Stage2 = model_vaep / coach_vaep / Δvaep / sel_acc
  - Output paths: e2e_vaep_* (does not touch existing e2e_*)
  - GK_SELECT=1 : split GK/OF pools via the is_gk mask → train gk_selector+of_selector jointly
  - EDGE_SCALAR=1: collapse 12D edges → 1D scalar at load time (.pt files unchanged)
  - NO_GNN=1 : per-node MLP encoder instead of SquadHAN → GNN ablation

Usage (from the repository root):
  python -m squadhan.train_e2e_vaep                               # default (team-VAEP, 5-fold)
  python -m squadhan.train_e2e_vaep --fold 0                      # fold 0 only
  VAEP_DIFF=1 python -m squadhan.train_e2e_vaep                   # VAEP-difference objective
  S1_EPOCHS=2 S2_EPOCHS=2 python -m squadhan.train_e2e_vaep       # smoke (2 epochs)
  NO_GNN=1 python -m squadhan.train_e2e_vaep                      # MLP ablation

Main environment variables (all optional):
  VAEP_DIFF   : 1=VAEP-difference objective (default 0=team-VAEP)
  OBJECTIVE   : points=expected-points objective (default vaep)
  GK_SELECT   : 1=competitive GK selection (default 1)
  EDGE_SCALAR : 1=collapse edges to 1D (default 1)
  LAMBDA_COORD: weight of the coordinate auxiliary loss (default 10)
  NO_GNN      : 1=MLP encoder (default 0=GNN)
  RUN_TAG     : suffix identifying output files
  S1_EPOCHS   : max Stage1 epochs
  S2_EPOCHS   : max Stage2 epochs
"""

import argparse
import csv
import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, log_loss
from tqdm import tqdm

warnings.filterwarnings("ignore")

from squadhan.config import (
    CHECKPOINTS_DIR, VAEP_OUTPUT_DIR, METRICS_DIR,
    NODE_DIM, EDGE_DIM, HIDDEN_CHANNELS, NUM_LAYERS, NUM_HEADS, DROPOUT,
    VALID_COMPETITION_IDS,
)
from squadhan.e2e_model_vaep import E2ELineupOptimizerVAEP
from squadhan.build_squad_dataset import SQUAD_GRAPHS_DIR


# ── Hyperparameters ────────────────────────────────────────────────────────────
STAGE1_EPOCHS = int(os.environ.get("S1_EPOCHS", 50))
STAGE1_LR = 3e-3     # adjusted for ACCUM_STEPS=32: 1e-3 × ~√32 ≈ 3e-3
STAGE1_WD = 1e-4
STAGE1_PATIENCE = 10
STAGE1_ACCUM_STEPS = 32   # reduces batch=1 noise: step after averaging gradients over 32 samples
LAMBDA_COORD = float(os.environ.get("LAMBDA_COORD", 10.0))   # coord auxiliary-loss weight (sweepable via env)

STAGE2_EPOCHS = int(os.environ.get("S2_EPOCHS", 50))
STAGE2_LR = float(os.environ.get("S2_LR", 1e-4))
STAGE2_PATIENCE = int(os.environ.get("S2_PATIENCE", 15))
ACCUM_STEPS = 32
GRAD_CLIP = 1.0
TEMP_START = 1.0   # temperature annealing start (tunable from a notebook via T.TEMP_START)
TEMP_END   = 0.1   # temperature annealing end

# Edge representation variant: EDGE_SCALAR=1 collapses IO/ID edges from 12 zones → 1D (zone sum) at load time (.pt files unchanged)
EDGE_SCALAR = os.environ.get("EDGE_SCALAR", "0") == "1"

# Competitive GK selection: GK_SELECT=1 picks top-1 from the GK pool + top-10 from the OF pool (backup GKs split off the OF pool). Default OFF.
GK_SELECT = os.environ.get("GK_SELECT", "0") == "1"

# Target variant: VAEP_DIFF=1 sets y = our total VAEP − opponent total VAEP (difference). Default is our total VAEP.
VAEP_DIFF = os.environ.get("VAEP_DIFF", "0") == "1"

# no-GNN ablation: NO_GNN=1 uses a per-node Linear projection instead of SquadHGT (edges ignored). Validates the GNN's contribution.
NO_GNN = os.environ.get("NO_GNN", "0") == "1"

# SEG_TOKEN=1: add our/opponent segment embeddings to the joint Transformer input (experiment C3).
SEG_TOKEN = os.environ.get("SEG_TOKEN", "0") == "1"

# Node-spatiality ablation: NODE_NOZONE=1 collapses 48D (4 groups × 12 zones) to 4D (zone sums) at load time.
NODE_NOZONE = os.environ.get("NODE_NOZONE", "0") == "1"

# Edge-type ablation: EDGE_DROP=ID|IO removes the edges of that relation at load time (empty edges).
EDGE_DROP = os.environ.get("EDGE_DROP", "")

# NO_TRANSFORMER=1: bypass the joint Transformer (ablation; selected embeddings go straight to the heads)
NO_TRANSFORMER = os.environ.get("NO_TRANSFORMER", "0") == "1"

# COORD_SKIP=1: skip connection of raw encoder embeddings into the coordinate head (position-prediction tuning)
COORD_SKIP = os.environ.get("COORD_SKIP", "0") == "1"

# VALUE_SKIP=1: skip-concat mean encoder embeddings into the value head (value-head version of coord_skip)
VALUE_SKIP = os.environ.get("VALUE_SKIP", "0") == "1"

# TRF_LAYERS: set the joint Transformer depth separately (0=default, same as NUM_LAYERS)
TRF_LAYERS = int(os.environ.get("TRF_LAYERS", "0"))

# Objective: OBJECTIVE=points trains 3-class win/draw/loss (expected points) — the "why VAEP" comparison. Default vaep (team-VAEP regression).
OBJECTIVE = os.environ.get("OBJECTIVE", "vaep")

# Minimum-minutes eligibility filter (hard): if >0, players whose actual minutes
# (21-25 players.csv sum) fall below the threshold are excluded from selector
# candidates (both inference and Stage2 training). If the pool runs short
# (eligible GK<1 / OF<10), relax by top minutes. Requires GK_SELECT=1.
# 0 (default) = 100% identical to the original behavior.
MIN_ELIG_MINUTES = float(os.environ.get("MIN_ELIG_MINUTES", "0"))
_MINUTES_MAP = None

VAL_RATIO_LOSO = 0.1
SEED = int(os.environ.get("SEED", "42"))   # env knob for Stage-2 reseeding diagnostics (default 42, unchanged)
SEASONS = [2021, 2022, 2023, 2024, 2025]

# Fixed split: FIXED_SPLIT=1 uses 21~23 train / 24 val / 25 test (a single run, no fold loop)
FIXED_SPLIT = os.environ.get("FIXED_SPLIT", "0") == "1"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _append_csv(path: Path, row: dict, header: list):
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if is_new:
            w.writeheader()
        w.writerow(row)


def _build_yvaep_map() -> dict:
    """(game_id, is_home=1/0) -> team total VAEP."""
    v = pd.read_parquet(VAEP_OUTPUT_DIR / "vaep_oof.parquet",
                        columns=["game_id", "team_id", "vaep_value"]).dropna()
    v["game_id"] = v["game_id"].astype(int)
    v["team_id"] = v["team_id"].astype(int)
    tm = v.groupby(["game_id", "team_id"])["vaep_value"].sum().to_dict()
    games = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv")
    games = games[games["competition_id"].isin(VALID_COMPETITION_IDS)].copy()
    ymap = {}
    for r in games.itertuples(index=False):
        gid = int(r.game_id)
        for is_home, tid in ((1, int(r.home_team_id)), (0, int(r.away_team_id))):
            if (gid, tid) in tm:
                ymap[(gid, is_home)] = float(tm[(gid, tid)])
    return ymap


def _build_gkids() -> set:
    """Set of player_ids ever fielded as a GK starter in players.csv (identifies backup GKs)."""
    p = pd.read_csv(VAEP_OUTPUT_DIR / "players.csv",
                    usecols=["player_id", "starting_position_name"])
    return set(p.loc[p["starting_position_name"] == "GK", "player_id"].astype(int).tolist())


def _reg_metrics(y_true: list, y_pred: list) -> dict:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    sse = float(((yt - yp) ** 2).sum())
    sst = float(((yt - yt.mean()) ** 2).sum())
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    rmse = float(np.sqrt(((yt - yp) ** 2).mean()))
    rng = float(yt.max() - yt.min())                     # range-normalized NRMSE (bounded, near [0,1])
    nrmse = rmse / rng if rng > 0 else float("nan")
    try:
        pear = float(pearsonr(yp, yt)[0])
    except Exception:
        pear = float("nan")
    return {"r2": float(r2), "pearson": pear, "rmse": rmse, "nrmse": nrmse}


# ── Expected-points (OBJECTIVE=points) metrics (same as train_e2e.py) ──────────

def _ece_multiclass(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    pred = y_proba.argmax(axis=1)
    conf = y_proba.max(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi)
        if mask.sum() > 0:
            ece += mask.sum() / n * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def _brier_multiclass(y_true: np.ndarray, y_proba: np.ndarray, n_classes: int = 3) -> float:
    y_onehot = np.zeros((len(y_true), n_classes), dtype=np.float32)
    y_onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(((y_proba - y_onehot) ** 2).sum(axis=1).mean())


def _cls_metrics(y_true: list, y_proba: list) -> dict:
    """AUC (OvR macro + per class), LogLoss, Brier, ECE. 0=loss/1=draw/2=win."""
    y_true = np.asarray(y_true)
    y_proba = np.stack(y_proba)
    y_proba_clipped = np.clip(y_proba, 1e-7, 1 - 1e-7)
    auc_pc = [float("nan")] * 3
    try:
        per = roc_auc_score(y_true, y_proba, multi_class="ovr", average=None)
        for i in range(3):
            auc_pc[i] = float(per[i])
        auc = float(np.nanmean(per))
    except ValueError:
        auc = float("nan")
    return {"auc": auc, "auc_loss": auc_pc[0], "auc_draw": auc_pc[1], "auc_win": auc_pc[2],
            "logloss": float(log_loss(y_true, y_proba_clipped, labels=[0, 1, 2])),
            "brier": _brier_multiclass(y_true, y_proba), "ece": _ece_multiclass(y_true, y_proba)}


def _exp_pts(logits) -> float:
    """logits(3,) → expected points 3·P_win + 1·P_draw."""
    P = F.softmax(logits, dim=-1)
    return float(3.0 * P[2] + 1.0 * P[1])


def _loss_stage2_points(logits):
    """Stage 2 (points): minimize -log(3·P_win + 1·P_draw) = maximize expected points."""
    P = F.softmax(logits, dim=-1)
    return -torch.log((3.0 * P[2] + 1.0 * P[1]).clamp(min=1e-8))


def _load_samples_fold(test_season: int, seed: int, ymap: dict, gkids: set = None):
    """Season LOSO + per-fold 10% random val. Attaches data._yv (target VAEP) at load time.
    With GK_SELECT, also attaches data['our_squad'].is_gk (bool mask) from gkids (.pt files unchanged)."""
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
    n_val = int(len(rest_t) * VAL_RATIO_LOSO)
    val_t, train_t = rest_t[:n_val], rest_t[n_val:]

    def _load(triples):
        out = []
        for p, is_home, gid in triples:
            if (gid, is_home) not in ymap:
                continue
            d = torch.load(p, weights_only=False)
            d._yv = ymap[(gid, is_home)]
            if EDGE_SCALAR:
                for et in d.edge_types:
                    d[et].edge_attr = d[et].edge_attr.sum(1, keepdim=True)
            if NODE_NOZONE:
                for nt in ("our_squad", "opp"):
                    x = d[nt].x
                    d[nt].x = x.view(x.size(0), 4, 12).sum(-1)
            if EDGE_DROP:
                for et in d.edge_types:
                    if et[1] == EDGE_DROP:
                        ed = d[et].edge_attr.size(1)
                        d[et].edge_index = torch.zeros((2, 0), dtype=torch.long)
                        d[et].edge_attr = torch.zeros((0, ed))
            if gkids is not None:
                pid = d["our_squad"].player_ids
                d["our_squad"].is_gk = torch.tensor(
                    [int(x) in gkids for x in pid.tolist()], dtype=torch.bool)
                _attach_elig(d)
            out.append(d)
        return out

    return _load(train_t), _load(val_t), _load(test_t)


def _load_samples_fixed(ymap: dict, gkids: set = None):
    """Fixed split: train=2021~2023 / val=2024 / test=2025 (separated at season boundaries)."""
    games = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv")
    games = games[games["competition_id"].isin(VALID_COMPETITION_IDS)].copy()
    games["season"] = games["season"].astype(int)
    season_map = dict(zip(games["game_id"].astype(int), games["season"]))

    train_t, val_t, test_t = [], [], []
    for gid, season in season_map.items():
        for side in ("home", "away"):
            p = SQUAD_GRAPHS_DIR / f"match_{gid}_{side}.pt"
            if not p.exists():
                continue
            triple = (p, 1 if side == "home" else 0, gid)
            if season == 2025:
                test_t.append(triple)
            elif season == 2024:
                val_t.append(triple)
            else:
                train_t.append(triple)

    def _load(triples):
        out = []
        for p, is_home, gid in triples:
            if (gid, is_home) not in ymap:
                continue
            d = torch.load(p, weights_only=False)
            d._yv = ymap[(gid, is_home)]
            if EDGE_SCALAR:
                for et in d.edge_types:
                    d[et].edge_attr = d[et].edge_attr.sum(1, keepdim=True)
            if NODE_NOZONE:
                for nt in ("our_squad", "opp"):
                    x = d[nt].x
                    d[nt].x = x.view(x.size(0), 4, 12).sum(-1)
            if EDGE_DROP:
                for et in d.edge_types:
                    if et[1] == EDGE_DROP:
                        ed = d[et].edge_attr.size(1)
                        d[et].edge_index = torch.zeros((2, 0), dtype=torch.long)
                        d[et].edge_attr = torch.zeros((0, ed))
            if gkids is not None:
                pid = d["our_squad"].player_ids
                d["our_squad"].is_gk = torch.tensor(
                    [int(x) in gkids for x in pid.tolist()], dtype=torch.bool)
                _attach_elig(d)
            out.append(d)
        return out

    return _load(train_t), _load(val_t), _load(test_t)


# ── Losses ────────────────────────────────────────────────────────────────────

def _loss_stage1(pred, coords, data, mu, sd):
    """Stage 1: coord MSE + (vaep: standardized VAEP MSE | points: win/draw/loss CE)."""
    coord_gt = data.our_positions.view(-1, 2)
    loss_coord = F.mse_loss(coords, coord_gt)
    if OBJECTIVE == "points":
        loss_main = F.cross_entropy(pred.unsqueeze(0), data.y.view(-1).long())
    else:
        y_std = torch.as_tensor((data._yv - mu) / sd, device=pred.device, dtype=pred.dtype)
        loss_main = F.mse_loss(pred, y_std)
    return LAMBDA_COORD * loss_coord + loss_main, loss_coord.item(), loss_main.item()


def _attach_elig(d):
    """MIN_ELIG_MINUTES>0: attach eligibility masks (gk_elig/of_elig) in pool order."""
    global _MINUTES_MAP
    if MIN_ELIG_MINUTES <= 0:
        return
    if _MINUTES_MAP is None:
        from squadhan.build_dataset import _ID_TO_ALL_IDS   # lazy import (merge table for duplicate player IDs)
        mp = (pd.read_csv(VAEP_OUTPUT_DIR / "players.csv")
              .groupby("player_id")["minutes_played"].sum())
        raw = {int(k): float(v) for k, v in mp.items()}
        # Node features use ID-merged profiles, so merge eligibility minutes the same way for consistency
        _MINUTES_MAP = {p: sum(raw.get(i, 0.0)
                               for i in _ID_TO_ALL_IDS.get(p, frozenset([p])))
                        for p in raw}
    pid = d["our_squad"].player_ids
    mins = torch.tensor([_MINUTES_MAP.get(int(p), 0.0) for p in pid.tolist()])
    is_gk = d["our_squad"].is_gk.view(-1).bool()
    elig = mins >= MIN_ELIG_MINUTES
    gk_e, of_e = elig[is_gk].clone(), elig[~is_gk].clone()
    if int(gk_e.sum()) < 1:                      # no eligible GK → allow the GK with the most minutes
        gk_e[mins[is_gk].argmax()] = True
    if int(of_e.sum()) < 10:                     # eligible OF<10 → fill with top minutes
        of_e[mins[~is_gk].topk(min(10, int((~is_gk).sum().item()))).indices] = True
    d["our_squad"].gk_elig = gk_e
    d["our_squad"].of_elig = of_e


def _fwd_sel(model, data):
    """Selection-mode forward — passes eligibility masks attached by the loader, if any."""
    return model(data, teacher_forcing=False,
                 gk_elig=getattr(data["our_squad"], "gk_elig", None),
                 of_elig=getattr(data["our_squad"], "of_elig", None))


def _selected_of_ids(model, data):
    """IDs of the 10 OF players the selector picks (frozen evaluator) + coach OF pool (node[1:]) indices.
    sel_acc is over the 10 OF players (intersection with the coach's OF starters). With GK_SELECT, picks come from the OF pool."""
    our_emb, opp_emb = model.encoder(data)
    opp_ctx = opp_emb.mean(dim=0)
    pid = data["our_squad"].player_ids
    if getattr(model, "gk_select", False):
        is_gk = data["our_squad"].is_gk.view(-1).bool()
        of_pool_emb = our_emb[~is_gk]
        of_pool_ids = pid[~is_gk].cpu().numpy()
        _, _, idx = model.of_selector(of_pool_emb, opp_ctx, k=10, training=False,
                                      elig=getattr(data["our_squad"], "of_elig", None))
        model_ids = set(of_pool_ids[idx.cpu().numpy()].tolist())
    else:
        _, _, idx = model.of_selector(our_emb[1:], opp_ctx, k=10, training=False)
        model_ids = set(pid[1:].cpu().numpy()[idx.cpu().numpy()].tolist())
    of_ids = pid[1:].cpu().numpy()
    coach_pool = data.our_starter_of_pool_idx.view(-1).cpu().numpy()
    return model_ids, of_ids, coach_pool


# ── Stage 1 ───────────────────────────────────────────────────────────────────

def train_stage1(model, train_s, val_s, device, ckpt, metrics_csv, mu, sd):
    opt = torch.optim.Adam(model.parameters(), lr=STAGE1_LR, weight_decay=STAGE1_WD)
    best_val, patience = float("inf"), 0
    if OBJECTIVE == "points":
        header = ["epoch", "train_loss", "train_coord", "train_result",
                  "val_loss", "val_auc", "val_ece", "val_logloss", "val_brier"]
    else:
        header = ["epoch", "train_loss", "train_coord", "train_vaep",
                  "val_loss", "val_r2", "val_pearson", "val_rmse"]
    last = ckpt.with_name(ckpt.stem + "_last.pt")
    start_epoch = 1
    if last.exists():
        st = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        best_val, patience, start_epoch = st["best_val"], st["patience"], st["epoch"] + 1
        print(f"[S1-vaep] resume ep{start_epoch} (best={best_val:.4f} patience={patience})")
    elif metrics_csv.exists():
        metrics_csv.unlink()

    for epoch in range(start_epoch, STAGE1_EPOCHS + 1):
        model.train()
        random.shuffle(train_s)
        tot, totc, totv = 0.0, 0.0, 0.0
        opt.zero_grad()
        valid_in_window = 0
        for i, data in enumerate(tqdm(train_s, desc=f"[S1 ep{epoch}]", leave=False)):
            data = data.to(device)
            pred, coords, _ = model(data, teacher_forcing=True)
            loss, c, vv = _loss_stage1(pred, coords, data, mu, sd)
            if not torch.isfinite(loss):
                continue                      # skip NaN/inf samples
            # Track valid samples in the window (denominator correction excluding non-finite samples)
            ws = (i // STAGE1_ACCUM_STEPS) * STAGE1_ACCUM_STEPS
            accs = min(ws + STAGE1_ACCUM_STEPS, len(train_s)) - ws
            (loss / accs).backward()
            valid_in_window += 1
            tot += loss.item(); totc += c; totv += vv
            if (i + 1) % STAGE1_ACCUM_STEPS == 0:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                if torch.isfinite(gn):
                    opt.step()
                opt.zero_grad(); valid_in_window = 0
        # Handle the final partial window
        if valid_in_window > 0:
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            if torch.isfinite(gn):
                opt.step()
            opt.zero_grad()
        n = max(len(train_s), 1)
        train_loss = tot / n

        model.eval()
        vloss = 0.0
        yt, yp = [], []          # vaep: ground truth/predictions | points: y_true/y_proba
        with torch.no_grad():
            for data in val_s:
                data = data.to(device)
                pred, coords, _ = model(data, teacher_forcing=True)
                loss, _, _ = _loss_stage1(pred, coords, data, mu, sd)
                vloss += loss.item()
                if OBJECTIVE == "points":
                    yp.append(F.softmax(pred, dim=-1).cpu().numpy())
                    yt.append(int(data.y.view(-1).item()))
                else:
                    yp.append(float(pred.item()) * sd + mu)
                    yt.append(data._yv)
        m = max(len(val_s), 1)
        val_loss = vloss / m

        if OBJECTIVE == "points":
            cm = _cls_metrics(yt, yp)
            print(f"[S1-pts] ep{epoch:03d} train={train_loss:.4f}(coord={totc/n:.4f} ce={totv/n:.4f}) "
                  f"val={val_loss:.4f} | AUC={cm['auc']:.3f} ECE={cm['ece']:.3f} LogLoss={cm['logloss']:.3f}")
            _append_csv(metrics_csv, {
                "epoch": epoch, "train_loss": train_loss, "train_coord": totc / n, "train_result": totv / n,
                "val_loss": val_loss, "val_auc": cm["auc"], "val_ece": cm["ece"],
                "val_logloss": cm["logloss"], "val_brier": cm["brier"],
            }, header)
        else:
            rm = _reg_metrics(yt, yp)
            print(f"[S1-vaep] ep{epoch:03d} train={train_loss:.4f}(coord={totc/n:.4f} vaep={totv/n:.4f}) "
                  f"val={val_loss:.4f} | R2={rm['r2']:+.3f} Pearson={rm['pearson']:+.3f} RMSE={rm['rmse']:.3f}")
            _append_csv(metrics_csv, {
                "epoch": epoch, "train_loss": train_loss, "train_coord": totc / n, "train_vaep": totv / n,
                "val_loss": val_loss, "val_r2": rm["r2"], "val_pearson": rm["pearson"], "val_rmse": rm["rmse"],
            }, header)

        if val_loss < best_val:
            best_val = val_loss; patience = 0
            torch.save(model.state_dict(), ckpt)
        else:
            patience += 1
        torch.save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(),
                    "best_val": best_val, "patience": patience}, last)
        if patience >= STAGE1_PATIENCE:
            print(f"[S1-vaep] Early stop ep{epoch} (best={best_val:.4f})")
            break
    ckpt.with_name(ckpt.stem + ".done").write_text("ok")
    last.unlink(missing_ok=True)
    print(f"[S1-vaep] done best val_loss={best_val:.4f} -> {ckpt}")


# ── Stage 2 ───────────────────────────────────────────────────────────────────

def train_stage2(model, train_s, val_s, device, ckpt, metrics_csv, mu, sd):
    for p in model.encoder.parameters(): p.requires_grad = False
    for p in model.transformer.parameters(): p.requires_grad = False
    for p in model.coord_head.parameters(): p.requires_grad = False
    for p in model.vaep_head.parameters(): p.requires_grad = False
    if hasattr(model, "result_head"):                       # freeze the points evaluator head too
        for p in model.result_head.parameters(): p.requires_grad = False

    trainable = list(model.of_selector.parameters())
    if getattr(model, "gk_select", False):
        trainable += list(model.gk_selector.parameters())   # the GK selector is also trained in Stage2
    print(f"[S2-vaep] trainable(selector)={sum(p.numel() for p in trainable)}")
    opt = torch.optim.Adam(trainable, lr=STAGE2_LR)
    best_val, patience = float("inf"), 0
    _vk = "exp_pts" if OBJECTIVE == "points" else "vaep"     # CSV column suffix (per objective)
    header = ["epoch", "temperature", "train_loss", "val_loss",
              f"val_model_{_vk}", f"val_coach_{_vk}", f"val_delta_{_vk}", "val_selection_acc"]
    last = ckpt.with_name(ckpt.stem + "_last.pt")
    start_epoch = 1
    if last.exists():
        st = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        best_val, patience, start_epoch = st["best_val"], st["patience"], st["epoch"] + 1
        print(f"[S2-vaep] resume ep{start_epoch} (best={best_val:.4f} patience={patience})")
    elif metrics_csv.exists():
        metrics_csv.unlink()

    for epoch in range(start_epoch, STAGE2_EPOCHS + 1):
        anneal = (TEMP_START - TEMP_END) / max(STAGE2_EPOCHS - 1, 1)
        temp = max(TEMP_END, TEMP_START - anneal * (epoch - 1))
        model.of_selector.temperature = temp
        if getattr(model, "gk_select", False):
            model.gk_selector.temperature = temp

        model.train()
        random.shuffle(train_s)
        opt.zero_grad()
        tot = 0.0
        for i, data in enumerate(tqdm(train_s, desc=f"[S2 ep{epoch}]", leave=False)):
            data = data.to(device)
            pred, _, _ = _fwd_sel(model, data)
            if not torch.isfinite(pred).all():
                continue                  # skip samples with NaN/inf predictions
            loss_step = _loss_stage2_points(pred) if OBJECTIVE == "points" else -pred
            tot += loss_step.item()
            ws = (i // ACCUM_STEPS) * ACCUM_STEPS
            accs = min(ws + ACCUM_STEPS, len(train_s)) - ws
            (loss_step / accs).backward()
            if (i + 1) % ACCUM_STEPS == 0:
                gn = torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
                if torch.isfinite(gn):
                    opt.step()
                opt.zero_grad()
        if len(train_s) % ACCUM_STEPS != 0:
            gn = torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            if torch.isfinite(gn):
                opt.step()
            opt.zero_grad()
        train_loss = tot / len(train_s)

        model.eval()
        mlist, clist = [], []
        smatch, stot = 0, 0
        vobj = 0.0
        with torch.no_grad():
            for data in val_s:
                data = data.to(device)
                pred_m, _, _ = _fwd_sel(model, data)
                pred_c, _, _ = model(data, teacher_forcing=True)
                if OBJECTIVE == "points":
                    mlist.append(_exp_pts(pred_m)); clist.append(_exp_pts(pred_c))
                    vobj += float(_loss_stage2_points(pred_m).item())
                else:
                    mlist.append(float(pred_m.item()) * sd + mu)
                    clist.append(float(pred_c.item()) * sd + mu)
                    vobj += -float(pred_m.item())
                msel, of_ids, coach_pool = _selected_of_ids(model, data)
                smatch += len(msel & set(of_ids[coach_pool].tolist()))
                stot += 10
        m = max(len(val_s), 1)
        vmodel, vcoach = float(np.mean(mlist)), float(np.mean(clist))
        vdelta, vsel, val_loss = vmodel - vcoach, smatch / max(stot, 1), vobj / m

        tag = "S2-pts" if OBJECTIVE == "points" else "S2-vaep"
        print(f"[{tag}] ep{epoch:03d} temp={temp:.2f} train={train_loss:.4f} "
              f"| model_{_vk}={vmodel:.3f} coach_{_vk}={vcoach:.3f} Δ={vdelta:+.3f} sel_acc={vsel:.3f}")
        _append_csv(metrics_csv, {
            "epoch": epoch, "temperature": temp, "train_loss": train_loss, "val_loss": val_loss,
            f"val_model_{_vk}": vmodel, f"val_coach_{_vk}": vcoach,
            f"val_delta_{_vk}": vdelta, "val_selection_acc": vsel,
        }, header)

        if val_loss < best_val:
            best_val = val_loss; patience = 0
            torch.save(model.state_dict(), ckpt)
        else:
            patience += 1
        torch.save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(),
                    "best_val": best_val, "patience": patience}, last)
        if patience >= STAGE2_PATIENCE:
            print(f"[S2-vaep] Early stop ep{epoch}")
            break
    ckpt.with_name(ckpt.stem + ".done").write_text("ok")
    last.unlink(missing_ok=True)
    print(f"[S2-vaep] done best val_loss={best_val:.4f} -> {ckpt}")


# ── Test ──────────────────────────────────────────────────────────────────────

def evaluate_test(model, test_s, device, mu, sd) -> dict:
    model.eval()
    yt, yp = [], []                       # vaep: ground truth/predictions | points: y_true/y_proba
    coord_mse = 0.0
    pos_t, pos_p = [], []                 # position head: flattened coordinates (pred/GT) → same 4 metrics
    mlist, clist = [], []                 # model/coach scores (VAEP value or exp_pts)
    smatch, stot = 0, 0
    with torch.no_grad():
        for data in test_s:
            data = data.to(device)
            pred_c, coords_c, _ = model(data, teacher_forcing=True)
            gt_pos = data.our_positions.view(-1, 2)
            coord_mse += F.mse_loss(coords_c, gt_pos).item()
            pos_p.extend(coords_c.reshape(-1).cpu().numpy().tolist())
            pos_t.extend(gt_pos.reshape(-1).cpu().numpy().tolist())

            pred_m, _, _ = _fwd_sel(model, data)
            if OBJECTIVE == "points":
                yp.append(F.softmax(pred_c, dim=-1).cpu().numpy())
                yt.append(int(data.y.view(-1).item()))
                clist.append(_exp_pts(pred_c)); mlist.append(_exp_pts(pred_m))
            else:
                yp.append(float(pred_c.item()) * sd + mu); yt.append(data._yv)
                clist.append(float(pred_c.item()) * sd + mu)
                mlist.append(float(pred_m.item()) * sd + mu)

            msel, of_ids, coach_pool = _selected_of_ids(model, data)
            smatch += len(msel & set(of_ids[coach_pool].tolist()))
            stot += 10

    pm = _reg_metrics(pos_t, pos_p)       # 4 position metrics (on flattened coordinates)
    n = max(len(test_s), 1)
    out = {
        "coord_mse": coord_mse / n,
        "pos_r2": pm["r2"], "pos_pearson": pm["pearson"], "pos_rmse": pm["rmse"], "pos_nrmse": pm["nrmse"],
        "model_score": float(np.mean(mlist)), "coach_score": float(np.mean(clist)),
        "delta": float(np.mean(mlist) - np.mean(clist)),
        "selection_acc": smatch / max(stot, 1),
    }
    if OBJECTIVE == "points":
        out.update(_cls_metrics(yt, yp))                  # auc/auc_*/logloss/brier/ece
    else:
        out.update(_reg_metrics(yt, yp))                  # r2/pearson/rmse/nrmse
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--fold", type=int, default=-1)
    args = ap.parse_args()

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Building y_vaep map …")
    ymap = _build_yvaep_map()
    if VAEP_DIFF:
        dmap = {}
        for (gid, ih), val in ymap.items():
            opp = ymap.get((gid, 1 - ih))
            if opp is not None:
                dmap[(gid, ih)] = val - opp     # our total VAEP − opponent total VAEP
        ymap = dmap
        print(f"  Target = VAEP difference (ours − opponent). team-matches: {len(ymap)}")
    else:
        print(f"  Target = team total VAEP. team-matches: {len(ymap)}")
    gkids = _build_gkids() if GK_SELECT else None
    if GK_SELECT:
        print(f"  Competitive GK selection ON — gkids: {len(gkids)} players")
    if MIN_ELIG_MINUTES > 0:
        assert GK_SELECT, "MIN_ELIG_MINUTES requires GK_SELECT=1"
        print(f"  Minimum-minutes eligibility filter ON — {MIN_ELIG_MINUTES:.0f} min (sum of 21-25 actual minutes)")

    folds = list(range(5)) if args.fold == -1 else [args.fold]
    TAG = ("_scalar" if EDGE_SCALAR else "") + os.environ.get("RUN_TAG", "")
    edge_dim_use = 1 if EDGE_SCALAR else EDGE_DIM
    node_dim_use = 4 if NODE_NOZONE else NODE_DIM
    if NODE_NOZONE or EDGE_DROP or NO_TRANSFORMER:
        print(f"ablation: NODE_NOZONE={NODE_NOZONE} EDGE_DROP='{EDGE_DROP}' NO_TRANSFORMER={NO_TRANSFORMER}")
    print(f"objective={OBJECTIVE} | edges={'scalar' if EDGE_SCALAR else '12-zone'} | LAMBDA_COORD={LAMBDA_COORD} | "
          f"GK_SELECT={GK_SELECT} | VAEP_DIFF={VAEP_DIFF} | NO_GNN={NO_GNN} | FIXED_SPLIT={FIXED_SPLIT} | tag='{TAG or '(none)'}'")
    test_csv = METRICS_DIR / f"e2e_vaep{TAG}_test_cv.csv"
    if OBJECTIVE == "points":
        test_header = ["fold", "test_season", "s1_auc", "s1_auc_loss", "s1_auc_draw", "s1_auc_win",
                       "s1_logloss", "s1_brier", "s1_ece", "s1_coord_mse",
                       "s1_pos_r2", "s1_pos_pearson", "s1_pos_rmse", "s1_pos_nrmse",
                       "s2_model_exp_pts", "s2_coach_exp_pts", "s2_delta_exp_pts", "s2_selection_acc"]
    else:
        test_header = ["fold", "test_season", "s1_r2", "s1_pearson", "s1_rmse", "s1_nrmse", "s1_coord_mse",
                       "s1_pos_r2", "s1_pos_pearson", "s1_pos_rmse", "s1_pos_nrmse",
                       "s2_model_vaep", "s2_coach_vaep", "s2_delta_vaep", "s2_selection_acc"]
    done_folds = set(pd.read_csv(test_csv)["fold"].astype(int)) if test_csv.exists() else set()

    if FIXED_SPLIT:
        # Fixed split: 21~23 train / 24 val / 25 test — a single run without the fold loop
        torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
        print(f"\n{'='*60}\nFixed Split — train:2021~2023 / val:2024 / test:2025\n{'='*60}")
        train_s, val_s, test_s = _load_samples_fixed(ymap, gkids)
        print(f"  Train {len(train_s)}  Val {len(val_s)}  Test {len(test_s)}")
        ys = np.array([d._yv for d in train_s], dtype=np.float64)
        mu, sd = float(ys.mean()), float(ys.std() + 1e-8)
        print(f"  y_vaep standardize: mu={mu:.3f} sd={sd:.3f}")
        folds = [0]    # stored as fold=0 (denotes the fixed split)

    for k in folds:
        if FIXED_SPLIT and k != 0:
            break
        if not FIXED_SPLIT:
            ts = SEASONS[k]
            seed_k = SEED + k
            torch.manual_seed(seed_k); np.random.seed(seed_k); random.seed(seed_k)
            print(f"\n{'='*60}\nFold {k} — test season={ts} (seed={seed_k})\n{'='*60}")
            train_s, val_s, test_s = _load_samples_fold(ts, seed_k, ymap, gkids)
            print(f"  Train {len(train_s)}  Val {len(val_s)}  Test {len(test_s)}")
            ys = np.array([d._yv for d in train_s], dtype=np.float64)
            mu, sd = float(ys.mean()), float(ys.std() + 1e-8)
            print(f"  y_vaep standardize: mu={mu:.3f} sd={sd:.3f}")

        model = E2ELineupOptimizerVAEP(
            node_dim=node_dim_use, edge_dim=edge_dim_use, hidden=HIDDEN_CHANNELS,
            n_heads=NUM_HEADS, n_layers=NUM_LAYERS, dropout=DROPOUT,
            gk_select=GK_SELECT, no_gnn=NO_GNN, objective=OBJECTIVE,
            seg_token=SEG_TOKEN, no_transformer=NO_TRANSFORMER,
            coord_skip=COORD_SKIP, value_skip=VALUE_SKIP,
            n_trf_layers=(TRF_LAYERS or None),
        ).to(device)

        s1 = CHECKPOINTS_DIR / f"e2e_vaep{TAG}_stage1_fold{k}.pt"
        s2 = CHECKPOINTS_DIR / f"e2e_vaep{TAG}_stage2_fold{k}.pt"
        m1 = METRICS_DIR / f"e2e_vaep{TAG}_stage1_metrics_fold{k}.csv"
        m2 = METRICS_DIR / f"e2e_vaep{TAG}_stage2_metrics_fold{k}.csv"

        if args.stage == 0 and k in done_folds:
            print(f"  fold {k} already complete (in {test_csv.name}) — skip")
            del train_s, val_s, test_s, model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        if args.stage in (0, 1):
            if s1.with_name(s1.stem + ".done").exists():
                print(f"\n=== Fold {k} Stage 1 — done, skip ===")
            else:
                print(f"\n=== Fold {k} Stage 1 (training the VAEP evaluator) ===")
                train_stage1(model, train_s, val_s, device, s1, m1, mu, sd)

        if args.stage in (0, 2):
            if not s1.exists():
                raise FileNotFoundError(f"Stage 1 checkpoint not found: {s1}")
            model.load_state_dict(torch.load(s1, map_location=device, weights_only=False))
            if s2.with_name(s2.stem + ".done").exists():
                print(f"\n=== Fold {k} Stage 2 — done, skip ===")
            else:
                print(f"\n=== Fold {k} Stage 2 (Selector) ===")
                train_stage2(model, train_s, val_s, device, s2, m2, mu, sd)

        if args.stage == 0:
            ts_label = 2025 if FIXED_SPLIT else ts
            print(f"\n=== Fold {k} Test (season {ts_label}) ===")
            model.load_state_dict(torch.load(s2, map_location=device, weights_only=False))
            t = evaluate_test(model, test_s, device, mu, sd)
            row = {"fold": k, "test_season": ts_label, "s1_coord_mse": t["coord_mse"],
                   "s1_pos_r2": t["pos_r2"], "s1_pos_pearson": t["pos_pearson"],
                   "s1_pos_rmse": t["pos_rmse"], "s1_pos_nrmse": t["pos_nrmse"],
                   "s2_selection_acc": t["selection_acc"]}
            if OBJECTIVE == "points":
                print(f"[Test fold{k}] AUC={t['auc']:.3f} (loss{t['auc_loss']:.3f}/draw{t['auc_draw']:.3f}/win{t['auc_win']:.3f}) "
                      f"ECE={t['ece']:.3f} LogLoss={t['logloss']:.3f}")
                print(f"[Test fold{k}] model_pts={t['model_score']:.3f} coach_pts={t['coach_score']:.3f} "
                      f"Δ={t['delta']:+.3f} sel_acc={t['selection_acc']:.3f}")
                row.update({"s1_auc": t["auc"], "s1_auc_loss": t["auc_loss"], "s1_auc_draw": t["auc_draw"],
                            "s1_auc_win": t["auc_win"], "s1_logloss": t["logloss"], "s1_brier": t["brier"],
                            "s1_ece": t["ece"], "s2_model_exp_pts": t["model_score"],
                            "s2_coach_exp_pts": t["coach_score"], "s2_delta_exp_pts": t["delta"]})
            else:
                print(f"[Test fold{k}] R2={t['r2']:+.3f} Pearson={t['pearson']:+.3f} "
                      f"RMSE={t['rmse']:.3f} coord_mse={t['coord_mse']:.4f}")
                print(f"[Test fold{k}] model_vaep={t['model_score']:.3f} coach_vaep={t['coach_score']:.3f} "
                      f"Δ={t['delta']:+.3f} sel_acc={t['selection_acc']:.3f}")
                row.update({"s1_r2": t["r2"], "s1_pearson": t["pearson"], "s1_rmse": t["rmse"],
                            "s1_nrmse": t["nrmse"], "s2_model_vaep": t["model_score"],
                            "s2_coach_vaep": t["coach_score"], "s2_delta_vaep": t["delta"]})
            _append_csv(test_csv, row, test_header)

        del train_s, val_s, test_s, model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.fold == -1 and args.stage == 0 and test_csv.exists():
        df = pd.read_csv(test_csv).drop_duplicates(subset="fold", keep="last").sort_values("fold")
        print(f"\n{'='*60}\n5-fold VAEP results (mean ± std)\n{'='*60}")
        for col in df.columns:
            if col in ("fold", "test_season"):
                continue
            print(f"  {col:20s}: {df[col].mean():.4f} ± {df[col].std():.4f}")


if __name__ == "__main__":
    main()
