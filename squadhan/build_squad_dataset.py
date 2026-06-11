"""Build the squad-pool (starters + bench) HeteroData dataset.

Position in the pipeline:
  run_vaep.py → vaep/output/  →  build_squad_dataset.py (here)  →  train_e2e_vaep.py

Differences from the original build_dataset.py:
  - Nodes: our squad pool of 20 (GK 1 + OF 19) + 11 opponent starters
  - 2 samples per match: home view (match_{gid}_home.pt) + away view (match_{gid}_away.pt)
  - Metadata: our_starter_of_pool_idx, our_positions, is_home_game, y (result), game_id

Feature aggregation — totals over the full 21~25 seasons + per-90 normalization (static skill representation):
  - VAEP events of all matches are aggregated first into per-player/per-team caches.
  - Each .pt file then looks features up from the caches in O(N+E).
  - No "past matches only" filter → even early-2021 matches get dense features.
  - No label leakage: y (result) is separated by LOSO folds at training time.

Main fields of the saved .pt files:
  data["our_squad"].x              : (20, 48) node features [GK(idx0) + OF(idx1~19)]
  data["our_squad"].player_ids     : (20,) player_id
  data["opp"].x                    : (11, 48) opponent starter node features
  data["opp"].player_ids           : (11,)
  data.<edge_type>.edge_attr       : (E, 12) edge features
  data.our_starter_of_pool_idx     : (10,) pool indices of the coach's starting OF players
  data.our_positions               : (11, 2) actual starter coordinates (width, depth), normalized [0,1]
  data.is_home_game                : (1,) float 1.0/0.0
  data.y                           : (1,) result (0=loss/1=draw/2=win, from our viewpoint)
  data.game_id                     : int

Output location: outputs/squad_graphs/match_{game_id}_{home|away}.pt

Usage (from the repository root):
  python -m squadhan.build_squad_dataset             # full build
  python -m squadhan.build_squad_dataset --smoke     # K1 2024 only (quick sanity check)
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

warnings.filterwarnings("ignore")

GNN_ROOT = Path(__file__).resolve().parent.parent

from squadhan.config import (
    RAW_DATA_DIR, VAEP_OUTPUT_DIR,
    EDGE_DIM, VALID_COMPETITION_IDS, K1_COMPETITION_ID,
)
from squadhan.build_dataset import (
    _player_node, _extract_io_pairs, _extract_id_pairs,
    _ID_TO_ALL_IDS,
)
from squadhan.zones import map_to_zone

# Minimum-minutes shrinkage variant: with MIN_SHRINK_M0>0, node features are built
# as "actual-minutes-based per-90 × mins/(mins+M0)" and saved to a separate directory.
# train_e2e_vaep imports SQUAD_GRAPHS_DIR from this module, so training with the same
# env applies it consistently.
# Default (0) = 100% identical to the original behavior and paths.
MIN_SHRINK_M0 = float(os.environ.get("MIN_SHRINK_M0", "0"))

# Output path
SQUAD_GRAPHS_DIR = GNN_ROOT / "outputs" / (
    f"squad_graphs_mshr{int(MIN_SHRINK_M0)}" if MIN_SHRINK_M0 > 0 else "squad_graphs")

# Result-label flip map for the our-team viewpoint
RESULT_FLIP = {0: 2, 1: 1, 2: 0}


# ── Feature cache build (full-season totals) ───────────────────────────────────

def _build_node_cache(full_hist: pd.DataFrame) -> dict[int, np.ndarray]:
    """player_id → 48D node features (over the full 21-25 seasons).

    MIN_SHRINK_M0>0: true per-90 based on actual minutes (players.csv minutes_played
    sum), multiplied by the confidence weight mins/(mins+M0) to shrink small-sample
    profiles (continuous discounting, no exclusion).
    Default (0): same as before — per-appearance, assuming matches played × 90.
    """
    minutes_map: dict[int, float] = {}
    if MIN_SHRINK_M0 > 0:
        mp = (pd.read_csv(VAEP_OUTPUT_DIR / "players.csv")
              .groupby("player_id")["minutes_played"].sum())
        minutes_map = {int(k): float(v) for k, v in mp.items()}
        print(f"  [mshr] actual-minutes per-90 + shrinkage M0={MIN_SHRINK_M0:.0f} min")
    all_pids = sorted(set(int(p) for p in full_hist["player_id"].unique() if pd.notna(p)))
    cache: dict[int, np.ndarray] = {}
    by_pid = {pid: g for pid, g in full_hist.groupby("player_id")}
    for pid in tqdm(all_pids, desc="node cache"):
        all_ids = _ID_TO_ALL_IDS.get(pid, frozenset([pid]))
        sub = by_pid.get(pid) if len(all_ids) == 1 else full_hist[full_hist["player_id"].isin(all_ids)]
        if sub is None or len(sub) == 0:
            continue
        if MIN_SHRINK_M0 > 0:
            mins_real = sum(minutes_map.get(int(i), 0.0) for i in all_ids)
            shrink = mins_real / (mins_real + MIN_SHRINK_M0)
            cache[pid] = _player_node(sub, max(mins_real, 1.0)) * np.float32(shrink)
        else:
            mins = float(sub["game_id"].nunique()) * 90.0
            cache[pid] = _player_node(sub, mins)
    return cache


def _build_io_cache(io_pairs: pd.DataFrame, team_exposure: dict[int, float]) -> dict:
    """{(team_id, p_min, p_max): 12D vec} — symmetric IO edge cache (per 90 min)."""
    cache: dict = {}
    for tid, g in tqdm(io_pairs.groupby("team_id"), desc="IO cache"):
        edge_acc: dict[tuple[int, int], np.ndarray] = {}
        mins = team_exposure.get(int(tid), 1.0)
        for row in g.itertuples(index=False):
            sp, dp = int(row.src_player), int(row.dst_player)
            key = (min(sp, dp), max(sp, dp))
            if key not in edge_acc:
                edge_acc[key] = np.zeros(EDGE_DIM, dtype=np.float32)
            z_s = map_to_zone(float(row.src_x), float(row.src_y))
            z_d = map_to_zone(float(row.dst_x), float(row.dst_y))
            edge_acc[key][z_s] += float(row.src_vaep)
            edge_acc[key][z_d] += float(row.dst_vaep)
        for key, vec in edge_acc.items():
            cache[(int(tid), *key)] = vec * 90.0 / max(mins, 1.0)
    return cache


def _build_id_cache(id_pairs: pd.DataFrame, team_exposure: dict[int, float]) -> dict:
    """{(src_team, dst_team, src_player, dst_player): 12D vec} — directional ID edge cache (per 90 min).

    src_only=True: only src's (the defensive reactor's) vaep is accumulated, in the src zone.
    """
    cache: dict = {}
    for (src_tid, dst_tid), g in tqdm(id_pairs.groupby(["src_team", "dst_team"]), desc="ID cache"):
        edge_acc: dict[tuple[int, int], np.ndarray] = {}
        mins = team_exposure.get(int(src_tid), 1.0)
        for row in g.itertuples(index=False):
            sp, dp = int(row.src_player), int(row.dst_player)
            key = (sp, dp)
            if key not in edge_acc:
                edge_acc[key] = np.zeros(EDGE_DIM, dtype=np.float32)
            z_s = map_to_zone(float(row.src_x), float(row.src_y))
            edge_acc[key][z_s] += float(row.src_vaep)
        for key, vec in edge_acc.items():
            cache[(int(src_tid), int(dst_tid), *key)] = vec * 90.0 / max(mins, 1.0)
    return cache


def _node_matrix_from_cache(player_ids: list[int], node_cache: dict) -> torch.Tensor:
    """From the cache: list of players → (N, 48) node feature matrix."""
    rows = [node_cache.get(int(p), np.zeros(48, dtype=np.float32)) for p in player_ids]
    return torch.tensor(np.stack(rows), dtype=torch.float32)


def _build_io_edges(team_id: int, player_ids: list[int],
                    io_cache: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract IO edges from the cache → expand to bidirectional edges."""
    pid_set = set(player_ids)
    idx_map = {p: i for i, p in enumerate(player_ids)}
    src_list, dst_list, attr_list = [], [], []
    for (tid, p_min, p_max), vec in io_cache.items():
        if tid != team_id:
            continue
        if p_min not in pid_set or p_max not in pid_set:
            continue
        i_min, i_max = idx_map[p_min], idx_map[p_max]
        src_list += [i_min, i_max]
        dst_list += [i_max, i_min]
        attr_list += [vec, vec]
    if not src_list:
        return (torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0, EDGE_DIM), dtype=torch.float32))
    return (
        torch.tensor([src_list, dst_list], dtype=torch.long),
        torch.tensor(np.stack(attr_list), dtype=torch.float32),
    )


def _build_id_edges(src_tid: int, dst_tid: int,
                    src_ids: list[int], dst_ids: list[int],
                    id_cache: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract ID edges from the cache (src_team → dst_team direction)."""
    src_set = set(src_ids)
    dst_set = set(dst_ids)
    src_idx = {p: i for i, p in enumerate(src_ids)}
    dst_idx = {p: i for i, p in enumerate(dst_ids)}
    src_list, dst_list, attr_list = [], [], []
    for (st, dt, sp, dp), vec in id_cache.items():
        if st != src_tid or dt != dst_tid:
            continue
        if sp not in src_set or dp not in dst_set:
            continue
        src_list.append(src_idx[sp])
        dst_list.append(dst_idx[dp])
        attr_list.append(vec)
    if not src_list:
        return (torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0, EDGE_DIM), dtype=torch.float32))
    return (
        torch.tensor([src_list, dst_list], dtype=torch.long),
        torch.tensor(np.stack(attr_list), dtype=torch.float32),
    )


# ── Lineup loader ──────────────────────────────────────────────────────────────

def _load_full_lineup(competition: str, season: str, match_id: int) -> dict:
    """Load both teams' full lineups (starters + bench + coordinates) for a match.

    Returns
    -------
    dict
        {team_id: {
            "gk":      [(player_id, x, y)]   # GK starter (exactly 1)
            "of":      [(player_id, x, y)]   # outfield starters (exactly 10)
            "bench":   [player_id]            # bench (no coordinates)
        }}
    """
    path = RAW_DATA_DIR / competition / season / "match" / str(match_id) / "lineup.json"
    with open(path) as f:
        entries = json.load(f)["result"]

    teams: dict[int, dict] = {}
    for e in entries:
        tid = int(e["team_id"])
        if tid not in teams:
            teams[tid] = {"gk": [], "of": [], "bench": []}
        pid = int(e["player_id"])
        if e.get("is_starting_lineup"):
            pos = e.get("position") or {}
            x, y = float(pos.get("x", 0.0)), float(pos.get("y", 0.0))
            if e.get("position_name") == "GK":
                teams[tid]["gk"].append((pid, x, y))
            else:
                teams[tid]["of"].append((pid, x, y))
        else:
            teams[tid]["bench"].append(pid)
    return teams


# ── Single-sample builder ──────────────────────────────────────────────────────

def _build_sample(
    our_lineup: dict,
    opp_lineup: dict,
    our_team_id: int,
    opp_team_id: int,
    node_cache: dict,
    io_cache: dict,
    id_cache: dict,
    is_home_game: bool,
    y: int,
    game_id: int,
) -> HeteroData | None:
    """Build a single HeteroData sample from the our-team vs opponent viewpoint.

    Graph structure:
      our_squad : 20 nodes = GK 1 + OF 19 (10 starters + 9 bench)
        - index 0       : GK (starter)
        - indices 1~10  : OF starters (10)
        - indices 11~19 : OF bench (9)
      opp       : 11 nodes = opponent starters (GK 1 + OF 10)
    """
    if len(our_lineup["gk"]) != 1 or len(our_lineup["of"]) != 10:
        return None
    if len(opp_lineup["gk"]) != 1 or len(opp_lineup["of"]) != 10:
        return None

    # Our-team nodes
    our_gk_pid, our_gk_x, our_gk_y = our_lineup["gk"][0]
    our_of_starter_ids = [pid for pid, _, _ in our_lineup["of"]]
    our_bench_ids = list(our_lineup["bench"])
    our_squad_ids = [our_gk_pid] + our_of_starter_ids + our_bench_ids
    our_starter_of_pool_idx = torch.arange(10, dtype=torch.long)

    # Opponent nodes (11 starters)
    opp_gk_pid = opp_lineup["gk"][0][0]
    opp_of_starter_ids = [pid for pid, _, _ in opp_lineup["of"]]
    opp_ids = [opp_gk_pid] + opp_of_starter_ids

    # Node features from the cache
    our_node_feats = _node_matrix_from_cache(our_squad_ids, node_cache)
    opp_node_feats = _node_matrix_from_cache(opp_ids, node_cache)

    # Edges from the cache
    our_io_ei, our_io_ea = _build_io_edges(our_team_id, our_squad_ids, io_cache)
    opp_io_ei, opp_io_ea = _build_io_edges(opp_team_id, opp_ids, io_cache)
    oo_id_ei, oo_id_ea = _build_id_edges(our_team_id, opp_team_id, our_squad_ids, opp_ids, id_cache)
    oa_id_ei, oa_id_ea = _build_id_edges(opp_team_id, our_team_id, opp_ids, our_squad_ids, id_cache)

    # Coordinate GT (11 starters, ascending player_id)
    starter_coord_list = [(our_gk_pid, our_gk_x, our_gk_y)]
    starter_coord_list.extend(our_lineup["of"])
    starter_coord_list.sort(key=lambda t: t[0])
    starter_sorted_ids = torch.tensor([t[0] for t in starter_coord_list], dtype=torch.long)
    starter_positions = torch.tensor(
        [[t[1], t[2]] for t in starter_coord_list], dtype=torch.float32
    )

    # Assemble the HeteroData
    data = HeteroData()
    data["our_squad"].x = our_node_feats
    data["our_squad"].player_ids = torch.tensor(our_squad_ids, dtype=torch.long)
    data["opp"].x = opp_node_feats
    data["opp"].player_ids = torch.tensor(opp_ids, dtype=torch.long)

    data["our_squad", "IO", "our_squad"].edge_index = our_io_ei
    data["our_squad", "IO", "our_squad"].edge_attr = our_io_ea
    data["opp", "IO", "opp"].edge_index = opp_io_ei
    data["opp", "IO", "opp"].edge_attr = opp_io_ea
    data["our_squad", "ID", "opp"].edge_index = oo_id_ei
    data["our_squad", "ID", "opp"].edge_attr = oo_id_ea
    data["opp", "ID", "our_squad"].edge_index = oa_id_ei
    data["opp", "ID", "our_squad"].edge_attr = oa_id_ea

    data.n_gk = 1
    data.our_gk_node_idx = torch.tensor([0], dtype=torch.long)
    data.our_starter_of_pool_idx = our_starter_of_pool_idx
    data.our_starter_sorted_ids = starter_sorted_ids
    data.our_positions = starter_positions
    data.is_home_game = torch.tensor([1.0 if is_home_game else 0.0], dtype=torch.float32)
    data.y = torch.tensor([y], dtype=torch.long)
    data.game_id = game_id
    return data


# ── Full orchestration ─────────────────────────────────────────────────────────

def build(smoke: bool = False) -> None:
    """Build and save squad-pool HeteroData for every match.

    With smoke=True, only K1 2024 is processed.
    """
    SQUAD_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    # Match metadata
    games = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv")
    games = games[games["competition_id"].isin(VALID_COMPETITION_IDS)].copy()
    games["game_date"] = pd.to_datetime(games["game_date"])
    games = games.sort_values("game_date").reset_index(drop=True)

    if smoke:
        games = games[
            (games["competition_id"] == K1_COMPETITION_ID) & (games["season"].astype(int) == 2024)
        ].copy().reset_index(drop=True)
        print(f"[smoke] {len(games)} matches (K1 2024)")

    valid_game_ids = set(games["game_id"].astype(int).tolist())
    game_to_teams: dict[int, tuple[int, int]] = {
        int(r.game_id): (int(r.home_team_id), int(r.away_team_id))
        for r in games.itertuples(index=False)
    }

    # Load full-season SPADL + VAEP
    spadl_cache = VAEP_OUTPUT_DIR / "spadl_all.parquet"
    if not spadl_cache.exists():
        raise FileNotFoundError(
            f"SPADL cache not found: {spadl_cache}\n"
            "Run vaep/run_vaep.py first to generate spadl_all.parquet."
        )
    print("Loading SPADL cache …")
    spadl_all = pd.read_parquet(spadl_cache)
    spadl_all["game_id"] = spadl_all["game_id"].astype(int)
    # Even in smoke mode, features aggregate over full-season data (static representation)
    all_valid = set(pd.read_csv(VAEP_OUTPUT_DIR / "games.csv").query(
        "competition_id in @VALID_COMPETITION_IDS"
    )["game_id"].astype(int).tolist())
    full_spadl = spadl_all[spadl_all["game_id"].isin(all_valid)].copy()

    vaep = pd.read_parquet(VAEP_OUTPUT_DIR / "vaep_oof.parquet")
    vaep["game_id"] = vaep["game_id"].astype(int)

    print("Merging VAEP …")
    full_hist = full_spadl.merge(
        vaep[["game_id", "action_id", "vaep_value"]],
        on=["game_id", "action_id"], how="left",
    )
    full_hist["vaep_value"] = full_hist["vaep_value"].fillna(0.0)
    print(f"  Total events: {len(full_hist):,}")

    # Per-team exposure (total matches × 90 min)
    team_games = full_hist.groupby("team_id")["game_id"].nunique().to_dict()
    team_exposure = {int(t): float(g) * 90.0 for t, g in team_games.items()}

    # Build the caches (once)
    print("Extracting IO/ID pairs …")
    io_pairs = _extract_io_pairs(full_hist)
    id_pairs = _extract_id_pairs(full_hist)
    print(f"  IO pairs: {len(io_pairs):,}, ID pairs: {len(id_pairs):,}")

    print("Building node cache …")
    node_cache = _build_node_cache(full_hist)
    print(f"  node cache: {len(node_cache)} players")

    print("Building IO edge cache …")
    io_cache = _build_io_cache(io_pairs, team_exposure)
    print(f"  IO edge cache: {len(io_cache)} entries")

    print("Building ID edge cache …")
    id_cache = _build_id_cache(id_pairs, team_exposure)
    print(f"  ID edge cache: {len(id_cache)} entries")

    del full_hist, io_pairs, id_pairs, spadl_all, full_spadl, vaep

    # Build samples per match
    built, skipped = 0, 0
    for row in tqdm(games.itertuples(index=False), total=len(games), desc="building squad graphs"):
        gid = int(row.game_id)
        home_path = SQUAD_GRAPHS_DIR / f"match_{gid}_home.pt"
        away_path = SQUAD_GRAPHS_DIR / f"match_{gid}_away.pt"

        if home_path.exists() and away_path.exists():
            built += 2
            continue

        home_tid, away_tid = int(row.home_team_id), int(row.away_team_id)
        hs, as_ = int(row.home_score), int(row.away_score)
        result_home = 2 if hs > as_ else (1 if hs == as_ else 0)
        result_away = RESULT_FLIP[result_home]

        competition = "KLEAGUE1" if int(row.competition_id) == K1_COMPETITION_ID else "KLEAGUE2"
        season_str = str(row.season)
        try:
            lineups = _load_full_lineup(competition, season_str, gid)
        except FileNotFoundError:
            skipped += 2
            continue

        if home_tid not in lineups or away_tid not in lineups:
            skipped += 2
            continue
        home_l, away_l = lineups[home_tid], lineups[away_tid]

        if not home_path.exists():
            sample_home = _build_sample(
                our_lineup=home_l, opp_lineup=away_l,
                our_team_id=home_tid, opp_team_id=away_tid,
                node_cache=node_cache, io_cache=io_cache, id_cache=id_cache,
                is_home_game=True, y=result_home, game_id=gid,
            )
            if sample_home is None:
                skipped += 1
            else:
                torch.save(sample_home, home_path)
                built += 1

        if not away_path.exists():
            sample_away = _build_sample(
                our_lineup=away_l, opp_lineup=home_l,
                our_team_id=away_tid, opp_team_id=home_tid,
                node_cache=node_cache, io_cache=io_cache, id_cache=id_cache,
                is_home_game=False, y=result_away, game_id=gid,
            )
            if sample_away is None:
                skipped += 1
            else:
                torch.save(sample_away, away_path)
                built += 1

    print(f"Done — saved: {built}, skipped: {skipped} ({len(games)*2} potential samples total)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="K1 2024 only (quick sanity check)")
    args = ap.parse_args()
    build(smoke=args.smoke)
