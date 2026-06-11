"""Build HeteroData graphs from VAEP + raw event data.

Pipeline:
  1. Load vaep_oof.parquet (LOSO 5-fold OOF, no data leakage)
  2. Convert to SPADL via BeproLoader → obtain start_x, start_y coordinates
  3. Merge VAEP values onto SPADL actions by (game_id, action_id)
  4. For each match m (in chronological order), build a graph from events of matches before m:
       Nodes (per player, 276d):
         23 action types × 12 zones matrix, accumulated vaep_value, per-90 normalized
       IO edges (same team, 12d, symmetric pairs summed):
         (a→b) and (b→a) events are summed as one pair → bidirectional edges
         vec[zone_a] += a.vaep_value
         vec[zone_b] += b.vaep_value
         → both a→b and b→a edges share the same attr
       ID edges (different teams d→o, 12d, src_only):
         vec[zone_d] += d.vaep_value  (only d's vaep; o's vaep excluded)
         condition: d.team_id != o.team_id  (must be different teams)
  5. Save the HeteroData to outputs/graphs/match_{game_id}.pt

Usage (from the repository root):
  python -m squadhan.build_dataset            # full build
  python -m squadhan.build_dataset --smoke    # K1 2024 only (quick sanity check)
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent

from squadhan.config import (
    RAW_DATA_DIR, VAEP_OUTPUT_DIR, GRAPHS_DIR,
    NUM_ZONES, NUM_ACTION_TYPES, NODE_DIM, EDGE_DIM,
    GROUP_MAP, NUM_GROUPS,
    VALID_COMPETITION_IDS, K1_COMPETITION_ID,
)
from squadhan.zones import map_to_zone
from vaep import core as vaep_core  # importing `vaep` puts vaep/lib on sys.path
from datatools.loaders.bepro import BeproLoader

# Starters per team (skip the match graph unless exactly 11)
PLAYERS_PER_TEAM = 11

# ── Action-group lookup ───────────────────────────────────────────────────────
# Reverse lookup type_id → group (type_ids not in GROUP_MAP return -1)
_TYPE_TO_GROUP: dict[int, int] = {
    t: g for g, types in GROUP_MAP.items() for t in types
}

# ── Merging duplicate player IDs across transfers ─────────────────────────────
def _load_unified_ids() -> dict[int, frozenset[int]]:
    """Load player_id_groups.json → {player_id: set of all player_ids of the same player}.

    File format: [[id, id, ...], ...] — lists of IDs grouped together when the
    same player has multiple player_ids across seasons/transfers (no identifying
    information).
    """
    path = REPO_ROOT / "player_id_groups.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        groups = json.load(f)
    result: dict[int, frozenset[int]] = {}
    for ids in groups:
        ids_set = frozenset(int(i) for i in ids)
        for pid in ids_set:
            result[pid] = ids_set
    return result

_ID_TO_ALL_IDS: dict[int, frozenset[int]] = _load_unified_ids()

# ── ELO helpers ───────────────────────────────────────────────────────────────
_ELO_K = 20
_ELO_HOME_ADV = 100  # home-advantage correction (home team +100 in the expectation)

def _elo_expected(rh: float, ra: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((ra - rh - _ELO_HOME_ADV) / 400.0))

def _elo_update(rh: float, ra: float, result: int) -> tuple[float, float]:
    """Update ELO from the match result (2=home win, 1=draw, 0=home loss)."""
    sh = {2: 1.0, 1: 0.5, 0: 0.0}[result]
    eh = _elo_expected(rh, ra)
    return rh + _ELO_K * (sh - eh), ra + _ELO_K * ((1 - sh) - (1 - eh))


# ── Lineup loader ──────────────────────────────────────────────────────────────

def _load_lineup(competition: str, season: str, match_id: int) -> dict[int, list[int]]:
    """Return each team's list of starters for a given match.

    Reads raw-data/{competition}/{season}/match/{match_id}/lineup.json and
    keeps only players with is_starting_lineup == True.

    Parameters
    ----------
    competition : str
        League folder name: "KLEAGUE1" or "KLEAGUE2"
    season : str
        Season year string, e.g. "2024"
    match_id : int
        Unique match ID

    Returns
    -------
    dict[int, list[int]]
        {team_id: [player_id, ...]}.
        Key: team ID; value: list of that team's starter player IDs
    """
    path = RAW_DATA_DIR / competition / season / "match" / str(match_id) / "lineup.json"
    with open(path) as f:
        entries = json.load(f)["result"]

    teams: dict[int, list[int]] = {}
    for e in entries:
        # Only players with is_starting_lineup == True count as starters
        if not e.get("is_starting_lineup"):
            continue
        teams.setdefault(int(e["team_id"]), []).append(int(e["player_id"]))
    return teams


# ── Node feature construction ──────────────────────────────────────────────────

def _player_node(events: pd.DataFrame, total_minutes: float) -> np.ndarray:
    """Compute the 48-dim node feature vector for one player.

    Node feature layout:
      - 4 action groups × 12 zones = 48-dim flattened vector
      - GROUP_MAP folds the 23 type_ids into 4 groups → less sparsity
      - vec[group * 12 + zone] += vaep_value  (accumulated per action)
      - finally normalized per 90 minutes

    Parameters
    ----------
    events : pd.DataFrame
        Past event rows of this player.
        Required columns: type_id, start_x, start_y, vaep_value
    total_minutes : float
        Player's total minutes played (approximated as matches × 90)

    Returns
    -------
    np.ndarray
        shape (48,), dtype float32
    """
    vec = np.zeros(NODE_DIM, dtype=np.float32)   # shape (48,)

    for row in events.itertuples(index=False):
        g = _TYPE_TO_GROUP.get(int(row.type_id), -1)
        if g < 0:
            continue
        z = map_to_zone(float(row.start_x), float(row.start_y))
        vec[g * NUM_ZONES + z] += float(row.vaep_value)

    if total_minutes > 0:
        vec *= 90.0 / total_minutes
    return vec


# ── Edge pair extractors ───────────────────────────────────────────────────────

def _extract_io_pairs(events: pd.DataFrame) -> pd.DataFrame:
    """Extract same-team consecutive event pairs (a→b). No action-type restriction.

    IO edge definition:
      Player a acts at time t; another player b of the same team acts at t+1.
      → captures intra-team cooperation/link-up patterns.

    Why action types are not restricted:
      Not only passes but also dribble → pass, shot → rebound, etc. —
      every form of consecutive intra-team action reflects cooperation patterns.

    Parameters
    ----------
    events : pd.DataFrame
        SPADL events of one or more matches, assumed sorted by time.
        Required columns: team_id, player_id, start_x, start_y, vaep_value

    Returns
    -------
    pd.DataFrame
        Columns: src_player, dst_player, src_x, src_y, src_vaep,
               dst_x, dst_y, dst_vaep, team_id
        Rows containing NaN are dropped.
    """
    e = events.reset_index(drop=True)
    # shift(-1): the row immediately after each row (the t+1 event)
    nxt = e.shift(-1)

    mask = (
        (e["team_id"] == nxt["team_id"])      # consecutive actions by the same team
        & (e["player_id"] != nxt["player_id"]) # must be a different player (no self-succession)
        & e["player_id"].notna()               # current row must have a player_id
        & nxt["player_id"].notna()             # the next row must exist
    )
    s, d = e[mask].reset_index(drop=True), nxt[mask].reset_index(drop=True)
    return pd.DataFrame({
        "src_player": s["player_id"].astype(int),   # player a's ID
        "dst_player": d["player_id"].astype(int),   # player b's ID
        "src_x": s["start_x"], "src_y": s["start_y"], "src_vaep": s["vaep_value"],
        "dst_x": d["start_x"], "dst_y": d["start_y"], "dst_vaep": d["vaep_value"],
        "team_id": s["team_id"].astype(int),
    }).dropna()


def _extract_id_pairs(events: pd.DataFrame) -> pd.DataFrame:
    """Extract cross-team consecutive event pairs (d→o). No action-type restriction.

    ID edge definition:
      Player d acts at time t, right after a player o of the other team acted at t-1.
      → captures one-on-one interactions where d reacts to o's action or wins the ball.

    Implementation:
      shift(1) fetches the immediately preceding event; keep rows where the teams differ.

    Why action types are not restricted:
      Beyond tackles/interceptions, we want every possession-switch moment,
      including those after the opponent loses the ball or it goes out of bounds.

    Parameters
    ----------
    events : pd.DataFrame
        SPADL events sorted by time.
        Required columns: team_id, player_id, start_x, start_y, vaep_value

    Returns
    -------
    pd.DataFrame
        Columns: src_player(d), dst_player(o), src_x, src_y, src_vaep,
               dst_x, dst_y, dst_vaep, src_team, dst_team
        Rows containing NaN are dropped.
    """
    e = events.reset_index(drop=True)
    # shift(1): the row immediately before each row (the t-1 event)
    prv = e.shift(1)

    mask = (
        prv["team_id"].notna()                  # the previous event must exist
        & (e["team_id"] != prv["team_id"])       # consecutive actions by different teams (cross-team guard)
        & e["player_id"].notna()                 # current row must have a player_id
        & prv["player_id"].notna()
    )
    d, o = e[mask].reset_index(drop=True), prv[mask].reset_index(drop=True)
    return pd.DataFrame({
        "src_player": d["player_id"].astype(int),   # player d (current actor)
        "dst_player": o["player_id"].astype(int),   # player o (previous actor, opposing team)
        "src_x": d["start_x"], "src_y": d["start_y"], "src_vaep": d["vaep_value"],
        "dst_x": o["start_x"], "dst_y": o["start_y"], "dst_vaep": o["vaep_value"],
        "src_team": d["team_id"].astype(int),
        "dst_team": o["team_id"].astype(int),
    }).dropna()


# ── Edge aggregator ───────────────────────────────────────────────────────────

def _aggregate_edges(
    pairs: pd.DataFrame,
    src_map: dict[int, int],
    dst_map: dict[int, int],
    exposure_minutes: float,
    symmetric: bool = False,
    src_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate per-pair events into 12-dim edge feature tensors.

    B-scheme aggregation:
      - src's vaep → accumulated in the zone slot where src acted
      - dst's vaep → accumulated in the zone slot where dst acted (skipped if src_only=True)

    symmetric=True (for IO edges):
      - (a,b) and (b,a) events are summed under the same pair key (merging both directions)
      - emit both a→b and b→a edges with identical attr (pair chemistry)

    src_only=True (for ID edges):
      - only src(d)'s vaep is accumulated; dst(o)'s vaep is excluded
      - the o node only learns "the defensive pressure received from d"

    Parameters
    ----------
    pairs : pd.DataFrame
        Output of _extract_io_pairs or _extract_id_pairs.
    src_map : dict[int, int]
        {player_id: node index within the graph} — for src players
    dst_map : dict[int, int]
        {player_id: node index within the graph} — for dst players
    exposure_minutes : float
        Total minutes used for normalization (team level, matches × 90)
    symmetric : bool
        If True, sum (a,b)·(b,a) events then emit bidirectional edges (IO)
    src_only : bool
        If True, only src vaep goes into attr (ID)

    Returns
    -------
    edge_index : torch.Tensor  shape (2, E), dtype long
        (src_idx, dst_idx) of E edges
    edge_attr  : torch.Tensor  shape (E, 12), dtype float32
        12-dim features of E edges (per-90 normalized)
    """
    valid = pairs[
        pairs["src_player"].isin(src_map) & pairs["dst_player"].isin(dst_map)
    ]
    if valid.empty:
        return (torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0, EDGE_DIM), dtype=torch.float32))

    edge_acc: dict[tuple[int, int], np.ndarray] = {}
    for row in valid.itertuples(index=False):
        sp, dp = int(row.src_player), int(row.dst_player)
        # symmetric: sum into the same pair regardless of direction
        key = (min(sp, dp), max(sp, dp)) if symmetric else (sp, dp)
        if key not in edge_acc:
            edge_acc[key] = np.zeros(EDGE_DIM, dtype=np.float32)
        z_s = map_to_zone(float(row.src_x), float(row.src_y))
        edge_acc[key][z_s] += float(row.src_vaep)
        if not src_only:
            z_d = map_to_zone(float(row.dst_x), float(row.dst_y))
            edge_acc[key][z_d] += float(row.dst_vaep)

    mins = max(exposure_minutes, 1.0)
    src_list, dst_list, attr_list = [], [], []
    for (sp, dp), vec in edge_acc.items():
        normalized = vec * 90.0 / mins
        si, di = src_map[sp], dst_map[dp]
        if symmetric:
            # bidirectional edges: emit both a→b and b→a with identical attr
            src_list += [si, di]
            dst_list += [di, si]
            attr_list += [normalized, normalized]
        else:
            src_list.append(si)
            dst_list.append(di)
            attr_list.append(normalized)

    return (
        torch.tensor([src_list, dst_list], dtype=torch.long),
        torch.tensor(np.stack(attr_list), dtype=torch.float32),
    )


# ── HeteroData graph builder ───────────────────────────────────────────────────

def _build_graph(
    home_players: list[int],
    away_players: list[int],
    home_team_id: int,
    away_team_id: int,
    hist: pd.DataFrame,
    result: int,
    home_elo: float = 1500.0,
    away_elo: float = 1500.0,
) -> HeteroData | None:
    """Build the HeteroData graph for a single match.

    Graph structure:
      2 node types:
        "home_team" : 11 home starters, 276-dim each
        "away_team" : 11 away starters, 276-dim each
      4 edge types (all 12-dim):
        (home_team, IO, home_team) : consecutive events within the home team
        (away_team, IO, away_team) : consecutive events within the away team
        (home_team, ID, away_team) : home→away cross-team events
        (away_team, ID, home_team) : away→home cross-team events
      Label:
        data.y = 0 (home loss), 1 (draw), 2 (home win)

    Parameters
    ----------
    home_players : list[int]
        Home starter player IDs (must be exactly 11)
    away_players : list[int]
        Away starter player IDs (must be exactly 11)
    home_team_id : int
        Home team ID (used for edge filtering)
    away_team_id : int
        Away team ID (used for edge filtering)
    hist : pd.DataFrame
        All events of the matches preceding this one.
        Columns: player_id, team_id, type_id, start_x, start_y, vaep_value, game_id
    result : int
        Match result label (0=home loss, 1=draw, 2=home win)

    Returns
    -------
    HeteroData or None
        Returns None unless each starting XI has exactly 11 players.
    """
    # Validate starters: the graph cannot be built unless exactly 11
    if len(home_players) != PLAYERS_PER_TEAM or len(away_players) != PLAYERS_PER_TEAM:
        return None

    def _minutes(pid: int) -> float:
        """Player's total minutes played — includes records under pre-transfer player_ids."""
        all_ids = _ID_TO_ALL_IDS.get(pid, frozenset([pid]))
        games_played = hist[hist["player_id"].isin(all_ids)]["game_id"].nunique()
        return float(games_played) * 90.0

    def _node_matrix(players: list[int]) -> torch.Tensor:
        """Return the (11, 48) node feature matrix for a list of players."""
        return torch.tensor(
            np.stack([
                _player_node(
                    hist[hist["player_id"].isin(_ID_TO_ALL_IDS.get(p, frozenset([p])))],
                    _minutes(p),
                )
                for p in players
            ]),
            dtype=torch.float32,
        )

    # Compute node feature matrices for home/away players
    x_home = _node_matrix(home_players)   # (11, 276)
    x_away = _node_matrix(away_players)   # (11, 276)

    # Map player_id → node index within the graph (0-10)
    home_idx = {p: i for i, p in enumerate(home_players)}
    away_idx = {p: i for i, p in enumerate(away_players)}

    # IO edges: extract same-team consecutive events and split home/away
    io = _extract_io_pairs(hist)
    home_io = io[io["team_id"] == home_team_id]
    away_io = io[io["team_id"] == away_team_id]

    # Total exposure minutes per team (normalization basis)
    home_exp = hist[hist["team_id"] == home_team_id]["game_id"].nunique() * 90.0
    away_exp = hist[hist["team_id"] == away_team_id]["game_id"].nunique() * 90.0

    # Aggregate IO edges: within home team, within away team (symmetric pairs → bidirectional edges)
    h_io_idx, h_io_attr = _aggregate_edges(home_io, home_idx, home_idx, home_exp, symmetric=True)
    a_io_idx, a_io_attr = _aggregate_edges(away_io, away_idx, away_idx, away_exp, symmetric=True)

    # ID edges: extract cross-team consecutive events and split by direction
    id_pairs = _extract_id_pairs(hist)
    # home→away direction: a home player acts right after an away player acted
    h_id = id_pairs[
        (id_pairs["src_team"] == home_team_id) & (id_pairs["dst_team"] == away_team_id)
    ]
    # away→home direction
    a_id = id_pairs[
        (id_pairs["src_team"] == away_team_id) & (id_pairs["dst_team"] == home_team_id)
    ]

    # Aggregate ID edges: only d's vaep (src_only); the o node learns the defensive pressure
    h_id_idx, h_id_attr = _aggregate_edges(h_id, home_idx, away_idx, home_exp, src_only=True)
    a_id_idx, a_id_attr = _aggregate_edges(a_id, away_idx, home_idx, away_exp, src_only=True)

    # Assemble the HeteroData
    data = HeteroData()
    data["home_team"].x = x_home                                        # (11, 48)
    data["away_team"].x = x_away                                        # (11, 48)
    data.home_elo = torch.tensor([home_elo], dtype=torch.float32)
    data.away_elo = torch.tensor([away_elo], dtype=torch.float32)
    data["home_team", "IO", "home_team"].edge_index = h_io_idx
    data["home_team", "IO", "home_team"].edge_attr  = h_io_attr
    data["away_team", "IO", "away_team"].edge_index = a_io_idx
    data["away_team", "IO", "away_team"].edge_attr  = a_io_attr
    data["home_team", "ID", "away_team"].edge_index = h_id_idx
    data["home_team", "ID", "away_team"].edge_attr  = h_id_attr
    data["away_team", "ID", "home_team"].edge_index = a_id_idx
    data["away_team", "ID", "home_team"].edge_attr  = a_id_attr
    data.y = torch.tensor(result, dtype=torch.long)   # match result label
    return data


# ── Full orchestration ─────────────────────────────────────────────────────────

def build(smoke: bool = False) -> None:
    """Build graphs for every match and save them as .pt files.

    With smoke=True, only the K1 2024 season is processed for a quick check.

    Steps:
      1. Load games.csv and sort chronologically
      2. Load vaep_oof.parquet
      3. Convert all matches to SPADL via BeproLoader
      4. Merge VAEP onto SPADL events
      5. For each match, build a graph from earlier matches' events and save it

    Already-saved .pt files are skipped (safe to re-run).

    Parameters
    ----------
    smoke : bool
        If True, process only K1 2024 matches (default: False = all)
    """
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: load match metadata and sort chronologically
    # Chronological order is crucial: the graph of match i uses only events up to match i-1
    games = pd.read_csv(VAEP_OUTPUT_DIR / "games.csv")
    games = games[games["competition_id"].isin(VALID_COMPETITION_IDS)].copy()
    games["game_date"] = pd.to_datetime(games["game_date"])
    games = games.sort_values("game_date").reset_index(drop=True)

    if smoke:
        games = games[
            (games["competition_id"] == K1_COMPETITION_ID) & (games["season"].astype(int) == 2024)
        ].copy()
        print(f"[smoke] {len(games)} matches (K1 2024)")

    # Step 2: load VAEP OOF values
    # vaep_oof.parquet: per-action VAEP values computed by run_vaep.py with LOSO
    vaep = pd.read_parquet(VAEP_OUTPUT_DIR / "vaep_oof.parquet")
    vaep["game_id"] = vaep["game_id"].astype(int)

    # Step 3: load SPADL (cache saved by run_vaep.py) or convert via BeproLoader (fallback)
    SPADL_CACHE = VAEP_OUTPUT_DIR / "spadl_all.parquet"
    if SPADL_CACHE.exists():
        print("Loading SPADL cache …")
        _cached = pd.read_parquet(SPADL_CACHE)
        actions_dict = {
            gid: df.drop(columns="game_id").reset_index(drop=True)
            for gid, df in _cached.groupby("game_id")
        }
        print(f"  Cache loaded: {len(actions_dict)} matches")
    else:
        print("Converting to SPADL … (skipped automatically once run_vaep.py has been run)")
        loader = BeproLoader(getter="local", root=RAW_DATA_DIR)
        vaep_core.load_all_games(loader)
        _, actions_dict = vaep_core.convert_games_to_spadl(
            loader=loader, games=games.copy(), verbose=True
        )
        print(f"  Conversion done: {len(actions_dict)} matches")
        if not smoke:
            pd.concat(
                [df.assign(game_id=gid) for gid, df in actions_dict.items()],
                ignore_index=True,
            ).to_parquet(SPADL_CACHE, index=False)
            print(f"  SPADL cache saved: {SPADL_CACHE}")

    # Step 4: merge VAEP values onto SPADL events
    # Matched by (game_id, action_id); unmatched rows get vaep_value=0
    all_frames = []
    for gid, acts in actions_dict.items():
        v = vaep[vaep["game_id"] == gid][["action_id", "vaep_value"]]
        merged = acts.merge(v, on="action_id", how="left")
        merged["game_id"] = gid
        all_frames.append(merged)

    all_events = pd.concat(all_frames, ignore_index=True)
    all_events["vaep_value"] = all_events["vaep_value"].fillna(0.0)

    # Step 5: build a graph per match
    skipped, built = 0, 0
    elo: dict[int, float] = {}  # team_id → current ELO (initial 1500)

    for i, row in tqdm(games.iterrows(), total=len(games), desc="building graphs"):
        gid = int(row["game_id"])
        out_path = GRAPHS_DIR / f"match_{gid}.pt"

        home_tid, away_tid = int(row["home_team_id"]), int(row["away_team_id"])
        # Match result label: 0=home loss, 1=draw, 2=home win
        hs, as_ = int(row["home_score"]), int(row["away_score"])
        result = 2 if hs > as_ else (1 if hs == as_ else 0)

        # Pre-match ELO snapshot (leakage-safe)
        h_elo = elo.get(home_tid, 1500.0)
        a_elo = elo.get(away_tid, 1500.0)

        # Always update ELO (also for matches skipped below)
        new_h, new_a = _elo_update(h_elo, a_elo, result)
        elo[home_tid] = new_h
        elo[away_tid] = new_a

        # Do not recompute graphs that already exist
        if out_path.exists():
            built += 1
            continue

        # Leakage guard: collect only IDs of matches before this one (index i)
        earlier_ids = set(games.loc[:i - 1, "game_id"].astype(int).tolist())
        hist = all_events[all_events["game_id"].isin(earlier_ids)]

        # The first match has no prior data — skip
        if hist.empty:
            skipped += 1
            continue

        # Load the lineup: identify this match's starters
        competition = "KLEAGUE1" if int(row["competition_id"]) == K1_COMPETITION_ID else "KLEAGUE2"
        season_str = str(row["season"])
        try:
            lineup = _load_lineup(competition, season_str, gid)
        except FileNotFoundError:
            skipped += 1
            continue

        home_players = lineup.get(home_tid, [])[:PLAYERS_PER_TEAM]
        away_players = lineup.get(away_tid, [])[:PLAYERS_PER_TEAM]

        data = _build_graph(
            home_players, away_players, home_tid, away_tid, hist, result,
            home_elo=h_elo, away_elo=a_elo,
        )
        if data is None:
            skipped += 1
            continue

        torch.save(data, out_path)
        built += 1

    print(f"Done — saved: {built}, skipped: {skipped} ({len(games)} matches total)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="K1 2024 only (quick sanity check)")
    args = ap.parse_args()
    build(smoke=args.smoke)
