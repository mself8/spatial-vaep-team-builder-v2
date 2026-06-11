"""
VAEP (Valuing Actions by Estimating Probabilities) pipeline — K-League / J-League

[Paper] Decroos et al., "Actions Speak Louder than Goals" (KDD 2019)
      https://dl.acm.org/doi/10.1145/3292500.3330758

[Overview]
  Trains XGBoost models to estimate how much each soccer action (pass, shot,
  dribble, ...) changes the probability of scoring/conceding within the next
  10 seconds, and converts this into per-player contributions.

[K-League pipeline]
  Bepro raw JSON
    → BeproLoader (lib/datatools/loaders/bepro.py)
    → SPADL conversion (standardized action format)
    → XGBoost feature/label computation
    → Leave-One-Season-Out OOF training
    → VAEP values (offensive / defensive / total)

[J-League pipeline]
  StatsBomb flat JSON (Statsbomb_J1_League.json, sb_matches.json)
    → flat → nested reconstruction (StatsBomb event format)
    → SPADL conversion (lib/datatools/representation/spadl/statsbomb.py)
    → VAEP prediction with the model trained on all K-League data (cross-league)
    → VAEP values
"""

import sys
import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Add lib/ to sys.path so the internal libraries can be imported.
# Resolved relative to __file__ so lib/ is found wherever this file runs from.
# ---------------------------------------------------------------------------
_LIB = Path(__file__).parent / "lib"
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))

# Data loader and SPADL conversion functions
from datatools.loaders.bepro import BeproLoader
from datatools.representation.spadl.bepro import convert_to_actions as bepro_convert_to_actions
from datatools.representation.spadl.statsbomb import convert_to_actions as sb_convert_to_actions
import datatools.representation.spadl as spadl
import datatools.vaep.spadl.features as fs   # VAEP feature functions
import datatools.vaep.spadl.labels as lab    # VAEP label functions

# ---------------------------------------------------------------------------
# VAEP feature functions (the 16 defaults from the socceraction paper)
#
# Each function maps a "game state" (current + previous N actions) to a
# feature DataFrame. Concatenating them horizontally with pd.concat yields
# the model input matrix X.
#
# e.g. fs.startlocation → 2 columns of (x, y) coordinates
#      fs.actiontype_onehot → one-hot vector of the action type
#      fs.goalscore → current score state (2 columns: attacking/defending team goals)
# ---------------------------------------------------------------------------
XFNS = [
    fs.actiontype,        # action type (integer code)
    fs.actiontype_onehot, # action type (one-hot vector)
    fs.bodypart,          # body part (foot/head etc., integer code)
    fs.bodypart_onehot,   # body part (one-hot)
    fs.result,            # action success/failure (integer code)
    fs.result_onehot,     # action result (one-hot)
    fs.goalscore,         # current score (attacking team, defending team)
    fs.startlocation,     # start coordinates (x, y)
    fs.endlocation,       # end coordinates (x, y)
    fs.movement,          # movement vector (dx, dy)
    fs.space_delta,       # change in occupied space (threat space delta)
    fs.startpolar,        # start location in polar coordinates (distance/angle to goal)
    fs.endpolar,          # end location in polar coordinates
    fs.team,              # whether consecutive actions are by the same team
    fs.time_delta,        # time gap from the previous action (seconds)
    fs.speed,             # ball movement speed (estimated m/s)
]

# Number of previous actions included in each game state.
# Features use the context of 4 actions: the current one plus the 3 before it.
NB_PREV_ACTIONS = 3


# ===========================================================================
# Data loading
# ===========================================================================

def build_loader(raw_data_dir: Path) -> BeproLoader:
    """
    Create a loader object that reads Bepro raw JSON data.

    Parameters
    ----------
    raw_data_dir : Path
        Path to the raw-data/ folder.
        Internal layout: {competition}/{season}/match/{game_id}/event_data.json etc.

    Returns
    -------
    BeproLoader
        Data loader backed by the local file system.
    """
    return BeproLoader(getter="local", root=raw_data_dir)


def load_all_games(
    loader: BeproLoader,
    competition_names: list[str] = ("KLEAGUE1", "KLEAGUE2"),
) -> pd.DataFrame:
    """
    Load match metadata for all seasons of the given leagues.

    BeproLoader organizes data in a competition/season hierarchy:
    1. List all competitions
    2. Filter by the requested competition names (KLEAGUE1, KLEAGUE2)
    3. Collect the match list for each (competition_id, season_id) pair
    4. Add 'season' (year string) and 'competition_name' columns to the games DataFrame

    Parameters
    ----------
    loader : BeproLoader
    competition_names : list[str]
        Competition names to load. Defaults to the K League first/second divisions.

    Returns
    -------
    pd.DataFrame
        Columns: game_id, home_team_id, away_team_id, competition_id,
               season_id, season (year), competition_name
    """
    # Keep only the requested competitions
    comps = loader.competitions()
    comps = comps[comps["competition_name"].isin(competition_names)]

    # Vertically concatenate the match lists of every competition×season pair
    games = pd.concat(
        [loader.games(r.competition_id, r.season_id) for r in comps.itertuples()],
        ignore_index=True,
    )

    # Map (competition_id, season_id) → season year string
    # e.g. (587, 10) → "2024"
    season_map = comps.set_index(["competition_id", "season_id"])["season_name"].to_dict()
    comp_name_map = comps.set_index("competition_id")["competition_name"].to_dict()

    games["season"] = games.apply(
        lambda r: season_map.get((r.competition_id, r.season_id), "?"), axis=1
    )
    games["competition_name"] = games["competition_id"].map(comp_name_map)
    return games


# ===========================================================================
# SPADL conversion
# ===========================================================================

def convert_games_to_spadl(
    loader: BeproLoader,
    games: pd.DataFrame,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Convert the Bepro event data of every K-League match to SPADL format.

    SPADL (Soccer Player Action Description Language) standardizes soccer
    event data into actions. Each action includes:
      - action type (20 types: pass, shot, dribble, ...)
      - start/end coordinates (standard 0-105m × 0-68m pitch)
      - body part (foot, head, other)
      - result (success/failure)
      - elapsed time

    Matches that fail to convert (e.g. incomplete data) are recorded in an
    error list and removed from the games DataFrame before returning.

    Parameters
    ----------
    loader : BeproLoader
    games : pd.DataFrame
        Return value of load_all_games().
    verbose : bool
        Whether to print progress.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        - games: DataFrame keeping only successfully converted matches
        - actions_dict: {game_id: SPADL actions DataFrame}
    """
    actions_dict: dict[int, pd.DataFrame] = {}
    errors: list[int] = []

    iterator = tqdm(list(games.itertuples()), desc="SPADL conversion") if verbose else games.itertuples()

    for game in iterator:
        try:
            # Load Bepro event data and sequence data
            events = loader.events(game.game_id)
            seqs = loader.sequences(game.game_id)

            # Bepro events → SPADL actions
            # xy_fidelity_version=2: decimal coordinates (high precision)
            # shot_fidelity_version=2: include shot details
            acts = bepro_convert_to_actions(
                events, seqs,
                home_team_id=game.home_team_id,
                xy_fidelity_version=2,
                shot_fidelity_version=2,
            )
            actions_dict[game.game_id] = acts

        except Exception as e:
            errors.append(game.game_id)
            if verbose:
                print(f"  ✗ game {game.game_id}: {e}")

    if errors and verbose:
        print(f"Conversion failed: {len(errors)} matches")

    # Return without the failed matches
    return games[~games.game_id.isin(errors)].reset_index(drop=True), actions_dict


def load_players_and_teams(loader: BeproLoader, games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load player/team info for all matches and deduplicate.

    Each match JSON contains its own player and team info, so it is requested
    per match and everything is vertically concatenated. Teams are
    deduplicated by team_id; players are kept as-is (preserving one record
    per match appearance of the same player).

    Parameters
    ----------
    loader : BeproLoader
    games : pd.DataFrame
        Matches that survived convert_games_to_spadl().

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        - teams_df: team info (team_id, team_name, ...)
        - players_df: player info (player_id, player_name, team_id, ...)
    """
    teams, players = [], []
    for game in tqdm(list(games.itertuples()), desc="Loading players/teams"):
        try:
            teams.append(loader.teams(game.game_id))
            players.append(loader.players(game.game_id))
        except Exception:
            pass  # skip matches with incomplete data

    teams_df = pd.concat(teams).drop_duplicates(subset="team_id").reset_index(drop=True)
    players_df = pd.concat(players).reset_index(drop=True)
    return teams_df, players_df


# ===========================================================================
# VAEP feature / label computation
# ===========================================================================

def _add_names(actions: pd.DataFrame) -> pd.DataFrame:
    """
    Augment the integer-code columns of SPADL actions with name columns.

    XGBoost's enable_categorical option requires string/categorical types.
    e.g. type_id=1 → type_name="pass",  bodypart_id=0 → bodypart_name="foot"

    Parameters
    ----------
    actions : pd.DataFrame
        Actions DataFrame in SPADL format.

    Returns
    -------
    pd.DataFrame
        DataFrame with type_name, bodypart_name, result_name columns added.
    """
    return spadl.add_names(actions)


def compute_features(actions: pd.DataFrame, home_team_id: int) -> pd.DataFrame:
    """
    Compute the VAEP model input feature matrix from one match's SPADL actions.

    [Processing steps]
    1. add_names: integer codes → names
    2. gamestates: bundle the current action with the previous N actions
       into a "game state"
       - row count is unchanged; previous-action info is appended as extra columns
    3. play_left_to_right: normalize the attacking direction to left→right
       regardless of home/away
       - flipping coordinates lets the model learn independently of direction
    4. Call each function in XFNS and horizontally concatenate the feature DataFrames

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL actions of one match.
    home_team_id : int
        Home team ID (used to normalize direction).

    Returns
    -------
    pd.DataFrame
        Rows = number of actions in the match,
        columns = combined outputs of all feature functions.
    """
    named = _add_names(actions)

    # game state: context of the current action plus the previous NB_PREV_ACTIONS actions
    gs = fs.gamestates(named, nb_prev_actions=NB_PREV_ACTIONS)

    # Standardize the attacking direction to always be left→right (mirror coordinates)
    gs = fs.play_left_to_right(gs, home_team_id=home_team_id)

    # Concatenate each feature function's output column-wise into the final feature matrix
    return pd.concat([fn(gs) for fn in XFNS], axis=1)


def compute_labels(actions: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the VAEP training labels from one match's SPADL actions.

    [Label definition]
    - scores_by_seconds:  1 if the acting team scores within 10 seconds of the action, else 0
    - concedes_by_seconds: 1 if the acting team concedes within 10 seconds of the action, else 0

    Training XGBoost binary classifiers on these two labels yields, for each
    action, a predicted probability of scoring and of conceding.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL actions of one match (before add_names).

    Returns
    -------
    pd.DataFrame
        Columns: scores_by_seconds, concedes_by_seconds
        Values: 0 or 1 (binary labels)
    """
    named = _add_names(actions)
    scores = lab.scores_by_seconds(named)    # goal-scored labels
    concedes = lab.concedes_by_seconds(named)  # goal-conceded labels
    return pd.concat([scores, concedes], axis=1)


# ===========================================================================
# OOF (Out-of-Fold) VAEP training and prediction
# ===========================================================================

def _train_xgb(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    """
    Train an XGBoost binary classifier for VAEP probability prediction.

    [Hyperparameter rationale]
    - n_estimators=100: conservative, to limit overfitting
    - max_depth=4: moderate complexity; deeper trees overfit noise
    - learning_rate=0.1: standard learning rate
    - enable_categorical=True: handles pandas Categorical columns in the
      feature matrix automatically (nominal variables such as the action type
      are processed by XGBoost directly, without one-hot encoding)

    Parameters
    ----------
    X_train : pd.DataFrame
        Output of compute_features(). Training feature matrix.
    y_train : pd.Series
        A single column of compute_labels(). Binary labels (0/1).

    Returns
    -------
    xgb.XGBClassifier
        Trained model. Probabilities available via predict_proba().
    """
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
        enable_categorical=True,  # native support for pandas Categorical columns
        random_state=42,
        n_jobs=-1,  # use all CPU cores
    )
    model.fit(X_train, y_train)
    return model


def run_oof_vaep(
    games: pd.DataFrame,
    actions_dict: dict,
    output_dir: Path,
    seasons: Optional[list] = None,
) -> pd.DataFrame:
    """
    Compute unbiased VAEP values for all seasons via Leave-One-Season-Out OOF.

    [Why OOF (Out-of-Fold)]
    Training and predicting on the same season lets the model reproduce the
    patterns it was trained on, yielding optimistically biased estimates.
    OOF holds out each season in turn as the test set and trains only on the
    remaining seasons, giving a fair evaluation across all seasons.

    [Per-fold steps]
    1. Split: held-out season = test set, remaining seasons = training set
    2. Train separate XGBoost models for scoring and conceding on the training set
    3. Apply both models to the test set → per-action P(score), P(concede)
    4. Record AUC (model quality check)
    5. Store predictions → converted to VAEP values in the next step

    [Overall result]
    Combining the predictions of all folds yields unbiased VAEP across all
    seasons. The final result is saved to vaep_oof.parquet.

    Parameters
    ----------
    games : pd.DataFrame
        Must include game_id, season, home_team_id columns.
    actions_dict : dict
        {game_id: SPADL DataFrame} — return value of convert_games_to_spadl().
    output_dir : Path
        Folder for the output files.
    seasons : list, optional
        Seasons to use as folds. If None, derived from games.

    Returns
    -------
    pd.DataFrame
        Columns: game_id, action_id, player_id, team_id, ...
              p_scores, p_concedes, offensive_value, defensive_value, vaep_value
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if seasons is None:
        seasons = sorted(games["season"].unique())

    # -----------------------------------------------------------------------
    # Precompute all features/labels (once, before the OOF loop)
    # Storing each match's features and labels in dicts keyed by game_id lets
    # us split folds quickly via pd.concat without recomputing per fold.
    # -----------------------------------------------------------------------
    print("Computing features/labels...")
    feats: dict[int, pd.DataFrame] = {}      # {game_id: feature DataFrame}
    labels: dict[int, pd.DataFrame] = {}     # {game_id: label DataFrame}
    action_rows: dict[int, pd.DataFrame] = {} # {game_id: raw action metadata DataFrame}

    for game in tqdm(list(games.itertuples()), desc="Computing features"):
        gid = game.game_id
        if gid not in actions_dict:
            continue
        acts = actions_dict[gid]
        try:
            feats[gid] = compute_features(acts, game.home_team_id)
            labels[gid] = compute_labels(acts)
            # Keep only the metadata columns needed to assign VAEP values (saves memory)
            action_rows[gid] = acts[
                ["game_id", "action_id", "period_id", "time_seconds",
                 "team_id", "player_id", "type_id", "result_id"]
            ].copy()
        except Exception as e:
            print(f"  ✗ feature {gid}: {e}")

    # -----------------------------------------------------------------------
    # OOF loop: repeat the train/test split per season
    # -----------------------------------------------------------------------
    all_preds: list[pd.DataFrame] = []
    fold_metrics: list[dict] = []

    for season in seasons:
        # Held-out season = test, the rest = train
        test_ids = set(games[games["season"] == season]["game_id"])
        train_ids = set(games[games["season"] != season]["game_id"])

        # Use only matches whose features were computed successfully
        test_ids = test_ids & set(feats.keys())
        train_ids = train_ids & set(feats.keys())

        if not test_ids or not train_ids:
            print(f"  [skip] season {season}: train={len(train_ids)}, test={len(test_ids)}")
            continue

        # Assemble train/test feature and label matrices
        X_train = pd.concat([feats[g] for g in train_ids])
        y_scores_train = pd.concat([labels[g]["scores_by_seconds"] for g in train_ids])
        y_concedes_train = pd.concat([labels[g]["concedes_by_seconds"] for g in train_ids])

        X_test = pd.concat([feats[g] for g in test_ids])
        y_scores_test = pd.concat([labels[g]["scores_by_seconds"] for g in test_ids])
        y_concedes_test = pd.concat([labels[g]["concedes_by_seconds"] for g in test_ids])

        print(f"\n[fold {season}] train={len(train_ids)} matches, test={len(test_ids)} matches")

        # Train the scoring model and the conceding model independently
        model_scores = _train_xgb(X_train, y_scores_train)
        model_concedes = _train_xgb(X_train, y_concedes_train)

        # Predict test-set probabilities ([:, 1] → positive class = scoring/conceding occurs)
        p_scores = model_scores.predict_proba(X_test)[:, 1]
        p_concedes = model_concedes.predict_proba(X_test)[:, 1]

        # Record model performance via AUC (target: ≥ 0.73, achieved: ~0.91)
        auc_s = roc_auc_score(y_scores_test, p_scores)
        auc_c = roc_auc_score(y_concedes_test, p_concedes)
        print(f"  AUC scores: {auc_s:.4f}, AUC concedes: {auc_c:.4f}")
        fold_metrics.append({
            "season": season,
            "auc_scores": auc_s,
            "auc_concedes": auc_c,
            "n_test": len(test_ids),
        })

        # Join predictions with action metadata and store
        test_acts = pd.concat([action_rows[g] for g in test_ids]).reset_index(drop=True)
        test_acts["p_scores"] = p_scores
        test_acts["p_concedes"] = p_concedes
        all_preds.append(test_acts)

    # -----------------------------------------------------------------------
    # Combine all folds' predictions, compute VAEP values, and save
    # -----------------------------------------------------------------------
    print("\nComputing VAEP values...")
    combined = pd.concat(all_preds, ignore_index=True)
    combined = _compute_vaep_values(combined)

    combined.to_parquet(output_dir / "vaep_oof.parquet", index=False)
    with open(output_dir / "vaep_oof_metrics.json", "w") as f:
        metrics = {
            "folds": fold_metrics,
            "mean_auc_scores": np.mean([m["auc_scores"] for m in fold_metrics]),
            "mean_auc_concedes": np.mean([m["auc_concedes"] for m in fold_metrics]),
        }
        json.dump(metrics, f, indent=2)

    print(f"\n✓ Done: VAEP computed for {len(combined):,} actions")
    print(f"  Mean AUC scores:   {metrics['mean_auc_scores']:.4f}")
    print(f"  Mean AUC concedes: {metrics['mean_auc_concedes']:.4f}")
    return combined


def _compute_vaep_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert predicted probabilities (p_scores, p_concedes) into VAEP values.

    [socceraction paper formulas]
    For the game-state transition S_{a-1} → S_a caused by action a:

      V_off(a) = P_score(S_a)   − P_score(S_{a-1})
                 ← how much the scoring probability increased

      V_def(a) = P_concede(S_{a-1}) − P_concede(S_a)
                 ← how much the conceding probability decreased

      V(a)     = V_off(a) + V_def(a)
                 ← offensive contribution + defensive contribution

    Examples:
      - shot just before a goal: scoring prob rises 0.01→0.60 → V_off = +0.59 (very high)
      - giveaway pass: conceding prob rises 0.01→0.15 → V_def = -0.14 (negative)
      - routine sideways pass: negligible probability change → V ≈ 0

    [Implementation note]
    shift() must be applied independently per match (game_id).
    Shifting the whole DataFrame across match boundaries would make the
    first action of a match inherit the last action of the previous match.

    Parameters
    ----------
    df : pd.DataFrame
        Must include p_scores, p_concedes columns. May mix multiple matches.

    Returns
    -------
    pd.DataFrame
        Adds offensive_value, defensive_value, vaep_value columns.
    """
    parts = []
    for gid, gdf in df.groupby("game_id"):
        # Sort chronologically within the match (by period, then time)
        gdf = gdf.sort_values(["period_id", "time_seconds"]).reset_index(drop=True)

        # 'Previous' probability of each action: push down one row with shift(1)
        # The first action of a match has no predecessor, so fill with 0 (kickoff state = probability 0)
        prev_p_scores = gdf["p_scores"].shift(1).fillna(0)
        prev_p_concedes = gdf["p_concedes"].shift(1).fillna(0)

        gdf["offensive_value"] = gdf["p_scores"] - prev_p_scores
        gdf["defensive_value"] = prev_p_concedes - gdf["p_concedes"]
        gdf["vaep_value"] = gdf["offensive_value"] + gdf["defensive_value"]
        parts.append(gdf)

    return pd.concat(parts, ignore_index=True)


# ===========================================================================
# J-League data loading and SPADL conversion
# ===========================================================================

# Action prefixes handled when building the extra dict from StatsBomb flat JSON.
# Dot-notation columns starting with these prefixes go into the extra dict.
# e.g. "pass.outcome.name" → extra["pass"]["outcome"]["name"]
_SB_ACTION_PREFIXES = {
    "pass", "carry", "dribble", "duel", "clearance", "shot",
    "goalkeeper", "foul_committed", "foul_won", "ball_recovery",
    "ball_receipt", "interception", "substitution", "injury_stoppage",
    "50_50", "miscontrol", "block", "bad_behaviour", "player_off", "tactics",
}

# Rename StatsBomb flat columns → StatsBombLoader.events() output columns
_SB_RENAME = {
    "id":                        "event_id",
    "match_id":                  "game_id",
    "period":                    "period_id",
    "type.id":                   "type_id",
    "type.name":                 "type_name",
    "possession_team.id":        "possession_team_id",
    "possession_team.name":      "possession_team_name",
    "play_pattern.id":           "play_pattern_id",
    "play_pattern.name":         "play_pattern_name",
    "team.id":                   "team_id",
    "team.name":                 "team_name",
    "player.id":                 "player_id",
    "player.name":               "player_name",
    "position.id":               "position_id",
    "position.name":             "position_name",
}


def load_jleague_games(raw_data_dir: Path) -> pd.DataFrame:
    """
    Load J-League 2024 match metadata.

    The StatsBomb J1 League data stores all match info in a single file,
    sb_matches.json, in flat form (dot-notation columns).
    This function normalizes it to the standard column names used by the
    VAEP pipeline.

    Parameters
    ----------
    raw_data_dir : Path
        Path to the raw-data/ folder. Must contain a J-league1/ subfolder.

    Returns
    -------
    pd.DataFrame
        Columns: game_id, home_team_id, away_team_id, season, competition_name
    """
    path = raw_data_dir / "J-league1" / "sb_matches.json"
    matches = pd.read_json(path)
    matches = matches.rename(columns={
        "match_id":                     "game_id",
        "home_team.home_team_id":       "home_team_id",
        "home_team.home_team_name":     "home_team_name",
        "away_team.away_team_id":       "away_team_id",
        "away_team.away_team_name":     "away_team_name",
        "season.season_name":           "season",
        "competition.competition_name": "competition_name",
    })
    matches["season"] = matches["season"].astype(str)
    return matches[["game_id", "home_team_id", "home_team_name",
                    "away_team_id", "away_team_name", "season", "competition_name"]]


def _build_extra(row: pd.Series) -> dict:
    """
    Reconstruct the StatsBomb event 'extra' dict from flat dot-notation columns.

    StatsBombLoader.events() parses nested JSON and stores per-action details
    in an 'extra' column (dict). The J-League data, however, comes already
    flattened (e.g. "pass.outcome.name", "shot.body_part.name"), so we must
    split on dots and rebuild the nested dict for the StatsBomb SPADL
    conversion function to work correctly.

    Examples:
      "pass.outcome.name" = "Complete"
        → extra["pass"]["outcome"]["name"] = "Complete"
      "shot.body_part.id" = 40
        → extra["shot"]["body_part"]["id"] = 40

    Parameters
    ----------
    row : pd.Series
        One row of the flat events DataFrame.

    Returns
    -------
    dict
        Same nested structure as the 'extra' column of StatsBombLoader.events().
    """
    extra: dict = {}
    for col, val in row.items():
        # Columns without a dot (type.id, period, etc.) are not included in extra
        if "." not in str(col):
            continue
        # Only columns whose prefix is in _SB_ACTION_PREFIXES go into extra
        prefix = col.split(".")[0]
        if prefix not in _SB_ACTION_PREFIXES:
            continue
        # Skip None / NaN / empty lists
        # pd.read_json turns JSON null into None (object) or float NaN depending on the column type
        if val is None:
            continue
        if isinstance(val, float) and pd.isna(val):
            continue
        if isinstance(val, list) and len(val) == 0:
            continue
        # Build the nested dict along the dot path
        parts = col.split(".")
        d = extra
        for k in parts[:-1]:
            d = d.setdefault(k, {})
        d[parts[-1]] = val
    return extra


def _flat_to_sb_events(flat_events: pd.DataFrame, game_id: int) -> pd.DataFrame:
    """
    Convert flat StatsBomb JSON rows to the StatsBombLoader.events() output format.

    The StatsBomb SPADL conversion function (sb_convert_to_actions) expects
    the standard column layout returned by StatsBombLoader. This function
    reshapes the flat J-League events accordingly:
      1. rename columns (type.id → type_id, ...)
      2. convert timestamp to timedelta
      3. rebuild the extra dict (via _build_extra)

    Parameters
    ----------
    flat_events : pd.DataFrame
        Full contents of Statsbomb_J1_League.json.
    game_id : int
        ID of the match to process.

    Returns
    -------
    pd.DataFrame
        Same column layout as StatsBombLoader.events().
    """
    # Select this match only, then normalize column names
    e = flat_events[flat_events["match_id"] == game_id].copy()
    e = e.rename(columns=_SB_RENAME)

    # Convert timestamp to timedelta (required inside convert_to_actions)
    e["timestamp"] = pd.to_timedelta(e["timestamp"])

    # related_events: NaN → empty list
    e["related_events"] = e["related_events"].apply(
        lambda d: d if isinstance(d, list) else []
    )

    # Normalize bool columns
    e["under_pressure"] = e["under_pressure"].fillna(False).astype(bool)
    if "counterpress" not in e.columns:
        e["counterpress"] = False
    e["counterpress"] = e["counterpress"].fillna(False).astype(bool)

    # Clean location: replace malformed coordinates from JSON parsing (e.g. ['null']) with None
    # A valid coordinate must be a length-2 list of [float, float].
    def _clean_loc(v):
        if isinstance(v, list) and len(v) == 2:
            try:
                float(v[0]); float(v[1])
                return v
            except (TypeError, ValueError):
                pass
        return None
    e["location"] = e["location"].apply(_clean_loc)

    # Rebuild the extra dict (per row: dot-notation columns → nested dict)
    e["extra"] = e.apply(_build_extra, axis=1)

    return e


def load_jleague_events(raw_data_dir: Path) -> pd.DataFrame:
    """
    Load all J-League 2024 match events from a single flat JSON file.

    The StatsBomb J1 League events are not split per match: all matches
    (380 matches, ~1.25M rows) are stored in one JSON file.
    This DataFrame is grouped by game_id in convert_jleague_to_spadl().

    Parameters
    ----------
    raw_data_dir : Path

    Returns
    -------
    pd.DataFrame
        ~1,257,772 rows, matches identified by the match_id column.
    """
    path = raw_data_dir / "J-league1" / "Statsbomb_J1_League.json"
    print(f"Loading J-League events: {path}")
    # convert_dates=False: the 'timestamp' column holds in-match time of the
    # form "00:00:00.000", so keep pandas from auto-parsing it as dates.
    return pd.read_json(path, convert_dates=False)


def convert_jleague_to_spadl(
    games: pd.DataFrame,
    all_events: pd.DataFrame,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Convert J-League match events to SPADL format.

    [Processing steps]
    1. Group flat events by match ID (pre-grouped for speed)
    2. For each match, rebuild the StatsBomb format via _flat_to_sb_events()
    3. Call sb_convert_to_actions() (StatsBomb SPADL conversion)
    4. Drop matches that fail to convert from games

    The return format matches the K-League convert_games_to_spadl(), so the
    downstream compute_features / compute_labels functions can be reused as-is.

    Parameters
    ----------
    games : pd.DataFrame
        Return value of load_jleague_games().
    all_events : pd.DataFrame
        Return value of load_jleague_events() (flat events of all matches).
    verbose : bool

    Returns
    -------
    tuple[pd.DataFrame, dict]
        - games: DataFrame keeping only successfully converted matches
        - actions_dict: {game_id: SPADL actions DataFrame}
    """
    # Pre-group flat events by match (avoids repeated filtering in the loop)
    grouped = {gid: grp for gid, grp in all_events.groupby("match_id")}

    # Build the dummy sequences DataFrame required by the StatsBomb SPADL conversion
    # (the converter takes sequences as an argument but does not use it internally)
    def _dummy_seqs(game_id: int) -> pd.DataFrame:
        return pd.DataFrame({
            "game_id":   [game_id, game_id],
            "period_id": [1, 2],
            "team_id":   [float("nan"), float("nan")],
            "start_time": [float("nan"), float("nan")],
            "end_time":   [float("nan"), float("nan")],
            "event_ids":  [[], []],
        })

    actions_dict: dict[int, pd.DataFrame] = {}
    errors: list[int] = []
    iterator = tqdm(list(games.itertuples()), desc="J-League SPADL conversion") if verbose else games.itertuples()

    for game in iterator:
        gid = game.game_id
        if gid not in grouped:
            errors.append(gid)
            continue
        try:
            events = _flat_to_sb_events(grouped[gid], gid)
            acts = sb_convert_to_actions(
                events,
                _dummy_seqs(gid),
                home_team_id=game.home_team_id,
                xy_fidelity_version=2,
                shot_fidelity_version=2,
            )
            actions_dict[gid] = acts
        except Exception as e:
            errors.append(gid)
            if verbose:
                print(f"  ✗ game {gid}: {e}")

    if errors and verbose:
        print(f"Conversion failed: {len(errors)} matches")
    return games[~games.game_id.isin(errors)].reset_index(drop=True), actions_dict


def run_jleague_vaep(
    kleague_games: pd.DataFrame,
    kleague_actions_dict: dict,
    jleague_games: pd.DataFrame,
    jleague_actions_dict: dict,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Train the model on all K-League data and apply VAEP to the J-League.

    [Why cross-league VAEP]
    The J-League covers a single season (2024), so OOF is not possible.
    Instead, all K-League seasons (2021-2025) are used as training data and
    the J-League is treated as the test set when predicting scoring/conceding
    probabilities. This evaluates the J-League on a K-League-calibrated
    "action value" scale.

    [Caveat] Distribution shift due to differing play styles between leagues
    may exist, but this is an acceptable assumption for league-level comparison.

    Parameters
    ----------
    kleague_games : pd.DataFrame
        K-League match metadata (includes game_id, home_team_id, season).
    kleague_actions_dict : dict
        {game_id: SPADL DataFrame} — all K-League seasons.
    jleague_games : pd.DataFrame
        J-League match metadata.
    jleague_actions_dict : dict
        {game_id: SPADL DataFrame} — J-League 2024.
    output_dir : Path
        Output path.

    Returns
    -------
    pd.DataFrame
        Per-action J-League VAEP values (offensive_value, defensive_value, vaep_value).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Compute features/labels for all K-League seasons (training data)
    # -----------------------------------------------------------------------
    print("Computing K-League features/labels (for training)...")
    kl_feats, kl_labels = {}, {}
    for game in tqdm(list(kleague_games.itertuples()), desc="K-League Feature"):
        gid = game.game_id
        if gid not in kleague_actions_dict:
            continue
        acts = kleague_actions_dict[gid]
        try:
            kl_feats[gid] = compute_features(acts, game.home_team_id)
            kl_labels[gid] = compute_labels(acts)
        except Exception as e:
            print(f"  ✗ feature {gid}: {e}")

    X_train = pd.concat(kl_feats.values())
    y_scores_train = pd.concat([v["scores_by_seconds"] for v in kl_labels.values()])
    y_concedes_train = pd.concat([v["concedes_by_seconds"] for v in kl_labels.values()])

    print(f"\nK-League training data: {len(X_train):,} actions")

    # Train the scoring model and the conceding model
    print("Training XGBoost (scoring model)...")
    model_scores = _train_xgb(X_train, y_scores_train)
    print("Training XGBoost (conceding model)...")
    model_concedes = _train_xgb(X_train, y_concedes_train)

    # -----------------------------------------------------------------------
    # Compute J-League features and predict VAEP
    # -----------------------------------------------------------------------
    print("\nComputing J-League features and predicting VAEP...")
    jl_feats, jl_action_rows = {}, {}
    for game in tqdm(list(jleague_games.itertuples()), desc="J-League Feature"):
        gid = game.game_id
        if gid not in jleague_actions_dict:
            continue
        acts = jleague_actions_dict[gid]
        try:
            jl_feats[gid] = compute_features(acts, game.home_team_id)
            jl_action_rows[gid] = acts[
                ["game_id", "action_id", "period_id", "time_seconds",
                 "team_id", "player_id", "type_id", "result_id"]
            ].copy()
        except Exception as e:
            print(f"  ✗ J-League feature {gid}: {e}")

    X_test = pd.concat(jl_feats.values())
    p_scores = model_scores.predict_proba(X_test)[:, 1]
    p_concedes = model_concedes.predict_proba(X_test)[:, 1]

    # Join predictions with action metadata
    test_acts = pd.concat(jl_action_rows.values()).reset_index(drop=True)
    test_acts["p_scores"] = p_scores
    test_acts["p_concedes"] = p_concedes

    # Compute VAEP values (per-match shift)
    result = _compute_vaep_values(test_acts)
    result.to_parquet(output_dir / "vaep_jleague.parquet", index=False)

    print(f"\n✓ Done: VAEP computed for {len(result):,} J-League actions")
    print(f"  VAEP range: {result['vaep_value'].min():.4f} ~ {result['vaep_value'].max():.4f}")
    return result
