"""bepro data to SPADL converter."""
from typing import Any, cast, Optional
import numpy as np
import pandas as pd  # type: ignore
from pandera.typing import DataFrame

from . import config as spadlconfig
from .base import (
    _add_dribbles,
    _fix_clearances,
    _fix_block,
    _fix_recovery,
    _add_dribbles_after_receive,
    shift_with_edge_fix,
    # _fix_direction_of_play,
)
from .schema import SPADLSchema

from .config import (
    field_length,
    field_width,
    HEIGHT_POST,
    TOUCH_LINE_LENGTH,
    GOAL_LINE_LENGTH,
    LEFT_POST,
    RIGHT_POST,
    CENTER_POST,
    ORIGINAL_LEFT_POST,
    ORIGINAL_RIGHT_POST,
    Eighteen_YARD,
)


def convert_to_actions(
    events: pd.DataFrame, 
    sequences: pd.DataFrame, 
    home_team_id: int, 
	xy_fidelity_version: Optional[int] = None,
	shot_fidelity_version: Optional[int] = None,
) -> DataFrame[SPADLSchema]:
    """
    Convert K-league events to SPADL actions.
    """

    events = events.sort_values(
        ["period_id", "event_time"], 
        kind="mergesort"
    ).reset_index(drop=True) 
    # Remove events not used in the analysis: missing values & duplicates
    events = _clean_events(events, remove_event_types=["Substitution", "Control Under Pressure"])
    events = _sort_sequence(events, sequences) 

    events = _convert_pass_locations(events) # estimate and impute pass end locations
    events = _convert_shot_locations(events) # estimate and impute shot end locations

    # Fix the playing direction of both teams: the home team plays bottom-to-top, the away team top-to-bottom.
    events = _fix_direction_of_play(events, home_team_id)

    events = _fix_offside(events)
    events = _fix_defensive_line_support(events)    # Convert defensive_line_support to tackle or interception

    # Split events where an offensive and a defensive event occur at the same time.
    events = insert_defensive_actions(events, defensive_action="Interception")
    events = insert_defensive_actions(events, defensive_action="Tackle")

    # Convert actions from the K-league dataset format to SPADL format
    events["type_name"] = events["event_types"].apply(_get_type_name)

    # For each action, define the action ID, body part, result, and end location
    events[["type_id", "bodypart_id", "result_id"]] = events.apply(_parse_event, axis=1, result_type="expand")

    actions = pd.DataFrame()
    actions["game_id"] = events.game_id.astype(int)
    actions["original_event_id"] = events.event_id.astype(object)
    actions["period_id"] = events.period_id.astype(int)

    # convert milliseconds to seconds
    # First half kick-off: 0(ms), second half kick-off: 2,700,000(ms)
    actions["time_seconds"] = (
        events["event_time"] * 0.001 
        - ((events.period_id > 1) * 45 * 60) # convert 45(minutes) to 45*60(seconds)
        - ((events.period_id > 2) * 45 * 60)
        - ((events.period_id > 3) * 15 * 60)
        - ((events.period_id > 4) * 15 * 60)
    )
     
    actions["team_id"] = events.team_id
    actions["player_id"] = events.player_id

    # Convert the K-league pitch layout to the SPADL one: 68x105 -> 105x68
    actions["start_x"] = events.y
    actions["start_y"] = GOAL_LINE_LENGTH - events.x
    actions["end_x"] = events.end_y
    actions["end_y"] = GOAL_LINE_LENGTH - events.end_x

    actions["type_id"] = events.type_id.astype(int)
    actions["bodypart_id"] = events.bodypart_id.astype(int)
    actions["result_id"] = events.result_id.astype(int)

    actions = (
        actions[actions.type_id != spadlconfig.actiontypes.index("non_action")]
        .sort_values(["period_id", "time_seconds"], kind="mergesort")
        .reset_index(drop=True)
    )

    actions = _fix_tackle_result(actions) # fix the result of defensive events with no recorded outcome.
    actions = _fix_dribble(actions)  # adjust dribble end locations
    actions = _fix_clearances(actions)

    actions["action_id"] = range(len(actions))
    actions = _add_dribbles(actions) # TODO: _add_dribbles_after_receive can generate dribbles more accurately, but is excluded for the baseline

    return cast(DataFrame[SPADLSchema], actions)

def _sort_sequence(df_events: pd.DataFrame, sequences: pd.DataFrame) -> pd.DataFrame:
    """
        Bepro data is not sorted in the order in which the events occurred.
        The cause is unknown, but the event sequence data (the sequences
        DataFrame) must be used to sort the events into their order of occurrence.
    """
    def _insert_seq_events(period_group: pd.DataFrame, period_sequence) -> pd.DataFrame:
        period_group = period_group.copy().set_index('event_id') # indexing: faster

        # Extract every event_id (list) contained in the sequence data
        seq_event_ids = [
            event_id 
            for event_ids in period_sequence["event_ids"] 
            for event_id in event_ids # event_ids: list
            if event_id in period_group.index # exclude events removed by the clean_events function
        ]

        seq_events = period_group.loc[seq_event_ids]
        seq_events = seq_events[
            ~seq_events.index.duplicated(keep="first") # note: the sequence data contains duplicates, e.g. at the start or end of a sequence
        ].reset_index(drop=False) # restore event_id
        
        # Insert events not contained in the sequence data.
        # # idxmax: Return index of "first" occurrence of maximum over requested axis.
        # ex) if time is 6, event_time = [1, 2, 7, 8] -> idxmax=[False, False, True, True] -> insert_time = 6 -> [1, 2, 6, 7, 8]
        not_seq_events = period_group[~period_group.index.isin(seq_events.event_id)].reset_index(drop=False)
        not_seq_events.index = not_seq_events["event_time"].apply(
            lambda time: (seq_events["event_time"] > time).idxmax() 
            if any(seq_events["event_time"] > time) else len(seq_events)
        )

        return pd.concat(
            [not_seq_events, seq_events], 
            ignore_index=False,
            axis=0
        ).sort_index(kind="mergesort").reset_index(drop=True)
    
    df_events = df_events.groupby("period_id").apply(
        lambda group: _insert_seq_events(group, sequences[sequences["period_id"] == group.name])
        )

    return df_events.reset_index(drop=True)

def _clean_events(df_events: pd.DataFrame, remove_event_types) -> pd.DataFrame:
    """
    Remove specific event types from the events DataFrame.
    remove_event_types (list): event types to remove

    - Missing-value conditions (missing_cond)
    1.24	[Duel]	[Aerial]	...	NaN	NaN	NaN	NaN	NaN	NaN	[{'event_type': 'Duel', 'sub_event_type': 'Aerial...	NaN	NaN	35
    67	85848	94916404	1	4641.0	259769.0	163181	0.193699	0.5342 event_types not recorded -> parsing impossible; cannot be inferred from the previous info alone
    2. event_types recorded but team_id & player_id info missing -> not inferable from the previous info, but inferable from the next.
    
    - Duplicate-data conditions (duplicated_cond)
    1. duplicated event_id -> remove
    2. different event_id but otherwise duplicated data -> remove
    """

    # Remove only the given event types
    # ex) Pass + Control Under Pressure -> Pass 
    df_events["event_types"] = df_events["event_types"].apply(
        lambda event_list: [event for event in event_list if event.get("event_type") not in remove_event_types]
    )

    # Drop rows whose event list is empty, either originally or after removing remove_event_types
    missing_cond = (
        (df_events['event_types'].apply(len) == 0)
        # | (events["team_id"].isna())   # we do not remove missing team_id to better capture the context
        # | (events["player_id"].isna()) # we do not remove missing player_id to better capture the context
    )
    df_events = df_events[~missing_cond].reset_index(drop=True)

    # keep=first: keep only the first occurrence among duplicates and drop the rest (only the first is False)
    df = df_events.copy()

    # duplicated_cond1: duplicated event_id
    duplicated_cond1 = df.duplicated(subset="event_id", keep="first") # only the first occurrence is False -> ~False=True

    # duplicated_cond2: different event_id but all other data duplicated
    non_event_cols = [col for col in df.columns if col != "event_id"] # columns other than event_id
    for col in non_event_cols:
        df[col] = df[col].apply(lambda x: str(x) if isinstance(x, list) or isinstance(x, dict) else x) # the duplicated function does not support list or dict
    duplicated_cond2 = df.duplicated(subset=non_event_cols, keep="first") # only the first occurrence is False -> ~False=True
    
    return df_events[
        ~(duplicated_cond1 | duplicated_cond2)
    ].reset_index(drop=True)

def _parse_event(event : pd.Series) -> tuple[int, int, float, float]:
    # 23 possible values : pass, cross, throw-in, 
    # crossed free kick, short free kick, crossed corner, short corner, 
    # take-on, foul, tackle, interception, 
    # shot, penalty shot, free kick shot, 
    # keeper save, keeper claim, keeper punch, keeper pick-up, 
    # clearance, bad touch, dribble and goal kick.
    events = {
        "pass": _parse_pass_event,
        "cross": _parse_pass_event,
        "throw_in": _parse_pass_event,
        "freekick_crossed": _parse_pass_event,
        "freekick_short": _parse_pass_event,
        "corner_crossed": _parse_pass_event,
        "corner_short": _parse_pass_event,

        "take_on": _parse_take_on_event,

        "foul": _parse_foul_event,

        "tackle" : _parse_tackle_event,

        "interception": _parse_interception_event,

        "shot": _parse_shot_event,
        "shot_penalty": _parse_shot_event,
        "shot_freekick": _parse_shot_event,

        "keeper_save" : _parse_goalkeeper_event,
        "keeper_claim" : _parse_goalkeeper_event,
        "keeper_punch" : _parse_goalkeeper_event,
        "keeper_pick_up" : _parse_goalkeeper_event,
        "Defensive_Line_Support" : _parse_goalkeeper_event,

        "clearance" : _parse_clearance_event,
        "bad_touch" : _parse_bad_touch_event,
        "dribble" : _parse_dribble_event,

        "goalkick" : _parse_pass_event,
    }

    parser = events.get(event["type_name"], _parse_event_as_non_action)
    bodypart, result = parser(event)

    actiontype = spadlconfig.actiontypes.index(event["type_name"])
    bodypart = spadlconfig.bodyparts.index(bodypart)
    result = spadlconfig.results.index(result)
    
    return actiontype, bodypart, result

def _get_type_name(event_types: list) -> str:
    if any(e["event_type"] == "Pass" for e in event_types):
        pass_dict = next(e for e in event_types if e["event_type"] == "Pass")
        if pass_dict.get("cross", False):
            if any(e.get("sub_event_type") == "Freekick" for e in event_types):
                a = "freekick_crossed"
            elif any(e.get("sub_event_type") == "Corner" for e in event_types):
                a = "corner_crossed"
            else:
                a = "cross"
        else:
            if any(e.get("sub_event_type") == "Freekick" for e in event_types):
                a = "freekick_short"
            elif any(e.get("sub_event_type") == "Corner" for e in event_types):
                a = "corner_short"
            elif any(e.get("sub_event_type") == "Throw-In" for e in event_types):
                a = "throw_in"
            elif any(e.get("sub_event_type") == "Goal Kick" for e in event_types):
                a = "goalkick"
            else:
                a = "pass"
    elif any(e["event_type"] == "Shot" for e in event_types):
        if any(e.get("sub_event_type") == "Freekick" for e in event_types):
            a = "shot_freekick"
        elif any(e.get("sub_event_type") == "Penalty Kick" for e in event_types):
            a = "shot_penalty"
        else:
            a = "shot"
    elif any(e["event_type"] == "Take-On" for e in event_types):
        a = "take_on"
    elif any(e["event_type"] == "Step-in" for e in event_types): # API 2025: rename Carry to Step-in
        a = "dribble"
    elif any(e["event_type"] == "Save" for e in event_types): # goalkeeper actions: Save, Aerial Clearnce, Defensive Line Support Succeeded
        if any(e.get("sub_event_type") == "Catch" for e in event_types):
            a = "keeper_save"
        elif any(e.get("sub_event_type") == "Parry" for e in event_types):
            a = "keeper_punch"
        else:
            a = "non_action" # Save only comes as Catch or Parry; no other cases exist
    elif any((e["event_type"] == "Aerial Clearance") & (e.get("outcome") == "Successful") for e in event_types):  
        # A failed Aerial Clearance does not affect the ball's direction, so treat it as non_action
        a = "keeper_claim"
    elif any(e["event_type"] == "Clearance" for e in event_types):
        a = "clearance"
    elif any(e["event_type"] == "Foul" for e in event_types):
        a = "foul"
    elif any(e["event_type"] in ["Tackle", "Intervention"] for e in event_types):
        a = "tackle"
    elif any(e["event_type"] == "Interception" for e in event_types):
        a = "interception"
    elif any(e["event_type"] == "Error" for e in event_types) or any(e["event_type"] == "Own Goal" for e in event_types): # own goals are also defined as bad_touch
        a = "bad_touch"
    else:
        a = "non_action"
    
    return a

def _fix_direction_of_play(df_events: pd.DataFrame, home_team_id: int) -> pd.DataFrame:
    away_idx = (df_events.team_id != home_team_id).values
    for col in ["x", "end_x"]:
        df_events.loc[away_idx, col] = GOAL_LINE_LENGTH - df_events.loc[away_idx, col].values
    for col in ["y", "end_y"]: 
        df_events.loc[away_idx, col] = TOUCH_LINE_LENGTH - df_events.loc[away_idx, col].values

    return df_events

def _fix_defensive_line_support(df_events: pd.DataFrame) -> pd.DataFrame:
    """Convert Defensive_Line_Support events to interception"""  
    df_events_next = shift_with_edge_fix(df_events, shift_value=-1)
    cond_defensive_line_support = df_events["event_types"].apply(lambda x: any(e["event_type"] == "Defensive Line Support" for e in x))
    cond_tackle = df_events["event_types"].apply(lambda x: any(e["event_type"] == "Tackle" for e in x))
    same_player = df_events.player_id == df_events_next.player_id

    cond_interception = cond_defensive_line_support & same_player & ~cond_tackle # if the player keeps possession after the defensive action, define it as an Interception
    cond_tackle = cond_defensive_line_support & (~same_player | cond_tackle)  # if the player does not keep possession after the defensive action, define it as a Tackle

    df_events.loc[cond_interception , "event_types"] = df_events.loc[cond_interception , "event_types"].apply(
        lambda event_list: event_list + [{"event_type": "Interception"}]
    )
    df_events.loc[cond_tackle , "event_types"] = df_events.loc[cond_tackle , "event_types"].apply(
        lambda event_list: event_list + [{"event_type": "Tackle", "outcome": next(e.get("outcome") for e in event_list if e["event_type"] == "Defensive Line Support")}]
    )

    return df_events

def _fix_offside(df_events: pd.DataFrame) -> pd.DataFrame:
    df_events_next = shift_with_edge_fix(df_events, shift_value=-1)

    cond_pass = df_events["event_types"].apply(
        lambda x: any(e["event_type"] == "Pass" for e in x)
    )
    cond_set_piece = df_events["event_types"].apply(
        lambda x: any(e.get("sub_event_type") in ["Corner", "Freekick"] for e in x)
    )
    cond_next_offside = df_events_next["event_types"].apply(
        lambda x: any(e.get("event_type") == "Offside" for e in x)
    )

    df_events.loc[cond_pass & cond_next_offside, "event_types"] = df_events.loc[cond_pass & cond_next_offside, "event_types"].apply(
        lambda event_list: [{**e, "outcome": "offside"} if e.get("event_type") == "Pass" else e for e in event_list]
    )
    df_events.loc[cond_set_piece & cond_next_offside, "event_types"] = df_events.loc[cond_set_piece & cond_next_offside, "event_types"].apply(
        lambda event_list: [{**e, "outcome": "offside"} if e.get("sub_event_type") in ["Corner", "Freekick"] else e for e in event_list]
    )

    return df_events

def _fix_dribble(df_actions: pd.DataFrame) -> pd.DataFrame:
    """
        _fix_dribble : Update the end position of dribble events based on their success or failure.
        If the dribble failed, the end position is set to the position of the next event.
        If the dribble succeeded, the end position is set to the position of the next event that is not a tackle.
    """

    df_actions_next = shift_with_edge_fix(df_actions, shift_value=-1)

    failed_tackle = (
        (df_actions_next['type_id'] == spadlconfig.actiontypes.index('tackle')) &
        (df_actions_next['result_id'] == spadlconfig.results.index('fail'))
    )
    failed_defensive = (
        failed_tackle & 
        (df_actions.team_id != df_actions_next.team_id)
    )

    # next_actions: impute the dribble end location from the next event that is not a failed tackle
    # ex) for dribble(team A) -> tackle(team B, fail) -> pass(team A), the dribble end location is the pass start location
    next_actions = df_actions_next.mask(failed_defensive)[["start_x", "start_y"]].bfill()

    cond_dribble = df_actions.type_id == spadlconfig.actiontypes.index("dribble")
    df_actions.loc[cond_dribble, "end_x"] = next_actions.loc[cond_dribble, "start_x"].values
    df_actions.loc[cond_dribble, "end_y"] = next_actions.loc[cond_dribble, "start_y"].values

    return df_actions

def _fix_tackle_result(df_actions: pd.DataFrame) -> pd.DataFrame:
    """
        In SPADL, a tackle's result is defined by possession.
        In Bepro, however, a tackle is also considered successful when it creates a loose-ball situation.
        Therefore, convert tackle results in the Bepro data to the SPADL definition.
    """
    cond_tackle = df_actions.type_id == spadlconfig.actiontypes.index("tackle")

    # Exception (the possession-based definition does not apply): a tackle performed right after a goal is considered a failed tackle
    # ex) for shot(team A) -> tackle(team B) -> kick_off(team B), the action is failed regardless of retained possession
    df_actions_prev = shift_with_edge_fix(df_actions, shift_value=1)
    tackle_after_goal = (
        (df_actions_prev.team_id != df_actions.team_id) & # defending team's tackle after the attacking team scores (successful shot)
        (df_actions_prev.type_id == spadlconfig.actiontypes.index("shot")) & 
        (df_actions_prev.result_id == spadlconfig.results.index("success"))
    )

    df_actions_next = shift_with_edge_fix(df_actions, shift_value=-1)
    same_team = df_actions.team_id == df_actions_next.team_id

    df_actions.loc[cond_tackle & same_team, "result_id"] = spadlconfig.results.index("success")
    df_actions.loc[cond_tackle & (~same_team | tackle_after_goal), "result_id"] = spadlconfig.results.index("fail")

    return df_actions

# Checks whether an offensive and a defensive event exist together
def insert_defensive_actions(df_events: pd.DataFrame, defensive_action : str) -> pd.DataFrame:
    """Insert defensive actions before offensive actions when both occur at the same time."""

    def is_attack_and_defense(event_types : list) -> bool:
        has_attack = any(e["event_type"] in ["Pass", "Shot", "Take-On", "Step-in", "Clearance"] for e in event_types) # offensive events
        has_defense = any(e["event_type"] == defensive_action for e in event_types)

        return has_attack and has_defense

    cond_attack_and_defense = df_events["event_types"].apply(is_attack_and_defense)
    df_events_defense = df_events[cond_attack_and_defense].copy()

    if not df_events_defense.empty:
        df_events_defense["event_time"] -= 1e-3
        df_events_defense["event_types"] = df_events_defense["event_types"].apply(
            lambda event_list: [event for event in event_list if event.get("event_type") == defensive_action]
        )
        df_events.loc[cond_attack_and_defense, "event_types"] = df_events.loc[cond_attack_and_defense, "event_types"].apply(
            lambda event_list: [event for event in event_list if event.get("event_type") != defensive_action]
        )

        df_events = pd.concat([df_events_defense, df_events], ignore_index=True)
        df_events = df_events.sort_values(["period_id", "event_time"], kind="mergesort")
        df_events = df_events.reset_index(drop=True)

    return df_events

def _convert_pass_locations(df_events: pd.DataFrame) -> pd.DataFrame:
    """Convert StatsBomb locations to spadl coordinates.
    
    K League pitch conventions:
    1. Pitch size: 68m x 105m
    2. Coordinate (0,0) is the bottom-left corner, (1,1) the top-right corner
    3. Regardless of the half, every event always starts from the goal line (y=0).
    4. The x coordinate is scaled by 68 (GOAL_LINE_LENGTH), the y coordinate by 105 (TOUCH_LINE_LENGTH).
    """
    def _get_end_location(relative_event: dict) -> tuple[Optional[float], Optional[float]]:
        if isinstance(relative_event, dict):
            return pd.Series([relative_event.get("x"), relative_event.get("y")])
        else:
            return pd.Series([np.nan, np.nan])

    df_events[["end_x", "end_y"]] = df_events["relative_event"].apply(_get_end_location)

    df_events[["x", "end_x"]] = np.clip(df_events[["x", "end_x"]] * GOAL_LINE_LENGTH, 0, GOAL_LINE_LENGTH)
    df_events[["y", "end_y"]] = np.clip(df_events[["y", "end_y"]] * TOUCH_LINE_LENGTH, 0, TOUCH_LINE_LENGTH)

    return df_events

def _convert_shot_locations(df_events: pd.DataFrame) -> pd.DataFrame:
    cond_shot = df_events["event_types"].apply(lambda x: any(e["event_type"] == "Shot" for e in x))
    cond_own_goal = df_events["event_types"].apply(lambda x: any(e["event_type"] == "Own Goal" for e in x))
    cond_blocked = df_events["event_types"].apply(lambda x: any(e.get("outcome") == "Blocked" for e in x))
    cond_low_quality = df_events["event_types"].apply(lambda x: any(e.get("outcome") == "Low Quality Shot" for e in x))
    cond_keeper_rush_out = df_events["event_types"].apply(lambda x: any(e.get("outcome") == "Keeper Rush-Out" for e in x))

    cond_missing_end_loc = df_events["ball_position"].apply(lambda b: not isinstance(b, dict))
    
    # If the shot end location is recorded, use those coordinates
    cond_existing_end_loc = (
        cond_shot & 
        ~cond_missing_end_loc
    )
    df_events.loc[cond_existing_end_loc, "end_x"] = df_events[
        cond_existing_end_loc
    ]["ball_position"].apply(lambda b: ORIGINAL_LEFT_POST + b.get("x") * (ORIGINAL_RIGHT_POST - ORIGINAL_LEFT_POST))
    df_events.loc[cond_existing_end_loc, "end_x"] = np.clip(df_events.loc[cond_existing_end_loc, "end_x"] * GOAL_LINE_LENGTH, 0, GOAL_LINE_LENGTH)
    df_events.loc[cond_existing_end_loc, "end_y"] = TOUCH_LINE_LENGTH # shot height information is not used

    # For own goals, set the end location to the center of the team's own goal (y=0)
    df_events.loc[
        cond_own_goal & cond_missing_end_loc, 
        ["end_x", "end_y"]
    ] = CENTER_POST, 0

    # Missing-value handling: Bepro data often lacks the end location of shots.
    # 1. Blocked: impute the shot end location from the blocking action's location
    # 2. Low Quality: the shot missed the goal frame by a lot or never reached it; impute a location depending on the situation
    # 3. Keeper Rush-Out: the keeper rushed out to stop the shot; impute the shot end location from the next goalkeeper action's location
    # 4. Others: the few remaining missing cases (Off-Target (99%), On-Target, Goal, ...) are imputed heuristically. ex) Off-Target-> Goal Kick, Corner, Deflection, Substitution, Recovery, Parry

    # 1. Blocked shots
    # Note: the location reference differs depending on whether the blocking team is the same team or the opponent, because both teams share the same attacking direction.
    df_events_next = shift_with_edge_fix(df_events, shift_value=-1) 
    blocked_idx_by_teammate = (
        cond_shot & 
        cond_blocked & 
        cond_missing_end_loc & 
        cond_blocked & 
        (df_events["team_id"] == df_events_next["team_id"]) # a teammate's blocking (hit) action shares the attacking direction, so no mirroring
    ) 
    blocked_idx_by_opponent = (
        cond_shot & 
        cond_blocked & 
        cond_missing_end_loc & 
        (df_events["team_id"] != df_events_next["team_id"]) # the defending team's blocking action is recorded relative to its own end, so it must be mirrored
    ) 

    # Impute the shot end location from the teammate's blocking action location
    df_events.loc[blocked_idx_by_teammate, ["end_x", "end_y"]] = df_events_next.loc[blocked_idx_by_teammate, ["x", "y"]].values

    # Impute the shot end location from the defending team's blocking action location
    df_events.loc[blocked_idx_by_opponent, "end_x"] = GOAL_LINE_LENGTH - df_events_next.loc[blocked_idx_by_opponent, "x"].values
    df_events.loc[blocked_idx_by_opponent, "end_y"] = TOUCH_LINE_LENGTH - df_events_next.loc[blocked_idx_by_opponent, "y"].values

    # 2. Low Quality shots (complicated)
    cond_next_set_piece = df_events_next["event_types"].apply(lambda x: any(e.get("sub_event_type") in ["Goal Kick", "Corner"] for e in x)) # goal kicks are recorded relative to the defending end, so mirroring is required
    low_quality_idx_by_set_piece = (
        cond_shot & 
        cond_low_quality & 
        cond_missing_end_loc & 
        cond_next_set_piece # if the next action is a set piece, the shot went out past the goal frame, so impute heuristically
    )
    low_quality_idx_others_by_teammate = (
        cond_shot & 
        cond_low_quality & 
        cond_missing_end_loc &
        ~low_quality_idx_by_set_piece &
        (df_events["team_id"] == df_events_next["team_id"]) # while play continues, a teammate's next action shares the attacking direction, so no mirroring
    )
    low_quality_idx_others_by_opponent = (
        cond_shot & 
        cond_low_quality & 
        cond_missing_end_loc & 
        ~low_quality_idx_by_set_piece & 
        (df_events["team_id"] != df_events_next["team_id"]) # while play continues, the defending team's next action is recorded relative to its own end, so it must be mirrored
    )

    cond_out_left = (
        df_events["x"] < (LEFT_POST - Eighteen_YARD) # shots from the left flank end outside the left post
    )
    cond_out_center = (

        (df_events["x"] >= (LEFT_POST - Eighteen_YARD)) # shots from the center end toward the center of the goal
        & (df_events["x"] <= (RIGHT_POST + Eighteen_YARD))
    )
    cond_out_right = (
        df_events["x"] > (RIGHT_POST + Eighteen_YARD) # shots from the right flank end outside the right post
    )
    df_events.loc[low_quality_idx_by_set_piece & cond_out_left, ["end_x", "end_y"]] = LEFT_POST - Eighteen_YARD, TOUCH_LINE_LENGTH
    df_events.loc[low_quality_idx_by_set_piece & cond_out_right, ["end_x", "end_y"]] = RIGHT_POST + Eighteen_YARD, TOUCH_LINE_LENGTH
    df_events.loc[low_quality_idx_by_set_piece & cond_out_center, ["end_x", "end_y"]] = CENTER_POST, TOUCH_LINE_LENGTH


    df_events.loc[low_quality_idx_others_by_teammate, ["end_x", "end_y"]] = df_events_next.loc[low_quality_idx_others_by_teammate, ["x", "y"]].values

    df_events.loc[low_quality_idx_others_by_opponent, "end_x"] = GOAL_LINE_LENGTH - df_events_next.loc[low_quality_idx_others_by_opponent, "x"].values
    df_events.loc[low_quality_idx_others_by_opponent, "end_y"] = TOUCH_LINE_LENGTH - df_events_next.loc[low_quality_idx_others_by_opponent, "y"].values

    # 3. Keeper Rush-Out shots
    # The opposing keeper's action location is used as the shot end location, so it must be mirrored
    df_events.loc[cond_shot & cond_keeper_rush_out & cond_missing_end_loc, "end_x"] = GOAL_LINE_LENGTH - df_events_next.loc[cond_shot & cond_keeper_rush_out & cond_missing_end_loc, "x"].values
    df_events.loc[cond_shot & cond_keeper_rush_out & cond_missing_end_loc, "end_y"] = TOUCH_LINE_LENGTH - df_events_next.loc[cond_shot & cond_keeper_rush_out & cond_missing_end_loc, "y"].values
    
    # 4. Remaining missing values
    cond_other_missing_end_loc = (
        cond_shot &
        (df_events["end_x"].isna() | df_events["end_y"].isna())
    )
    
    df_events.loc[cond_other_missing_end_loc & cond_out_left, ["end_x", "end_y"]] = LEFT_POST - Eighteen_YARD, TOUCH_LINE_LENGTH
    df_events.loc[cond_other_missing_end_loc & cond_out_right, ["end_x", "end_y"]] = RIGHT_POST + Eighteen_YARD, TOUCH_LINE_LENGTH
    df_events.loc[cond_other_missing_end_loc & cond_out_center, ["end_x", "end_y"]] = CENTER_POST, TOUCH_LINE_LENGTH

    return df_events

def _parse_event_as_non_action(event):
    bodypart = "other"
    result = "fail"
    return bodypart, result

def _parse_pass_event(event):
    cond_aerial = any(
        (e.get("event_type") == "Duel") and 
        (e.get("sub_event_type") == "Aerial") 
        for e in event["event_types"]
    )
    cond_throw_in = any(
        (e.get("event_type") == "Set Piece") and 
        (e.get("sub_event_type") == "Throw-In")
        for e in event["event_types"]
    )
    if cond_aerial:
        bodypart = "head" 
    elif cond_throw_in:
        bodypart = "other"
    else:
        bodypart = "foot"

    pass_outcome =  next(
        (e.get('outcome') for e in event['event_types'] if e.get('event_type') == 'Pass'), 
        None
    )
    if pass_outcome == "Successful":
        result = "success"
    elif pass_outcome == "Unsuccessful":
        result = "fail"  # Offside situations are handled in _fix_offside
    elif pass_outcome == "offside":
        result = "offside"
    else:
        raise ValueError(f"Unexpected outcome value: {pass_outcome}")

    return bodypart, result

def _parse_take_on_event(event):
    bodypart = next(
        (e.get("body_part") for e in event["event_types"] if e.get("event_type") == "Take-On"), 
        None
    )
    if bodypart == "Hands":
        bodypart = "other"
    elif bodypart == "Head":
        bodypart = "head"
    elif bodypart == "Left Foot":
        bodypart = "foot_left"
    elif bodypart == "Right Foot":
        bodypart = "foot_right"
    elif bodypart in ["Lower Body", "Upper Body", "Other"]:
        bodypart = "other"
    else:
        bodypart = "foot"

    take_on_outcome =  next(
        (e.get('outcome') for e in event['event_types'] if e.get('event_type') == 'Take-On'), 
        None
    )
    if take_on_outcome  == "Successful":
        result = "success"
    elif take_on_outcome  == "Unsuccessful":
        result = "fail"
    else:
        raise ValueError(f"Unexpected outcome value: {take_on_outcome}")

    return bodypart, result

def _parse_foul_event(event):
    if any(e.get("sub_event_type") in ["Handball Foul", "Foul Throw"] for e in event["event_types"]):
        bodypart = "other" 
    else:
        bodypart = "foot"

    # a foul can have multiple outcomes at once (e.g. yellow card + sending-off)
    if any(e.get("outcome") == "Red Card" for e in event["event_types"]):
        result = "red_card"
    elif any(e.get("outcome") == "Yellow Card" for e in event["event_types"]):
        result = "yellow_card"
    else:
        result = "fail"
    
    return bodypart, result

def _parse_tackle_event(event):
    # Even in cases of "Aerial Duel + Tackle", the tackle itself is performed with the foot after the aerial duel, so "foot" is used.
    bodypart = "foot"

    tackle_outcome = next(
        (e.get("outcome") for e in event["event_types"] if e.get("event_type") == "Tackle"), 
        None
    )
    defensive_line_support_outcome = next(
        (e.get("outcome") for e in event["event_types"] if e.get("event_type") == "Defensive Line Support"), 
        None
    )

    if tackle_outcome:
        if tackle_outcome  == "Successful":
            result = "success"
        elif tackle_outcome  == "Unsuccessful":
            result = "fail"
        else:
            raise ValueError(f"Unexpected outcome value: {tackle_outcome}")
    elif defensive_line_support_outcome:
        if defensive_line_support_outcome  == "Successful":
            result = "success"
        elif defensive_line_support_outcome  == "Unsuccessful":
            result = "fail"
        else:
            raise ValueError(f"Unexpected outcome value: {defensive_line_support_outcome}")
    elif any(e.get("event_type") == "Intervention" for e in event["event_types"]):
        # Result is set to "success" for interventions with no recorded outcome, to be fixed later in _fix_defense_result.
        result = "success" 
    else:
        raise ValueError(f'Unexpected event_types: {event}')

    return bodypart, result

def _parse_interception_event(event):
    # Even in cases of "Aerial Duel + Interception", the interception itself is performed with the foot after the aerial duel, so "foot" is used.
    bodypart = "foot"

    defensive_line_support_outcome = next(
        (e.get("outcome") for e in event["event_types"] if e.get("event_type") == "Defensive Line Support"), 
        None
    )
    if defensive_line_support_outcome:
        if defensive_line_support_outcome  == "Successful":
            result = "success"
        elif defensive_line_support_outcome  == "Unsuccessful":
            result = "fail"
        else:
            raise ValueError(f"Unexpected outcome value: {defensive_line_support_outcome}")
    else:
        # Result is set to "success" for interceptions and interventions with no recorded outcome, to be fixed later in _fix_defense_result.
        result = "success"  

    return bodypart, result

def _parse_shot_event(event):
    bodypart = next(
        (e.get("body_part") for e in event["event_types"] if e.get("event_type") == "Shot"), 
        None
    )
    if bodypart == "Hands":
        bodypart = "other"
    elif bodypart == "Head":
        bodypart = "head"
    elif bodypart == "Left Foot":
        bodypart = "foot_left"
    elif bodypart == "Right Foot":
        bodypart = "foot_right"
    elif bodypart in ["Lower Body", "Upper Body", "Other"]:
        bodypart = "other"
    else:
        bodypart = "foot"

    shot_outcome = next(
        (e.get("outcome") for e in event["event_types"] if e.get("event_type") == "Shot"), 
        None
    )
    result = "success" if shot_outcome == "Goal" else "fail"

    return bodypart, result

def _parse_goalkeeper_event(event):
    bodypart = next(
        (e.get("body_part") for e in event["event_types"] if e.get("event_type") == "Save"), 
        None
    )
    if bodypart == "Hands":
        bodypart = "other"
    elif bodypart == "Head":
        bodypart = "head"
    elif bodypart == "Left Foot":
        bodypart = "foot_left"
    elif bodypart == "Right Foot":
        bodypart = "foot_right"
    elif bodypart in ["Lower Body", "Upper Body", "Other"]:
        bodypart = "other"
    else:
        bodypart = "other" # Aerial Clearance has no bodypart info

    # Determine the result based on the event type
    if any(e.get("event_type") == "Save" for e in event["event_types"]): # Catch and parry actions are always successful
        result = "success"
    elif any(e.get("event_type") == "Aerial Clearance" for e in event["event_types"]): # Claim actions can be successful or unsuccessful
        aerial_clearance_outcome = next(
            (e.get("outcome") for e in event["event_types"] if e.get("event_type") == "Aerial Clearance"), 
            None
        )
        if aerial_clearance_outcome == "Successful": 
            result = "success"
        elif aerial_clearance_outcome  == "Unsuccessful":
            result = "fail"
        else:
            raise ValueError(f"Unexpected outcome value: {aerial_clearance_outcome}")
    elif any(e.get("event_type") == "Defensive Line Support" for e in event["event_types"]): # Defensive Line Support can be successful or unsuccessful
        defensive_line_support_outcome = next(
            (e.get("outcome") for e in event["event_types"] if e.get("event_type") == "Defensive Line Support"), 
            None
        )
        if defensive_line_support_outcome  == "Successful":
            result = "success"
        elif defensive_line_support_outcome  == "Unsuccessful":
            result = "fail"
        else:
            raise ValueError(f"Unexpected outcome value: {defensive_line_support_outcome}")
    else:
        raise ValueError(f'Unexpected event_types: {event}')

    return bodypart, result

def _parse_clearance_event(event):
    cond_aerial = any(
        (e.get("event_type") == "Duel") and 
        (e.get("sub_event_type") == "Aerial") 
        for e in event["event_types"]
    )
    bodypart = "head" if cond_aerial else "foot"
    result = "success"

    return bodypart, result

def _parse_bad_touch_event(event):
    bodypart = "foot"
    result = "owngoal" if any(e.get("event_type") == "Own Goal" for e in event["event_types"]) else "fail"

    return bodypart, result

def _parse_dribble_event(event):
    bodypart = "foot"
    result = "success"

    return bodypart, result