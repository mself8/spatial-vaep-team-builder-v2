"""Utility functions for all event stream to SPADL converters.

A converter should implement 'convert_to_actions' to convert the events to the
SPADL format.

"""

import numpy as np
import pandas as pd  # type: ignore

from . import config as spadlconfig

min_dribble_length: float = 3.0
max_dribble_length: float = 60.0
max_dribble_duration: float = 10.0

def shift_with_edge_fix(actions: pd.DataFrame, shift_value: int) -> pd.DataFrame:
    """
    Shift each group by a specified value and fill NaN only in the first or last row.
    Does not alter NaN values in the middle of the group to avoid affecting naturally missing data.
    """

    shift_action = actions.groupby("period_id").shift(shift_value)

    if shift_value < 0:
        # When shifting upwards, last row gets NaN, so fill it with the original last value
        fill_indices = actions.groupby("period_id").tail(abs(shift_value)).index
    else:
        # When shifting downwards, first row gets NaN, so fill it with the original first value
        fill_indices = actions.groupby("period_id").head(abs(shift_value)).index

    # Fill the NaN rows with the corresponding original values
    shift_action.loc[fill_indices] = actions.loc[fill_indices]
    shift_action["period_id"] = actions["period_id"]

    return shift_action

def _fix_clearances(actions: pd.DataFrame) -> pd.DataFrame:
    next_actions = shift_with_edge_fix(actions, shift_value=-1)

    clearance_idx = actions.type_id == spadlconfig.actiontypes.index("clearance")

    actions.loc[clearance_idx, "end_x"] = next_actions.loc[clearance_idx, "start_x"].values
    actions.loc[clearance_idx, "end_y"] = next_actions.loc[clearance_idx, "start_y"].values

    return actions

def _fix_block(actions: pd.DataFrame) -> pd.DataFrame:
    """ 
    Fixes the 'block' actions in the DataFrame by setting the end coordinates
    to the next action's start coordinates.
    """

    next_actions = shift_with_edge_fix(actions, shift_value=-1)

    block_idx = actions.type_id == spadlconfig.actiontypes.index("block")

    actions.loc[block_idx, "end_x"] = next_actions.loc[block_idx, "start_x"].values
    actions.loc[block_idx, "end_y"] = next_actions.loc[block_idx, "start_y"].values

    return actions

def _fix_recovery(actions: pd.DataFrame, selector_recovery: pd.Series) -> pd.DataFrame:
    """converts recovery to dribble"""

    next_actions = shift_with_edge_fix(actions, shift_value=-1)
    non_action = actions.type_id == spadlconfig.actiontypes.index("non_action")

    failed_tackle = (
        (next_actions['type_id'] == spadlconfig.actiontypes.index('tackle')) &
        (next_actions['result_id'] == spadlconfig.results.index('fail'))
    )
    failed_interception = (
        (next_actions['type_id'] == spadlconfig.actiontypes.index('interception')) &
        (next_actions['result_id'] == spadlconfig.results.index('fail'))
    )
    same_team = actions.team_id == next_actions.team_id
    # Determine failed defensive actions by the opposing team
    failed_defensive = (failed_tackle | failed_interception) & ~same_team

    # set the dribble's end position to the next valid action's start location, skipping 'non_action' or failed defensive actions.
    next_actions = next_actions.mask(
        (next_actions.type_id == spadlconfig.actiontypes.index("non_action")) | failed_defensive
    )[["start_x", "start_y"]].bfill()

    dx = actions.start_x - next_actions.start_x
    dy = actions.start_y - next_actions.start_y
    far_enough = dx**2 + dy**2 >= min_dribble_length**2

    dribble_idx = (
        selector_recovery
        & non_action
        & far_enough
    )
 
    actions.loc[dribble_idx, "type_id"] = spadlconfig.actiontypes.index("dribble")
    actions.loc[dribble_idx, "result_id"] = spadlconfig.results.index("success")
    actions.loc[dribble_idx, "bodypart_id"] = spadlconfig.bodyparts.index("foot")

    actions.loc[dribble_idx, ["end_x", "end_y"]] = next_actions.loc[
        dribble_idx, ["start_x", "start_y"]
    ].values

    return actions

def _add_dribbles_after_receive(actions: pd.DataFrame, selector_receive: pd.Series) -> pd.DataFrame:
    """ Adds dribbles after receiving the ball if the player moves a certain distance."""

    next_actions = shift_with_edge_fix(actions, shift_value=-1)
    same_player = actions.player_id == next_actions.player_id
    
    next_actions = next_actions.mask(
        next_actions.type_id == spadlconfig.actiontypes.index("non_action")
    )[["start_x", "start_y"]].bfill()

    dx = actions.end_x - next_actions.start_x
    dy = actions.end_y - next_actions.start_y
    far_enough = dx**2 + dy**2 >= min_dribble_length**2

    dribble_idx = (
        selector_receive
        & same_player
        & far_enough
    )

    receive = actions[dribble_idx]
    dribbles = receive.copy() 
    next = next_actions[dribble_idx]

    if not dribbles.empty:
        dribbles["original_event_id"] = np.nan
        dribbles["time_seconds"] = receive.time_seconds + 1e-3 # Dribbles occur right after receiving the ball

        dribbles["start_x"] = receive.start_x
        dribbles["start_y"] = receive.start_y
        dribbles["end_x"] = next.start_x
        dribbles["end_y"] = next.start_y

        dribbles["bodypart_id"] = spadlconfig.bodyparts.index("foot")
        dribbles["type_id"] = spadlconfig.actiontypes.index("dribble")
        dribbles["result_id"] = spadlconfig.results.index("success")

        actions = pd.concat([dribbles, actions], ignore_index=True, sort=False)
        actions = actions.sort_values(["period_id", "time_seconds"], kind="mergesort").reset_index(drop=True)

    return actions

def _fix_direction_of_play(actions: pd.DataFrame, home_team_id: int) -> pd.DataFrame:
    away_idx = (actions.team_id != home_team_id).values
    for col in ["start_x", "end_x"]:
        actions.loc[away_idx, col] = spadlconfig.field_length - actions[away_idx][col].values
    for col in ["start_y", "end_y"]:
        actions.loc[away_idx, col] = spadlconfig.field_width - actions[away_idx][col].values

    return actions

def _add_dribbles(actions: pd.DataFrame) -> pd.DataFrame:
    next_actions = shift_with_edge_fix(actions, shift_value=-1)

    same_team = actions.team_id == next_actions.team_id
    # not_clearance = actions.type_id != actiontypes.index("clearance")
    not_offensive_foul = same_team & (
        next_actions.type_id != spadlconfig.actiontypes.index("foul")
    )
    not_headed_shot = (next_actions.type_id != spadlconfig.actiontypes.index("shot")) & (
        next_actions.bodypart_id != spadlconfig.bodyparts.index("head")
    )

    # case where a bad_touch occurred
    not_bad_touch = (next_actions.type_id != spadlconfig.actiontypes.index("bad_touch")) 
    # no consecutive dribbles by the same player
    not_dribble = (next_actions.type_id != spadlconfig.actiontypes.index("dribble")) & (
        next_actions.type_id != spadlconfig.actiontypes.index("take_on")
    )

    dx = actions.end_x - next_actions.start_x
    dy = actions.end_y - next_actions.start_y
    far_enough = dx**2 + dy**2 >= min_dribble_length**2
    not_too_far = dx**2 + dy**2 <= max_dribble_length**2

    dt = next_actions.time_seconds - actions.time_seconds
    same_phase = dt < max_dribble_duration
    same_period = actions.period_id == next_actions.period_id
    
    dribble_idx = (
        same_team
        & far_enough
        & not_too_far
        & same_phase
        & same_period
        & not_offensive_foul
        & not_headed_shot
        & not_bad_touch
        & not_dribble
    )

    dribbles = pd.DataFrame()
    prev = actions[dribble_idx]
    nex = next_actions[dribble_idx]
    dribbles["game_id"] = nex.game_id
    dribbles["period_id"] = nex.period_id
    dribbles["action_id"] = prev.action_id + 0.1
    dribbles["time_seconds"] = (prev.time_seconds + nex.time_seconds) / 2
    if "timestamp" in actions.columns:
        dribbles["timestamp"] = nex.timestamp
    dribbles["team_id"] = nex.team_id
    dribbles["player_id"] = nex.player_id

    dribbles["start_x"] = prev.end_x
    dribbles["start_y"] = prev.end_y
    dribbles["end_x"] = nex.start_x
    dribbles["end_y"] = nex.start_y

    dribbles["bodypart_id"] = spadlconfig.bodyparts.index("foot")
    dribbles["type_id"] = spadlconfig.actiontypes.index("dribble")
    dribbles["result_id"] = spadlconfig.results.index("success")

    actions = pd.concat([actions, dribbles], ignore_index=True, sort=False)
    actions = actions.sort_values(["game_id", "period_id", "action_id"]).reset_index(drop=True)
    actions["action_id"] = range(len(actions))
    return actions

