"""Implements the label tranformers of the VAEP framework."""

import pandas as pd  # type: ignore
from pandera.typing import DataFrame

import datatools.representation.spadl.config as spadl
from datatools.representation.spadl.schema import SPADLSchema

N_SECONDS = 10
N_ACTIONS = 10

# When a set piece occurs within n_seconds, compute the label over n_seconds starting after the set piece.
# Logic added to check for a set piece within n_seconds and count n_seconds from after the set piece.
# ex) pass(280) -> out(282) -> Throw-In(290) -> pass(292) -> pass(293) -> goal(294)
# Original: with n_seconds = 10, the positive labels run from Throw-In(290) to the goal(294) scored within 10 seconds
# Improved: with n_seconds = 10, the positive labels run from Pass(290-n_seconds) to the goal(294)
set_piece_types = ["freekick_crossed", "freekick_short ", "shot_freekick", "corner_crossed", "corner_short", 
                    "shot_penalty", "throw_in", "goalkick"]  # set-piece event types


def scores_by_actions(actions: DataFrame[SPADLSchema], nr_actions: int = N_ACTIONS) -> pd.DataFrame:
    """Determine whether the team possessing the ball scored a goal within the next x actions.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.
    nr_actions : int, default=10  # noqa: DAR103
        Number of actions after the current action to consider.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'scores' and a row for each action set to
        True if a goal was scored by the team possessing the ball within the
        next x actions; otherwise False.
    """
    # merging goals, owngoals and team_ids

    goals = actions["type_name"].str.contains("shot") & (
        actions["result_id"] == spadl.results.index("success")
    )
    # error in the own-goal action definition
    # owngoals = actions["type_name"].str.contains("shot") & (
    #     actions["result_id"] == spadl.results.index("owngoal")
    # )
    owngoals = actions["result_id"] == spadl.results.index("owngoal")
    y = pd.concat([goals, owngoals, actions["team_id"]], axis=1)
    y.columns = ["goal", "owngoal", "team_id"]

    # adding future results
    for i in range(1, nr_actions):
        for c in ["team_id", "goal", "owngoal"]:
            shifted = y[c].shift(-i)
            shifted[-i:] = y[c].iloc[len(y) - 1]
            y["%s+%d" % (c, i)] = shifted

    res = y["goal"]
    for i in range(1, nr_actions):
        gi = y["goal+%d" % i] & (y["team_id+%d" % i] == y["team_id"])
        ogi = y["owngoal+%d" % i] & (y["team_id+%d" % i] != y["team_id"])
        res = res | gi | ogi

    return pd.DataFrame(res, columns=["scores_by_actions"])


def concedes_by_actions(actions: DataFrame[SPADLSchema], nr_actions: int = N_ACTIONS) -> pd.DataFrame:
    """Determine whether the team possessing the ball conceded a goal within the next x actions.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.
    nr_actions : int, default=10  # noqa: DAR103
        Number of actions after the current action to consider.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'concedes' and a row for each action set to
        True if a goal was conceded by the team possessing the ball within the
        next x actions; otherwise False.
    """
    # merging goals,owngoals and team_ids
    goals = actions["type_name"].str.contains("shot") & (
        actions["result_id"] == spadl.results.index("success")
    )
    # owngoals = actions["type_name"].str.contains("shot") & (
    #     actions["result_id"] == spadl.results.index("owngoal")
    # )
    owngoals = actions["result_id"] == spadl.results.index("owngoal")
    y = pd.concat([goals, owngoals, actions["team_id"]], axis=1)
    y.columns = ["goal", "owngoal", "team_id"]

    # adding future results
    for i in range(1, nr_actions):
        for c in ["team_id", "goal", "owngoal"]:
            shifted = y[c].shift(-i)
            shifted[-i:] = y[c].iloc[len(y) - 1]
            y["%s+%d" % (c, i)] = shifted

    res = y["owngoal"]
    for i in range(1, nr_actions):
        gi = y["goal+%d" % i] & (y["team_id+%d" % i] != y["team_id"])
        ogi = y["owngoal+%d" % i] & (y["team_id+%d" % i] == y["team_id"])
        res = res | gi | ogi

    return pd.DataFrame(res, columns=["concedes_by_actions"])

def scores_by_seconds(actions: DataFrame[SPADLSchema], n_seconds: int = N_SECONDS) -> pd.DataFrame:
    """Determine whether the team possessing the ball scored a goal within the next x seconds.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.
    nr_seconds : int, default=10  # noqa: DAR103
        Number of seconds after the current action to consider.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'scores' and a row for each action set to
        True if a goal was scored by the team possessing the ball within the
        next x seconds; otherwise False.
    """
    # merging goals, owngoals and team_ids
    goal_idx = actions[actions["type_name"].str.contains("shot") & (
        actions["result_id"] == spadl.results.index("success")
    )].index
    # error in the own-goal action definition
    # owngoals = actions["type_name"].str.contains("shot") & (
    #     actions["result_id"] == spadl.results.index("owngoal")
    # )
    owngoal_idx = actions[actions["result_id"] == spadl.results.index("owngoal")].index

    res = pd.Series([False] * len(actions))
    for idx in goal_idx:
        time = actions.at[idx, "time_seconds"]
        period_id = actions.at[idx, "period_id"]
        team_id = actions.at[idx, "team_id"]
        
        # Check for a set piece within n_seconds (regardless of possession)
        set_piece_within_n_seconds = (
              (actions["type_name"].isin(set_piece_types)) &
              (actions["time_seconds"] >= (time - n_seconds)) & 
              (actions["time_seconds"] <= time) &
              (actions["period_id"] == period_id) &
              (actions.index <= idx) 
        )  

        # pandas.Series.diff(int, default 1): Periods to shift for calculating difference
        additional_time = actions["time_seconds"].diff().fillna(0).loc[set_piece_within_n_seconds].sum() if any(set_piece_within_n_seconds) else 0
        additional_n_seconds = n_seconds + additional_time

        goal_cond = (
            (actions["time_seconds"] >= (time - additional_n_seconds)) & 
            (actions["time_seconds"] <= time) &
            (actions["period_id"] == period_id) &
            (actions["team_id"] == team_id) &
            (actions.index <= idx) # the event stream order is adjusted, so labels cannot be assigned from time info alone
        )  
        res = res | goal_cond

    for idx in owngoal_idx:
        time = actions.at[idx, "time_seconds"]
        period_id = actions.at[idx, "period_id"]
        team_id = actions.at[idx, "team_id"]

        # Check for a set piece within n_seconds (regardless of possession)
        set_piece_within_n_seconds = (
              (actions["type_name"].isin(set_piece_types)) &
              (actions["time_seconds"] >= (time - n_seconds)) & 
              (actions["time_seconds"] <= time) &
              (actions["period_id"] == period_id) &
              (actions.index <= idx) 
        )  

        # pandas.Series.diff(int, default 1): Periods to shift for calculating difference
        additional_time = actions["time_seconds"].diff().fillna(0).loc[set_piece_within_n_seconds].sum() if any(set_piece_within_n_seconds) else 0
        additional_n_seconds = n_seconds + additional_time

        owngoal_cond = (
            (actions["time_seconds"] >= (time - additional_n_seconds)) & 
            (actions["time_seconds"] <= time) &
            (actions["period_id"] == period_id) &
            (actions["team_id"] != team_id) &
            (actions.index <= idx) # the event stream order is adjusted, so labels cannot be assigned from time info alone
        )
        res = res | owngoal_cond

    return pd.DataFrame(res, columns=["scores_by_seconds"])

def concedes_by_seconds(actions: DataFrame[SPADLSchema], n_seconds: int = N_SECONDS) -> pd.DataFrame:
    """Determine whether the team possessing the ball scored a goal within the next x seconds.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.
    nr_seconds : int, default=10  # noqa: DAR103
        Number of seconds after the current action to consider.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'scores' and a row for each action set to
        True if a goal was scored by the team possessing the ball within the
        next x seconds; otherwise False.
    """
    # merging goals, owngoals and team_ids
    goal_idx = actions[actions["type_name"].str.contains("shot") & (
        actions["result_id"] == spadl.results.index("success")
    )].index
    # error in the own-goal action definition
    # owngoals = actions["type_name"].str.contains("shot") & (
    #     actions["result_id"] == spadl.results.index("owngoal")
    # )
    owngoal_idx = actions[actions["result_id"] == spadl.results.index("owngoal")].index
    
    res = pd.Series([False] * len(actions))
    for idx in goal_idx:
        time = actions.at[idx, "time_seconds"]
        period_id = actions.at[idx, "period_id"]
        team_id = actions.at[idx, "team_id"]
        
        # Check for a set piece within n_seconds (regardless of possession)
        set_piece_within_n_seconds = (
              (actions["type_name"].isin(set_piece_types)) &
              (actions["time_seconds"] >= (time - n_seconds)) & 
              (actions["time_seconds"] <= time) &
              (actions["period_id"] == period_id) &
              (actions.index <= idx) 
        )  

        # pandas.Series.diff(int, default 1): Periods to shift for calculating difference
        additional_time = actions["time_seconds"].diff().fillna(0).loc[set_piece_within_n_seconds].sum() if any(set_piece_within_n_seconds) else 0
        additional_n_seconds = n_seconds + additional_time

        goal_cond = (
            (actions["time_seconds"] >= (time - additional_n_seconds)) & 
            (actions["time_seconds"] <= time) &
            (actions["period_id"] == period_id) &
            (actions["team_id"] != team_id) &
            (actions.index <= idx) # the event stream order is adjusted, so labels cannot be assigned from time info alone
        )  

        res = res | goal_cond

    for idx in owngoal_idx:
        time = actions.at[idx, "time_seconds"]
        period_id = actions.at[idx, "period_id"]
        team_id = actions.at[idx, "team_id"]

        # Check for a set piece within n_seconds (regardless of possession)
        set_piece_within_n_seconds = (
              (actions["type_name"].isin(set_piece_types)) &
              (actions["time_seconds"] >= (time - n_seconds)) & 
              (actions["time_seconds"] <= time) &
              (actions["period_id"] == period_id) &
              (actions.index <= idx) 
        )  

        # pandas.Series.diff(int, default 1): Periods to shift for calculating difference
        additional_time = actions["time_seconds"].diff().fillna(0).loc[set_piece_within_n_seconds].sum() if any(set_piece_within_n_seconds) else 0
        additional_n_seconds = n_seconds + additional_time

        owngoal_cond = (
            (actions["time_seconds"] >= (time - additional_n_seconds)) & 
            (actions["time_seconds"] <= time) &
            (actions["period_id"] == period_id) &
            (actions["team_id"] == team_id) &
            (actions.index <= idx) # the event stream order is adjusted, so labels cannot be assigned from time info alone
        )
        res = res | owngoal_cond

    return pd.DataFrame(res, columns=["concedes_by_seconds"])

def goal_from_shot(actions: DataFrame[SPADLSchema]) -> pd.DataFrame:
    """Determine whether a goal was scored from the current action.

    This label can be use to train an xG model.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'goal' and a row for each action set to
        True if a goal was scored from the current action; otherwise False.
    """
    goals = actions["type_name"].str.contains("shot") & (
        actions["result_id"] == spadl.results.index("success")
    )

    return pd.DataFrame(goals, columns=["goal_from_shot"])
