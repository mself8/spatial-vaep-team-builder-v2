"""Minimal SPADL package — bepro only."""
from . import config
from .bepro import convert_to_actions  # noqa: F401

import pandas as pd
import datatools.representation.spadl.config as spadlcfg

def add_names(actions: pd.DataFrame) -> pd.DataFrame:
    """Add name columns for type_id/bodypart_id/result_id."""
    actions = actions.copy()
    if "type_id" in actions.columns and "type_name" not in actions.columns:
        actions["type_name"] = actions["type_id"].map(
            lambda i: spadlcfg.actiontypes[int(i)] if pd.notna(i) and 0 <= int(i) < len(spadlcfg.actiontypes) else "non_action"
        )
    if "bodypart_id" in actions.columns and "bodypart_name" not in actions.columns:
        actions["bodypart_name"] = actions["bodypart_id"].map(
            lambda i: spadlcfg.bodyparts[int(i)] if pd.notna(i) and 0 <= int(i) < len(spadlcfg.bodyparts) else "foot"
        )
    if "result_id" in actions.columns and "result_name" not in actions.columns:
        actions["result_name"] = actions["result_id"].map(
            lambda i: spadlcfg.results[int(i)] if pd.notna(i) and 0 <= int(i) < len(spadlcfg.results) else "fail"
        )
    return actions
