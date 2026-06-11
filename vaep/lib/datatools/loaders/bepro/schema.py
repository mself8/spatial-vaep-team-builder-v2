"""SPADL schema for bepro data."""

from typing import Optional

import pandera as pa
from pandera.typing import Object, Series, DateTime

from datatools.loaders.schema import (
    CompetitionSchema,
    EventSchema,
    GameSchema,
    PlayerSchema,
    TeamSchema,
    SequenceSchema,
)

class BeproCompetitionSchema(CompetitionSchema):
    """Definition of a dataframe containing a list of competitions and seasons."""

    country_name: Series[str]
    """The name of the country the competition relates to."""

class BeproGameSchema(GameSchema):
    """Definition of a dataframe containing a list of games."""

    home_score: Series[int]
    """The final score of the home team."""
    away_score: Series[int]
    """The final score of the away team."""
    venue: Series[str] = pa.Field(nullable=True)
    """The name of the stadium where the game was played."""

class BeproPlayerSchema(PlayerSchema):
    """Definition of a dataframe containing the list of players of a game."""

    #nickname: Series[str] = pa.Field(nullable=True)

    """The nickname of the player on the team. -> Not exist."""
    nickname: Series[str] = pa.Field()
    """birth_date of the player."""
    # birth_date: Series[Object] = pa.Field()
    # """The korean name of the player."""
    # main_position: Series[str] = pa.Field(nullable=True)
    # """The name of the main position of the player on the team."""
    starting_position_name: Series[str] = pa.Field(nullable=True)
    """The name of the starting position of the player on the team."""
    rating: Series[float] = pa.Field(nullable=True, ge=0.0, le=10.0)
    """The players' rating."""

class BeproTeamSchema(TeamSchema):
    """Definition of a dataframe containing the list of teams of a game."""

    team_name_ko: Series[str] = pa.Field()
    """The korean name of the team."""

class BeproPlayerStatsSchema(pa.SchemaModel):
    """Definition of a dataframe containing the list of teams of a game."""

    team_id: Series[int]
    """The unique identifier for the player's team."""
    player_id: Series[int]
    """The unique identifier for the player."""
    play_time: Series[int]
    """The number of miliseconds the player played in the game."""
    rating: Series[float] = pa.Field(ge=0.0, le=10.0)
    """The players' rating."""

class BeproEventSchema(EventSchema):
    """Definition of a dataframe containing event stream data of a game."""

    event_time: Series[int] 
    """ Time when the event occurred during the match, in the format of milliseconds after the match started."""

    x: Series[float] 
    """The x coordinate of the event on the pitch."""
    y: Series[float] 
    """The y coordinate of the event on the pitch."""

    event_types: Series[Object] 
    """list format : eventType + subEventType + cross + outcome + KeyPass + assist + body_part"""
    relative_event: Series[Object] = pa.Field(nullable=True)
    """json format : relative_id + relative_event_time + relative_player_id + relative_x + relative_y"""
    ball_position: Series[Object] = pa.Field(nullable=True)
    """json format : ball_position_x + ball_position_y"""

    # type_id, type_name do not exist in the bepro dataset
    type_id: Optional[int]
    """The unique identifier for the type of this event."""
    type_name: Optional[str]
    """The name of the type of this event."""

class BeproSequenceSchema(SequenceSchema):
    """Definition of a dataframe containing sequence data of a game."""
