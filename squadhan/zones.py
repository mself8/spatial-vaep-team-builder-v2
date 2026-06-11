"""Map pitch coordinates onto 12 asymmetric zones.

SPADL coordinate system:
  - Pitch size: 105m (x) × 68m (y)
  - Normalized so that every team attacks from left (x=0) to right (x=105)
  - i.e., the larger x is, the closer to the opponent's goal (attacking area)

12-zone layout (ported from team-builder):
  x ∈ [  0,  26.25) : defensive third → 3-way split on y → zones 0, 1, 2
  x ∈ [26.25, 52.5) : defensive midfield → single zone, no split → zone 3
  x ∈ [52.5, 78.75) : attacking midfield → 3-way split on y → zones 4, 5, 6
  x ∈ [78.75, 105]  : attacking third → 5-way split on y → zones 7, 8, 9, 10, 11

Why the attacking third gets 5 zones:
  Actions in the goal-decisive area (in and around the penalty box)
  have high VAEP variance, so they are split more finely.
"""

NUM_ZONES = 12

# y boundaries for the 3-way split: 68m into thirds
_Y_BREAKS_3 = (68 / 3, 68 * 2 / 3)          # ≈ 22.67m, 45.33m

# y boundaries for the 5-way split: 68m into fifths (used in the attacking third)
_Y_BREAKS_5 = (68 / 5, 68 * 2 / 5, 68 * 3 / 5, 68 * 4 / 5)   # ≈ 13.6, 27.2, 40.8, 54.4


def map_to_zone(x: float, y: float) -> int:
    """Convert an (x, y) coordinate to a zone number 0-11.

    Parameters
    ----------
    x : float
        Pitch length coordinate (0=own goal, 105=opponent's goal). Clipped if out of range.
    y : float
        Pitch width coordinate (0=left touchline, 68=right touchline). Clipped if out of range.

    Returns
    -------
    int
        Zone number (0-11)

    Zone layout (→ = attacking direction):
        ┌──────────┬──────────┬──────────┬──────────────────────┐
        │  2       │          │  6       │  11                  │
        ├──────────┤    3     ├──────────┼──────────────────────┤
        │  1       │ (def.mid)│  5       │  10  9  8  (att. 1/3)│
        ├──────────┤          ├──────────┼──────────────────────┤
        │  0       │          │  4       │  7                   │
        └──────────┴──────────┴──────────┴──────────────────────┘
         def. third   def. mid   att. mid   att. third (5 zones)
    """
    # Clip coordinates to the valid range (guards against sensor noise, on-boundary values, etc.)
    x = max(0.0, min(float(x), 105.0))
    y = max(0.0, min(float(y), 68.0))

    if x < 26.25:
        # Defensive third: 3-way split on y
        if y < _Y_BREAKS_3[0]:
            return 0   # defensive third, left (bottom)
        elif y < _Y_BREAKS_3[1]:
            return 1   # defensive third, center
        else:
            return 2   # defensive third, right (top)

    elif x < 52.5:
        # Defensive midfield: a single zone, no subdivision
        # (VAEP variance here is relatively small, so splitting buys little)
        return 3

    elif x < 78.75:
        # Attacking midfield: 3-way split on y
        if y < _Y_BREAKS_3[0]:
            return 4   # attacking midfield, left
        elif y < _Y_BREAKS_3[1]:
            return 5   # attacking midfield, center
        else:
            return 6   # attacking midfield, right

    else:
        # Attacking third: 5-way split on y
        # Finer split in/around the penalty area to isolate the dangerous zones
        if y < _Y_BREAKS_5[0]:
            return 7    # attacking third, far left (near the left byline)
        elif y < _Y_BREAKS_5[1]:
            return 8    # attacking third, left
        elif y < _Y_BREAKS_5[2]:
            return 9    # attacking third, center (in front of the penalty box)
        elif y < _Y_BREAKS_5[3]:
            return 10   # attacking third, right
        else:
            return 11   # attacking third, far right (near the right byline)
