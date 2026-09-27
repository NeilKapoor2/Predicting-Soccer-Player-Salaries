"""Scout a player the models have never seen: a verdict with a confidence level.

Both models work on a made-up player:
- Salary model (Project 2): predicts pay from the stats, with its typical miss.
- Performance score (Project 3): ranks the player's per-90 stats against real
  players in the same league and position, and ranks the salary the same way.
  The gap between the two is the score's signal.

The verdict combines the two, using the flag rules in flags.py. Confidence
depends on how strongly they agree. The thresholds were set from the 1,472
real players: only about 2.6% of them reach "High".
"""

import numpy as np

from flags import GAP_THRESHOLD, model_signal, score_signal
from performance_score import stat_columns

STRONG_GAP = 40      # about the most extreme 10% of real gaps, each way
STRONG_MISSES = 2    # salary at least 2 typical misses from the model's guess

UNDERPAID, FAIR, OVERPAID, UNCLEAR = "Underpaid", "Fairly paid", "Overpaid", "Unclear"
HIGH, MEDIUM, LOW = "High", "Medium", "Low"


def percentile_among(peer_values, value):
    """Percentile (0-100) of value if it were added to peer_values.

    Uses the same "average rank" rule as pandas rank(pct=True), which the
    performance score uses for real players, so the numbers are comparable.
    """
    peer_values = np.asarray(peer_values, dtype=float)
    below = (peer_values < value).sum()
    ties = (peer_values == value).sum()
    rank = below + (ties + 2) / 2          # average rank within the tie group, 1-based
    return 100 * rank / (len(peer_values) + 1)


def score_new_player(stats, peers):
    """Per-stat percentiles and performance score for a made-up player.

    stats has the season totals (Gls, Ast, xG, ...) and Min. peers are the
    real players in the same league and position group (with *_per90 columns).
    """
    nineties = round(stats["Min"] / 90, 1)
    percentiles = {
        stat: percentile_among(peers[stat + "_per90"], stats[stat] / nineties)
        for stat in stat_columns
    }
    return percentiles, float(np.mean(list(percentiles.values())))


def verdict(gap, salary, guess, typical_miss):
    """Combine the two methods into (verdict, confidence)."""
    s = score_signal(gap)
    m = model_signal(salary, guess, typical_miss)

    if s == m == 0:
        return FAIR, LOW              # nothing stands out, but the model's range is wide
    if s * m == -1:
        return UNCLEAR, LOW           # the two methods point opposite ways
    direction = s or m
    label = UNDERPAID if direction == 1 else OVERPAID
    if s != m:
        return label, LOW             # only one method sees it
    strong = abs(gap) >= STRONG_GAP and abs(guess - salary) >= STRONG_MISSES * typical_miss
    return label, HIGH if strong else MEDIUM


CONFIDENCE_RULES = f"""
| Confidence | When |
|---|---|
| **High** | Both methods agree **strongly**: a score gap of ±{STRONG_GAP} or more, *and* a salary at least {STRONG_MISSES} typical misses from the model's guess. About 2.6% of real players. |
| **Medium** | Both methods agree: a score gap of ±{GAP_THRESHOLD} or more, *and* a salary outside the model's range (guess ± 1 typical miss). |
| **Low** | Only one method sees a gap, the two disagree ("Unclear"), or neither sees one ("Fairly paid"). |
"""
