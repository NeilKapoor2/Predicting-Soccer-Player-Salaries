"""Scouting flags: flag a player only when two different methods agree.

Method 1, the performance score (Project 3): is the player's production
percentile far above or below their salary percentile?

Method 2, the salary model (Project 2): is the actual salary outside the
model's typical error range (prediction ± its average miss)?

A flag needs both methods to point the same way. When they point opposite
ways, that's "Mixed signals". Everything else gets no flag, which is the most
common and most honest result. The two methods are built from the same stats,
so agreement is stronger evidence, not proof.
"""

UNDERPAID = "Possibly underpaid"
OVERPAID = "Possibly overpaid"
MIXED = "Mixed signals"
NO_FLAG = "No flag"
FLAGS = [UNDERPAID, OVERPAID, MIXED, NO_FLAG]

GAP_THRESHOLD = 25   # score gap that counts as a signal (about the top/bottom quarter)


def score_signal(gap, threshold=GAP_THRESHOLD):
    """+1 if production is well above pay, -1 if well below, 0 otherwise."""
    if gap >= threshold:
        return 1
    if gap <= -threshold:
        return -1
    return 0


def model_signal(actual, predicted, typical_miss):
    """+1 if paid below the model's range, -1 if above it, 0 if inside it."""
    if actual < predicted - typical_miss:
        return 1
    if actual > predicted + typical_miss:
        return -1
    return 0


def combine(score, model):
    """Turn the two signals into a flag."""
    if score == model == 1:
        return UNDERPAID
    if score == model == -1:
        return OVERPAID
    if score * model == -1:
        return MIXED
    return NO_FLAG


def add_flags(scores, salary_model):
    """Add model_guess, score_signal, model_signal and flag columns."""
    df = scores.copy()
    df["model_guess"] = [salary_model.predict(row) for row in df.to_dict("records")]
    df["score_signal"] = df["gap"].map(score_signal)
    df["model_signal"] = [model_signal(actual, guess, salary_model.test_mae)
                          for actual, guess in zip(df["Annual USD"], df["model_guess"])]
    df["flag"] = [combine(s, m) for s, m in zip(df["score_signal"], df["model_signal"])]
    return df
