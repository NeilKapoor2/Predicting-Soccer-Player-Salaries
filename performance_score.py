"""Performance score: rate players WITHOUT using salary.

Project 2 showed that on-field stats only explain about a fifth of salary (R² 0.20).
This script turns that around: it scores every player from production alone,
then compares the score to what the player is paid.

The score never sees "Annual USD". Salary is only used afterwards, to see who
is paid far above or far below what their production would suggest.
"""

from data import load_big_leagues

# Stats that go into the score (all measured per 90 minutes)
stat_columns = ["Gls", "Ast", "xG", "xAG", "PrgC", "PrgP", "PrgR"]

MIN_90S = 10   # ignore players with fewer than 10 full matches of playing time

# Compare players only with peers: same league AND same position group.
# A defender's goals should not be judged against a striker's, and a wage that
# is huge in MLS is ordinary in the Premier League.
peers = ["League", "position_group"]


def select_players(df):
    """Keep players with enough minutes and complete data.

    - Goalkeepers are dropped: none of these stats measure goalkeeping.
    - Champions League rows are dropped: those players already have a row for
      their domestic league, so keeping both would count them twice.
    """
    df = df.dropna(subset=stat_columns + ["90s", "Annual USD", "Pos_x"])
    df = df[
        (df["90s"] >= MIN_90S)
        & (df["Pos_x"] != "GK")
        & (df["League"] != "Champions League")
    ].copy()
    df["position_group"] = df["Pos_x"].str.split(",").str[0].replace({"CB": "DF"})
    return df


def score_players(df):
    """Performance score (0-100) from production alone. NO salary here."""
    df = df.copy()

    # Rate stats: per-90 so a player is not rewarded just for playing more
    for col in stat_columns:
        df[col + "_per90"] = df[col] / df["90s"]

    # Percentile rank (0-100) of each stat within the peer group
    per90_columns = [col + "_per90" for col in stat_columns]
    percentiles = df.groupby(peers)[per90_columns].rank(pct=True) * 100

    # Performance score = average percentile across the stats
    df["performance_score"] = percentiles.mean(axis=1)
    return df


def add_salary_gap(df):
    """Only now bring in salary: its percentile within the same peer group."""
    df = df.copy()
    df["salary_percentile"] = df.groupby(peers)["Annual USD"].rank(pct=True) * 100

    # Positive gap = plays better than the pay suggests, negative = paid more than production suggests
    df["gap"] = df["performance_score"] - df["salary_percentile"]
    return df


def results_table(df):
    """What gets saved to performance_score_results.csv."""
    # Stable sort: players with the same gap keep their order from the dataset,
    # so the CSV comes out identical on every computer
    return df.sort_values("gap", ascending=False, kind="stable")[["Player", "Squad_x", "League", "Pos_x", "90s",
        "performance_score", "salary_percentile", "gap", "Annual USD"]].round(2)


if __name__ == "__main__":
    # Same dataset as Project 2
    df = add_salary_gap(score_players(select_players(load_big_leagues())))

    print(f"Players scored: {len(df)} (at least {MIN_90S} full matches, no goalkeepers, no Champions League rows)")

    correlation = df["performance_score"].corr(df["salary_percentile"], method="spearman")
    print(f"Rank correlation between performance score and salary: {correlation:.2f}")
    print("(1.0 would mean pay perfectly follows production; 0 would mean no relationship)")

    # Rank only forwards and midfielders. These stats say little about a
    # defender's job (defending), so defenders are scored but not ranked.
    ranked = df[df["position_group"].isin(["FW", "MF"])]

    show = ["Player", "Squad_x", "League", "Pos_x", "performance_score", "salary_percentile", "Annual USD"]

    print("\nForwards and midfielders most underpaid relative to production:")
    print(ranked.nlargest(15, "gap")[show].round(1).to_string(index=False))

    print("\nForwards and midfielders most overpaid relative to production:")
    print(ranked.nsmallest(15, "gap")[show].round(1).to_string(index=False))

    print("\nCaution: the score uses only 7 attacking and ball-progression stats, so it")
    print("misses defensive work (a defensive midfielder will look overpaid), and it")
    print("ignores age, contract length and transfer fees. Treat the gap as a")
    print("question, not a verdict.")

    results_table(df).to_csv("performance_score_results.csv", index=False)
