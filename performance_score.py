"""Performance score: rate players WITHOUT using salary.

Project 2 showed that on-field stats only explain about a quarter of salary.
This script turns that around: it scores every player from production alone,
then compares the score to what the player is paid.

The score never sees "Annual USD". Salary is only used afterwards, to see who
is paid far above or far below what their production would suggest.
"""

import os
import numpy as np
import pandas as pd
import kagglehub

# Load the same dataset as Project 2
path = kagglehub.dataset_download("armaanmartins21/undervalued-football-players")
csv_file = [file for file in os.listdir(path) if file.endswith(".csv")][0]
df = pd.read_csv(os.path.join(path, csv_file))

# Stats that go into the score (all measured per 90 minutes)
stat_columns = ["Gls", "Ast", "xG", "xAG", "PrgC", "PrgP", "PrgR"]

MIN_90S = 10   # ignore players with fewer than 10 full matches of playing time

# Keep players with enough minutes and complete data.
# - Goalkeepers are dropped: none of these stats measure goalkeeping.
# - Champions League rows are dropped: those players already have a row for
#   their domestic league, so keeping both would count them twice.
df = df.dropna(subset=stat_columns + ["90s", "Annual USD", "Pos_x"])
df = df[
    (df["90s"] >= MIN_90S)
    & (df["Pos_x"] != "GK")
    & (df["League"] != "Champions League")
].copy()

# Rate stats: per-90 so a player is not rewarded just for playing more
for col in stat_columns:
    df[col + "_per90"] = df[col] / df["90s"]

# Compare players only with peers: same league AND same position group.
# A defender's goals should not be judged against a striker's, and a wage that
# is huge in MLS is ordinary in the Premier League.
df["position_group"] = df["Pos_x"].str.split(",").str[0].replace({"CB": "DF"})
peers = ["League", "position_group"]

# Percentile rank (0-100) of each stat within the peer group
per90_columns = [col + "_per90" for col in stat_columns]
percentiles = df.groupby(peers)[per90_columns].rank(pct=True) * 100

# Performance score = average percentile across the stats. NO salary here.
df["performance_score"] = percentiles.mean(axis=1)

# Only now bring in salary: its percentile within the same peer group
df["salary_percentile"] = df.groupby(peers)["Annual USD"].rank(pct=True) * 100

# Positive gap = plays better than the pay suggests, negative = paid more than production suggests
df["gap"] = df["performance_score"] - df["salary_percentile"]

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

df.sort_values("gap", ascending=False)[["Player", "Squad_x", "League", "Pos_x", "90s",
    "performance_score", "salary_percentile", "gap", "Annual USD"]].round(2).to_csv(
    "performance_score_results.csv", index=False)
