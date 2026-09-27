"""Streamlit app: explore the performance score (Project 3).

Run locally:   streamlit run app.py

The app uses the same functions as performance_score.py, so it always shows
exactly what the analysis computes. It reads dfAll.csv (identical to the
Kaggle file), so it loads fast and needs no Kaggle login.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from performance_score import add_salary_gap, score_players, select_players

DATA_FILE = Path(__file__).parent / "dfAll.csv"
REPO_URL = "https://github.com/NeilKapoor2/Predicting-Soccer-Player-Salaries"


@st.cache_data
def load_scores():
    """Every eligible player with performance score, salary percentile and gap."""
    df = pd.read_csv(DATA_FILE)
    return add_salary_gap(score_players(select_players(df)))


st.set_page_config(page_title="Does salary measure skill?", page_icon="⚽", layout="wide")

scores = load_scores()

# ----- Header -----
st.title("Does salary measure skill?")
st.caption(f"A performance score built without ever looking at pay. [How it works]({REPO_URL})")


# ----- Filters -----
with st.sidebar:
    st.header("Filters")
    leagues = st.multiselect("League", sorted(scores["League"].unique()))
    positions = st.multiselect("Position group", sorted(scores["position_group"].unique()))

shown = scores
if leagues:
    shown = shown[shown["League"].isin(leagues)]
if positions:
    shown = shown[shown["position_group"].isin(positions)]


# ----- Player lookup -----
# TODO : the core view. Some questions to decide:
#   - Pick a player from a search box (st.selectbox) and show what?
#     Score, salary percentile and gap as st.metric cards? Their 7 stats?
#   - How do you explain what a "gap of +40" means to a non-expert?
#   - Should defenders get a warning, since the score can't see defending?


# ----- Chart: score vs salary -----
# TODO (you): a scatter plot of performance_score (x) vs salary_percentile (y)
# would show the 0.29 rank correlation at a glance. st.scatter_chart works,
# or use matplotlib like the README chart. Which players do you label?


# ----- Full table (working example, so the app can be deployed now) -----
st.subheader(f"{len(shown):,} players")
st.dataframe(
    shown.sort_values("gap", ascending=False)[
        ["Player", "Squad_x", "League", "Pos_x", "performance_score", "salary_percentile", "gap", "Annual USD"]
    ],
    hide_index=True,
    width="stretch",
    column_config={
        "Squad_x": "Club",
        "Pos_x": "Position",
        "performance_score": st.column_config.NumberColumn("Performance score", format="%.0f"),
        "salary_percentile": st.column_config.NumberColumn("Salary percentile", format="%.0f"),
        "gap": st.column_config.NumberColumn("Gap", format="%+.0f",
                                             help="Positive: plays better than the pay suggests"),
        "Annual USD": st.column_config.NumberColumn("Salary (USD)", format="dollar"),
    },
)


# ----- Caveats -----

st.caption(
    "The score uses 7 attacking and ball-progression stats per 90 minutes, ranked "
    "against players in the same league and position. It can't see defending, age, "
    "contract length or transfer fees, so treat the gap as a question, not a verdict."
)
