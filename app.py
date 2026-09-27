"""Streamlit app: explore the performance score (Project 3).

Run locally:   streamlit run app.py

The app uses the same functions as performance_score.py, so it always shows
exactly what the analysis computes. It reads dfAll.csv (identical to the
Kaggle file), so it loads fast and needs no Kaggle login.
"""

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from performance_score import add_salary_gap, score_players, select_players, stat_percentiles
from salary_model import train_salary_model

DATA_FILE = Path(__file__).parent / "dfAll.csv"
REPO_URL = "https://github.com/NeilKapoor2/Predicting-Soccer-Player-Salaries"


@st.cache_data
def load_scores():
    """Every eligible player with performance score, salary percentile and gap."""
    df = pd.read_csv(DATA_FILE)
    return add_salary_gap(score_players(select_players(df)))


@st.cache_resource
def load_salary_model():
    """Project 2's linear regression (see salary_model.py)."""
    return train_salary_model(pd.read_csv(DATA_FILE))


STAT_NAMES = {
    "Gls": "Goals", "Ast": "Assists", "xG": "Expected goals",
    "xAG": "Expected assists", "PrgC": "Progressive carries",
    "PrgP": "Progressive passes", "PrgR": "Passes received",
}
POSITION_NAMES = {"FW": "forwards", "MF": "midfielders", "DF": "defenders"}
# The score only sees attacking and ball-progression stats
DEFENSIVE_POSITIONS = {"DF", "CB", "LB", "RB", "DM"}
GAP_THRESHOLD = 25   # about the top and bottom quarter of gaps


def money(x):
    return f"${x:,.0f}"


def money_md(x):
    # In st.write / help text, two $ signs turn the text between them into a
    # math formula, so escape them
    return money(x).replace("$", "\\$")


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
st.header("Look up a player")

salary_model = load_salary_model()
percentiles = stat_percentiles(scores)   # ranked against ALL peers, not just the filtered ones
repeated_names = set(scores.loc[scores["Player"].duplicated(), "Player"])


def player_label(i):
    row = scores.loc[i]
    label = f"{row['Player']} — {row['Squad_x']} ({row['League']})"
    if row["Player"] in repeated_names:
        label += f", {row['90s']:g} full matches"
    return label


if shown.empty:
    st.info("No players match these filters.")
else:
    row_id = st.selectbox("Player", shown.sort_values("Player").index, format_func=player_label,
                          index=None, placeholder="Type a name...")

    if row_id is not None:
        player = scores.loc[row_id]
        group = player["position_group"]
        peers = f"{POSITION_NAMES.get(group, group + ' players')} in {player['League']}"
        peer_count = ((scores["League"] == player["League"]) & (scores["position_group"] == group)).sum()

        # --- Headline numbers ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Performance score", f"{player['performance_score']:.0f} / 100",
                    help="Average percentile across 7 per-90 stats, against " + peers)
        col2.metric("Salary percentile", f"{player['salary_percentile']:.0f} / 100",
                    help=f"Paid more than about this share of {peers} ({money_md(player['Annual USD'])} a year)")
        col3.metric("Gap", f"{player['gap']:+.0f}",
                    help="Performance score minus salary percentile")

        st.write(
            f"Among {peer_count} {peers}, this player's production ranks around the "
            f"**{player['performance_score']:.0f}th percentile**, and their salary ranks around the "
            f"**{player['salary_percentile']:.0f}th**."
        )

        # --- Frame the gap as a question, not a verdict ---
        if player["gap"] >= GAP_THRESHOLD:
            st.info("**Produces more than the pay suggests.** Is this a bargain, or is something "
                    "missing, like age, a short contract, or a league the market overlooks?")
        elif player["gap"] <= -GAP_THRESHOLD:
            st.info("**Paid more than production suggests.** What might the score be missing? "
                    "Defending, leadership, reputation, or past seasons?")
        else:
            st.info("**Pay and production roughly line up.**")

        if group in DEFENSIVE_POSITIONS or "DM" in str(player["Pos_x"]):
            st.warning("The score can't see defending (tackles, interceptions, positioning), "
                       "so it underrates defensive players.")
        if peer_count < 10:
            sharing = "player has" if peer_count == 1 else "players share"
            st.warning(f"Only {peer_count} {sharing} this position code in this league, "
                       "so these percentiles compare against very few players.")

        # --- What the score is made of ---
        st.subheader("Where the score comes from")
        breakdown = pd.DataFrame({
            "Stat": [STAT_NAMES[s] for s in percentiles.columns],
            "Percentile": percentiles.loc[row_id].round(0).values,
            "Per 90 minutes": [round(player[s + "_per90"], 2) for s in percentiles.columns],
        })
        # Fixed 0-100 axis and no zoom, so scrolling the page doesn't rescale the chart
        chart = alt.Chart(breakdown).mark_bar().encode(
            x=alt.X("Percentile", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y("Stat", sort=None, title=None),
            tooltip=["Stat", "Percentile", "Per 90 minutes"],
        ).properties(height=240)
        st.altair_chart(chart, width="stretch")
        st.caption(f"Each bar: how this player's per-90 number ranks against {peers} (0–100).")

        # --- The salary model (Project 2) ---
        st.subheader("What a salary model trained on stats would guess")
        predicted = salary_model.predict(player.to_dict())
        low, high = max(0, predicted - salary_model.test_mae), predicted + salary_model.test_mae

        col1, col2 = st.columns(2)
        col1.metric("Actual salary", money(player["Annual USD"]))
        if predicted > 0:
            col2.metric("Model's guess", money(predicted))
        else:
            col2.metric("Model's guess", "below $0")
        st.write(
            f"The model is typically off by **±{money_md(salary_model.test_mae)}** on players it has "
            f"never seen, so its honest answer is somewhere between **{money_md(low)}** and "
            f"**{money_md(high)}**. That range is the point: stats alone can't pin down pay "
            f"(R² {salary_model.test_r2:.2f})."
        )
        if predicted <= 0:
            st.caption("A linear model can go below zero for players with few minutes. "
                       "That's a limit of the model, not a real salary.")
        st.caption("This player may have been in the model's training data, "
                   "so the guess can look closer than it would for a brand-new player.")

        # --- What-if sliders ---
        with st.expander("What if this player's numbers were different?"):
            what_if = player.to_dict()
            for stat, label, top in [("Min", "Minutes played", 3420), ("Gls", "Goals", 40),
                                     ("Ast", "Assists", 25), ("xG", "Expected goals (xG)", 30.0),
                                     ("xAG", "Expected assists (xAG)", 20.0)]:
                start = float(player[stat]) if isinstance(top, float) else int(player[stat])
                what_if[stat] = st.slider(label, 0 * top, max(top, start), start, key=f"{stat}-{row_id}")
            what_if.pop("G+A")   # recalculated from goals + assists
            what_if.pop("90s")   # recalculated from minutes

            new_guess = salary_model.predict(what_if)
            st.metric("Model's new guess", money(new_guess) if new_guess > 0 else "below $0",
                      delta=f"{new_guess - predicted:+,.0f} dollars")
            st.caption("Try adding assists: the guess can go *down*. Several stats overlap "
                       "(goals, xG, goals + assists), so the model splits credit between them "
                       "in strange ways. It learned patterns in pay, not what makes a player good.")


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
