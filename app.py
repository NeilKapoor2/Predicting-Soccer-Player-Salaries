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

from flags import FLAGS, GAP_THRESHOLD, MIXED, NO_FLAG, OVERPAID, UNDERPAID, add_flags
from performance_score import add_salary_gap, score_players, select_players, stat_percentiles
from salary_model import train_salary_model

DATA_FILE = Path(__file__).parent / "dfAll.csv"
REPO_URL = "https://github.com/NeilKapoor2/Predicting-Soccer-Player-Salaries"


@st.cache_data
def load_scores():
    """Every eligible player with performance score, salary percentile, gap and flag."""
    df = pd.read_csv(DATA_FILE)
    return add_flags(add_salary_gap(score_players(select_players(df))), train_salary_model(df))


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
FLAG_COLORS = {UNDERPAID: "var(--more)", OVERPAID: "var(--less)", MIXED: "var(--gold)", NO_FLAG: "var(--even)"}
FLAG_WHY = {
    UNDERPAID: "Both methods agree. Worth a closer look: is this a bargain, or is something "
               "missing, like age, a short contract, or a league the market overlooks?",
    OVERPAID: "Both methods agree. What might they be missing? Defending, leadership, "
              "reputation, or past seasons?",
    MIXED: "The two methods disagree about this player. Worth asking why.",
    NO_FLAG: "The two methods don't agree on anything strong. That's true for most players, "
             "and it's a result too.",
}


def money(x):
    return f"${x:,.0f}"


def money_md(x):
    # In st.write / help text, two $ signs turn the text between them into a
    # math formula, so escape them
    return money(x).replace("$", "\\$")


# Visual style from the Fair Wage design sample. Colors and fonts are set in
# .streamlit/config.toml; this CSS adds what the theme can't: the background
# glow, uppercase headings, scoreboard-style metric cards, the stamp and gauge.
STYLE = """
<style>
  :root { --ink:#0C1A17; --panel:#122420; --panel2:#1B322B; --line:#254339;
          --chalk:#EAF1EA; --muted:#7F9A8D; --gold:#E8B84B;
          --more:#46D08A; --even:#B7C7BD; --less:#F2706F; }
  .stApp { background: radial-gradient(120% 80% at 50% -10%, #16302a 0%, var(--ink) 55%) fixed; }
  h1, h2, h3 { text-transform: uppercase; letter-spacing: .02em; }

  /* Header, like the sample's masthead */
  .masthead { display:flex; align-items:flex-end; gap:16px; flex-wrap:wrap;
              border-bottom:1px solid var(--line); padding-bottom:18px; margin-bottom:8px; }
  .eyebrow, .block-label { font-family:"Space Mono",monospace; font-size:11px;
              letter-spacing:.24em; text-transform:uppercase; }
  .eyebrow { color:var(--gold); }
  .block-label { color:var(--muted); margin:0 0 10px; }
  .masthead h1 { font-family:"Barlow Condensed",sans-serif; font-weight:700; line-height:.95;
                 font-size:clamp(34px,6vw,58px); margin:4px 0 0; padding:0; }
  .masthead h1 .thin { font-weight:500; color:var(--muted); }
  .masthead .sub { margin-left:auto; color:var(--muted); font-size:13px; max-width:260px; text-align:right; }
  .masthead .sub a { color:var(--gold); }

  /* Left pane section labels */
  [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
      font-family:"Space Mono",monospace; font-size:11px; letter-spacing:.22em;
      color:var(--muted); font-weight:700; }

  /* Metric cards = the sample's scoreboard */
  [data-testid="stMetric"] { background:var(--ink); border:1px solid var(--line);
      border-radius:12px; padding:16px 18px; }
  [data-testid="stMetricLabel"] p { font-family:"Space Mono",monospace; font-size:10px;
      letter-spacing:.2em; text-transform:uppercase; color:var(--muted); }
  [data-testid="stMetricValue"] { font-family:"Barlow Condensed",sans-serif; font-weight:700;
      font-size:clamp(34px,4.5vw,50px); line-height:1.05; font-variant-numeric:tabular-nums; }
  .st-key-model-guess [data-testid="stMetricValue"] { color:var(--gold); }

  /* Player panel */
  .st-key-player-panel { background:linear-gradient(180deg,var(--panel),#0f211c);
      border:1px solid var(--line); border-radius:16px; padding:clamp(16px,2.4vw,26px);
      box-shadow:0 18px 40px -20px rgba(0,0,0,.7); }

  /* Stamp: the gap framed as a question */
  .stamp { border:2px solid; border-radius:12px; padding:16px 18px; display:flex;
           align-items:center; justify-content:space-between; gap:18px; margin:6px 0 14px; }
  .stamp .verdict { font-family:"Barlow Condensed",sans-serif; font-weight:700; letter-spacing:.04em;
           text-transform:uppercase; font-size:clamp(22px,3.4vw,32px); line-height:1.05; }
  .stamp .why { font-size:13px; color:var(--muted); max-width:340px; text-align:right; }

  /* Value gauge: where the gap falls */
  .gauge { margin-bottom:22px; }
  .gauge .track { position:relative; height:38px; border-radius:9px; overflow:hidden;
      border:1px solid var(--line); }
  .gauge .tick { position:absolute; top:0; bottom:0; width:1px; background:rgba(234,241,234,.35); }
  .gauge .marker { position:absolute; top:-4px; bottom:-4px; width:3px; background:var(--chalk);
      border-radius:2px; box-shadow:0 0 0 3px rgba(12,26,23,.6); }
  .gauge .ends { display:flex; justify-content:space-between; margin-top:7px;
      font-family:"Space Mono",monospace; font-size:10px; letter-spacing:.14em;
      text-transform:uppercase; color:var(--muted); }
  .footnote { color:var(--muted); font-size:12px; font-family:"Space Mono",monospace;
      letter-spacing:.04em; text-align:center; margin-top:18px; }
</style>
"""


def stamp(title, why, color):
    return (f'<div class="stamp" style="border-color:{color}">'
            f'<div class="verdict" style="color:{color}">{title}</div>'
            f'<div class="why">{why}</div></div>')


def gauge(gap):
    """Gap (-100 to +100) on a track with the ±GAP_THRESHOLD bands marked."""
    left = 50 - GAP_THRESHOLD / 2      # the -25 line, as a % of the track
    right = 50 + GAP_THRESHOLD / 2     # the +25 line
    marker = 50 + max(-100, min(100, gap)) / 2
    track = (f"linear-gradient(90deg, rgba(242,112,111,.30) 0%, rgba(242,112,111,.30) {left}%, "
             f"rgba(183,199,189,.16) {left}%, rgba(183,199,189,.16) {right}%, "
             f"rgba(70,208,138,.30) {right}%, rgba(70,208,138,.30) 100%)")
    return (f'<div class="gauge"><p class="block-label">Where the gap falls</p>'
            f'<div class="track" style="background:{track}">'
            f'<div class="tick" style="left:{left}%"></div><div class="tick" style="left:{right}%"></div>'
            f'<div class="marker" style="left:calc({marker}% - 1.5px)"></div></div>'
            f'<div class="ends"><span>Paid more than production</span><span>In line</span>'
            f'<span>Produces more than pay</span></div></div>')


st.set_page_config(page_title="Does salary measure skill?", page_icon="⚽", layout="wide")
st.markdown(STYLE, unsafe_allow_html=True)

scores = load_scores()

# ----- Header -----
st.markdown(
    f'''<div class="masthead">
      <div><div class="eyebrow">Pay vs. production · 1,472 players</div>
           <h1>Does salary <span class="thin">measure skill?</span></h1></div>
      <div class="sub">A performance score built without ever looking at pay.
           <a href="{REPO_URL}">How it works</a></div>
    </div>''',
    unsafe_allow_html=True,
)


salary_model = load_salary_model()
percentiles = stat_percentiles(scores)   # ranked against ALL peers, not just the filtered ones
repeated_names = set(scores.loc[scores["Player"].duplicated(), "Player"])


def player_label(i):
    row = scores.loc[i]
    label = f"{row['Player']} — {row['Squad_x']} ({row['League']})"
    if row["Player"] in repeated_names:
        label += f", {row['90s']:g} full matches"
    return label


# ----- Left pane: player picker on top, filters below -----
# The picker's list depends on the filters, so reserve its spot at the top
# first and fill it in after the filters have been read.
player_slot = st.sidebar.container()

with st.sidebar:
    st.divider()
    st.subheader("Filters")
    st.caption("Narrow the player list and the table below.")
    leagues = st.multiselect("League", sorted(scores["League"].unique()))
    positions = st.multiselect("Position group", sorted(scores["position_group"].unique()))
    flag_counts = scores["flag"].value_counts()
    flags = st.multiselect("Scouting flag", FLAGS,
                           format_func=lambda f: f"{f} ({flag_counts.get(f, 0)})",
                           help="A flag appears only when the performance score and the salary model agree")

shown = scores
if leagues:
    shown = shown[shown["League"].isin(leagues)]
if positions:
    shown = shown[shown["position_group"].isin(positions)]
if flags:
    shown = shown[shown["flag"].isin(flags)]

with player_slot:
    st.header("Find a player")
    row_id = st.selectbox("Player", shown.sort_values("Player").index, format_func=player_label,
                          index=None, placeholder="Type a name...")


# ----- Player lookup -----
st.header("Look up a player")

if shown.empty:
    st.info("No players match these filters.")
else:
    if row_id is None:
        st.info("Pick a player at the top of the left pane (tap » on a phone) to see how "
                "their pay compares with their production.")

    if row_id is not None:
        with st.container(key="player-panel"):
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

            # --- Scouting flag: only when both methods agree ---
            low = max(0, player["model_guess"] - salary_model.test_mae)
            high = player["model_guess"] + salary_model.test_mae
            flag = player["flag"]
            title = flag if flag in (MIXED, NO_FLAG) else f"Flag: {flag.lower()}"
            st.markdown(stamp(title, FLAG_WHY[flag], FLAG_COLORS[flag]), unsafe_allow_html=True)

            score_reading = {1: "points to *underpaid*", -1: "points to *overpaid*",
                             0: f"within ±{GAP_THRESHOLD}, no signal"}[player["score_signal"]]
            model_reading = {1: "below the model's range, points to *underpaid*",
                             -1: "above the model's range, points to *overpaid*",
                             0: "inside the model's range, no signal"}[player["model_signal"]]
            st.markdown(
                f"- **Performance score:** gap of {player['gap']:+.0f} → {score_reading}\n"
                f"- **Salary model:** actual {money_md(player['Annual USD'])} vs. range "
                f"{money_md(low)}–{money_md(high)} → {model_reading}"
            )
            st.caption("Both methods are built from the same stats, so agreement is stronger "
                       "evidence, not proof.")

            st.markdown(gauge(player["gap"]), unsafe_allow_html=True)

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
            chart = alt.Chart(breakdown).mark_bar(color="#E8B84B", cornerRadiusEnd=3).encode(
                x=alt.X("Percentile", scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("Stat", sort=None, title=None),
                tooltip=["Stat", "Percentile", "Per 90 minutes"],
            ).properties(height=240, background="transparent").configure_axis(
                labelColor="#7F9A8D", titleColor="#7F9A8D", gridColor="#254339", domainColor="#254339",
                labelFont="Inter", titleFont="Space Mono", labelLimit=220,
            ).configure_view(stroke=None)
            st.altair_chart(chart, width="stretch")
            st.caption(f"Each bar: how this player's per-90 number ranks against {peers} (0–100).")

            # --- The salary model (Project 2) ---
            st.subheader("What a salary model trained on stats would guess")
            predicted = player["model_guess"]

            col1, col2 = st.columns(2)
            col1.metric("Actual salary", money(player["Annual USD"]))
            with col2.container(key="model-guess"):
                st.metric("Model's guess", money(predicted) if predicted > 0 else "below $0")
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
        ["Player", "Squad_x", "League", "Pos_x", "performance_score", "salary_percentile", "gap", "flag",
         "Annual USD"]
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
        "flag": st.column_config.TextColumn("Flag", help="Only when the score and the salary model agree"),
        "Annual USD": st.column_config.NumberColumn("Salary (USD)", format="dollar"),
    },
)


# ----- Caveats -----
st.markdown(
    '<p class="footnote">7 attacking and ball-progression stats per 90 minutes, ranked against '
    "players in the same league and position · can't see defending, age, contract length or "
    "transfer fees · treat the gap as a question, not a verdict</p>",
    unsafe_allow_html=True,
)
