import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import streamlit as st
import pandas as pd

from src.prediction import (
    load_prediction_assets,
    predict_match
)

st.set_page_config(
    page_title="FIFA World Cup Team Predictor",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ FIFA World Cup Team Predictor")
st.write(
    "Predict international football match outcomes using recent form, goals, "
    "FIFA ranking, ranking points, and machine learning."
)

try:
    model, features, latest_team_features = load_prediction_assets()

    st.success("Improved ranking-based model loaded successfully.")

    teams = sorted(latest_team_features["team"].dropna().unique())

    st.sidebar.header("Select Match")

    default_home_index = teams.index("Brazil") if "Brazil" in teams else 0
    default_away_index = teams.index("Argentina") if "Argentina" in teams else 1

    home_team = st.sidebar.selectbox(
        "Select Home Team",
        teams,
        index=default_home_index
    )

    away_team = st.sidebar.selectbox(
        "Select Away Team",
        teams,
        index=default_away_index
    )

    neutral = st.sidebar.checkbox("Neutral Venue", value=True)

    if st.sidebar.button("Predict Match"):
        if home_team == away_team:
            st.error("Please select two different teams.")
        else:
            result, error = predict_match(
                home_team=home_team,
                away_team=away_team,
                model=model,
                latest_team_features=latest_team_features,
                neutral=neutral
            )

            if error:
                st.error(error)
            else:
                st.subheader(f"{home_team} vs {away_team}")
                st.success(f"Predicted Result: {result['predicted_result']}")

                probability_table = result["probabilities"].copy()
                probability_table["Probability (%)"] = (
                    probability_table["Probability"] * 100
                ).round(2)

                col1, col2, col3 = st.columns(3)

                for i, row in probability_table.iterrows():
                    with [col1, col2, col3][i]:
                        st.metric(
                            label=row["Result"],
                            value=f"{row['Probability (%)']}%"
                        )

                st.write("### Prediction Probabilities")
                st.dataframe(probability_table)

                st.write("### Probability Chart")
                st.bar_chart(
                    probability_table.set_index("Result")["Probability (%)"]
                )

                home_stats = result["home_stats"]
                away_stats = result["away_stats"]

                comparison = pd.DataFrame({
                    "Metric": [
                        "Recent Win Rate",
                        "Recent Avg Goals Scored",
                        "Recent Avg Goals Conceded",
                        "FIFA Rank",
                        "FIFA Ranking Points"
                    ],
                    home_team: [
                        round(home_stats["recent_win_rate"], 3),
                        round(home_stats["recent_avg_goals_scored"], 3),
                        round(home_stats["recent_avg_goals_conceded"], 3),
                        int(home_stats["team_rank"]),
                        round(home_stats["team_rank_points"], 2)
                    ],
                    away_team: [
                        round(away_stats["recent_win_rate"], 3),
                        round(away_stats["recent_avg_goals_scored"], 3),
                        round(away_stats["recent_avg_goals_conceded"], 3),
                        int(away_stats["team_rank"]),
                        round(away_stats["team_rank_points"], 2)
                    ]
                })

                st.write("### Team Comparison")
                st.dataframe(comparison)

                st.write("### Model Features Used")
                st.write(features)

    else:
        st.info("Select two teams from the sidebar and click Predict Match.")

except Exception as e:
    st.error("The app could not load.")
    st.write("Error details:")
    st.code(str(e))