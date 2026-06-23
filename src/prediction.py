import pandas as pd
import joblib


def load_prediction_assets():
    model_data = pd.read_csv("data/processed/model_data_with_rankings.csv")
    model_data["date"] = pd.to_datetime(model_data["date"])

    model = joblib.load("models/world_cup_predictor_model_ranked.pkl")
    features = joblib.load("models/model_features_ranked.pkl")

    return model_data, model, features


def build_latest_team_features(model_data):
    home_team_features = model_data[[
        "date",
        "home_team",
        "home_recent_win_rate",
        "home_recent_avg_goals_scored",
        "home_recent_avg_goals_conceded",
        "home_team_rank",
        "home_team_rank_points"
    ]].copy()

    home_team_features = home_team_features.rename(columns={
        "home_team": "team",
        "home_recent_win_rate": "recent_win_rate",
        "home_recent_avg_goals_scored": "recent_avg_goals_scored",
        "home_recent_avg_goals_conceded": "recent_avg_goals_conceded",
        "home_team_rank": "team_rank",
        "home_team_rank_points": "team_rank_points"
    })

    away_team_features = model_data[[
        "date",
        "away_team",
        "away_recent_win_rate",
        "away_recent_avg_goals_scored",
        "away_recent_avg_goals_conceded",
        "away_team_rank",
        "away_team_rank_points"
    ]].copy()

    away_team_features = away_team_features.rename(columns={
        "away_team": "team",
        "away_recent_win_rate": "recent_win_rate",
        "away_recent_avg_goals_scored": "recent_avg_goals_scored",
        "away_recent_avg_goals_conceded": "recent_avg_goals_conceded",
        "away_team_rank": "team_rank",
        "away_team_rank_points": "team_rank_points"
    })

    team_features = pd.concat([home_team_features, away_team_features], ignore_index=True)
    team_features = team_features.sort_values("date")

    latest_team_features = team_features.groupby("team").tail(1).reset_index(drop=True)

    return latest_team_features


def predict_match(home_team, away_team, model, latest_team_features, neutral=True):
    if home_team not in latest_team_features["team"].values:
        return None, f"{home_team} not found in dataset."

    if away_team not in latest_team_features["team"].values:
        return None, f"{away_team} not found in dataset."

    home_stats = latest_team_features[latest_team_features["team"] == home_team].iloc[0]
    away_stats = latest_team_features[latest_team_features["team"] == away_team].iloc[0]

    input_data = pd.DataFrame([{
        "home_recent_win_rate": home_stats["recent_win_rate"],
        "home_recent_avg_goals_scored": home_stats["recent_avg_goals_scored"],
        "home_recent_avg_goals_conceded": home_stats["recent_avg_goals_conceded"],

        "away_recent_win_rate": away_stats["recent_win_rate"],
        "away_recent_avg_goals_scored": away_stats["recent_avg_goals_scored"],
        "away_recent_avg_goals_conceded": away_stats["recent_avg_goals_conceded"],

        "win_rate_difference": home_stats["recent_win_rate"] - away_stats["recent_win_rate"],
        "attack_difference": home_stats["recent_avg_goals_scored"] - away_stats["recent_avg_goals_scored"],
        "defense_difference": home_stats["recent_avg_goals_conceded"] - away_stats["recent_avg_goals_conceded"],

        "neutral": neutral,

        "home_team_rank": home_stats["team_rank"],
        "away_team_rank": away_stats["team_rank"],
        "home_team_rank_points": home_stats["team_rank_points"],
        "away_team_rank_points": away_stats["team_rank_points"],

        "rank_difference": away_stats["team_rank"] - home_stats["team_rank"],
        "rank_points_difference": home_stats["team_rank_points"] - away_stats["team_rank_points"]
    }])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    label_map = {
        -1: "Away Win",
         0: "Draw",
         1: "Home Win"
    }

    probability_table = pd.DataFrame({
        "Result": [label_map[label] for label in model.classes_],
        "Probability": probabilities
    })

    probability_table["Probability"] = probability_table["Probability"].round(3)

    result = {
        "home_team": home_team,
        "away_team": away_team,
        "predicted_result": label_map[prediction],
        "probabilities": probability_table,
        "home_stats": home_stats,
        "away_stats": away_stats
    }

    return result, None