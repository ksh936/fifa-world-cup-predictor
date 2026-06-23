# AI-Powered FIFA World Cup Team Predictor

## Project Overview

This project is an end-to-end machine learning application that predicts international football match outcomes using historical match results, recent team form, goals scored, goals conceded, FIFA rankings, ranking points, and venue information.

The final application allows users to select two national teams and generate predicted probabilities for:

- Home Win
- Draw
- Away Win

The project includes data cleaning, exploratory data analysis, feature engineering, machine learning model training, model evaluation, and an interactive Streamlit web app.

---

## Tools & Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest Classifier
- Logistic Regression
- Joblib
- Matplotlib
- Streamlit
- Jupyter Notebook
- GitHub

---

## Datasets Used

1. International football match results dataset  
2. FIFA Men's World Ranking dataset  

The project uses historical international match data and FIFA ranking data to create model-ready features.

---

## Project Workflow

### 1. Data Cleaning
- Loaded historical international football match data.
- Converted date columns into datetime format.
- Created match result labels:
  - Home Win = 1
  - Draw = 0
  - Away Win = -1
- Saved cleaned dataset for further analysis.

### 2. Exploratory Data Analysis
- Analyzed match result distribution.
- Studied total goals per match.
- Filtered FIFA World Cup matches.
- Identified top teams by World Cup wins.
- Identified top teams by World Cup goals scored.

### 3. Feature Engineering
Created rolling recent-form features using each team's previous five matches:

- Recent win rate
- Recent average goals scored
- Recent average goals conceded
- Attack difference
- Defense difference
- Win rate difference
- Neutral venue indicator

### 4. FIFA Ranking Integration
Merged FIFA ranking data with historical match data using an as-of merge to capture the most recent ranking available before each match date.

Added ranking-based features:

- Home team FIFA rank
- Away team FIFA rank
- Home team ranking points
- Away team ranking points
- Rank difference
- Ranking points difference

### 5. Machine Learning Model
Trained and compared:

- Logistic Regression
- Random Forest Classifier

The final model uses Random Forest because it performs well on structured tabular data and provides feature importance insights.

### 6. Streamlit Web App
Built an interactive app where users can:

- Select home and away teams
- Choose neutral venue
- Generate match prediction probabilities
- View team comparison metrics
- View FIFA ranking comparison
- View model features used for prediction

---

## Model Features

The final model uses:

- home_recent_win_rate
- home_recent_avg_goals_scored
- home_recent_avg_goals_conceded
- away_recent_win_rate
- away_recent_avg_goals_scored
- away_recent_avg_goals_conceded
- win_rate_difference
- attack_difference
- defense_difference
- neutral
- home_team_rank
- away_team_rank
- home_team_rank_points
- away_team_rank_points
- rank_difference
- rank_points_difference

---

## Project Structure

```text
fifa-world-cup-predictor/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── world_cup_predictor_model.pkl
│   ├── model_features.pkl
│   ├── world_cup_predictor_model_ranked.pkl
│   └── model_features_ranked.pkl
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_match_prediction_function.ipynb
│   ├── 06_add_fifa_ranking_features.ipynb
│   └── 07_model_training_with_rankings.ipynb
│
├── src/
│   └── prediction.py
│
├── README.md
└── requirements.txt