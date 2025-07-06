# imdb_rating_prediction
This project predicts IMDb ratings of TV shows using features like genre, release year, and vote count. It uses models such as Linear Regression, Random Forest, Gradient Boosting, and CatBoost with tuning. The goal is to build an accurate model and evaluate performance using MAE, MSE, and R².

# IMDb Rating Prediction using Machine Learning
This mini-project aims to predict IMDb ratings of TV shows based on genre, release year, and number of votes. The dataset is processed and modeled using various regressors including Linear Regression, Random Forest, Gradient Boosting, and CatBoost with hyperparameter tuning.

## Dataset Overview

- **Dataset**: IMDb Top 250 TV Shows
- **Features Used**:
  - `Genre` (multi-labeled)
  - `Year` of release
  - `IMDb Votes`
  - `Rank`, `Duration`, and more
- **Target Variable**: `IMDb Rating`

## Project Workflow

### 1. Data Preprocessing
- Cleaned and renamed columns
- Converted IMDb votes to numeric
- Transformed `duration` to minutes
- Handled missing values

### 2. Exploratory Data Analysis
- Scatter plot: Votes vs Ratings
- Histogram of IMDb ratings
- Correlation heatmap
- Top 10 genres by average rating

### 3. Feature Engineering
- Multi-hot encoded genres
- Binned `year` into decade categories
- One-hot encoded `year_range`
- Scaled numeric features using `StandardScaler`

### 4. Model Building & Evaluation
- Trained 3 base models:
  - Linear Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
- Evaluated with:
  - Mean Squared Error (MSE)
  - Root MSE (RMSE)
  - R² Score
- Plotted residuals of best-performing model

### 5. Hyperparameter Tuning
- Performed grid search with `GradientBoostingRegressor`
- Optimized for `n_estimators`, `max_depth`, and `learning_rate`
- Selected best estimator based on R² score

### 6. Advanced Modeling with CatBoost
- Used `CatBoostRegressor` for improved handling of categorical variables
- Built-in handling of `main_genre` as categorical
- Hyperparameter tuning using CatBoost's `grid_search()`
- Final model saved in `.cbm` format

## Final Model Performance

| Model                | MAE   | MSE   | R² Score |
|---------------------|-------|-------|----------|
| **Tuned CatBoost**  | 0.1607| 0.0399| 0.3184   |

> CatBoost provided the best performance with MAE ≈ 0.16 and explained 32% of the variance in ratings.

## Model Export
best_model.save_model("catboost_imdb_rating_model.cbm")

## To load the model:
from catboost import CatBoostRegressor
model = CatBoostRegressor()
model.load_model("catboost_imdb_rating_model.cbm")

## Libraries Used
pandas, numpy
matplotlib, seaborn
scikit-learn
catboost

## Author : Khizareen Taj
