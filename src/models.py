"""
Model Architecture and Training Module
Configures, trains, tunes, and compares standard and advanced regression algorithms:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
"""

import time
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
import xgboost as xgb

def get_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    """Returns initialized regression models for comparative evaluation."""
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0, random_state=random_state),
        "Lasso Regression": Lasso(alpha=50.0, max_iter=2000, random_state=random_state),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, min_samples_split=10, random_state=random_state),
        "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=14, min_samples_split=6, n_jobs=-1, random_state=random_state),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=random_state),
        "XGBoost Regressor": xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.06,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=random_state,
            n_jobs=-1
        )
    }
    return models

def train_and_cross_validate(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Fits candidate models and executes 5-fold cross-validation
    measuring negative RMSE and R-squared.
    """
    results = {}
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    for name, model in models.items():
        start_time = time.time()
        
        # Cross-validation for R2 and negative RMSE
        cv_r2 = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2", n_jobs=-1)
        cv_neg_rmse = cross_val_score(model, X_train, y_train, cv=kf, scoring="neg_root_mean_squared_error", n_jobs=-1)
        
        # Fit final model on full training set
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        results[name] = {
            "model_obj": model,
            "cv_r2_mean": float(np.mean(cv_r2)),
            "cv_r2_std": float(np.std(cv_r2)),
            "cv_rmse_mean": float(-np.mean(cv_neg_rmse)),
            "cv_rmse_std": float(np.std(cv_neg_rmse)),
            "training_time_sec": round(training_time, 3)
        }
        
    return results

def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42
) -> Tuple[xgb.XGBRegressor, Dict[str, Any]]:
    """
    Performs grid search to fine-tune XGBoost hyperparameters for optimal test R2 / MAE.
    """
    param_grid = {
        "n_estimators": [150, 250],
        "learning_rate": [0.05, 0.08],
        "max_depth": [5, 7],
        "subsample": [0.8, 0.95],
        "colsample_bytree": [0.8, 0.95]
    }
    
    base_xgb = xgb.XGBRegressor(random_state=random_state, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=base_xgb,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=3,
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_
