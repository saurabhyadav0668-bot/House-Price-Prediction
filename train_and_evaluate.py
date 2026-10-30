"""
End-to-End Training, Cross-Validation, and Evaluation Pipeline
Orchestrates the entire ML lifecycle:
- Ingests data and computes EDA visualizations
- Preprocesses and engineers domain features
- Trains and cross-validates 7 ML algorithms
- Tunes hyperparameters for gradient boosted trees
- Evaluates models on holdout test set (MAE, MSE, RMSE, R2, MAPE)
- Saves diagnostic plots and serializes production-ready model artifacts
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from src.data_preprocessing import prepare_data, load_data, clean_data
from src.feature_engineering import engineer_features
from src.models import get_candidate_models, train_and_cross_validate, tune_xgboost
from src.evaluate import (
    calculate_metrics,
    evaluate_all_models,
    plot_eda_distributions,
    plot_correlation_heatmap,
    plot_model_comparison,
    plot_residuals_and_predictions,
    plot_feature_importance
)

def run_pipeline():
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_root, "data", "house_prices.csv")
    models_dir = os.path.join(project_root, "models")
    static_dir = os.path.join(project_root, "static")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)
    
    print("=" * 70)
    print("  HOME PRICE PREDICTION SYSTEM: TRAINING & EVALUATION PIPELINE  ")
    print("=" * 70)
    
    # 1. Load Data and Generate Exploratory Data Analysis Plots
    print("\n[Step 1/6] Loading dataset and performing Exploratory Data Analysis (EDA)...")
    raw_df = load_data(data_path)
    cleaned_df = clean_data(raw_df)
    print(f"Dataset loaded: {len(cleaned_df)} records after cleaning.")
    
    plot_eda_distributions(cleaned_df, static_dir)
    plot_correlation_heatmap(cleaned_df, static_dir)
    print(f"EDA plots saved to: {static_dir}")
    
    # 2. Data Preparation & Feature Engineering
    print("\n[Step 2/6] Preparing data, engineering domain features, and splitting...")
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(
        data_path, test_size=0.2, random_state=42
    )
    print(f"Training set: {X_train.shape[0]} samples | Testing set: {X_test.shape[0]} samples")
    print(f"Total features utilized: {len(feature_names)}")
    
    # 3. Model Training & 5-Fold Cross-Validation
    print("\n[Step 3/6] Training and 5-Fold Cross-Validating candidate regression models...")
    candidate_models = get_candidate_models(random_state=42)
    trained_results = train_and_cross_validate(candidate_models, X_train, y_train, cv_folds=5)
    
    for name, res in trained_results.items():
        print(f"  -> {name:<22}: CV R2 = {res['cv_r2_mean']:.4f} (+/- {res['cv_r2_std']:.4f}) | CV RMSE = ${res['cv_rmse_mean']:,.0f} | Time: {res['training_time_sec']}s")
        
    # 4. Hyperparameter Tuning for XGBoost
    print("\n[Step 4/6] Fine-tuning XGBoost Regressor via Grid Search...")
    best_xgb, best_params = tune_xgboost(X_train, y_train, random_state=42)
    print(f"Optimal XGBoost Hyperparameters: {best_params}")
    
    # Add Tuned XGBoost to evaluation suite
    best_xgb.fit(X_train, y_train)
    trained_results["XGBoost (Tuned)"] = {
        "model_obj": best_xgb,
        "cv_r2_mean": trained_results["XGBoost Regressor"]["cv_r2_mean"],
        "cv_rmse_mean": trained_results["XGBoost Regressor"]["cv_rmse_mean"],
        "training_time_sec": trained_results["XGBoost Regressor"]["training_time_sec"]
    }
    
    # 5. Holdout Evaluation and Benchmarking
    print("\n[Step 5/6] Evaluating all models on Holdout Test Set (20%)...")
    eval_table = evaluate_all_models(trained_results, X_test, y_test)
    print("\n" + "=" * 90)
    print(eval_table.to_string(index=False))
    print("=" * 90)
    
    # Save comparison table
    eval_csv_path = os.path.join(models_dir, "model_comparison.csv")
    eval_json_path = os.path.join(models_dir, "model_comparison.json")
    eval_table.to_csv(eval_csv_path, index=False)
    eval_table.to_json(eval_json_path, orient="records", indent=2)
    
    # 6. Diagnostic Plots & Model Artifact Serialization
    print("\n[Step 6/6] Generating diagnostics and serializing best production model...")
    plot_model_comparison(eval_table, static_dir)
    
    # Identify top model
    best_model_name = eval_table.iloc[0]["Model"]
    best_model_info = trained_results[best_model_name]
    best_model = best_model_info["model_obj"]
    rmse_col = "Test_RMSE (₹)" if "Test_RMSE (₹)" in eval_table.columns else "Test_RMSE ($)"
    best_rmse = eval_table.iloc[0][rmse_col]
    best_r2 = eval_table.iloc[0]["Test_R2"]
    
    print(f"\nWinning Model Selected: {best_model_name} (Test R2: {best_r2:.4f}, Test RMSE: ₹{best_rmse:,.0f})")
    
    # Test predictions and diagnostic plots for best model
    best_preds = best_model.predict(X_test)
    plot_residuals_and_predictions(y_test, best_preds, best_model_name, static_dir)
    plot_feature_importance(best_model, feature_names, static_dir)
    
    # Persist artifacts
    model_save_path = os.path.join(models_dir, "best_model.joblib")
    preprocessor_save_path = os.path.join(models_dir, "preprocessor.joblib")
    
    joblib.dump({
        "model": best_model,
        "model_name": best_model_name,
        "rmse": best_rmse,
        "r2": best_r2
    }, model_save_path)
    
    joblib.dump({
        "preprocessor": preprocessor,
        "feature_names": feature_names
    }, preprocessor_save_path)
    
    print(f"Saved best model to: {model_save_path}")
    print(f"Saved preprocessor to: {preprocessor_save_path}")
    print("\nPipeline execution completed successfully!")

if __name__ == "__main__":
    run_pipeline()
