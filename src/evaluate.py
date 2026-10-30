"""
Model Evaluation and Diagnostic Visualization Module
Computes regression metrics (MAE, MSE, RMSE, R2, MAPE) and generates
publication-grade diagnostic charts for exploratory data analysis and model assessment.
"""

import os
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# Configure visual styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 150

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes comprehensive regression evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    return {
        "MAE": round(float(mae), 2),
        "MSE": round(float(mse), 2),
        "RMSE": round(float(rmse), 2),
        "R2_Score": round(float(r2), 4),
        "MAPE_percent": round(float(mape), 2)
    }

def evaluate_all_models(
    models_dict: Dict[str, Dict[str, Any]],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """Evaluates all candidate models on the hold-out test set."""
    records = []
    
    for name, info in models_dict.items():
        model = info["model_obj"]
        y_pred = model.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred)
        
        records.append({
            "Model": name,
            "CV_R2_Mean": round(info["cv_r2_mean"], 4),
            "CV_RMSE": round(info["cv_rmse_mean"], 2),
            "Test_R2": metrics["R2_Score"],
            "Test_MAE (₹)": metrics["MAE"],
            "Test_RMSE (₹)": metrics["RMSE"],
            "MAPE (%)": metrics["MAPE_percent"],
            "Train_Time_s": info["training_time_sec"]
        })
        
    df_eval = pd.DataFrame(records).sort_values(by="Test_R2", ascending=False)
    return df_eval

def plot_eda_distributions(df: pd.DataFrame, output_dir: str):
    """Generates distribution plots for target price and key structural features."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Price Distribution
    sns.histplot(df["price"] / 100000, kde=True, ax=axes[0, 0], color="#2b5c8f", bins=35)
    axes[0, 0].set_title("Property Price Distribution (in ₹ Lakhs)", fontsize=13, fontweight="bold")
    axes[0, 0].set_xlabel("Price (₹ Lakhs)")
    axes[0, 0].set_ylabel("Frequency")
    
    # 2. Square Footage Living Area Distribution
    sns.histplot(df["sqft_living"], kde=True, ax=axes[0, 1], color="#2a9d8f", bins=35)
    axes[0, 1].set_title("Living Area Distribution (Sq. Ft.)", fontsize=13, fontweight="bold")
    axes[0, 1].set_xlabel("Living Area (sq.ft)")
    axes[0, 1].set_ylabel("Frequency")
    
    # 3. Price by Location
    order = df.groupby("location")["price"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="price", y="location", order=order, ax=axes[1, 0], palette="crest", orient="h")
    axes[1, 0].set_title("House Prices by Location / Neighborhood", fontsize=13, fontweight="bold")
    axes[1, 0].set_xlabel("Price (₹)")
    axes[1, 0].set_ylabel("")
    axes[1, 0].xaxis.set_major_formatter("₹{x:,.0f}")
    
    # 4. Price vs Living Area with Grade Hue
    scatter = sns.scatterplot(
        data=df,
        x="sqft_living",
        y="price",
        hue="grade",
        palette="viridis",
        alpha=0.6,
        ax=axes[1, 1]
    )
    axes[1, 1].set_title("Price vs. Living Area (Colored by Quality Grade)", fontsize=13, fontweight="bold")
    axes[1, 1].set_xlabel("Living Area (sq.ft)")
    axes[1, 1].set_ylabel("Price (₹)")
    axes[1, 1].yaxis.set_major_formatter("₹{x:,.0f}")
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "eda_price_distribution.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str):
    """Generates correlation heatmap of numeric features with house price."""
    os.makedirs(output_dir, exist_ok=True)
    
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    # Sort by correlation with price
    top_corr_features = corr["price"].sort_values(ascending=False).index
    corr_sorted = corr.loc[top_corr_features, top_corr_features]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_sorted,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Correlation Matrix with House Price", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "eda_correlation_matrix.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

def plot_model_comparison(df_eval: pd.DataFrame, output_dir: str):
    """Generates side-by-side bar plots comparing Model R2 Scores and RMSE."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    palette = sns.color_palette("mako", len(df_eval))
    
    # 1. R2 Score comparison
    sns.barplot(data=df_eval, x="Test_R2", y="Model", palette=palette, ax=ax1)
    ax1.set_title("Model Comparison: Test R² Score (Higher is Better)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("R² Score")
    ax1.set_xlim(0, 1.05)
    for i, v in enumerate(df_eval["Test_R2"]):
        ax1.text(v + 0.01, i, f"{v:.4f}", va="center", fontweight="bold", fontsize=10)
        
    # 2. RMSE comparison
    rmse_col = "Test_RMSE (₹)" if "Test_RMSE (₹)" in df_eval.columns else "Test_RMSE ($)"
    sns.barplot(data=df_eval, x=rmse_col, y="Model", palette="flare", ax=ax2)
    ax2.set_title("Model Comparison: Test RMSE in ₹ (Lower is Better)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Root Mean Squared Error (₹)")
    ax2.xaxis.set_major_formatter("₹{x:,.0f}")
    for i, v in enumerate(df_eval[rmse_col]):
        ax2.text(v + 2000, i, f"₹{v:,.0f}", va="center", fontweight="bold", fontsize=10)
        
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "model_comparison_chart.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

def plot_residuals_and_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    output_dir: str
):
    """Generates Actual vs Predicted scatter and Residual distribution diagnostics."""
    os.makedirs(output_dir, exist_ok=True)
    
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Actual vs Predicted
    axes[0].scatter(y_true, y_pred, alpha=0.45, color="#1f77b4", edgecolors="none")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Fit (y = x)")
    axes[0].set_title(f"{model_name}: Actual vs. Predicted Prices", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Actual Price (₹)")
    axes[0].set_ylabel("Predicted Price (₹)")
    axes[0].xaxis.set_major_formatter("₹{x:,.0f}")
    axes[0].yaxis.set_major_formatter("₹{x:,.0f}")
    axes[0].legend()
    
    # 2. Residual Distribution
    sns.histplot(residuals, kde=True, ax=axes[1], color="#e76f51", bins=30)
    axes[1].axvline(0, color="black", linestyle="--", lw=1.5)
    axes[1].set_title(f"{model_name}: Residuals Distribution (Actual - Predicted)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Residual Error (₹)")
    axes[1].set_ylabel("Frequency")
    axes[1].xaxis.set_major_formatter("₹{x:,.0f}")
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "residuals_plot.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    output_dir: str,
    top_n: int = 15
):
    """Extracts and plots Top N feature importances from tree-based or linear models."""
    os.makedirs(output_dir, exist_ok=True)
    
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        metric_label = "Feature Importance (Gini / Gain)"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
        metric_label = "Standardized Coefficient Magnitude (|Beta|)"
    else:
        return
        
    feat_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 7))
    sns.barplot(data=feat_df, x="Importance", y="Feature", palette="viridis")
    plt.title(f"Top {top_n} Most Influential Price Determinants", fontsize=13, fontweight="bold")
    plt.xlabel(metric_label)
    plt.ylabel("")
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
