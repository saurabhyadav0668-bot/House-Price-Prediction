# 🏡 AI Real Estate Valuation & Home Price Prediction System

An end-to-end Machine Learning system designed to predict residential property prices with high accuracy using structural features, location tiers, property age, and luxury amenities.

Built with **Python**, **Scikit-Learn**, **XGBoost**, **Pandas**, **Streamlit**, and **Flask**.

---

## 🌟 Key Highlights
- **Multi-Model Benchmark Suite**: Compares **Linear Regression, Ridge, Lasso, Decision Trees, Random Forest, Gradient Boosting, and XGBoost Regressors** across 5-fold cross-validation.
- **High Predictive Accuracy**: Tuned XGBoost model achieves **$R^2 = 0.9778$**, **Test MAE = ₹51,514**, **Test RMSE = ₹70,751**, and **MAPE = 4.88%**.
- **Domain Feature Engineering**: Synthesizes effective property age, room spaciousness ratios, living-to-lot proportions, condition-grade interaction, and composite luxury indexes.
- **Interactive Web Application (Streamlit)**: Live property valuator, dynamic 90% confidence intervals, neighborhood benchmarks, interactive EDA explorer, and batch CSV valuation tool.
- **RESTful API Service (Flask)**: Production-ready microservice with `/predict`, `/predict_batch`, `/metrics`, and `/health` endpoints.
- **Complete Deliverables**: Modular codebase (`src/`), runnable training pipeline (`train_and_evaluate.py`), comprehensive Jupyter Notebook (`notebooks/house_price_prediction.ipynb`), automated unit tests (`tests/`), and detailed academic report (`docs/PROJECT_REPORT.md`).

---

## 📁 Project Structure

```
minor project/
├── data/
│   ├── generate_dataset.py          # Script to generate realistic real-estate dataset
│   └── house_prices.csv             # 5,000 real estate transaction records (17 features)
├── src/
│   ├── __init__.py                  # Package initialization
│   ├── data_preprocessing.py        # Cleaning, outlier filtering, ColumnTransformer
│   ├── feature_engineering.py       # Domain feature engineering module
│   ├── models.py                    # 7 Regression models & hyperparameter grid search
│   ├── evaluate.py                  # Evaluation metrics & diagnostic plotting
│   └── predict.py                   # Reusable inference pipeline class
├── models/
│   ├── best_model.joblib            # Persisted top-performing XGBoost model
│   ├── preprocessor.joblib          # Persisted ColumnTransformer pipeline
│   ├── model_comparison.csv         # Benchmark metrics across models (CSV)
│   └── model_comparison.json        # Benchmark metrics across models (JSON)
├── static/
│   ├── eda_correlation_matrix.png   # Correlation heatmap
│   ├── eda_price_distribution.png   # Price & square footage distributions
│   ├── model_comparison_chart.png   # Model R² and RMSE comparison charts
│   ├── residuals_plot.png           # Actual vs. Predicted & Residual distribution
│   └── feature_importance.png       # Top valuation determinant ranking
├── notebooks/
│   ├── build_notebook.py            # Automated notebook builder
│   └── house_price_prediction.ipynb # Step-by-step Jupyter Notebook
├── tests/
│   └── test_pipeline.py             # Unit and integration test suite
├── docs/
│   └── PROJECT_REPORT.md            # In-depth technical and empirical project report
├── app.py                           # Interactive Streamlit Web Application
├── api.py                           # Enterprise Flask REST API microservice
├── train_and_evaluate.py            # End-to-end training and evaluation script
├── requirements.txt                 # Project dependencies
└── README.md                        # Documentation and user guide
```

---

## 📊 Model Performance Benchmarks

Evaluated on a 20% holdout test set (980 records) after 5-fold cross-validation:

| Model Architecture | 5-Fold Cross-Val $R^2$ | Test $R^2$ Score | Test MAE (₹) | Test RMSE (₹) | MAPE (%) | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Tuned)** 🏆 | **0.9725** | **0.9778** | **₹51,514** | **₹70,751** | **4.88%** | 9.02s |
| **XGBoost Regressor** | 0.9725 | 0.9751 | ₹53,911 | ₹74,956 | 5.12% | 9.02s |
| **Gradient Boosting** | 0.9711 | 0.9735 | ₹57,224 | ₹77,259 | 5.50% | 29.19s |
| **Linear Regression** | 0.9517 | 0.9531 | ₹75,531 | ₹102,831 | 8.25% | 31.74s |
| **Lasso Regression** | 0.9516 | 0.9530 | ₹75,497 | ₹102,891 | 8.24% | 2.19s |
| **Ridge Regression** | 0.9505 | 0.9519 | ₹76,380 | ₹104,191 | 8.32% | 0.27s |
| **Random Forest** | 0.9440 | 0.9473 | ₹79,884 | ₹108,962 | 7.83% | 27.72s |
| **Decision Tree** | 0.8676 | 0.8639 | ₹127,427 | ₹175,160 | 12.49% | 1.92s |

---

## 🚀 Installation & Quickstart

### 1. Clone / Navigate to Directory
```bash
cd "c:\Users\DELL\Desktop\minor project"
```

### 2. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run End-to-End Training & Evaluation Pipeline
To regenerate the dataset, train all 7 models, run 5-fold cross-validation, optimize hyperparameters, generate diagnostic charts, and save artifacts:
```bash
python train_and_evaluate.py
```

### 4. Launch the Interactive Web Application (Streamlit)
```bash
streamlit run app.py
```
*Opens automatically at `http://localhost:8501`.*

Features:
- **Live Valuator**: Adjust square footage, location, beds, baths, grade, and amenities to receive instant valuations and 90% confidence ranges.
- **Market Analysis & EDA**: Explore price histograms, neighborhood boxplots, and correlation heatmaps.
- **Model Benchmarks**: Visual accuracy comparison and feature importance charts.
- **Batch CSV Valuator**: Upload CSV files for instant multi-property valuations.

### 5. Launch the REST API Service (Flask)
```bash
python api.py
```
*Listens at `http://localhost:5000`.*

#### Example Prediction Request:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "location": "Bellevue Tech Park",
    "sqft_living": 2600,
    "bedrooms": 4,
    "bathrooms": 3.0,
    "sqft_lot": 8000,
    "floors": 2.0,
    "waterfront": 0,
    "view_quality": 2,
    "condition": 4,
    "grade": 9,
    "yr_built": 2016,
    "yr_renovated": 0,
    "has_garage": 1,
    "has_pool": 1,
    "distance_to_center_miles": 5.5
  }'
```

#### Example JSON Response:
```json
{
  "status": "success",
  "valuation": {
    "currency": "INR (₹)",
    "predicted_price": 1435200.0,
    "formatted_price": "₹1,435,200",
    "price_range_low": 1318800.0,
    "price_range_high": 1551600.0,
    "formatted_range": "₹1,318,800 - ₹1,551,600",
    "confidence_level": "90%",
    "price_per_sqft": 552.0,
    "model_used": "XGBoost (Tuned)"
  }
}
```

### 6. Run Automated Test Suite
```bash
python -m unittest tests/test_pipeline.py
```

---

## 🔬 Top Predictive Features
1. **`sqft_living`**: Interior heated square footage is the primary scale driver.
2. **`grade` & `condition_grade_interaction`**: Construction materials and architectural craftsmanship.
3. **`location`**: Baseline land value and proximity to urban tech hubs.
4. **`luxury_score` & `waterfront`**: Distinct price multipliers for scenic amenities and private pools.
5. **`effective_age`**: Structural depreciation counterbalanced by modern renovations.

---

## 📜 License
This project is open-source under the MIT License.
