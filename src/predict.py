"""
Inference Pipeline Module
Loads persisted preprocessing and model artifacts, validates raw inputs,
applies feature engineering, and generates live price predictions with confidence intervals.
"""

import os
from typing import Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
import joblib

from src.feature_engineering import engineer_features
from src.data_preprocessing import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

class HousePricePredictor:
    """Predictor class for inferencing housing prices."""
    
    def __init__(self, model_path: str = None, preprocessor_path: str = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.model_path = model_path or os.path.join(base_dir, "models", "best_model.joblib")
        self.preprocessor_path = preprocessor_path or os.path.join(base_dir, "models", "preprocessor.joblib")
        
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.rmse_error = 35000.0  # Default benchmark standard error
        self.load_artifacts()
        
    def load_artifacts(self):
        """Loads serialized model and preprocessor pipelines."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}. Please train the model first.")
        if not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(f"Preprocessor file not found at: {self.preprocessor_path}. Please train the model first.")
            
        model_payload = joblib.load(self.model_path)
        if isinstance(model_payload, dict) and "model" in model_payload:
            self.model = model_payload["model"]
            self.rmse_error = model_payload.get("rmse", 35000.0)
            self.model_name = model_payload.get("model_name", "XGBoost")
        else:
            self.model = model_payload
            self.model_name = "Best Model"
            
        prep_payload = joblib.load(self.preprocessor_path)
        if isinstance(prep_payload, dict) and "preprocessor" in prep_payload:
            self.preprocessor = prep_payload["preprocessor"]
            self.feature_names = prep_payload.get("feature_names", None)
        else:
            self.preprocessor = prep_payload
            
    def predict(self, input_data: Union[Dict[str, Any], pd.DataFrame]) -> Dict[str, Any]:
        """
        Takes raw property feature dictionary or DataFrame and returns predicted price,
        confidence interval, and breakdown metrics.
        """
        if isinstance(input_data, dict):
            df_input = pd.DataFrame([input_data])
        else:
            df_input = input_data.copy()
            
        # Support distance in kilometers as well as miles
        if "distance_to_center_km" in df_input.columns and "distance_to_center_miles" not in df_input.columns:
            df_input["distance_to_center_miles"] = df_input["distance_to_center_km"]
            
        # Ensure default values for any omitted optional features
        defaults = {
            "location": "Downtown Metro",
            "sqft_living": 2000,
            "bedrooms": 3,
            "bathrooms": 2.0,
            "sqft_lot": 5000,
            "floors": 1.5,
            "waterfront": 0,
            "view_quality": 0,
            "condition": 3,
            "grade": 7,
            "yr_built": 2000,
            "yr_renovated": 0,
            "has_garage": 1,
            "has_pool": 0,
            "distance_to_center_miles": 10.0
        }
        for k, v in defaults.items():
            if k not in df_input.columns:
                df_input[k] = v
                
        # 1. Feature Engineering
        engineered = engineer_features(df_input)
        
        # 2. Preprocess / Scale & Encode
        feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
        X = engineered[feature_cols]
        X_trans = self.preprocessor.transform(X)
        
        # 3. Predict
        preds = self.model.predict(X_trans)
        predicted_price = float(round(preds[0], -2))
        
        # Calculate 90% confidence interval (approx 1.645 * standard error)
        margin = float(round(1.645 * self.rmse_error, -2))
        lower_bound = max(50000.0, predicted_price - margin)
        upper_bound = predicted_price + margin
        
        price_per_sqft = round(predicted_price / max(1, df_input["sqft_living"].iloc[0]), 2)
        
        return {
            "predicted_price": predicted_price,
            "currency": "INR (₹)",
            "formatted_price": f"₹{predicted_price:,.0f}",
            "price_range_low": lower_bound,
            "price_range_high": upper_bound,
            "formatted_range": f"₹{lower_bound:,.0f} - ₹{upper_bound:,.0f}",
            "confidence_level": "90%",
            "price_per_sqft": price_per_sqft,
            "model_used": self.model_name,
            "input_summary": {
                "location": str(df_input["location"].iloc[0]),
                "sqft_living": int(df_input["sqft_living"].iloc[0]),
                "bedrooms": int(df_input["bedrooms"].iloc[0]),
                "bathrooms": float(df_input["bathrooms"].iloc[0]),
                "grade": int(df_input["grade"].iloc[0]),
                "condition": int(df_input["condition"].iloc[0]),
                "property_age": int(engineered["property_age"].iloc[0]),
                "luxury_score": float(engineered["luxury_score"].iloc[0])
            }
        }
