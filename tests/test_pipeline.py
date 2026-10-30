"""
Automated Unit and Integration Test Suite
Tests data preprocessing, feature engineering, model prediction, and API endpoints.
"""

import os
import unittest
import numpy as np
import pandas as pd

from src.feature_engineering import engineer_features
from src.data_preprocessing import clean_data, build_preprocessor, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from src.predict import HousePricePredictor
from api import app

class TestHousingPipeline(unittest.TestCase):
    
    def setUp(self):
        self.sample_raw_property = {
            "location": "Bellevue Tech Park",
            "sqft_living": 2400,
            "bedrooms": 3,
            "bathrooms": 2.5,
            "sqft_lot": 6500,
            "floors": 2.0,
            "waterfront": 0,
            "view_quality": 1,
            "condition": 4,
            "grade": 8,
            "yr_built": 2010,
            "yr_renovated": 0,
            "has_garage": 1,
            "has_pool": 0,
            "distance_to_center_miles": 5.0
        }
        
    def test_feature_engineering(self):
        """Verify domain feature transformations."""
        df = pd.DataFrame([self.sample_raw_property])
        engineered = engineer_features(df, current_year=2025)
        
        # Check property age
        self.assertIn("property_age", engineered.columns)
        self.assertEqual(engineered["property_age"].iloc[0], 15)
        
        # Check total rooms
        self.assertIn("total_rooms", engineered.columns)
        self.assertEqual(engineered["total_rooms"].iloc[0], 5.5)
        
        # Check sqft per room
        self.assertIn("sqft_per_room", engineered.columns)
        self.assertAlmostEqual(engineered["sqft_per_room"].iloc[0], 2400 / 5.5, places=2)
        
        # Check condition grade interaction
        self.assertIn("condition_grade_interaction", engineered.columns)
        self.assertEqual(engineered["condition_grade_interaction"].iloc[0], 32)
        
    def test_predictor_inference(self):
        """Verify model inference pipeline returns valid output with confidence bounds."""
        predictor = HousePricePredictor()
        result = predictor.predict(self.sample_raw_property)
        
        self.assertIn("predicted_price", result)
        self.assertIn("price_range_low", result)
        self.assertIn("price_range_high", result)
        self.assertIn("price_per_sqft", result)
        
        self.assertGreater(result["predicted_price"], 200000)
        self.assertLess(result["price_range_low"], result["predicted_price"])
        self.assertGreater(result["price_range_high"], result["predicted_price"])
        self.assertEqual(result["currency"], "INR (₹)")
        
        # Test inference with distance in kilometers
        km_property = dict(self.sample_raw_property)
        del km_property["distance_to_center_miles"]
        km_property["distance_to_center_km"] = 8.0
        result_km = predictor.predict(km_property)
        self.assertGreater(result_km["predicted_price"], 200000)
        
    def test_flask_health_endpoint(self):
        """Test Flask API /health endpoint."""
        client = app.test_client()
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])
        
    def test_flask_predict_endpoint(self):
        """Test Flask API /predict endpoint with JSON payload."""
        client = app.test_client()
        response = client.post("/predict", json=self.sample_raw_property)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("predicted_price", data["valuation"])

if __name__ == "__main__":
    unittest.main()
