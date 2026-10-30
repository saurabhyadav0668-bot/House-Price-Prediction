"""
Feature Engineering Module
Creates domain-specific real estate features to boost regression model predictive accuracy.
"""

import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame, current_year: int = 2025) -> pd.DataFrame:
    """
    Applies real estate domain-specific feature transformations:
    - Age of the property and effective age accounting for renovations
    - Total room count and living space per room
    - Ratio of living space to lot area
    - Luxury index combining waterfront, view quality, pool, and structural grade
    - Interaction terms between property condition and architectural grade
    """
    data = df.copy()
    
    # 1. Age calculations
    if "yr_built" in data.columns:
        data["property_age"] = np.clip(current_year - data["yr_built"], 0, None)
    else:
        data["property_age"] = 0
        
    if "yr_renovated" in data.columns:
        data["is_renovated"] = (data["yr_renovated"] > 0).astype(int)
        data["effective_age"] = np.where(
            data["yr_renovated"] > 0,
            np.clip(current_year - data["yr_renovated"], 0, None),
            data["property_age"]
        )
        data["years_since_renovation"] = np.where(
            data["yr_renovated"] > 0,
            np.clip(current_year - data["yr_renovated"], 0, None),
            0
        )
    else:
        data["is_renovated"] = 0
        data["effective_age"] = data["property_age"]
        data["years_since_renovation"] = 0

    # 2. Room metrics
    bedrooms = data["bedrooms"] if "bedrooms" in data.columns else 1
    bathrooms = data["bathrooms"] if "bathrooms" in data.columns else 1.0
    data["total_rooms"] = bedrooms + bathrooms
    data["bed_bath_ratio"] = np.where(bathrooms > 0, bedrooms / bathrooms, bedrooms)
    
    # 3. Space metrics
    if "sqft_living" in data.columns:
        data["sqft_per_room"] = np.where(data["total_rooms"] > 0, data["sqft_living"] / data["total_rooms"], data["sqft_living"])
        
        if "sqft_lot" in data.columns:
            data["living_to_lot_ratio"] = np.where(data["sqft_lot"] > 0, data["sqft_living"] / data["sqft_lot"], 0.0)
            data["unbuilt_lot_area"] = np.clip(data["sqft_lot"] - data["sqft_living"], 0, None)
            
    # 4. Luxury and quality score
    waterfront = data["waterfront"] if "waterfront" in data.columns else 0
    view_quality = data["view_quality"] if "view_quality" in data.columns else 0
    has_pool = data["has_pool"] if "has_pool" in data.columns else 0
    grade = data["grade"] if "grade" in data.columns else 7
    condition = data["condition"] if "condition" in data.columns else 3
    
    data["luxury_score"] = (
        waterfront * 3.0 +
        view_quality * 1.5 +
        has_pool * 1.5 +
        np.where(grade >= 9, (grade - 8) * 1.2, 0.0)
    )
    
    # 5. Quality interaction term
    data["condition_grade_interaction"] = condition * grade
    
    return data
