"""
Data Preprocessing Pipeline Module
Handles data loading, cleaning, categorical encoding, scaling, and transformation.
"""

import os
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib

from src.feature_engineering import engineer_features

CATEGORICAL_FEATURES = ["location"]

NUMERICAL_FEATURES = [
    "sqft_living",
    "bedrooms",
    "bathrooms",
    "sqft_lot",
    "floors",
    "waterfront",
    "view_quality",
    "condition",
    "grade",
    "distance_to_center_miles",
    "has_garage",
    "has_pool",
    # Engineered features
    "property_age",
    "effective_age",
    "is_renovated",
    "years_since_renovation",
    "total_rooms",
    "bed_bath_ratio",
    "sqft_per_room",
    "living_to_lot_ratio",
    "unbuilt_lot_area",
    "luxury_score",
    "condition_grade_interaction"
]

TARGET_COLUMN = "price"

def load_data(file_path: str) -> pd.DataFrame:
    """Loads CSV dataset and verifies essential columns."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
    df = pd.read_csv(file_path)
    return df

def clean_data(df: pd.DataFrame, drop_outliers: bool = True) -> pd.DataFrame:
    """
    Cleans raw dataframe:
    - Removes duplicate entries
    - Handles missing values
    - Optionally removes extreme target outliers using IQR capping
    """
    data = df.copy()
    data = data.drop_duplicates()
    
    # Fill missing values if any
    num_cols = data.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if data[col].isnull().sum() > 0:
            data[col] = data[col].fillna(data[col].median())
            
    cat_cols = data.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if data[col].isnull().sum() > 0:
            data[col] = data[col].fillna(data[col].mode()[0])
            
    if drop_outliers and TARGET_COLUMN in data.columns:
        # IQR filtering on price to prevent extreme noise artifacts
        q1 = data[TARGET_COLUMN].quantile(0.01)
        q99 = data[TARGET_COLUMN].quantile(0.99)
        data = data[(data[TARGET_COLUMN] >= q1) & (data[TARGET_COLUMN] <= q99)]
        
    return data

def build_preprocessor() -> ColumnTransformer:
    """Creates a ColumnTransformer for categorical and numerical features."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )
    return preprocessor

def prepare_data(
    file_path: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer, List[str]]:
    """
    End-to-end data preparation workflow:
    1. Loads raw dataset
    2. Cleans data
    3. Performs feature engineering
    4. Splits into Train and Test sets
    5. Fits preprocessor on training data
    6. Extracts feature names
    """
    raw_df = load_data(file_path)
    cleaned_df = clean_data(raw_df)
    engineered_df = engineer_features(cleaned_df)
    
    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    X = engineered_df[feature_cols]
    y = engineered_df[TARGET_COLUMN]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Extract feature names after OneHotEncoding
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = NUMERICAL_FEATURES + cat_feature_names
    
    X_train_df = pd.DataFrame(X_train_transformed, columns=all_feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_transformed, columns=all_feature_names, index=X_test.index)
    
    return X_train_df, X_test_df, y_train, y_test, preprocessor, all_feature_names
