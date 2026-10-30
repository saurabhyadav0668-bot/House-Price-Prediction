"""
Dataset Generation Script for Real Estate House Price Prediction
Generates a realistic, multifaceted real-estate dataset modeling realistic
economic relationships, location multipliers, structural features, and amenities.
"""

import os
import numpy as np
import pandas as pd

def generate_housing_dataset(n_samples: int = 5000, random_seed: int = 42) -> pd.DataFrame:
    np.random.seed(random_seed)
    
    locations = [
        "Downtown Metro",
        "Bellevue Tech Park",
        "Lakeview Hills",
        "Greenwood Suburb",
        "Northgate District",
        "Westside Heights",
        "Eastgate Valley",
        "Riverdale Park",
        "South Ridge",
        "Pinecrest Meadows"
    ]
    
    # Neighborhood baseline price per sqft multiplier and mean distance to city center
    location_profiles = {
        "Downtown Metro": {"rate_sqft": 420, "base_val": 220000, "dist_center": 2.5, "prob": 0.12},
        "Bellevue Tech Park": {"rate_sqft": 400, "base_val": 240000, "dist_center": 6.0, "prob": 0.13},
        "Lakeview Hills": {"rate_sqft": 360, "base_val": 210000, "dist_center": 9.5, "prob": 0.10},
        "Greenwood Suburb": {"rate_sqft": 280, "base_val": 160000, "dist_center": 12.0, "prob": 0.14},
        "Northgate District": {"rate_sqft": 260, "base_val": 150000, "dist_center": 14.0, "prob": 0.11},
        "Westside Heights": {"rate_sqft": 250, "base_val": 140000, "dist_center": 16.5, "prob": 0.10},
        "Eastgate Valley": {"rate_sqft": 220, "base_val": 130000, "dist_center": 18.0, "prob": 0.10},
        "Riverdale Park": {"rate_sqft": 200, "base_val": 120000, "dist_center": 20.0, "prob": 0.08},
        "South Ridge": {"rate_sqft": 180, "base_val": 110000, "dist_center": 23.5, "prob": 0.07},
        "Pinecrest Meadows": {"rate_sqft": 170, "base_val": 100000, "dist_center": 28.0, "prob": 0.05}
    }
    
    loc_names = list(location_profiles.keys())
    loc_probs = [location_profiles[loc]["prob"] for loc in loc_names]
    selected_locs = np.random.choice(loc_names, size=n_samples, p=loc_probs)
    
    # Square footage living area: log-normal distribution centered around ~2,100 sqft
    sqft_living = np.clip(np.random.lognormal(mean=7.55, sigma=0.42, size=n_samples), 650, 6200).astype(int)
    
    # Bedrooms correlated with sqft_living
    expected_beds = (sqft_living / 580) + np.random.normal(0, 0.45, size=n_samples)
    bedrooms = np.clip(np.round(expected_beds), 1, 6).astype(int)
    
    # Bathrooms correlated with bedrooms and living area
    expected_baths = 0.8 + 0.55 * bedrooms + (sqft_living / 2200) + np.random.normal(0, 0.25, size=n_samples)
    # Round to nearest 0.25
    bathrooms = np.clip(np.round(expected_baths * 4) / 4, 1.0, 5.0)
    
    # Floors: 1.0, 1.5, 2.0, 2.5, 3.0
    floor_choices = [1.0, 1.5, 2.0, 2.5, 3.0]
    floor_probs = [0.42, 0.10, 0.38, 0.04, 0.06]
    floors = np.random.choice(floor_choices, size=n_samples, p=floor_probs)
    
    # Lot size
    sqft_lot = np.clip(sqft_living * np.random.uniform(1.4, 7.5, size=n_samples) + np.random.exponential(3500, size=n_samples), 1200, 35000).astype(int)
    
    # Waterfront (rare luxury, 3% overall, more in Lakeview Hills)
    waterfront_prob = np.where(selected_locs == "Lakeview Hills", 0.15, 0.02)
    waterfront = (np.random.rand(n_samples) < waterfront_prob).astype(int)
    
    # View quality (0 to 4)
    view_quality = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.72, 0.12, 0.08, 0.05, 0.03])
    view_quality = np.where(waterfront == 1, np.clip(view_quality + 2, 2, 4), view_quality)
    
    # Condition (1 to 5, mean ~ 3.4)
    condition = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.02, 0.08, 0.52, 0.28, 0.10])
    
    # Construction grade (4 to 11, correlated with living area and location)
    base_grade = 6 + (sqft_living / 1000) * 0.75 + np.random.normal(0, 0.7, size=n_samples)
    grade = np.clip(np.round(base_grade), 4, 11).astype(int)
    
    # Year built (1945 to 2024)
    yr_built = np.random.randint(1945, 2025, size=n_samples)
    
    # Year renovated (approx 20% renovated if built before 2005)
    renovate_candidate = (yr_built < 2005) & (np.random.rand(n_samples) < 0.22)
    yr_renovated = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if renovate_candidate[i]:
            yr_renovated[i] = np.random.randint(yr_built[i] + 10, 2025)
            
    # Garage & Pool
    has_garage = (np.random.rand(n_samples) < 0.82).astype(int)
    has_pool = (np.random.rand(n_samples) < 0.14).astype(int)
    
    # Distance to city center with noise
    distances = np.array([location_profiles[loc]["dist_center"] for loc in selected_locs])
    distance_to_center = np.clip(distances + np.random.normal(0, 1.2, size=n_samples), 0.8, 32.0).round(1)
    
    # Target variable: PRICE Calculation
    prices = []
    for i in range(n_samples):
        loc = selected_locs[i]
        profile = location_profiles[loc]
        
        # Base economic component
        loc_base = profile["base_val"]
        living_val = sqft_living[i] * profile["rate_sqft"]
        lot_val = sqft_lot[i] * 12.0
        
        # Bedroom & Bathroom adjustments
        rooms_val = (bedrooms[i] * 18000) + (bathrooms[i] * 22000)
        
        # Architectural Grade Multiplier
        # Grade 7 is standard benchmark; each point above/below adjusts price
        grade_multiplier = 1.0 + (grade[i] - 7) * 0.11
        
        # Condition Adjustment
        cond_multiplier = 1.0 + (condition[i] - 3) * 0.05
        
        # Age Depreciation & Renovation
        effective_year = yr_renovated[i] if yr_renovated[i] > 0 else yr_built[i]
        age = 2025 - effective_year
        age_discount = max(-0.25, -0.0035 * age)  # Depreciates gently, capped
        if yr_renovated[i] > 0:
            age_discount += 0.08  # Renovation premium
            
        # Luxury Amenities
        waterfront_val = 220000 if waterfront[i] == 1 else 0
        view_val = view_quality[i] * 28000
        garage_val = 25000 if has_garage[i] == 1 else 0
        pool_val = 38000 if has_pool[i] == 1 else 0
        
        # Distance penalty
        distance_penalty = -1800 * distance_to_center[i]
        
        subtotal = (
            (loc_base + living_val + lot_val + rooms_val) * grade_multiplier * cond_multiplier * (1.0 + age_discount)
            + waterfront_val + view_val + garage_val + pool_val + distance_penalty
        )
        
        # Realistic market variance (log-normal residual around 5% noise)
        noise = np.random.normal(1.0, 0.045)
        calculated_price = np.round(max(110000, subtotal * noise), -2)
        prices.append(calculated_price)
        
    df = pd.DataFrame({
        "id": [f"PROP-{10001 + i}" for i in range(n_samples)],
        "location": selected_locs,
        "sqft_living": sqft_living,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_lot": sqft_lot,
        "floors": floors,
        "waterfront": waterfront,
        "view_quality": view_quality,
        "condition": condition,
        "grade": grade,
        "yr_built": yr_built,
        "yr_renovated": yr_renovated,
        "has_garage": has_garage,
        "has_pool": has_pool,
        "distance_to_center_miles": distance_to_center,
        "price": prices
    })
    
    return df

if __name__ == "__main__":
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "house_prices.csv")
    
    print(f"Generating 5,000 real estate records...")
    df = generate_housing_dataset(n_samples=5000, random_seed=42)
    df.to_csv(csv_path, index=False)
    print(f"Dataset successfully saved to: {csv_path}")
    print(f"Shape: {df.shape}")
    print("\nSummary Statistics:")
    print(df[["sqft_living", "bedrooms", "bathrooms", "grade", "price"]].describe().round(2))
