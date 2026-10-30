"""
Streamlit Web Application for Dynamic Home Price Prediction
Provides an interactive GUI for homebuyers, sellers, and real estate analysts:
- Live dynamic property valuation with confidence intervals
- Market comparison metrics against neighborhood benchmarks
- Interactive Exploratory Data Analysis (EDA) explorer
- Model benchmark performance dashboard and feature importances
- Batch property valuation tool
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from src.predict import HousePricePredictor
from src.feature_engineering import engineer_features

# Set page configuration
st.set_page_config(
    page_title="AI Home Price Prediction System",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .price-badge {
        font-size: 2.2rem;
        font-weight: 800;
        color: #047857;
    }
    .range-badge {
        font-size: 1.05rem;
        color: #6B7280;
    }
</style>
""", unsafe_allow_html=True)

# Cache predictor and data loaders
@st.cache_resource
def get_predictor():
    return HousePricePredictor()

@st.cache_data
def load_housing_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "data", "house_prices.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

@st.cache_data
def load_model_comparison():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "models", "model_comparison.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return None

predictor = get_predictor()
df_data = load_housing_data()
comparison_data = load_model_comparison()

def format_inr(val, include_words=True):
    """Formats numeric value into Indian Rupee (INR ₹) representation with optional Lakhs/Crores notation."""
    if val is None or (isinstance(val, (int, float)) and np.isnan(val)):
        return "₹0"
    num = round(float(val))
    s = str(abs(num))
    if len(s) <= 3:
        formatted = s
    else:
        last3 = s[-3:]
        rem = s[:-3]
        groups = []
        while len(rem) > 2:
            groups.insert(0, rem[-2:])
            rem = rem[:-2]
        if rem:
            groups.insert(0, rem)
        formatted = ",".join(groups) + "," + last3
    
    prefix = "-₹" if num < 0 else "₹"
    res = f"{prefix}{formatted}"
    if include_words:
        if abs(num) >= 10_000_000:
            res += f" ({abs(num)/10_000_000:.2f} Cr)"
        elif abs(num) >= 100_000:
            res += f" ({abs(num)/100_000:.2f} Lakhs)"
    return res

# Application Header
st.markdown('<div class="main-title">🏡 AI Home Price Prediction & Real Estate Valuation System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Powered by XGBoost, Random Forest, Gradient Boosting & Scikit-Learn pipelines</div>', unsafe_allow_html=True)

# Main Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Live Property Valuation",
    "📊 Market Analysis & EDA",
    "🤖 Model Benchmarks & Explainability",
    "📂 Batch Valuation Tool"
])

# ==============================================================================
# TAB 1: LIVE PROPERTY VALUATION
# ==============================================================================
with tab1:
    st.subheader("Estimate Real Estate Market Value")
    st.write("Specify property features below to generate an AI-driven price estimate with confidence bounds.")
    
    col_input, col_result = st.columns([1.1, 0.9], gap="large")
    
    with col_input:
        st.markdown("#### 1. Location & Space Attributes")
        loc_col1, loc_col2 = st.columns(2)
        with loc_col1:
            location_options = [
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
            location = st.selectbox("Location / Neighborhood", location_options, index=1)
        with loc_col2:
            distance_to_center = st.slider("Distance to City Center (kilometers)", 0.5, 50.0, 10.0, 0.5)
            
        space_col1, space_col2 = st.columns(2)
        with space_col1:
            sqft_living = st.number_input("Living Area (sq. ft.)", min_value=600, max_value=8000, value=2400, step=50)
        with space_col2:
            sqft_lot = st.number_input("Lot Size (sq. ft.)", min_value=1000, max_value=50000, value=7500, step=250)
            
        st.markdown("#### 2. Layout & Architecture")
        room_col1, room_col2, room_col3 = st.columns(3)
        with room_col1:
            bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5, 6], index=2)
        with room_col2:
            bathrooms = st.selectbox("Bathrooms", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], index=3)
        with room_col3:
            floors = st.selectbox("Floors / Stories", [1.0, 1.5, 2.0, 2.5, 3.0], index=2)
            
        grade_col1, grade_col2 = st.columns(2)
        with grade_col1:
            grade = st.slider("Construction Grade (4: Low to 11: Luxury Mansion)", 4, 11, 8)
        with grade_col2:
            condition = st.select_slider("Overall Condition", options=[1, 2, 3, 4, 5], value=4, format_func=lambda x: {
                1: "1 - Poor", 2: "2 - Fair", 3: "3 - Average", 4: "4 - Good", 5: "5 - Pristine"
            }[x])
            
        st.markdown("#### 3. Age & Historical Renovations")
        age_col1, age_col2 = st.columns(2)
        with age_col1:
            yr_built = st.slider("Year Built", 1940, 2025, 2012)
        with age_col2:
            is_renovated = st.checkbox("Property Has Been Renovated", value=False)
            if is_renovated:
                yr_renovated = st.slider("Year Renovated", yr_built, 2025, max(yr_built + 5, 2020))
            else:
                yr_renovated = 0
                
        st.markdown("#### 4. Premium Amenities")
        amenity_col1, amenity_col2, amenity_col3, amenity_col4 = st.columns(4)
        with amenity_col1:
            waterfront = 1 if st.checkbox("Waterfront View", value=False) else 0
        with amenity_col2:
            view_quality = st.selectbox("Scenic View", [0, 1, 2, 3, 4], format_func=lambda x: ["0 - None", "1 - Fair", "2 - Good", "3 - Excellent", "4 - Panoramic"][x])
        with amenity_col3:
            has_garage = 1 if st.checkbox("Garage Included", value=True) else 0
        with amenity_col4:
            has_pool = 1 if st.checkbox("Private Swimming Pool", value=False) else 0
            
    with col_result:
        st.markdown("#### 📈 Valuation Estimate Summary")
        
        # Prepare input dictionary
        input_data = {
            "location": location,
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
            "distance_to_center_miles": distance_to_center
        }
        
        # Run inference
        prediction_output = predictor.predict(input_data)
        
        pred_price = prediction_output["predicted_price"]
        low_range = prediction_output["price_range_low"]
        high_range = prediction_output["price_range_high"]
        price_sqft = prediction_output["price_per_sqft"]
        
        st.markdown(f"""
        <div class="metric-card">
            <span style="font-size: 0.95rem; font-weight: 600; color: #4B5563;">ESTIMATED MARKET VALUE</span><br>
            <div class="price-badge">{format_inr(pred_price)}</div>
            <div class="range-badge">90% Confidence Interval: <b>{format_inr(low_range, False)} &ndash; {format_inr(high_range, False)}</b></div>
            <div style="margin-top: 10px; font-size: 0.95rem; color: #1F2937;">
                <b>Unit Rate:</b> ₹{price_sqft:,.2f} per sq.ft. &nbsp;|&nbsp; 
                <b>Model:</b> {prediction_output['model_used']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        
        # Neighborhood comparative benchmark
        if df_data is not None:
            neighborhood_df = df_data[df_data["location"] == location]
            avg_neigh_price = neighborhood_df["price"].mean()
            diff_pct = ((pred_price - avg_neigh_price) / avg_neigh_price) * 100
            diff_sign = "+" if diff_pct >= 0 else ""
            
            b1, b2 = st.columns(2)
            b1.metric("Neighborhood Avg Price", format_inr(avg_neigh_price, False))
            b2.metric("Relative to Neighborhood", f"{diff_sign}{diff_pct:.1f}%")
            
        st.markdown("##### Valuation Driver Breakdown")
        breakdown_items = [
            ("Base Location Quality", location),
            ("Distance to City Center", f"{distance_to_center:.1f} kilometers"),
            ("Living Space", f"{sqft_living:,} sq.ft."),
            ("Total Rooms", f"{bedrooms} Beds, {bathrooms} Baths"),
            ("Effective Property Age", f"{2025 - (yr_renovated if yr_renovated > 0 else yr_built)} years"),
            ("Construction Grade", f"Level {grade}/11"),
            ("Luxury Amenities", ("Waterfront " if waterfront else "") + ("Pool " if has_pool else "") + ("Garage " if has_garage else "") or "Standard")
        ]
        
        driver_df = pd.DataFrame(breakdown_items, columns=["Valuation Factor", "Property Specification"])
        st.dataframe(driver_df, hide_index=True, use_container_width=True)

# ==============================================================================
# TAB 2: MARKET ANALYSIS & EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
with tab2:
    st.subheader("Exploratory Data Analysis & Market Trends")
    st.write("Understand the distribution of property prices and key drivers across 5,000 recorded real estate transactions.")
    
    if df_data is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{len(df_data):,}")
        c2.metric("Median Sale Price", format_inr(df_data['price'].median(), False))
        c3.metric("Average Living Area", f"{df_data['sqft_living'].mean():,.0f} sq.ft.")
        c4.metric("Avg Price / Sq.Ft.", f"₹{(df_data['price'] / df_data['sqft_living']).mean():,.2f}")
        
        st.markdown("---")
        
        col_eda1, col_eda2 = st.columns(2)
        with col_eda1:
            st.markdown("#### Price Distribution Across Market")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(df_data["price"] / 100000, kde=True, ax=ax, color="#1e40af", bins=30)
            ax.set_title("Property Price Distribution (in ₹ Lakhs)", fontsize=11, fontweight="bold")
            ax.set_xlabel("Price (₹ Lakhs)")
            st.pyplot(fig)
            plt.close()
            
        with col_eda2:
            st.markdown("#### Price by Neighborhood Tier")
            fig, ax = plt.subplots(figsize=(8, 5))
            order = df_data.groupby("location")["price"].median().sort_values(ascending=False).index
            sns.boxplot(data=df_data, x="price", y="location", order=order, ax=ax, palette="mako")
            ax.set_title("Sale Price by Location", fontsize=11, fontweight="bold")
            ax.set_xlabel("Price (₹)")
            ax.xaxis.set_major_formatter("₹{x:,.0f}")
            st.pyplot(fig)
            plt.close()
            
        col_eda3, col_eda4 = st.columns(2)
        with col_eda3:
            st.markdown("#### Square Footage vs Price by Architectural Grade")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=df_data, x="sqft_living", y="price", hue="grade", palette="viridis", alpha=0.6, ax=ax)
            ax.set_title("Living Area vs. Price", fontsize=11, fontweight="bold")
            ax.set_xlabel("Living Area (sq.ft)")
            ax.yaxis.set_major_formatter("₹{x:,.0f}")
            st.pyplot(fig)
            plt.close()
            
        with col_eda4:
            st.markdown("#### Feature Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(8, 5))
            num_cols = ["price", "sqft_living", "grade", "bathrooms", "bedrooms", "view_quality", "yr_built"]
            corr_mat = df_data[num_cols].corr()
            sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
            ax.set_title("Feature Correlations with Target Price", fontsize=11, fontweight="bold")
            st.pyplot(fig)
            plt.close()

# ==============================================================================
# TAB 3: MODEL BENCHMARKS & EXPLAINABILITY
# ==============================================================================
with tab3:
    st.subheader("Model Performance Evaluation & Diagnostic Benchmarks")
    st.write("Rigorous comparison across Linear, Regularized, Decision Tree, and Advanced Boosting architectures evaluated with 5-Fold Cross-Validation on holdout test set.")
    
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        # Rename USD column labels to INR (₹)
        rename_map = {
            "Test_MAE ($)": "Test_MAE (₹)",
            "Test_RMSE ($)": "Test_RMSE (₹)"
        }
        display_comp_df = comp_df.rename(columns=rename_map)
        st.dataframe(display_comp_df, use_container_width=True)
        
        st.markdown("---")
        
        diag_col1, diag_col2 = st.columns(2)
        with diag_col1:
            st.markdown("#### Model Accuracy Comparison (R² Score)")
            fig, ax = plt.subplots(figsize=(8, 5))
            palette = sns.color_palette("crest", len(comp_df))
            sns.barplot(data=comp_df, x="Test_R2", y="Model", palette=palette, ax=ax)
            ax.set_title("Test R² Score Comparison (Higher is Better)", fontsize=11, fontweight="bold")
            ax.set_xlim(0, 1.05)
            for i, v in enumerate(comp_df["Test_R2"]):
                ax.text(v + 0.01, i, f"{v:.4f}", va="center", fontweight="bold")
            st.pyplot(fig)
            plt.close()
            
        with diag_col2:
            st.markdown("#### Test Error Metric (RMSE in ₹ INR)")
            fig, ax = plt.subplots(figsize=(8, 5))
            rmse_col = "Test_RMSE (₹)" if "Test_RMSE (₹)" in display_comp_df.columns else "Test_RMSE ($)"
            sns.barplot(data=display_comp_df, x=rmse_col, y="Model", palette="flare", ax=ax)
            ax.set_title("Test Root Mean Squared Error (Lower is Better)", fontsize=11, fontweight="bold")
            ax.xaxis.set_major_formatter("₹{x:,.0f}")
            for i, v in enumerate(display_comp_df[rmse_col]):
                ax.text(v + 1500, i, f"₹{v:,.0f}", va="center", fontweight="bold")
            st.pyplot(fig)
            plt.close()
            
        # Feature Importance section
        st.markdown("#### Winning Model Feature Importances")
        feat_img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "feature_importance.png")
        resid_img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "residuals_plot.png")
        
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            if os.path.exists(feat_img_path):
                st.image(feat_img_path, caption="Top Influential Valuation Determinants (XGBoost Gain)")
        with f_col2:
            if os.path.exists(resid_img_path):
                st.image(resid_img_path, caption="Actual vs. Predicted Price & Residual Distribution")

# ==============================================================================
# TAB 4: BATCH PROPERTY VALUATION
# ==============================================================================
with tab4:
    st.subheader("Batch Property Valuation")
    st.write("Upload a CSV file of properties or download a template to estimate valuations for multiple homes simultaneously.")
    
    sample_df = pd.DataFrame([
        {"location": "Bellevue Tech Park", "sqft_living": 2800, "bedrooms": 4, "bathrooms": 3.0, "sqft_lot": 8000, "floors": 2.0, "waterfront": 0, "view_quality": 1, "condition": 4, "grade": 9, "yr_built": 2018, "yr_renovated": 0, "has_garage": 1, "has_pool": 1, "distance_to_center_km": 9.0},
        {"location": "Lakeview Hills", "sqft_living": 3400, "bedrooms": 4, "bathrooms": 3.5, "sqft_lot": 12000, "floors": 2.0, "waterfront": 1, "view_quality": 4, "condition": 5, "grade": 10, "yr_built": 2015, "yr_renovated": 0, "has_garage": 1, "has_pool": 1, "distance_to_center_km": 14.5},
        {"location": "Downtown Metro", "sqft_living": 1450, "bedrooms": 2, "bathrooms": 2.0, "sqft_lot": 2000, "floors": 1.0, "waterfront": 0, "view_quality": 2, "condition": 4, "grade": 8, "yr_built": 2020, "yr_renovated": 0, "has_garage": 1, "has_pool": 0, "distance_to_center_km": 2.0},
        {"location": "South Ridge", "sqft_living": 1800, "bedrooms": 3, "bathrooms": 2.0, "sqft_lot": 6500, "floors": 1.5, "waterfront": 0, "view_quality": 0, "condition": 3, "grade": 7, "yr_built": 1995, "yr_renovated": 2016, "has_garage": 1, "has_pool": 0, "distance_to_center_km": 38.0}
    ])
    
    st.download_button(
        label="📥 Download CSV Sample Template",
        data=sample_df.to_csv(index=False),
        file_name="batch_valuation_template.csv",
        mime="text/csv"
    )
    
    uploaded_file = st.file_uploader("Upload CSV for Valuation", type=["csv"])
    
    batch_data = sample_df if uploaded_file is None else pd.read_csv(uploaded_file)
    
    st.write(f"Evaluating {len(batch_data)} properties:")
    
    if st.button("🚀 Calculate Batch Predictions", type="primary"):
        with st.spinner("Valuating portfolio..."):
            predictions = []
            for _, row in batch_data.iterrows():
                pred_res = predictor.predict(row.to_dict())
                predictions.append({
                    "Estimated_Price": pred_res["formatted_price"],
                    "Confidence_Range": pred_res["formatted_range"],
                    "Price_Per_SqFt": f"₹{pred_res['price_per_sqft']:,.2f}"
                })
            results_df = pd.concat([batch_data, pd.DataFrame(predictions)], axis=1)
            st.success("Batch Valuation complete!")
            st.dataframe(results_df, use_container_width=True)
            
            st.download_button(
                label="📥 Export Valuations to CSV",
                data=results_df.to_csv(index=False),
                file_name="valuated_properties.csv",
                mime="text/csv"
            )
