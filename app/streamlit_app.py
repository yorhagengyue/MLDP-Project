import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Ensure we can import from local directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from predictor_simple import SimplePredictor

# Page configuration
st.set_page_config(
    page_title="HousePrice AI | Intelligent Valuation",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern UI
st.markdown("""
<style>
    /* Global Styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Card Component */
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #f0f0f0;
        margin-bottom: 1rem;
        transition: transform 0.2s ease-in-out;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #1f2937;
    }
    
    .metric-label {
        font-size: 14px;
        color: #6b7280;
        font-weight: 500;
        margin-bottom: 4px;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.5);
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    /* Button Styling */
    .stButton > button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2);
        transition: all 0.2s;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #1d4ed8;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.3);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding-top: 1rem;
        padding-bottom: 1rem;
        font-weight: 600;
    }
    
    /* Success/Error Message Styling */
    .stAlert {
        border-radius: 8px;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("""
<div class="main-header">
    <div class="header-title">🏠 HousePrice AI</div>
    <div class="header-subtitle">Advanced Machine Learning Valuation System for Ames Housing Market</div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    """Load the simple predictor"""
    try:
        predictor = SimplePredictor()
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_data
def load_data():
    """Load original training data for reference"""
    try:
        train = pd.read_csv('../data/train.csv')
        return train
    except:
        return None

# Load resources
predictor = load_predictor()
train_data = load_data()

if predictor is not None and train_data is not None:
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/home.png", width=80)
        st.markdown("### Control Panel")
        st.info("Select a sample property from the dataset to generate a real-time valuation prediction.")
        
        st.markdown("---")
        st.markdown("### System Status")
        st.success("● Model Loaded (Stacking Ensemble)")
        st.success("● Database Connected (1,460 Records)")
        
        st.markdown("---")
        st.markdown("© 2026 HousePrice AI")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Price Prediction", "📊 Model Analytics", "📈 Market Insights"])

    with tab1:
        st.markdown("### Property Valuation")
        
        # Selection
        sample_indices = predictor.get_sample_indices(20)
        
        col_sel, col_btn = st.columns([3, 1])
        with col_sel:
            selected_idx = st.selectbox(
                "Select Property Sample:",
                options=sample_indices,
                format_func=lambda x: f"Property ID #{x} - {predictor.get_house_summary(x).get('Neighborhood', 'Unknown')}"
            )
        
        # Get house details
        house = predictor.get_house_summary(selected_idx)
        
        # Display Property Cards
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        def metric_card(label, value, col):
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

        metric_card("Living Area", f"{house['Living Area (sq ft)']} sqft", col1)
        metric_card("Overall Quality", f"{house['Overall Quality']}/10", col2)
        metric_card("Year Built", house['Year Built'], col3)
        metric_card("Neighborhood", house['Neighborhood'], col4)
        
        col5, col6, col7, col8 = st.columns(4)
        metric_card("Bedrooms", house['Bedrooms'], col5)
        metric_card("Bathrooms", house['Bathrooms'], col6)
        metric_card("Total Rooms", house['Total Rooms'], col7)
        metric_card("Garage", f"{house['Garage Cars']} Cars", col8)

        # Predict Action
        st.markdown("---")
        with col_btn:
            st.write("") # Spacer
            st.write("") 
            predict_btn = st.button("Generate Valuation", type="primary")

        if predict_btn:
            with st.spinner("Analyzing property features..."):
                try:
                    predicted_price, individual_preds, pred_log = predictor.predict_by_index(selected_idx)
                    actual_price_val = float(house['Actual Price'].replace('$', '').replace(',', ''))
                    error_pct = abs(predicted_price - actual_price_val) / actual_price_val * 100
                    
                    # Result Display
                    st.markdown("### Valuation Results")
                    
                    res_col1, res_col2, res_col3 = st.columns(3)
                    
                    with res_col1:
                        st.markdown(f"""
                        <div class="metric-card" style="border-left: 5px solid #2563eb;">
                            <div class="metric-label">AI Valuation</div>
                            <div class="metric-value" style="color: #2563eb;">${predicted_price:,.0f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with res_col2:
                        st.markdown(f"""
                        <div class="metric-card" style="border-left: 5px solid #10b981;">
                            <div class="metric-label">Actual Sale Price</div>
                            <div class="metric-value" style="color: #10b981;">{house['Actual Price']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with res_col3:
                        color = "#ef4444" if error_pct > 10 else "#10b981"
                        st.markdown(f"""
                        <div class="metric-card" style="border-left: 5px solid {color};">
                            <div class="metric-label">Prediction Error</div>
                            <div class="metric-value" style="color: {color};">{error_pct:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Visualization
                    st.subheader("Model Confidence Analysis")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    
                    models = list(individual_preds.keys()) + ['Ensemble Final']
                    values = list(individual_preds.values()) + [predicted_price]
                    colors = ['#cbd5e1'] * len(individual_preds) + ['#2563eb']
                    
                    y_pos = np.arange(len(models))
                    ax.barh(y_pos, values, align='center', color=colors)
                    ax.axvline(actual_price_val, color='#ef4444', linestyle='--', label='Actual Price')
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(models)
                    ax.set_xlabel('Valuation ($)')
                    ax.legend()
                    ax.grid(axis='x', alpha=0.3)
                    
                    # Clean styling
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")

    with tab2:
        st.markdown("### Model Architecture & Performance")
        
        perf_col1, perf_col2 = st.columns(2)
        
        with perf_col1:
            st.markdown("#### Performance Metrics")
            metrics_df = pd.DataFrame({
                'Metric': ['RMSE (Cross-Val)', 'Kaggle Score', 'R² Score', 'CV-Test Gap'],
                'Value': ['0.1065', '0.1220', '0.920', '1.55%'],
                'Status': ['Excellent', 'Top 30%', 'High Fit', 'Stable']
            })
            st.table(metrics_df)
            
        with perf_col2:
            st.markdown("#### Feature Importance (Top 5)")
            st.markdown("""
            1. **Overall Quality** (0.79 corr)
            2. **Living Area** (0.71 corr)
            3. **Garage Cars** (Capacity)
            4. **Total Bathrooms** (Calculated)
            5. **Year Built** (Age factor)
            """)
            
        st.markdown("#### Model Comparison")
        fig, ax = plt.subplots(figsize=(10, 5))
        models = list(predictor.cv_scores.keys())
        scores = list(predictor.cv_scores.values())
        
        bars = ax.bar(models, scores, color='#3b82f6')
        ax.set_ylabel('RMSE (Lower is Better)')
        ax.set_ylim(0.10, 0.12)
        ax.grid(axis='y', alpha=0.3)
        
        # Highlight best model
        best_idx = np.argmin(scores)
        bars[best_idx].set_color('#10b981')
        
        st.pyplot(fig)

    with tab3:
        st.markdown("### Market Intelligence")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Price Distribution")
            fig, ax = plt.subplots()
            ax.hist(train_data['SalePrice'], bins=50, color='#3b82f6', alpha=0.7)
            ax.set_xlabel('Price ($)')
            ax.set_ylabel('Count')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)
            
        with col2:
            st.markdown("#### Market Stats")
            st.markdown(f"""
            - **Total Properties:** {len(train_data):,}
            - **Average Price:** ${train_data['SalePrice'].mean():,.0f}
            - **Median Price:** ${train_data['SalePrice'].median():,.0f}
            - **Price Range:** ${train_data['SalePrice'].min():,.0f} - ${train_data['SalePrice'].max():,.0f}
            """)
            
            st.info("""
            **Insight:** The market shows a right-skewed distribution, indicating most homes are affordable 
            with a long tail of luxury properties.
            """)

else:
    st.error("System Initialization Failed. Please check data files.")
