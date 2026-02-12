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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional UI
st.markdown("""
<style>
    /* Global Styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #1e293b;
    }
    
    /* Card Component */
    .metric-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 24px;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
        height: 100%;
        transition: all 0.2s ease-in-out;
    }
    
    .metric-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.5px;
        margin-top: 8px;
    }
    
    .metric-label {
        font-size: 13px;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Header Styling */
    .main-header {
        background: #0f172a;
        padding: 3rem 2rem;
        border-radius: 0;
        color: white;
        margin: -4rem -4rem 2rem -4rem;
        border-bottom: 4px solid #3b82f6;
    }
    
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -1px;
    }
    
    .header-subtitle {
        font-size: 1rem;
        color: #94a3b8;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Button Styling */
    .stButton > button {
        background-color: #0f172a;
        color: white;
        border-radius: 6px;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        border: 1px solid #0f172a;
        transition: all 0.2s;
        width: 100%;
        text-transform: uppercase;
        font-size: 14px;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        background-color: #1e293b;
        border-color: #1e293b;
        transform: translateY(-1px);
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding-top: 1rem;
        padding-bottom: 1rem;
        font-weight: 600;
        color: #64748b;
        font-size: 15px;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #3b82f6;
        border-bottom-color: #3b82f6;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
        background-color: #dbeafe;
        color: #1d4ed8;
    }
    
    /* Table Styling */
    .dataframe {
        font-family: 'Inter', sans-serif !important;
        border: 1px solid #e2e8f0 !important;
    }
    
    .dataframe th {
        background-color: #f8fafc !important;
        font-weight: 600 !important;
        color: #475569 !important;
        padding: 12px !important;
    }
    
    .dataframe td {
        padding: 12px !important;
        color: #334155 !important;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("""
<div class="main-header">
    <div style="max-width: 1200px; margin: 0 auto;">
        <div class="header-title">HousePrice AI</div>
        <div class="header-subtitle">Advanced Machine Learning Valuation System for Ames Housing Market</div>
    </div>
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
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        train = pd.read_csv(os.path.join(base, 'data', 'train.csv'))
        return train
    except:
        return None

# Load resources
predictor = load_predictor()
train_data = load_data()

if predictor is not None and train_data is not None:
    # Sidebar
    with st.sidebar:
        st.markdown("### Control Panel")
        st.markdown("Select a sample property from the dataset to generate a real-time valuation prediction.")
        
        st.markdown("---")
        st.markdown("### System Status")
        st.markdown("""
        <div style="background-color: white; padding: 15px; border-radius: 8px; border: 1px solid #e2e8f0;">
            <div style="margin-bottom: 8px; font-size: 13px; color: #64748b;">MODEL STATUS</div>
            <div style="color: #10b981; font-weight: 600; display: flex; align-items: center; gap: 6px;">
                <div style="width: 8px; height: 8px; background-color: #10b981; border-radius: 50%;"></div>
                Active (Stacking Ensemble)
            </div>
            <div style="margin-top: 12px; margin-bottom: 8px; font-size: 13px; color: #64748b;">DATABASE</div>
            <div style="color: #3b82f6; font-weight: 600; display: flex; align-items: center; gap: 6px;">
                <div style="width: 8px; height: 8px; background-color: #3b82f6; border-radius: 50%;"></div>
                Connected (1,460 Records)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.caption("© 2026 HousePrice AI")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Price Prediction", "Model Analytics", "Market Insights"])

    with tab1:
        st.markdown("### Property Valuation")
        
        # Selection
        sample_indices = predictor.get_sample_indices(20)
        
        col_sel, col_btn = st.columns([3, 1])
        with col_sel:
            selected_idx = st.selectbox(
                "Select Property Sample",
                options=sample_indices,
                format_func=lambda x: f"Property ID #{x} - {predictor.get_house_summary(x).get('Neighborhood', 'Unknown')}",
                label_visibility="collapsed"
            )
        
        # Get house details
        house = predictor.get_house_summary(selected_idx)
        
        # Display Property Cards
        st.markdown("<br>", unsafe_allow_html=True)
        
        def metric_card(label, value, col):
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

        row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)
        metric_card("Living Area", f"{house['Living Area (sq ft)']} sqft", row1_col1)
        metric_card("Overall Quality", f"{house['Overall Quality']}/10", row1_col2)
        metric_card("Year Built", house['Year Built'], row1_col3)
        metric_card("Neighborhood", house['Neighborhood'], row1_col4)
        
        row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)
        metric_card("Bedrooms", house['Bedrooms'], row2_col1)
        metric_card("Bathrooms", house['Bathrooms'], row2_col2)
        metric_card("Total Rooms", house['Total Rooms'], row2_col3)
        metric_card("Garage Capacity", f"{house['Garage Cars']} Cars", row2_col4)

        # Predict Action
        st.markdown("---")
        with col_btn:
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
                        <div class="metric-card" style="border-top: 4px solid #3b82f6;">
                            <div class="metric-label">AI Valuation</div>
                            <div class="metric-value" style="color: #3b82f6;">${predicted_price:,.0f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with res_col2:
                        st.markdown(f"""
                        <div class="metric-card" style="border-top: 4px solid #10b981;">
                            <div class="metric-label">Actual Sale Price</div>
                            <div class="metric-value" style="color: #10b981;">{house['Actual Price']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with res_col3:
                        color = "#ef4444" if error_pct > 10 else "#64748b"
                        st.markdown(f"""
                        <div class="metric-card" style="border-top: 4px solid {color};">
                            <div class="metric-label">Prediction Error</div>
                            <div class="metric-value" style="color: {color};">{error_pct:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Visualization
                    st.markdown("### Model Confidence Analysis")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    
                    models = list(individual_preds.keys()) + ['Ensemble Final']
                    values = list(individual_preds.values()) + [predicted_price]
                    colors = ['#e2e8f0'] * len(individual_preds) + ['#3b82f6']
                    
                    y_pos = np.arange(len(models))
                    ax.barh(y_pos, values, align='center', color=colors, height=0.6)
                    ax.axvline(actual_price_val, color='#ef4444', linestyle='--', label='Actual Price', linewidth=2)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(models)
                    ax.set_xlabel('Valuation ($)')
                    ax.legend(frameon=False)
                    ax.grid(axis='x', alpha=0.2, linestyle='--')
                    
                    # Clean styling
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.spines['bottom'].set_color('#cbd5e1')
                    
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
        
        bars = ax.bar(models, scores, color='#cbd5e1', width=0.6)
        ax.set_ylabel('RMSE (Lower is Better)')
        ax.set_ylim(0.10, 0.12)
        ax.grid(axis='y', alpha=0.2, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Highlight best model
        best_idx = np.argmin(scores)
        bars[best_idx].set_color('#3b82f6')
        
        st.pyplot(fig)

    with tab3:
        st.markdown("### Market Intelligence")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Price Distribution")
            fig, ax = plt.subplots()
            ax.hist(train_data['SalePrice'], bins=50, color='#3b82f6', alpha=0.8, edgecolor='white')
            ax.set_xlabel('Price ($)')
            ax.set_ylabel('Frequency')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='y', alpha=0.2, linestyle='--')
            st.pyplot(fig)
            
        with col2:
            st.markdown("#### Market Statistics")
            st.markdown(f"""
            - **Total Properties:** {len(train_data):,}
            - **Average Price:** ${train_data['SalePrice'].mean():,.0f}
            - **Median Price:** ${train_data['SalePrice'].median():,.0f}
            - **Price Range:** ${train_data['SalePrice'].min():,.0f} - ${train_data['SalePrice'].max():,.0f}
            """)
            
            st.info("""
            **Market Insight:** The market shows a right-skewed distribution, indicating most homes are affordable 
            with a long tail of luxury properties affecting the average.
            """)

else:
    st.error("System Initialization Failed. Please check data files.")
