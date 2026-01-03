import io
import base64
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Restaurant Sales Data Cleaning Studio", layout="wide", page_icon="🍽️")

st.markdown("""
<style>
body {
  background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%) !important;
}

.block-container {
  padding-top: 2rem;
}

.glass-card {
  backdrop-filter: blur(12px) saturate(160%);
  -webkit-backdrop-filter: blur(12px) saturate(160%);
  background: rgba(255, 255, 255, 0.35);
  border-radius: 20px;
  border: 1px solid rgba(255, 255, 255, 0.3);
  padding: 1.5rem 2rem;
  margin-bottom: 1.5rem;
  box-shadow: 0 8px 20px rgba(0,0,0,0.1);
}

.stTabs [data-baseweb="tab-list"] {
  backdrop-filter: blur(10px) saturate(150%);
  background: rgba(255,255,255,0.3);
  border-radius: 16px;
  padding: 0.4rem 0.6rem;
}

.stTabs [data-baseweb="tab"] {
  backdrop-filter: blur(8px) saturate(160%);
  background: rgba(255, 255, 255, 0.4);
  border-radius: 14px;
  margin: 0 6px;
  padding: 0.6rem 1rem;
  transition: 0.25s ease;
  font-weight: 600;
}

.stTabs [data-baseweb="tab"]:hover {
  background: rgba(255, 255, 255, 0.6);
  transform: translateY(-2px);
}

h1, h2, h3, h4, h5, p, label, span {
  color: #2d3436 !important;
  font-weight: 500 !important;
}

[data-testid="stMetricValue"] {
  font-size: 2rem;
  font-weight: 700;
}

.stat-box {
  backdrop-filter: blur(10px);
  background: rgba(255, 255, 255, 0.5);
  padding: 1rem;
  border-radius: 12px;
  text-align: center;
  margin: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="glass-card">
  <div style='display:flex; align-items:center; justify-content:space-between'>
    <div>
      <h1>🍽️ Restaurant Sales Data Cleaning Studio</h1>
      <div style='color:#636e72; font-size:1.1rem;'>PhD-Level Data Cleaning, EDA & Machine Learning Pipeline</div>
    </div>
    <div style='text-align:right'>
      <div style='font-size:0.9rem; color:#2d3436;'>Made by Aurangzeb </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Session state initialization
if 'data' not in st.session_state:
    st.session_state.data = None
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

def clean_data(df):
    """PhD-level data cleaning"""
    stats = {
        'original_rows': len(df),
        'duplicates_removed': 0,
        'missing_filled': {},
        'outliers_capped': 0
    }
    
    # Remove duplicates
    df_clean = df.drop_duplicates()
    stats['duplicates_removed'] = len(df) - len(df_clean)
    
    # Identify column types
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    # Fill missing values - Numeric (median)
    for col in numeric_cols:
        missing = df_clean[col].isnull().sum()
        if missing > 0:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            stats['missing_filled'][col] = f"{missing} values filled with median ({median_val:.2f})"
    
    # Fill missing values - Categorical (mode)
    for col in categorical_cols:
        missing = df_clean[col].isnull().sum()
        if missing > 0:
            mode_val = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown'
            df_clean[col].fillna(mode_val, inplace=True)
            stats['missing_filled'][col] = f"{missing} values filled with mode ('{mode_val}')"
    
    # Handle outliers (IQR method)
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        outliers = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
        if outliers > 0:
            df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
            stats['outliers_capped'] += outliers
    
    stats['final_rows'] = len(df_clean)
    
    return df_clean, stats, numeric_cols, categorical_cols

def train_ml_model(df, target_col, feature_cols):
    """Train multiple ML models"""
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    results['Linear Regression'] = {
        'model': lr,
        'rmse': np.sqrt(mean_squared_error(y_test, lr_pred)),
        'r2': r2_score(y_test, lr_pred),
        'predictions': lr_pred
    }
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results['Random Forest'] = {
        'model': rf,
        'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'r2': r2_score(y_test, rf_pred),
        'predictions': rf_pred,
        'feature_importance': dict(zip(feature_cols, rf.feature_importances_))
    }
    
    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    results['Gradient Boosting'] = {
        'model': gb,
        'rmse': np.sqrt(mean_squared_error(y_test, gb_pred)),
        'r2': r2_score(y_test, gb_pred),
        'predictions': gb_pred
    }
    
    return results, X_test, y_test, scaler

# Sidebar - File Upload
st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload Restaurant Sales CSV", type=['csv'])

if uploaded_file is not None:
    st.session_state.data = pd.read_csv(uploaded_file)
    st.sidebar.success(f"✅ Loaded: {len(st.session_state.data)} rows")

# Main Tabs
if st.session_state.data is not None:
    tabs = st.tabs(["📊 Data Overview", "🧹 Clean Data", "📈 EDA & Insights", "🤖 ML Training", "💾 Export"])
    
    # Tab 1: Data Overview
    with tabs[0]:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Raw Data Preview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", len(st.session_state.data))
        with col2:
            st.metric("Total Columns", len(st.session_state.data.columns))
        with col3:
            st.metric("Missing Values", st.session_state.data.isnull().sum().sum())
        with col4:
            st.metric("Duplicates", st.session_state.data.duplicated().sum())
        
        st.dataframe(st.session_state.data.head(10), use_container_width=True)
        
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column': st.session_state.data.columns,
            'Data Type': st.session_state.data.dtypes.values,
            'Missing': st.session_state.data.isnull().sum().values,
            'Unique Values': [st.session_state.data[col].nunique() for col in st.session_state.data.columns]
        })
        st.dataframe(col_info, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 2: Clean Data
    with tabs[1]:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🧹 Data Cleaning")
        
        if st.button("🚀 Clean Data Now!", type="primary"):
            with st.spinner("Cleaning data using PhD-level techniques..."):
                cleaned, stats, num_cols, cat_cols = clean_data(st.session_state.data)
                st.session_state.cleaned_data = cleaned
                st.session_state.numeric_cols = num_cols
                st.session_state.categorical_cols = cat_cols
                st.session_state.cleaning_stats = stats
                st.success("✅ Data Cleaned Successfully!")
        
        if st.session_state.cleaned_data is not None:
            stats = st.session_state.cleaning_stats
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Original Rows", stats['original_rows'])
            with col2:
                st.metric("Duplicates Removed", stats['duplicates_removed'])
            with col3:
                st.metric("Outliers Capped", stats['outliers_capped'])
            with col4:
                st.metric("Final Rows", stats['final_rows'])
            
            st.subheader("Cleaning Actions Performed:")
            for col, action in stats['missing_filled'].items():
                st.write(f"✅ **{col}**: {action}")
            
            st.subheader("Cleaned Data Preview")
            st.dataframe(st.session_state.cleaned_data.head(10), use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Tab 3: EDA
    with tabs[2]:
        if st.session_state.cleaned_data is not None:
            df = st.session_state.cleaned_data
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("📈 Statistical Summary")
            st.dataframe(df.describe(), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("📊 Distribution Plots")
            
            numeric_cols = st.session_state.numeric_cols
            if len(numeric_cols) > 0:
                selected_col = st.selectbox("Select column to visualize", numeric_cols)
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.box(df, y=selected_col, title=f"Boxplot of {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("🔗 Correlation Heatmap")
            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()
                fig = px.imshow(corr, text_auto=True, aspect="auto", 
                               color_continuous_scale='RdBu_r',
                               title="Feature Correlations")
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please clean the data first in the 'Clean Data' tab!")
    
    # Tab 4: ML Training
    with tabs[3]:
        if st.session_state.cleaned_data is not None:
            df = st.session_state.cleaned_data
            numeric_cols = st.session_state.numeric_cols
            
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.subheader("🤖 Machine Learning Model Training")
            
            if len(numeric_cols) > 1:
                target = st.selectbox("Select Target Variable (to predict)", numeric_cols)
                feature_cols = [col for col in numeric_cols if col != target]
                
                if st.button("🚀 Train Models", type="primary"):
                    with st.spinner("Training multiple ML models..."):
                        results, X_test, y_test, scaler = train_ml_model(df, target, feature_cols)
                        st.session_state.ml_results = results
                        st.session_state.ml_target = target
                        st.success("✅ Models Trained Successfully!")
                
                if 'ml_results' in st.session_state:
                    results = st.session_state.ml_results
                    
                    st.subheader("📊 Model Performance Comparison")
                    
                    comparison = pd.DataFrame({
                        'Model': list(results.keys()),
                        'RMSE': [results[m]['rmse'] for m in results],
                        'R² Score': [results[m]['r2'] for m in results]
                    }).sort_values('R² Score', ascending=False)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(comparison, use_container_width=True)
                    
                    with col2:
                        fig = px.bar(comparison, x='Model', y='R² Score', 
                                    title="Model R² Scores",
                                    color='R² Score',
                                    color_continuous_scale='viridis')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    best_model = comparison.iloc[0]['Model']
                    st.success(f"🏆 Best Model: **{best_model}** (R² = {comparison.iloc[0]['R² Score']:.4f})")
                    
                    # Feature Importance
                    if 'Random Forest' in results and 'feature_importance' in results['Random Forest']:
                        st.subheader("📊 Feature Importance (Random Forest)")
                        importance_df = pd.DataFrame(
                            list(results['Random Forest']['feature_importance'].items()),
                            columns=['Feature', 'Importance']
                        ).sort_values('Importance', ascending=False)
                        
                        fig = px.bar(importance_df, x='Feature', y='Importance',
                                    title="Feature Importance Rankings")
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Need at least 2 numeric columns for ML training!")
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please clean the data first!")
    
    # Tab 5: Export
    with tabs[4]:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("💾 Export Cleaned Data")
        
        if st.session_state.cleaned_data is not None:
            csv = st.session_state.cleaned_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Cleaned CSV",
                data=csv,
                file_name="restaurant_sales_cleaned.csv",
                mime="text/csv",
                type="primary"
            )
            
            st.subheader("📊 Export Statistics Report")
            if 'cleaning_stats' in st.session_state:
                stats = st.session_state.cleaning_stats
                report = f"""
                # Restaurant Sales Data Cleaning Report
                
                ## Summary Statistics
                - Original Rows: {stats['original_rows']}
                - Final Rows: {stats['final_rows']}
                - Duplicates Removed: {stats['duplicates_removed']}
                - Outliers Capped: {stats['outliers_capped']}
                
                ## Missing Values Handled
                """
                for col, action in stats['missing_filled'].items():
                    report += f"\n- {col}: {action}"
                
                st.download_button(
                    label="📄 Download Report (TXT)",
                    data=report,
                    file_name="cleaning_report.txt",
                    mime="text/plain"
                )
        else:
            st.warning("⚠️ No cleaned data available. Please clean data first!")
        
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.info("👈 Please upload a Restaurant Sales CSV file from the sidebar to begin!")
    
st.markdown("""
<div style='margin-top:20px; padding:15px; background: rgba(255,255,255,0.5); border-radius:12px; text-align:center'>
<strong>🍽️ Restaurant Sales Data Cleaning Studio</strong><br>
Made by Aurangzeb 
</div>
""", unsafe_allow_html=True)
