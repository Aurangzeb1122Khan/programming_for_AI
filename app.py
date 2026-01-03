import io
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import KNNImputer
from scipy import stats

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Restaurant Sales Analytics", layout="wide", page_icon="🍽️")

st.markdown("""
<style>
body {
  background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%) !important;
}

.block-container {
  padding-top: 2rem;
}

.glass-header {
  backdrop-filter: blur(12px) saturate(160%);
  -webkit-backdrop-filter: blur(12px) saturate(160%);
  background: rgba(255, 255, 255, 0.28);
  border-radius: 20px;
  border: 1px solid rgba(255, 255, 255, 0.3);
  padding: 1.5rem 2rem;
  margin-bottom: 1.5rem;
  box-shadow: 0 8px 20px rgba(0,0,0,0.1);
}

.stTabs [data-baseweb="tab-list"] {
  backdrop-filter: blur(10px) saturate(150%);
  background: rgba(255,255,255,0.2);
  border-radius: 16px;
  padding: 0.4rem 0.6rem;
}

.stTabs [data-baseweb="tab"] {
  backdrop-filter: blur(8px) saturate(160%);
  background: rgba(255, 255, 255, 0.35);
  border-radius: 14px;
  margin: 0 6px;
  padding: 0.6rem 1rem;
  transition: 0.25s ease;
  font-weight: 600;
}

.stTabs [data-baseweb="tab"]:hover {
  background: rgba(255, 255, 255, 0.55);
  transform: translateY(-2px);
}

h1, h2, h3, h4, h5, p, label, span {
  color: #2d2d2d !important;
  font-weight: 500 !important;
}

[data-testid="stSlider"] > div {
  backdrop-filter: blur(10px) saturate(180%);
  background: rgba(255, 255, 255, 0.45);
  padding: 1rem;
  border-radius: 16px;
}

.metric-box {
  backdrop-filter: blur(10px);
  background: rgba(255, 255, 255, 0.5);
  padding: 1rem;
  border-radius: 12px;
  text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="glass-header">
  <div style='display:flex; align-items:center; justify-content:space-between'>
    <div>
      <h1>🍽️ Restaurant Sales Analytics Studio</h1>
      <div style='color:#636e72; font-size:1.1rem;'>PhD-Level Data Analysis, Cleaning & Machine Learning</div>
    </div>
    <div style='text-align:right'>
      <div style='font-size:0.9rem; color:#2d3436;'></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_restaurant_data():
    
    # Create sample restaurant data (you can replace with actual CSV loading)
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Order ID': [f'ORD_{i:06d}' for i in range(n_samples)],
        'Customer ID': [f'CUST_{i%100:03d}' for i in range(n_samples)],
        'Category': np.random.choice(['Main Dishes', 'Starters', 'Desserts', 'Drinks', 'Side Dishes'], n_samples),
        'Item': np.random.choice(['Pasta Alfredo', 'Grilled Chicken', 'Ice Cream', 'Water', 'French Fries', 'Salad'], n_samples),
        'Price': np.random.uniform(3, 20, n_samples).round(2),
        'Quantity': np.random.randint(1, 6, n_samples),
        'Order Date': pd.date_range('2022-01-01', periods=n_samples, freq='8H').astype(str),
        'Payment Method': np.random.choice(['Credit Card', 'Cash', 'Digital Wallet'], n_samples)
    }
    
    df = pd.DataFrame(data)
    df['Order Total'] = df['Price'] * df['Quantity']
    
    # Add some missing values (like real dirty data)
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    df.loc[missing_indices[:len(missing_indices)//3], 'Price'] = np.nan
    df.loc[missing_indices[len(missing_indices)//3:2*len(missing_indices)//3], 'Quantity'] = np.nan
    df.loc[missing_indices[2*len(missing_indices)//3:], 'Payment Method'] = np.nan
    
    return df

def clean_data_phd(df):
    """PhD-level data cleaning"""
    df_clean = df.copy()
    
    # Remove duplicates
    df_clean = df_clean.drop_duplicates()
    
    # Identify column types
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    # KNN Imputation for numeric
    if len(numeric_cols) > 0:
        imputer = KNNImputer(n_neighbors=5)
        df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
    
    # Mode imputation for categorical
    for col in categorical_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    
    # IQR Outlier Capping
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
    
    # Text standardization
    for col in categorical_cols:
        df_clean[col] = df_clean[col].astype(str).str.lower().str.strip()
    
    return df_clean, numeric_cols, categorical_cols

def train_models(X, y):
    """Train ML models"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_cv = cross_val_score(lr, X_train_scaled, y_train, cv=5, scoring='r2')
    results['Linear Regression'] = {
        'model': lr,
        'rmse': np.sqrt(mean_squared_error(y_test, lr_pred)),
        'r2': r2_score(y_test, lr_pred),
        'cv_mean': lr_cv.mean()
    }
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results['Random Forest'] = {
        'model': rf,
        'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'r2': r2_score(y_test, rf_pred),
        'feature_importance': dict(zip(X.columns, rf.feature_importances_))
    }
    
    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    results['Gradient Boosting'] = {
        'model': gb,
        'rmse': np.sqrt(mean_squared_error(y_test, gb_pred)),
        'r2': r2_score(y_test, gb_pred)
    }
    
    return results, X_test, y_test, scaler

# Sidebar controls
st.sidebar.header("🎛️ Analysis Controls")

show_raw = st.sidebar.checkbox("Show Raw Data", value=False)
show_cleaned = st.sidebar.checkbox("Show Cleaned Data", value=True)

st.sidebar.markdown('---')
st.sidebar.write('📊 ML Model Settings')
target_var = st.sidebar.selectbox('Target Variable', ['Order Total', 'Price', 'Quantity'])
retrain = st.sidebar.button('🔄 Retrain Models')

st.sidebar.markdown('---')
st.sidebar.write('🎯 Quick Actions')
if st.sidebar.button('🧹 Clean Data'):
    st.session_state.clean_trigger = True
if st.sidebar.button('🤖 Train Models'):
    st.session_state.train_trigger = True

# Load data
df_raw = load_restaurant_data()

# Initialize session state
if 'cleaned_data' not in st.session_state or 'clean_trigger' in st.session_state:
    df_clean, numeric_cols, categorical_cols = clean_data_phd(df_raw)
    st.session_state.cleaned_data = df_clean
    st.session_state.numeric_cols = numeric_cols
    st.session_state.categorical_cols = categorical_cols
    if 'clean_trigger' in st.session_state:
        del st.session_state.clean_trigger
        st.sidebar.success("✅ Data Cleaned!")

df = st.session_state.cleaned_data
numeric_cols = st.session_state.numeric_cols
categorical_cols = st.session_state.categorical_cols

# Tabs
tabs = st.tabs(["📊 Overview", "🧹 Data Cleaning", "📈 EDA & Visualizations", "🤖 ML Training", "📥 Export"])

# TAB 1: Overview
with tabs[0]:
    st.markdown('<div class="glass-header">', unsafe_allow_html=True)
    st.subheader('📊 Dataset Overview')
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Orders", f"{len(df):,}")
    with col2:
        st.metric("Total Revenue", f"${df['Order Total'].sum():,.2f}")
    with col3:
        st.metric("Avg Order Value", f"${df['Order Total'].mean():.2f}")
    with col4:
        st.metric("Unique Customers", df['Customer ID'].nunique())
    with col5:
        st.metric("Product Categories", df['Category'].nunique())
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-header">', unsafe_allow_html=True)
    st.subheader('🔍 Data Preview')
    
    col1, col2 = st.columns(2)
    with col1:
        if show_raw:
            st.markdown("**Raw Data (First 10 rows)**")
            st.dataframe(df_raw.head(10), use_container_width=True)
    with col2:
        if show_cleaned:
            st.markdown("**Cleaned Data (First 10 rows)**")
            st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-header">', unsafe_allow_html=True)
    st.subheader('📋 Data Quality Metrics')
    
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.astype(str),
        'Missing (Raw)': df_raw.isnull().sum().values,
        'Missing (Cleaned)': df.isnull().sum().values,
        'Unique Values': [df[col].nunique() for col in df.columns]
    })
    st.dataframe(col_info, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 2: Data Cleaning
with tabs[1]:
    st.markdown('<div class="glass-header">', unsafe_allow_html=True)
    st.subheader('🧹 PhD-Level Data Cleaning Report')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Original Rows", len(df_raw))
        st.metric("Missing Values (Raw)", df_raw.isnull().sum().sum())
    with col2:
        st.metric("Cleaned Rows", len(df))
        st.metric("Missing Values (Cleaned)", df.isnull().sum().sum())
    with col3:
        st.metric("Duplicates Removed", len(df_raw) - len(df_raw.drop_duplicates()))
        st.metric("Outliers Handled", "✓")
    
    st.markdown("### 🔧 Cleaning Techniques Applied")
    st.markdown("""
    - ✅ **KNN Imputation (k=5)** for missing numeric values
    - ✅ **Mode Imputation** for missing categorical values
    - ✅ **IQR Method** for outlier detection and capping
    - ✅ **Text Standardization** (lowercase, trimming)
    - ✅ **Duplicate Removal**
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-header">', unsafe_allow_html=True)
    st.subheader('📊 Before vs After Comparison')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Raw Data Statistics**")
        st.dataframe(df_raw[numeric_cols].describe(), use_container_width=True)
    
    with col2:
        st.markdown("**Cleaned Data Statistics**")
        st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 3: EDA
with tabs[2]:
    st.markdown('<div class="glass-header">', unsafe_allow_html=True)
    st.subheader('📈 Statistical Analysis')
    
    # Advanced stats
    stats_data = {}
    for col in numeric_cols:
        stats_data[col] = {
            'Mean': df[col].mean(),
            'Median': df[col].median(),
            'Std Dev': df[col].std(),
            'Skewness': stats.skew(df[col]),
            'Kurtosis': stats.kurtosis(df[col])
        }
    
    stats_df = pd.DataFrame(stats_data).T
    st.dataframe(stats_df.round(3), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-header">', unsafe_allow_html=True)
    st.subheader('📊 Distribution Analysis')
    
    selected_col = st.selectbox("Select variable to visualize", numeric_cols)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df, x=selected_col, nbins=30, title=f"Distribution of {selected_col}")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(df, y=selected_col, title=f"Boxplot of {selected_col}")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-header">', unsafe_allow_html=True)
    st.subheader('🔗 Correlation Heatmap')
    
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto='.2f', aspect="auto", 
                       color_continuous_scale='RdBu_r',
                       title="Feature Correlations")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-header">', unsafe_allow_html=True)
    st.subheader('📊 Categorical Analysis')
    
    for col in categorical_cols[:3]:
        st.markdown(f"**{col} Distribution**")
        value_counts = df[col].value_counts().head(10)
        fig = px.bar(x=value_counts.index, y=value_counts.values, 
                    title=f"Top 10 {col}",
                    labels={'x': col, 'y': 'Count'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 4: ML Training
with tabs[3]:
    st.markdown('<div class="glass-header">', unsafe_allow_html=True)
    st.subheader('🤖 Machine Learning Training Center')
    
    if 'ml_results' not in st.session_state or 'train_trigger' in st.session_state or retrain:
        with st.spinner("Training models..."):
            feature_cols = [col for col in numeric_cols if col != target_var]
            if len(feature_cols) > 0:
                X = df[feature_cols]
                y = df[target_var]
                results, X_test, y_test, scaler = train_models(X, y)
                st.session_state.ml_results = results
                st.session_state.ml_target = target_var
                st.success("✅ Models Trained!")
            if 'train_trigger' in st.session_state:
                del st.session_state.train_trigger
    
    if 'ml_results' in st.session_state:
        results = st.session_state.ml_results
        
        st.markdown(f"**Target Variable:** {st.session_state.ml_target}")
        
        comparison = pd.DataFrame({
            'Model': list(results.keys()),
            'RMSE': [results[m]['rmse'] for m in results],
            'R² Score': [results[m]['r2'] for m in results]
        }).sort_values('R² Score', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(comparison.round(4), use_container_width=True)
            best_model = comparison.iloc[0]['Model']
            st.success(f"🏆 Best Model: **{best_model}** (R² = {comparison.iloc[0]['R² Score']:.4f})")
        
        with col2:
            fig = px.bar(comparison, x='Model', y='R² Score',
                        title="Model Performance Comparison",
                        color='R² Score',
                        color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        if 'Random Forest' in results and 'feature_importance' in results['Random Forest']:
            st.markdown("### 📊 Feature Importance")
            importance_df = pd.DataFrame(
                list(results['Random Forest']['feature_importance'].items()),
                columns=['Feature', 'Importance']
            ).sort_values('Importance', ascending=False)
            
            fig = px.bar(importance_df, x='Importance', y='Feature',
                        orientation='h',
                        title="Feature Importance Rankings")
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 5: Export
with tabs[4]:
    st.markdown('<div class="glass-header">', unsafe_allow_html=True)
    st.subheader('💾 Export Cleaned Data & Models')
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Cleaned CSV",
            data=csv,
            file_name="restaurant_sales_cleaned.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        if 'ml_results' in st.session_state:
            best_model = st.session_state.ml_results['Random Forest']['model']
            model_bytes = pickle.dumps(best_model)
            st.download_button(
                label="🤖 Download Best Model",
                data=model_bytes,
                file_name="restaurant_model.pkl",
                mime="application/octet-stream",
                use_container_width=True
            )
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div style='margin-top:20px; padding:15px; background: rgba(255,255,255,0.6); border-radius:12px; text-align:center'>
<strong>Made by aurangzeb</strong> | PhD-Level Data Science Platform
</div>
""", unsafe_allow_html=True)
