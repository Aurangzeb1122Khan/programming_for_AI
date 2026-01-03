import io
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy import stats

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

[data-testid="stMetricValue"] {
  font-size: 2rem !important;
  font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="glass-header">
  <div style='display:flex; align-items:center; justify-content:space-between'>
    <div>
      <h1>🍽️ Restaurant Sales Analytics Studio</h1>
      <div style='color:#636e72; font-size:1.1rem;'>Clustering Analysis & Sales Insights Dashboard</div>
    </div>
    <div style='text-align:right'>
      <div style='font-size:0.9rem; color:#2d3436;'></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_restaurant_data():
    """Load restaurant sales data"""
    np.random.seed(42)
    n_samples = 1000
    
    categories = ['Main Dishes', 'Starters', 'Desserts', 'Drinks', 'Side Dishes']
    items_by_category = {
        'Main Dishes': ['Pasta Alfredo', 'Grilled Chicken', 'Steak', 'Fish'],
        'Starters': ['Caesar Salad', 'Soup', 'Bruschetta', 'Wings'],
        'Desserts': ['Ice Cream', 'Chocolate Cake', 'Tiramisu', 'Cheesecake'],
        'Drinks': ['Water', 'Coca Cola', 'Orange Juice', 'Coffee'],
        'Side Dishes': ['French Fries', 'Mashed Potatoes', 'Coleslaw', 'Rice']
    }
    
    data = {
        'Order ID': [f'ORD_{i:06d}' for i in range(n_samples)],
        'Customer ID': [f'CUST_{i%100:03d}' for i in range(n_samples)],
        'Category': np.random.choice(categories, n_samples),
    }
    
    # Assign items based on category
    items = []
    for cat in data['Category']:
        items.append(np.random.choice(items_by_category[cat]))
    data['Item'] = items
    
    # Price based on category
    prices = []
    for cat in data['Category']:
        if cat == 'Main Dishes':
            prices.append(np.random.uniform(12, 25))
        elif cat == 'Starters':
            prices.append(np.random.uniform(5, 12))
        elif cat == 'Desserts':
            prices.append(np.random.uniform(4, 10))
        elif cat == 'Drinks':
            prices.append(np.random.uniform(2, 8))
        else:
            prices.append(np.random.uniform(3, 8))
    
    data['Price'] = np.round(prices, 2)
    data['Quantity'] = np.random.randint(1, 6, n_samples)
    data['Order Date'] = pd.date_range('2022-01-01', periods=n_samples, freq='8H').astype(str)
    data['Payment Method'] = np.random.choice(['Credit Card', 'Cash', 'Digital Wallet'], n_samples)
    
    df = pd.DataFrame(data)
    df['Order Total'] = df['Price'] * df['Quantity']
    
    return df

def prepare_clustering_data(df):
    """Prepare data for clustering"""
    # Encode categorical variables
    le_category = LabelEncoder()
    le_item = LabelEncoder()
    le_payment = LabelEncoder()
    
    df_cluster = df.copy()
    df_cluster['Category_Encoded'] = le_category.fit_transform(df['Category'])
    df_cluster['Item_Encoded'] = le_item.fit_transform(df['Item'])
    df_cluster['Payment_Encoded'] = le_payment.fit_transform(df['Payment Method'])
    
    # Features for clustering
    features = ['Price', 'Quantity', 'Order Total', 'Category_Encoded', 'Item_Encoded', 'Payment_Encoded']
    X = df_cluster[features]
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, features, df_cluster

def perform_clustering(X, n_clusters=3, method='kmeans'):
    """Perform clustering"""
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    elif method == 'hierarchical':
        model = AgglomerativeClustering(n_clusters=n_clusters)
    else:
        model = DBSCAN(eps=0.5, min_samples=5)
    
    clusters = model.fit_predict(X)
    
    # Calculate metrics
    if len(np.unique(clusters)) > 1:
        silhouette = silhouette_score(X, clusters)
        davies_bouldin = davies_bouldin_score(X, clusters)
    else:
        silhouette = 0
        davies_bouldin = 0
    
    return clusters, model, silhouette, davies_bouldin

# Load data
df = load_restaurant_data()

# Sidebar - Control Panel
st.sidebar.header("🎛️ Control Panel")

st.sidebar.markdown("### 📊 Clustering Settings")
cluster_method = st.sidebar.selectbox(
    "Clustering Algorithm",
    ['kmeans', 'hierarchical', 'dbscan'],
    format_func=lambda x: {'kmeans': 'K-Means', 'hierarchical': 'Hierarchical', 'dbscan': 'DBSCAN'}[x]
)

if cluster_method != 'dbscan':
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 8, 3)
else:
    n_clusters = None

run_clustering = st.sidebar.button("🚀 Run Clustering Analysis", type="primary")

st.sidebar.markdown("---")

st.sidebar.markdown("### 🔍 Data Filters")
selected_category = st.sidebar.multiselect(
    "Filter by Category",
    options=df['Category'].unique(),
    default=df['Category'].unique()
)

selected_payment = st.sidebar.multiselect(
    "Filter by Payment Method",
    options=df['Payment Method'].unique(),
    default=df['Payment Method'].unique()
)

price_range = st.sidebar.slider(
    "Price Range ($)",
    float(df['Price'].min()),
    float(df['Price'].max()),
    (float(df['Price'].min()), float(df['Price'].max()))
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Statistics")
st.sidebar.metric("Total Orders", f"{len(df):,}")
st.sidebar.metric("Total Revenue", f"${df['Order Total'].sum():,.2f}")
st.sidebar.metric("Unique Items", df['Item'].nunique())

# Apply filters
df_filtered = df[
    (df['Category'].isin(selected_category)) &
    (df['Payment Method'].isin(selected_payment)) &
    (df['Price'] >= price_range[0]) &
    (df['Price'] <= price_range[1])
]

# Main tabs
tabs = st.tabs([
    "🏠 Dashboard",
    "📊 Sales Analysis",
    "🎯 Clustering Analysis",
    "📈 Advanced Visualizations",
    "💾 Export"
])

# TAB 1: Dashboard
with tabs[0]:
    st.markdown('<div class="glass-header">', unsafe_allow_html=True)
    st.subheader("📊 Sales Overview Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Orders", f"{len(df_filtered):,}")
    with col2:
        st.metric("Total Revenue", f"${df_filtered['Order Total'].sum():,.2f}")
    with col3:
        st.metric("Avg Order Value", f"${df_filtered['Order Total'].mean():.2f}")
    with col4:
        st.metric("Unique Customers", df_filtered['Customer ID'].nunique())
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-header">', unsafe_allow_html=True)
    st.subheader("📊 Sales by Category")
    
    col1, col2 = st.columns(2)
    
    with col1:
        category_sales = df_filtered.groupby('Category')['Order Total'].sum().sort_values(ascending=False)
        fig = px.pie(values=category_sales.values, names=category_sales.index,
                    title="Revenue Distribution by Category",
                    hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        category_count = df_filtered['Category'].value_counts()
        fig = px.bar(x=category_count.index, y=category_count.values,
                    title="Order Count by Category",
                    labels={'x': 'Category', 'y': 'Orders'},
                    color=category_count.values,
                    color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-header">', unsafe_allow_html=True)
    st.subheader("🔝 Top Selling Items")
    
    col1, col2 = st.columns(2)
    
    with col1:
        top_items = df_filtered.groupby('Item')['Order Total'].sum().sort_values(ascending=False).head(10)
        fig = px.bar(x=top_items.index, y=top_items.values,
                    title="Top 10 Items by Revenue",
                    labels={'x': 'Item', 'y': 'Revenue ($)'},
                    color=top_items.values,
                    color_continuous_scale='plasma')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        top_items_count = df_filtered['Item'].value_counts().head(10)
        fig = px.bar(x=top_items_count.index, y=top_items_count.values,
                    title="Top 10 Items by Order Count",
                    labels={'x': 'Item', 'y': 'Orders'},
                    color=top_items_count.values,
                    color_continuous_scale='sunset')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 2: Sales Analysis
with tabs[1]:
    st.markdown('<div class="glass-header">', unsafe_allow_html=True)
    st.subheader("💰 Payment Method Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        payment_sales = df_filtered.groupby('Payment Method')['Order Total'].sum()
        fig = px.pie(values=payment_sales.values, names=payment_sales.index,
                    title="Revenue by Payment Method")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        payment_count = df_filtered['Payment Method'].value_counts()
        fig = px.bar(x=payment_count.index, y=payment_count.values,
                    title="Orders by Payment Method",
                    color=payment_count.values,
                    color_continuous_scale='blues')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-header">', unsafe_allow_html=True)
    st.subheader("📈 Price Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df_filtered, x='Price', nbins=30,
                          title="Price Distribution",
                          labels={'Price': 'Price ($)'},
                          color_discrete_sequence=['#fab1a0'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(df_filtered, x='Category', y='Price',
                    title="Price Range by Category",
                    color='Category')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-header">', unsafe_allow_html=True)
    st.subheader("📊 Category-wise Item Performance")
    
    for category in df_filtered['Category'].unique():
        st.markdown(f"### {category}")
        cat_data = df_filtered[df_filtered['Category'] == category]
        item_sales = cat_data.groupby('Item')['Order Total'].agg(['sum', 'count', 'mean'])
        item_sales.columns = ['Total Revenue', 'Order Count', 'Avg Order Value']
        item_sales = item_sales.sort_values('Total Revenue', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(item_sales, x=item_sales.index, y='Total Revenue',
                        title=f"{category} - Revenue by Item",
                        color='Total Revenue',
                        color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(item_sales.round(2), use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 3: Clustering
with tabs[2]:
    st.markdown('<div class="glass-header">', unsafe_allow_html=True)
    st.subheader("🎯 Customer Segmentation - Clustering Analysis")
    
    if run_clustering or 'clusters' not in st.session_state:
        with st.spinner("Running clustering analysis..."):
            X_scaled, features, df_cluster = prepare_clustering_data(df_filtered)
            clusters, model, silhouette, davies_bouldin = perform_clustering(
                X_scaled, n_clusters, cluster_method
            )
            
            st.session_state.clusters = clusters
            st.session_state.silhouette = silhouette
            st.session_state.davies_bouldin = davies_bouldin
            st.session_state.X_scaled = X_scaled
            st.session_state.df_cluster = df_cluster
            
            st.success("✅ Clustering Complete!")
    
    if 'clusters' in st.session_state:
        clusters = st.session_state.clusters
        df_cluster = st.session_state.df_cluster
        df_cluster['Cluster'] = clusters
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Clusters", len(np.unique(clusters)))
        with col2:
            st.metric("Silhouette Score", f"{st.session_state.silhouette:.3f}")
        with col3:
            st.metric("Davies-Bouldin Score", f"{st.session_state.davies_bouldin:.3f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="glass-header">', unsafe_allow_html=True)
        st.subheader("📊 Cluster Visualization (PCA)")
        
        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(st.session_state.X_scaled)
        
        fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=clusters.astype(str),
                        title="Customer Segments (PCA Projection)",
                        labels={'x': 'PC1', 'y': 'PC2', 'color': 'Cluster'},
                        color_discrete_sequence=px.colors.qualitative.Set1)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="glass-header">', unsafe_allow_html=True)
        st.subheader("📈 Cluster Characteristics")
        
        cluster_summary = df_cluster.groupby('Cluster').agg({
            'Price': ['mean', 'min', 'max'],
            'Quantity': ['mean', 'sum'],
            'Order Total': ['mean', 'sum', 'count']
        }).round(2)
        
        st.dataframe(cluster_summary, use_container_width=True)
        
        st.markdown("### 📊 Cluster Distribution by Category")
        cluster_category = pd.crosstab(df_cluster['Cluster'], df_cluster['Category'])
        fig = px.imshow(cluster_category, text_auto=True,
                       title="Cluster vs Category Heatmap",
                       color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# TAB 4: Advanced Visualizations
with tabs[3]:
    st.markdown('<div class="glass-header">', unsafe_allow_html=True)
    st.subheader("🔥 Advanced Sales Insights")
    
    # Scatter plot
    fig = px.scatter(df_filtered, x='Price', y='Quantity', color='Category',
                    size='Order Total', hover_data=['Item'],
                    title="Price vs Quantity (sized by Order Total)")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-header">', unsafe_allow_html=True)
    st.subheader("📊 Correlation Heatmap")
    
    numeric_cols = ['Price', 'Quantity', 'Order Total']
    corr = df_filtered[numeric_cols].corr()
    fig = px.imshow(corr, text_auto='.2f',
                   title="Feature Correlations",
                   color_continuous_scale='RdBu_r')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-header">', unsafe_allow_html=True)
    st.subheader("📈 Sales Trends Over Time")
    
    df_time = df_filtered.copy()
    df_time['Date'] = pd.to_datetime(df_time['Order Date'])
    df_time['Month'] = df_time['Date'].dt.to_period('M').astype(str)
    
    monthly_sales = df_time.groupby('Month')['Order Total'].sum().reset_index()
    fig = px.line(monthly_sales, x='Month', y='Order Total',
                 title="Monthly Revenue Trend",
                 markers=True)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# TAB 5: Export
with tabs[4]:
    st.markdown('<div class="glass-header">', unsafe_allow_html=True)
    st.subheader("💾 Export Data & Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv,
            file_name="restaurant_sales_filtered.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        if 'clusters' in st.session_state:
            df_with_clusters = st.session_state.df_cluster
            csv_clusters = df_with_clusters.to_csv(index=False)
            st.download_button(
                label="🎯 Download with Clusters (CSV)",
                data=csv_clusters,
                file_name="restaurant_sales_clustered.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div style='margin-top:20px; padding:15px; background: rgba(255,255,255,0.6); border-radius:12px; text-align:center'>
<strong>Made by ENIGJES</strong> | Restaurant Sales Analytics & Clustering Platform
</div>
""", unsafe_allow_html=True)
