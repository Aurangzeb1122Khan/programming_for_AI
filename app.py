import io
import base64
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Iris SVM — Next Level UI", layout="wide", page_icon="🌸")

st.markdown("""
<style>
body {
  background: linear-gradient(135deg, #e0f7fa 0%, #fbe9e7 100%) !important;
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

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card petals">
  <div style='display:flex; align-items:center; justify-content:space-between'>
    <div>
      <h1 class='display'>Iris SVM Studio 🌸</h1>
      <div class='small-muted'>An interactive classifier — train, explore, predict, and export.</div>
    </div>
    <div style='text-align:right'>
      <div style='font-size:0.9rem; color:#334;'>Made for Sir Zeeshan, with love from Enigjes 💕💦</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_iris_df():
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df, iris

def train_model(X, y, C=1.0, kernel='rbf', gamma='scale', probability=True):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = SVC(C=C, kernel=kernel, gamma=gamma, probability=probability)
    model.fit(Xs, y)
    return model, scaler

def predict_with_model(model, scaler, X):
    Xs = scaler.transform(X)
    probs = model.predict_proba(Xs)
    pred = model.predict(Xs)
    return pred, probs

def model_to_bytes(model, scaler):
    bio = io.BytesIO()
    pickle.dump({'model': model, 'scaler': scaler}, bio)
    bio.seek(0)
    return bio.read()

st.sidebar.header("Model & Controls")
C = st.sidebar.number_input('C (regularization)', min_value=0.01, max_value=100.0, value=1.0, step=0.01, format="%.2f")
kernel = st.sidebar.selectbox('Kernel', options=['rbf', 'linear', 'poly', 'sigmoid'], index=0)
gamma = st.sidebar.selectbox('Gamma', options=['scale', 'auto'], index=0)
retrain = st.sidebar.button('Retrain model with these hyperparameters')

st.sidebar.markdown('---')
st.sidebar.write('Presets')
if st.sidebar.button('Classic SVM (C=1, rbf)'):
    C, kernel, gamma = 1.0, 'rbf', 'scale'
if st.sidebar.button('Linear (C=0.5, linear)'):
    C, kernel, gamma = 0.5, 'linear', 'scale'

df, iris = load_iris_df()

tabs = st.tabs(["Predict","Visualize","Train & Metrics","Dataset & Export"])

with tabs[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader('Live Predictor — interact to predict species')
    col1, col2 = st.columns([1,2])
    with col1:
        st.markdown('**Input features**')
        s_len = st.slider('sepal length (cm)', float(df.iloc[:,0].min()), float(df.iloc[:,0].max()), float(df.iloc[0,0]))
        s_wid = st.slider('sepal width (cm)', float(df.iloc[:,1].min()), float(df.iloc[:,1].max()), float(df.iloc[0,1]))
        p_len = st.slider('petal length (cm)', float(df.iloc[:,2].min()), float(df.iloc[:,2].max()), float(df.iloc[0,2]))
        p_wid = st.slider('petal width (cm)', float(df.iloc[:,3].min()), float(df.iloc[:,3].max()), float(df.iloc[0,3]))

        st.markdown('**Example quick picks**')
        if st.button('Set: Setosa-like'):
            s_len, s_wid, p_len, p_wid = 5.0, 3.6, 1.4, 0.2
        if st.button('Set: Virginica-like'):
            s_len, s_wid, p_len, p_wid = 6.5, 3.0, 5.5, 2.0

    with col2:
        st.markdown('**Prediction result**')
        model_key = f"svm_{C}_{kernel}_{gamma}"
        if 'trained' not in st.session_state:
            st.session_state.trained = False
        if (not st.session_state.trained) or retrain:
            X = df[iris.feature_names].values
            y = df['target'].values
            model, scaler = train_model(X, y, C=C, kernel=kernel, gamma=gamma, probability=True)
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.trained = True

        Xnew = np.array([[s_len, s_wid, p_len, p_wid]])
        pred, probs = predict_with_model(st.session_state.model, st.session_state.scaler, Xnew)
        label = iris.target_names[pred[0]]
        st.markdown(f"### Predicted species: **{label}**")
        prob_df = pd.DataFrame({'species': iris.target_names, 'probability': probs[0]})
        st.bar_chart(prob_df.set_index('species'))
        st.write('Raw probabilities:')
        st.dataframe(prob_df)

    st.markdown('</div>', unsafe_allow_html=True)

with tabs[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader('PCA Projection & Decision Surface (approx)')
    X = df[iris.feature_names].values
    y = df['target'].values
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(StandardScaler().fit_transform(X))

    pca_model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
    pca_model.fit(X2, y)
    fig = px.scatter(x=X2[:,0], y=X2[:,1], color=df['species'], labels={'x':'PC1','y':'PC2'}, title='PCA scatter of Iris')
    xmin, xmax = X2[:,0].min()-1, X2[:,0].max()+1
    ymin, ymax = X2[:,1].min()-1, X2[:,1].max()+1
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 200), np.linspace(ymin, ymax, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = pca_model.predict(grid).reshape(xx.shape)
    fig.add_trace(go.Contour(x=np.linspace(xmin, xmax, 200), y=np.linspace(ymin, ymax, 200), z=Z, showscale=False, opacity=0.25, contours=dict(showlines=False)))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

with tabs[2]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader('Training details & cross-validation')
    X = df[iris.feature_names].values
    y = df['target'].values
    scores = cross_val_score(SVC(C=C, kernel=kernel, gamma=gamma, probability=True), StandardScaler().fit_transform(X), y, cv=5)
    st.metric('CV Accuracy (5-fold)', f"{scores.mean():.3f}", delta=f"{(scores.mean()-scores.min()):.3f}")
    st.write('Fold accuracies:')
    st.write(scores)

    st.markdown('**Model coefficients in PCA space (approximate importance)**')
    lin = SVC(C=C, kernel='linear', probability=True)
    lin.fit(StandardScaler().fit_transform(X), y)
    coefs = np.mean(np.abs(lin.coef_), axis=0)
    feat_imp = pd.DataFrame({'feature': iris.feature_names, 'importance': coefs})
    feat_imp = feat_imp.sort_values('importance', ascending=False)
    st.dataframe(feat_imp)
    st.bar_chart(feat_imp.set_index('feature'))

    st.markdown('</div>', unsafe_allow_html=True)

with tabs[3]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader('Dataset explorer & export')
    st.write('First five rows:')
    st.dataframe(df.head())

    st.markdown('Download trained model (pickle)')
    model_bytes = model_to_bytes(st.session_state.model, st.session_state.scaler)
    st.download_button('Download model (.pkl)', data=model_bytes, file_name='iris_svm_model.pkl', mime='application/octet-stream')

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div style='margin-top:18px; padding:12px; background: rgba(255,255,255,0.6); border-radius:12px'>
<strong>Made by ENIGJES - Running in Google Colab!</strong><br>
</div>
""", unsafe_allow_html=True)
