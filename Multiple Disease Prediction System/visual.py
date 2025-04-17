import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import pickle
import time
import os

# Set page config
st.set_page_config(page_title="🩺 Disease Dashboard", layout="wide")

# Animated heart background (CSS + Lottie)
st.markdown("""
    <style>
    .heart-bg {
        position: fixed;
        top: 0;
        left: 0;
        z-index: -1;
        opacity: 0.1;
        width: 100vw;
        height: 100vh;
        display: flex;
        justify-content: center;
        align-items: center;
        pointer-events: none;
    }
    </style>
    <div class="heart-bg">
        <lottie-player src="https://assets3.lottiefiles.com/packages/lf20_jz0qg93e.json" background="transparent" speed="1" style="width: 300px; height: 300px;" loop autoplay></lottie-player>
    </div>
    <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
""", unsafe_allow_html=True)

# Load datasets
diabetes_data = pd.read_csv(r"C:\Users\Abhay Patil\Desktop\Project\MDPS\dataset\diabetes.csv")
heart_data = pd.read_csv(r"C:\Users\Abhay Patil\Desktop\Project\MDPS\dataset\heart.csv")
parkinsons_data = pd.read_csv(r"C:\Users\Abhay Patil\Desktop\Project\MDPS\dataset\parkinsons.csv")

# Disease selection
selected_disease = st.sidebar.radio("Select Disease Dataset:", ("Diabetes", "Heart Disease", "Parkinson's"))

# Path to audio narration folder
audio_path = r"C:\Users\Abhay Patil\Desktop\Project\MDPS\narration_audio_files"

# Manual play for narration using st.audio() with full path
if selected_disease == "Diabetes":
    st.audio(os.path.join(audio_path, "diabetes_narration.mp3"), format="audio/mp3", start_time=0)
elif selected_disease == "Heart Disease":
    st.audio(os.path.join(audio_path, "heart_narration.mp3"), format="audio/mp3", start_time=0)
elif selected_disease == "Parkinson's":
    st.audio(os.path.join(audio_path, "parkinsons_narration.mp3"), format="audio/mp3", start_time=0)

st.title("📊 Interactive Disease Dashboard")

# Utility function for common visuals
def common_visuals(df, target_col, age_col=None):
    if age_col:
        st.subheader("📈 Trendline by Age Groups")
        df['Age Group'] = pd.cut(df[age_col], bins=[20,30,40,50,60,70,80], labels=['20s','30s','40s','50s','60s','70s'])
        trend_data = df.groupby('Age Group')[target_col].mean().reset_index()
        fig_trend = px.line(trend_data, x='Age Group', y=target_col, title="Risk Trend by Age Group")
        st.plotly_chart(fig_trend, use_container_width=True)

    st.subheader("🌡️ Health Risk Heatmap")
    corr = df.select_dtypes(include=np.number).corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader("📊 Feature Distribution vs Class")
    for col in df.select_dtypes(include=np.number).columns[:3]:
        if col != target_col:
            fig_violin = px.violin(df, y=col, color=target_col, box=True, points="all", title=f"{col} Distribution by Class")
            st.plotly_chart(fig_violin, use_container_width=True)

    st.subheader("🧮 Prediction Counter")
    sick_count = df[target_col].sum()
    healthy_count = len(df) - sick_count
    st.metric("Total Sick Predictions", sick_count)
    st.metric("Total Healthy Predictions", healthy_count)

    st.subheader("📉 Model Performance Charts (Simulated)")
    performance_data = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Score': [0.85, 0.80, 0.83, 0.82]
    })
    fig_perf = px.bar(performance_data, x='Metric', y='Score', title="Simulated Model Performance")
    st.plotly_chart(fig_perf, use_container_width=True)

    if age_col:
        st.subheader("🌈 Animated Time Evolution")
        anim_data = df.groupby(['Age Group', target_col]).size().reset_index(name='Count')
        fig_anim = px.bar(anim_data, x=target_col, y='Count', color=target_col, animation_frame='Age Group', title="Outcome Evolution Over Age Groups")
        st.plotly_chart(fig_anim, use_container_width=True)

    st.subheader("📊 Parallel Coordinates")
    df_parallel = df.select_dtypes(include=np.number).copy()
    df_parallel[target_col] = df[target_col]
    fig_parallel = px.parallel_coordinates(df_parallel, color=target_col, dimensions=df_parallel.columns[:5], color_continuous_scale=px.colors.diverging.Tealrose)
    st.plotly_chart(fig_parallel, use_container_width=True)

    st.subheader("🧬 PCA Cluster Plot")
    features = df.select_dtypes(include=np.number).drop(columns=[target_col], errors='ignore')
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=2)
    components = pca.fit_transform(features_scaled)
    pca_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
    pca_df[target_col] = df[target_col]
    fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color=target_col, color_continuous_scale=px.colors.sequential.RdBu, title="PCA Cluster Projection")
    st.plotly_chart(fig_pca, use_container_width=True)

# Render based on selected disease
if selected_disease == "Diabetes":
    st.header("🩺 Diabetes Insights")
    common_visuals(diabetes_data, target_col='Outcome', age_col='Age')
elif selected_disease == "Heart Disease":
    st.header("❤️ Heart Disease Insights")
    common_visuals(heart_data, target_col='target', age_col='age')
elif selected_disease == "Parkinson's":
    st.header("🧠 Parkinson's Insights")
    common_visuals(parkinsons_data.drop(columns=['name'], errors='ignore'), target_col='status')

st.info("✅ Select a disease from the sidebar to explore its complete data journey, from trends to advanced projections!")
