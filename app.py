import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI News Verifier",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
:root {
    --bg-main: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    --glass-bg: rgba(255, 255, 255, 0.12);
    --glass-border: rgba(255, 255, 255, 0.25);
    --neon-blue: #38bdf8;
    --neon-purple: #a855f7;
    --neon-pink: #ec4899;
    --gold: #facc15;
}
.stApp { background: var(--bg-main); color: #f8fafc; }
h1 { font-size: 3rem; font-weight: 900; text-align: center; }
.metric-card { background: rgba(255,255,255,0.05); border-radius: 18px; padding: 1rem; }
</style>
""", unsafe_allow_html=True)

# --- LOAD OR TRAIN MODEL ---
@st.cache_resource
def load_or_train_model(uploaded_df=None):
    model_path = "news_model.pkl"
    vectorizer_path = "tfidf_vectorizer.pkl"

    if uploaded_df is None and os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        tfidf_vectorizer = joblib.load(vectorizer_path)
        return tfidf_vectorizer, model, None, None, None, None, False

    if uploaded_df is not None:
        df = uploaded_df
    else:
        if not os.path.exists("News.csv"):
            return None, None, None, None, None, None, False
        df = pd.read_csv("News.csv")

    # Balance dataset
    df_fake = df[df.label == 'FAKE']
    df_real = df[df.label == 'REAL']
    df_real_upsampled = resample(df_real, replace=True, n_samples=len(df_fake), random_state=42)
    df_balanced = pd.concat([df_fake, df_real_upsampled])

    x_train, x_test, y_train, y_test = train_test_split(
        df_balanced['text'], df_balanced['label'], test_size=0.2, random_state=7
    )

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(x_train)

    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(tfidf_train, y_train)

    y_pred = model.predict(tfidf_vectorizer.transform(x_test))
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])

    joblib.dump(model, model_path)
    joblib.dump(tfidf_vectorizer, vectorizer_path)

    return tfidf_vectorizer, model, y_test, y_pred, acc, cm, True

# --- MAIN APP ---
st.title("🕵️ AI News Verifier")
st.markdown("<p class='subtitle'>Upload CSV files for intelligent fake news detection with visualizations</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🎓 Train Model")
uploaded_file = st.sidebar.file_uploader("📂 Upload Training CSV", type=['csv'], help="CSV must have 'text' and 'label' columns")

if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file)
    if 'text' in uploaded_df.columns and 'label' in uploaded_df.columns:
        st.sidebar.success(f"✅ {len(uploaded_df)} rows loaded successfully")
        if st.sidebar.button("🚀 Train Model"):
            with st.spinner("🔄 Training AI model..."):
                tfidf_vectorizer, model, y_test, y_pred, acc, cm, trained_fresh = load_or_train_model(uploaded_df)
                st.sidebar.success(f"✅ Model trained! Accuracy: {acc * 100:.2f}%")
                st.rerun()
    else:
        st.sidebar.error("❌ CSV must have 'text' and 'label' columns")
        uploaded_df = None
else:
    uploaded_df = None

# Load model
tfidf_vectorizer, model, y_test, y_pred, acc, cm, trained_fresh = load_or_train_model()
if model is None:
    st.error("⚠️ No model found. Please upload a training CSV file in the sidebar.")
    st.stop()

# --- Performance ---
st.sidebar.header("📈 Model Performance")
if trained_fresh and y_test is not None:
    st.sidebar.metric("Accuracy", f"{acc*100:.2f}%")
    if st.sidebar.checkbox("📊 Show Confusion Matrix", value=True):
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', ax=ax,
                    xticklabels=['Predicted FAKE', 'Predicted REAL'],
                    yticklabels=['Actual FAKE', 'Actual REAL'])
        st.sidebar.pyplot(fig)

# --- Batch Analysis ---
st.header("📂 Batch Analysis")
batch_file = st.file_uploader("📄 Upload CSV for Analysis (must have 'text' column)", type=['csv'])
if batch_file is not None:
    batch_df = pd.read_csv(batch_file)
    if 'text' not in batch_df.columns:
        st.error("❌ CSV must contain a 'text' column")
    else:
        st.success(f"✅ Loaded {len(batch_df)} articles for analysis")
        if st.button("🚀 Analyze All Articles"):
            predictions, fake_probabilities, real_probabilities = [], [], []
            for text in batch_df['text']:
                tfidf_test = tfidf_vectorizer.transform([str(text)])
                pred = model.predict(tfidf_test)[0]
                proba = model.predict_proba(tfidf_test)[0]
                predictions.append(pred)
                fake_probabilities.append(f"{proba[0]:.2%}")
                real_probabilities.append(f"{proba[1]:.2%}")

            batch_df['Prediction'] = predictions
            batch_df['FAKE_Probability'] = fake_probabilities
            batch_df['REAL_Probability'] = real_probabilities

            st.success("✅ Analysis Complete!")

            # Results table
            def highlight_prediction(val):
                if val == 'REAL':
                    return 'background-color: #d1fae5; color: #065f46; font-weight: bold'
                elif val == 'FAKE':
                    return 'background-color: #fee2e2; color: #991b1b; font-weight: bold'
                else:
                    return ''
            styled_df = batch_df.style.map(highlight_prediction, subset=['Prediction'])
            st.dataframe(styled_df, width='stretch', height=500)

            # Download results
            csv = batch_df.to_csv(index=False)
            st.download_button("📥 Download Results CSV", csv, "news_analysis_results.csv", "text/csv")
else:
    st.info("👆 Upload your CSV file to get started")
