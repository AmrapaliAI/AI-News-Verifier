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
    page_icon="&#128373;",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
    }
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        font-size: 3.5rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
    }
    h2, h3 {
        color: #1e40af;
        font-weight: 700;
    }
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stButton>button {
    position: relative;
    overflow: hidden;
    /* ... existing styles ... */
}

.stButton>button::after {
    content: '';
    position: absolute;
    top: -50%;
    left: -60%;
    width: 20%;
    height: 200%;
    background: rgba(255, 255, 255, 0.4);
    transform: rotate(30deg);
    transition: all 0.6s ease;
}

.stButton>button:hover::after {
    left: 120%;
}
    
    .upload-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #e3e7f1 100%);
        border: 3px dashed #667eea;
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        margin: 2rem 0;
    }
    .metric-card {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    .success-box {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #10b981;
        margin: 1rem 0;
    }
    .info-box {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #3b82f6;
        margin: 1rem 0;
    }/* ===== Success Box (Luxury Notification) ===== */
.success-box {
    background: rgba(16, 185, 129, 0.1); /* Subtle emerald tint */
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #34d399 !important; /* Bright emerald text */
    padding: 12px 18px;
    border-radius: 14px;
    font-size: 0.9rem;
    text-align: center;
    box-shadow: 
        0 4px 15px rgba(0, 0, 0, 0.3),
        inset 0 0 10px rgba(16, 185, 129, 0.1);
    backdrop-filter: blur(8px);
    margin: 10px 0;
    animation: slideInRight 0.5s ease-out, pulse-border 3s infinite;
}

/* Subtle border pulse animation */
@keyframes pulse-border {
    0% { border-color: rgba(16, 185, 129, 0.4); box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3); }
    50% { border-color: rgba(16, 185, 129, 0.8); box-shadow: 0 0 20px rgba(16, 185, 129, 0.3); }
    100% { border-color: rgba(16, 185, 129, 0.4); box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3); }
}

/* Entrance animation */
@keyframes slideInRight {
    from { opacity: 0; transform: translateX(-20px); }
    to { opacity: 1; transform: translateX(0); }
}

.success-box strong {
    color: #ffffff;
    text-shadow: 0 0 10px rgba(52, 211, 153, 0.6);
    font-weight: 800;
}
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
st.title("&#128373; AI News Verifier")
st.markdown(
    "<p class='subtitle'>Upload CSV files for intelligent fake news detection with beautiful visualizations</p>",
    unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.header("🎓 Train Model")
st.sidebar.markdown("Upload your training dataset to create or update the AI model")

st.sidebar.markdown(f"""
    <div style="
        background: rgba(255, 255, 255, 0.04);
        border-left: 3px solid var(--neon-blue);
        padding: 15px;
        border-radius: 12px;
        margin: 10px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    ">
        <p style="
            color: #cbd5e1;
            font-size: 0.9rem;
            line-height: 1.5;
            margin: 0;
            font-weight: 400;
        ">
            <span style="color: var(--neon-blue); font-weight: bold;">Step 1:</span> 
            Upload your training dataset to create or update the AI model.
        </p>
    </div>
""", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("📂 Upload Training CSV", type=['csv'],
                                         help="CSV must have 'text' and 'label' columns")

if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file)
    if 'text' in uploaded_df.columns and 'label' in uploaded_df.columns:
        st.sidebar.markdown(
            f"<div class='success-box'>✅ <strong>{len(uploaded_df)} rows</strong> loaded successfully</div>",
            unsafe_allow_html=True)
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

# --- SIDEBAR PERFORMANCE ---
st.sidebar.markdown("---")
st.sidebar.header("📈 Model Performance")
if trained_fresh and y_test is not None:
    st.sidebar.markdown(f"""
    <div class='metric-card'>
        <h2 style='color: #10b981; margin: 0;'>{acc * 100:.2f}%</h2>
        <p style='color: #64748b; margin: 0;'>Model Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

    if st.sidebar.checkbox("📊 Show Confusion Matrix", value=True):
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='RdYlGn', ax=ax,
            xticklabels=['Predicted FAKE', 'Predicted REAL'],
            yticklabels=['Actual FAKE', 'Actual REAL'],
            cbar_kws={'label': 'Count'},
            linewidths=3,
            linecolor='white',
            annot_kws={'size': 14, 'weight': 'bold'}
        )
        plt.title("Confusion Matrix", fontsize=18, fontweight='bold', pad=20)
        plt.xlabel("Predicted Label", fontsize=13, fontweight='bold')
        plt.ylabel("Actual Label", fontsize=13, fontweight='bold')
        st.sidebar.pyplot(fig)
        plt.close()

        # Download heatmap
        fig.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
        with open("confusion_matrix.png", "rb") as file:
            st.sidebar.download_button(
                label="📥 Download Heatmap",
                data=file,
                file_name="confusion_matrix.png",
                mime="image/png"
            )
else:
    st.sidebar.info("ℹ️ Model loaded from cache")

# --- MAIN BATCH ANALYSIS ---
st.markdown("---")
st.header("📂 Batch Analysis")

# Upload section
col1, col2 = st.columns([2, 1])

with col1:
    batch_file = st.file_uploader(
        "📄 Upload CSV for Analysis (must have 'text' column)",
        type=['csv'],
        key='batch_analyzer',
        help="Upload a CSV file containing news articles to analyze"
    )

with col2:
    st.markdown("""
    <div class='info-box'>
        <h4>📋 Requirements</h4>
        <ul>
            <li>CSV format</li>
            <li>'text' column required</li>
            <li>Optional: 'headline' column</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

if batch_file is not None:
    try:
        batch_df = pd.read_csv(batch_file)

        if 'text' not in batch_df.columns:
            st.error("❌ CSV must contain a 'text' column with news articles")
        else:
            st.markdown(f"<div class='success-box'><h3>✅ Loaded {len(batch_df)} articles for analysis</h3></div>",
                        unsafe_allow_html=True)

            # Analysis button
            if st.button("🚀 Analyze All Articles", use_container_width=True, key="analyze_btn"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                predictions = []
                fake_probabilities = []
                real_probabilities = []

                for idx, row in batch_df.iterrows():
                    status_text.markdown(
                        f"<p style='text-align: center; font-size: 1.2rem;'>🔍 Analyzing article <strong>{idx + 1}/{len(batch_df)}</strong>...</p>",
                        unsafe_allow_html=True)
                    progress_bar.progress((idx + 1) / len(batch_df))

                    text = str(row['text'])

                    # Model prediction
                    tfidf_test = tfidf_vectorizer.transform([text])
                    pred = model.predict(tfidf_test)[0]
                    proba = model.predict_proba(tfidf_test)[0]

                    predictions.append(pred)
                    fake_probabilities.append(f"{proba[0]:.2%}")
                    real_probabilities.append(f"{proba[1]:.2%}")

                progress_bar.empty()
                status_text.empty()

                # Add results to dataframe
                batch_df['Prediction'] = predictions
                batch_df['FAKE_Probability'] = fake_probabilities
                batch_df['REAL_Probability'] = real_probabilities

                # Display results summary
                st.markdown("<div class='success-box'><h2>✅ Analysis Complete!</h2></div>", unsafe_allow_html=True)

                # Metrics
                col1, col2, col3 = st.columns(3)
                real_count = predictions.count('REAL')
                fake_count = predictions.count('FAKE')

                with col1:
                    st.markdown(f"""
                    <div class='metric-card' style='border-left-color: #10b981;'>
                        <h1 style='color: #10b981; margin: 0;'>{real_count}</h1>
                        <h3 style='color: #64748b; margin: 0;'>REAL Articles</h3>
                        <p style='color: #94a3b8; font-size: 1.2rem;'>{real_count / len(predictions) * 100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class='metric-card' style='border-left-color: #ef4444;'>
                        <h1 style='color: #ef4444; margin: 0;'>{fake_count}</h1>
                        <h3 style='color: #64748b; margin: 0;'>FAKE Articles</h3>
                        <p style='color: #94a3b8; font-size: 1.2rem;'>{fake_count / len(predictions) * 100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class='metric-card' style='border-left-color: #667eea;'>
                        <h1 style='color: #667eea; margin: 0;'>{len(predictions)}</h1>
                        <h3 style='color: #64748b; margin: 0;'>Total Analyzed</h3>
                        <p style='color: #94a3b8; font-size: 1.2rem;'>100%</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Visualizations
                st.markdown("---")
                st.subheader("📊 Visual Analytics")

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

                # Pie chart
                labels = ['REAL', 'FAKE']
                sizes = [real_count, fake_count]
                colors = ['#10b981', '#ef4444']
                explode = (0.05, 0.05)

                ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                        shadow=True, startangle=90, textprops={'fontsize': 14, 'fontweight': 'bold'})
                ax1.set_title('News Classification Distribution', fontsize=18, fontweight='bold', pad=20)

                # Bar chart
                ax2.bar(labels, sizes, color=colors, edgecolor='white', linewidth=3, width=0.6)
                ax2.set_ylabel('Number of Articles', fontsize=14, fontweight='bold')
                ax2.set_title('Article Count by Category', fontsize=18, fontweight='bold', pad=20)
                ax2.grid(axis='y', alpha=0.3, linestyle='--')
                ax2.set_facecolor('#f8fafc')

                for i, v in enumerate(sizes):
                    ax2.text(i, v + max(sizes) * 0.02, str(v), ha='center', va='bottom',
                             fontweight='bold', fontsize=16)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # Results table
                st.markdown("---")
                st.subheader("📋 Detailed Results")

                # Add headline column if not exists
                if 'headline' not in batch_df.columns:
                    batch_df['Headline'] = batch_df['text'].str[:100] + '...'
                    cols = ['Headline'] + [col for col in batch_df.columns if col != 'Headline']
                    batch_df = batch_df[cols]


                # Color-code the prediction column
                def highlight_prediction(val):
                    if val == 'REAL':
                        return 'background-color: #d1fae5; color: #065f46; font-weight: bold'
                    elif val == 'FAKE':
                        return 'background-color: #fee2e2; color: #991b1b; font-weight: bold'
                    else:
                        return 'background-color: #fef3c7; color: #92400e; font-weight: bold'


                styled_df = batch_df.style.applymap(
                    highlight_prediction,
                    subset=['Prediction']
                )

                st.dataframe(styled_df, use_container_width=True, height=500)

                # Download results
                st.markdown("---")
                csv = batch_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Analysis Results (CSV)",
                    data=csv,
                    file_name="news_analysis_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                # Statistics
                with st.expander("📈 Detailed Statistics & Insights"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### 📊 Summary Statistics")
                        st.write(f"**Total Articles Analyzed:** {len(batch_df)}")
                        st.write(f"**REAL Articles:** {real_count} ({real_count / len(predictions) * 100:.1f}%)")
                        st.write(f"**FAKE Articles:** {fake_count} ({fake_count / len(predictions) * 100:.1f}%)")

                    with col2:
                        st.markdown("### 🎯 Confidence Metrics")
                        st.write(
                            f"**Average FAKE Probability:** {np.mean([float(p.strip('%')) / 100 for p in fake_probabilities]):.2%}")
                        st.write(
                            f"**Average REAL Probability:** {np.mean([float(p.strip('%')) / 100 for p in real_probabilities]):.2%}")

                        # High confidence predictions
                        high_conf_fake = sum(1 for p in fake_probabilities if float(p.strip('%')) > 80)
                        high_conf_real = sum(1 for p in real_probabilities if float(p.strip('%')) > 80)
                        st.write(f"**High Confidence (>80%):** {high_conf_fake + high_conf_real} articles")

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please ensure your CSV file has the correct format with a 'text' column.")

else:
    # Show instructions when no file uploaded
    st.markdown("""
    <div class='upload-box'>
        <h2>👆 Upload your CSV file to get started</h2>
        <p style='font-size: 1.1rem; color: #64748b; margin-top: 1rem;'>
            Drag and drop your CSV file above or click to browse
        </p>
        <p style='color: #94a3b8; margin-top: 1rem;'>
            Supported format: CSV with 'text' column containing news articles
        </p>
    </div>
    """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<style>

/* ===== Root Theme Variables ===== */
:root {
    --bg-main: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    --glass-bg: rgba(255, 255, 255, 0.12);
    --glass-border: rgba(255, 255, 255, 0.25);
    --neon-blue: #38bdf8;
    --neon-purple: #a855f7;
    --neon-pink: #ec4899;
    --gold: #facc15;
}

/* ===== App Background (Aurora Effect) ===== */
.stApp {
    background: var(--bg-main);
    background-attachment: fixed;
    color: #f8fafc;
    animation: aurora 15s ease infinite;
}

@keyframes aurora {
    0% {filter: hue-rotate(0deg);}
    50% {filter: hue-rotate(25deg);}
    100% {filter: hue-rotate(0deg);}
}

/* ===== Main Container Glass ===== */
.main {
    backdrop-filter: blur(18px) saturate(180%);
    background: var(--glass-bg);
    border-radius: 28px;
    padding: 2.5rem;
    margin: 1.5rem;
    border: 1px solid var(--glass-border);
    box-shadow: 0 20px 60px rgba(0,0,0,0.45);
}

/* ===== Headings (Luxury Neon) ===== */
/* Update your h1 with this shimmer effect */
h1 {
    font-size: 4rem;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(
        to right, 
        #38bdf8 20%, 
        #ec4899 40%, 
        #facc15 60%, 
        #38bdf8 80%
    );
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 4s linear infinite;
}

@keyframes shine {
    to { background-position: 200% center; }
}/* ===== Ultra-Stylish Sidebar Header ===== */
[data-testid="stSidebar"] h2 {
    font-family: 'Inter', sans-serif;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.05em !important;
    background: linear-gradient(135deg, #ffffff 30%, var(--neon-blue));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    padding: 1rem 0;
    border-bottom: 2px solid rgba(255, 255, 255, 0.1);
    margin-bottom: 1.5rem !important;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Adding a glowing pulse dot next to the text */
[data-testid="stSidebar"] h2::before {
    content: "";
    display: inline-block;
    width: 12px;
    height: 12px;
    background-color: var(--neon-pink);
    border-radius: 50%;
    box-shadow: 0 0 15px var(--neon-pink);
    margin-right: 12px;
    animation: pulse-glow 2s infinite;
}

@keyframes pulse-glow {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(236, 72, 153, 0.7); }
    70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(236, 72, 153, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(236, 72, 153, 0); }
}

h2, h3 {
    color: var(--gold);
    font-weight: 800;
    text-shadow: 0 0 8px rgba(250,204,21,0.6);
}

/* ===== Buttons (3D Neon Pulse) ===== */
.stButton>button {
    background: linear-gradient(120deg, var(--neon-blue), var(--neon-purple), var(--neon-pink));
    background-size: 300% 300%;
    animation: gradientMove 5s ease infinite;
    color: white;
    border-radius: 999px;
    padding: 1rem 3rem;
    font-size: 1.25rem;
    font-weight: 800;
    border: none;
    box-shadow: 0 10px 30px rgba(168,85,247,0.5);
    transition: all 0.35s ease;
}

.stButton>button:hover {
    transform: translateY(-4px) scale(1.06);
    box-shadow: 0 20px 50px rgba(236,72,153,0.8);
}

@keyframes gradientMove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* ===== Inputs (Frosted Neon) ===== */
input, textarea, .stTextInput>div>div>input {
    background: rgba(255,255,255,0.08) !important;
    color: white !important;
    border-radius: 16px !important;
    border: 1px solid rgba(56,189,248,0.4) !important;
    padding: 0.75rem 1rem !important;
    transition: all 0.3s ease;
}

input:focus, textarea:focus {
    border-color: var(--neon-purple) !important;
    box-shadow: 0 0 15px var(--neon-purple);
}

/* ===== Upload Box (Hover Glow) ===== */
.upload-box {
    background: rgba(255,255,255,0.08);
    border: 2px dashed var(--neon-blue);
    border-radius: 24px;
    padding: 2.5rem;
    text-align: center;
    box-shadow: 0 0 25px rgba(56,189,248,0.6);
    transition: all 0.35s ease;
}

.upload-box:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 0 40px rgba(168,85,247,0.8);
}

/* ===== Metric Cards (Floating Glass) ===== */
/* Replace or augment your .metric-card */
.metric-card {
    position: relative;
    overflow: hidden;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    transition: border 0.3s ease;
}

.metric-card::before {
    content: "";
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: conic-gradient(
        transparent, 
        var(--neon-blue), 
        transparent 30%
    );
    animation: rotate-border 6s linear infinite;
    z-index: -1;
}

@keyframes rotate-border {
    100% { transform: rotate(360deg); }
}

@keyframes float {
    0% {transform: translateY(0px);}
    50% {transform: translateY(-6px);}
    100% {transform: translateY(0px);}
}

/* ===== Sidebar Luxury ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(15,12,41,0.95), rgba(48,43,99,0.95));
    border-right: 1px solid rgba(255,255,255,0.15);
}

/* ===== Scrollbar Styling ===== */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(var(--neon-purple), var(--neon-pink));
    border-radius: 10px;
}

/* ===== Fade-in Animation ===== */
.fade-in {
    animation: fadeIn 1.2s ease forwards;
}

@keyframes fadeIn {
    from {opacity: 0; transform: translateY(20px);}
    to {opacity: 1; transform: translateY(0);}
}

</style>
""", unsafe_allow_html=True)
