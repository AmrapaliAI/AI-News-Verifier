import streamlit as st
import pandas as pd
from newsapi import NewsApiClient
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI News Verifier", page_icon="⚖️", layout="wide")

# Initialize NewsAPI
newsapi = NewsApiClient(api_key='6e868cef15364ffcb96079847a7794a1')


# --- STEP 1: LOAD & TRAIN ---
@st.cache_resource
def train_model():
    df = pd.read_csv('News.csv')
    x_train, x_test, y_train, y_test = train_test_split(df['text'], df.label, test_size=0.2, random_state=7)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(x_train)
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)
    y_pred = pac.predict(tfidf_vectorizer.transform(x_test))
    acc = accuracy_score(y_test, y_pred)
    return tfidf_vectorizer, pac, y_test, y_pred, acc


tfidf_vectorizer, pac, y_test, y_pred, acc = train_model()

# --- STEP 2: SIDEBAR WITH SEABORN HEATMAP ---
st.sidebar.header("📊 Model Performance")
st.sidebar.write(f"**Accuracy:** {acc * 100:.2f}%")

if st.sidebar.checkbox("Show Confusion Heatmap"):
    st.sidebar.write("Confusion Matrix (Seaborn):")
    cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', ax=ax,
                xticklabels=['Predicted FAKE', 'Predicted REAL'],
                yticklabels=['Actual FAKE', 'Actual REAL'])
    plt.title("Model Error Analysis")
    st.sidebar.pyplot(fig)

# --- STEP 3: MAIN INTERFACE ---
st.title("🕵️‍♂️ Fake News Pro: API & Analytics")
tab1, tab2 = st.tabs(["Upload CSV Verification", "Live API News"])

with tab1:
    st.subheader("Check Multiple News via CSV")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)

        # Check if 'text' column exists in uploaded file
        if 'text' in user_df.columns:
            if st.button("Analyze Uploaded File"):
                # Transform and Predict
                tfidf_test = tfidf_vectorizer.transform(user_df['text'].fillna(''))
                predictions = pac.predict(tfidf_test)

                # Add predictions to dataframe
                user_df['AI_Verdict'] = predictions

                # Display Summary
                fake_count = (predictions == 'FAKE').sum()
                real_count = (predictions == 'REAL').sum()

                c1, c2 = st.columns(2)
                c1.metric("Fake News Detected", fake_count)
                c2.metric("Real News Detected", real_count)

                # Show results table
                st.write("### Prediction Results")
                st.dataframe(user_df[['text', 'AI_Verdict']], use_container_width=True)
        else:
            st.error("The uploaded CSV must have a column named **'text'** to perform analysis.")

with tab2:
    st.subheader("Global Live Headlines")
    col1, col2 = st.columns([1, 3])
    with col1:
        category = st.selectbox("Select Category:", ["technology", "health", "science", "business", "sports"])
        count = st.slider("Number of articles:", 5, 20, 5)

    if st.button("Fetch Latest News"):
        top_headlines = newsapi.get_top_headlines(category=category, language='en', page_size=count)

        for article in top_headlines['articles']:
            text_to_test = f"{article['title']} {article['description']}"
            pred = pac.predict(tfidf_vectorizer.transform([text_to_test]))[0]

            with st.expander(f"[{pred}] - {article['title']}"):
                st.write(f"**Source:** {article['source']['name']}")
                st.write(article['description'])
                st.write(f"[Read full article]({article['url']})")