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

# Initialize NewsAPI (Replace with your own key)
# Get a free key at: https://newsapi.org/
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

    # Create Seaborn Heatmap
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', ax=ax,  # Yellow-Green-Blue palette
                xticklabels=['Predicted FAKE', 'Predicted REAL'],
                yticklabels=['Actual FAKE', 'Actual REAL'])
    plt.title("Model Error Analysis")
    st.sidebar.pyplot(fig)

# --- STEP 3: MAIN INTERFACE ---
st.title("🕵️‍♂️ Fake News Pro: API & Analytics")
tab1, tab2 = st.tabs(["Manual Verification", "Live API News"])

with tab1:
    user_input = st.text_area("Analyze specific text:", height=200)
    if st.button("Analyze Now"):
        if user_input:
            data = tfidf_vectorizer.transform([user_input])
            res = pac.predict(data)[0]
            if res == 'REAL':
                st.success(f"Result: {res}")
            else:
                st.error(f"Result: {res}")

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