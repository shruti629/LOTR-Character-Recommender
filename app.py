
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="LOTR Character Recommender",
    page_icon="💍",
    layout="wide"
)

# ---------------- Load Dataset ----------------
movies = pd.read_pickle("movies.pkl")

movies = movies.dropna(subset=["char", "dialog"])

movies["char"] = movies["char"].astype(str)
movies["dialog"] = movies["dialog"].astype(str)

# ---------------- TF-IDF ----------------
vectorizer = TfidfVectorizer(stop_words="english")

tfidf_matrix = vectorizer.fit_transform(movies["dialog"])

# ---------------- UI ----------------
st.title("💍 Lord of the Rings")
st.subheader("Character Recommendation System Based on Dialogue Similarity")

st.info(
    "Select any dialogue to find the Top 5 characters "
    "whose dialogues are most similar to it."
)

# ---------------- Dialogue Selection ----------------
selected_dialog = st.selectbox(
    "Select Dialogue",
    movies["dialog"].unique()
)

# ---------------- Recommendation Button ----------------
if st.button("🔍 Recommend Characters"):

    # Transform selected dialogue
    selected_vector = vectorizer.transform([selected_dialog])

    # Calculate similarity with every dialogue
    similarity_scores = cosine_similarity(
        selected_vector,
        tfidf_matrix
    ).flatten()

    # Add similarity scores to dataset
    movies_copy = movies.copy()

    movies_copy["Similarity Score"] = similarity_scores

    # Group by character and calculate average similarity
    character_scores = (
        movies_copy.groupby("char")["Similarity Score"]
        .mean()
        .sort_values(ascending=False)
    )

    # Top 5 characters
    top_5 = character_scores.head(5)

    # Display results
    st.subheader("🏆 Top 5 Recommended Characters")

    result = pd.DataFrame({
        "Character": top_5.index
    })

    st.dataframe(
        result,
        use_container_width=True,
        hide_index=True
    )

# ---------------- Footer ----------------
st.caption(
    "Built with Python | Streamlit | NLP | TF-IDF | Cosine Similarity"
)


