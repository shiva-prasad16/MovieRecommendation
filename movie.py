import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🎬 Movie Recommendation")

movies = [
    "The Dark Knight is an action-crime thriller",
    "Interstellar is a science-fiction adventure drama",
    "Inception is a science-fiction action thriller",
    "The Godfather is a crime drama",
    "Fight Club is a psychological drama",
    "The Shawshank Redemption is a drama",
    "Forrest Gump is a drama-romance",
    "The Matrix is a science-fiction action film",
    "Gladiator is an action historical drama",
    "Whiplash is a drama-music film"
]

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

@st.cache_data
def get_embeddings():
    return model.encode(movies)

embeddings = get_embeddings()

query = st.text_input(
    "Enter movie description",
    placeholder="space travel, black holes, dreams..."
)

if query:
    query_embedding = model.encode([query])

    scores = cosine_similarity(query_embedding, embeddings)[0]

    df = pd.DataFrame({
        "Movie": movies,
        "Similarity": scores
    }).sort_values(
        by="Similarity",
        ascending=False
    )

    st.subheader("Best Match")
    st.success(df.iloc[0]["Movie"])

    st.subheader("Top Results")
    st.dataframe(df, use_container_width=True)