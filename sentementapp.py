!pip install streamlit transformers torch
import streamlit as st
from transformers import pipeline

st.title("Sentiment Analyzer 🤖")
st.write("Enter text and get a sentiment prediction (positive or negative).")

@st.cache_resource
def load_pipeline():
    return pipeline("sentiment-analysis")

nlp = load_pipeline()

# User input
user_input = st.text_area("Type your sentence here:")

if user_input:
    with st.spinner("Analyzing..."):
        result = nlp(user_input)[0]
        st.subheader("Prediction:")
        st.write(f"**Label:** {result['label']}  \n**Confidence:** {result['score']:.2f}")
