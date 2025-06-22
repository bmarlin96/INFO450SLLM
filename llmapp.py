import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

st.title("Local Falcon Chatbot 🦅💬")
st.write("Chat with a small open model — no API keys required!")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
    return tokenizer, model

tokenizer, model = load_model()

if "history" not in st.session_state:
    st.session_state.history = ""

user_input = st.text_area("You:", height=100)

if st.button("Send") and user_input:
    prompt = st.session_state.history + f"\nUser: {user_input}\nAssistant:"

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output_ids = model.generate(
        input_ids,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    reply = output_text.split("Assistant:")[-1].strip()

    # Save history
    st.session_state.history += f"\nUser: {user_input}\nAssistant: {reply}"

    st.markdown(f"**🧑 You:** {user_input}")
    st.markdown(f"**🤖 Assistant:** {reply}")
