import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

st.title("Simple Chatbot 🤖💬")
st.write("Ask me anything!")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    return tokenizer, model

tokenizer, model = load_model()

# Track the conversation history
if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None
if "past_user_inputs" not in st.session_state:
    st.session_state.past_user_inputs = []

user_input = st.text_input("You:", key="input")

if user_input:
    # Encode the input
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

    # Append to previous conversation if exists
    bot_input_ids = torch.cat([st.session_state.chat_history_ids, input_ids], dim=-1) if st.session_state.chat_history_ids is not None else input_ids

    # Generate a response
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8
    )

    # Decode and display
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    st.session_state.chat_history_ids = chat_history_ids
    st.session_state.past_user_inputs.append((user_input, response))

# Display chat history
for i, (user_msg, bot_msg) in enumerate(reversed(st.session_state.past_user_inputs[-5:]), 1):
    st.markdown(f"**You:** {user_msg}")
    st.markdown(f"**Bot:** {bot_msg}")
