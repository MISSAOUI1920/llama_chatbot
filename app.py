import streamlit as st
import torch

# Ensure 'bitsandbytes' and other dependencies are installed correctly
# This code will work assuming you are using CPU-only

# Load the model and tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os 

model_id = os.getcwd()
if len(sys.argv) > 1:
    model_id = sys.argv[1]

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).cuda().bfloat16()
prompt = "Lily picked up a flower."
inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to('cuda')
out = model.generate(**inputs, max_new_tokens=80).ravel()
out = tokenizer.decode(out)

# Streamlit app layout
st.title("Chatbot with LLaMA Model")
st.write("Enter your message below and get a response from the chatbot.")
st.write(f"Chatbot: {out}")
