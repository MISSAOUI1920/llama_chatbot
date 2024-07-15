import streamlit as st
import torch

# Ensure 'bitsandbytes' and other dependencies are installed correctly
# This code will work assuming you are using CPU-only

# Load the model and tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os 

tokenizer = AutoTokenizer.from_pretrained("Maykeye/TinyLLama-v0")
model = AutoModelForCausalLM.from_pretrained("Maykeye/TinyLLama-v0")

# Define the prompt you want to test
prompt = "Once upon a time in a land far, far away,"

# Encode the prompt
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text
output = model.generate(inputs.input_ids, max_length=100, num_return_sequences=1)

# Decode the generated text
out= tokenizer.decode(output[0], skip_special_tokens=True)

# Streamlit app layout
st.title("Chatbot with LLaMA Model")
st.write("Enter your message below and get a response from the chatbot.")
st.write(f"Chatbot: {out}")
