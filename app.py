import streamlit as st
from peft import PeftModel, PeftConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
config = PeftConfig.from_pretrained("MISSAOUI/llama_model_2")
base_model = AutoModelForCausalLM.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
model = PeftModel.from_pretrained(base_model, "MISSAOUI/llama_model_2")
tokenizer = AutoTokenizer.from_pretrained("MISSAOUI/llama_model_2")

# Streamlit app layout
st.title("Chatbot with LLaMA Model")
st.write("Enter your message below and get a response from the chatbot.")

user_input = st.text_input("You: ")

if st.button("Send"):
    if user_input:
        input_ids = tokenizer.encode(user_input, return_tensors="pt")
        
        with torch.no_grad():
            output = model.generate(input_ids, max_length=50, num_return_sequences=1)
        
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        st.write(f"Chatbot: {response}")
    else:
        st.write("Please enter a message.")

