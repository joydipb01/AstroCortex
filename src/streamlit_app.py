import streamlit as st
from agentic_graph import graph
from ChromaDB_HuggingFace import retriever
from langchain_huggingface import HuggingFacePipeline
from langchain_core.messages import HumanMessage

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
import torch

# LLM and Retriever:

MODEL_NAME = '' # enter model name

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
generation_config.max_new_tokens = 1024
generation_config.temperature = 0.0001
generation_config.top_p = 0.7
generation_config.do_sample = True
generation_config.repetition_penalty = 1.15

text_pipeline = pipeline(
    "text-generation",
    model = model,
    tokenizer = tokenizer,
    generation_config = generation_config
)

llm = HuggingFacePipeline(pipeline=text_pipeline)

# Streamlit Page:

st.set_page_config(page_title="AstroCortex", page_icon="🚀")

st.title("AstroCortex")
st.markdown("**AstroCortex: Your Cosmic Co-Pilot.**")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    st.write(message)

with st.form(key="chat_form", clear_on_submit=True):
    input_col, button_col = st.columns([4, 1])
    
    with input_col:
        user_input = st.text_input("Ask your question here:")
        
    with button_col:
        submit = st.form_submit_button("Submit")
    
    if submit and user_input:
        
        inputs = {"messages": [HumanMessage(user_input)], "question": user_input, "answer_dict": {}}
        config = {"configurable": {"llm": llm, "retriever": retriever}}
        response = graph.invoke(inputs, config)
        st.write(response["final_answer"])