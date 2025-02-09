from agentic_graph import graph
from langchain_core.messages import HumanMessage
from ChromaDB_HuggingFace import retriever
from langchain_huggingface import HuggingFacePipeline

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
import torch

MODEL_NAME = 'ankner/chat-llama3-1b-base-rm' # enter model name

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

question = "Give me a complete mission plan to launch a satellite to the moon with a 1 million dollar budget"

inputs = {"messages": [HumanMessage(question)], "question": question, "answer_dict": {}}
config = {"configurable": {"llm": llm, "retriever": retriever}}
response = graph.invoke(inputs, config)
print(response)