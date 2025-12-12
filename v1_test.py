import os
from PathRAG import PathRAG, QueryParam
from PathRAG.llm import gpt_4o_mini_complete

WORKING_DIR = "./pathrag_cache"  # Set your working directory

api_key="your-openai-api-key-here"  # Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = api_key
base_url="https://api.openai.com/v1"
os.environ["OPENAI_API_BASE"]=base_url


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = PathRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete,  
)

data_file="./text.txt"  # Path to your input document
question="What is the main topic of this document?"  # Your query
with open(data_file) as f:
    rag.insert(f.read())

print(rag.query(question, param=QueryParam(mode="hybrid")))














