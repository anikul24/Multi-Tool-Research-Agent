import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path="./cred.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")
INDEX_NAME = 'langgraph-research-agent'

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Missing API Keys in .env file")