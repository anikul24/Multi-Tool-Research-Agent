import requests
import re
from langchain_core.tools import tool
from pydantic.v1 import BaseModel, Field
from serpapi import GoogleSearch
from vector_store import get_index, get_encoder
from config import SERP_API_KEY

index = get_index()
encoder = get_encoder()

# --- Tool 1: Fetch Arxiv ---
class ArxivInput(BaseModel):
    arvix_id: str = Field(description="The arXiv ID (e.g., '2407.03964') for the paper.")

@tool(args_schema=ArxivInput)
def fetch_arxiv(arvix_id: str) -> str:
    '''Fetch arXiv abstract based on ID'''
    res = requests.get(f'https://arxiv.org/abs/{arvix_id}')
    abstract_pattern = re.compile(
        r'<blockquote class="abstract mathjax">\s*<span class="descriptor">Abstract:</span>\s*(.*?)\s*</blockquote>',
        re.DOTALL
    )
    match = abstract_pattern.search(res.text)
    if match:
        return match.group(1).strip()
    
    # Fallback pattern
    fallback_pattern = re.compile(r'<div id="abstract">\s*<blockquote.*?>(.*?)</blockquote>', re.DOTALL)
    fallback_match = fallback_pattern.search(res.text)
    if fallback_match:
        return fallback_match.group(1).strip()
        
    return f"Abstract not found for {arvix_id}."

# --- Tool 2: Web Search ---
@tool('web_search')
def web_search(query: str) -> str:
    '''Finds general knowledge information using a Google search.'''
    params = {"engine": "google", "api_key": SERP_API_KEY, "q": query, "num": 5}
    search = GoogleSearch(params)
    results = search.get_dict().get('organic_results', [])
    if results:
        return '\n---\n'.join([f"{x['title']}\n{x['snippet']}\n{x['link']}" for x in results])
    return 'No results found.'

# --- Helper for RAG ---
def format_rag_text(matches: list) -> str:
    formatted_text = []
    for x in matches:
        text = (f"Title: {x['metadata']['title']}\n"
                f"Chunk: {x['metadata']['chunk']}\n"
                f"ArXiv ID: {x['metadata']['arxiv_id']}\n")
        formatted_text.append(text)
    return '\n---\n'.join(formatted_text)

# --- Tool 3: RAG Search Filter ---
@tool('rag_search_filter')
def rag_search_filter(query: str, arvix_id: str) -> str:
    '''RAG search filter based on arvix id'''
    query_encode = encoder([query])
    input_vector = index.query(vector=query_encode, top_k=5, include_metadata=True, filter={'arxiv_id': arvix_id})
    return format_rag_text(input_vector['matches'])

# --- Tool 4: RAG Search ---
@tool('rag_search')
def rag_search(query: str) -> str:
    '''RAG search without filter'''
    query_encode = encoder([query])
    input_vector = index.query(vector=query_encode, top_k=5, include_metadata=True)
    return format_rag_text(input_vector['matches'])

# --- Tool 5: Final Answer ---
@tool
def final_answer(introduction: str, research_steps: str | list, main_body: str, conclusion: str, sources: str | list) -> str:
    '''Returns a natural language response in the form of a research report.'''
    if isinstance(research_steps, list):
        research_steps = '\n'.join([f'- {r}' for r in research_steps])
    if isinstance(sources, list):
        sources = '\n'.join([f'- {s}' for s in sources])
    return f"{introduction}\n\nResearch Steps:\n{research_steps}\n\nMain Body:\n{main_body}\n\nConclusion:\n{conclusion}\n\nSources:\n{sources}"

# Export tools list
ALL_TOOLS = [rag_search_filter, rag_search, fetch_arxiv, web_search, final_answer]