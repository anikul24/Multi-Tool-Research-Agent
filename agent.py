import operator
from typing import TypedDict, Annotated, List, cast
from langchain_core.messages import BaseMessage, ToolCall
from langchain_core.agents import AgentAction
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from config import OPENAI_API_KEY
from tools import ALL_TOOLS, rag_search_filter, rag_search, fetch_arxiv, web_search, final_answer

# Define State
class AgentState(TypedDict):
    input: str
    chat_history: List[BaseMessage]
    intermediate_steps: Annotated[List[tuple[AgentAction, str]], operator.add]

# Initialize LLM
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0, openai_api_key=OPENAI_API_KEY)

# Define System Prompt
system_prompt = (
    '''You are the Agent LLM, the great AI decision-maker.
    Given the user's query, you must decide what to do with it based on the
    list of tools provided to you.
    Aim to collect information from a diverse range of sources before
    providing the answer using final_answer tool.'''
)

prompt = ChatPromptTemplate.from_messages([
    ('system', system_prompt),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{input}'),
    ('assistant', 'scratchpad: {scratchpad}'),
])

# Scratchpad Logic
def create_scratchpad(intermediate_steps: list[AgentAction]) -> str:
    research_steps = []
    for action, observation in intermediate_steps:
        research_steps.append(f"Tool: {action.tool}, input: {action.tool_input}\nOutput: {observation}")
    return '\n---\n'.join(research_steps)

# Orchestrator Node
def run_orchestrator(state: AgentState) -> dict:
    scratchpad = create_scratchpad(state['intermediate_steps'])
    
    # Prepare input for chain
    chain_input = {
        'input': state['input'],
        'chat_history': state['chat_history'],
        'scratchpad': scratchpad
    }
    
    orchestrator_chain = prompt | llm.bind_tools(ALL_TOOLS, tool_choice='any')
    out = orchestrator_chain.invoke(chain_input)
    
    # Extract tool call
    tool_name = out.tool_calls[0]['name']
    tool_args = out.tool_calls[0]['args']
    
    action_out = AgentAction(tool=tool_name, tool_input=tool_args, log='TBD')
    return {'intermediate_steps': [(action_out, "TBD")]} # Tuple placeholder

# Tool Execution Node
tool_str_to_func = {
    'rag_search_filter': rag_search_filter,
    'rag_search': rag_search,
    'fetch_arxiv': fetch_arxiv,
    'web_search': web_search,
    'final_answer': final_answer
}

def run_tool(state: AgentState) -> dict:
    last_action, _ = state['intermediate_steps'][-1]
    tool_name = last_action.tool
    tool_args = last_action.tool_input
    
    print(f"Running Tool: {tool_name}")
    
    # Execute Tool
    func = tool_str_to_func[tool_name]
    observation = str(func.invoke(input=tool_args))
    
    # Replace the last placeholder with actual result
    # In a real immutable append scenario, we just add the completed tuple
    # But since we use operator.add, we return a new list containing just this step
    return {'intermediate_steps': [(last_action, observation)]}

# Router
def router(state: AgentState) -> str:
    last_action, _ = state['intermediate_steps'][-1]
    return last_action.tool

# Build Graph
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node('orchestrator', run_orchestrator)
    
    # Add nodes for every tool
    for tool_obj in ALL_TOOLS:
        graph.add_node(tool_obj.name, run_tool)
    
    graph.set_entry_point('orchestrator')
    
    # Edges
    possible_destinations = {t.name: t.name for t in ALL_TOOLS}
    graph.add_conditional_edges('orchestrator', router, possible_destinations)
    
    for tool_obj in ALL_TOOLS:
        if tool_obj.name == 'final_answer':
            graph.add_edge(tool_obj.name, END)
        else:
            graph.add_edge(tool_obj.name, 'orchestrator')
            
    return graph.compile()