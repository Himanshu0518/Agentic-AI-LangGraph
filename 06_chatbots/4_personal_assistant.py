from langchain_groq import ChatGroq 
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from typing import TypedDict, List, Annotated
from langgraph.prebuilt import ToolNode
import operator
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langchain_core.tools import tool
import requests
import base64
import os 

load_dotenv()
USERNAME = "Himanshu0518" 
GITHUB_ACCESS_TOKEN = os.getenv("GITHUB_ACCESS_TOKEN")

# State definition
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    skills_cached: bool  # Track if skills have been fetched

# Memory setup
sqlite_conn = sqlite3.connect("checkpoint.sqlite", check_same_thread=False)
memory = SqliteSaver(sqlite_conn)

# Cache for README content
readme_cache = {"content": None, "fetched": False}

@tool("fetch_repos")
def fetch_repos() -> list:
    """
   Fetch all repos along with their details,links and all 
    """
    repos_url = "https://api.github.com/user/repos"
    repos_response = requests.get(repos_url, auth=(USERNAME,GITHUB_ACCESS_TOKEN  ))

    if repos_response.status_code != 200:
        return [{"error": f"Failed to fetch repos: {repos_response.text}"}]

    response = repos_response.json()
    details = []
    for repo in response:
        detail = {
            "repo_name": repo["name"],
            "repo_url": repo["html_url"],
            "repo_description": repo["description"],
            "demo_link":repo['homepage'],
            "topics": repo["topics"],
        }
        details.append(detail)
    return details

# Tool definition
@tool("fetch_skills_and_education")
def fetch_skills_and_education():
    """
    Fetch GitHub profile README.
    All my skills and tech stack are listed here.
    Call this when user asks about skills, education, tech stack, or projects.
    """
    # Return cached content if already fetched
    if readme_cache["fetched"]:
        return readme_cache["content"]
    
    readme_url = f"https://api.github.com/repos/{USERNAME}/{USERNAME}/readme"
    
    try:
        response = requests.get(readme_url, auth=(USERNAME, GITHUB_ACCESS_TOKEN))
        
        if response.status_code != 200:
            return f"Failed to fetch README: {response.text}"
        
        data = response.json()
        readme_content = base64.b64decode(data["content"]).decode("utf-8")
        
        # Cache the content
        readme_cache["content"] = readme_content
        readme_cache["fetched"] = True
        
        return readme_content
    except Exception as e:
        return f"Error fetching skills: {str(e)}"

tools = [fetch_skills_and_education]

# LLM setup
llm = ChatGroq(model="llama-3.1-8b-instant")
llm_with_tools = llm.bind_tools(tools=tools)

# System prompt
SYSTEM_PROMPT = """
You are Himanshu Singh, a BTech student at IIIT Una, Batch of 2027. 
You should ONLY answer questions related to:
- Your background, education, projects, GitHub repos, skills.
- Your achievements, hackathons, or portfolio.
- General greetings and introductions about yourself.

IMPORTANT TOOL USAGE RULES:
1. If you have ALREADY fetched skills/education data in this conversation (check previous messages for tool results), DO NOT call the tool again. Use the information from the previous tool call.
2. ONLY call  tools if:
   - User asks about skills/tech stack/education/projects for the FIRST time
   - You haven't seen any tool results in the conversation history yet
3. The tool data is cached, so refer back to previous tool responses in the conversation.

Speak in first person (use "I", "my", "me") as you ARE Himanshu Singh.

If the user asks about anything NOT related to you (Himanshu Singh), reply with:
"I can only answer questions related to me, Himanshu Singh."
"""

# Router function
def should_continue(state: ChatState):
    """Determine if we should continue to tool node or end"""
    last_message = state["messages"][-1]
    
    # If there are tool calls, route to tool node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # Otherwise end
    return "end"

# Chat node
def chat_node(state: ChatState):
    """Main chat node that calls the LLM"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Tool node
tool_node = ToolNode(tools=tools)

# Build the graph
workflow = StateGraph(ChatState)

# Add nodes
workflow.add_node("agent", chat_node)
workflow.add_node("tools", tool_node)

# Set entry point
workflow.set_entry_point("agent")

# Add conditional edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)

# Add edge from tools back to agent
workflow.add_edge("tools", "agent")

# Compile the graph
app = workflow.compile(checkpointer=memory)

# Configuration for conversation persistence
config = {
    "configurable": {
        "thread_id": "himanshu_chat_001"
    }
}

# Initialize conversation with system prompt
print("Chatbot initialized! Type 'END' to quit.\n")

# Track if this is the first message
first_message = True

while True:
    user_input = input("User: ")
    if user_input.strip().upper() == "END":
        print("Goodbye!")
        break
    
    # Build the input state
    if first_message:
        # Include system prompt on first message
        state = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_input)
            ]
        }
        first_message = False
    else:
        # Just add the user message
        state = {
            "messages": [HumanMessage(content=user_input)]
        }
    
    # Invoke the graph
    try:
        result = app.invoke(state, config=config)
        
        # Get the last AI message
        last_message = result["messages"][-1]
        
        if isinstance(last_message, AIMessage):
            print(f"Bot: {last_message.content}\n")
        else:
            print(f"Bot: {last_message}\n")
            
    except Exception as e:
        print(f"Error: {str(e)}\n")