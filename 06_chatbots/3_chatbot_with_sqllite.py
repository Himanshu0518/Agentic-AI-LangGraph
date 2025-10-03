from langchain_groq import ChatGroq 
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing import TypedDict, List, Annotated
from langchain_community.tools import TavilySearchResults
from langgraph.prebuilt import ToolNode
import operator
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

load_dotenv()

sqlite_conn = sqlite3.connect("checkpoint.sqlite", check_same_thread=False)
memory = SqliteSaver(sqlite_conn)

# Define State properly
class BasicChatBotState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.concat]

# Tavily Search tool
search_tool = TavilySearchResults()
tools = [search_tool]

# LLM with tool binding
llm = ChatGroq(model="llama-3.1-8b-instant")
llm = llm.bind_tools(tools=tools)

def chat_node(state: BasicChatBotState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def tool_router(state: BasicChatBotState):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"
    return END

tool_node = ToolNode(tools=tools, messages_key="messages")

# Build graph
graph = StateGraph(BasicChatBotState)
graph.add_node("chat_node", chat_node)
graph.add_node("tool_node", tool_node)

graph.add_edge("tool_node", "chat_node")
graph.add_conditional_edges(
    "chat_node",
    tool_router,
    {"tool_node": "tool_node", END: END}
)
graph.set_entry_point("chat_node")

app = graph.compile(checkpointer=memory)
config = {
    "configurable": {
        "thread_id": "1"
    }
}

# Interactive loop
while True:
    user_input = input("User: ")
    if user_input.strip().upper() == "END":
        break
    
    state = {"messages": [HumanMessage(content=user_input)]}
    response = app.invoke(state, config=config)

    print(f"Bot: {response}")
