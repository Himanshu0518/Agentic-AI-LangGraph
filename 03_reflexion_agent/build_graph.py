from langgraph.graph import StateGraph, END
from langgraph.graph.message import MessagesState
from langchain_core.messages import ToolMessage, HumanMessage
from dotenv import load_dotenv
from chains import first_responder_chain,reviser_chain
from execute_tools import execute_tools
load_dotenv()

MAX_LOOPS = 2

graph = StateGraph(MessagesState)

graph.add_node("first_responder",first_responder_chain )
graph.add_node("tool_executor",execute_tools)
graph.add_node("reviser",reviser_chain)

graph.add_edge("first_responder","tool_executor")
graph.add_edge("tool_executor","reviser")


def should_continue(state: MessagesState):
    iterations = sum(isinstance(msg, ToolMessage) for msg in state["messages"])
    if iterations >=  2*MAX_LOOPS:
        return END
    return "tool_executor"


graph.add_conditional_edges(
    "reviser",
    should_continue,
    {"tool_executor": "tool_executor", END: END}
)
graph.set_entry_point("first_responder")
app = graph.compile()

print(app.get_graph().draw_mermaid())
app.get_graph().print_ascii()

response = app.invoke({"messages": [HumanMessage(content="Write a blog post on Democracy vs dictatorship ?")]})
print(response)