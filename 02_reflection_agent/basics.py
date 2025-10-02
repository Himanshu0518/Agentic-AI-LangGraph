from dotenv import load_dotenv
from chains import generate_chain, reflection_chain
from langgraph.graph import StateGraph, END
from langgraph.graph.message import MessagesState
from langchain_core.messages import HumanMessage

load_dotenv()

# ---- Configurable loops ----
MAX_LOOPS = 2   # how many reflection cycles you want

# ---- Graph setup ----
graph = StateGraph(MessagesState)

REFLECT = "reflect"
GENERATE = "generate"

# ---- Nodes ----
def generate_node(state: MessagesState):
    response = generate_chain.invoke({"messages": state["messages"]})
    return {"messages": [response]}  

def reflect_node(state: MessagesState):
    response = reflection_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=response.content)]}   

graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)

graph.set_entry_point(GENERATE)

# ---- Conditional edges ----
def should_continue(state: MessagesState):
    if len(state["messages"]) >= 2 * MAX_LOOPS:
        return END
    return REFLECT

graph.add_conditional_edges(
    GENERATE,
    should_continue,
    {REFLECT: REFLECT, END: END}
)
graph.add_edge(REFLECT, GENERATE)

# ---- Compile and visualize ----
app = graph.compile()

print(app.get_graph().draw_mermaid())
app.get_graph().print_ascii()

# ---- Test run ----
# ---- Test run ----
response = app.invoke({"messages": [HumanMessage(content="AI is curse or bless?")]})
print("\nFinal messages:")
for msg in response["messages"]:
    print(f"{msg.type}: {msg.content}")

