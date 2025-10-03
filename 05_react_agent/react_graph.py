from nodes import reason_node,act_node
from react_state import AgentState
from langgraph.graph import StateGraph, END
from langchain_core.agents import AgentFinish

def should_continue(state: AgentState):
    if isinstance(state["agent_outcome"],AgentFinish):
        return END
    return "act_node"

graph = StateGraph(AgentState)

graph.add_node("reason_node",reason_node)
graph.add_node("act_node",act_node)

graph.set_entry_point("reason_node")

graph.add_conditional_edges(
    "reason_node",
    should_continue,
    {"act_node": "act_node", END: END}
)

app = graph.compile()

print(app.get_graph().draw_mermaid())

result = app.invoke({"input": "give summary of latest india vs pakistan mens cricket match ?"})
print(result)