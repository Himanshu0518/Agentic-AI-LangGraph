from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from typing import TypedDict,List,Annotated
import operator

# load_dotenv()

class SimpleState(TypedDict):
    count: int  
    sum : Annotated[int,operator.add]
    history:Annotated[ List[int], operator.concat]

def increment(state: SimpleState) -> SimpleState:
    new_count =   state["count"] + 1
  
    return {
        "count": new_count,
        "sum":  new_count,
        "history": [new_count] ,
        }

def should_continue(state: SimpleState):
    if state["count"] >= 5:
        return END
    return "increment"

graph = StateGraph(SimpleState)

graph.add_node("increment", increment)

graph.set_entry_point("increment")

graph.add_conditional_edges(
    "increment",
    should_continue,
    {"increment": "increment", END: END}
)

app = graph.compile()

print(app.get_graph().draw_mermaid())

state = {"count": 0,"sum":0,"history":[0]}
response = app.invoke(state)
print(response)

