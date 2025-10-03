from dotenv import load_dotenv
from agent_reason_runnable import react_agent_runnable, tools
from react_state import AgentState
from langchain_core.agents import AgentAction, AgentFinish

load_dotenv()

def reason_node(state: AgentState):
    # Ensure intermediate_steps always exists
    intermediate_steps = state.get("intermediate_steps", [])
    
    agent_outcome = react_agent_runnable.invoke({
        "input": state["input"],
        "intermediate_steps": intermediate_steps
    })
    
    if isinstance(agent_outcome, AgentFinish):
        return {"output": agent_outcome.return_values.get("output", "")}
    
    return {"agent_outcome": agent_outcome}


def act_node(state: AgentState):
    agent_action = state["agent_outcome"]
    tool_name = agent_action.tool
    tool_input = agent_action.tool_input

    tool_function = next((t for t in tools if t.name == tool_name), None)

    if tool_function:
        if isinstance(tool_input, dict):
            output = tool_function.invoke(**tool_input)
        else:
            output = tool_function.invoke(tool_input)
    else:
        output = "Tool not found"

    return {
        "intermediate_steps": [(agent_action, str(output))]
    }
