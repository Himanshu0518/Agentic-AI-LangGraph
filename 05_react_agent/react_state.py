from typing import TypedDict, List, Annotated, Union, Tuple
import operator
from langchain_core.agents import AgentAction, AgentFinish

class AgentState(TypedDict):
    input: str
    agent_outcome: Union[AgentAction, AgentFinish]
    # Each step is (AgentAction, observation)
    intermediate_steps: Annotated[List[Tuple[AgentAction, str]], operator.concat]
    output: str
