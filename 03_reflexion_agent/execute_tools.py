import json 
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage,AIMessage,HumanMessage,ToolMessage
from langchain_community.tools import TavilySearchResults
import os 

tavily_tool = TavilySearchResults(
    tavily_api_key=os.getenv("TAVILY_API_KEY"),
     max_results=3
)

def execute_tools(state: Dict[str, Any]) -> Dict[str, List[BaseMessage]]:
    messages = state["messages"]  
    if not messages:
        return {"messages": []}

    last_ai_message = messages[-1]
    if not hasattr(last_ai_message, "tool_calls") or not last_ai_message.tool_calls:
        return {"messages": []}

    tool_messages = []

    for tool_call in last_ai_message.tool_calls:
        if tool_call["name"] in ["AnswerQuestion", "RevisedResponse"]:
            call_id = tool_call["id"]
            search_queries = tool_call["args"]["search_queries"]

            query_results = {}

            for query in search_queries:
                results = tavily_tool.run(query)
                query_results[query] = results

            tool_messages.append(ToolMessage(
                name=tool_call["name"],
                content=json.dumps(query_results),
                tool_call_id=call_id
            ))

    return {"messages": tool_messages}
