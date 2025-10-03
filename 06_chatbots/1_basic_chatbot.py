from langchain_groq import ChatGroq 
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage,AIMessage,BaseMessage
from typing import TypedDict,List,Annotated
import operator


load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant",)

class BasicChatBotState(TypedDict):
    messages: Annotated[List,operator.concat]

def chatbot(state: BasicChatBotState) -> BaseMessage:
    
    response = llm.invoke(state["messages"])
    return {"messages":[response]}


graph = StateGraph(BasicChatBotState)

graph.add_node("chatbot",chatbot)

graph.set_entry_point("chatbot")

app = graph.compile()




while(True):
    user_input = input("User: ")
    if(user_input == "END"):
        break
    state = {"messages":[HumanMessage(content=user_input)]}
    response = app.invoke(state)
    print(f"Bot: {response}")
    

  

