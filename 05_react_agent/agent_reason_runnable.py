from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import TavilySearchResults
from langchain import hub 
from langchain.agents import create_react_agent 
load_dotenv()

# Tavily Search tool
search = TavilySearchResults(
    tavily_api_key=os.getenv("TAVILY_API_KEY"),
    search_depth="advanced",
    verbose=True
)

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    verbose=True
)

tools = [search] 

react_prompt = hub.pull("hwchase17/react")
react_agent_runnable = create_react_agent(tools=tools,llm=llm,prompt=react_prompt)