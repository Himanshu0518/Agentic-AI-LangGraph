from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import TavilySearchResults
from langchain.agents import initialize_agent

load_dotenv()

# Tavily Search tool
search = TavilySearchResults(
    tavily_api_key=os.getenv("TAVILY_API_KEY"),
    search_depth="advanced",
    verbose=True
)

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    verbose=True
)

# Create an agent that can use tools
agent = initialize_agent(
    tools=[search],
    llm=llm,
    agent='zero-shot-react-description',
    verbose=True
)

# Ask a question that requires both generation & search
query = "Write a 2 line poem about cricket and also tell me the latest Asia Cup 2025 men's cricket schedule."
result = agent.run(query)

print("\nFinal Answer:\n", result)
