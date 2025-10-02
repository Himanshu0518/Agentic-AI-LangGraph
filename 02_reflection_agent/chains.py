from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    verbose=True
)
generate_prompt = ChatPromptTemplate.from_messages(
    [
       (
            "system",
            """You are a helpful tweet writer. Generate 
            the best possible tweet which goes viral. 
            If user provides critique, respond with a revised version of your previous attempts."""
        ),
        MessagesPlaceholder(variable_name="messages"),  
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
      ("system",
       """You are a viral twitter influencer grading a tweet. 
       Generate critique and recommendations for the user's tweet. 
       Always provide detailed recommendations, including requests for length, virality,
       style, etc."""
      ),
      MessagesPlaceholder(variable_name="messages"),  
    ]   
)


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
)

generate_chain = generate_prompt|llm

reflection_chain = reflection_prompt|llm
