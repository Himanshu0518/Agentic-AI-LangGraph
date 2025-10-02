import datetime
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticToolsParser
from schema import AnswerQuestion, RevisedResponse
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    verbose=True
)

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert AI researcher.
current time : {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximise improvement.
3. After the reflection, **list 1-3 search queries** to improve your answer.
   Do not include them in reflection.
"""
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system","Answer the user's question in required format.")
    ]
)

actor_prompt_template = actor_prompt_template.partial(
    time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    first_instruction="provide 250 word answer to the question"
)

parser = PydanticToolsParser(tools=[AnswerQuestion, RevisedResponse])

first_responder_chain = actor_prompt_template | llm.bind(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
) | (lambda x: {"messages": x}) 


response = first_responder_chain.invoke({
    "messages": [HumanMessage(content="Write a blog post on Democracy vs dictatorship ?")],
})

revise_instructions = """
 Revise your previous response to the user's question using new information.
 - You should use the previous critique to add important details.
    - You must include numerical citations from the previous critique.
    - Add a "Refrences" section which do not count towards word limit give links.
 - You should use previous superflous and missing points to improve your answer.

"""

revisior_prompt_template = actor_prompt_template.partial(
    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    first_instruction=revise_instructions
)

reviser_chain = revisior_prompt_template | llm.bind(tools=[RevisedResponse], tool_choice="RevisedResponse") | (lambda x: {"messages": x})

