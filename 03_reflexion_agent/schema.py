from pydantic import BaseModel, Field
from typing import List

class Reflection(BaseModel):
    missing:str = Field( description="Critique what is missing")
    superflous:str = Field( description="critique what is superflous")

class AnswerQuestion(BaseModel):
    """ Answer the question """
    answer:str = Field( description="~250 word detailed answer")
    search_queries:List[str] = Field( description="1-3 search queries for rsearching improvements to adress the critique to current answer.")
    reflection:Reflection=Field( description="your reflection on the initial answer")

class RevisedResponse(AnswerQuestion):
    citations:List[str] = Field( description="citations for the answer")
   