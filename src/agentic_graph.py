from langgraph.graph import END, StateGraph, add_messages

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, AIMessage, convert_to_messages
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.output_parsers import BaseOutputParser

class ClassifyQuestion(BaseModel):
    """Question label to classify question for further processing"""

    question_label: str = Field(
        description="Question is of type 'general' or 'resource budget' or 'plan'"
    )

class QuestionClassificationParser(BaseOutputParser):

    def parse(self, text: str) -> ClassifyQuestion:

        if "general" in text.lower():
            return ClassifyQuestion(question_label = "general")
        
        elif "resource budget" in text.lower():
            return ClassifyQuestion(question_label = "resource budget")
        
        return ClassifyQuestion(question_label = "plan")