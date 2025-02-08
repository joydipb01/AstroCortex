from typing import Annotated, Iterator, Literal, TypedDict

from transformers import Pipeline

from langgraph.graph import END, StateGraph, add_messages

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage, convert_to_messages
from langchain_core.prompts.prompt import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.output_parsers import BaseOutputParser

class ClassifyQuestion(BaseModel):
    """Question label to classify question for further processing"""

    question_label: str = Field(
        description="Question is of type 'general' or 'resource budget' or 'comms' or 'misc'"
    )

class QuestionClassificationParser(BaseOutputParser):

    def parse(self, text: str) -> ClassifyQuestion:

        if "general" in text.lower():
            return ClassifyQuestion(question_label = "general")
        
        elif "resource budget" in text.lower():
            return ClassifyQuestion(question_label = "resource budget")
        
        elif "comms" in text.lower():
            return ClassifyQuestion(question_label = "comms")
        
        return ClassifyQuestion(question_label="misc")

class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    documents: list[Document]
    answer_list: list[str]

class GraphConfig(TypedDict):
    retriever: BaseRetriever
    llm: Pipeline

def rephrase_question(state: GraphState, config: GraphConfig):
    question = state["question"]
    llm = config["configurable"]["llm"]

    question_rephraser_prompt = PromptTemplate.from_file('prompts/prompt_question_rephraser.txt')

    question_rephraser = question_rephraser_prompt | llm | StrOutputParser()
    better_question = question_rephraser.invoke({"question": question})

    return {"question": better_question}

def classify_question(state: GraphState, config: GraphConfig):
    question = state["question"]
    llm = config["configurable"]["llm"]

    question_classifier_prompt = PromptTemplate.from_file('prompts/prompt_question_classifier.txt')

    question_classifier = question_classifier_prompt | llm | QuestionClassificationParser()
    question_label: ClassifyQuestion = question_classifier.invoke(
        {"question": question}
    )

workflow = StateGraph(GraphState, GraphConfig)

workflow.add_node()

graph = workflow.compile()