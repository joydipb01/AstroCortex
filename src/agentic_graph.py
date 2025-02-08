from typing import Annotated, Iterator, Literal, TypedDict

from transformers import Pipeline

from langgraph.graph import START, END, StateGraph, MessagesState

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults
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

class GraphState(MessagesState):
    question: str
    documents: list[Document]
    answer_dict: dict

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

def get_relevant_documents(state: GraphState, config: GraphConfig):
    retriever = config["configurable"]["retriever"]
    question = state["question"]

    relevant_documents = retriever.invoke(question)
    
    return {"documents": relevant_documents}


def plan_resources_and_cost(state: GraphState, config: GraphConfig):
    llm = config["configurable"]["llm"]
    question = state["question"]
    documents = state["documents"]
    answer_dict = state["answer_dict"]

    resource_cost_plan_prompt = PromptTemplate.from_file('prompts/prompt_resource_cost_plan.txt')

    resource_lister = resource_cost_plan_prompt | llm | StrOutputParser()
    resource_list = resource_lister.invoke({
        "question": question,
        "context": documents
    })

    answer_dict["resource_list"] = resource_list

def classify_question(state: GraphState, config: GraphConfig):
    question = state["question"]
    llm = config["configurable"]["llm"]

    question_classifier_prompt = PromptTemplate.from_file('prompts/prompt_question_classifier.txt')

    question_classifier = question_classifier_prompt | llm | QuestionClassificationParser()
    question_label: ClassifyQuestion = question_classifier.invoke(
        {"question": question}
    )

workflow = StateGraph(GraphState, GraphConfig)

workflow.add_node("rephraser", rephrase_question)

graph = workflow.compile()