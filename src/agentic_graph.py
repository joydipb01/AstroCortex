from typing import Annotated, Iterator, Literal, TypedDict

from transformers import Pipeline

from utils import *

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

class ClassifyCostBudget(BaseModel):

    binary_score: str = Field(
        description="List of resources meet the provided budget, 'yes' or 'no'"
    )

class BudgetClassificationParser(BaseOutputParser):
    def parse(Self, text: str) -> ClassifyCostBudget:
        if "yes" in text.lower():
            return ClassifyCostBudget(binary_score = "yes")
        return ClassifyCostBudget(binary_score="no")

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


def plan_resources(state: GraphState, config: GraphConfig):
    llm = config["configurable"]["llm"]
    question = state["question"]
    documents = state["documents"]
    answer_dict = state["answer_dict"]

    resource_plan_prompt = PromptTemplate.from_file('prompts/prompt_resource_plan.txt')

    resource_lister = resource_plan_prompt | llm | StrOutputParser()
    resource_list = resource_lister.invoke({
        "question": question,
        "context": documents
    })

    answer_dict["resource_list"] = resource_list

    return {"answer_dict": answer_dict}

def get_resource_cost(state: GraphState, config: GraphConfig):
    documents = state["documents"]
    question = state["question"]
    answer_dict = state["answer_dict"]
    tavily_search_tool = TavilySearchResults(max_results=3)

    search_questions = get_search_prompts(answer_dict["resource_list"], question)

    for q in search_questions:
        search_results = tavily_search_tool.invoke(q)
        search_content = "\n".join([d["content"] for d in search_results])
        documents.append(Document(page_content = search_content, metadata = {"source": "websearch"}))
    
    return {"documents": documents}

def budget_resources(state: GraphState, config: GraphConfig):
    question = state["question"]
    documents = state["documents"]
    resource_list = state["answer_dict"]["resource_list"]
    llm = config["configurable"]["llm"]

    budget_prompt = PromptTemplate.from_file('prompts/prompt_budgeting_resources.txt')

    budget_classifier = budget_prompt | llm | BudgetClassificationParser()
    budget_classification: ClassifyCostBudget = budget_classifier.invoke(
        {
            "question": question,
            "resources": resource_list,
            "context": documents
        }
    )

    return budget_classification.binary_score

def classify_question(state: GraphState, config: GraphConfig):
    question = state["question"]
    llm = config["configurable"]["llm"]

    question_classifier_prompt = PromptTemplate.from_file('prompts/prompt_question_classifier.txt')

    question_classifier = question_classifier_prompt | llm | QuestionClassificationParser()
    question_classification: ClassifyQuestion = question_classifier.invoke(
        {"question": question}
    )

    return question_classification.question_label

workflow = StateGraph(GraphState, GraphConfig)

workflow.add_node("rephraser", rephrase_question)

graph = workflow.compile()