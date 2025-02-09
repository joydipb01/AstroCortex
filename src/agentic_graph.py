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

# Data Models:

class ClassifyQuestion(BaseModel):
    """Question label to classify question for further processing"""

    question_label: str = Field(
        description="Question is of type 'general' or 'resource budget' or 'misc'"
    )

class QuestionClassificationParser(BaseOutputParser):

    def parse(self, text: str) -> ClassifyQuestion:

        if "general" in text.lower():
            return ClassifyQuestion(question_label = "general")
        
        elif "resource budget" in text.lower():
            return ClassifyQuestion(question_label = "resource budget")
        
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

# Graph Structure:

class GraphState(MessagesState):
    question: str
    question_class: str
    documents: list[Document]
    answer_dict: dict
    final_answer: str

class GraphConfig(TypedDict):
    retriever: BaseRetriever
    llm: Pipeline

# Graph Nodes

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

def classify_question(state: GraphState, config: GraphConfig):
    question = state["question"]
    llm = config["configurable"]["llm"]

    question_classifier_prompt = PromptTemplate.from_file('prompts/prompt_question_classifier.txt')

    question_classifier = question_classifier_prompt | llm | QuestionClassificationParser()
    question_classification: ClassifyQuestion = question_classifier.invoke(
        {"question": question}
    )

    return {"question_class": question_classification.question_label}

def plan_mission(state: GraphState, config: GraphConfig):
    llm = config["configurable"]["llm"]
    question = state["question"]
    documents = state["documents"]
    answer_dict = state["answer_dict"]

    mission_plan_prompt = PromptTemplate.from_file("prompt/prompt_mission_plan.txt")

    mission_planner = mission_plan_prompt | llm | StrOutputParser()
    mission_plan = mission_planner.invoke(
        {
            "question": question,
            "resources": answer_dict["resource_list"],
            "context": documents
        }
    )

    answer_dict["plan"] = mission_plan

    return {"answer_dict": answer_dict}

def plan_resources(state: GraphState, config: GraphConfig):
    llm = config["configurable"]["llm"]
    question = state["question"]
    documents = state["documents"]
    answer_dict = state["answer_dict"]

    if "resource_list" not in answer_dict.keys():
        resource_plan_prompt = PromptTemplate.from_file('prompts/prompt_resource_plan.txt')

        resource_lister = resource_plan_prompt | llm | StrOutputParser()
        resource_list = resource_lister.invoke({
            "question": question,
            "context": documents
        })
    
    else:
        resource_budget_plan_prompt = PromptTemplate.from_file('prompts/prompt_resource_budget_plan.txt')

        resource_lister_ii = resource_budget_plan_prompt | llm | StrOutputParser()
        resource_list = resource_lister_ii.invoke({
            "question": question,
            "mission": answer_dict["plan"],
            "resources": answer_dict["resource_list"],
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

def finalize_answer(state: GraphState, config: GraphConfig):
    question = state["question"]
    answer_dict = state["answer_dict"]
    documents = state["documents"]
    llm = config["configurable"]["llm"]

    finalizer_prompt = PromptTemplate.from_file('prompts/prompt_response_finalizer.txt')

    response_finalizer = finalizer_prompt | llm | StrOutputParser()
    answer = response_finalizer.invoke({
        "question": question,
        "resources": answer_dict["resource_list"],
        "context": documents
    })

    return {"messages": [AIMessage(content=answer)], "final_answer": answer}

# Conditional Edges:

def budget_resources(state: GraphState, config: GraphConfig):
    question = state["question"]
    documents = state["documents"]
    question_class = state["question_class"]
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

    if budget_classification.binary_score == "yes":
        if question_class == "general":
            return "yes,gen"
        if question_class == "resource_budget":
            return "yes,res"

    return budget_classification.binary_score

def get_question_class(state: GraphState):
    return state["question_class"]


workflow = StateGraph(GraphState, GraphConfig)

workflow.add_node("rephraser", rephrase_question)
workflow.add_node("retriever", get_relevant_documents)
workflow.add_node("classify_question", classify_question)
workflow.add_node("resource_planner", plan_resources)
workflow.add_node("resource_costing", get_resource_cost)
workflow.add_node("mission_planner", plan_mission)
workflow.add_node("finalizer", finalize_answer)

workflow.add_edge(START, "rephraser")
workflow.add_edge("rephraser", "retriever")
workflow.add_edge("retriever", "classify_question")
workflow.add_edge("classify_question", "plan_resources")
workflow.add_edge("plan_resources", "get_resource_cost")
workflow.add_edge("finalizer", END)

workflow.add_conditional_edge(
    "get_resource_cost", budget_resources, {"no": "plan_resources", "yes,gen": "plan_mission", "yes,res": "finalizer"}
)

graph = workflow.compile()