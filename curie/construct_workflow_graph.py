# Define overall graph
from typing import Annotated, Literal

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver

import model
import utils
import tool
import exp_agent
import worker_agent
import settings
import scheduler as sched
import verifier

# import router

import sys
import traceback
import json

config_filename = sys.argv[1]

# Read config_file which is a json file:
with open(config_filename, 'r') as file:
    config = json.load(file)
    question_filename = config["question_filename"]

    log_filename = config["log_filename"]
    log_file = open(log_filename, 'w')
    sys.stdout = log_file
    sys.stderr = log_file

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    prev_agent: Literal[*settings.AGENT_LIST]
    next_agent: Literal[*settings.AGENT_LIST]
    is_terminate: bool

graph_builder = StateGraph(State)

store = InMemoryStore()
metadata_store = InMemoryStore()
memory = MemorySaver()

# Create scheduler:
sched_tool = sched.SchedTool(store, metadata_store, config)
sched_node = sched.create_SchedNode(sched_tool, State, metadata_store)
graph_builder.add_node("scheduler", sched_node)
# Create supervisor graph:
supervisor_graph = exp_agent.create_ExpSupervisorGraph(State, store, metadata_store, memory, config_filename)
graph_builder.add_node("supervisor", supervisor_graph)
# Create worker graph:
experiment_worker_graph, control_worker_graph = worker_agent.create_all_worker_graphs(
    State, 
    store, 
    metadata_store, 
    memory,
    config_filename = config_filename)
worker_names = settings.list_worker_names() # we only have one worker for now
experimental_worker_name = worker_names[0]
worker_names = settings.list_control_worker_names() # we only have one worker for now
control_worker_name = worker_names[0]
graph_builder.add_node(experimental_worker_name, experiment_worker_graph)
graph_builder.add_node(control_worker_name, control_worker_graph)
# Create LLM verifier graph:
verifier_graph = verifier.create_LLMVerifierGraph(State, store, metadata_store)
graph_builder.add_node("llm_verifier", verifier_graph)
# Create LLM patcher graph:
verifier_graph = verifier.create_PatchVerifierGraph(State, store, metadata_store)
graph_builder.add_node("patch_verifier", verifier_graph)
# Create Analyzer graph:
verifier_graph = verifier.create_AnalyzerGraph(State, store, metadata_store)
graph_builder.add_node("analyzer", verifier_graph)
# Create Concluder graph:
verifier_graph = verifier.create_ConcluderGraph(State, store, metadata_store)
graph_builder.add_node("concluder", verifier_graph)
# Add edges: (we only need to define the higher level graph edges, the internal graphs have edges taken care of already)
graph_builder.add_edge(START, "supervisor") # start edge: we always start from the supervisor who will create some kind of plan based on user input
graph_builder.add_edge("supervisor", "scheduler") # supervisor will always call the scheduler to determine the next agent
graph_builder.add_edge(experimental_worker_name, "scheduler") # worker will always call the scheduler to determine the next agent
graph_builder.add_edge(control_worker_name, "scheduler") # worker will always call the scheduler to determine the next agent
graph_builder.add_edge("llm_verifier", "scheduler") # verifier will always call the scheduler to determine the next agent
graph_builder.add_edge("patch_verifier", "scheduler") # verifier will always call the scheduler to determine the next agent
graph_builder.add_edge("analyzer", "scheduler") # verifier will always call the scheduler to determine the next agent
graph_builder.add_edge("concluder", "scheduler") # verifier will always call the scheduler to determine the next agent
graph_builder.add_conditional_edges("scheduler", lambda state: state["next_agent"]) # Inspired partly from: https://langchain-ai.github.io/langgraph/tutorials/multi_agent/hierarchical_agent_teams/#add-layers

graph = graph_builder.compile(checkpointer=memory)

utils.save_langgraph_graph(graph, "misc/overall_graph_image.png") 

# Section: Run agent
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)], "is_terminate": False}, {"recursion_limit": 200, "configurable": {"thread_id": "main_graph_id"}}):
        print("Event:", event)
        for value in event.values():
            print("Event value:", value["messages"][-1].content)
        print("--------------------------------------------------")

def get_question(question_file_path: str):
    question = ""
    with open(question_file_path, "r") as question_file:
        for line in question_file:
            question += line.strip() + "\n"
    return question

while True:
    try:
        # user_input = input("User: ")
        # Read from question file to user_input:
        user_input = get_question(question_filename)

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        # Save question to long term store:
        user_id = "admin"
        application_context = "exp-sched"
        sched_namespace = (user_id, application_context)
        metadata_store.put(sched_namespace, "question", user_input)

        stream_graph_updates(user_input)
        break
    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
        break