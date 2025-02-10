import sys
import json
import traceback
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage

# Local module imports
import model
import utils
import tool
import exp_agent
import worker_agent
import settings
import scheduler as sched
import verifier

if len(sys.argv) < 2:
    print("Usage: python script.py <config_file>")
    sys.exit(1)

config_filename = sys.argv[1]

# Read config file
with open(config_filename, 'r') as file:
    config = json.load(file)
    question_filename = f"../{config['question_filename']}"
    log_filename = f"../{config['log_filename']}"
    log_file = open(log_filename, 'w')
    sys.stdout = log_file
    sys.stderr = log_file

class State(TypedDict):
    messages: Annotated[list, add_messages]
    prev_agent: Literal[*settings.AGENT_LIST]
    next_agent: Literal[*settings.AGENT_LIST]
    is_terminate: bool 

def setup_logging(log_filename: str):
    """
    Configure logging to redirect stdout and stderr to a log file.
    
    Args:
        log_filename (str): Path to the log file
    """
    log_file = open(log_filename, 'w')
    sys.stdout = log_file
    sys.stderr = log_file

def create_graph_stores():
    """
    Create stores and memory for the graph.
    
    Returns:
        tuple: Stores and memory objects
    """
    store = InMemoryStore()
    metadata_store = InMemoryStore()
    memory = MemorySaver()
    return store, metadata_store, memory

def create_scheduler_node(store, metadata_store, config, State):
    """
    Create scheduler node for the graph.
    
    Args:
        store: InMemoryStore for data storage
        metadata_store: InMemoryStore for metadata
        config: Configuration dictionary
        State: Graph state type
    
    Returns:
        Node for the scheduler
    """
    sched_tool = sched.SchedTool(store, metadata_store, config)
    return sched.create_SchedNode(sched_tool, State, metadata_store)

def create_worker_nodes(State, store, metadata_store, memory, config_filename):
    """
    Create worker nodes for the graph.
    
    Args:
        State: Graph state type
        store: InMemoryStore for data storage
        metadata_store: InMemoryStore for metadata
        memory: MemorySaver for checkpointing
        config_filename: Path to configuration file
    
    Returns:
        tuple: Experimental and control worker nodes with their names
    """
    experiment_worker_graph, control_worker_graph = worker_agent.create_all_worker_graphs(
        State, 
        store, 
        metadata_store, 
        memory,
        config_filename=config_filename
    )
    
    worker_names = settings.list_worker_names()
    experimental_worker_name = worker_names[0]
    
    worker_names = settings.list_control_worker_names()
    control_worker_name = worker_names[0]
    
    return (experimental_worker_name, experiment_worker_graph), \
           (control_worker_name, control_worker_graph)

def create_verification_nodes(State, store, metadata_store):
    """
    Create verification nodes for the graph.
    
    Args:
        State: Graph state type
        store: InMemoryStore for data storage
        metadata_store: InMemoryStore for metadata
    
    Returns:
        list: Verification nodes with their names
    """
    return [
        ("llm_verifier", verifier.create_LLMVerifierGraph(State, store, metadata_store)),
        ("patch_verifier", verifier.create_PatchVerifierGraph(State, store, metadata_store)),
        ("analyzer", verifier.create_AnalyzerGraph(State, store, metadata_store)),
        ("concluder", verifier.create_ConcluderGraph(State, store, metadata_store))
    ]

def build_graph(State, config_filename):
    """
    Build the complete LangGraph workflow.
    
    Args:
        State: Graph state type
        config_filename: Path to configuration file
    
    Returns:
        Compiled graph
    """
    # Read configuration
    with open(config_filename, 'r') as file:
        config = json.load(file)
    
    # Setup logging
    # setup_logging(f"../{config['log_filename']}")
    
    # Create stores
    store, metadata_store, memory = create_graph_stores()
    
    # Create graph builder
    graph_builder = StateGraph(State)
    
    # Add scheduler node
    sched_node = create_scheduler_node(store, metadata_store, config, State)
    graph_builder.add_node("scheduler", sched_node)
    
    # Add supervisor node
    supervisor_graph = exp_agent.create_ExpSupervisorGraph(State, store, metadata_store, memory, config_filename)
    graph_builder.add_node("supervisor", supervisor_graph)
    
    # Add worker nodes
    experimental_worker = create_worker_nodes(State, store, metadata_store, memory, config_filename)[0]
    control_worker = create_worker_nodes(State, store, metadata_store, memory, config_filename)[1]
    graph_builder.add_node(experimental_worker[0], experimental_worker[1])
    graph_builder.add_node(control_worker[0], control_worker[1])
    
    # Add verification nodes
    verification_nodes = create_verification_nodes(State, store, metadata_store)
    for name, node in verification_nodes:
        graph_builder.add_node(name, node)
    
    # Add graph edges
    graph_builder.add_edge(START, "supervisor")
    graph_builder.add_edge("supervisor", "scheduler")
    graph_builder.add_edge(experimental_worker[0], "scheduler")
    graph_builder.add_edge(control_worker[0], "scheduler")
    
    for name, _ in verification_nodes:
        graph_builder.add_edge(name, "scheduler")
    
    graph_builder.add_conditional_edges("scheduler", lambda state: state["next_agent"])
    
    # Compile and visualize graph
    graph = graph_builder.compile(checkpointer=memory)
    utils.save_langgraph_graph(graph, "../logs/misc/overall_graph_image.png")
    
    return graph, metadata_store, config

def get_question(question_file_path: str) -> str:
    """
    Read question from a file.

    Args:
        question_file_path (str): Path to the question file

    Returns:
        str: Question text
    """
    with open(question_file_path, "r") as question_file:
        return question_file.read().strip()

def stream_graph_updates(graph, user_input: str):
    """
    Stream graph updates during workflow execution.

    Args:
        graph: Compiled LangGraph workflow
        user_input (str): User's input question
    """
    for event in graph.stream(
        {"messages": [("user", user_input)], "is_terminate": False}, 
        {"recursion_limit": 200, "configurable": {"thread_id": "main_graph_id"}}
    ):
        print("Event:", event)
        for value in event.values():
            print("Event value:", value["messages"][-1].content)
        print("--------------------------------------------------")

def main():
    """
    Main execution function for the LangGraph workflow.
    """
    if len(sys.argv) < 2:
        print("Usage: python script.py <config_file>")
        sys.exit(1)

    config_filename = sys.argv[1]

    class State(TypedDict):
        messages: Annotated[list, add_messages]
        prev_agent: Literal[*settings.AGENT_LIST]
        next_agent: Literal[*settings.AGENT_LIST]
        is_terminate: bool

    try:
        # Build graph
        graph, metadata_store, config = build_graph(State, config_filename)
        
        # Read question from file
        question_filename = f"../{config['question_filename']}"
        user_input = get_question(question_filename) 
        sched_namespace = ("admin", "exp-sched")
        metadata_store.put(sched_namespace, "question", user_input)

        # Stream graph updates
        stream_graph_updates(graph, user_input)

    except Exception as e:
        print(f"Execution error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()