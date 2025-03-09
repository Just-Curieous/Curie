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
from langgraph.managed.is_last_step import RemainingSteps

# Local module imports
import model
import utils
import tool
import settings
import scheduler as sched

from logger import init_logger
from model import setup_model_logging
from tool import setup_tool_logging
from nodes.exec_validator import setup_exec_validator_logging

from nodes.architect import Architect
from nodes.technician import Technician
from nodes.base_node import NodeConfig
from nodes.llm_validator import LLMValidator
from nodes.patcher import Patcher
from nodes.analyzer import Analyzer
from nodes.concluder import Concluder
from reporter import generate_report

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
    
    curie_logger = init_logger(log_filename)
    setup_model_logging(log_filename)
    setup_exec_validator_logging(log_filename)
    setup_tool_logging(log_filename)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    prev_agent: Literal[*settings.AGENT_LIST]
    next_agent: Literal[*settings.AGENT_LIST]
    is_terminate: bool 
    remaining_steps: RemainingSteps
    remaining_steps_display: int # remaining_steps cannot be seen in event.values since it is an Annotated value managed by RemainingStepsManager https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/managed/is_last_step.py

class AllNodes():
    def __init__(self, log_filename: str, config_filename: str, state: State, store, metadata_store, memory):
        self.nodes = {}
        self.State = State
        self.log_filename = log_filename
        self.store = store
        self.metadata_store = metadata_store
        self.memory = memory
        self.config_filename = config_filename
        self.instantiate_nodes()
        self.instantiate_subgraphs()
    
    def instantiate_nodes(self):
        with open(self.config_filename, 'r') as file:
            config_dict = json.load(file)

        # Create scheduler node:
        self.sched_node = self.create_sched_node(config_dict) # always create sched node first

        # Create other nodes, passing in self.sched_node as a param:
        self.architect = self.create_architect_node()
        worker, control_worker = self.create_worker_nodes()
        self.workers = [worker]
        self.control_workers = [control_worker]
        self.validators = self.create_validators() # list of validators

        # Create sched tool, passing in other agent's transition funcs as a dict
        config_dict["transition_funcs"] = {
            "supervisor": lambda: self.architect.transition_handle_func(),
            "worker": lambda: self.workers[0].transition_handle_func(),
            "control_worker": lambda: self.control_workers[0].transition_handle_func(),
            "llm_verifier": lambda: self.validators[0].transition_handle_func(),
            "patch_verifier": lambda: self.validators[1].transition_handle_func(),
            "analyzer": lambda: self.validators[2].transition_handle_func(),
            "concluder": lambda state: self.validators[3].transition_handle_func(state)
        }
        self.sched_tool = sched.SchedTool(self.store, self.metadata_store, config_dict)

    def instantiate_subgraphs(self):
        self.sched_subgraph = self.sched_node.create_SchedNode_subgraph(self.sched_tool)
        self.architect_subgraph = self.architect.create_subgraph()
        self.worker_subgraph = self.workers[0].create_subgraph()
        self.control_worker_subgraph = self.control_workers[0].create_subgraph()
        self.validator_subgraphs = [validator.create_subgraph() for validator in self.validators]
    
    def get_sched_subgraph(self):
        return self.sched_subgraph
    
    def get_architect_subgraph(self):
        return self.architect_subgraph
    
    def get_worker_subgraphs(self):
        return self.worker_subgraph, self.control_worker_subgraph
    
    def get_validator_subgraphs(self):
        return self.validator_subgraphs

    def get_architect_node(self):
        return self.architect

    def get_worker_node(self):
        return self.workers[0]
    
    def get_control_worker_node(self):
        return self.control_workers[0]
    
    def get_validator_nodes(self):
        return self.validators

    def create_sched_node(self, config_dict):
        return sched.SchedNode(self.store, self.metadata_store, self.State, config_dict)

    def create_architect_node(self):
        # Customizable node config 
        node_config = NodeConfig(
            name="supervisor",
            node_icon="👑",
            log_filename=self.log_filename, 
            config_filename=self.config_filename,
            system_prompt_key="supervisor_system_prompt_filename",
            default_system_prompt_filename="prompts/exp-supervisor.txt"
        )
        # Customizable tools
        store_write_tool = tool.NewExpPlanStoreWriteTool(self.store, self.metadata_store)
        redo_write_tool = tool.RedoExpPartitionTool(self.store, self.metadata_store)
        store_get_tool = tool.StoreGetTool(self.store)
        edit_priority_tool = tool.EditExpPriorityTool(self.store, self.metadata_store)
        tools = [store_write_tool, edit_priority_tool, redo_write_tool, store_get_tool, tool.read_file_contents]

        return Architect(self.sched_node, node_config, self.State, self.store, self.metadata_store, self.memory, tools)

    def create_worker_nodes(self):
        # Create common tools:
        # Customizable tools
        store_write_tool = tool.ExpPlanCompletedWriteTool(self.store, self.metadata_store)
        store_get_tool = tool.StoreGetTool(self.store)
        with open(self.config_filename, 'r') as file:
            config_dict = json.load(file) 
        codeagent_openhands = tool.CodeAgentTool(config_dict)
        tools = [codeagent_openhands, tool.execute_shell_command, store_write_tool, store_get_tool]

        # Create 1 worker: 
        # Customizable node config 
        worker_names = settings.list_worker_names()
        assert len(worker_names) == 1
        node_config = NodeConfig(
            name=worker_names[0],
            node_icon="👷",
            log_filename=self.log_filename, 
            config_filename=self.config_filename,
            system_prompt_key="worker_system_prompt_filename",
            default_system_prompt_filename="prompts/exp-worker.txt"
        )

        worker = Technician(self.sched_node, node_config, self.State, self.store, self.metadata_store, self.memory, tools)

        # Create 1 control worker: 
        # Customizable node config 
        worker_names = settings.list_control_worker_names()
        assert len(worker_names) == 1
        node_config = NodeConfig(
            name=worker_names[0],
            node_icon="👷",
            log_filename=self.log_filename, 
            config_filename=self.config_filename,
            system_prompt_key="control_worker_system_prompt_filename",
            default_system_prompt_filename="prompts/controlled-worker.txt"
        )

        control_worker = Technician(self.sched_node, node_config, self.State, self.store, self.metadata_store, self.memory, tools)

        return worker, control_worker

    def create_validators(self):
        # Create LLM validator: 
        # Customizable node config 
        node_config = NodeConfig(
            name="llm_verifier",
            node_icon="✅",
            log_filename=self.log_filename, 
            config_filename=self.config_filename,
            system_prompt_key="llm_verifier_system_prompt_filename",
            default_system_prompt_filename="prompts/llm-verifier.txt"
        )

        # Customizable tools
        verifier_write_tool = tool.LLMVerifierWriteTool(self.store, self.metadata_store)
        store_get_tool = tool.StoreGetTool(self.store)
        tools = [tool.execute_shell_command, store_get_tool, verifier_write_tool]
        
        llm_validator = LLMValidator(self.sched_node, node_config, self.State, self.store, self.metadata_store, self.memory, tools)

        # Create Patcher: 
        # Customizable node config 
        node_config = NodeConfig(
            name="patch_verifier",
            node_icon="✅",
            log_filename=self.log_filename, 
            config_filename=self.config_filename,
            system_prompt_key="patcher_system_prompt_filename",
            default_system_prompt_filename="prompts/exp-patcher.txt"
        )

        # Customizable tools
        patcher_record_tool = tool.PatchVerifierWriteTool(self.store, self.metadata_store)
        with open(self.config_filename, 'r') as file:
            config_dict = json.load(file) 
        patch_agent_tool = tool.PatcherAgentTool(config_dict)
        store_get_tool = tool.StoreGetTool(self.store)
        tools = [patch_agent_tool, tool.execute_shell_command, patcher_record_tool, store_get_tool] 
        
        patcher = Patcher(self.sched_node, node_config, self.State, self.store, self.metadata_store, self.memory, tools)

        # Create Analyer: 
        # Customizable node config 
        node_config = NodeConfig(
            name="analyzer",
            node_icon="✅",
            log_filename=self.log_filename, 
            config_filename=self.config_filename,
            system_prompt_key="analyzer_system_prompt_filename",
            default_system_prompt_filename="prompts/exp-analyzer.txt"
        )

        # Customizable tools
        patcher_record_tool = tool.AnalyzerWriteTool(self.store, self.metadata_store)
        store_get_tool = tool.StoreGetTool(self.store)
        tools = [tool.read_file_contents, patcher_record_tool, store_get_tool]
        
        analyzer = Analyzer(self.sched_node, node_config, self.State, self.store, self.metadata_store, self.memory, tools)

        # Create Concluder: 
        # Customizable node config 
        node_config = NodeConfig(
            name="concluder",
            node_icon="✅",
            log_filename=self.log_filename, 
            config_filename=self.config_filename,
            system_prompt_key="concluder_system_prompt_filename",
            default_system_prompt_filename="prompts/exp-concluder.txt"
        )

        # Customizable tools
        patcher_record_tool = tool.ConcluderWriteTool(self.store, self.metadata_store)
        store_get_tool = tool.StoreGetTool(self.store)
        tools = [tool.read_file_contents, patcher_record_tool, store_get_tool] # Only tool is code execution for now
        
        concluder = Concluder(self.sched_node, node_config, self.State, self.store, self.metadata_store, self.memory, tools)

        return [llm_validator, patcher, analyzer, concluder]

    def get_all_nodes(self):
        return self.nodes
 
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

    all_nodes = AllNodes(log_filename, config_filename, State, store, metadata_store, memory)

    
    # Add scheduler node
    sched_subgraph = all_nodes.get_sched_subgraph()
    graph_builder.add_node("scheduler", sched_subgraph)
    
    # Add supervisor node
    supervisor_graph = all_nodes.get_architect_subgraph()
    supervisor_name = all_nodes.get_architect_node().get_name()
    graph_builder.add_node(supervisor_name, supervisor_graph)
    
    # Add worker nodes
    experimental_worker, control_worker = all_nodes.get_worker_subgraphs()
    experimental_worker_name = all_nodes.get_worker_node().get_name()
    control_worker_name = all_nodes.get_control_worker_node().get_name()

    graph_builder.add_node(experimental_worker_name, experimental_worker)
    graph_builder.add_node(control_worker_name, control_worker)
    
    # Add verification nodes
    verification_subgraphs = all_nodes.get_validator_subgraphs()
    verification_nodes = all_nodes.get_validator_nodes()
    for index, node in enumerate(verification_nodes):
        graph_builder.add_node(node.get_name(), verification_subgraphs[index])
    
    # Add graph edges
    graph_builder.add_edge(START, supervisor_name)
    graph_builder.add_edge(supervisor_name, "scheduler")
    # graph_builder.add_conditional_edges("supervisor", router, ["scheduler", END])
    graph_builder.add_edge(experimental_worker_name, "scheduler")
    graph_builder.add_edge(control_worker_name, "scheduler")
    
    for _, node in enumerate(verification_nodes):
        graph_builder.add_edge(node.get_name(), "scheduler")
    
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

def stream_graph_updates(graph, user_input: str, config: dict):
    """
    Stream graph updates during workflow execution.

    Args:
        graph: Compiled LangGraph workflow
        user_input (str): User's input question
    """
    max_global_steps = config.get("max_global_steps", 50)
    for event in graph.stream(
        {"messages": [("user", user_input)], "is_terminate": False}, 
        {"recursion_limit": max_global_steps, "configurable": {"thread_id": "main_graph_id"}}
    ):
        event_vals = list(event.values())
        step = max_global_steps - event_vals[0]["remaining_steps_display"] # if there are multiple event values, we believe they will have the same remaining steps (only possible in parallel execution?)..
        curie_logger.info(f"============================ Global Step {step} ============================")    
        curie_logger.debug(f"Event: {event}")
        for value in event.values():
            curie_logger.info(f"Event value: {value['messages'][-1].content}")

def main():
    """
    Main execution function for the LangGraph workflow.
    """
    if len(sys.argv) < 2:
        curie_logger.error("Usage: python script.py <config_file>")
        sys.exit(1)

    config_filename = sys.argv[1]

    try:
        # Build graph
        graph, metadata_store, config = build_graph(State, config_filename)

        # Read question from file
        question_filename = f"../{config['question_filename']}"
        user_input = get_question(question_filename) 
        sched_namespace = ("admin", "exp-sched")
        metadata_store.put(sched_namespace, "question", user_input)

        # Stream graph updates
        stream_graph_updates(graph, user_input, config)
        if config['report'] == True:
            generate_report(config)

    except Exception as e:
        curie_logger.error(f"Execution error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()