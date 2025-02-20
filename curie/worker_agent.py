# Create worker agent graph here:
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from typing import Annotated, Literal

import settings
import model
from settings import list_worker_names, list_control_worker_names
import utils
import tool
import json 

from logger import init_logger
def setup_worker_logging(log_filename: str):
    global curie_logger 
    curie_logger = init_logger(log_filename)

def create_all_worker_graphs(State, store, metadata_store, memory, config_filename):

    # Read config_file which is a json file:

    with open(config_filename, 'r') as file:
        config = json.load(file) 
    system_prompt_key = "worker_system_prompt_filename"
    default_system_prompt_filename = "prompts/exp-worker.txt"
    worker_system_prompt_filename = config.get(system_prompt_key, default_system_prompt_filename)

    system_prompt_key = "control_worker_system_prompt_filename"
    default_system_prompt_filename = "prompts/controlled-worker.txt"
    control_worker_system_prompt_filename = config.get(system_prompt_key, default_system_prompt_filename)

    """Entry point for creating all workers. Call this."""
    worker_details = {
        "system_prompt_file": worker_system_prompt_filename,
        "graph_image_name": "../logs/misc/worker_graph_image.png",
        "graph_name": "worker_graph",
        "type": "experimental_worker",
        "config_filename": config_filename
    }
    experiment_worker_graph = _create_WorkerGraph(State, store, metadata_store, memory, worker_details, config)

    worker_details = {
        "system_prompt_file": control_worker_system_prompt_filename,
        "graph_image_name": "../logs/misc/control_worker_graph_image.png",
        "graph_name": "control_worker_graph",
        "type": "control_worker",
        "config_filename": config_filename
    }
    control_worker_graph = _create_WorkerGraph(State, store, metadata_store, memory, worker_details, config)

    return experiment_worker_graph, control_worker_graph

def _create_WorkerGraph(State, store, metadata_store, memory, worker_details, config_dict):
    """ Creates a Worker graph that runs experimental groups given a working controlled experiment setup that was created by a controlled worker earlier. """
    # TODO: only creating one worker now. Later, we will create multiple workers.
    
    def router(state: State):
        if state["remaining_steps"] <= 2:
            return END
        return worker_names[0]
    
    worker_builder = StateGraph(State)
    system_prompt_file = worker_details["system_prompt_file"]
    config_file = worker_details["config_filename"]
    if worker_details["type"] == "experimental_worker":
        worker_names = list_worker_names()
    elif worker_details["type"] == "control_worker":
        worker_names = list_control_worker_names()
    assert len(worker_names) == 1 
    store_write_tool = tool.ExpPlanCompletedWriteTool(store, metadata_store)
    store_get_tool = tool.StoreGetTool(store)
    codeagent_openhands = tool.CodeAgentTool(config_dict)
    tools = [codeagent_openhands, tool.execute_shell_command, store_write_tool, store_get_tool] # Only tool is code execution for now
    worker_node = create_Worker(tools, system_prompt_file, config_file, State, worker_names[0]) 

    worker_builder.add_node(worker_names[0], worker_node)
    worker_builder.add_edge(START, worker_names[0])
    tool_node = ToolNode(tools=tools)
    worker_builder.add_node("tools", tool_node)

    worker_builder.add_conditional_edges(
        worker_names[0],
        tools_condition,
    )
    worker_builder.add_conditional_edges("tools", router, [worker_names[0], END])
    
    worker_graph = worker_builder.compile()
    utils.save_langgraph_graph(worker_graph, worker_details["graph_image_name"]) 

    def call_worker_graph(state: State) -> State:
        response = worker_graph.invoke({
                                        "messages": state["messages"][-1]
                                    },
                                    {
                                        # "recursion_limit": 20,
                                        "configurable": {
                                            "thread_id": "worker_graph_thread"
                                        }
                                    })
        # Give a very lenient recursion limit since we expect control experiment construction to take more trials https://langchain-ai.github.io/langgraph/how-tos/recursion-limit/#use-the-graph
        return {
            "messages": [
                HumanMessage(content=response["messages"][-1].content, name=worker_details["graph_name"])
            ],
            "prev_agent": response["prev_agent"],
        }
    return call_worker_graph

def create_Worker(tools, system_prompt_file, config_file, State, worker_name):  
    """ Normal worker that runs experimental groups given a working controlled experiment setup that was created by a controlled worker earlier. """

    def Worker(state: State):
        # Read from prompt file:
        with open(system_prompt_file, "r") as file:
            system_prompt = file.read()

        # Read config_file which is a json file:
        with open(config_file, 'r') as file:
            config = json.load(file)
            if config["benchmark_specific_context"] != "none": # this happens only for base config

                # Open txt filename represented by the benchmark_specific_context key in the config file, and append it to the system prompt:
                with open(config["benchmark_specific_context"], "r") as file2:
                    task_specific_context = file2.read()
                    system_prompt = system_prompt + "\n\n" + task_specific_context

        # system_prompt = """
        # You are an agent that will use the test_search_tool tool provided when answering questions about UMich.  
        # """
        system_message = SystemMessage(
            content=system_prompt,
        )

        # Query model and append response to chat history 
        messages = state["messages"]

        # Ensure the system prompt is included at the start of the conversation
        if not any(isinstance(msg, SystemMessage) for msg in messages):
            messages.insert(0, system_message)

        curie_logger.debug(f"Messages TO Worker 👷: {messages}")

        response = model.query_model_safe(messages, tools)
        curie_logger.info(f"👷 FROM Worker: {worker_name}")
        concise_msg = response.content.split('\n\n')[0]
        if concise_msg:
            curie_logger.info(f'Concise message: {concise_msg}')
        if response.tool_calls:
            curie_logger.info(f"Tool calls: {response.tool_calls[0]['name']}")
            if 'prompt' in response.tool_calls[0]['args']:
                curie_logger.info(f"Message received: {response.tool_calls[0]['args']['prompt']}")
            else:
                curie_logger.info(f"Message: {response.tool_calls[0]['args']}")
        curie_logger.debug(json.dumps(response.content, indent=4) )

        return {"messages": [response], "prev_agent": worker_name}
    
    return Worker