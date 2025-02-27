# Currently consists of two components: a LLM-based verifier, and an execution verifier (program).
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage

import os
import subprocess
import filecmp
import json

import model
import utils
import tool

from logger import init_logger
def setup_verifier_logging(log_filename: str):
    global curie_logger 
    curie_logger = init_logger(log_filename)

def create_LLMVerifierGraph(State, store, metadata_store, config_dict):
    """ Creates a verifier graph consisting of the LLM-based verifier that checks through the experimental workflow to verify that actual data is produced. """
    default_prompt_path = "prompts/llm-verifier.txt"
    system_prompt_key = "llm_verifier_system_prompt_filename"
    system_prompt_file = config_dict.get(system_prompt_key, default_prompt_path)

    verifier_write_tool = tool.LLMVerifierWriteTool(store, metadata_store)
    store_get_tool = tool.StoreGetTool(store)
    tools = [tool.execute_shell_command, store_get_tool, verifier_write_tool]
    verifier_graph = create_VerifierGraph(State, store, metadata_store, system_prompt_file, tools, "llm_verifier")
    return verifier_graph

def create_PatchVerifierGraph(State, store, metadata_store, config_dict):
    """ Creates a patcher graph consisting of the LLM-based patcher/debugger that debugs the experimental workflow to ensure that actual data is produced. """
    system_prompt_key = "patcher_system_prompt_filename"
    default_prompt_path = "prompts/exp-patcher.txt"
    system_prompt_file = config_dict.get(system_prompt_key, default_prompt_path)
    patcher_record_tool = tool.PatchVerifierWriteTool(store, metadata_store)
    patch_agent_tool = tool.PatcherAgentTool(config_dict)
    store_get_tool = tool.StoreGetTool(store)
    tools = [patch_agent_tool, tool.execute_shell_command, patcher_record_tool, store_get_tool] 
    verifier_graph = create_VerifierGraph(State, store, metadata_store, system_prompt_file, tools, "patch_verifier")
    return verifier_graph

def create_AnalyzerGraph(State, store, metadata_store, config_dict):
    """ Creates a analyzer graph consisting of the LLM-based analyzer that analyses results from the experimental workflow."""
    system_prompt_key = "analyzer_system_prompt_filename"
    default_prompt_path = "prompts/exp-analyzer.txt"
    system_prompt_file = config_dict.get(system_prompt_key, default_prompt_path)

    patcher_record_tool = tool.AnalyzerWriteTool(store, metadata_store)
    store_get_tool = tool.StoreGetTool(store)
    tools = [tool.read_file_contents, patcher_record_tool, store_get_tool] # Only tool is code execution for now
    verifier_graph = create_VerifierGraph(State, store, metadata_store, system_prompt_file, tools, "analyzer")
    return verifier_graph

def create_ConcluderGraph(State, store, metadata_store, config_dict):
    """ Creates a concluder graph consisting of the LLM-based concluder that is only activated when all results have been produced."""
    system_prompt_key = "concluder_system_prompt_filename"
    default_prompt_path = "prompts/exp-concluder.txt"
    system_prompt_file = config_dict.get(system_prompt_key, default_prompt_path)

    patcher_record_tool = tool.ConcluderWriteTool(store, metadata_store)
    store_get_tool = tool.StoreGetTool(store)
    tools = [tool.read_file_contents, patcher_record_tool, store_get_tool] # Only tool is code execution for now
    verifier_graph = create_VerifierGraph(State, store, metadata_store, system_prompt_file, tools, "concluder")
    return verifier_graph

def create_VerifierGraph(State, store, metadata_store, system_prompt_file, tools, node_name):
    """ Creates a verifier graph consisting of the LLM-based verifier that checks through the experimental workflow to verify that actual data is produced. """
    # TODO: only creating one worker now. Not sure if we will create multiple workers.
    # NOTE: since there is no parallelization now we assume that there is only one control_experiment.sh that we need to look into. We can hardcode this for now. 
    
    def router(state):
        if state["remaining_steps"] <= 2:
            return END
        return node_name
        
    verifier_builder = StateGraph(State)
    verifier_node = create_Verifier(tools, system_prompt_file, State, node_name) 

    verifier_builder.add_node(node_name, verifier_node)
    verifier_builder.add_edge(START, node_name)
    tool_node = ToolNode(tools=tools)
    verifier_builder.add_node("tools", tool_node)

    verifier_builder.add_conditional_edges(
        node_name,
        tools_condition,
    )
    # verifier_builder.add_conditional_edges("tools", router, [node_name, END])
    verifier_builder.add_edge("tools", node_name)

    verifier_graph = verifier_builder.compile()
    utils.save_langgraph_graph(verifier_graph, f"../logs/misc/{node_name}_graph_image.png")

    def call_verifier_graph(state: State) -> State:
        response = verifier_graph.invoke({"messages": state["messages"][-1]})
        return {
            "messages": [
                HumanMessage(content=response["messages"][-1].content, name=f"{node_name}_graph")
            ],
            "prev_agent": response["prev_agent"],
            "remaining_steps_display": state["remaining_steps"],
        }
    return call_verifier_graph

def create_Verifier(tools, system_prompt_file, State, node_name):    
    def Verifier(state: State):
        if state["remaining_steps"] <= 4:
            return {
                "messages": [], 
                "prev_agent": node_name,
            }

        # Read from prompt file:
        with open(system_prompt_file, "r") as file:
            system_prompt = file.read()

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
        
        response = model.query_model_safe(messages, tools)
        curie_logger.info(f"⭕⭕⭕ FROM {node_name} ✅✅✅") 
        if response.content:
            curie_logger.info(response.content)
        if response.tool_calls:
            curie_logger.info(f"Tool calls: {response.tool_calls[0]['name']}")
            if 'verifier_log_message' in response.tool_calls[0]['args']:
                curie_logger.info(f"Message: {response.tool_calls[0]['args']['verifier_log_message']}")
            else:
                curie_logger.debug(f"Message: {response.tool_calls[0]['args']}")
            curie_logger.debug(json.dumps(response.tool_calls, indent=4) )

        return {"messages": [response], "prev_agent": node_name}
    
    return Verifier