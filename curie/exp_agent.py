from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage

import model
import utils
import tool
import os
import json

from logger import init_logger
def setup_supervisor_logging(log_filename: str):
    global curie_logger 
    curie_logger = init_logger(log_filename)

def create_ExpSupervisorGraph(State, store, metadata_store, memory, config_filename):
    """ Creates a Supervisor graph that proposes experimental plans, and makes experimental progress. """
    # TODO: only creating one worker now. Later, we will create multiple workers.
    supervisor_builder = StateGraph(State)
    system_prompt_file = "prompts/exp-supervisor.txt"
    # Read config_file which is a json file:
    with open(config_filename, 'r') as file:
        config = json.load(file)
        if config["supervisor_system_prompt_filename"] != "none": # this happens only for base config
            system_prompt_file = config["supervisor_system_prompt_filename"] 

    store_write_tool = tool.NewExpPlanStoreWriteTool(store, metadata_store)
    store_write_tool_2 = tool.ExistingExpPlanStoreWriteTool(store, metadata_store) # TODO: make this more fine grained later
    redo_write_tool = tool.RedoExpPartitionTool(store, metadata_store)
    store_get_tool = tool.StoreGetTool(store)
    edit_priority_tool = tool.EditExpPriorityTool(store, metadata_store)
    tools = [store_write_tool, edit_priority_tool, redo_write_tool, store_get_tool, tool.read_file_contents, 
    # tool.execute_shell_command
    ] # Only tool is code execution for now
    supervisor_node = create_ExpSupervisor(tools, system_prompt_file, State) 

    supervisor_builder.add_node("supervisor", supervisor_node)
    supervisor_builder.add_edge(START, "supervisor")
    tool_node = ToolNode(tools=tools)
    supervisor_builder.add_node("tools", tool_node)

    supervisor_builder.add_conditional_edges(
        "supervisor",
        tools_condition,
    )
    supervisor_builder.add_edge("tools", "supervisor")
    
    supervisor_graph = supervisor_builder.compile(checkpointer=memory)
    os.makedirs("../logs/misc") if not os.path.exists("../logs/misc") else None
    utils.save_langgraph_graph(supervisor_graph, "../logs/misc/supervisor_graph_image.png") 

    def call_supervisor_graph(state: State) -> State:
        response = supervisor_graph.invoke({"messages": state["messages"][-1]}, {"recursion_limit": 20, "configurable": {"thread_id": "supervisor_graph_id"}})
        return {
            "messages": [
                HumanMessage(content=response["messages"][-1].content, name="supervisor_graph")
            ],
            "prev_agent": response["prev_agent"],
        }
    return call_supervisor_graph

def create_ExpSupervisor(tools, system_prompt_file, State):    
    # FIXME: better way to get model names; from config?
    # FIXME: can move model name to model.py 
    def ExpSupervisor(state: State):
        # Read from prompt file:
        with open(system_prompt_file, "r") as file:
            system_prompt = file.read()

        system_message = SystemMessage(
            content=system_prompt,
        )

        # Query model and append response to chat history 
        messages = state["messages"]

        # Ensure the system prompt is included at the start of the conversation
        if not any(isinstance(msg, SystemMessage) for msg in messages):
            messages.insert(0, system_message)
        
        response = model.query_model_safe(messages, tools=tools)
        curie_logger.info("<> FROM SUPERVISOR:")
        curie_logger.info(utils.parse_langchain_llm_output(response))
        curie_logger.info("------------------ END Supervisor -----------------")
        return {"messages": [response], "prev_agent": "supervisor"}
    
    return ExpSupervisor