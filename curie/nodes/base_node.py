from pydantic import BaseModel, Field
from typing_extensions import TypedDict 
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from abc import ABC, abstractmethod
from memory.memory_manager import CurieMemoryManager

import os
import json

import model
import utils
import tool
import settings
from logger import init_logger
from scheduler import SchedNode
from utils import count_tokens

class NodeConfig(BaseModel):
    name: str = Field(
        description="The name of the node, used for logging and debugging."
    )

    node_icon: str = Field( 
        default="👑",
        description="The icon for the node."
    )

    log_filename: str = Field(
        description="The filename for the log file."
    )

    config_filename: str = Field(
        description="The filename for the config file."
    )

    transition_objs: dict = Field(
        default_factory=dict,
        description="A dictionary of transition objects for the node."
    )
    
    system_prompt_key: str = Field(
        default="supervisor_system_prompt_filename",
        description="The key for the system prompt in the config file."
    )

    default_system_prompt_filename: str = Field(
        default="prompts/exp-supervisor.txt",
        description="The default filename for the system prompt."
    )

class BaseNode(ABC):
    def __init__(self, sched_node: SchedNode, config: NodeConfig, State, store, metadata_store, memory, tools: list):
        self.node_config = config
        self.sched_node = sched_node # we will be using helper functions from scheduler within transition_handle_func
        self.curie_logger = init_logger(self.node_config.log_filename)
        self.State = State
        self.store = store
        self.metadata_store = metadata_store
        self.memory = memory
        self.tools = tools
        self.sched_namespace = self.sched_node.sched_namespace
        self.plan_namespace = self.sched_node.plan_namespace
        self.memory_manager = CurieMemoryManager()
    
    def get_name(self):
        return self.node_config.name

    def create_subgraph(self):
        """ Creates a Node subgraph."""

        subgraph_builder = StateGraph(self.State)
        
        with open(self.node_config.config_filename, 'r') as file:
            config = json.load(file)
    
        system_prompt_file = config.get(self.node_config.system_prompt_key, self.node_config.default_system_prompt_filename)

        subgraph_node = self._create_model_response(system_prompt_file) 

        subgraph_builder.add_node(self.node_config.name, subgraph_node)
        subgraph_builder.add_edge(START, self.node_config.name)
        tool_node = ToolNode(tools=self.tools)
        subgraph_builder.add_node("tools", tool_node)

        subgraph_builder.add_conditional_edges(self.node_config.name, tools_condition)
        # supervisor_builder.add_conditional_edges( "tools", router, ["supervisor", END])
        subgraph_builder.add_edge("tools", self.node_config.name)

        subgraph = subgraph_builder.compile(checkpointer=self.memory) 

        def call_subgraph(state: self.State) -> self.State: 
            response = subgraph.invoke({
                    "messages": state["messages"][-1]
                },
                {
                    # "recursion_limit": 20,
                    "configurable": {
                        "thread_id": f"{self.node_config.name}_graph_id"
                    }
                }
            )
            
            return {
                "messages": [
                    HumanMessage(content=response["messages"][-1].content, name=f"{self.node_config.name}_graph")
                ],
                "prev_agent": response["prev_agent"],
                "remaining_steps_display": state["remaining_steps"],
            }
        return call_subgraph

    def _create_model_response(self, system_prompt_file):    
        # FIXME: better way to get model names; from config?
        # FIXME: can move model name to model.py 
        def Node(state: self.State):
            if state["remaining_steps"] <= settings.CONCLUDER_BUFFER_STEPS:
                return {
                    "messages": [], 
                    "prev_agent": self.node_config.name,
                }
                
            # Read from prompt file:
            with open(system_prompt_file, "r") as file:
                system_prompt = file.read()

            system_message = SystemMessage(
                content=system_prompt,
            )

            # Use memory manager to get and prune messages
            original_messages = state["messages"]
            original_token_count = count_tokens(original_messages)

            self.memory_manager.replace(self.plan_namespace, original_messages)
            self.memory_manager.prune_and_summarize(self.plan_namespace)
            messages = self.memory_manager.get(self.plan_namespace)
            
            managed_token_count = count_tokens(messages)

            # self.curie_logger.info(f"❕❕❕ before filtering (len: {len(state['messages'])} messages): {state['messages']}")
            # self.curie_logger.info(f"❕❕❕ after filtering (len: {len(filtered_messages)} messages ): {filtered_messages}")
            self.curie_logger.debug(f"🏦 {self.node_config.node_icon} number of saved messages: {len(original_messages)} --> {len(messages)}")
            self.curie_logger.debug(f"🏦 {self.node_config.node_icon} token count: {original_token_count} --> {managed_token_count} (Reduction: {original_token_count - managed_token_count})")

            state["messages"] = messages
            
            # Ensure the system prompt is included at the start of the conversation
            if not any(isinstance(msg, SystemMessage) for msg in messages):
                messages.insert(0, system_message)

            self.curie_logger.debug(f"Messages TO {self.node_config.name.upper()} {self.node_config.node_icon}: {messages}")
            
            response = model.query_model_safe(messages, tools=self.tools)

            self.curie_logger.info(f"<><><><><> {self.node_config.node_icon} {self.node_config.name.upper()} {self.node_config.node_icon} <><><><><>")

            if response.tool_calls:
                self.curie_logger.info(f"Tool calls: {response.tool_calls[0]['name']}")
                if 'prompt' in response.tool_calls[0]['args']:
                    self.curie_logger.info(f"Message received: {response.tool_calls[0]['args']['prompt']}")
                elif 'verifier_log_message' in response.tool_calls[0]['args']:
                    self.curie_logger.info(f"Message: {response.tool_calls[0]['args']['verifier_log_message']}")
                else:
                    self.curie_logger.info(f"Message: {response.tool_calls[0]['args']}")
            
            concise_msg = response.content.split('\n\n')[0]
            if self.node_config.name.upper() == "CONCLUDER":
                self.curie_logger.info(f"✌️ Concluder response: {response.content}")
            elif concise_msg:
                self.curie_logger.info(f'Concise response: {concise_msg}')

            self.curie_logger.debug(f"Full response from {self.node_config.name.upper()} {self.node_config.node_icon}: {response}")
            
            # Update memory with the new response
            self.memory_manager.push(self.plan_namespace, response)

            return {"messages": [response], "prev_agent": self.node_config.name}
            # need to change 'add_messages' if you want to permanently update the message state.
        
        return Node

    @abstractmethod
    def transition_handle_func(self):
        """Handles transition logic and determines next action to take."""
        pass

    @abstractmethod
    def create_transition_objs(self):
        """Creates transition objects for the node."""
        pass