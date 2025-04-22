# References used:
# https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop
from nodes.base_node import BaseNode, NodeConfig
from langgraph.graph import StateGraph, START, END
from scheduler import SchedNode
from langgraph.types import interrupt, Command
from langchain_core.messages import HumanMessage, SystemMessage 

import os
import json

import settings
import model
import utils

class UserInput(BaseNode):

    def __init__(self, sched_node: SchedNode, config: NodeConfig, State, store, metadata_store, memory, tools: list):
        super().__init__(sched_node, config, State, store, metadata_store, memory, tools)  # Call parent class's __init__
        self.create_transition_objs()

    def create_transition_objs(self):
        # intro_message = "Here is the user's response:\n"
        self.node_config.transition_objs["done"] = lambda user_response: {
            "messages": user_response, 
            "next_agent": "user_input_router"
        }

    def transition_handle_func(self, state):
        """
        We will only forward to user_input_router. 
        """
        self.curie_logger.info("------------ Handle User Input ------------")

        items = self.store.search(self.plan_namespace)
        plans_list = [item.dict()["value"] for item in items]

        user_response = "Here are the previously proposed plans:\n" + str(plans_list) + "\n\nHere is the user's response: " + state["messages"][-1].content

        return self.node_config.transition_objs["done"](user_response)

    def create_subgraph(self):
        """ Creates a Node subgraph specific to UserInput. Override BaseNode implementation."""

        subgraph_builder = StateGraph(self.State)
        system_prompt_file = "dummy_system_prompt.txt"
        subgraph_node = self._create_model_response(system_prompt_file) 

        subgraph_builder.add_node(self.node_config.name, subgraph_node)
        subgraph_builder.add_edge(START, self.node_config.name)
        # subgraph_builder.add_edge(self.node_config.name, END)

        subgraph = subgraph_builder.compile(checkpointer=self.memory)
        os.makedirs("../../logs/misc") if not os.path.exists("../../logs/misc") else None
        # utils.save_langgraph_graph(subgraph, f"../../logs/misc/{self.node_config.name}_graph_image.png") 

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
            # ANSI escape codes for colors
            RED = "\033[91m"
            GREEN = "\033[92m"
            CYAN = "\033[96m"
            BOLD = "\033[1m"
            RESET = "\033[0m"

            # Custom input prompt
            user_input = input(f"{BOLD}{GREEN}✅ Do you approve of the architect's proposed plan? {RESET}"
                                f"{RED}❌ If not, why? {RESET}"
                                f"{CYAN}💬 Provide your response here: {RESET}")
            response = subgraph.invoke(Command(resume=user_input),
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
                "prev_agent": response.get("prev_agent", "scheduler"),
                "remaining_steps_display": state["remaining_steps"],
            }
        return call_subgraph

    def _create_model_response(self, system_prompt_file): 
        """Override BaseNode implementation."""   
        # FIXME: better way to get model names; from config?
        # FIXME: can move model name to model.py 
        def Node(state: self.State):
            if state["remaining_steps"] <= settings.CONCLUDER_BUFFER_STEPS:
                return {
                    "messages": [], 
                    "prev_agent": self.node_config.name,
                }

            response = interrupt("Do you approve of the architect's proposed plan?")

            self.curie_logger.info(f"<><><><><> {self.node_config.node_icon} {self.node_config.name.upper()} {self.node_config.node_icon} <><><><><>")

            self.curie_logger.info(f"Full response from user: {self.node_config.node_icon}: {response}")

            return {"messages": [response], "prev_agent": self.node_config.name}
        
        return Node

class UserInputRouter(BaseNode):

    def __init__(self, sched_node: SchedNode, config: NodeConfig, State, store, metadata_store, memory, tools: list):
        super().__init__(sched_node, config, State, store, metadata_store, memory, tools)  # Call parent class's __init__
        self.create_transition_objs()

    def create_transition_objs(self):
        intro_message = "You need to record down your decision using the user_router_wrote_list tool.\n"
        self.node_config.transition_objs["progress_not_recorded"] = lambda: {
            "messages": intro_message, 
            "next_agent": self.node_config.name
        }
        intro_message2 = "The user is satisfied with your proposed plan. Terminate and move on.\n"
        self.node_config.transition_objs["inform_architect_done"] = lambda: {
            "messages": intro_message2, 
            "next_agent": "supervisor",
            "is_user_input_done": True,
        }

        # intro_message = "The user is not satisfied with your proposed plan. Review your existing plans (via 'exp_plan_get'), then delete the plans (via 'exp_plan_remove'), then finally review the user's feedback below and propose a new plan that satisfies it:\n"
        intro_message3 = "The user is not satisfied with your previous proposed plan(s), which are attached for your reference. Review the user's feedback below and propose a new plan that satisfies it:\n"
        self.node_config.transition_objs["redo_architect"] = lambda router_log_message: {
            "messages": intro_message3 + router_log_message, 
            "next_agent": "supervisor"
        }

    def transition_handle_func(self, state):
        """
        Our router will essentially be called after the user input node. The router will analyze the user input, record its decision using the appropriate write tool. Here, we will analyze its decision, and decide what node to call next. Currently, there are only 2 possible choices:
            - If the user decides that the architect's plan is insufficient, we will go back to the architect node, and attach the router's analysis as context.
            - Otherwise, we will also return to architect, but let it know it is done.
        """
        self.curie_logger.info("------------ Handle User Input Router ------------")

        self.curie_logger.info("Checking user_router_wrote_list..")
        memory_id = str("user_router_wrote_list")
        wrote_list = self.metadata_store.get(self.sched_namespace, memory_id).dict()["value"]
        if len(wrote_list) == 0:
            return self.node_config.transition_objs["progress_not_recorded"]()
        if len(wrote_list) > 1:
            self.curie_logger.debug("WARNING: user input router wrote to user_router_wrote_list more than once..")

        # Reset wrote_list:
        self.metadata_store.put(self.sched_namespace, memory_id, [])

        if wrote_list[-1]["is_correct"]:
            return self.node_config.transition_objs["inform_architect_done"]()
        else:
            # Retrieve existing plans:
            items = self.store.search(self.plan_namespace)
            plans_list = [item.dict()["value"] for item in items]
            router_log_message = wrote_list[-1]["router_log_message"]

            feedback_msg = "Previous plan(s):\n" + str(plans_list) + "\n\nUser feedback:\n" + router_log_message

            # Delete all existing plans:
            for plan in plans_list:
                self.store.delete(self.plan_namespace, plan["plan_id"])
            # Reset architect wrote list:
            memory_id = str("supervisor_wrote_list")
            self.metadata_store.put(self.sched_namespace, memory_id, [])

            return self.node_config.transition_objs["redo_architect"](feedback_msg)