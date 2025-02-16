from langchain_core.tools import tool
from typing import Annotated, List
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import InjectedStore
from langgraph.prebuilt import InjectedState
from langgraph.graph import END
import uuid
import os
from typing import Optional, Type, Dict
import heapq

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from collections import deque, defaultdict
import re
import json
import subprocess
import logging

import formatter
import settings
import utils
import worker_agent
import verifier
from modified_deps.langchain_bash.tool import ShellTool
from logger import init_logger


shell_tool = ShellTool(timeout=1200)

def create_SchedNode(sched_tool, State, metadata_store):    
    """
        No LLM involved. Manual flow control. Reason: tools cannot modify state directly (and workaround are too troublesome https://github.com/langchain-ai/langgraph/discussions/1247) which we need to control which nodes to call next. 

        Workflow:
        - Supervisor -> SchedNode -> sched_tool -> SchedNode -> Worker X (multiple are possible. We need async too TODO)
        - Worker X -> SchedNode -> sched_tool -> SchedNode -> Supervisor
        - Note that supervisor to supervisor and worker to worker transitions are technically possible too, it is up to the sched_tool. 
    """
    user_id = "admin"
    application_context = "exp-sched" 
    sched_namespace = (user_id, application_context) # just a random namespace name for now
    setup_sched(metadata_store, sched_namespace)

    def SchedNode(state: State):
        # Invoke sched_tool: 
        response = sched_tool.invoke({"state":state}) # response is guaranteed to be a dict
        # Based on sched_tool, response, we decide which node to call next:
        if "next_agent" in response and response["next_agent"] == END: # terminate experiment. Langgraph will call END based on next_agent as defined in our conditional_edge in main.py
            return {"messages": state["messages"], "next_agent": END, "prev_agent": "sched_node"}
        if "next_agent" in response and response["next_agent"] == "supervisor": # next agent is the supervisor
            if response["prev_agent"] == "supervisor":
                intro_message = "This is the question to answer, make sure to formulate it in terms of an experimental plan(s) using the 'write_new_exp_plan' tool:\n"
                return {"messages": [
                            HumanMessage(content=intro_message + str(response["messages"]), name="scheduler")
                        ],
                    "prev_agent": "sched_node",
                    "next_agent": response["next_agent"]
                }
            elif response["prev_agent"] == "analyzer":
                intro_message = "The following experimental plan partitions (with plan IDs, groups, and partitions) have completed execution, each run twice with the same inputs for reproducibility. Their results were analyzed, and next-step suggestions appended. Review each suggestion to assess result validity. If incorrect, mark for redo using 'redo_exp_partition'; otherwise, leave the plan unchanged. Modify or create new plans as needed.\n"
                return {"messages": [
                            HumanMessage(content= intro_message + str(response["messages"]), name="scheduler")
                        ],
                    "prev_agent": "sched_node", 
                    "next_agent": response["next_agent"]
                }
            elif response["prev_agent"] == "patch_verifier":
                intro_message = "The following experimental plan workflows (containing plan IDs, group, partitions) have been attempted to be patched by a code debugging agent but failed. Please review. You may re-execute partitions where the workflow is not correct using the 'redo_exp_partition' tool. Otherwise, you can leave the plans unchanged, write new plans, or modify existing ones as needed.\n"
                return {"messages": [
                            HumanMessage(content= intro_message + str(response["messages"]), name="scheduler")
                        ],
                    "prev_agent": "sched_node", 
                    "next_agent": response["next_agent"]
                }
            elif response["prev_agent"] == "concluder":
                intro_message = '''
All partitions for all experimental plans have completed, with results produced and analyzed. A next-step suggestion is appended. Conclude the experiment if you believe it provides a rigorous and comprehensive answer. Otherwise, if results are insufficient or further questions remain, create a new experimental plan.
'''
                return {"messages": [
                            HumanMessage(content= intro_message + str(response["messages"]), name="scheduler")
                        ],
                    "prev_agent": "sched_node", 
                    "next_agent": response["next_agent"], 
                }
                
            assert True == False # should never reach here

        elif "next_agent" in response and response["next_agent"] == "llm_verifier":
            return {"messages": [ 
                        HumanMessage(content=str(response["messages"]), name="scheduler")
                    ],
                "prev_agent": "sched_node", 
                "next_agent": response["next_agent"]
            }
        elif "next_agent" in response and response["next_agent"] == "patch_verifier":
            return {"messages": [ 
                        HumanMessage(content=str(response["messages"]), name="scheduler")
                    ],
                "prev_agent": "sched_node", 
                "next_agent": response["next_agent"]
            }
        elif "next_agent" in response and response["next_agent"] == "analyzer":
            intro_message = "The following partitions have completed execution and have also been executed twice with the same independent variable inputs to check for reproducibility.\n"
            return {"messages": [
                        HumanMessage(content= intro_message + str(response["messages"]), name="scheduler")
                    ],
                "prev_agent": "sched_node", 
                "next_agent": response["next_agent"]
            }
        elif "next_agent" in response and response["next_agent"] == "concluder":
            return {"messages": [
                        HumanMessage(content= str(response["messages"]), name="scheduler")
                    ],
                "prev_agent": "sched_node", 
                "next_agent": response["next_agent"]
            }
        else: # next agent is a worker, and we need to determine which worker it will be.
            # TODO: not doing parallelism for now, so we just assume for instances that messages will only contain one worker's name, and we will only need to call that one worker. Parallelism will be implemented later.
            control_empty = not response["control_work"]["messages"]
            experimental_empty = not response["experimental_work"]["messages"]
            assert control_empty != experimental_empty # only one of them should be non-empty. in our current iteration, since there is only one plan (I hope), this means one of these should be empty, since we cannot have that same plan not having a control group, but having some experimental groups to be run, and vice versa. TODO: thus, we will need to change this and the following lines later to accommodate more than one plan existing
            
            if response["experimental_work"]["messages"]:
                type_name = "experimental_work"
            elif response["control_work"]["messages"]:
                type_name = "control_work"
            assert len(list(response[type_name]["messages"].keys())) == 1 # only one worker for a given worker type has been assigned partitions to run
            return {"messages": [
                        HumanMessage(content=json.dumps(list(response[type_name]["messages"].values())[0]), name="scheduler")
                    ], 
                "prev_agent": "sched_node", 
                "next_agent": list(response[type_name]["messages"].keys())[0]
            } # next_agent: worker_1
    
    return SchedNode

def setup_sched(metadata_store, sched_namespace):
    memory_id = str("worker_assignment_dict") # Format of this dict: {"worker_name": [(exp_plan_id1, "experimental_group_partition_1"), (exp_plan_id2, "experimental_group_partition_1"), ...]}
    assignment_dict = {}
    for name in settings.list_worker_names():
        assignment_dict[name] = []
    metadata_store.put(sched_namespace, memory_id, assignment_dict)

    memory_id = str("control_worker_assignment_dict") # Format of this dict: {"worker_name": [(exp_plan_id1, "control_group"), (exp_plan_id2, "control_group"), ...]}
    assignment_dict = {}
    for name in settings.list_control_worker_names():
        assignment_dict[name] = []
    metadata_store.put(sched_namespace, memory_id, assignment_dict)

    memory_ids = {
        "llm_verifier_assignment_dict": "llm_verifier",
        "exec_verifier_assignment_dict": "exec_verifier",
        "patch_verifier_assignment_dict": "patch_verifier",
        "analyzer_assignment_dict": "analyzer",
        "concluder_assignment_dict": "concluder",
    }

    def create_assignment_dict(memory_ids, metadata_store, sched_namespace):
        for memory_id, verifier_name in memory_ids.items():
            assignment_dict = {verifier_name: []}
            metadata_store.put(sched_namespace, memory_id, assignment_dict)

    create_assignment_dict(memory_ids, metadata_store, sched_namespace)

    write_lists = [
        "supervisor_wrote_list", # Format of this list: [exp_plan_id1, exp_plan_id2, ...] This is to record down the calls to exp_plan_write tool by supervisor, and the plans that were written/modified. 
        "llm_verifier_wrote_list", # Format of this list: [{"plan_id": 123, "partition_name": "control_group", "is_correct": True, "verifier_log_message": "is no error haha"}, ...] This is to record down the calls to workflow_verified_record tool by llm verifier, and the workflows that were evaluated. 
        "patch_verifier_wrote_list", # Format of this list: [{"plan_id": 123, "partition_name": "control_group", "is_correct": True, "patcher_log_message": "is no error haha"}, ...] This is to record down the calls to workflow_verified_record tool by patch verifier, and the workflows that were evaluated. 
        "analyzer_wrote_list", # Format of this list: [{"plan_id": 123, "partition_name": "control_group", "is_correct": True, "patcher_log_message": "is no error haha"}, ...] This is to record down the calls to workflow_verified_record tool by patch verifier, and the workflows that were evaluated. 
        "concluder_wrote_list", # Format of this list: [{"plan_id": 123, "partition_name": "control_group", "is_correct": True, "patcher_log_message": "is no error haha"}, ...] This is to record down the calls to workflow_verified_record tool by patch verifier, and the workflows that were evaluated. 
        "supervisor_redo_partition_list", # Format of this list: [{"plan_id": 123, "group": "control_group", "partition_name": "partition_1", "error_feedback": "xtz"}, ...] This is to record down the calls to redo_exp_partition tool by supervisor.
        "standby_exp_plan_list"
    ]

    queues = [
        "worker_queue", # Format of this queue: [(priority, exp_plan_id1, "experimental_group_partition_2"), (priority, exp_plan_id2, "experimental_group_partition_2")] # Priority queue implemented using min heap 
        "control_worker_queue"
    ]

    def create_empty_list(memory_ids, metadata_store, sched_namespace):
        for memory_id in memory_ids:
            metadata_store.put(sched_namespace, memory_id, [])

    create_empty_list(write_lists, metadata_store, sched_namespace)
    create_empty_list(queues, metadata_store, sched_namespace)

class SchedInput(BaseModel):
    state: Annotated[dict, InjectedState] # For scheduler, we are guaranteed that the prev_agent will either be supervisor or worker. 

class SchedTool(BaseTool):
    name: str = "scheduler"
    description: str = "Programmatic scheduling."
    args_schema: Type[BaseModel] = SchedInput
    # None of the following work:
    # https://langchain-ai.github.io/langgraph/how-tos/pass-run-time-values-to-tools/#define-the-tools_1
    # https://github.com/langchain-ai/langchain/discussions/24906
    # and so on..
    store: Optional[InMemoryStore] = None  # Declare store as an optional field
    metadata_store: Optional[InMemoryStore] = None  # Declare store as an optional field. This is for storing sched related metadata. 
    sched_namespace: Optional[tuple] = None
    plan_namespace: Optional[tuple] = None
    config: Optional[dict] = None
    curie_logger: Optional[logging.Logger] = None

    def __init__(self, store: InMemoryStore, metadata_store: InMemoryStore, config: dict):
        super().__init__()
        self.store = store
        self.metadata_store = metadata_store
        self.sched_namespace = self.get_sched_namespace()
        self.plan_namespace = self.get_plan_namespace()
        self.config = config
        log_filename = f'../{config["log_filename"]}'
        self.curie_logger = init_logger(log_filename)
    
    class Config:
        arbitrary_types_allowed = True  # Allow non-Pydantic types like InMemoryStore


    def _run(
        self,
        state: Annotated[dict, InjectedState],
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str: 
        prev_agent = state["prev_agent"]
        next_agent = self._handle_agent(prev_agent)
        next_agent_name = 'N/A'
        for k, v in next_agent.items():
            if k == 'next_agent':
                next_agent_name = v
                break
            if 'next_agent' in v:
                next_agent_name = v['next_agent']
                break
        
        self.curie_logger.info(
            f"------- Scheduling {prev_agent} --> {next_agent_name} ---------"
        )
        if next_agent_name == 'N/A':
            self.curie_logger.info(
                f"------- No next agent found for {next_agent} ---------"
            )
        
        return next_agent
    
    def _handle_agent(self, prev_agent: str) -> str:
        """
        Route to appropriate handler based on previous agent.
        
        Args:
            prev_agent (str): Name of the previous agent
            
        Returns:
            str: Response from the appropriate handler
        """
        handlers = {
            "supervisor": self.handle_supervisor,
            "worker": lambda: self.handle_worker(prev_agent),
            "llm_verifier": lambda: self.handle_llm_verifier(prev_agent),
            "patch_verifier": lambda: self.handle_patch_verifier(prev_agent),
            "analyzer": lambda: self.handle_analyzer(prev_agent),
            "concluder": lambda: self.handle_concluder(prev_agent)
        }
        
        # Handle special case for worker which has a prefix
        if "worker" in prev_agent:
            return handlers["worker"]()
        
        # Get the appropriate handler or raise an error if not found
        handler = handlers.get(prev_agent)
        if not handler:
            raise ValueError(f"No handler found for agent: {prev_agent}")
            
        return handler()

    def is_init_run(self):
        memory_id = str("worker_assignment_dict")
        return self.metadata_store.get(self.sched_namespace, memory_id).dict()["value"] is None

    def handle_supervisor(self):
        """
            All of the plans that were edited/written by the supervisor will be scheduled for execution now or added to the queue. 
            If the plan existed before, we check if the experimental groups have been modified: 
                if yes, we will preempt the existing group (TODO: not implemented yet. this requires that we for instance also record the old plan to know if it has changed..) from the current worker, 
                or modify within the queue. 
            If the plan is new, 
                we simply add to worker if some are idle, 
                or add to queue.
        """
        self.curie_logger.info("------------ Supervisor ------------")

        # Zero, if no plan exists at all, we need to re-prompt the architect to force it to create a plan:
        if self.is_no_plan_exists():
            return {"messages": self.get_question(), "next_agent": "supervisor", "prev_agent": "supervisor"}

        # First, check for exp termination condition:
        is_terminate = self.check_exp_termination_condition()
        if is_terminate:
            return {"next_agent": END}

        # Second, for control groups that are done, move their experimental groups to the worker queue:
        self.curie_logger.info("Checking control group done..")
        memory_id = str("standby_exp_plan_list")
        standby_exp_plan_list = self.metadata_store.get(self.sched_namespace, memory_id).dict()["value"]
        for plan_id in standby_exp_plan_list[:]: # iterate over a copy of standby_exp_plan_list
            self.update_queues(plan_id, assert_new_control=True)

        # Third, for plans that were written/modified by the supervisor, add them to the correct queue:
        self.curie_logger.info("Checking supervisor wrote list..")
        memory_id = str("supervisor_wrote_list")
        supervisor_wrote_list = self.metadata_store.get(self.sched_namespace, memory_id).dict()["value"]
        # Create a new /workspace dir for each new plan_id, and other related inits for a new plan:
        self.init_new_plan(supervisor_wrote_list)

        # Update queues:
        for _ in range(len(supervisor_wrote_list)):
            # NOTE: currently, we ignore plan groups that are already executing in the worker.. 
            plan_id = supervisor_wrote_list.pop(0)
            self.update_queues(plan_id) 
        # Reset supervisor_wrote_list:
        assert len(supervisor_wrote_list) == 0
        self.metadata_store.put(self.sched_namespace, memory_id, supervisor_wrote_list)

        # Fourth, for partition's that need to be redone, add them to the correct queue: the difference here is that an explicit error feedback is provided by the supervisor.
        self.curie_logger.info("Checking supervisor redo partition list..")
        memory_id = str("supervisor_redo_partition_list")
        supervisor_redo_partition_list = self.metadata_store.get(self.sched_namespace, memory_id).dict()["value"]
        for _ in range(len(supervisor_redo_partition_list)):
            redo_details = supervisor_redo_partition_list.pop(0)
            self.update_queues(redo_details["plan_id"], redo_details=redo_details)
        # Reset supervisor_redo_partition_list:
        assert len(supervisor_redo_partition_list) == 0
        self.metadata_store.put(self.sched_namespace, memory_id, supervisor_redo_partition_list)

        self.curie_logger.info("------------Exiting handle supervisor!!!------------")
        # Fifth, Assign to control and experimental workers if there are idle workers: according to priority
        # TODO: currently since we have NOT implemented ASYNC, we will only run 1 worker at a time (i.e., either control or normal worker). Note that we also only have 1 control and 1 normal worker only. Since there is no async I did not implement parallelism yet. 
        assignment_messages = self.assign_worker("control")
        if not assignment_messages: # if there is no control group to run, we will run the experimental group of some partition
            return {
                "control_work": {"messages": assignment_messages, "next_agent": "control_worker"},
                "experimental_work": {"messages": self.assign_worker("experimental"), "next_agent": "worker"}
            }
        else: # if there exist a control group to run, we do not run the experimental group of any partition
            return {
                "control_work": {"messages": assignment_messages, "next_agent": "control_worker"},
                "experimental_work": {"messages": [], "next_agent": "control_worker"}
            }

        # Original version:  
        # return {
        #     "control_work": {"messages": self.assign_worker("control"), "next_agent": "control_worker"},
        #     "experimental_work": {"messages": self.assign_worker("experimental"), "next_agent": "worker"}
        # }
    
    def handle_worker(self, worker_name: str):
        """
            A worker has completed a run. 
            We will now:
            - remove the worker from the worker assignment dict. 
            - set the executed group to done. NOTE: update, this will be handled by the worker itself instead.
            - return information back to supervisor.
        """
        self.curie_logger.info("------------Entering handle worker!!!------------")
        # Get plan id and partition names assigned to worker name:
        assignments = self.get_worker_assignment(worker_name) # format: [(plan_id1, partition_name1), (plan_id2, partition_name2), ...]

        group_type = self.get_worker_group_type(worker_name)

        completion_messages = []

        not_done_groups = []

        # Assert that all assigned partition names are now done
        for assignment in assignments:
            plan_id, group, partition_name = assignment["plan_id"], assignment["group"], assignment["partition_name"]
            plan = self.store.get(self.plan_namespace, plan_id).dict()["value"]
            self.curie_logger.info(f'Plan ID: {plan_id}')
            
            self.curie_logger.info(f"Partition Name: {partition_name}")
            self.curie_logger.info(f"Plan details: {plan}")
            # if plan_id not in completion_messages:
            #     completion_messages[plan_id] = plan
            if plan[group][partition_name]["done"] != True:
                not_done_groups.append((plan_id, group, partition_name))

        if not_done_groups: # we will reexecute the groups that are not done, by the remaining groups to the same worker
            # Determine the worker type:
            if group_type == "experimental":
                return {
                    "control_work": {"messages": {}, "next_agent": "control_worker"},
                    "experimental_work": {"messages": {worker_name: not_done_groups}, "next_agent": "worker"}
                }
            elif group_type == "control":
                return {
                    "control_work": {"messages": {worker_name: not_done_groups}, "next_agent": "control_worker"},
                    "experimental_work": {"messages": {}, "next_agent": "worker"}
                }
            assert False # should not reach here
            
        # Remove worker from worker assignment dict:
        self.unassign_worker_all(worker_name) # NOTE: a worker will only return to the supervisor once all its groups are done. 

        # Pass all assignments to llm_verifier:
        for assignment in assignments:
            plan_id, group, partition_name = assignment["plan_id"], assignment["group"], assignment["partition_name"]
            plan = self.store.get(self.plan_namespace, plan_id).dict()["value"]
            task_details = {
                "plan_id": plan_id,
                "group": group,
                "partition_name": partition_name,
                "workspace_dir": self.get_workspace_dirname(plan_id),
                "control_experiment_filename": self.get_control_experiment_filename(plan_id, group, partition_name),
                "control_experiment_results_filename": self.get_control_experiment_results_filename(plan_id, group, partition_name),
            }
            completion_messages.append(task_details)
            self.assign_verifier("llm_verifier", task_details)

        # utils.print_workspace_contents()
        self.curie_logger.info("------------Exiting handle worker!!!------------")
        # Inform supervisor that worker has completed a run:
        return {"messages": completion_messages, "next_agent": "llm_verifier"}

    def handle_llm_verifier(self, verifier_name: str):
        """
            LLM verifier has completed a run. 
            We will now:
            - remove the verifier from the verifier assignment dict. 
            - return information back to supervisor.
        """
        self.curie_logger.info("------------Entering handle llm verifier!!!------------")
        # Get plan id and partition names assigned to verifier name:
        assignments = self.get_verifier_assignment(verifier_name) # format: [(plan_id1, partition_name1), (plan_id2, partition_name2), ...]

        completion_messages = [] # format: [{"plan_id": plan_id1, "partition_name": partition_name1, "is_correct": True, "verifier_log_message": "no error"}, ...]

        # Assert that all assigned partition names are now done
        has_false = False # if there exist one workflow that is considered incorrect by the verifier.
        for assignment in assignments:
            plan_id, group, partition_name = assignment["plan_id"], assignment["group"], assignment["partition_name"]
            # Check if the verifier has written to the verifier_wrote_list:
            item = self.get_verifier_wrote_list_item(verifier_name, plan_id, group, partition_name)
            if item is None:
                self.curie_logger.info("Warning: LLM verifier has not written plan_id {}, group {}, partition_name {} to verifier_wrote_list yet. We will rerun LLM verifier.".format(plan_id, group, partition_name))
                return {"messages": assignments, "next_agent": "llm_verifier"}
            completion_messages.append(item)
            if not item["is_correct"]:
                has_false = True

        # Remove verifier from verifier assignment dict:
        self.unassign_verifier_all(verifier_name)

        # Remove from verifier_wrote_list:
        self.remove_verifier_wrote_list_all(verifier_name)

        # utils.print_workspace_contents()

        self.curie_logger.info("------------Exiting handle llm verifier!!!------------")
        # NOTE: currently because I don't think divergent parallel execution is possible, we will just return to supervisor if even one workflow is considered incorrect (even though there may be others that are correct which we can in principle forward to the exec_verifier)
        # Inform supervisor that verifier has completed a run:
        if has_false: # go to patch verifier
            # Pass all assignments to patch_verifier:
            for task_details in completion_messages:
                self.assign_verifier("patch_verifier", task_details)
            return {"messages": completion_messages, "prev_agent": "llm_verifier", "next_agent": "patch_verifier"}
        else: # go to exec verifier -> supervisor 
            for item in completion_messages:
                item["control_experiment_results_filename"] = self.get_control_experiment_results_filename(item["plan_id"], item["group"], item["partition_name"])
            completion_messages = verifier.exec_verifier(completion_messages)
            self.write_all_control_experiment_results_filenames(completion_messages)
            for task_details in completion_messages:
                self.assign_verifier("analyzer", task_details)
            return {"messages": completion_messages, "prev_agent": "exec_verifier", "next_agent": "analyzer"}

    def handle_patch_verifier(self, verifier_name: str):
        """
            Patch verifier has completed a run. 
            We will now:
            - remove the verifier from the verifier assignment dict. 
            - return information back to supervisor.
        """
        self.curie_logger.info("------------Entering handle patch verifier!!!------------")
        # Get plan id and partition names assigned to verifier name:
        assignments = self.get_verifier_assignment(verifier_name) # format: [(plan_id1, partition_name1), (plan_id2, partition_name2), ...]

        completion_messages = [] # format: [{"plan_id": plan_id1, "partition_name": partition_name1, "is_correct": True, "verifier_log_message": "no error"}, ...]

        # Assert that all assigned partition names are now done
        has_false = False # if there exist one workflow that is considered incorrect by the verifier.
        for assignment in assignments:
            plan_id, group, partition_name = assignment["plan_id"], assignment["group"], assignment["partition_name"]
            # Check if the verifier has written to the verifier_wrote_list:
            item = self.get_verifier_wrote_list_item(verifier_name, plan_id, group, partition_name)
            if item is None:
                self.curie_logger.info("Warning: Patch verifier has not written plan_id {}, group {}, partition_name {} to verifier_wrote_list yet. We will rerun patch verifier.".format(plan_id, group, partition_name))
                return {"messages": assignments, "next_agent": "patch_verifier"}
            completion_messages.append(item)
            if not item["is_correct"]:
                has_false = True

        # Remove verifier from verifier assignment dict:
        self.unassign_verifier_all(verifier_name)

        # Remove from verifier_wrote_list:
        self.remove_verifier_wrote_list_all(verifier_name)

        # utils.print_workspace_contents()

        self.curie_logger.info("------------Exiting handle patch verifier!!!------------")
        # NOTE: currently because I don't think divergent parallel execution is possible, we will just return to supervisor if even one workflow is considered incorrect (even though there may be others that are correct which we can in principle forward to the exec_verifier)
        # Inform supervisor that verifier has completed a run:
        if has_false: # go to supervisor
            return {"messages": completion_messages, "prev_agent": "patch_verifier", "next_agent": "supervisor"}
        else: # go to exec verifier -> supervisor 
            for item in completion_messages:
                item["control_experiment_results_filename"] = self.get_control_experiment_results_filename(item["plan_id"], item["group"], item["partition_name"])
            completion_messages = verifier.exec_verifier(completion_messages)
            self.write_all_control_experiment_results_filenames(completion_messages)
            for task_details in completion_messages:
                self.assign_verifier("analyzer", task_details)
            return {"messages": completion_messages, "prev_agent": "exec_verifier", "next_agent": "analyzer"}

    def handle_analyzer(self, verifier_name: str):
        """
            Analyzer has completed a run. 
            We will now:
            - remove the analyzer from the analyzer assignment dict. 
            - assign to concluder (conditionally).
        """
        self.curie_logger.info("------------Entering handle analyzer!!!------------")
        # Get plan id and partition names assigned to verifier name:
        assignments = self.get_verifier_assignment(verifier_name) # format: [(plan_id1, partition_name1), (plan_id2, partition_name2), ...]

        completion_messages = [] # format: [{"plan_id": plan_id1, "partition_name": partition_name1, "is_correct": True, "verifier_log_message": "no error"}, ...]

        # Assert that all assigned partition names are now done
        has_false = False # if there exist one workflow that is considered incorrect by the verifier.
        for assignment in assignments:
            plan_id, group, partition_name = assignment["plan_id"], assignment["group"], assignment["partition_name"]
            # Check if the verifier has written to the verifier_wrote_list:
            item = self.get_verifier_wrote_list_item(verifier_name, plan_id, group, partition_name)
            if item is None:
                self.curie_logger.info("Warning: Analyzer has not written plan_id {}, group {}, partition_name {} to analyzer_wrote_list yet. We will rerun analyzer.".format(plan_id, group, partition_name))
                return {"messages": assignments, "next_agent": "analyzer"}
            completion_messages.append(item)
            if not item["no_change"]:
                has_false = True

        # Remove verifier from verifier assignment dict:
        self.unassign_verifier_all(verifier_name)

        # Remove from verifier_wrote_list:
        self.remove_verifier_wrote_list_all(verifier_name)

        # utils.print_workspace_contents()

        self.curie_logger.info("------------Exiting handle analyzer!!!------------")
        # NOTE: currently because I don't think divergent parallel execution is possible, we will just return to supervisor if even one workflow is considered incorrect (even though there may be others that are correct which we can in principle forward to the exec_verifier)
        # Inform supervisor that verifier has completed a run:
        is_terminate = self.check_exp_termination_condition()
        if not has_false and is_terminate: # go to concluder -> supervisor 
            return {"messages": [], "prev_agent": "analyzer", "next_agent": "concluder"}
        else:
            return {"messages": completion_messages, "prev_agent": "analyzer", "next_agent": "supervisor"}

    def handle_concluder(self, verifier_name: str):
        """
            Concluder has completed a run. 
            We will now:
            - remove the concluder from the concluder assignment dict. 
            - assign to supervisor.
        """
        self.curie_logger.info("------------Entering handle concluder!!!------------")
        # Get plan id and partition names assigned to verifier name:
        assignments = self.get_verifier_assignment(verifier_name) # format: [(plan_id1, partition_name1), (plan_id2, partition_name2), ...]

        completion_messages = [] # format: [{"plan_id": plan_id1, "partition_name": partition_name1, "is_correct": True, "verifier_log_message": "no error"}, ...]

        # Assert that all assigned partition names are now done
        item = self.get_concluder_wrote_list_item()
        if item is None:
            self.curie_logger.info("Warning: Concluder has not written to concluder_wrote_list yet. We will rerun concluder.")
            return {"messages": [], "next_agent": "concluder"}

        # Remove verifier from verifier assignment dict:
        self.unassign_verifier_all(verifier_name)

        # Remove from verifier_wrote_list:
        self.remove_verifier_wrote_list_all(verifier_name)

        # utils.print_workspace_contents()

        self.curie_logger.info("------------Exiting handle concluder!!!------------")
        # NOTE: currently because I don't think divergent parallel execution is possible, we will just return to supervisor if even one workflow is considered incorrect (even though there may be others that are correct which we can in principle forward to the exec_verifier)
        # Inform supervisor that verifier has completed a run:
        return {"messages": item, "prev_agent": "concluder", "next_agent": "supervisor"}

    def update_queues(
        self, 
        plan_id, 
        redo_details: Annotated[dict, "If this is not None, this means that the supervisor has requested that this partition be redone, and the error feedback is provided."]=None,
        assert_new_control: Annotated[bool, "If true, this means that the control group was just completed, meaning all experimental groups should not be completed."]=False
    ):
        """Given a plan ID, this function will:
            - if control group is done: add experimental groups that don't yet exist in the worker queue, or modify existing groups as needed. Remove plan from standy list if exist. 
            - if control group is not done: add control group that don't yet exist in the control worker queue, or modify existing control group as needed. Add plan to standby list, or modify existing as needed. 
        """
        self.curie_logger.info("------------ Update Queues ------------")
        plan = self.store.get(self.plan_namespace, plan_id).dict()["value"]
        self.curie_logger.info(f"Plan is: {utils.pretty_json(plan)} ")

        # First, if control group is not done:
        if plan["control_group"]['partition_1']["done"] == False: # only 1 partition for now in control group
            self.curie_logger.info("Control group is not done..")
            # Add plan to control queue if not exist or modify existing control group in queue as needed:
            partition_name = "partition_1" # Only 1 partition for now in control group
            if redo_details:
                # partition_name = redo_details["partition_name"]
                assert redo_details["group"] == "control_group"
                assert redo_details["partition_name"] == partition_name # NOTE: there is only one control group partition now, so we are certain that the error feedback will be directed to partition_1
                task_details = {
                    "priority": int(plan["priority"]),
                    "plan_id": plan_id,
                    "group": "control_group",
                    "partition_name": partition_name,
                    "workspace_dir": self.get_workspace_dirname(plan_id),
                    "error_feedback": redo_details["error_feedback"]
                }
            else:
                task_details = {
                    "priority": int(plan["priority"]),
                    "plan_id": plan_id,
                    "group": "control_group",
                    "workspace_dir": self.get_workspace_dirname(plan_id),
                    "partition_name": partition_name,
                }

            self.insert_control_worker_queue(task_details)

            # Add plan to standby list if not exist: 
            self.insert_standby_exp_plan_list(plan_id)
            self.curie_logger.info(f"Current control group worker queue: {self.get_control_worker_queue()}")
        else: # Second, if control group is done:
            self.curie_logger.info("Control group is done..")
            # Remove plan from standby list if exist:
            self.remove_standby_exp_plan_list(plan_id)

            # Add new experimental groups to worker queue or modify existing groups in queue as needed: 
            pq = self.metadata_store.get(self.sched_namespace, "worker_queue").dict()["value"]
            
            all_groups = self.get_groups_from_plan(plan["experimental_group"])

            self.curie_logger.info(f"All experimental group's partitions are: {all_groups}")

            if redo_details: # if redo partition, only the partition needs to be added to queue
                task_details = {
                    "priority": int(plan["priority"]),
                    "plan_id": plan_id,
                    "group": "experimental_group",
                    "partition_name": redo_details["partition_name"],
                    "workspace_dir": self.get_workspace_dirname(plan_id),
                    "error_feedback": redo_details["error_feedback"]
                }
                self.insert_worker_queue(task_details)
            else:
                # If not redo_partition, this means that all experimental groups should be added to queue 
                for partition_name in all_groups:
                    # We do not modify partitions that are already done: (they wouldn't be in the queue anyway)
                    if plan["experimental_group"][partition_name]["done"] == True:
                        if assert_new_control:
                            raise RuntimeError("Control group just completed done, therefore no experimental groups should be done yet.")
                        continue

                    # Insert into queue:
                    task_details = {
                        "priority": int(plan["priority"]),
                        "plan_id": plan_id,
                        "group": "experimental_group",
                        "workspace_dir": self.get_workspace_dirname(plan_id),
                        "partition_name": partition_name,
                    }
                    self.insert_worker_queue(task_details)
            self.curie_logger.info(f"Current worker queue: {self.get_worker_queue()}")
        
    def get_groups_from_plan(self, group_dict: dict) -> int:
        # Obtains group (either experimental_group or control_group) partitions from the plan
        # pattern = r"experimental_group_partition_\d+(?!_done)"

        # Collect experimental groups matches from the list
        matches = []
        for key in group_dict: # we know for sure that key will be partition_<number> (this is guaranteed by formatter)
            matches.append(key)
        return matches

    def get_worker_group_type(self, worker_name: str) -> str:
        if worker_name in settings.list_control_worker_names():
            return "control"
        elif worker_name in settings.list_worker_names():
            return "experimental"
        else:
            raise ValueError("Worker name not found in list of workers.")
    
    def has_idle_worker(self, group_type: str) -> (bool, str):
        memory_id = self.get_assignment_dict_mem_id(group_type)

        assignment_dict = self.metadata_store.get(self.sched_namespace, memory_id).dict()["value"]
        for name in assignment_dict:
            if not assignment_dict[name] or len(assignment_dict[name]) < settings.PARTITIONS_PER_WORKER:
                return True, name
        return False, None

    def assign_worker(self, group_type: str) -> dict:
        assignment_messages = defaultdict(list) # {worker_1: [{plan_id: xyz, partition_name: xyz}], worker_2: [{plan_id: xyz, partition_name: xyz}]}
        while True: # Keep assigning until no more idle workers
            has_idle_worker, worker_name = self.has_idle_worker(group_type)
            if has_idle_worker:
                task_details = self.pop_worker_queue(group_type)
                if task_details: # if queue is not empty
                    priority, plan_id, group, partition_name = task_details["priority"], task_details["plan_id"], task_details["group"], task_details["partition_name"]
                    self._assign_worker(worker_name, task_details, group_type)
                    if "error_feedback" in task_details:
                        task_details["error_feedback"] = self.augment_redo_partition_error_feedback(task_details)
                    assignment_messages[worker_name].append(task_details) # only the worker itself needs to be passed in context that may include an error feedback
                else:
                    break # no more plans in queue
            else:
                break

        return assignment_messages

    def augment_redo_partition_error_feedback(self, task_details: dict) -> str:
        self.curie_logger.info("------------Entering augment redo partition error feedback!!!------------")
        plan_id = task_details["plan_id"]
        group = task_details["group"]
        partition_name = task_details["partition_name"]
        error_feedback = task_details["error_feedback"]
        # Get plan from plan_id:
        plan = self.store.get(self.plan_namespace, plan_id).dict()["value"]
        # Get control experiment filename:
        control_experiment_filename = plan[group][partition_name]["control_experiment_filename"]
        self.curie_logger.info("------------Exiting augment redo partition error feedback!!!------------")
        # Return augmented error feedback:
        if not control_experiment_filename:
            return error_feedback
        else:
            return error_feedback + " Consider reusing the existing experiment workflow setup that you made earlier as your starting point, but feel free to start from scratch if you believe it can't be fixed or salvaged. The workflow filename is: {}".format(control_experiment_filename)

    
    def _assign_worker(self, worker_name: str, assignment_dict: dict, group_type: str):
        memory_id = self.get_assignment_dict_mem_id(group_type)
        self._assign_to_entity(worker_name, assignment_dict, memory_id)
    
    def assign_verifier(self, verifier_name, assignment_dict: dict):
        memory_id = self.get_assignment_dict_mem_id(verifier_name)
        self._assign_to_entity(verifier_name, assignment_dict, memory_id)

    def _assign_to_entity(self, entity_name: str, assignment_dict: dict, memory_id: str):
        overall_assignment_dict = self.metadata_store.get(self.sched_namespace, memory_id).dict()["value"]
        overall_assignment_dict[entity_name].append(assignment_dict)
        self.metadata_store.put(self.sched_namespace, memory_id, overall_assignment_dict)

    def get_worker_assignment(self, worker_name: str) -> list:
        group_type = self.get_worker_group_type(worker_name)
        memory_id = self.get_assignment_dict_mem_id(group_type)
        return self._get_entity_assignment(worker_name, memory_id)
    
    def get_verifier_assignment(self, verifier_name: str) -> list:
        memory_id = self.get_assignment_dict_mem_id(verifier_name)
        return self._get_entity_assignment(verifier_name, memory_id) # format: [(plan_id1, "experimental_group", "partition_1"), (plan_id2, "experimental_group", "partition_1"), ...]

    def _get_entity_assignment(self, entity_name, memory_id):
        assignment_dict = self.metadata_store.get(self.sched_namespace, memory_id).dict()["value"]
        return assignment_dict[entity_name]
    
    def unassign_worker_all(self, worker_name: str):
        """ Unassign all groups from a worker. """
        group_type = self.get_worker_group_type(worker_name)
        memory_id = self.get_assignment_dict_mem_id(group_type)
        self._unassign_entity_all(worker_name, memory_id)
    
    def unassign_verifier_all(self, verifier_name: str):
        memory_id = self.get_assignment_dict_mem_id(verifier_name)
        self._unassign_entity_all(verifier_name, memory_id)
    
    def _unassign_entity_all(self, entity_name: str, memory_id: str):
        assignment_dict = self.metadata_store.get(self.sched_namespace, memory_id).dict()["value"]
        assignment_dict[entity_name] = []
        self.metadata_store.put(self.sched_namespace, memory_id, assignment_dict)

    def pop_worker_queue(self, group_type: str) -> (dict):
        # https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore.get
        if group_type == "experimental":
            pq = self.metadata_store.get(self.sched_namespace, "worker_queue").dict()["value"]
        elif group_type == "control":
            pq = self.metadata_store.get(self.sched_namespace, "control_worker_queue").dict()["value"]
        if pq:
            _, _, _, _, task_details = heapq.heappop(pq)
            if group_type == "experimental":
                self.metadata_store.put(self.sched_namespace, "worker_queue", pq)
            elif group_type == "control":
                self.metadata_store.put(self.sched_namespace, "control_worker_queue", pq)
            return task_details
        else:
            return None
    
    def insert_worker_queue(self, task_details: dict):
        pq = self.metadata_store.get(self.sched_namespace, "worker_queue").dict()["value"]

        priority, plan_id, group, partition_name = task_details["priority"], task_details["plan_id"], task_details["group"], task_details["partition_name"]

        # Check if plan is in queue (linear time). If plan exists, remove it from queue. Note we need to do this because there could be a change in priority even if the plan is already in the queue. 
        for index in range(len(pq)):
            _, _, _, _, task_details2 = pq[index]
            priority2, plan_id2, group2, partition_name2 = task_details2["priority"], task_details2["plan_id"], task_details2["group"], task_details2["partition_name"]
            if plan_id == plan_id2 and group == group2 and partition_name == partition_name2:
                pq.pop(index)
                break # there should be only one such an instance anyway

        heapq.heappush(pq, (priority, plan_id, group, partition_name, task_details)) # heapq needs priority and the other fields except for task_details since its a dict, to make comparison
        self.metadata_store.put(self.sched_namespace, "worker_queue", pq)

    def insert_control_worker_queue(self, task_details: dict):
        pq = self.metadata_store.get(self.sched_namespace, "control_worker_queue").dict()["value"] # format: [(priority, plan_id1)]

        priority, plan_id, group, partition_name = task_details["priority"], task_details["plan_id"], task_details["group"], task_details["partition_name"]

        # Check if plan is in queue (linear time). If plan exists, remove it from queue. Note we need to do this because there could be a change in priority even if the plan is already in the queue. 
        for index in range(len(pq)):
            _, _, _, _, task_details2 = pq[index]
            priority2, plan_id2, group2, partition_name2 = task_details2["priority"], task_details2["plan_id"], task_details2["group"], task_details2["partition_name"]
            if plan_id == plan_id2 and group == group2 and partition_name == partition_name2:
                pq.pop(index)
                break # there should be only one such an instance anyway

        heapq.heappush(pq, (priority, plan_id, group, partition_name, task_details))
        self.metadata_store.put(self.sched_namespace, "control_worker_queue", pq)

    def get_worker_queue(self):
        pq = self.metadata_store.get(self.sched_namespace, "worker_queue").dict()["value"]
        return pq

    def get_control_worker_queue(self):
        pq = self.metadata_store.get(self.sched_namespace, "control_worker_queue").dict()["value"] # format: [(priority, plan_id1)]
        return pq

    def write_all_control_experiment_results_filenames(self, completion_messages: list):
        # for the format of completion_messages, see tool.LLMVerifierWriteTool for the format of this list
        for item in completion_messages:
            plan_id, group, partition_name = item["plan_id"], item["group"], item["partition_name"]
            plan = self.store.get(self.plan_namespace, plan_id).dict()["value"]
            filename = self.get_all_control_experiment_results_filename(plan_id, group, partition_name)
            with open(filename, 'w') as file:
                file.write(item["verifier_log_message"])
            plan[group][partition_name]["all_control_experiment_results_filename"] = filename
            self.store.put(self.plan_namespace, plan_id, plan)

    def insert_standby_exp_plan_list(self, plan_id: str):
        standby_exp_plan_list = self.metadata_store.get(self.sched_namespace, "standby_exp_plan_list").dict()["value"]

        # Check if plan is in queue (linear time). If plan exists, remove it from queue. Note we need to do this because there could be a change in priority even if the plan is already in the queue. 
        for index in range(len(standby_exp_plan_list)):
            plan_id2 = standby_exp_plan_list[index]
            if plan_id == plan_id2:
                standby_exp_plan_list.pop(index)
                break # there should be only one such an instance anyway

        standby_exp_plan_list.append(plan_id)
        self.metadata_store.put(self.sched_namespace, "standby_exp_plan_list", standby_exp_plan_list)

    def remove_standby_exp_plan_list(self, plan_id: str):
        standby_exp_plan_list = self.metadata_store.get(self.sched_namespace, "standby_exp_plan_list").dict()["value"]

        # Check if plan is in list (linear time). If plan exists, remove it from list. 
        for index in range(len(standby_exp_plan_list)):
            plan_id2 = standby_exp_plan_list[index]
            if plan_id == plan_id2:
                standby_exp_plan_list.pop(index)
                break
        
        self.metadata_store.put(self.sched_namespace, "standby_exp_plan_list", standby_exp_plan_list)

    def get_verifier_wrote_list_item(self, verifier_name:str, plan_id: str, group: str, partition_name: str):
        memory_id = self.get_wrote_list_mem_id(verifier_name)

        verifier_wrote_list = self.metadata_store.get(self.sched_namespace, memory_id).dict()["value"]
        for item in verifier_wrote_list:
            if item["plan_id"] == plan_id and item["group"] == group and item["partition_name"] == partition_name:
                return item
        return None

    def get_concluder_wrote_list_item(self):
        memory_id = str("concluder_wrote_list")
        verifier_wrote_list = self.metadata_store.get(self.sched_namespace, memory_id).dict()["value"]
        return verifier_wrote_list

    def remove_verifier_wrote_list_all(self, verifier_name: str):
        memory_id = self.get_wrote_list_mem_id(verifier_name)
        self.metadata_store.put(self.sched_namespace, memory_id, [])

    def get_control_experiment_filename(self, plan_id: str, group: str, partition_name: str) -> str:
        return "/workspace/{}_{}/control_experiment_{}_{}_{}.sh".format(self.config['workspace_name'], plan_id, plan_id, group, partition_name)

    def get_control_experiment_results_filename(self, plan_id: str, group: str, partition_name: str) -> str:
        return "/workspace/{}_{}/results_{}_{}_{}.txt".format(self.config['workspace_name'], plan_id, plan_id, group, partition_name)

    def get_all_control_experiment_results_filename(self, plan_id: str, group: str, partition_name: str) -> str:
        # results for multiple runs (i.e., a single run by exec verifier for now) for a single partition
        return "/workspace/{}_{}/all_results_{}_{}_{}.txt".format(self.config['workspace_name'], plan_id, plan_id, group, partition_name)

    def get_workspace_dirname(self, plan_id: str) -> str:
        return "/workspace/{}_{}".format(self.config["workspace_name"], plan_id)

    def is_no_plan_exists(self):
        items = self.store.search(self.plan_namespace)
        self.curie_logger.info(f"ITEMS SIS: {items}, with len: {len(items)}")
        plans_list = [item.dict()["value"] for item in items]
        if len(plans_list) == 0:
            return True
        else:
            return False

    def check_exp_termination_condition(self):
        """
        If all control and experimental groups are done for all plans, return True. Otherwise, return False.
        """
        items = self.store.search(self.plan_namespace)

        plans_list = [item.dict()["value"] for item in items]

        for plan in plans_list:
            for partition_name in plan["control_group"]:
                if plan["control_group"][partition_name]["done"] == False:
                    return False
            for partition_name in plan["experimental_group"]:
                if plan["experimental_group"][partition_name]["done"] == False:
                    return False
        return True

    def get_sched_namespace(self):
        user_id = "admin"
        application_context = "exp-sched" 
        sched_namespace = (user_id, application_context) # just a random namespace name for now
        return sched_namespace
    
    def get_plan_namespace(self):
        user_id = "admin"
        application_context = "exp-plans" 
        plan_namespace = (user_id, application_context) # just a random namespace name for now
        return plan_namespace

    def get_wrote_list_mem_id(self, verifier_name: str) -> str:
        if verifier_name == "llm_verifier":
            memory_id = str("llm_verifier_wrote_list") # check tool.LLMVerifierWriteTool for the format of this list
        elif verifier_name == "patch_verifier":
            memory_id = str("patch_verifier_wrote_list")
        elif verifier_name == str("analyzer"):
            memory_id = str("analyzer_wrote_list")
        elif verifier_name == str("concluder"):
            memory_id = str("concluder_wrote_list")
        return memory_id
    
    def get_assignment_dict_mem_id(self, entity_name: str) -> str:
        if entity_name == "llm_verifier":
            memory_id = "llm_verifier_assignment_dict"
        elif entity_name == "exec_verifier":
            memory_id = "exec_verifier_assignment_dict"
        elif entity_name == "patch_verifier":
            memory_id = "patch_verifier_assignment_dict"
        elif entity_name == "analyzer":
            memory_id = "analyzer_assignment_dict"
        elif entity_name == "concluder":
            memory_id = "concluder_assignment_dict"
        elif entity_name == "experimental":
            memory_id = str("worker_assignment_dict")
        elif entity_name == "control":
            memory_id = str("control_worker_assignment_dict")
        return memory_id

    def init_new_plan(self, plan_ids: list):
        for plan_id in plan_ids:
            new_plan = self.create_workspace_dir(plan_id)
            if new_plan:
                # Add "workspace_dir" attribute to plan
                self.add_workspace_to_plan(plan_id)
                # Edit plan question:
                self.edit_plan_question(plan_id)
    
    def create_workspace_dir(self, plan_id: str):
        # If we are running a question from Curie benchmark (specified in config["workspace_name"]), copy its associated starter files from ../starter_file and move it to ../workspace. 
        # Otherwise, if running a question not from Curie benchmark, we assume that starter_file does not exist, and we do not copy. We still create the new_starter_file_dir folder but leave it empty. 
        new_starter_file_dir = f"../workspace/{self.config['workspace_name']}_{plan_id}"
        new_starter_file_dir = os.path.abspath(new_starter_file_dir) + "/"
        old_starter_file_dir = f"../starter_file/{self.config['workspace_name']}" 
        old_starter_file_dir = os.path.abspath(old_starter_file_dir) + "/"
        if not os.path.exists(new_starter_file_dir):
            os.makedirs(new_starter_file_dir)
            os.chmod(new_starter_file_dir, 0o777)
            try:
                if os.path.exists(old_starter_file_dir): 
                    # This will copy only the contents of old_starter_file_dir into new_starter_file_dir, not the directory itself.
                    subprocess.run(["cp", "-r", old_starter_file_dir + ".", new_starter_file_dir], check=True)
                    # output = shell_tool.run({"commands": [f"cp -r {old_starter_file_dir} {new_starter_file_dir}"]})
                    self.curie_logger.info(f"Created {new_starter_file_dir}. Starter files from {old_starter_file_dir} copied successfully!")
                else:
                    self.curie_logger.info(f"Created {new_starter_file_dir}. No starter files to copy.")
            except Exception as e:
                self.curie_logger.info(f"Error copying files: {e}")
                raise

            return True
        else:
            return False

    def add_workspace_to_plan(self, plan_id: str):
        plan = self.store.get(self.plan_namespace, plan_id).dict()["value"]
        plan["workspace_dir"] = self.get_workspace_dirname(plan_id)
        self.store.put(self.plan_namespace, plan_id, plan)

    def edit_plan_question(self, plan_id: str):
        """
            Edit plan question to point to the correct writable workspace directory, that the technician agents are able to tweeak/modify. 
        """
        plan = self.store.get(self.plan_namespace, plan_id).dict()["value"]

        plan["question"] = plan["question"].replace(f"/starter_file/{self.config['workspace_name']}", self.get_workspace_dirname(plan_id))

        self.store.put(self.plan_namespace, plan_id, plan)

    def get_question(self):
        memory_id = str("question")
        question = self.metadata_store.get(self.sched_namespace, memory_id)
        question = question.dict()["value"]
        return question