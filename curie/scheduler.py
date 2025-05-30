from langchain_core.tools import tool
from typing import Annotated, List
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import InjectedStore
from langgraph.prebuilt import InjectedState
from langgraph.graph import END
import shutil
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

import model
from langchain_core.messages import HumanMessage, SystemMessage 
import formatter
import settings
import utils
from modified_deps.langchain_bash.tool import ShellTool
from logger import init_logger

class SchedNode():
    def __init__(self, store: InMemoryStore, metadata_store: InMemoryStore, State, config: dict):
        self.store = store
        self.metadata_store = metadata_store
        self.State = State
        self.sched_namespace = self.get_sched_namespace()
        self.plan_namespace = self.get_plan_namespace()
        self.config = config
        log_filename = f'../{config["log_filename"]}'
        self.curie_logger = init_logger(log_filename)
 
        self.workspace_dirname = ( self.config["workspace_name"] or 
                          self.config["job_name"] or 
                          "project" )   
        self.setup_sched()

    def setup_sched(self):
        memory_id = str("worker_assignment_dict") # Format of this dict: {"worker_name": [(exp_plan_id1, "experimental_group_partition_1"), (exp_plan_id2, "experimental_group_partition_1"), ...]}
        assignment_dict = {}
        for name in settings.list_worker_names():
            assignment_dict[name] = []
        self.metadata_store.put(self.sched_namespace, memory_id, assignment_dict)

        memory_id = str("control_worker_assignment_dict") # Format of this dict: {"worker_name": [(exp_plan_id1, "control_group"), (exp_plan_id2, "control_group"), ...]}
        assignment_dict = {}
        for name in settings.list_control_worker_names():
            assignment_dict[name] = []
        self.metadata_store.put(self.sched_namespace, memory_id, assignment_dict)

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

        create_assignment_dict(memory_ids, self.metadata_store, self.sched_namespace)

        # Lists to track various workflow tool interactions
        write_lists = [
            # Records calls to exp_plan_write by the supervisor.
            "supervisor_wrote_list",  # [exp_plan_id1, exp_plan_id2, ...]
            # Records calls to workflow_verified_record by LLM verifier.
            "llm_verifier_wrote_list",  # [{"plan_id": ..., "partition_name": ..., "is_correct": ..., "verifier_log_message": ...}, ...]
            # Records calls to workflow_verified_record by patch verifier.
            "patch_verifier_wrote_list",  # Same format as above
            # Records calls to workflow_verified_record by analyzer.
            "analyzer_wrote_list",  # Same format as above
            # Records calls to workflow_verified_record by concluder.
            "concluder_wrote_list",  # Same format as above
            # Records calls to redo_exp_partition by supervisor.
            "supervisor_redo_partition_list",  # [{"plan_id": ..., "group": ..., "partition_name": ..., "error_feedback": ...}, ...]
            "standby_exp_plan_list",  
            "user_router_wrote_list",  
        ]


        queues = [
            "worker_queue", # Format of this queue: [(priority, exp_plan_id1, "experimental_group_partition_2"), (priority, exp_plan_id2, "experimental_group_partition_2")] # Priority queue implemented using min heap 
            "control_worker_queue"
        ]

        def create_empty_list(memory_ids, metadata_store, sched_namespace):
            for memory_id in memory_ids:
                metadata_store.put(sched_namespace, memory_id, [])

        create_empty_list(write_lists, self.metadata_store, self.sched_namespace)
        create_empty_list(queues, self.metadata_store, self.sched_namespace)
    
    def create_SchedNode_subgraph(self, sched_tool):    
        """
            No LLM involved. Manual flow control. Reason: tools cannot modify state directly (and workaround are too troublesome https://github.com/langchain-ai/langgraph/discussions/1247) which we need to control which nodes to call next. 

            Workflow:
            - Supervisor -> SchedNode -> sched_tool -> SchedNode -> Worker X (multiple are possible. We need async too TODO)
            - Worker X -> SchedNode -> sched_tool -> SchedNode -> Supervisor
            - Note that supervisor to supervisor and worker to worker transitions are technically possible too, it is up to the sched_tool. 
        """

        def SchedNode(state: self.State):
            # Invoke sched_tool: 
            response = sched_tool.invoke({"state":state}) # response is guaranteed to be a dict
            # Based on sched_tool, response, we decide which node to call next:
            if "next_agent" in response and response["next_agent"] == END: # terminate experiment. Langgraph will call END based on next_agent as defined in our conditional_edge in main.py
                self.write_down_exp_plan()
                return {"messages": state["messages"], "next_agent": END, "prev_agent": "sched_node", "remaining_steps_display": state["remaining_steps"]}
            elif "next_agent" not in response: # next agent is a worker, and we need to determine which worker it will be.
                # TODO: not doing parallelism for now, so we just assume for instances that messages will only contain one worker's name, and we will only need to call that one worker. Parallelism will be implemented later.
                control_empty = not response["control_work"]["messages"]
                experimental_empty = not response["experimental_work"]["messages"]
                assert control_empty != experimental_empty # only one of them should be non-empty. in our current iteration, since there is only one plan (I hope), this means one of these should be empty, since we cannot have that same plan not having a control group, but having some experimental groups to be run, and vice versa. TODO: thus, we will need to change this and the following lines later to accommodate more than one plan existing
                
                if response["experimental_work"]["messages"]:
                    type_name = "experimental_work"
                elif response["control_work"]["messages"]:
                    type_name = "control_work"
                assert len(list(response[type_name]["messages"].keys())) == 1 # only one worker for a given worker type has been assigned partitions to run
                self.write_down_exp_plan()
                return {"messages": [
                            HumanMessage(content=json.dumps(list(response[type_name]["messages"].values())[0]), name="scheduler")
                        ], 
                    "prev_agent": "sched_node", 
                    "next_agent": list(response[type_name]["messages"].keys())[0],
                    "remaining_steps_display": state["remaining_steps"],
                } # next_agent: worker_1
            else:
                new_dict = response.copy()
                new_dict["messages"] = [HumanMessage(content=str(response["messages"]), name="scheduler")]
                new_dict["prev_agent"] = "sched_node"
                new_dict["remaining_steps_display"] = state["remaining_steps"]
                # print(new_dict)
                # if "reset_steps" in response and response["reset_steps"] == True:
                #     new_dict["remaining_steps"] = state["remaining_steps"] + 1
                #     new_dict["remaining_steps_display"] = new_dict["remaining_steps"]
                return new_dict
        
        return SchedNode

    def write_down_exp_plan(self):

        items = self.store.search(self.plan_namespace)
        plans = [item.dict()["value"] for item in items]

        filename = self.config['exp_plan_filename'].split("/")[-1].replace(".txt", ".json")
        dirname = self.config['log_filename'].split("/")[:-1]
        with open('/' + '/'.join(dirname) + '/' + filename, 'w') as file:
            for plan in plans:
                file.write(json.dumps(plan) + "\n")

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
        self.curie_logger.debug("------------ Update Queues ------------")
        plan = self.store.get(self.plan_namespace, plan_id).dict()["value"]
        self.curie_logger.info(f"Plan is: {utils.pretty_json(plan)} ")

        # First, if control group is not done:
        if plan["control_group"]['partition_1']["done"] == False: # only 1 partition for now in control group
            self.curie_logger.debug("Control group is not done..")
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
            self.curie_logger.debug(f"Current control group worker queue: {self.get_control_worker_queue()}")
        else: # Second, if control group is done:
            self.curie_logger.debug("Control group is done..")
            # Remove plan from standby list if exist:
            self.remove_standby_exp_plan_list(plan_id)

            # Add new experimental groups to worker queue or modify existing groups in queue as needed: 
            pq = self.metadata_store.get(self.sched_namespace, "worker_queue").dict()["value"]
            
            all_groups = self.get_groups_from_plan(plan["experimental_group"])

            self.curie_logger.debug(f"All experimental group's partitions are: {all_groups}")

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
            self.curie_logger.debug(f"Current worker queue: {self.get_worker_queue()}")
        
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
        return "/workspace/{}_{}/control_experiment_{}_{}_{}.sh".format(os.path.basename(self.workspace_dirname) , plan_id, plan_id, group, partition_name)

    def get_control_experiment_results_filename(self, plan_id: str, group: str, partition_name: str) -> str:
        return "/workspace/{}_{}/results_{}_{}_{}.txt".format(os.path.basename(self.workspace_dirname), plan_id, plan_id, group, partition_name)

    def get_all_control_experiment_results_filename(self, plan_id: str, group: str, partition_name: str) -> str:
        # results for multiple runs (i.e., a single run by exec verifier for now) for a single partition
        return "/workspace/{}_{}/all_results_{}_{}_{}.txt".format(os.path.basename(self.workspace_dirname) , plan_id, plan_id, group, partition_name)

    def get_workspace_dirname(self, plan_id: str) -> str: 
        return "/workspace/{}_{}".format(os.path.basename(self.workspace_dirname), plan_id) 

    def is_no_plan_exists(self):
        items = self.store.search(self.plan_namespace)
        self.curie_logger.debug(f"Plans that exist: {items}, with len: {len(items)}")
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
        elif "control" in entity_name:
            memory_id = str("control_worker_assignment_dict")
        else:
            raise ValueError(f"Entity name not found for {entity_name}")
        return memory_id
    
    def copy_dataset_to_workspace(self):
        if self.config["dataset_dir"] != "": 
            # Copy dataset to workspace
            dataset_dir = os.path.join('/all', self.config["dataset_dir"].lstrip('/').rstrip('/'))

            if not os.path.exists(dataset_dir):
                raise FileNotFoundError(f"Dataset directory does not exist: {self.config['dataset_dir']}. Please check the path.")            

            workspace_dir = "/workspace/" 
            dataset_dir_name = (os.path.basename(self.config["workspace_name"]) or 
                                os.path.basename(self.config["job_name"]) or
                                "project" ) + "_dataset"
            new_dataset_dir = os.path.join(workspace_dir, dataset_dir_name)
            dataset_name = dataset_dir.split("/")[-1]

            if os.path.isfile(dataset_dir):
                dataset_dir_size = os.path.getsize(dataset_dir)
            else:
                dataset_dir_size = sum(os.path.getsize(os.path.join(dataset_dir, f)) for f in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, f)))

            if os.path.exists(new_dataset_dir):
                if os.path.isfile(new_dataset_dir):
                    new_dataset_dir_size = os.path.getsize(new_dataset_dir)
                else:
                    new_dataset_dir_size = sum(os.path.getsize(os.path.join(new_dataset_dir, f)) for f in os.listdir(new_dataset_dir) if os.path.isfile(os.path.join(new_dataset_dir, f)))
            else:
                new_dataset_dir_size = 0
            
            if os.path.isfile(dataset_dir):
                if os.path.exists(new_dataset_dir):
                    shutil.rmtree(new_dataset_dir)

                # mkdir new_dataset_dir
                os.makedirs(new_dataset_dir, exist_ok=True)
                shutil.copy(dataset_dir, new_dataset_dir)
                self.curie_logger.info(f"Copying file {dataset_dir} --> {new_dataset_dir} successfully!") 

            elif os.path.isdir(dataset_dir):
                if not os.path.exists(new_dataset_dir) or ( os.path.exists(new_dataset_dir) and dataset_dir_size != new_dataset_dir_size):
                    if os.path.exists(new_dataset_dir): 
                        shutil.rmtree(new_dataset_dir) 

                    try:
                        subprocess.run(["cp", "-r", f"{dataset_dir}",  '/workspace'], check=True)
                        os.rename(os.path.join('/workspace', dataset_name), new_dataset_dir)
                        self.curie_logger.info(f"Copying {dataset_dir} --> {new_dataset_dir} successfully!") 
                    except Exception as e:
                        self.curie_logger.info(f"Error copying files: {e}")
                        raise
                else:
                    self.curie_logger.info(f"Dataset directory already exists: {new_dataset_dir}. Skipping copy.")

            return new_dataset_dir
        else:
            self.curie_logger.info(f"No dataset directory specified. Skipping dataset copy.")
            
    def init_new_plan(self, plan_ids: list):
        import concurrent.futures

        new_dataset_dir = self.copy_dataset_to_workspace()
        self.get_packages_to_install()

        def process_plan(plan_id):
            # FIXME: initialization time too long, trigger this when deciding to execute this plan
            self.curie_logger.info(f"Preparing environment for plan ID: {plan_id}")
            new_plan = self.create_workspace_dir(plan_id)
            if new_plan:
                # Add "workspace_dir" attribute to plan
                self.add_workspace_to_plan(plan_id)
                # Edit plan question:
                self.edit_plan_question(plan_id)
                self.add_dataset_to_plan(plan_id, new_dataset_dir)
            return plan_id
        
        # Execute the plans in parallel using a ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all tasks and get future objects
            futures = [executor.submit(process_plan, plan_id) for plan_id in plan_ids]
            
            # Wait for all futures to complete (optional)
            concurrent.futures.wait(futures)

    def init_coding_env(self, work_dir: str, env_name: str='venv'):
        
        def install_packages(env_path, packages):
            """
            Install specified packages into a micromamba environment
            
            Args:
                env_path (str): Path to the micromamba environment
                packages (list): List of packages to install
            """
            if not os.path.exists(env_path) or len(packages) == 0:
                return 
            # Activate the environment
            successful_packages = []
            failed_packages = []
            # start_time = time.time()
            for i, package in enumerate(packages):
                # Construct the installation command for the current package
                activate_cmd = [
                    "micromamba", "install", "-y", "--quiet",
                    "-p", env_path, package, '&'
                ]
                try:
                    # Run the installation command for the current package
                    result = subprocess.run(
                        activate_cmd, 
                        check=True, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE, 
                        text=True
                    )
                    successful_packages.append(package)
                    self.curie_logger.info(f"[{i}/{len(packages)}] Successfully installed {package}.")

                except subprocess.CalledProcessError as e:
                    try:
                        activate_cmd = [
                            "micromamba", "run", "-p", env_path,
                            "pip", "install", package
                        ]
                        result = subprocess.run(
                            activate_cmd, 
                            check=True, 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE, 
                            text=True
                        )
                        successful_packages.append(package)
                    except subprocess.CalledProcessError as e: 
                        failed_packages.append(package) 
                        self.curie_logger.info(f"Fail to install the packages {package}. Error: {e.stderr}")
    
            # self.curie_logger.info(f"Sucessfully install packages: {', '.join(successful_packages)} in {time.time() - start_time:.2f} seconds.")
            self.curie_logger.info(f"Sucessfully install packages: {', '.join(successful_packages)}.")

        # FIXME: some use cases may need old versions of Python 
        env_path = os.path.join(work_dir, env_name)
        if not os.path.exists(env_path):
            command = ["micromamba", "create", "-p", env_path, "python=3.12", "-y", "--quiet"]
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                # Install the packages
                install_packages(env_path, self.packages_to_install)
            except json.JSONDecodeError as e:
                self.curie_logger.info(f"No python package needs to be installed") 
        else:
            self.curie_logger.info(f"Environment is pre-built at {env_path}. Skipping creation.")

        return env_path

    def get_packages_to_install(self):
        with open('/'+self.config['exp_plan_filename'], "r") as file:  
            question = file.read() 
        with open("/curie/prompts/exp-env-manager.txt", "r") as file:
            system_prompt = file.read() 
            
        messages = [SystemMessage(content=system_prompt),
                HumanMessage(content=question)]
        response = model.query_model_safe(messages) 
        cleaned_response = response.content.replace('```json', '').replace('```', '').strip()
        try:
            self.packages_to_install  = json.loads(cleaned_response)["packages"] 
        except json.JSONDecodeError as e:
            self.curie_logger.info(f"No python package needs to be installed") 
            self.packages_to_install = []
    
    def create_workspace_dir(self, plan_id: str):
        # If we are running a question from Curie benchmark (specified in config["workspace_name"]), copy its associated starter files from ../starter_file and move it to ../workspace. 
        # Otherwise, if running a question not from Curie benchmark, we assume that starter_file does not exist, and we do not copy. We still create the new_starter_file_dir folder but leave it empty. 
        
        workspace_base_name = (os.path.basename(self.config["workspace_name"]) or
                               os.path.basename(self.config["job_name"]) or
                               "project" )
        new_starter_file_dir = f"../workspace/{workspace_base_name}_{plan_id}"
        if self.config["workspace_name"] != "":
            old_starter_file_dir = os.path.join('/all', self.config["workspace_name"].lstrip('/')) 
            new_starter_file_dir = os.path.abspath(new_starter_file_dir)  
        else:
            new_starter_file_dir = os.path.abspath(new_starter_file_dir)  
            old_starter_file_dir = None
        
        if not os.path.exists(new_starter_file_dir):
            
            try:
                if old_starter_file_dir and os.path.exists(old_starter_file_dir): 
                    # This will copy only the contents of old_starter_file_dir into new_starter_file_dir, not the directory itself.
                    subprocess.run(["cp", "-r", old_starter_file_dir,  new_starter_file_dir], check=True)

                    self.curie_logger.info(f"Created 📁 {new_starter_file_dir}. \
                                            Starter files from {old_starter_file_dir.replace('/all/', '')} copied successfully!")
                else:
                    self.curie_logger.info(f"Created 📁 {new_starter_file_dir}. No starter files to copy.")
                # FIXME: install environment for each plan_id -- too slow.
                env_path = self.init_coding_env(new_starter_file_dir)
                self.curie_logger.info(f"Micromamba environment created at {env_path}")

            except Exception as e:
                self.curie_logger.info(f"Error copying files: {e}")
                raise

            return True
        else:
            return False
        
    @staticmethod
    def write_at_beginning(filename, text_to_add): 
        try:
            with open(filename, "r") as file:
                existing_content = file.read()
        except FileNotFoundError:
            existing_content = ""   

        with open(filename, "w") as file:
            file.write(text_to_add)
            file.write(existing_content)

    def add_dataset_to_plan(self, plan_id: str, new_dataset_dir: str):
        # for plan_id in plan_id_list:
        plan = self.store.get(self.plan_namespace, plan_id).dict()["value"]
        plan["dataset_dir"] = new_dataset_dir
        self.store.put(self.plan_namespace, plan_id, plan) 
        description_file = os.path.join(plan["workspace_dir"], "description.md") 
        self.write_at_beginning(description_file,f"\nDataset directory: {plan['dataset_dir']}. \
                                Briefly explore the directory `ls * | head -n 50` to understand the structure.\n")
        
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

    def get_concluder_terminate_message(self):
        return "Your task: Some results are incomplete, but you must conclude the experiment now (make sure to set 'is_conclude' to True; 'concluder_log_message' must also make it clear that we must conclude now). Analyze the results per the instructions and provide a detailed breakdown of what the user could do on their own (e.g. propose follow-up experiments)."

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
    config: Optional[dict] = None
    curie_logger: Optional[logging.Logger] = None

    def __init__(self, store: InMemoryStore, metadata_store: InMemoryStore, config: dict):
        super().__init__()
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
        if state["remaining_steps"] <= settings.CONCLUDER_BUFFER_STEPS:
            if prev_agent != "concluder":
                self.curie_logger.info("------------ Not enough remaining steps to continue with experimentation. Entering concluder now!------------")
                return {"messages": [self.get_concluder_terminate_message()], "prev_agent": "analyzer", "next_agent": "concluder"} 
            else:
                self.curie_logger.info("------------ Not enough remaining steps to continue with experimentation. Handling concluder output before EXITING!!!------------")
        next_agent = self._handle_agent(prev_agent, state)
        next_agent_name = 'N/A'
        for k, v in next_agent.items():
            if k == 'next_agent':
                next_agent_name = v
                break
            if 'next_agent' in v:
                next_agent_name = v['next_agent']
                break
        
        self.curie_logger.info(
            f"<<<<<<<< Scheduling {prev_agent} ⏩ {next_agent_name} >>>>>>>>"
        )
        if next_agent_name == 'N/A':
            self.curie_logger.info(
                f"------- No next agent found for {next_agent} ---------"
            )
        
        return next_agent
    
    # TODO: currently assumes single nodes for each node type. Need to account for multiple nodes of same type.
    def _handle_agent(self, prev_agent: str, state: Annotated[dict, InjectedState]) -> str:
        """
        Route to appropriate handler based on previous agent.
        
        Args:
            prev_agent (str): Name of the previous agent
            
        Returns:
            str: Response from the appropriate handler
        """
        handlers = self.config["transition_funcs"]
        # print(handlers)
        
        # Handle special case for worker which has a prefix
        if "control_worker" in prev_agent:
            return handlers["control_worker"]()
        if "worker" in prev_agent:
            return handlers["worker"]()
        if "concluder" == prev_agent:
            return handlers["concluder"](state)
        if "supervisor" == prev_agent:
            return handlers["supervisor"](state)
        if "user_input_router" == prev_agent:
            return handlers["user_input_router"](state)
        if "user_input" == prev_agent:
            return handlers["user_input"](state)
        
        # Get the appropriate handler or raise an error if not found
        handler = handlers.get(prev_agent)
        if not handler:
            raise ValueError(f"No handler found for agent: {prev_agent}")
            
        return handler()
    
    def get_concluder_terminate_message(self):
        return "Your task: Some results are incomplete, but you must conclude the experiment now (make sure to set 'is_conclude' to True; 'concluder_log_message' must also make it clear that we must conclude now). Analyze the results per the instructions and provide a detailed breakdown of what the user could do on their own (e.g. propose follow-up experiments)."