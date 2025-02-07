# https://python.langchain.com/docs/how_to/custom_tools/
from langchain_core.tools import tool
from typing import Annotated, List
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import InjectedStore
from langgraph.prebuilt import InjectedState
import uuid
import time
from modified_deps.langchain_bash.tool import ShellTool
from typing import Optional, Type, Dict, Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

import formatter
import settings

import json
import re
import os

@tool
def test_search_tool(a: Annotated[str, "search string"]) -> str:
    """Searches for useful information"""
    return "University of Michigan!"

shell_tool = ShellTool(timeout=1200)

def extract_plan_id(prompt: str) -> str:
    """
    Extracts the plan ID from the given prompt.

    Args:
        prompt (str): The input text containing a plan ID.

    Returns:
        str: The extracted plan ID if found, else an empty string.
    """
    # Regular expression to match UUID-like patterns (plan_id format)
    pattern = r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"

    # Search for the pattern in the prompt
    match = re.search(pattern, prompt)
    
    # Return the matched plan ID if found, else return an empty string
    return match.group(0) if match else int(time.time())

def extract_partition_name(prompt: str) -> str:
    """
    Extracts the partition name from a given prompt.

    Args:
        prompt (str): The input text that may contain a partition name.

    Returns:
        str: The extracted partition name if found, else an empty string.
    """
    # Regular expression to match partition names (e.g., 'partition_1')
    pattern = r"['\"]?(partition_\d+)['\"]?"
    
    # Search for the pattern in the prompt
    match = re.search(pattern, prompt)
    
    # Return the matched partition name if found, else return an empty string
    return match.group(1) if match else "partition_0"

def extract_workspace_dir(text: str) -> str:
    """
    Extracts the directory name that appears after '/workspace/' in the given text,
    excluding any surrounding single quotes.

    Args:
        text (str): The input string containing the workspace path.

    Returns:
        str: The directory name after '/workspace/', or an empty string if not found.
    """
    match = re.search(r"/workspace/([^\s'/]+)", text)  # Excludes single quotes
    return match.group(1) if match else ""

@tool
def codeagent_openhands(prompt: Annotated[str, "The prompt to generate code for"] = 'Write a function print Hi.') -> str:
    """Write/Modify code and write the script for a given experimentation plan, and return experimental workflow script."""
    # given the code workspace 
    # wait; return log and result script 
    try:  
        # TODO: remove the hardcoded path
        # TODO: inform the new work dir
    
        current_dir = extract_workspace_dir(prompt)
        if not current_dir:
            return "Failed to extract the current workspace directory. please provide in the prompt in the format '/workspace/<workspace_dir>'"
        plan_id = extract_plan_id(prompt)
        partition_name = extract_partition_name(prompt)
        print(f"Plan ID: {plan_id}. Partition Name: {partition_name}")

        new_starter_file_dir = f"../workspace/{current_dir}_{plan_id}"
        old_starter_file_dir = f"../workspace/{current_dir}" 
        output = shell_tool.run({"commands": [f"cp -r {old_starter_file_dir} {new_starter_file_dir}"]})
        print(f"Output: {output}")

        # TODO: Put the system prompt into a file
        system_prompt = f'''You are a Coding Agent tasked with generating a fully reproducible experimental workflow based on the provided experiment plan. 
        Your working directory is /workspace/{current_dir}_{plan_id}, you should not touch files outside this directory.
        Key Requirements for OpenHands:
        - (1) Control Group:
            - Your primary responsibility is to construct the workflow, including the environment setup, and produce results for the control group.
        - (2) Experimental Group:
            - Represents test conditions where one or more independent variables differ from the control group.
            - Ensure the workflow supports seamless substitution of variables for experimental groups without requiring further modifications.
            - Achieve this by either: (i) Writing additional scripts or functions for the experimental groups. (ii) Modifying reusable parts of the workflow to parameterize control and experimental variables.

        Program Requirement: 
        The entire controlled experiment workflow (which may involve multiple scripts) must be callable through a single script named as "/workspace/control_experiment_{plan_id}_control_group_{partition_name}.sh", 
        This program must take no arguments and should handle the execution of the entire workflow for the control group. Include neccesary explanation assiciated with the plan in the script comment.
        The program must store the control group results in a single file named "/workspace/results_{plan_id}_control_group_{partition_name}.txt", 
        and what the results mean should be easily understood (e.g., including measurement units). You may develop and test smaller standalone programs initially, 
        Don't try to execute commands like "ls -R".
        Here is the experiment plan: '''

        prompt = f'''{system_prompt}\n{prompt}'''
        # write to a file
        prompt_file = f"../logs/tmp_prompt.txt"
        with open(prompt_file, "w") as file:
            file.write(prompt)
        
        # FIXME, remove organization for public use
        output = shell_tool.run({
            "commands": [
                f"export LOG_ALL_EVENTS=true; "
                f'sed -i "474i \          \'organization\': \'499023\'," /root/.cache/pypoetry/virtualenvs/openhands-ai-*-py3.12/lib/python3.12/site-packages/litellm/llms/azure/azure.py; '
                f"export WORKSPACE_BASE=/home/ubuntu/Curie/starter_file/; " # 
                f"chmod 777 -R {new_starter_file_dir}; "
                f"/root/.cache/pypoetry/virtualenvs/openhands-ai-*-py3.12/bin/python -m openhands.core.main -f {prompt_file} 2>&1 | tee -a /logs/openhands_{plan_id}_logging.txt; "
            ]
        })

        # copy the starter file outside the container to the new directory inside the container
        # FIXME: this does not support running outside the container.
        print(f"Output: {output}") 

    except BaseException as e:
        print(f"Error for openhands agent: {repr(e)}")
        return f"Failed to generate code for prompt: {prompt}\nError: {repr(e)}"
    return (f"Update the workspace dir to be /workspace/{current_dir}_{plan_id}\n"
            f"Design plan ID: {plan_id}, Partition Name: {partition_name}"
            f"The experiment workflow can be found in '/workspace/control_experiment_{plan_id}_control_group_{partition_name}.sh'\n"
            f"Control group results are stored in '/workspace/results_{plan_id}_control_group_{partition_name}.txt'\n"
            f"[Minor] Openhands logging can be found in '/logs/openhands_{plan_id}_logging.txt'"
            )
 
    # TODO: return the concise logs.

# Note: shell_tool itself can in theory be passed into the agent as a tool already https://python.langchain.com/docs/integrations/tools/bash/ https://www.youtube.com/watch?v=-ybgQK0BE-I
@tool
def execute_shell_command(
    command: Annotated[str, "The shell command to execute"]
    ) -> str:
    """Execute one line shell command and return the output."""
    try:
        # For all $ symbols in the command, that don't have a \ appended right before the $, add a \ right before the $ symbol:
        command = re.sub(r'(?<!\\)\$', r'\\$', command)
        # command = re.sub(r'(?<!\\)\n', r'\\n', command)

        if "ls -lR" in command or "ls -R" in command:
            return "Please don't use 'ls -lR' or 'ls -R' commands. They are not allowed, as they will cause you to exceed context length."

        output = shell_tool.run({"commands": [command]}) # only run one command at a time 
        # print(f"Command executed: {command}")
        print(f"Output: {output}")
    except BaseException as e:
        print(f"Error executing command: {command}")
        print(f"Error: {repr(e)}")
        return f"Failed to execute command: ```bash\n{command}\n```\nError: {repr(e)}"
    return f"Command executed: ```bash\n{command}\n```\nStdout: {output}"

@tool
def write_to_file(
    input_string: Annotated[str, "The string to write to the file"],
    file_path: Annotated[str, "The path of the file where the string will be written"]
) -> str:
    """Write a given string to a file line by line after processing it."""
    try:
        # Split the string into lines
        lines = input_string.splitlines()  # Preserve actual newlines

        with open(file_path, "w") as file:
            for line in lines:
                # Replace literal '\n' with '\\n'
                processed_line = line.replace("\n", "\\n")
                print(f"Writing line: {processed_line}")
                file.write(processed_line + "\n")

        return f"String written successfully to {file_path}"
    except BaseException as e:
        print(f"Error writing to file: {file_path}")
        print(f"Error: {repr(e)}")
        return f"Failed to write to file: {file_path}\nError: {repr(e)}"

@tool
def read_file_contents(
    filename: Annotated[str, "The absolute path of the file to read"]
    ) -> str:
    """
    Reads and returns the contents of a file.

    Parameters:
        filename (str): The absolute path of the file to read.

    Returns:
        str: The contents of the file as a string.
    """
    try:
        with open(filename, 'r') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return f"Error: The file '{filename}' does not exist."
    except PermissionError:
        return f"Error: Insufficient permissions to read the file '{filename}'."
    except Exception as e:
        return f"Error: An unexpected error occurred while reading the file: {str(e)}"

# Long term memory: https://blog.langchain.dev/launching-long-term-memory-support-in-langgraph/
# i.e., in memory store: https://langchain-ai.github.io/langgraph/concepts/persistence/#memory-store
# API reference: https://langchain-ai.github.io/langgraph/reference/store/#langgraph.store.base.BaseStore.get
class NewExpPlanStoreWriteInput(BaseModel):
    plan: formatter.NewExperimentalPlanResponseFormatter = Field(
        ...,
        description="Experimental plan dict to store in long-term storage. This field is required."
    )

# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class NewExpPlanStoreWriteTool(BaseTool):
    name: str = "write_new_exp_plan"
    description: str = "Write a new experimental plan to long term storage."
    args_schema: Type[BaseModel] = NewExpPlanStoreWriteInput
    # None of the following work:
    # https://langchain-ai.github.io/langgraph/how-tos/pass-run-time-values-to-tools/#define-the-tools_1
    # https://github.com/langchain-ai/langchain/discussions/24906
    # and so on..
    store: Optional[InMemoryStore] = None  # Declare store as an optional field
    metadata_store: Optional[InMemoryStore] = None  # Declare store as an optional field. This is for storing sched related metadata. 

    def __init__(self, store: InMemoryStore, metadata_store: InMemoryStore):
        super().__init__()
        self.store = store
        self.metadata_store = metadata_store
    
    class Config:
        arbitrary_types_allowed = True  # Allow non-Pydantic types like InMemoryStore

    def _run(
        self, 
        plan: formatter.NewExperimentalPlanResponseFormatter, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Use the tool.
        
        Note: we are guaranteed that plan will conform to the required format
        """
        user_id = "admin"
        application_context = "exp-plans" 
        namespace = (user_id, application_context) # just a random namespace name for now
        
        print("Writing new plan...")

        memory_id = str(uuid.uuid4())
        # Add metadata and reformat plan:
        plan_data = plan.dict()
        plan_data = self.add_plan_metadata(plan_data)

        # Add plan to store:
        plan_data["plan_id"] = memory_id
        self.async_notify_sched_modify_plan(memory_id)
        self.store.put(namespace, memory_id, plan_data)

        return memory_id
    
    def async_notify_sched_modify_plan(self, plan_id: str):
        user_id = "admin"
        application_context = "exp-sched" 
        sched_namespace = (user_id, application_context) # just a random namespace name for now

        memory_id = str("supervisor_wrote_list")
    
        wrote_list = self.metadata_store.get(sched_namespace, memory_id)
        wrote_list = wrote_list.dict()["value"] # https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/?h=item#langgraph_sdk.schema.RunCreate.multitask_strategy
        wrote_list.append(plan_id)
        self.metadata_store.put(sched_namespace, memory_id, wrote_list)

    def add_plan_metadata(self, plan_data: dict) -> dict:
        """ This partitions and adds required metadata to the plan. 
            Example modified plan: # this is the only plan format ever saved in long term store. 
            {...all fields except for control and experimental group, 
                "question": "What is the best instance type for a CPU workload?", # original user_input
                "control_group": {
                    "partition_1": {
                        "independent_vars": [{
                            "region": "us-east-1",
                            "instance_type": "t2.micro"
                        }],
                        "control_experiment_filename": "/workspace/control_experiment_<plan_id>.sh",
                        "control_experiment_results_filename": "/workspace/results_<plan_id>_control_group.txt",
                        "done": False, 
                        "error_feedback": "", 
                    }
                },
                "experimental_group": {
                    "partition_1": {
                        "independent_vars": [{
                            "region": "us-east-1",
                            "instance_type": "t2.micro"
                        }],
                        "control_experiment_filename": "/workspace/control_experiment_<plan_id>_experimental_group_partition_<partition_number>.sh",
                        "control_experiment_results_filename": "/workspace/results_<plan_id>_experimental_group_partition_<partition_number>.txt",
                        "done": False, 
                        "error_feedback": "", 
                    },
                    "partition_2": {
                        "independent_vars": [{
                            "region": "us-west-2",
                            "instance_type": "t2.micro"
                        }],
                        "control_experiment_filename": "/workspace/control_experiment_<plan_id>_experimental_group_partition_<partition_number>.sh",
                        "control_experiment_results_filename": "/workspace/results_<plan_id>_experimental_group_partition_<partition_number>.txt",
                        "done": False, 
                        "error_feedback": "", 
                    }
                }
            }

            (just for reference) Archived outdated plan format:
            {..., "experimental_group":{"vcpu":[1,2,3,4]}, "experimental_group_partition_1":{"vcpu":[1,2]}, "experimental_group_partition_2":{"vcpu":[3,4]}, "experimental_group_partition_1_done: False, "experimental_group_partition_2_done: False, control_group_done: False}
        """
        print("---------Enter add_plan_metadata---------")

        user_id = "admin"
        application_context = "exp-sched" 
        sched_namespace = (user_id, application_context) # just a random namespace name for now
        memory_id = str("question")
        question = self.metadata_store.get(sched_namespace, memory_id)
        question = question.dict()["value"]

        partitioned_plan_data = {"control_group": {}, "experimental_group": {}, "question": question}
        # Partition applied to experimental groups only for now:
        for key, value in plan_data.items():
            if key != "experimental_group" and key != "control_group":
                partitioned_plan_data[key] = value                
            else:
                group_type = key
                # print(key)
                num_exp_groups = len(plan_data[group_type]) # using the example, this would be 4
                i = 0
                partition_count = 0
                while i < num_exp_groups:
                    var_group = []
                    for j in range(settings.VARS_PER_PARTITION):  
                        if i + j >= num_exp_groups:
                            break
                        var_group.append(plan_data[group_type][j+i])

                    partitioned_plan_data[group_type][f"partition_{partition_count+1}"] = {
                        "independent_vars": var_group,
                        "control_experiment_filename": "",
                        "control_experiment_results_filename": "",
                        "all_control_experiment_results_filename": "",
                        "done": False,
                        # "error_feedback": ""
                    }
                    
                    i += settings.VARS_PER_PARTITION
                    partition_count += 1

        print("---------Exit add_plan_metadata---------")
        return partitioned_plan_data

class ExistingExpPlanStoreWriteInput(BaseModel):
    plan_id: str = Field(
        None,
        description="Plan ID of the specific experimental plan from storage."
    )
    plan: formatter.ExistingExperimentalPlanResponseFormatter = Field(
        ...,
        description="Experimental plan dict to store in long-term storage. This field is required."
    )

# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class ExistingExpPlanStoreWriteTool(BaseTool):
    name: str = "edit_existing_exp_plan"
    description: str = "Modify existing experimental plan in long term storage. Warning: This enables overwriting any field within the plan, try avoiding this tool if possible (if other suitable tools are available)."
    args_schema: Type[BaseModel] = ExistingExpPlanStoreWriteInput
    # None of the following work:
    # https://langchain-ai.github.io/langgraph/how-tos/pass-run-time-values-to-tools/#define-the-tools_1
    # https://github.com/langchain-ai/langchain/discussions/24906
    # and so on..
    store: Optional[InMemoryStore] = None  # Declare store as an optional field
    metadata_store: Optional[InMemoryStore] = None  # Declare store as an optional field. This is for storing sched related metadata. 

    def __init__(self, store: InMemoryStore, metadata_store: InMemoryStore):
        super().__init__()
        self.store = store
        self.metadata_store = metadata_store
    
    class Config:
        arbitrary_types_allowed = True  # Allow non-Pydantic types like InMemoryStore

    def _run(
        self, 
        plan: formatter.ExistingExperimentalPlanResponseFormatter, 
        plan_id: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Use the tool.
        
        Note: we are guaranteed that plan will conform to the required format
        """
        user_id = "admin"
        application_context = "exp-plans" 
        namespace = (user_id, application_context) # just a random namespace name for now
        
        print("Modifying existing plan...")
        memory_id = plan_id
        plan_data = plan.dict()
        # remove existing plan:
        self.store.delete(namespace, memory_id)

        # Add plan to store:
        self.async_notify_sched_modify_plan(memory_id)
        self.store.put(namespace, memory_id, plan_data)

        return memory_id
    
    def async_notify_sched_modify_plan(self, plan_id: str):
        user_id = "admin"
        application_context = "exp-sched" 
        sched_namespace = (user_id, application_context) # just a random namespace name for now

        memory_id = str("supervisor_wrote_list")
    
        wrote_list = self.metadata_store.get(sched_namespace, memory_id)
        wrote_list = wrote_list.dict()["value"] # https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/?h=item#langgraph_sdk.schema.RunCreate.multitask_strategy
        wrote_list.append(plan_id)
        self.metadata_store.put(sched_namespace, memory_id, wrote_list)

class RedoExpPartitionInput(BaseModel):
    plan_id: str = Field(
        None,
        description="Plan ID of the specific experimental plan from storage."
    )
    group: str = Field(
        None,
        description="This is either 'control_group' or 'experimental_group'."
    )
    partition_name: str = Field(
        None,
        description="The partition_name belonging to the group that needs re-doing."
    )
    error_feedback: str = Field(
        None,
        description="Feedback to the lab technician on what needs to be fixed to get the partition working."
    )

    # state: Annotated[dict, InjectedState]

# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class RedoExpPartitionTool(BaseTool):
    name: str = "redo_exp_partition"
    description: str = "Redo a specific partition of a group in an existing experimental plan. This should be used if you believe the partition's experimental workflow is incorrect, or not working as expected."
    args_schema: Type[BaseModel] = RedoExpPartitionInput
    # None of the following work:
    # https://langchain-ai.github.io/langgraph/how-tos/pass-run-time-values-to-tools/#define-the-tools_1
    # https://github.com/langchain-ai/langchain/discussions/24906
    # and so on..
    store: Optional[InMemoryStore] = None  # Declare store as an optional field
    metadata_store: Optional[InMemoryStore] = None  # Declare store as an optional field. This is for storing sched related metadata. 

    def __init__(self, store: InMemoryStore, metadata_store: InMemoryStore):
        super().__init__()
        self.store = store
        self.metadata_store = metadata_store
        # self.user_id = "admin"
        # self.application_context = "exp-plans" 
        # self.namespace = (self.user_id, self.application_context) # just a random namespace name for now
    
    class Config:
        arbitrary_types_allowed = True  # Allow non-Pydantic types like InMemoryStore

    def _run(
        self, 
        # state: Annotated[dict, InjectedState], 
        plan_id: str, 
        group: str,
        partition_name: str,
        error_feedback: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Use the tool.
        
        Note: we are guaranteed that plan will conform to the required format
        """
        user_id = "admin"
        application_context = "exp-plans" 
        plan_namespace = (user_id, application_context) # just a random namespace name for now

        # print(state["prev_agent"])
        
        print("Modifying existing plan...")

        # Get plan from memory
        plan = self.store.get(plan_namespace, plan_id).dict()["value"]

        # plan[group][partition_name]["error_feedback"] = error_feedback
        plan[group][partition_name]["done"] = False
        plan[group][partition_name]["all_control_experiment_results_filename"] = "" # reset this since the current exec verifiers results will not be needed. # NOTE: we may choose to preserve all history in the future

        # Remove existing plan:
        self.store.delete(plan_namespace, plan_id)

        # Add plan to store:
        self.store.put(plan_namespace, plan_id, plan)

        self.async_notify_sched_redo_partition(plan_id, group, partition_name, error_feedback)

        return "Record partition for redo successful."

    def async_notify_sched_redo_partition(self, plan_id: str, group: str, partition_name: str, error_feedback: str):
        user_id = "admin"
        application_context = "exp-sched" 
        sched_namespace = (user_id, application_context) # just a random namespace name for now

        memory_id = str("supervisor_redo_partition_list")
    
        wrote_list = self.metadata_store.get(sched_namespace, memory_id)
        wrote_list = wrote_list.dict()["value"] # https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/?h=item#langgraph_sdk.schema.RunCreate.multitask_strategy
        # print(wrote_list)
        wrote_list.append({
            "plan_id": plan_id,
            "group": group,
            "partition_name": partition_name,
            "error_feedback": error_feedback
        })
        self.metadata_store.put(sched_namespace, memory_id, wrote_list)

        # wrote_list = self.metadata_store.get(sched_namespace, memory_id)
        # wrote_list = wrote_list.dict()["value"]
        # print(wrote_list)

class RemoveExpPartitionInput(BaseModel):
    plan_id: str = Field(
        None,
        description="Plan ID of the specific experimental plan from storage."
    )
    group: str = Field(
        None,
        description="This is either 'control_group' or 'experimental_group'."
    )
    partition_name: str = Field(
        None,
        description="The partition_name belonging to the group that needs its independent variables changed."
    )
    # state: Annotated[dict, InjectedState]

# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class RemoveExpPartitionTool(BaseTool):
    name: str = "remove_exp_plan_partition"
    description: str = "Remove a partition from some existing experimental plan."
    args_schema: Type[BaseModel] = RemoveExpPartitionInput
    # None of the following work:
    # https://langchain-ai.github.io/langgraph/how-tos/pass-run-time-values-to-tools/#define-the-tools_1
    # https://github.com/langchain-ai/langchain/discussions/24906
    # and so on..
    store: Optional[InMemoryStore] = None  # Declare store as an optional field
    metadata_store: Optional[InMemoryStore] = None  # Declare store as an optional field. This is for storing sched related metadata. 

    def __init__(self, store: InMemoryStore, metadata_store: InMemoryStore):
        super().__init__()
        self.store = store
        self.metadata_store = metadata_store
        # self.user_id = "admin"
        # self.application_context = "exp-plans" 
        # self.namespace = (self.user_id, self.application_context) # just a random namespace name for now
    
    class Config:
        arbitrary_types_allowed = True  # Allow non-Pydantic types like InMemoryStore

    def _run(
        self, 
        # state: Annotated[dict, InjectedState], 
        plan_id: str, 
        group: str,
        partition_name: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Use the tool.
        
        Note: we are guaranteed that plan will conform to the required format
        """
        user_id = "admin"
        application_context = "exp-plans" 
        plan_namespace = (user_id, application_context) # just a random namespace name for now

        # print(state["prev_agent"])
        
        print("Modifying existing plan...")

        # Get plan from memory
        plan = self.store.get(plan_namespace, plan_id).dict()["value"]

        if group not in plan:
            return "The group provided does not exist in the plan."
        if partition_name not in plan[group]:
            return "The partition_name provided does not exist in the group."
        
        del plan[group][partition_name]

        # Remove existing plan:
        self.store.delete(plan_namespace, plan_id)

        # Add plan to store:
        self.store.put(plan_namespace, plan_id, plan)

        self.async_notify_sched_modify_plan(plan_id)

        return "Edit priority successful."

    def async_notify_sched_modify_plan(self, plan_id: str):
        user_id = "admin"
        application_context = "exp-sched" 
        sched_namespace = (user_id, application_context) # just a random namespace name for now

        memory_id = str("supervisor_wrote_list")
    
        wrote_list = self.metadata_store.get(sched_namespace, memory_id)
        wrote_list = wrote_list.dict()["value"] # https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/?h=item#langgraph_sdk.schema.RunCreate.multitask_strategy
        # print(wrote_list)
        wrote_list.append(plan_id)
        self.metadata_store.put(sched_namespace, memory_id, wrote_list)

class ArchiveExpPlanInput(BaseModel):
    plan_id: str = Field(
        None,
        description="Plan ID of the specific experimental plan from storage."
    )

class ArchiveExpPlanTool(BaseTool):
    name: str = "exp_plan_archive"
    description: str = "Remove an existing experimental plan."
    args_schema: Type[BaseModel] = ArchiveExpPlanInput
    # None of the following work:
    # https://langchain-ai.github.io/langgraph/how-tos/pass-run-time-values-to-tools/#define-the-tools_1
    # https://github.com/langchain-ai/langchain/discussions/24906
    # and so on..
    store: Optional[InMemoryStore] = None  # Declare store as an optional field
    metadata_store: Optional[InMemoryStore] = None  # Declare store as an optional field. This is for storing sched related metadata. 

    def __init__(self, store: InMemoryStore, metadata_store: InMemoryStore):
        super().__init__()
        self.store = store
        self.metadata_store = metadata_store
    
    class Config:
        arbitrary_types_allowed = True  # Allow non-Pydantic types like InMemoryStore

    def _run(
        self, 
        plan_id: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Use the tool.
        
        Note: we are guaranteed that plan will conform to the required format
        """
        user_id = "admin"
        application_context = "exp-plans" 
        plan_namespace = (user_id, application_context) # just a random namespace name for now

        # print(state["prev_agent"])
        
        print("Modifying existing plan...")

        # Get plan from memory
        try:
            plan = self.store.get(plan_namespace, plan_id).dict()["value"]
        except:
            return "The plan does not exist."

        # Remove existing plan:
        self.store.delete(plan_namespace, plan_id)

        self.del_sched_metadata(plan_id)

        return "Plan removal successful."

    def del_sched_metadata(self, plan_id: str):
        user_id = "admin"
        application_context = "exp-sched" 
        sched_namespace = (user_id, application_context) # just a random namespace name for now

        # Remove existing plan from supervisor_wrote_list:
        memory_id = str("supervisor_wrote_list")
        supervisor_wrote_list = self.metadata_store.get(sched_namespace, memory_id).dict()["value"]
        new_list = []
        for plan_id2 in supervisor_wrote_list:
            # NOTE: currently, we ignore plan groups that are already executing in the worker.. 
            if plan_id2 != plan_id:
                new_list.append(plan_id2)
        self.metadata_store.put(sched_namespace, memory_id, new_list)

        # Remove existing plan from standby_exp_plan_list:
        memory_id = str("standby_exp_plan_list")
        standby_exp_plan_list = self.metadata_store.get(sched_namespace, memory_id).dict()["value"]
        new_list = []
        for plan_id2 in standby_exp_plan_list:
            # NOTE: currently, we ignore plan groups that are already executing in the worker.. 
            if plan_id2 != plan_id:
                new_list.append(plan_id2)
        self.metadata_store.put(sched_namespace, memory_id, new_list)

        # Remove existing plan from supervisor_redo_partition_list:
        memory_id = str("supervisor_redo_partition_list")
        supervisor_redo_partition_list = self.metadata_store.get(sched_namespace, memory_id).dict()["value"]
        new_list = []
        for redo_details in supervisor_redo_partition_list:
            if redo_details["plan_id"] != plan_id:
                new_list.append(redo_details)
        self.metadata_store.put(sched_namespace, memory_id, new_list)

class EditExpPriorityInput(BaseModel):
    plan_id: str = Field(
        None,
        description="Plan ID of the specific experimental plan from storage."
    )
    # group: str = Field(
    #     None,
    #     description="This is either 'control_group' or 'experimental_group'."
    # )
    # partition_name: str = Field(
    #     None,
    #     description="The partition_name belonging to the group that needs its independent variables changed."
    # )
    priority: int = Field(
        None, gt=0,
        description="An integer representing the priority of the experiment. Lower values indicate higher priority."
    )
    # state: Annotated[dict, InjectedState]

# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class EditExpPriorityTool(BaseTool):
    name: str = "edit_exp_plan_priority"
    description: str = "Modify the priority of an existing experimental plan."
    args_schema: Type[BaseModel] = EditExpPriorityInput
    # None of the following work:
    # https://langchain-ai.github.io/langgraph/how-tos/pass-run-time-values-to-tools/#define-the-tools_1
    # https://github.com/langchain-ai/langchain/discussions/24906
    # and so on..
    store: Optional[InMemoryStore] = None  # Declare store as an optional field
    metadata_store: Optional[InMemoryStore] = None  # Declare store as an optional field. This is for storing sched related metadata. 

    def __init__(self, store: InMemoryStore, metadata_store: InMemoryStore):
        super().__init__()
        self.store = store
        self.metadata_store = metadata_store
        # self.user_id = "admin"
        # self.application_context = "exp-plans" 
        # self.namespace = (self.user_id, self.application_context) # just a random namespace name for now
    
    class Config:
        arbitrary_types_allowed = True  # Allow non-Pydantic types like InMemoryStore

    def _run(
        self, 
        # state: Annotated[dict, InjectedState], 
        plan_id: str, 
        priority: int,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Use the tool.
        
        Note: we are guaranteed that plan will conform to the required format
        """
        user_id = "admin"
        application_context = "exp-plans" 
        plan_namespace = (user_id, application_context) # just a random namespace name for now

        # print(state["prev_agent"])
        
        print("Modifying existing plan...")

        # Get plan from memory
        plan = self.store.get(plan_namespace, plan_id).dict()["value"]

        plan["priority"] = priority

        # Remove existing plan:
        self.store.delete(plan_namespace, plan_id)

        # Add plan to store:
        self.store.put(plan_namespace, plan_id, plan)

        self.async_notify_sched_modify_plan(plan_id)

        return "Edit priority successful."

    def async_notify_sched_modify_plan(self, plan_id: str):
        user_id = "admin"
        application_context = "exp-sched" 
        sched_namespace = (user_id, application_context) # just a random namespace name for now

        memory_id = str("supervisor_wrote_list")
    
        wrote_list = self.metadata_store.get(sched_namespace, memory_id)
        wrote_list = wrote_list.dict()["value"] # https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/?h=item#langgraph_sdk.schema.RunCreate.multitask_strategy
        # print(wrote_list)
        wrote_list.append(plan_id)
        self.metadata_store.put(sched_namespace, memory_id, wrote_list)

        # wrote_list = self.metadata_store.get(sched_namespace, memory_id)
        # wrote_list = wrote_list.dict()["value"]
        # print(wrote_list)

# class EditExpPartitionInput(BaseModel):
#     plan_id: str = Field(
#         None,
#         description="Plan ID of the specific experimental plan from storage."
#     )
#     group: str = Field(
#         None,
#         description="This is either 'control_group' or 'experimental_group'."
#     )
#     partition_name: str = Field(
#         None,
#         description="The partition_name belonging to the group that needs its independent variables changed."
#     )
#     independent_vars: List[Dict[str, Any]] = Field(
#         None,
#         description="New independent variables to replace the current one for this partition."
#     )

#     # state: Annotated[dict, InjectedState]

# # Note: It's important that every field has type hints. BaseTool is a
# # Pydantic class and not having type hints can lead to unexpected behavior.
# class EditExpPartitionsTool(BaseTool):
#     name: str = "edit_exp_partition_independent_vars"
#     description: str = "Modify the independent vars of a specific partition of a group in an existing experimental plan. Use with caution: this will discard all previous results and state related to the partition."
#     args_schema: Type[BaseModel] = EditExpPartitionInput
#     # Holy cow this was frustrating to figure out... 
#     # None of the following work:
#     # https://langchain-ai.github.io/langgraph/how-tos/pass-run-time-values-to-tools/#define-the-tools_1
#     # https://github.com/langchain-ai/langchain/discussions/24906
#     # and so on..
#     store: Optional[InMemoryStore] = None  # Declare store as an optional field
#     metadata_store: Optional[InMemoryStore] = None  # Declare store as an optional field. This is for storing sched related metadata. 

#     def __init__(self, store: InMemoryStore, metadata_store: InMemoryStore):
#         super().__init__()
#         self.store = store
#         self.metadata_store = metadata_store
#         # self.user_id = "admin"
#         # self.application_context = "exp-plans" 
#         # self.namespace = (self.user_id, self.application_context) # just a random namespace name for now
    
#     class Config:
#         arbitrary_types_allowed = True  # Allow non-Pydantic types like InMemoryStore

#     def _run(
#         self, 
#         # state: Annotated[dict, InjectedState], 
#         plan_id: str, 
#         group: str,
#         partition_name: str,
#         error_feedback: str,
#         run_manager: Optional[CallbackManagerForToolRun] = None
#     ) -> str:
#         """
#         Use the tool.
        
#         Note: we are guaranteed that plan will conform to the required format
#         """
#         user_id = "admin"
#         application_context = "exp-plans" 
#         plan_namespace = (user_id, application_context) # just a random namespace name for now

#         # print(state["prev_agent"])
        
#         print("Modifying existing plan...")

#         # Get plan from memory
#         plan = self.store.get(plan_namespace, plan_id).dict()["value"]

#         plan[group][partition_name]["error_feedback"] = error_feedback
#         plan[group][partition_name]["done"] = False

#         # Remove existing plan:
#         self.store.delete(plan_namespace, plan_id)

#         # Add plan to store:
#         self.store.put(plan_namespace, plan_id, plan)

#         self.async_notify_sched_redo_partition(plan_id, group, partition_name, error_feedback)

#         return "Success."

#     def async_notify_sched_modify_plan(self, plan_id: str):
#         user_id = "admin"
#         application_context = "exp-sched" 
#         sched_namespace = (user_id, application_context) # just a random namespace name for now

#         memory_id = str("supervisor_wrote_list")
    
#         wrote_list = self.metadata_store.get(sched_namespace, memory_id)
#         wrote_list = wrote_list.dict()["value"] # https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/?h=item#langgraph_sdk.schema.RunCreate.multitask_strategy
#         # print(wrote_list)
#         wrote_list.append(plan_id)
#         self.metadata_store.put(sched_namespace, memory_id, wrote_list)

#         # wrote_list = self.metadata_store.get(sched_namespace, memory_id)
#         # wrote_list = wrote_list.dict()["value"]
#         # print(wrote_list)

class StoreGetInput(BaseModel):
    plan_id: Optional[str] = Field(
        None,
        description="Plan ID to retrieve the specific experimental plan from storage. Optional. Provide if modifying an existing plan. If not provided, all existing plans will be retrieved."
    )
    # state: Annotated[dict, InjectedState]

class StoreGetTool(BaseTool):
    name: str = "exp_plan_get"
    description: str = "Get all experimental plans, or get a specific experimental plan by plan_id, from long term storage."
    args_schema: Type[BaseModel] = StoreGetInput
    store: Optional[InMemoryStore] = None  # Declare store as an optional field

    def __init__(self, store: InMemoryStore):
        super().__init__()
        self.store = store # Ensure that the same store is passed across all tools and agents
        
    def _run(
        self, 
        # state: Annotated[dict, InjectedState], 
        plan_id: Optional[str] = None, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Use the tool.
        """
        user_id = "admin"
        application_context = "exp-plans" 
        namespace = (user_id, application_context) # just a random namespace name for now
        
        if not plan_id: # get all plans
            items = self.store.search(namespace)
        else: # specific plan
            # list "memories" within this namespace, filtering on content equivalence
            items = self.store.search(namespace, filter={"plan_id": plan_id})

        # Unpack all items and return them as a list
        return [item.dict()["value"] for item in items] # Cleaner return: only return the "value" key, since the "value" key's value (which presents the plan dict) contains plan-key, we don't need the other attributes        

    # async def _arun(
    #     self,
    #     a: int,
    #     b: int,
    #     run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    # ) -> str:
    #     """Use the tool asynchronously."""
    #     # If the calculation is cheap, you can just delegate to the sync implementation
    #     # as shown below.
    #     # If the sync calculation is expensive, you should delete the entire _arun method.
    #     # LangChain will automatically provide a better implementation that will
    #     # kick off the task in a thread to make sure it doesn't block other async code.
    #     return self._run(a, b, run_manager=run_manager.get_sync())

# # @tool
# def store_write_tool(a: Annotated[str, "search string"], store) -> str:
#     """Stores useful information"""
#     # InMemoryStore saves data to an in-memory dictionary. Use a DB-backed store in production use.
#     # store = InMemoryStore()
#     user_id = "admin"
#     application_context = "exp-plans" 
#     namespace = (user_id, application_context) # just a random namespace name for now
#     memory_id = str(uuid.uuid4())
    
#     memory = {"rules": ["User likes short, direct language", "User only speaks English & python"], "my-key": "my-value"}
#     store.put(namespace, memory_id, memory)
#     # get the "memory" by ID
#     item = store.get(namespace, memory_id)
#     # list "memories" within this namespace, filtering on content equivalence
#     items = store.search(namespace, filter={"my-key": "my-value"})

#     return memory_id

# def store_get_tool(memory_id, store):
#     # store = InMemoryStore()
#     user_id = "admin"
#     application_context = "exp-plans" 
#     namespace = (user_id, application_context) # just a random namespace name for now
#     # memory_id = "9b177ab7-d35e-4cca-b5a0-e5af08bfbeb2"
#     item = store.get(namespace, memory_id)
#     print(item)
#     # list "memories" within this namespace, filtering on content equivalence
#     items = store.search(namespace, filter={"my-key": "my-value"})
#     print(items[-1].dict())

# store = InMemoryStore()
# memory_id = store_write_tool("test", store)
# store_get_tool(memory_id, store)

class ExpPlanCompletedWriteInput(BaseModel):
    plan_id: str = Field(
        ...,
        description="The plan_id that was passed to you as input."
    )
    group: str = Field(
        ...,
        description="This was the group that was passed to you as input, it is either 'control_group' or 'experimental_group'."
    )
    partition_name: str = Field(
        ...,
        description="The partition_name that was passed to you as input."
    )
    control_experiment_filename: str = Field(
        ...,
        description="The filename of the controlled experiment workflow."
    )
    control_experiment_results_filename: str = Field(
        ...,
        description="The filename of the result produced by running the controlled experiment workflow."
    )

# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class ExpPlanCompletedWriteTool(BaseTool):
    name: str = "exp_plan_partition_done_write"
    description: str = "Write the completed partition of an experimental plan's group to long term storage."
    args_schema: Type[BaseModel] = ExpPlanCompletedWriteInput
    # None of the following work:
    # https://langchain-ai.github.io/langgraph/how-tos/pass-run-time-values-to-tools/#define-the-tools_1
    # https://github.com/langchain-ai/langchain/discussions/24906
    # and so on..
    store: Optional[InMemoryStore] = None  # Declare store as an optional field
    metadata_store: Optional[InMemoryStore] = None  # Declare store as an optional field. This is for storing sched related metadata. 

    def __init__(self, store: InMemoryStore, metadata_store: InMemoryStore):
        super().__init__()
        self.store = store
        self.metadata_store = metadata_store
    
    class Config:
        arbitrary_types_allowed = True  # Allow non-Pydantic types like InMemoryStore

    def _run(
        self, 
        # state: Annotated[dict, InjectedState], 
        plan_id: str, 
        group: str,
        partition_name: str,
        control_experiment_filename: str,
        control_experiment_results_filename: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Use the tool.
        """
        
        user_id = "admin"
        application_context = "exp-plans" 
        plan_namespace = (user_id, application_context) # just a random namespace name for now

        print("Modifying existing plan...")

        # Get plan from memory
        plan = self.store.get(plan_namespace, plan_id).dict()["value"]

        plan[group][partition_name]["done"] = True
        plan[group][partition_name]["control_experiment_filename"] = control_experiment_filename
        plan[group][partition_name]["control_experiment_results_filename"] = control_experiment_results_filename

        # Remove existing plan:
        self.store.delete(plan_namespace, plan_id)

        # Add plan to store:
        self.store.put(plan_namespace, plan_id, plan)

        return "Successfully recorded the completion of this partition."

class LLMVerifierWriteInput(BaseModel):
    plan_id: str = Field(
        ...,
        description="The plan_id that was passed to you as input."
    )
    group: str = Field(
        ...,
        description="This is either 'control_group' or 'experimental_group'."
    )
    partition_name: str = Field(
        ...,
        description="The partition_name that was passed to you as input."
    )
    control_experiment_filename: str = Field(
        ...,
        description="The filename of the control experiment, that was passed to you as input."
    )
    control_experiment_results_filename: str = Field(
        ...,
        description="The filename of the result produced by running the control experiment workflow."
    )
    is_correct: bool = Field(
        ...,
        description="Indicates whether the experimental workflow is correct."
    )
    verifier_log_message: str = Field(
        ...,
        description="Log or error message describing any issues with the experimental workflow."
    )

# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class LLMVerifierWriteTool(BaseTool):
    name: str = "workflow_verified_record"
    description: str = "Record workflows that you have evaluated to long term storage."
    args_schema: Type[BaseModel] = LLMVerifierWriteInput
    # None of the following work:
    # https://langchain-ai.github.io/langgraph/how-tos/pass-run-time-values-to-tools/#define-the-tools_1
    # https://github.com/langchain-ai/langchain/discussions/24906
    # and so on..
    store: Optional[InMemoryStore] = None  # Declare store as an optional field
    metadata_store: Optional[InMemoryStore] = None  # Declare store as an optional field. This is for storing sched related metadata. 

    def __init__(self, store: InMemoryStore, metadata_store: InMemoryStore):
        super().__init__()
        self.store = store
        self.metadata_store = metadata_store
    
    class Config:
        arbitrary_types_allowed = True  # Allow non-Pydantic types like InMemoryStore

    def _run(
        self, 
        plan_id: str, 
        group: str,
        partition_name: str,
        control_experiment_filename: str,
        control_experiment_results_filename: str,
        is_correct: bool,
        verifier_log_message: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Use the tool.
        """
        
        user_id = "admin"
        application_context = "exp-sched" 
        sched_namespace = (user_id, application_context) # just a random namespace name for now

        memory_id = str("llm_verifier_wrote_list")
    
        wrote_list = self.metadata_store.get(sched_namespace, memory_id)
        wrote_list = wrote_list.dict()["value"] # https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/?h=item#langgraph_sdk.schema.RunCreate.multitask_strategy
        # print(wrote_list)

        # Check if the plan_id and partition_name already exist in the list:
        for item in wrote_list:
            if item["plan_id"] == plan_id and item["group"] == group and item["partition_name"] == partition_name:
                return "This plan_id and partition_name has already been evaluated by you."

        wrote_list.append({
            "plan_id": plan_id,
            "group": group,
            "partition_name": partition_name,
            "control_experiment_filename": control_experiment_filename,
            "control_experiment_results_filename": control_experiment_results_filename,
            "is_correct": is_correct,
            "verifier_log_message": verifier_log_message
        })
        self.metadata_store.put(sched_namespace, memory_id, wrote_list)

        return "Successfully recorded the evaluation."

class PatchVerifierWriteInput(BaseModel):
    plan_id: str = Field(
        ...,
        description="The plan_id that was passed to you as input."
    )
    group: str = Field(
        ...,
        description="This is either 'control_group' or 'experimental_group'."
    )
    partition_name: str = Field(
        ...,
        description="The partition_name that was passed to you as input."
    )
    control_experiment_filename: str = Field(
        ...,
        description="The filename of the control experiment, that was passed to you as input."
    )
    control_experiment_results_filename: str = Field(
        ...,
        description="The filename of the result produced by running the control experiment workflow."
    )
    is_correct: bool = Field(
        ...,
        description="Indicates whether the experimental workflow is correct."
    )
    patcher_log_message: str = Field(
        ...,
        description="Log or error message describing any issues with the experimental workflow."
    )

# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class PatchVerifierWriteTool(BaseTool):
    name: str = "workflow_patched_record"
    description: str = "Record workflows that you have evaluated to long term storage."
    args_schema: Type[BaseModel] = PatchVerifierWriteInput
    # None of the following work:
    # https://langchain-ai.github.io/langgraph/how-tos/pass-run-time-values-to-tools/#define-the-tools_1
    # https://github.com/langchain-ai/langchain/discussions/24906
    # and so on..
    store: Optional[InMemoryStore] = None  # Declare store as an optional field
    metadata_store: Optional[InMemoryStore] = None  # Declare store as an optional field. This is for storing sched related metadata. 

    def __init__(self, store: InMemoryStore, metadata_store: InMemoryStore):
        super().__init__()
        self.store = store
        self.metadata_store = metadata_store
        # self.user_id = "admin"
        # self.application_context = "exp-plans" 
        # self.namespace = (self.user_id, self.application_context) # just a random namespace name for now
    
    class Config:
        arbitrary_types_allowed = True  # Allow non-Pydantic types like InMemoryStore

    def _run(
        self, 
        # state: Annotated[dict, InjectedState], 
        plan_id: str, 
        group: str,
        partition_name: str,
        control_experiment_filename: str,
        control_experiment_results_filename: str,
        is_correct: bool,
        patcher_log_message: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Use the tool.
        """
        
        user_id = "admin"
        application_context = "exp-sched" 
        sched_namespace = (user_id, application_context) # just a random namespace name for now

        memory_id = str("patch_verifier_wrote_list")
    
        wrote_list = self.metadata_store.get(sched_namespace, memory_id)
        wrote_list = wrote_list.dict()["value"] # https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/?h=item#langgraph_sdk.schema.RunCreate.multitask_strategy
        # print(wrote_list)

        # Check if the plan_id and partition_name already exist in the list:
        for item in wrote_list:
            if item["plan_id"] == plan_id and item["group"] == group and item["partition_name"] == partition_name:
                return "This plan_id and partition_name has already been evaluated by you."

        wrote_list.append({
            "plan_id": plan_id,
            "group": group,
            "partition_name": partition_name,
            "control_experiment_filename": control_experiment_filename,
            "control_experiment_results_filename": control_experiment_results_filename,
            "is_correct": is_correct,
            "patcher_log_message": patcher_log_message
        })
        self.metadata_store.put(sched_namespace, memory_id, wrote_list)

        return "Successfully recorded the evaluation."

class AnalyzerWriteInput(BaseModel):
    plan_id: str = Field(
        ...,
        description="The plan_id that was passed to you as input."
    )
    group: str = Field(
        ...,
        description="This is either 'control_group' or 'experimental_group'."
    )
    partition_name: str = Field(
        ...,
        description="The partition_name that was passed to you as input."
    )
    no_change: bool = Field(
        ...,
        description="No change means that after analyzing the partition’s results, you determine that the existing plan and partitions remain valid, requiring no modifications or new plan creation."
    )
    analyzer_log_message: str = Field(
        ...,
        description="Log message detailing the analysis performed and decisions made along with justification."
    )

# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class AnalyzerWriteTool(BaseTool):
    name: str = "analyzer_record"
    description: str = "Record partitions that you have analyzed to long term storage."
    args_schema: Type[BaseModel] = AnalyzerWriteInput
    # None of the following work:
    # https://langchain-ai.github.io/langgraph/how-tos/pass-run-time-values-to-tools/#define-the-tools_1
    # https://github.com/langchain-ai/langchain/discussions/24906
    # and so on..
    store: Optional[InMemoryStore] = None  # Declare store as an optional field
    metadata_store: Optional[InMemoryStore] = None  # Declare store as an optional field. This is for storing sched related metadata. 

    def __init__(self, store: InMemoryStore, metadata_store: InMemoryStore):
        super().__init__()
        self.store = store
        self.metadata_store = metadata_store
        # self.user_id = "admin"
        # self.application_context = "exp-plans" 
        # self.namespace = (self.user_id, self.application_context) # just a random namespace name for now
    
    class Config:
        arbitrary_types_allowed = True  # Allow non-Pydantic types like InMemoryStore

    def _run(
        self, 
        # state: Annotated[dict, InjectedState], 
        plan_id: str, 
        group: str,
        partition_name: str,
        no_change: bool,
        analyzer_log_message: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Use the tool.
        """
        
        user_id = "admin"
        application_context = "exp-sched" 
        sched_namespace = (user_id, application_context) # just a random namespace name for now

        memory_id = str("analyzer_wrote_list")
    
        wrote_list = self.metadata_store.get(sched_namespace, memory_id)
        wrote_list = wrote_list.dict()["value"] # https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/?h=item#langgraph_sdk.schema.RunCreate.multitask_strategy
        # print(wrote_list)

        # Check if the plan_id and partition_name already exist in the list:
        for item in wrote_list:
            if item["plan_id"] == plan_id and item["group"] == group and item["partition_name"] == partition_name:
                return "This plan_id and partition_name has already been evaluated by you."

        wrote_list.append({
            "plan_id": plan_id,
            "group": group,
            "partition_name": partition_name,
            "no_change": no_change,
            "analyzer_log_message": analyzer_log_message,
        })
        self.metadata_store.put(sched_namespace, memory_id, wrote_list)

        return "Successfully recorded the evaluation."

class ConcluderWriteInput(BaseModel):
    is_conclude: bool = Field(
        ...,
        description="This means you decide to conclude the experiment, i.e. after analyzing all partitions results, you determine that the existing plans and partitions remain valid, requiring no modifications or new plan creation."
    )
    concluder_log_message: str = Field(
        ...,
        description="Log message detailing the analysis performed and decisions made along with justification."
    )

# Note: It's important that every field has type hints. BaseTool is a
# Pydantic class and not having type hints can lead to unexpected behavior.
class ConcluderWriteTool(BaseTool):
    name: str = "concluder_record"
    description: str = "Record partitions that you have analyzed to long term storage."
    args_schema: Type[BaseModel] = ConcluderWriteInput
    # None of the following work:
    # https://langchain-ai.github.io/langgraph/how-tos/pass-run-time-values-to-tools/#define-the-tools_1
    # https://github.com/langchain-ai/langchain/discussions/24906
    # and so on..
    store: Optional[InMemoryStore] = None  # Declare store as an optional field
    metadata_store: Optional[InMemoryStore] = None  # Declare store as an optional field. This is for storing sched related metadata. 

    def __init__(self, store: InMemoryStore, metadata_store: InMemoryStore):
        super().__init__()
        self.store = store
        self.metadata_store = metadata_store
        # self.user_id = "admin"
        # self.application_context = "exp-plans" 
        # self.namespace = (self.user_id, self.application_context) # just a random namespace name for now
    
    class Config:
        arbitrary_types_allowed = True  # Allow non-Pydantic types like InMemoryStore

    def _run(
        self, 
        # state: Annotated[dict, InjectedState], 
        is_conclude: bool,
        concluder_log_message: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Use the tool.
        """
        
        user_id = "admin"
        application_context = "exp-sched" 
        sched_namespace = (user_id, application_context) # just a random namespace name for now

        memory_id = str("concluder_wrote_list")
    
        wrote_list = self.metadata_store.get(sched_namespace, memory_id)
        wrote_list = wrote_list.dict()["value"] # https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/?h=item#langgraph_sdk.schema.RunCreate.multitask_strategy
        # print(wrote_list)

        wrote_list.append({
            "is_conclude": is_conclude,
            "concluder_log_message": concluder_log_message,
        })
        self.metadata_store.put(sched_namespace, memory_id, wrote_list)

        return "Successfully recorded the evaluation."