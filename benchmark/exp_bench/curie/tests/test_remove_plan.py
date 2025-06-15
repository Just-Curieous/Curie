import pytest
from pydantic import ValidationError
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import model
import scheduler as sched
import tool

from langgraph.store.memory import InMemoryStore

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

state_instance: State = {
    "messages": []
}

store = InMemoryStore()
metadata_store = InMemoryStore()

store_write_tool = tool.NewExpPlanStoreWriteTool(store, metadata_store)
removal_plan_tool = tool.ArchiveExpPlanTool(store, metadata_store)

user_id = "admin"
application_context = "exp-sched" 
sched_namespace = (user_id, application_context)

# Necessary now:
user_input = "test question"
metadata_store.put(sched_namespace, "question", user_input)

sched.setup_sched(metadata_store, sched_namespace)

def test_valid_remove_plan():
    """
    Test that plan removal works given an existing valid plan_id. 
    """

    # Initialize and store the plan:
    valid_input = {
        "plan": {'hypothesis': 'AWS EC2 t3.medium instances are less CPU efficient compared to c5.large instances under a fixed compute-bound workload.', 'constant_vars': ['workload'], 'independent_vars': ['instance_type'], 'dependent_vars': ['execution_time'], 'controlled_experiment_setup_description': 'Create EC2 VM, run compute-bound task, measure execution time.', 'priority': 1, 'control_group': [{'instance_type': 't3.medium'}], 'experimental_group': [{'instance_type': 'c5.large'}]}
    }

    plan_id = store_write_tool.invoke(valid_input, state_instance)
    store_get_tool = tool.StoreGetTool(store)
    vals = store_get_tool.invoke({"plan_key": plan_id})

    print("The stored plan is structured as: ", vals)

    valid_input = {
        "plan_id": plan_id, 
    }

    # Store valid metadata:
    memory_id = str("supervisor_wrote_list")
    new_list = [plan_id]
    metadata_store.put(sched_namespace, memory_id, new_list)
    supervisor_wrote_list = metadata_store.get(sched_namespace, memory_id).dict()["value"]
    assert len(supervisor_wrote_list) == 1

    memory_id = str("standby_exp_plan_list")
    new_list = [plan_id]
    metadata_store.put(sched_namespace, memory_id, new_list)
    standby_exp_plan_list = metadata_store.get(sched_namespace, memory_id).dict()["value"]
    assert len(standby_exp_plan_list) == 1

    memory_id = str("supervisor_redo_partition_list")
    new_list = [{"plan_id": plan_id}]
    metadata_store.put(sched_namespace, memory_id, new_list)
    supervisor_redo_partition_list = metadata_store.get(sched_namespace, memory_id).dict()["value"]
    assert len(supervisor_redo_partition_list) == 1

    # Remove the plan:
    return_string = removal_plan_tool.invoke(valid_input, state_instance)

    assert return_string == "Plan removal successful."

    # Check that the plan was removed from all metadata lists:
    memory_id = str("supervisor_wrote_list")
    wrote_list = metadata_store.get(sched_namespace, memory_id).dict()["value"]
    assert len(wrote_list) == 0

    memory_id = str("standby_exp_plan_list")
    standby_list = metadata_store.get(sched_namespace, memory_id).dict()["value"]
    assert len(standby_list) == 0

    memory_id = str("supervisor_redo_partition_list")
    redo_partition_list = metadata_store.get(sched_namespace, memory_id).dict()["value"]
    assert len(redo_partition_list) == 0


def test_invalid_remove_plan():
    """
    Test that plan removal does not work and has no effect on other plans given a non-existing invalid plan_id. 
    """

    # Initialize and store the plan:
    valid_input = {
        "plan": {'hypothesis': 'AWS EC2 t3.medium instances are less CPU efficient compared to c5.large instances under a fixed compute-bound workload.', 'constant_vars': ['workload'], 'independent_vars': ['instance_type'], 'dependent_vars': ['execution_time'], 'controlled_experiment_setup_description': 'Create EC2 VM, run compute-bound task, measure execution time.', 'priority': 1, 'control_group': [{'instance_type': 't3.medium'}], 'experimental_group': [{'instance_type': 'c5.large'}]}
    }

    plan_id = store_write_tool.invoke(valid_input, state_instance)
    store_get_tool = tool.StoreGetTool(store)
    vals = store_get_tool.invoke({"plan_key": plan_id})

    print("The stored plan is structured as: ", vals)

    valid_input = {
        "plan_id": plan_id, 
    }

    # Store valid metadata:
    user_id = "admin"
    application_context = "exp-sched" 
    sched_namespace = (user_id, application_context)
    memory_id = str("supervisor_wrote_list")
    new_list = [plan_id]
    metadata_store.put(sched_namespace, memory_id, new_list)
    supervisor_wrote_list = metadata_store.get(sched_namespace, memory_id).dict()["value"]
    assert len(supervisor_wrote_list) == 1

    memory_id = str("standby_exp_plan_list")
    new_list = [plan_id]
    metadata_store.put(sched_namespace, memory_id, new_list)
    standby_exp_plan_list = metadata_store.get(sched_namespace, memory_id).dict()["value"]
    assert len(standby_exp_plan_list) == 1

    memory_id = str("supervisor_redo_partition_list")
    new_list = [{"plan_id": plan_id}]
    metadata_store.put(sched_namespace, memory_id, new_list)
    supervisor_redo_partition_list = metadata_store.get(sched_namespace, memory_id).dict()["value"]
    assert len(supervisor_redo_partition_list) == 1

    # Attempt to Remove an invalid plan:
    invalid_input = {
        "plan_id": "invalid_plan_id", 
    }
    return_string = removal_plan_tool.invoke(invalid_input, state_instance)

    assert return_string == "The plan does not exist."

    # Check that the plan was removed from all metadata lists:
    memory_id = str("supervisor_wrote_list")
    wrote_list = metadata_store.get(sched_namespace, memory_id).dict()["value"]
    assert len(wrote_list) == 1

    memory_id = str("standby_exp_plan_list")
    standby_list = metadata_store.get(sched_namespace, memory_id).dict()["value"]
    assert len(standby_list) == 1

    memory_id = str("supervisor_redo_partition_list")
    redo_partition_list = metadata_store.get(sched_namespace, memory_id).dict()["value"]
    assert len(redo_partition_list) == 1

print(test_valid_remove_plan())
print(test_invalid_remove_plan())

