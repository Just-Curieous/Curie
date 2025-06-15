import pytest
from pydantic import ValidationError
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

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
redo_partition_tool = tool.RedoExpPartitionTool(store, metadata_store)
sched.setup_sched(metadata_store)

def test_valid_redo_partition():
    """
    Test that invoking store_write_tool with a valid 'plan' structure returns a memory_id.
    """
    valid_input = {
        "plan": {'hypothesis': 'AWS EC2 t3.medium instances are less CPU efficient compared to c5.large instances under a fixed compute-bound workload.', 'constant_vars': ['workload'], 'independent_vars': ['instance_type'], 'dependent_vars': ['execution_time'], 'controlled_experiment_setup_description': 'Create EC2 VM, run compute-bound task, measure execution time.', 'priority': 1, 'control_group': [{'instance_type': 't3.medium'}], 'experimental_group': [{'instance_type': 'c5.large'}]}
    }

    plan_id = store_write_tool.invoke(valid_input, state_instance)
    store_get_tool = tool.StoreGetTool(store)
    vals = store_get_tool.invoke({"plan_key": plan_id})

    print("The stored plan is structured as: ", vals)

    valid_input = {
        "plan_id": plan_id, 
        "group": "control_group",
        "partition_name": "partition_1",
        "error_feedback": "The results from the two runs show significant variability in the number of events per second, latency, and maximum latency. Please investigate potential sources of variability in the computational environment and ensure consistent conditions across runs."
    }

    memory_id = redo_partition_tool.invoke(valid_input, state_instance)

    memory_id = str("supervisor_redo_partition_list")

    user_id = "admin"
    application_context = "exp-sched" 
    sched_namespace = (user_id, application_context)

    wrote_list = metadata_store.get(sched_namespace, memory_id)
    wrote_list = wrote_list.dict()["value"]

    assert len(wrote_list) == 1

    redo_partition = wrote_list[0]

    assert redo_partition["plan_id"] == plan_id
    assert redo_partition["group"] == "control_group"
    assert redo_partition["partition_name"] == "partition_1"
    assert redo_partition["error_feedback"] == "The results from the two runs show significant variability in the number of events per second, latency, and maximum latency. Please investigate potential sources of variability in the computational environment and ensure consistent conditions across runs."

    assert memory_id is not None and vals is not None

print(test_valid_redo_partition())