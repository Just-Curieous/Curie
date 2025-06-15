# TEST 1: test through regular langchain tool invoke:
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

store_write_tool = tool.StoreWriteTool(store, metadata_store)
sched.setup_sched(metadata_store)

def test_invalid_plan_input(store_write_tool):
    """
    Test that invoking store_write_tool with an invalid 'plan' structure raises an error.
    """
    invalid_input = {
        "plan": {"testkey": "University of Michigan!"}  # Missing required keys from the schema
    }

    # Expect a ValidationError when invoking with invalid input
    with pytest.raises(ValidationError) as exc_info:
        memory_id = store_write_tool.invoke(invalid_input)

    print("Test passed: Invalid input raised a ValidationError.")

def test_valid_plan_input(store_write_tool):
    """
    Test that invoking store_write_tool with a valid 'plan' structure returns a memory_id.
    """
    valid_input = {
        "plan": {
            "hypothesis": "AWS EC2 VMs in us-east-1 are slower than us-east-2.",
            "constant_vars": ["EC2 VM type"],
            "independent_vars": ["AWS region"],
            "dependent_vars": ["execution_time"],
            "controlled_experiment_setup_description": "Create EC2 VM, SSH into VM, run predefined task.",
            "control_group": {"region": ["us-east-1"]},
            "experimental_group": {"region": ["us-east-2"]},
            "priority": 1,
        }
    }

    memory_id = store_write_tool.invoke(valid_input, state_instance)
    store_get_tool = tool.StoreGetTool(store)
    vals = store_get_tool.invoke({"plan_key": memory_id})
    assert memory_id is not None and vals is not None

    print("Test passed: Valid input returned a memory_id.")

def test_multi_valid_plan_input(store_write_tool):
    """
    Test that invoking store_write_tool with a valid 'plan' structure, followed by an edit of the written plan (note: with non-validated input keys), works.
    """
    valid_input = {
        "plan": {
            "hypothesis": "AWS EC2 VMs in us-east-1 are slower than us-east-2.",
            "constant_vars": ["EC2 VM type"],
            "independent_vars": ["AWS region"],
            "dependent_vars": ["execution_time"],
            "controlled_experiment_setup_description": "Create EC2 VM, SSH into VM, run predefined task.",
            "control_group": {"region": ["us-east-1"]},
            "experimental_group": {"region": ["us-east-2"]},
            "priority": 1,
        }
    }

    memory_id = store_write_tool.invoke(valid_input)
    store_get_tool = tool.StoreGetTool(store)
    vals = store_get_tool.invoke({"plan_id": memory_id})

    valid_input = {
        "plan": {
            "hypothesis": "AWS EC2 VMs in us-east-1 are slower than us-east-2.",
            "constant_vars": ["EC2 VM type"],
            "independent_vars": ["AWS region"],
            "dependent_vars": ["execution_time"],
            "controlled_experiment_setup_description": "Create EC2 VM, SSH into VM, run predefined task.",
            "control_group": {"region": ["us-east-1"]},
            "experimental_group": {"region": ["us-east-2"]},
            "priority": 1,
            "control_group_done": True, # new key value
        },
        "plan_id": memory_id
    }
    # print("entering edit")
    new_memory_id = store_write_tool.invoke(valid_input)
    assert new_memory_id == memory_id
    store_get_tool = tool.StoreGetTool(store)
    vals = store_get_tool.invoke({"plan_id": new_memory_id})
    print("New plan:")
    print(vals)

    assert memory_id is not None and vals is not None

    assert len(vals) == 1
    assert vals[0]["control_group_done"] == True

    print("Test passed: Valid input returned a memory_id, and editing the memory_id with a new plan worked.")

test_invalid_plan_input(store_write_tool)
test_valid_plan_input(store_write_tool)
test_multi_valid_plan_input(store_write_tool)

# TEST 2: Store through agent test:
import store_agent
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.store.memory import InMemoryStore

import model
import utils
import tool

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

store = InMemoryStore()
metadata_store = InMemoryStore()
store_write_tool = tool.StoreWriteTool(store, metadata_store)
sched.setup_sched(metadata_store)
store_get_tool = tool.StoreGetTool(store)
tools = [store_write_tool, 
    store_get_tool
]
# system_prompt_file = "prompts/exp-store-writer.txt"
system_prompt_file = "prompts/exp-store-writer-test.txt"
store_agent = store_agent.create_StoreWriteAgent(tools, system_prompt_file, State)

graph_builder.add_node("chatbot", store_agent)
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
# graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

utils.save_langgraph_graph(graph, "misc/graph_image.png") 

# Section: Run agent
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break