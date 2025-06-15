from langchain_core.tools import tool
from typing import Annotated, List
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import InjectedStore
from langgraph.prebuilt import InjectedState
import uuid
from typing import Optional, Type, Dict

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from collections import deque, defaultdict
import re

import formatter
import settings
import worker_agent

import scheduler as sched

import pytest
from pydantic import ValidationError
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

import tool
from langgraph.store.memory import InMemoryStore

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    prev_agent: Literal[*settings.AGENT_LIST] # The agent that called the tool (we are using this mainly for the standard exp-agent interface functions, such as write, get..., since these are agent->tool interactions. Note that Sched tool does not need this, since we will always do a worker -> sched -> supervisor, or supervisor -> sched -> worker transitions) # To view prev_agent within tools, we need: https://langchain-ai.github.io/langgraph/how-tos/pass-run-time-values-to-tools/#pass-graph-state-to-tools
    next_agent: Literal[*settings.AGENT_LIST]

state: State = {
    "messages": [],
    "prev_agent": "supervisor",
    # "next_agent": "sched"
}

store = InMemoryStore()
metadata_store = InMemoryStore()

sched.setup_sched(metadata_store)

sched_tool = sched.SchedTool(store, metadata_store)
response = sched_tool.invoke({"state":state}) # response is guaranteed to be a dict
print(response)