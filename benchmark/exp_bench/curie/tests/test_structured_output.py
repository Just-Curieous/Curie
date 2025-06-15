# Overall guide: https://python.langchain.com/docs/how_to/structured_output/#typeddict-or-json-schema
from pydantic import BaseModel, Field
from typing import List, Dict, Any

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from formatter import NewExperimentalPlanResponseFormatter

# store_agent = store_agent.with_structured_output(ExperimentalPlanResponseFormatter)

print("Testing valid input 1...")
valid_input = {'hypothesis': 'The c5.large instance type can handle 500 concurrent requests to the add_to_cart function with a 99th percentile latency below 100ms, while minimizing cost.', 'constant_vars': ['e-commerce web application code', 'number of concurrent requests (500)'], 'independent_vars': ['EC2 instance type'], 'dependent_vars': ['99th percentile latency', 'cost'], 'controlled_experiment_setup_description': 'Create a c5.large EC2 instance, deploy the e-commerce web application on the instance, simulate 500 concurrent requests to the add_to_cart function, measure the 99th percentile latency and cost.', 'control_group': [{'instance_type': 'c5.large'}], 'experimental_group': [{'instance_type': 'c5.medium'}], 'priority': 1}

formatter = NewExperimentalPlanResponseFormatter(**valid_input)
print(formatter.dict()) # https://docs.pydantic.dev/1.10/usage/exporting_models/

print("Testing valid input 2... (without experimental_group)")
valid_input = {'hypothesis': 'The c5.large instance type can handle 500 concurrent requests to the add_to_cart function with a 99th percentile latency below 100ms, while minimizing cost.', 'constant_vars': ['e-commerce web application code', 'number of concurrent requests (500)'], 'independent_vars': ['EC2 instance type'], 'dependent_vars': ['99th percentile latency', 'cost'], 'controlled_experiment_setup_description': 'Create a c5.large EC2 instance, deploy the e-commerce web application on the instance, simulate 500 concurrent requests to the add_to_cart function, measure the 99th percentile latency and cost.', 'control_group': [{'instance_type': 'c5.large'}], 'priority': 1}

formatter = NewExperimentalPlanResponseFormatter(**valid_input)
print(formatter.dict()) # https://docs.pydantic.dev/1.10/usage/exporting_models/


print("Testing invalid input 1... (this should trigger an error)")
invalid_input = {
    "hypothesis": "AWS EC2 VMs in us-east-1 are slower than us-east-2.",
    "constant_vars": ["EC2 VM type"],
    "independent_vars": ["AWS region"],
    "dependent_vars": ["execution_time"],
    "controlled_experiment_setup_description": "Create EC2 VM, SSH into VM, run predefined task.",
    "control_group": {"region": "us-east-1", "execution_time": "baseline"},
    "experimental_group": {"region": "us-east-2", "execution_time": "to compare"},
    "priority": 1,
}

formatter = NewExperimentalPlanResponseFormatter(**invalid_input)
print(formatter.dict()) # https://docs.pydantic.dev/1.10/usage/exporting_models/