NUM_WORKERS = 1
NUM_CONTROL_WORKERS = 1

# Scheduler specific:
VARS_PER_PARTITION = 5 # Number of variable values to assign to each partition. This means we may partition a single experimental plan into multiple smaller plans that are identical except for the experimental/control groups used. E.g., [{instance_type: t2.micro}, {instance_type: t2.small}, {instance_type: t2.medium}, {instance_type: t2.large}, {instance_type: t2.xlarge}] would be 5 variable values. 
PARTITIONS_PER_WORKER = 1 # Number of experimental/control group partitions to assign to one worker. Note this is only used later after the controlled experiment has been set up correctly. 

def list_worker_names() -> list:
    worker_names = []
    for i in range(NUM_WORKERS):
        worker_names.append(f"worker_{i}")
    return worker_names

def list_control_worker_names() -> list:
    worker_names = []
    for i in range(NUM_CONTROL_WORKERS):
        worker_names.append(f"control_worker_{i}")
    return worker_names

AGENT_LIST = ["supervisor"]
AGENT_LIST.extend(list_worker_names()) # add worker agents to the agent list
AGENT_LIST.extend(list_control_worker_names()) # add control worker agents to the agent list

# Do not override the following variables:
CONCLUDER_BUFFER_STEPS = 15