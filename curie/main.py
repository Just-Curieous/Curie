import subprocess
import time
import os
import argparse
from enum import Enum
import json
import re
from datetime import datetime 
import sys
import uuid 
from curie.logger import init_logger, send_question_telemetry
import shutil

# Create a function to parse input arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Process input arguments for the script.")
    
    # Add argument for iterations
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations (must be an integer)."
    )
    
    parser.add_argument(
        "--question_file",
        "-f",
        type=str,
        required=False,
        help="Question file to run"
    )

    parser.add_argument(
        "--question",
        "-q",
        type=str,
        required=False,
        help="Question to run"
    )

    parser.add_argument(
        "--task_config",
        type=str, 
        default="curie/configs/base_config.json",
        help="Task configuration file for advanced developers."
    )

    parser.add_argument(
        "--report",
        action="store_true",
        help="Whether to write a formal experiment report."
    )

    return parser.parse_args()

def prune_openhands_docker(): 
    # Get the list of container names matching 'openhands'
    result = subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}"],
        capture_output=True, text=True, check=True
    )
    # Filter names starting with 'openhands'
    container_names = [name for name in result.stdout.splitlines() if name.startswith("openhands")]

    # If there are containers to remove, run the `docker rm -f` command
    if container_names:
        subprocess.run(["docker", "rm", "-f"] + container_names, check=True)
        print("Removed containers:", ", ".join(container_names))
    else:
        print("No matching containers found.")

# Function to create a configuration file
def create_config_file(question_file, unique_id, iteration, task_config):
    work_name = (
        task_config.get('workspace_name', '').lstrip('/').rstrip('/').split('/')[-1] or 
        task_config.get('job_name', '') or 
        "default_research"
    )
    # Setup logging directory and files
    exp_log_dir = os.path.join("logs", f"{work_name}_{unique_id}_iter{iteration}")
    os.makedirs(exp_log_dir, exist_ok=True)

    # Generate filenames
    question_base = os.path.basename(question_file).replace('.txt', '')
    log_filename = os.path.join(exp_log_dir, f"{question_base}_{unique_id}_iter{iteration}.log")
    config_filename = os.path.join(exp_log_dir, 
                                 f"{work_name}_config_{question_base}_{unique_id}_iter{iteration}.json")

    # Update task configuration
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    task_config.update({
        "unique_id": unique_id,
        "iteration": iteration,
        "log_filename": log_filename,
        "exp_plan_filename": question_file,
        "base_dir": base_dir
        # "workspace_name": workspace_name
    })
    
    os.makedirs(os.path.dirname(config_filename), exist_ok=True)
    send_question_telemetry(question_file)
    
    with open(config_filename, "w") as f:
        json.dump(task_config, f, indent=4)
    send_question_telemetry(config_filename)
    
    global curie_logger
    curie_logger = init_logger(log_filename)

    curie_logger.info(f"Config file created: {config_filename}")
    curie_logger.info(f"Check out the log file: {log_filename}")
    return task_config, config_filename


def docker_image_exists(image):
    """Check if a Docker image exists locally."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        return result.returncode == 0  # Return True if image exists
    except Exception as e:
        print(f"Error checking Docker image: {e}")
        return False


# Function to run a Docker container
def run_docker_container(unique_id, iteration, task_config):
    rand_uuid = uuid.uuid4()
    container_name = f"exp-agent-container-{unique_id}-{rand_uuid}-iter{iteration}"
    curie_logger.info(f"Building Docker image for iteration {iteration}...")
    
    image_name = task_config["docker_image"]
    docker_filename = task_config["base_dir"] + "/curie/" + task_config["dockerfile_name"]

    if docker_image_exists(image_name):
        curie_logger.info(f"Using existing Docker image: {image_name}")
    else:
        # FIXME: enable auto rebuild if the docker image or its dependencies are changed
        curie_logger.info(f"Start building Docker image {image_name} ... ") 
        command = [
            "sudo", "docker", "build",
            "--no-cache", "--progress=plain",
            "-t",  image_name,
            "-f",  docker_filename,
            "."
        ] 
        subprocess.run(command, check=True)
    
    base_dir = task_config['base_dir']
    command = [
        "docker", "run",
        "-v", "/var/run/docker.sock:/var/run/docker.sock",
        "-v", f"{base_dir}/curie:/curie:ro",
        "-v", f"{base_dir}/benchmark:/benchmark:ro",
        "-v", f"{base_dir}/logs:/logs",
        "-v", f"{base_dir}/starter_file:/starter_file:ro",
        "-v", f"{base_dir}/workspace:/workspace",
        "-v", f"/:/all:ro",
        "--network=host",
        "-d",
    ]
    has_gpu = shutil.which("nvidia-smi") is not None and subprocess.call(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
    if has_gpu:
        command += ["--gpus", "all"]
    command += ["--name", container_name, image_name]

    curie_logger.info(f"Running command: {' '.join(command)}")
    # Run the command
    subprocess.run(command, check=True) 
    return container_name

# Function to execute the experiment inside the Docker container
def execute_experiment_in_container(container_name, task_config, config_file):
    """
    Executes the experiment inside the specified Docker container and retrieves log files.

    Args:
        container_name (str): The name of the Docker container.
        config_file (str): The path to the configuration file for the experiment.

    Raises:
        Exception: If any subprocess command fails.
    """
    curie_logger.info(f"Starting experiment in container {container_name} with config in {config_file}")
    try:
        # check for the existence of curie/setup/env.sh
        if not os.path.exists("curie/setup/env.sh"):
            curie_logger.error("env.sh does not exist under curie/setup. Please input your API credentials.")
            return

        # Run the experiment inside the container
        subprocess.run([
            "docker", "exec", "-it", container_name,
            "bash", "-c", (
                # "source ~/.bashrc && "
                "source setup/env.sh && "
                '''eval "$(micromamba shell hook --shell bash)" && '''
                "micromamba activate curie && "
                "sed -i '474i \\                    \"organization\": \"499023\",' /root/.cache/pypoetry/virtualenvs/openhands-ai-*-py3.12/lib/python3.12/site-packages/litellm/llms/azure/azure.py &&"
                "sed -i '474i \\    \"organization\": \"499023\",' /opt/micromamba/envs/curie/lib/python3.11/site-packages/litellm/llms/azure/azure.py  &&"
                "sed -i '49d' /root/.cache/pypoetry/virtualenvs/openhands-ai-*-py3.12/lib/python3.12/site-packages/litellm/llms/azure/chat/o_series_handler.py &&"
                "sed -i '49i \\                    organization=\"014482\",' /root/.cache/pypoetry/virtualenvs/openhands-ai-*-py3.12/lib/python3.12/site-packages/litellm/llms/azure/chat/o_series_handler.py  &&"
                f"python3 construct_workflow_graph.py /{config_file}"
            )
        ], check=True)  # This will block until main.py finishes.

    except subprocess.CalledProcessError as e:
        curie_logger.error(f"Experiment failed with exit code {e.returncode}. Error: {e}")
        raise

# Function to stop and remove the Docker container
def cleanup_docker_container(container_name):
    print(f"Stopping and removing Docker container: {container_name}...")
    subprocess.run(["docker", "stop", container_name], check=True)
    subprocess.run(["docker", "rm", container_name], check=True)
    print(f"Docker container {container_name} cleaned up.")

def run_prune_commands():
    commands = [
        [ "docker", "container", "prune", "-f"],
        [ "docker", "image", "prune", "-f"],
        [ "docker", "volume", "prune", "-f"],
        [ "docker", "builder", "prune", "-f"],
    ]

    for command in commands:
        try:
            print(f"Running docker: {' '.join(command)}")
            result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(result.stdout.decode())  # Print the standard output
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {' '.join(command)}")
            print(e.stderr.decode())  # Print the standard error
    prune_openhands_docker()

def execute_curie(question_filename, unique_id, iteration, task_config):
    # Create configuration file
    task_config, config_filename = create_config_file(question_filename, unique_id, iteration, task_config)

    # Run Docker container for this iteration
    container_name = None
    try:
        container_name = run_docker_container(unique_id, iteration, task_config)

        # Execute experiment in Docker container
        execute_experiment_in_container(container_name, task_config, config_filename)

    finally:
        # Clean up Docker container after each iteration
        if container_name:
            cleanup_docker_container(container_name)
        run_prune_commands()
        pass
    
def main():
    args = parse_args()
    config_file = args.task_config
    try:
        with open(config_file, 'r') as f:
            task_config = json.load(f)
            task_config['report'] = args.report
    except Exception as e:
        print(f"Error reading config file: {e}")
        return
    
    print(f"Curie is running with the following configuration: {task_config}")
    if args.question_file is None and args.question is None:
        print("Please provide either a question file or a question.")
        return
    elif args.question_file is not None and args.question is not None:
        print("Please provide only one of either a question file or a question.")
        return
    elif args.question_file is None:
        q_file = (
                task_config["workspace_name"].rstrip('/').split("/")[-1] or
                task_config.get('job_name') or 
                "default_research"  
                )
        
        question_file = f'workspace/{q_file}_{int(time.time())}.txt'
        try:
            os.makedirs(os.path.dirname(question_file), exist_ok=True)
            # write the question to the file
            with open(question_file, 'w') as f:
                f.write(args.question)
        except Exception as e:
            print(f"Error writing question to file: {e}")
            print("Please give permission to write to `workspace/`.")
            sys.exit(1)
    else:
        question_file = args.question_file
    
    # print(f"Processing {question_file} for {args.iterations} iterations...")
    for iteration in range(1, args.iterations + 1):
        # Perform the required operation for each iteration (to be provided later)
        # print(f"Iteration {iteration} ")
        start_time = time.time() 
        unique_id = datetime.now().strftime("%Y%m%d%H%M%S")

        execute_curie(question_file, unique_id, iteration, task_config)
        send_question_telemetry(task_config['log_filename'])
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Iteration {iteration} for {question_file} completed in {elapsed_time:.2f} seconds.")
    
if __name__ == "__main__": 
    main() 
