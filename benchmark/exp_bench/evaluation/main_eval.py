"""
    Processes at the granularity of the paper ID level. Instantiates a new docker container that will run tasks in the paper (eval.py).
"""
import subprocess
import time
import os
import argparse
from enum import Enum
import json
import re
from pathlib import Path
from datetime import datetime 
import sys
import uuid 
from helper.logger import init_logger
from utils import get_log_filename_eval_gen, get_log_filename_judge_gen
import shutil

# Create a function to parse input arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Process input arguments for the script.")
    
    parser.add_argument(
        "--task_config",
        type=str, 
        default="evaluation/configs/main_eval_gen_config_template.json",
        help="Experimental Task configuration file."
    )

    return parser.parse_args()

def convert_dir_to_logfile_path(input_path: str, suffix: str = "setup_gen.log") -> Path:
    input_path = Path(input_path)
    paper_id = input_path.name  # e.g., "19044"
    parts = input_path.parts[1:]  # Skip "benchmark"
    output_path = Path("outputs", "logs", *parts, f"{paper_id}_{suffix}")
    return str(output_path)

def convert_file_to_logfile_path(input_file: str, suffix: str = "setup_gen.log") -> Path:
    input_path = Path(input_file)
    stem = input_path.stem  # e.g., vdb_enhanced_research_tasks_topomlp_reduced_test
    parts = input_path.parts[1:-1]  # skip "logs", drop filename
    output_path = Path("outputs", "logs", *parts, stem, f"{stem}_{suffix}")
    return str(output_path)

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
def create_config_file(unique_id, iteration, task_config):
    # log_filename = f"logs/{os.path.basename(question_file).replace('.txt', '')}_{unique_id}_iter{iteration}.log"
    config_filename = f"outputs/evaluation/configs/{str(task_config["paper_id"])}_{task_config["mode"]}_{unique_id}_config.json"
    log_filename = get_log_filename_eval_gen(task_config, unique_id, iteration)
    log_judge_filename = get_log_filename_judge_gen(task_config, unique_id, iteration)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    task_config.update({"unique_id": unique_id, 
                        "iteration": iteration, # unlike "iterations" which is the total number of iterations to run
                        "log_filename": log_filename, 
                        "log_judge_filename": log_judge_filename,
                        'base_dir': base_dir})
    
    os.makedirs(os.path.dirname(config_filename), exist_ok=True)
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    
    with open(config_filename, "w") as f:
        json.dump(task_config, f, indent=4)
    
    global bench_logger
    if task_config["mode"] == "generate":
        bench_logger = init_logger(log_filename)
    elif task_config["mode"] == "judge":
        bench_logger = init_logger(log_judge_filename)
    else:
        bench_logger = init_logger(log_filename)
        bench_logger.error(f"Invalid mode {task_config["mode"]} provided in the task configuration.")
        raise ValueError(f"Invalid mode {task_config["mode"]} provided in the task configuration.")

    bench_logger.info(f"Config file created: {config_filename}")
    bench_logger.info(f"Check out the log file: {log_filename}")
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
    bench_logger.info(f"Building Docker image for iteration {iteration}...")
    
    image_name = task_config["docker_image"]
    docker_filename = task_config["base_dir"] + task_config["dockerfile_name"]

    context_dir = "." # of task_config["agent_name"] != "inspectai" else "./inspect_agent"

    if docker_image_exists(image_name):
        bench_logger.info(f"Using existing Docker image: {image_name}")
    else:
        # FIXME: enable auto rebuild if the docker image or its dependencies are changed
        bench_logger.info(f"Start building Docker image {image_name} ... ") 
        command = [
            "sudo", "docker", "build",
            "--no-cache", "--progress=plain",
            "-t",  image_name,
            "-f",  docker_filename,
            context_dir
        ] 
        subprocess.run(command, check=True)
    
    base_dir = task_config['base_dir']
    command = [
        "docker", "run",
        "-v", "/var/run/docker.sock:/var/run/docker.sock",
        "-v", f"{base_dir}/benchmark:/benchmark:ro",
        "-v", f"{base_dir}/logs:/logs",
        "-v", f"{base_dir}/evaluation:/evaluation:ro",
        "-v", f"{base_dir}/inspect_agent:/inspect_agent",
        "-v", f"{base_dir}/outputs:/outputs",
        "-v", f"{base_dir}/setup:/setup:ro",
        "-v", f"{base_dir}/prompts:/prompts:ro",
        "-v", f"{base_dir}/helper:/helper:ro",
        "-v", f"{base_dir}/workspace:/workspace",
        "-v", f"{base_dir}:/exp_bench:ro", # this contains our primary extraction scripts
        "--network=host",
        "-d",
    ]
    has_gpu = shutil.which("nvidia-smi") is not None and subprocess.call(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
    if has_gpu:
        command += ["--gpus", "all"]
    command += ["--name", container_name, image_name]

    bench_logger.info(f"Running command: {' '.join(command)}")
    # Run the command
    subprocess.run(command, check=True) 
    return container_name

# Function to execute the experiment inside the Docker container
def execute_experiment_in_container(container_name, task_config, config_filename):
    """
    Executes the experiment inside the specified Docker container and retrieves log files.

    Args:
        container_name (str): The name of the Docker container.
        config_filename (str): The path to the configuration file for the experiment.

    Raises:
        Exception: If any subprocess command fails.
    """
    bench_logger.info(f"Starting experiment in container {container_name} with config in {config_filename}")
    try:
        # check for the existence of curie/setup/env.sh
        if not os.path.exists(task_config["llm_config_filename"]):
            bench_logger.error(f"{task_config['llm_config_filename']} does not exist under setup. Please input your API credentials.")
            return
        if task_config["mode"] == "judge" and not os.path.exists(task_config["llm_judge_config_filename"]):
            bench_logger.error(f"{task_config['llm_judge_config_filename']} does not exist under setup. Please input your judge API credentials.")
            return
        
        if "generate" == task_config["mode"]:
            # Run the experiment inside the container  
            subprocess.run([
                "docker", "exec", "-it", container_name,
                "bash", "-c", (
                    # "source ~/.bashrc && "
                    # "ls -la exp-bench && "
                    f"source {task_config["llm_config_filename"]} && "
                    '''eval "$(micromamba shell hook --shell bash)" && '''
                    "micromamba activate exp-bench && "
                    f"python3 evaluation/eval.py --config-file={config_filename}"
                )
            ], check=True)  # This will block until main.py finishes.
        elif "judge" == task_config["mode"]:
            # Run the experiment inside the container
            subprocess.run([
                "docker", "exec", "-it", container_name,
                "bash", "-c", (
                    # "source ~/.bashrc && "
                    f"source {task_config["llm_judge_config_filename"]} && "
                    '''eval "$(micromamba shell hook --shell bash)" && '''
                    "micromamba activate exp-bench && "
                    f"python3 evaluation/judge.py --config-file={config_filename}"
                )
            ], check=True)  # This will block until main.py finishes.      
        else:
            bench_logger.error(f"Invalid mode {task_config["mode"]} provided in the task configuration.")
            return      

    except subprocess.CalledProcessError as e:
        bench_logger.error(f"Experiment failed with exit code {e.returncode}. Error: {e}")
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

def execute_eval(unique_id, iteration, task_config):
    # Create configuration file
    task_config, config_filename = create_config_file(unique_id, iteration, task_config)

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
    except Exception as e:
        print(f"Error reading config file: {e}")
        return
    
    print(f"EXP-Bench main eval is running with the following configuration: {task_config}")
    iterations = task_config.get("iterations", 1)
    unique_id = datetime.now().strftime("%Y%m%d%H%M%S")
    for iteration in range(1, iterations + 1):
        # Perform the required operation for each iteration (to be provided later)
        # print(f"Iteration {iteration} ")
        task_id = task_config["paper_id"]
        start_time = time.time() 
        execute_eval(unique_id, iteration, task_config)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Iteration {iteration} for {task_id} completed in {elapsed_time:.2f} seconds.")
        bench_logger.info(f"Iteration {iteration} for {task_id} completed in {elapsed_time:.2f} seconds.")
    
if __name__ == "__main__": 
    main() 
