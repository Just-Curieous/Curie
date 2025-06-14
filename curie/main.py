import subprocess
import time
import os
import argparse
import json
import uuid
from datetime import datetime
import sys
import shutil

from curie.logger import init_logger, send_question_telemetry

# Constants
DEFAULT_CONFIG_PATH = "curie/configs/base_config.json"
DEFAULT_JOB_NAME = "default_research"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process input arguments for the script.")
    
    parser.add_argument("--iterations", type=int, default=1, 
                        help="Number of iterations (must be an integer).")
    
    parser.add_argument("--question_file", "-f", type=str, required=False,
                        help="Question file to run")

    parser.add_argument("--task_config", type=str, default=DEFAULT_CONFIG_PATH,
                        help="Task configuration file for advanced developers.")
    
    # these arguments will overwrite the ones in the task_config file if provided
    parser.add_argument("--codebase_dir", "-c", type=str, default=None, required=False,
                        help="Workspace name (starter code dir) to be used in the experiment.")

    parser.add_argument("--dataset_dir", "-d", type=str, default=None, required=False,
                        help="Dataset directory to be used in the experiment.")
    parser.add_argument("--env_requirements", "-e", type=str, default=None, required=False,
                        help="Environment requirements file to be used in the experiment.")

    parser.add_argument("--question", "-q", type=str, required=False,
                        help="Question to run")
    return parser.parse_args()

def prune_openhands_docker():
    """Remove Docker containers with names starting with 'openhands'."""
    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.Names}}"],
            capture_output=True, text=True, check=True
        )
        container_names = [name for name in result.stdout.splitlines() if name.startswith("openhands")]

        if container_names:
            subprocess.run(["docker", "rm", "-f"] + container_names, check=True)
            print("Removed containers:", ", ".join(container_names))
        else:
            print("No matching containers found.")
    except subprocess.SubprocessError as e:
        print(f"Error pruning Docker containers: {e}")

def get_workspace_name(task_config):
    """Extract workspace name from task config."""
    return (
        (os.path.basename(task_config.get('workspace_name', '')) or '' ) or 
        task_config.get('job_name', '') or 
        DEFAULT_JOB_NAME
    )

def create_config_file(question_file, unique_id, iteration, task_config, env_requirements):
    """Create experiment configuration file and set up logging."""
    work_name = get_workspace_name(task_config)
    
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
        "base_dir": base_dir,
        "env_requirements": env_requirements,
    })
        
    os.makedirs(os.path.dirname(config_filename), exist_ok=True)
    send_question_telemetry(question_file)
    
    with open(config_filename, "w") as f:
        json.dump(task_config, f, indent=4)
    send_question_telemetry(config_filename)
    
    logger = init_logger(log_filename)
    logger.info(f"Config file created: {config_filename}")
    logger.info(f"Check out the log file: {log_filename}")
    
    return task_config, config_filename, logger

def docker_image_exists(image):
    """Check if a Docker image exists locally."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error checking Docker image: {e}")
        return False

def build_docker_image(image_name, dockerfile):
    """Build Docker image if it doesn't exist."""
    sudo_available = shutil.which("sudo") is not None
    command = [
        f"{'sudo ' if sudo_available else ''}docker", "build",
        "--no-cache", "--progress=plain",
        "-t", image_name,
        "-f", dockerfile,
        "."
    ]
    subprocess.run(command, check=True)

def run_docker_container(unique_id, iteration, task_config, logger):
    """Run a Docker container for the experiment."""
    rand_uuid = uuid.uuid4()
    container_name = f"exp-agent-container-{unique_id}-{rand_uuid}-iter_{iteration}"
    
    image_name = task_config["docker_image"]
    docker_filename = task_config["base_dir"] + "/curie/" + task_config["dockerfile_name"]

    if docker_image_exists(image_name):
        logger.info(f"Using existing Docker image: {image_name}")
    else:
        logger.info(f"Start building Docker image {image_name} ... ")
        build_docker_image(image_name, docker_filename)
    
    base_dir = task_config['base_dir']
    if 'dataset_dir' in task_config and task_config['dataset_dir'] is not None:
        dataset_name = f"{task_config['job_name']}_dataset"
        dataset_dir = os.path.abspath(task_config['dataset_dir']).rstrip('/')
        mount_dataset = ["-v",  f"{dataset_dir}:/workspace/{dataset_name}:ro"]
    else:
        mount_dataset = []
    
    command = [
        "docker", "run",
        "-v", "/var/run/docker.sock:/var/run/docker.sock",
        "-v", f"{base_dir}/curie:/curie:ro",
        "-v", f"{base_dir}/benchmark:/benchmark:ro",
        "-v", f"{base_dir}/logs:/logs",
        "-v", f"{base_dir}/starter_file:/starter_file:ro",
        "-v", f"{base_dir}/workspace:/workspace"] + mount_dataset + [
            "-v", f"/:/all:ro",
            "--network=host",
            "-d",
        ]
    
    # Add GPU support if available
    has_gpu = shutil.which("nvidia-smi") is not None and subprocess.call(
        ["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
    if has_gpu:
        command += ["--gpus", "all"]
        
    command += ["--name", container_name, image_name]

    logger.info(f"Running command: {' '.join(command)}")
    subprocess.run(command, check=True) 
    return container_name

def execute_experiment_in_container(container_name, config_file, logger):
    """Execute the experiment inside the Docker container."""
    logger.info(f"Starting experiment in container {container_name} with config in {config_file}")
    
    # Check for required environment file
    if not os.path.exists("curie/setup/env.sh"):
        logger.error("env.sh does not exist under curie/setup. Please input your API credentials.")
        return False
    
    env_output = subprocess.check_output(["/bin/bash", "-c", "source curie/setup/env.sh && env"], text=True)
    for line in env_output.splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            os.environ[key] = value
            
    organization_id = os.environ.get("ORGANIZATION") if os.environ.get("ORGANIZATION") else "014482"
    # Command to run inside container
    container_command = (
        "source setup/env.sh && " 
        '''eval "$(micromamba shell hook --shell bash)" && '''
        "micromamba activate curie && "
        f"sed -i '474i \\                    \"organization\": \"{organization_id}\",' /root/.cache/pypoetry/virtualenvs/openhands-ai-*-py3.12/lib/python3.12/site-packages/litellm/llms/azure/azure.py &&"
        f"sed -i '474i \\    \"organization\": \"{organization_id}\",' /opt/micromamba/envs/curie/lib/python3.11/site-packages/litellm/llms/azure/azure.py  &&"
        "sed -i '49d' /root/.cache/pypoetry/virtualenvs/openhands-ai-*-py3.12/lib/python3.12/site-packages/litellm/llms/azure/chat/o_series_handler.py &&"
        f"sed -i '49i \\                    organization=\"{organization_id}\",' /root/.cache/pypoetry/virtualenvs/openhands-ai-*-py3.12/lib/python3.12/site-packages/litellm/llms/azure/chat/o_series_handler.py  &&"
        f"python3 construct_workflow_graph.py /{config_file}"
    )
    
    try:
        subprocess.run([
            "docker", "exec", "-it", container_name,
            "bash", "-c", container_command
        ], check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Experiment failed with exit code {e.returncode}. Error: {e}")
        return False

def cleanup_docker_container(container_name):
    """Stop and remove the Docker container."""
    try:
        print(f"Stopping and removing Docker container: {container_name}...")
        subprocess.run(["docker", "stop", container_name], check=True)
        subprocess.run(["docker", "rm", container_name], check=True)
        print(f"Docker container {container_name} cleaned up.")
    except subprocess.SubprocessError as e:
        print(f"Error cleaning up container: {e}")

def run_prune_commands():
    """Run Docker pruning commands to free up resources."""
    commands = [
        ["docker", "container", "prune", "-f"],
        ["docker", "image", "prune", "-f"],
        ["docker", "volume", "prune", "-f"],
        ["docker", "builder", "prune", "-f"],
    ]

    for command in commands:
        try:
            print(f"Running docker: {' '.join(command)}")
            result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(result.stdout.decode())
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {' '.join(command)}")
            print(e.stderr.decode())
    
    prune_openhands_docker()

def prepare_question_file(task_config, question_text, question_file):
    """Create a question file from question text."""
    if question_file is not None:
        question_file = os.path.abspath(question_file)
        with open(question_file, 'r') as f:
            question_text = f.read()
        task_config["question"] = question_file
        return question_file, task_config
    
    q_file = get_workspace_name(task_config)
    question_file = f'workspace/{q_file}_{int(time.time())}.txt'
    task_config["question"] = question_file
    
    try:
        os.makedirs(os.path.dirname(question_file), exist_ok=True)
        with open(question_file, 'w') as f:
            f.write(question_text)
        question_file = os.path.abspath(question_file)
        task_config["question"] = question_file
        return question_file, task_config
    except Exception as e:
        print(f"Error writing question to file: {e}")
        print("Please give permission to write to `workspace/`.")
        sys.exit(1)

def execute_curie(question_filename, unique_id, iteration, task_config, env_requirements):
    """Execute a single Curie iteration."""
    # Create configuration file and get logger
    task_config, config_filename, logger = create_config_file(
        question_filename, unique_id, iteration, task_config, env_requirements)

    # Run Docker container for this iteration
    container_name = None
    try:
        container_name = run_docker_container(unique_id, iteration, task_config, logger)
        execute_experiment_in_container(container_name, config_filename, logger)
    finally:
        # Clean up Docker container after each iteration
        if container_name:
            cleanup_docker_container(container_name)
        run_prune_commands()
    
    send_question_telemetry(task_config['log_filename'])

def load_and_update_config(config_path, workspace_name=None, dataset_dir=None):
    """Load and update task configuration with command line arguments."""
    try:
        with open(config_path, 'r') as f:
            task_config = json.load(f)
            task_config['workspace_name'] = workspace_name or task_config['workspace_name']
            task_config['dataset_dir'] = dataset_dir or task_config['dataset_dir']
        return task_config
    except Exception as e:
        print(f"Error reading config file: {e}")
        return None

def validate_question_input(question_file, question):
    """Validate that exactly one of question_file or question is provided."""
    if question_file is None and question is None:
        print("Please provide either a question file or a question.")
        return False
    elif question_file is not None and question is not None:
        print("Please provide only one of either a question file or a question.")
        return False
    return True

def run_iterations(question_file, iterations, task_config, env_requirements):
    """Run the specified number of iterations for the given question file."""
    for iteration in range(1, iterations + 1):
        start_time = time.time()
        unique_id = datetime.now().strftime("%Y%m%d%H%M%S")
        
        execute_curie(question_file, unique_id, iteration, task_config, env_requirements)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Iteration {iteration} for {question_file} completed in {elapsed_time:.2f} seconds.")

def main():
    """Main function to parse args and run iterations."""
    args = parse_args()
    
    # Load and update configuration
    task_config = load_and_update_config(args.task_config, args.codebase_dir, args.dataset_dir)
    if task_config is None:
        return
        
    # Validate question input
    if not validate_question_input(args.question_file, args.question):
        return
    
    # Prepare question file
    question_file, task_config = prepare_question_file(task_config, args.question, args.question_file)
    
    # Run iterations
    run_iterations(question_file, args.iterations, task_config, args.env_requirements)

if __name__ == "__main__":
    main()