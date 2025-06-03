import subprocess
import time
import os
import json
import uuid
from datetime import datetime
import sys
import shutil

from curie.logger import init_logger, send_question_telemetry

# Constants
DEFAULT_TASK_CONFIG = {
    "job_name": "default_research",
    "workspace_name": "",
    "dataset_dir": "",
    "docker_image": "curie-pip-image",
    "dockerfile_name": "ExpDockerfile_pip", 
    "benchmark_specific_context": "none",
    "is_user_interrupt_allowed": False,
    "timeout": 600,
    "max_global_steps": 50,
    "max_coding_iterations": 25,
    "supervisor_system_prompt_filename": "prompts/simple/simple-supervisor.txt",
    "control_worker_system_prompt_filename": "prompts/simple/simple-control-worker.txt",
    "patcher_system_prompt_filename": "prompts/simple/simple-patcher.txt",
    "llm_verifier_system_prompt_filename": "prompts/simple/simple-llm-verifier.txt",
    "coding_prompt_filename": "prompts/simple/simple-coding.txt",
    "worker_system_prompt_filename": "prompts/simple/simple-worker.txt"
}

DEFAULT_JOB_NAME = "default_research"

def write_api_keys_to_env(api_keys: dict):
    """Write API keys to env.sh file."""
    env_path = os.path.join(os.getcwd(), '.setup', 'env.sh')
    os.makedirs(os.path.dirname(env_path), exist_ok=True)
    # print(f"Writing API keys to {env_path}")
    
    with open(env_path, 'w') as f:
        for key, value in api_keys.items():
            print(f"Writing {key} to {env_path}")
            f.write(f'export {key}="{value}"\n')

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
    command = [
        "sudo", "docker", "build",
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
    
    base_dir = os.getcwd()
    api_key_dir = os.path.join(os.getcwd(), '.setup')
    command = [
        "docker", "run",
        "-v", "/var/run/docker.sock:/var/run/docker.sock",
        "-v", f"{api_key_dir}:/curie/setup/:ro",
        "-v", f"{base_dir}/logs:/logs",
        "-v", f"{base_dir}/workspace:/workspace",
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

def get_workspace_name(task_config):
    """Extract workspace name from task config."""
    return (
        (os.path.basename(task_config.get('workspace_name', '')) or '' ) or 
        task_config.get('job_name', '') or 
        DEFAULT_JOB_NAME
    )

def create_config_file(question_file, unique_id, iteration, task_config):
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

def prepare_question_file(task_config, question_text):
    """Create a question file from question text."""
    q_file = get_workspace_name(task_config)
    question_file = f'workspace/{q_file}_{int(time.time())}.txt'
    
    try:
        os.makedirs(os.path.dirname(question_file), exist_ok=True)
        with open(question_file, 'w') as f:
            f.write(question_text)
        return question_file
    except Exception as e:
        print(f"Error writing question to file: {e}")
        print("Please give permission to write to `workspace/`.")
        sys.exit(1)

def execute_curie(question_filename, unique_id, iteration, task_config):
    """Execute a single Curie iteration."""
    # Create configuration file and get logger
    task_config, config_filename, logger = create_config_file(
        question_filename, unique_id, iteration, task_config)

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
            # force override the docker image and dockerfile name
            task_config['docker_image'] = "curie-pip-image"
            task_config['dockerfile_name'] = "ExpDockerfile_pip"
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

def experiment(api_keys=None, dataset_dir=None, workspace_name=None, question_file=None, question=None, iterations=1, task_config=None):
    """Main experiment function that orchestrates the experiment workflow."""
    # Write API keys to env file if provided
    if api_keys:
        write_api_keys_to_env(api_keys)
    
    # Load and update configuration
    if task_config is None:
        task_config = DEFAULT_TASK_CONFIG
    else:
        task_config = load_and_update_config(task_config, workspace_name, dataset_dir)
    
    if task_config is None:
        return
    
    print(f"Curie is running with the following configuration: {task_config}")
    
    # Validate question input
    if not validate_question_input(question_file, question):
        return
    
    # Prepare question file
    question_file = question_file if question_file else prepare_question_file(task_config, question)
    
    # Run iterations
    for iteration in range(1, iterations + 1):
        start_time = time.time()
        unique_id = datetime.now().strftime("%Y%m%d%H%M%S")
        
        execute_curie(question_file, unique_id, iteration, task_config)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Iteration {iteration} for {question_file} completed in {elapsed_time:.2f} seconds.")

