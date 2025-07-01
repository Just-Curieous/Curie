import subprocess
import time
import os
import json
import uuid
from datetime import datetime
import sys
import shutil
from importlib.resources import files
from typing import Optional, Dict, Any, Union
from curie.logger import init_logger, send_question_telemetry
from curie.docker_setup import ensure_docker_installed

# Constants
DEFAULT_TASK_CONFIG = {
    "job_name": "default_research",
    "docker_image": "amberljc/curie:latest",
    "dockerfile_name": "ExpDockerfile_pip", 
    "benchmark_specific_context": "none",
    "is_user_interrupt_allowed": False,
    "timeout": 600,
    "max_coding_iterations": 25,
    "max_global_steps": 20,
    "supervisor_system_prompt_filename": "prompts/simple/simple-supervisor.txt",
    "control_worker_system_prompt_filename": "prompts/simple/simple-control-worker.txt",
    "patcher_system_prompt_filename": "prompts/simple/simple-patcher.txt",
    "llm_verifier_system_prompt_filename": "prompts/simple/simple-llm-verifier.txt",
    "coding_prompt_filename": "prompts/simple/simple-coding.txt",
    "worker_system_prompt_filename": "prompts/simple/simple-worker.txt",
    "workspace_name": "", # to be filled up by the user
    "dataset_dir": "", # to be filled up by the user
    "env_requirements": "", # to be filled up by the user
    "code_instructions": "", # to be filled up by the user
}

DEFAULT_JOB_NAME = "default_research"

def write_api_keys_to_env(api_keys: Dict[str, str]) -> None:
    """Write API keys to env.sh file."""
    env_path = os.path.join(os.getcwd(), '.setup', 'env.sh')
    os.makedirs(os.path.dirname(env_path), exist_ok=True) 
    # print(f"Writing API keys to {env_path}")
    
    with open(env_path, 'w') as f:
        for key, value in api_keys.items():
            print(f"Writing {key} to {env_path}")
            f.write(f'export {key}="{value}"\n')

def docker_image_exists(image: str) -> bool:
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

def run_docker_container(unique_id: str, iteration: int, task_config: Dict[str, Any], logger: Any) -> str:
    """Run a Docker container for the experiment."""
    rand_uuid = uuid.uuid4()
    container_name = f"exp-agent-container-{unique_id}-{rand_uuid}-iter_{iteration}"
    
    image_name = task_config["docker_image"]
    docker_filename = files("curie") / task_config["dockerfile_name"]

    if docker_image_exists(image_name):
        logger.info(f"Using existing Docker image: {image_name}")
    else:
        logger.info(f"Pulling Docker image {image_name}...")
        subprocess.run(["docker", "pull", image_name], check=True)
    
    if 'dataset_dir' in task_config and task_config['dataset_dir'] != '':
        dataset_name = f"{task_config['job_name']}_dataset"
        dataset_dir = os.path.abspath(task_config['dataset_dir']).rstrip('/')
        mount_dataset = ["-v",  f"{dataset_dir}:/workspace/{dataset_name}:ro"]
    else:
        mount_dataset = []
    
    base_dir = os.getcwd()
    api_key_dir = os.path.join(os.getcwd(), '.setup')
    command = [
        "docker", "run",
        "-v", "/var/run/docker.sock:/var/run/docker.sock",
        # "-v", f"{base_dir}/curie:/curie", # for local development
        "-v", f"{api_key_dir}:/curie/setup/",
        "-v", f"{base_dir}/logs:/logs",
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

    logger.info(f"💻 Running command: {' '.join(command)}")
    subprocess.run(command, check=True) 
    return container_name

def execute_experiment_in_container(container_name: str, config_file: str, logger: Any) -> bool:
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

def cleanup_docker_container(container_name: str) -> None:
    """Stop and remove the Docker container."""
    try:
        print(f"Stopping and removing Docker container: {container_name}...")
        subprocess.run(["docker", "stop", container_name], check=True)
        subprocess.run(["docker", "rm", container_name], check=True)
        print(f"Docker container {container_name} cleaned up.")
    except subprocess.SubprocessError as e:
        print(f"Error cleaning up container: {e}")

def run_prune_commands() -> None:
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

def get_workspace_name(task_config: Dict[str, Any]) -> str:
    """Extract workspace name from task config."""
    return (
        (os.path.basename(task_config.get('workspace_name', '')) or '' ) or 
        task_config.get('job_name', '') or 
        DEFAULT_JOB_NAME
    )

def create_config_file(question_file: str, unique_id: str, iteration: int, task_config: Dict[str, Any]) -> tuple[Dict[str, Any], str, Any]:
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
    
    with open(config_filename, "w") as f:
        json.dump(task_config, f, indent=4)

    send_question_telemetry(question_file)
    send_question_telemetry(config_filename)
    
    logger = init_logger(log_filename)
    logger.info(f"Config file created: {config_filename}")
    logger.info(f"Check out the log file: {log_filename}")
    
    return task_config, config_filename, logger

def prepare_question_file(task_config: Dict[str, Any], question_text: Optional[str] = None, question_file: Optional[str] = None) -> str:
    if question_file is not None:
        with open(question_file, 'r') as f:
            question_text = f.read()
    
    q_file = get_workspace_name(task_config)
    question_file = os.path.join('workspace', f'{q_file}_{int(time.time())}.txt')
    
    try:
        os.makedirs(os.path.dirname(question_file), exist_ok=True)
        with open(question_file, 'w') as f:
            f.write(question_text)
        question_file = os.path.abspath(question_file)
        print(f"❓ Question file created: {question_file}")
        task_config["question"] = question_file
        return question_file, task_config
    except Exception as e:
        print(f"Error writing question to file: {e}")
        print("Please give permission to write to `workspace/`.")
        sys.exit(1)

def execute_curie(question_filename: str, unique_id: str, iteration: int, task_config: Dict[str, Any]) -> None:
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

def prepare_config(task_config: Optional[Dict[str, Any]] = None, 
                 codebase_dir: Optional[str] = None, 
                 dataset_dir: Optional[str] = None, 
                 code_instructions: Optional[str] = None,
                 max_global_steps: int = 30,
                 env_requirements: Optional[str] = None) -> Dict[str, Any]:
    """Load and update task configuration with command line arguments."""
    codebase_dir = os.path.abspath(codebase_dir) if codebase_dir else ''
    dataset_dir = os.path.abspath(dataset_dir) if dataset_dir else ''
    env_requirements = os.path.abspath(env_requirements) if env_requirements else ''
    
    if codebase_dir and not os.path.exists(os.path.abspath(codebase_dir)):
        raise ValueError(f"Codebase directory {codebase_dir} is not a valid path.")
    if dataset_dir and not os.path.exists(os.path.abspath(dataset_dir)):
        raise ValueError(f"Dataset directory {dataset_dir} is not a valid path.") 
    if env_requirements and not os.path.exists(os.path.abspath(env_requirements)):
        raise ValueError(f"Environment requirements file {env_requirements} is not a valid path.")
    
    if code_instructions and not codebase_dir:
        raise ValueError("Code instructions file is provided but codebase directory is not provided. \
                        Please specify the codebase directory by `codebase_dir=''`.")
    elif codebase_dir and not (code_instructions or os.path.exists(os.path.join(codebase_dir, "description.md"))):
        print("[Recommendation] Please specify the code instructions by `code_instructions=''`.")
    elif code_instructions:
        with open(os.path.join(codebase_dir, "description.md"), "w") as f:
            f.write(f"Code Instructions:\n{code_instructions}")

    if codebase_dir and os.path.exists(os.path.join(codebase_dir, "requirements.txt")):
        env_requirements = os.path.join(codebase_dir, "requirements.txt")
        print(f"Found requirements.txt in the codebase directory {codebase_dir}. Using it as the environment requirements file.")
    
    if task_config is None:
        task_config = DEFAULT_TASK_CONFIG.copy()

    # Update paths with absolute paths if provided
    path_updates = {
        'workspace_name': codebase_dir,
        'dataset_dir': dataset_dir,
        'env_requirements': env_requirements
    }

    for key, path in path_updates.items():
        task_config[key] = os.path.abspath(path) if path else task_config.get(key, '')

    # Fill missing fields from defaults
    for key, value in DEFAULT_TASK_CONFIG.items():
        if key not in task_config:
            task_config[key] = value

    # Set required overrides
    task_config.update({
        'max_global_steps': max_global_steps,
        'docker_image': "amberljc/curie:latest",
        'dockerfile_name': "ExpDockerfile_pip"
    })

    return task_config

def validate_input(question_file: Optional[str], 
                            question: Optional[str]) -> bool:
    """Validate that exactly one of question_file or question is provided."""
    if question_file is None and question is None:
        raise ValueError("Please provide either a question file or a question.")
    elif question_file is not None and question is not None:
        raise ValueError("Please provide only one of either a question file or a question.")
    elif question_file is not None and not os.path.exists(question_file):
        raise ValueError(f"Question file {question_file} does not exist.") 
    return 

def experiment(api_keys: Optional[Dict[str, str]] = None, 
               dataset_dir: Optional[str] = None, 
               codebase_dir: Optional[str] = None, 
               code_instructions: Optional[str] = None,
               question_file: Optional[str] = None, 
               question: Optional[str] = None, 
               task_config: Optional[Dict[str, Any]] = None, 
               env_requirements: Optional[str] = None,
               max_global_steps: int = 30) -> None:
    """Main experiment function that orchestrates the experiment workflow."""
    # Write API keys to env file if provided
    if api_keys:
        write_api_keys_to_env(api_keys)
    ensure_docker_installed()
    # Load and update configuration
    task_config = prepare_config(task_config, codebase_dir, dataset_dir, code_instructions, max_global_steps, env_requirements)
    
    print(f"Curie is running with the following configuration: {task_config}")
    
    # Validate question input
    validate_input(question_file, question)
        
    # Prepare question file
    question_file, task_config = prepare_question_file(task_config, question, question_file)
    
    # Run iterations
    iterations = 1
    for iteration in range(1, iterations + 1):
        start_time = time.time()
        unique_id = datetime.now().strftime("%Y%m%d%H%M%S")
        
        execute_curie(question_file, unique_id, iteration, task_config)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Iteration {iteration} for {question_file} completed in {elapsed_time:.2f} seconds.")

