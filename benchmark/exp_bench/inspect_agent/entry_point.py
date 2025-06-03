#!/usr/bin/env python3
import argparse
import subprocess
import os
import uuid
import sys
import shlex
import time
import json
from pathlib import Path

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

def get_relative_log_path(prompt_path: str, unique_id: str = None) -> str:
    """
    Get a unique log file path based on the input prompt path.
    The prompt path is expected to be in format: .../paper_id/task_index/iter/...
    """
    if unique_id is None:
        unique_id = str(int(time.time()))

    # Create log directory structure
    log_dir = os.path.join(os.path.dirname(prompt_path), "inspect_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    log_filename = f"inspect_log_{unique_id}.log"
    return os.path.join(log_dir, log_filename)

def run_docker_command(base_dir, prompt_path, code_repo_path, max_timeout_in_seconds=3600, inspect_agent_dir_path=None, env_file=None):
    """
    Run the start.sh script inside a Docker container with the provided paths.
    
    Args:
        base_dir (str): Path to the base directory ("/home/patkon/Benchmark-Construction")
        prompt_path (str): Path to the JSON file ("logs/temp_prompt_eval_{}.txt")
        code_repo_path (str): Path to the code repository ("workspace/<repo_name>")
        inspect_agent_dir_path (str): Path to the inspect_agent directory ("inspect_agent")
        env_file (str): Path to the environment file to use
    """
    # Validate that the paths exist
    if not os.path.exists(prompt_path):
        print(f"Error: JSON file '{prompt_path}' does not exist.")
        sys.exit(1)
    
    if not os.path.exists(code_repo_path):
        print(f"Error: Code repository '{code_repo_path}' does not exist.")
        sys.exit(1)
    
    # Get absolute paths
    inspect_agent_dir_abs_path = base_dir + "/" + inspect_agent_dir_path # abs path in host, not container that eval.py is running in
    code_repo_path = base_dir + "/" + code_repo_path # abs path in host, not container that eval.py is running in

    # Check and build Docker image if needed
    image_name = "pb-env"
    dockerfile_name = os.path.join(inspect_agent_dir_path, "Dockerfile.base")
    
    if docker_image_exists(image_name):
        print(f"Using existing Docker image: {image_name}")
    else:
        print(f"Building Docker image {image_name}...")
        build_command = [
            "docker", "build",
            "--platform=linux/amd64",
            "--no-cache",
            "--progress=plain",
            "-t", image_name,
            "-f", dockerfile_name, 
            inspect_agent_dir_path  # Use inspect_agent_dir_path as build context. For building container, we use the paths within the outer container, i.e., not the host 
        ]
        try:
            subprocess.run(build_command, check=True)
            print(f"Docker image {image_name} built successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error building Docker image: {e}")
            return e.returncode

    # Construct the Docker run command
    docker_id = f"{int(time.time())}-{uuid.uuid4().hex[:6]}"
    docker_command = [
        "docker", "run",
        "-it",
        "--name", f"pb-env-{docker_id}",
        "-v", f"{inspect_agent_dir_abs_path}:/inspect_agent",
        "-v", f"{code_repo_path}:/workspace",  # This will store the code repo, and is the only dir that the agent can write to (i.e., including the final design/conclusion output we are looking for)
        image_name,
        "/bin/bash", "-c",
        f"cd /inspect_agent && bash start.sh {env_file}"
    ]
    
    # Print the command that will be executed
    print(f"👩‍💻 Executing command: {' '.join(docker_command)}")
    
    import signal
    
    return_code = 0
    try:
        # Run the Docker command with timeout
        p = subprocess.Popen(
            docker_command,
            start_new_session=True  # ensures a new process group
        )
        p.wait(timeout=max_timeout_in_seconds) # 1 hour timeout
        print("Docker command executed successfully.")
    except subprocess.TimeoutExpired:
        print(f"Timeout expired for Docker command, killing process group...", flush=True)
        try:
            # Get the process group ID and kill the process group
            pgid = os.getpgid(p.pid)
            os.killpg(pgid, signal.SIGTERM)  # send SIGTERM to the process group
            try:
                p.wait(timeout=10)
                print("Process group terminated with SIGTERM.")
            except subprocess.TimeoutExpired:
                print("SIGTERM failed, sending SIGKILL...")
                os.killpg(pgid, signal.SIGKILL)
                p.wait()
        except Exception as e:
            print(f"Failed to terminate process group: {e}")
        return_code = 1
    except subprocess.CalledProcessError as e:
        print(f"Error executing Docker command: {e}")
        return_code = e.returncode
    except Exception as e:
        print(f"Unexpected error: {e}")
        return_code = 1
    finally:
        try:
            subprocess.run(["docker", "rm", "-f", f"pb-env-{docker_id}"], check=True)
            print("Container removed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error removing container: {e}")
        
        return return_code

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run start.sh script inside Docker container with specified paths.")
    parser.add_argument("--base_dir", required=True, help="Path to the base directory")
    parser.add_argument("--prompt_path", required=True, help="Path to the JSON file")
    parser.add_argument("--code_repo_path", required=True, help="Path to the code repository")
    parser.add_argument("--inspect_agent_dir_path", required=True, help="Path to the inspect file")
    parser.add_argument("--env_file", required=True, help="Path to the environment file")
    parser.add_argument("--max_timeout_in_seconds", required=False, help="Max timeout in seconds", default=3600)
    parser.add_argument("--remove-container", action="store_true", help="Remove the container after execution")

    # Parse arguments
    args = parser.parse_args()
    
    # Run the Docker command
    return_code = run_docker_command(args.base_dir, args.prompt_path, args.code_repo_path, args.max_timeout_in_seconds, args.inspect_agent_dir_path, args.env_file)
    
    sys.exit(return_code)

if __name__ == "__main__":
    main()

# python entry_point.py --json_path /home/ubuntu/Benchmark-Construction/logs/neurips2024/95262.json --code_repo_path /home/ubuntu/Benchmark-Construction/logs/neurips2024/MoE-Jetpack --inspect_path /home/ubuntu/inspect-agent