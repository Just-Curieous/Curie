import os
import shutil
import json
import sys
import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from helper.utils import load_system_prompt, load_prompt_from_file, _collect_openhands_cost, setup_openhands_credential, setup_utils_logging, safe_json_load, call_oh_with_prompt, print_exception_and_traceback, get_messages_needed, breakdown_message, concatenate_setup_scripts
from utils import get_relative_output_path_eval
from exp_bench.model import llm_query_plain

from helper.logger import init_logger
def setup_gen_setup_logging(log_filename: str):
    global bench_logger 
    bench_logger = init_logger(log_filename)

def run_command(cmd, cwd=None):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    print(f"Running command: {cmd} (in {cwd or 'current directory'})")
    bench_logger.info(f"Running command: {cmd} (in {cwd or 'current directory'})")
    if result.stdout:
        print(f"stdout: {result.stdout.strip()}")
        bench_logger.info(f"stdout: {result.stdout.strip()}")
    if result.stderr:
        print(f"stderr: {result.stderr.strip()}")
        bench_logger.error(f"stderr: {result.stderr.strip()}")
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        bench_logger.error(f"Command failed with return code {result.returncode}")
    return result

def extract_setup_scripts(task_data: dict, config: dict):
    if "masked_source" in task_data:
        # Extract the setup scripts from the task data:
        setup_scripts = task_data["masked_source"]
        # Concatenate the setup scripts into a single string:
        concatenated_setup_scripts = concatenate_setup_scripts(config["github_url"], setup_scripts)
    elif "source" in task_data:
        # Extract the setup scripts from the task data:
        setup_scripts = task_data["source"]
        # Concatenate the setup scripts into a single string:
        concatenated_setup_scripts = concatenate_setup_scripts(config["github_url"], setup_scripts)
    else:
        concatenated_setup_scripts = ""

    return concatenated_setup_scripts

def generate_task_prompt(task_data: dict, config: dict, task_counter: int):
    """
        Generate the task prompt for the agent.
        Inputs:
            task_data: the actual task dict obtained from 1 task within a file such as: outputs/logs/neurips2024/93022/93022_complete_final.json
    """
    # question = task_data["question"] if "question" in task_data else task_data["hypothesis"]
    # method = task_data["method"]
    # agent_instructions = task_data["agent_instructions"] if "agent_instructions" in task_data else "" # could be a type 3 task

    eval_judge_prompt_filename = config["eval_judge_prompt"]
    eval_judge_setup_prompt_filename = config["eval_judge_setup_prompt"]
    eval_judge_setup_partial_prompt_filename = config["eval_judge_setup_partial_prompt"]
    eval_judge_setup_monitor_prompt_filename = config["eval_judge_setup_monitor_prompt"]

    # Determine the output path of the EVALUATION GENERATION phase that we will use as input for judging:
    output_path = get_relative_output_path(config, task_counter) # assume agent workspace view == local workspace view
    print(f"output_path: {output_path}")
    bench_logger.info(f"output_path: {output_path}")
    # Load the output path as a dict:
    with open(output_path, 'r') as f:
        output_json = json.load(f)

    # Obtain agent outputs:
    if "invalid_task" in output_json:
        # If the task is invalid, we need to return a default response:
        print("Task is invalid. Skipping...")
        bench_logger.info("Task is invalid. Skipping...")
        return None, None, None, None
    
    if "no_answer" in output_json:
        design_output = output_json["no_answer"]
        conclusion_output = output_json["no_answer"]
    else:
        design_output = output_json["design"]
        conclusion_output = output_json["conclusion"]

    # Load the setup output path as string:
    setup_output_path = get_relative_output_patch_path(config, task_counter)
    # Check if the setup output path exists:
    if not os.path.exists(setup_output_path):
        setup_output_str = "Empty content"
    else:
        with open(setup_output_path, 'r') as f:
            setup_output_str = f.read()

    setup_output_strs = []
    msg_needed = get_messages_needed(setup_output_str)
    if msg_needed > 1:
        # Split the setup output string into multiple strings:
        print(f"WARNING: MSG TOO LARGE. Breaking down setup output string into {msg_needed} parts")
        bench_logger.info(f"WARNING: MSG TOO LARGE. Breaking down setup output string into {msg_needed} parts")
        setup_output_strs = breakdown_message(setup_output_str, msg_needed)
        print(f"setup_output_strs: {[get_messages_needed(setup_output_strs[i]) for i in range(len(setup_output_strs))]}")
        # while True:
        #     x=1
    else:
        setup_output_strs = [setup_output_str]

    # Load the logs:
    oh_log_path = get_relative_log_path(config, task_counter)
    with open(oh_log_path, 'r') as f:
        logs = f.read()

    prompt_content_template = load_prompt_from_file(eval_judge_setup_monitor_prompt_filename)
    task_monitor_prompt = load_system_prompt(
        prompt_content_template,
        logs=logs,
    )

    default_monitor_json_output = {
        "paper_access": "Error parsing response",
        "git_operations": "Error parsing response",
        "faked_or_nonexperimental_data": "Error parsing response",
        "setup_monitor_comprehensive_reason": "Error parsing response"
    }

    prompt_content_template = load_prompt_from_file(eval_judge_prompt_filename)
    partial_prompt_content_template = load_prompt_from_file(eval_judge_setup_partial_prompt_filename)
    task_prompt = load_system_prompt(
        prompt_content_template,
        design_gt=task_data["design_complexity"],
        conclusion_gt=task_data["expected_outcome"],
        design_output=design_output,
        conclusion_output=conclusion_output,
    )

    default_json_output = {
        "design_evaluation_explanation": "Error parsing response",
        "design_score": "Error parsing response",
        "design_error_analysis": "Error parsing response",
        "conclusion_evaluation_explanation": "Error parsing response", 
        "conclusion_score": "Error parsing response",
        "conclusion_error_analysis": "Error parsing response"
    }

    concatenated_setup_scripts = extract_setup_scripts(task_data, config)

    prompt_content_template = load_prompt_from_file(eval_judge_setup_prompt_filename)
    task_setup_prompt = load_system_prompt(
        prompt_content_template,
        setup_gt=task_data["requirements"] if "requirements" in task_data else task_data["method"], # use method as an approximation for type 3 tasks. # TODO: may consider using the depth setup complexity as well/instead
        setup_output=setup_output_str,
        setup_scripts=concatenated_setup_scripts
    )

    default_setup_json_output = {
        "setup_evaluation_explanation": "Error parsing response",
        "setup_score": "Error parsing response",
        "setup_error_analysis": "Error parsing response",
    }

    if msg_needed == 1:
        return task_prompt, default_json_output, task_setup_prompt, default_setup_json_output, task_monitor_prompt, default_monitor_json_output
    else:
        return task_prompt, default_json_output, {"setup_output_strs": setup_output_strs, "prompt_template": prompt_content_template, "partial_prompt_template": partial_prompt_content_template, "setup_gt": task_data["requirements"] if "requirements" in task_data else task_data["method"], "setup_scripts": concatenated_setup_scripts}, default_setup_json_output, task_monitor_prompt, default_monitor_json_output # will reconstruct this setup judge prompt later

def query_openhands(config, task_prompt, task_counter, github_workspace_path):

    # Save full prompt to a temp file:
    temp_prompt_path = "logs/temp_prompt_eval_{}.txt".format(os.getpid())
    with open(temp_prompt_path, 'w') as f:
        f.write(task_prompt)
    # Get absolute path of the temp prompt file:
    temp_prompt_path = os.path.abspath(temp_prompt_path)
    
    output_path = get_relative_output_path(config, task_counter)
    workspace_response_path = get_agent_relative_output_path(config, task_counter)
    workspace_local_response_path = get_agent_relative_local_output_path(config, task_counter, github_workspace_path)
    oh_log_path = get_relative_log_path(config, task_counter)
    os.makedirs(os.path.dirname(oh_log_path), exist_ok=True)
    os.makedirs(os.path.dirname(workspace_local_response_path), exist_ok=True)

    max_iterations = 2
    iteration = 0

    while True: # repeat until the expected format of JSON is returned
        print("Generating Agent output..")
        bench_logger.info("Generating Agent output..")

        call_oh_with_prompt(task_prompt, temp_prompt_path, config, github_workspace_path, oh_log_path, iterations=30)

        if os.path.exists(workspace_local_response_path):
            # Copy the workspace response to output path:
            subprocess.run(f"cp {workspace_local_response_path} {output_path}", shell=True, check=True)

            # Remove the workspace response file:
            subprocess.run(f"rm -rf {workspace_local_response_path}", shell=True, check=True)

            # Inspect the output file to check if the JSON contains the keys we need:
            response = safe_json_load(output_path)
            if set(response.keys()) == {"design", "conclusion"}:
                print("Successfully parsed Agent output.")
                bench_logger.info("Successfully parsed Agent output.")
                break
            else:
                print("Expected keys not found in the response. Received output: {}. Might Retrying...".format(response))
                bench_logger.info("Expected keys not found in the response. Received output: {}. Might Retrying...".format(response))
                # Optionally, you could add a break condition after a certain number of retries
        else:
            print(f"Workspace response file {workspace_local_response_path} not found. Might Retrying...")
            bench_logger.info(f"Workspace response file {workspace_local_response_path} not found. Might Retrying...")
            # Optionally, you could add a break condition after a certain number of retries
        
        iteration += 1
        if iteration >= max_iterations:
            print("Max iterations reached. Skipping...")
            bench_logger.info("Max iterations reached. Skipping...")
            # Create a json like response with key no_answer:
            response = {"no_answer": "No answer found in analyze_created_setup step of phase 2 after {} iterations.".format(max_iterations)}
            with open(output_path, 'w') as f:
                json.dump(response, f, indent=2)
            break
        else:
            print("Retrying... (iteration {})".format(iteration))
            bench_logger.info("Retrying... (iteration {})".format(iteration))

def query_plain(task_prompt, default_json_output):
    max_iterations = 3
    iteration = 0

    while True: # repeat until the expected format of JSON is returned
        print("Generating Agent output..")
        bench_logger.info("Generating Agent output..")

        response = llm_query_plain("", 
            task_prompt, 
            default_json_output,
        ) # response will be a dict

        if set(response.keys()) == set(default_json_output.keys()): # default_json_output contains the keys we need
            # Check if any key's value is "Error parsing response":
            if not any(value == "Error parsing response" for value in response.values()):
                print("Successfully parsed Agent output.")
                bench_logger.info("Successfully parsed Agent output.")
                break
        
        iteration += 1
        if iteration >= max_iterations:
            print("Max iterations reached. Skipping...")
            bench_logger.info("Max iterations reached. Skipping...")
            # Create a json like response with key no_answer:
            break
        else:
            print("Expected keys not found in the response. Received output: {}. Retrying...".format(response))
            bench_logger.info("Expected keys not found in the response. Received output: {}. Retrying...".format(response))
            print("Retrying... (iteration {})".format(iteration))
            bench_logger.info("Retrying... (iteration {})".format(iteration))
    
    return response

def query_normal(config, task_prompt_details, task_counter):
    try:
        task_prompt, default_json_output, task_setup_prompt, default_setup_json_output, task_monitor_prompt, default_monitor_json_output = task_prompt_details

        # Save full prompt to a temp file:
        temp_prompt_path = "logs/temp_prompt_eval_judge_{}.txt".format(os.getpid())
        with open(temp_prompt_path, 'w') as f:
            f.write(task_prompt)
        # Get absolute path of the temp prompt file:
        temp_prompt_path = os.path.abspath(temp_prompt_path)
        
        output_path = get_relative_judge_output_path(config, task_counter)

        if "skip_before_exec_check" in config:
            with open(output_path, 'r') as f:
                data = json.load(f)
            if "Agent performed forbidden operations" in data["design_evaluation_explanation"]:
                print("Agent performed forbidden operations. Skipping execution...")
                bench_logger.info("Agent performed forbidden operations. Skipping execution...")
                return
            print("Judge file already exists. Skipping directly to execution check since that is not completed...")
            bench_logger.info("Judge file already exists. Skipping directly to execution check since that is not completed...")
            do_not_save, exec_response = execute_setup(config, task_counter)
            # Update the existing output file with the exec response:
            if not do_not_save:
                save_judge_response(data, exec_response, output_path)
            return

        # Zero-eth, extract judge output for setup monitor:
        monitor_response = query_plain(task_monitor_prompt, default_monitor_json_output)
        # Inspect response. If any key's value is True, or if any key's value is "Error parsing response", we need to skip this task.
        if any(value == True for value in monitor_response.values()) or any(value == "Error parsing response" for value in monitor_response.values()):
            print("Agent performed forbidden operations. Skipping...")
            bench_logger.info("Agent performed forbidden operations. Skipping...")
            default_json_output = {
                "design_evaluation_explanation": "Agent performed forbidden operations. See setup_monitor_comprehensive_reason for more details.",
                "design_score": 0,
                "design_error_analysis": "Agent performed forbidden operations. See setup_monitor_comprehensive_reason for more details.",
                "conclusion_evaluation_explanation": "Agent performed forbidden operations. See setup_monitor_comprehensive_reason for more details.", 
                "conclusion_score": 0,
                "conclusion_error_analysis": "Agent performed forbidden operations. See setup_monitor_comprehensive_reason for more details."
            }
            default_setup_json_output = {
                "setup_evaluation_explanation": "Agent performed forbidden operations. See setup_monitor_comprehensive_reason for more details.",
                "setup_score": 0,
                "setup_error_analysis": "Agent performed forbidden operations. See setup_monitor_comprehensive_reason for more details.",
                "setup_monitor_comprehensive_reason": monitor_response["setup_monitor_comprehensive_reason"] if monitor_response["setup_monitor_comprehensive_reason"] != "Error parsing response" else "Specific reason: " + ", ".join([f"{key}: {value}" for key, value in monitor_response.items() if value == True or value == "Error parsing response"]),
            }
            save_judge_response(default_json_output, default_setup_json_output, output_path)
            return

        # First, extract judge output for conclusion and design:
        print(f"Generating design and conclusion output for: {output_path}")
        bench_logger.info("Generating design and conclusion output for: {}".format(output_path))
        response = query_plain(task_prompt, default_json_output)

        # Second extract setup judge output for setup:
        
        # Check if task_setup_prompt is a dict:
        if isinstance(task_setup_prompt, dict): # setup_output contains very long string, need to breakdown into multiple queries
            # task_setup_prompt format: {"setup_output_strs": setup_output_strs, "prompt_template": prompt_content_template, "partial_prompt_template": partial_prompt_content_template, "setup_gt": task_data["requirements"] if "requirements" in task_data else task_data["method"]}
            prompt_details = task_setup_prompt
            for i in range(len(prompt_details["setup_output_strs"])):
                print("Generating setup output for portion {}".format(i))
                bench_logger.info("Generating setup output for portion {}..".format(i))
                setup_gt = prompt_details["setup_gt"]
                setup_output = prompt_details["setup_output_strs"][i]
                if i == 0: # first portion of message:
                    prompt_template = prompt_details["prompt_template"]
                    task_setup_prompt = load_system_prompt(
                        prompt_template,
                        setup_gt=setup_gt,
                        setup_output=setup_output,
                        setup_scripts=prompt_details["setup_scripts"]
                    )
                elif i == 15: # set this as a break point for now, otherwise faced situations where the response is filled with invalid information
                    print("Skipping remaining setup portions.")
                    bench_logger.info("Skipping remaining setup portions.")
                    break
                else:
                    prompt_template = prompt_details["partial_prompt_template"]
                    task_setup_prompt = load_system_prompt(
                        prompt_template,
                        setup_gt=setup_gt,
                        setup_output=setup_output,
                        setup_scripts=prompt_details["setup_scripts"],
                        previous_partial_evaluation=setup_response,
                    )
                setup_response = query_plain(task_setup_prompt, default_setup_json_output)
        else: # task_setup_prompt is a string
            print(f"Generating setup output for: {output_path}")
            bench_logger.info("Generating setup output for: {}".format(output_path))
            setup_response = query_plain(task_setup_prompt, default_setup_json_output)
        
        save_judge_response(response, setup_response, output_path)

        if config["do_exec_check"]:
            do_not_save, exec_response = execute_setup(config, task_counter)
            # Save the response:
            if not do_not_save:
                save_judge_response(response, setup_response, output_path, exec_response)

    except Exception as e:
        print("Error occurred in query_normal:")
        print_exception_and_traceback(e)
        raise  # Re-raise the exception after printing

def execute_setup(config, task_counter):
    # skip if the config file does not exist (older versions don't have this)
    config_path = get_relative_output_config_path(config, task_counter)
    if not os.path.exists(config_path):
        print("Config file does not exist. Skipping execution...")
        bench_logger.info("Config file does not exist. Skipping execution...")
        return True, {
            "execution_success": False,
            "execution_message": "Skipped. Config file does not exist"
        }

    with open(config_path, 'r') as f:
        config_data = json.load(f)
    github_workspace_path = config_data["github_workspace_path"] # e.g., workspace/<repo_name>
    image_name = config_data["docker_image"]
    base_dir = config_data['base_dir'] # e.g., /home/patkon/Benchmark-Construction
    # Generate a unique container name
    container_name = f"judge_setup_exec_{task_counter}_{os.getpid()}"
    
    # skip if the workspace dir does not exists: (meaning we are not in the correct node where the exec gen was generated for this task)
    if not os.path.exists(os.path.join(github_workspace_path)): # we cannot search as os.path.join(base_dir, github_workspace_path) because the host workspace dir is not mounted in the judge container.
        print(f"Workspace dir does not exist: {os.path.abspath(os.path.join(github_workspace_path))}")
        bench_logger.info(f"Workspace dir does not exist: {os.path.join(github_workspace_path)}")
        return True, {
            "execution_success": False,
            "execution_message": f"Workspace dir does not exist: {os.path.join(github_workspace_path)}"
        }
    
    try:
        # Start a new container with proper volume mounts
        command = [
            "docker", "run",
            "-v", f"{base_dir}/{github_workspace_path}:/workspace", # we must copy the host workspace dir to this inner container. Note that judge itself is already running in a container. This is counter to what we did for os.path.exists() above.
            "--network=host",
            "-d",
        ]
        # Add GPU support if available
        has_gpu = shutil.which("nvidia-smi") is not None and subprocess.call(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
        if has_gpu:
            command += ["--gpus", "all"]
        command += ["--name", container_name, image_name]

        bench_logger.info(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=True)

        # Execute the setup script
        try:
            result = subprocess.run([
                "docker", "exec",
                container_name,
                "bash", "-c",
                "cd /workspace && bash reproduce_exp_bench.sh"
            ], capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            
            if result.returncode != 0:
                print(f"Error running setup script: {result.stderr}")
                bench_logger.error(f"Error running setup script: {result.stderr}")
                return False, {
                    "execution_success": False,
                    "execution_message": f"Error running setup script: {result.stderr}"
                }
            
            print(f"Setup script output: {result.stdout}")
            bench_logger.info(f"Setup script output: {result.stdout}")
            return False, {
                "execution_success": True,
                "execution_message": f"Setup script executed successfully. Output: {result.stdout}"
            }
            
        except subprocess.TimeoutExpired:
            print(f"Timeout running setup script")
            bench_logger.warning(f"Timeout running setup script")
            return False, {
                "execution_success": False,
                "execution_message": "Timeout running setup script"
            }
            
    except subprocess.CalledProcessError as e:
        print(f"Docker command failed: {e}")
        bench_logger.error(f"Docker command failed: {e}")
        return False, {
            "execution_success": False,
            "execution_message": f"Docker command failed: {e}"
        }
        
    finally:
        # Clean up - remove the container
        try:
            subprocess.run([
                "docker", "rm", "-f", container_name
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to remove container {container_name}: {e}")
            bench_logger.error(f"Failed to remove container {container_name}: {e}")

def save_judge_response(response, setup_response, output_path, exec_response=None):
    # Combine the two responses into single dict by combining dicts:
    combined_response = {**response, **setup_response}
    if exec_response is not None:
        combined_response = {**response, **setup_response, **exec_response}
    # Save the combined response to the output path:
    print("Saving output to: {}".format(output_path))
    bench_logger.info("Saving output to: {}".format(output_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(combined_response, f, indent=2)

def query_agent(config: dict, task_prompt_details: tuple, task_counter: int):
    if config["judge_agent_name"] == "openhands":
        # Query OpenHands:
        query_openhands(config, task_prompt_details, task_counter, "") # we will never use this for now.
        # Calculate openhands cost:
        filename_suffix = f"{str(config["paper_id"])}_task_index_{task_counter}_iter{config["iteration"]}_duration{config["max_duration_per_task_in_hours"]}_eval_gen"
        _collect_openhands_cost(f"task {task_counter}", filename_suffix, mode="eval_gen")
    elif config["judge_agent_name"] == "": # Currently, we always use this, this will be the raw LLM, no agent.
        query_normal(config, task_prompt_details, task_counter)
        # TODO: collet cost 
    else:
        # Raise warning:
        print(f"Agent {config["judge_agent_name"]} not supported")
        bench_logger.info(f"Agent {config["judge_agent_name"]} not supported")
        raise ValueError(f"Agent {config["judge_agent_name"]} not supported")

def postprocessing(config: dict, repo_path: str, task_counter: int):
    # Obtain git diff patch:
    generate_patch(repo_path, config, task_counter)

    # Remove repo:
    # git_remove_repo(repo_path)

def strip_workspace_prefix(path: str) -> str:
    """
        Remove the "/workspace/" or /tmp/" prefix from the path if it exists.
        Remove starting "/" if it exists.
    """
    if path.startswith("/workspace/"):
        # Remove "/workspace/" prefix
        return path[len("/workspace/"):]
    elif path.startswith("/tmp/"): # an edge case we noticed when the setup extractor agent did not know where the repo is, and cloned their own repo in /tmp of the existing filesystem. We assume that not github repo will have a tmp folder..
        # Remove "/tmp/" prefix
        return path[len("/tmp/"):]
    elif path.startswith("/"): # remove starting "/"
        return path[1:]
    else:
        # No prefix to remove
        return path

def mask_repo(task_data: dict, repo_path: str):
    """
        Inputs:
            task_data: the actual task dict obtained from 1 task within a file such as: outputs/logs/neurips2024/93022/93022_complete_final.json
            repo_path: the path to the cloned repo such as: workspace/<repo_name> (note there is no slash at the end)
    """
    is_mask_fail = False
    failed_masked_sources = []
    # Mask the repo according to the masked_source:
    # We make sure that if a new file with the same name is created by the agent later, we will only show the new lines
    print("Masking repo at:", repo_path)
    bench_logger.info("Masking repo at: " + repo_path)
    if "masked_source" in task_data: # type 1 task
        masked_sources = task_data["masked_source"]
    elif "source" in task_data: # type 2 task
        masked_sources = task_data["source"]
    else: # type 3 task
        return is_mask_fail, failed_masked_sources
    for source in masked_sources:
        # Convert to correct filepath:
        workspace_filename = strip_workspace_prefix(source) # e.g., <some_dir>/<some_dir>/file.py
        actual_source = repo_path + "/" + workspace_filename
        dir_path, just_filename = os.path.split(actual_source)
        # Check if source is a valid file:
        if os.path.isfile(actual_source): 
            print("Masking file:", actual_source)
            bench_logger.info("Masking file: " + actual_source)
            # Remove file from git tracking:
            run_command(f"git rm --cached {just_filename} -f", cwd=dir_path) # we need to enter the "end" dir, because this may be a submodule and repo_path alone (which is main repo) will not work
            # Force remove the file from the filesystem:
            run_command(f"rm -f {workspace_filename}", cwd=repo_path)
        else:
            print("WARNING: File not found:", actual_source)
            bench_logger.info("WARNING: File not found: " + actual_source)
            is_mask_fail = True
            failed_masked_sources.append(actual_source)
    return is_mask_fail, failed_masked_sources

def process_inputs(config: dict, task_data: dict, task_counter: int):
    print("Processing inputs..")
    bench_logger.info("Processing inputs..")

    print("Generating task prompt...")
    bench_logger.info("Generating task prompt...")
    task_prompt_details = generate_task_prompt(task_data, config, task_counter)
    print("Generate task prompt done.")
    bench_logger.info("Generate task prompt done.")

    return task_prompt_details

def get_output_eval_filename(config: dict, task_counter: int, mode: str = "generate"):
    if mode == "generate":
        return f"/{str(config["paper_id"])}_task_index_{task_counter}_iter_{config["iteration"]}_duration_{config["max_duration_per_task_in_hours"]}_eval_gen.json"
    elif mode == "judge":
        return f"/{str(config["paper_id"])}_task_index_{task_counter}_iter_{config["iteration"]}_duration_{config["max_duration_per_task_in_hours"]}_eval_judge.json"

def get_agent_relative_output_path(config: dict, task_counter: int):
    """
        Get the final relative output path (for agent) for design and conclusion
    """
    return "/workspace" + get_output_eval_filename(config, task_counter)

def get_relative_log_path(config: dict, task_counter: int):
    return config["output_log_folder"] + f"/{str(config["paper_id"])}_task_index_{task_counter}_iter_{config["iteration"]}_duration_{config["max_duration_per_task_in_hours"]}_eval_gen_{config["agent_name"]}_logs.txt"

def get_agent_relative_local_output_path(config: dict, task_counter: int, repo_path: str):
    """
        Get the relative output path (local, i.e., exp-bench docker or actual local) version of get_agent_relative_output_path for design and conclusion

        Inputs:
            repo_path: the path to cloned repo such as: workspace/<repo_name> (note there is no slash at the end)
    """
    return repo_path + get_output_eval_filename(config, task_counter)
    
def get_relative_output_path(config: dict, task_counter: int):
    """
        Get the final relative output path (local, i.e., exp-bench docker or actual local) for design and conclusion
    """
    return config["output_folder"] + get_output_eval_filename(config, task_counter)

def get_relative_judge_output_path(config: dict, task_counter: int):
    """
        Get the final relative output path (local, i.e., exp-bench docker or actual local) for design and conclusion
    """
    return config["output_folder"] + get_output_eval_filename(config, task_counter, mode="judge")

def get_relative_output_patch_path(config: dict, task_counter: int):
    """
        Get the final relative output path (local, i.e., exp-bench docker or actual local) for git diff patch
    """
    return config["output_folder"] + f"/{str(config["paper_id"])}_task_index_{task_counter}_iter_{config["iteration"]}_duration_{config["max_duration_per_task_in_hours"]}_eval_gen.patch"

def get_relative_output_config_path(config: dict, task_counter: int):
    """
        Get the final relative output path (local, i.e., exp-bench docker or actual local) for config
    """
    return config["output_folder"] + f"/{str(config["paper_id"])}_task_index_{task_counter}_iter_{config["iteration"]}_duration_{config["max_duration_per_task_in_hours"]}_eval_gen_config.json"

def process_task(args):
    """
        Inputs:
            task_counter: the index of the task in the list, just a counter for repo construction
            task: the actual task dict obtained from 1 task within a file such as: outputs/logs/neurips2024/93022/93022_complete_final.json
            config: the config dict containing all the parameters for the pipeline
    """
    task_counter, task_data, config = args
    # Check if iteration for this task:
    output_judge_path = get_relative_judge_output_path(config, task_counter) # output path for conclusion & design
    if os.path.isfile(output_judge_path):
        if config["do_exec_check"]:
            # Open file and check if "execution_success" exists:
            with open(output_judge_path, 'r') as f:
                data = json.load(f)
                if "execution_success" in data:
                    print("Task {} for paper {} already processed (i.e., judged). Skipping...".format(task_counter, str(config["paper_id"])))
                    bench_logger.info("Task {} for paper already processed (i.e., judged). Skipping...".format(task_counter, str(config["paper_id"])))
                    return
                else: # everything else is done except execution check
                    config["skip_before_exec_check"] = True
        else:
            print("Task {} for paper {} already processed (i.e., judged). Skipping...".format(task_counter, str(config["paper_id"])))
            bench_logger.info("Task {} for paper already processed (i.e., judged). Skipping...".format(task_counter, str(config["paper_id"])))
            return

    # Prepare inputs:
    # - Question prompt construction: hypothesis+method+agent_instruction
    # - Repo related: (1) clone repo, (2) remove scripts according to masked_source
    task_prompt_details = process_inputs(config, task_data, task_counter)

    # Call LLM with appropriate prompts: use appropriate agent
    # - judge output file: if not saved in correct place repeat x runs
    if task_prompt_details[0] is not None:
        query_agent(config, task_prompt_details, task_counter)

def main(
        config: dict
    ):
    # for loop for parallel execution. check through paper folder to obtain tasks
    with open(config["input_paper_tasks_filename"], 'r') as f:
        paper_tasks_complete = json.load(f)
    tasks_list = []
    for task_counter, task in enumerate(paper_tasks_complete["questions"]):
        # Check if eval_gen.json exists for this task:
        output_file = get_relative_output_path_eval(config, task_counter, config["iteration"], "generate")
        if not os.path.exists(output_file):
            print(f"❌ NOTE: {output_file} does not exist or is empty, skipping.")
            continue
        tasks_list.append((task_counter, task, config))

    with ProcessPoolExecutor(max_workers=6) as executor:
        processed_tasks = list(executor.map(process_task, tasks_list))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run evaluation/judge with various agents/LLMs pipeline")
    parser.add_argument("--config-file", type=str, default="config.yaml", help="Path to the config file (e.g., config.yaml)")

    args = parser.parse_args()

    with open(args.config_file, 'r') as file:
        config = json.load(file)

    log_filename = config["log_judge_filename"]
    log_path = Path(log_filename)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # setup_openhands_credential(config["llm_config_filename"])
    setup_gen_setup_logging(log_filename)
    setup_utils_logging(log_filename)

    main(
        config=config,
    )