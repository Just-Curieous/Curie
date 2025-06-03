import os
import shutil
import json
import sys
import re
import time
import subprocess
import datetime
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from helper.utils import load_system_prompt, load_prompt_from_file, _collect_openhands_cost, setup_openhands_credential, setup_utils_logging, safe_json_load, call_oh_with_prompt, print_exception_and_traceback, _collect_inspectai_cost

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

def generate_patch(repo_path, config, task_counter):
    print("Generating git diff patch...")
    bench_logger.info("Generating git diff patch...")
    patch_file = get_relative_output_patch_path(config, task_counter)
    # Get absolute path of the patch file:
    patch_file = os.path.abspath(patch_file)
    original_dir = os.getcwd()
    os.system(f"git config --global --add safe.directory {repo_path}")
    os.chdir(repo_path) 
    os.system("git config core.fileMode false")

    # Add new files in the main repo
    os.system("git ls-files --others --exclude-standard | xargs git add -N")

    # Add new files in submodules (ignore failure with '|| true')
    os.system("git submodule foreach --quiet --recursive 'git ls-files --others --exclude-standard | xargs git add -N || true'")

    # Show git diff (main repo)
    os.system(f'git diff -G"." | grep -v \'^diff --git\' | grep -v \'^index\' > {patch_file}')

    # Show git diff (submodules) and append to patch file
    os.system(f'git submodule foreach --recursive \'git diff -G"."\' | grep -v \'^diff --git\' | grep -v \'^index\' >> {patch_file}')

    os.chdir(original_dir)

    # Read contents of patch file and print warning if empty:
    with open(patch_file, 'r') as f:
        patch_contents = f.read()

    # Shorten patch file if it's too large
    size_bytes = os.path.getsize(patch_file)
    size_mb = size_bytes / (1024 * 1024)
    original_size_mb = size_mb
    print(f"Original patch file size: {original_size_mb:.2f}MB")
    bench_logger.info(f"Original patch file size: {original_size_mb:.2f}MB")
    size_threshold_mb = 70
    head_lines = 5000
    tail_lines = 5000
    min_lines = 10  # Minimum number of lines to keep for each section

    if size_mb > size_threshold_mb:
        print(f"WARNING: Patch file exceeds {size_threshold_mb}MB, truncating...")
        bench_logger.info(f"WARNING: Patch file exceeds {size_threshold_mb}MB, truncating...")
        
        with open(patch_file, 'r') as f:
            lines = f.readlines()

        while size_mb > size_threshold_mb and head_lines >= min_lines and tail_lines >= min_lines:
            if len(lines) <= head_lines + tail_lines:
                clipped = lines
            else:
                clipped = lines[:head_lines] + ['...\n'] + lines[-tail_lines:]
            
            # Write temporary clipped content
            with open(patch_file, 'w') as f:
                f.writelines(clipped)
            
            # Check new size
            size_bytes = os.path.getsize(patch_file)
            size_mb = size_bytes / (1024 * 1024)
            
            if size_mb > size_threshold_mb:
                # Halve the number of lines for next iteration
                head_lines = max(head_lines // 2, min_lines)
                tail_lines = max(tail_lines // 2, min_lines)
                print(f"Still too large ({size_mb:.2f}MB), reducing to {head_lines} head and {tail_lines} tail lines...")
                bench_logger.info(f"Still too large ({size_mb:.2f}MB), reducing to {head_lines} head and {tail_lines} tail lines...")
            
    if not patch_contents.strip():
        print("WARNING: Patch file is empty. No changes detected.")
        bench_logger.info("WARNING: Patch file is empty. No changes detected.")
    else:
        final_size_mb = os.path.getsize(patch_file) / (1024 * 1024)
        print(f"Patch file generated successfully: {patch_file} (final size: {final_size_mb:.2f}MB)")
        bench_logger.info(f"Patch file generated successfully: {patch_file} (final size: {final_size_mb:.2f}MB)")

def convert_ssh_submodules_to_https(repo_dir="."):
    gitmodules_path = Path(repo_dir) / ".gitmodules"
    if not gitmodules_path.exists():
        print("No .gitmodules file found.")
        bench_logger.info("No .gitmodules file found.")
        return ""

    with open(gitmodules_path, "r") as f:
        content = f.read()

    # Convert SSH to HTTPS in URLs
    new_content = re.sub(
        r"url = git@github\.com:(.+)\.git",
        r"url = https://github.com/\1.git",
        content
    )

    if new_content != content:
        print("🔧 SSH submodule URLs detected. Rewriting to HTTPS...")
        bench_logger.info("SSH submodule URLs detected. Rewriting to HTTPS...")
        with open(gitmodules_path, "w") as f:
            f.write(new_content)

        print("✅ Submodules successfully updated using HTTPS.")
        bench_logger.info("Submodules successfully updated using HTTPS.")
    else:
        print("✅ All submodule URLs are already using HTTPS.")
        bench_logger.info("All submodule URLs are already using HTTPS.")

def update_submodules(repo_path: str):
    original_dir = os.getcwd()  # Save current working directory
    try:
        os.chdir(repo_path)  # Change to repo directory
        print(f"📁 Changed to repo dir: {repo_path}")

        # Sync and update submodules
        os.system("git submodule sync")
        os.system("git submodule update --init --recursive")

        print("✅ Submodules synced and updated.")
    except Exception as e:
        print_exception_and_traceback(e, prefix=f"❌ Error updating submodules")
    finally:
        os.chdir(original_dir)  # Change back to original dir
        print(f"↩️ Returned to original dir: {original_dir}")

def git_clone_repo(github_url, base_dir="workspace", suffix=""):
    """
    Clones the GitHub repo into base_dir/ (letting git decide the folder name),
    and returns the expected full path of the cloned repo.
    """
    # Get repo name from URL (e.g., 'repo.git' → 'repo')
    repo_name = os.path.splitext(os.path.basename(urlparse(github_url).path))[0]
    expected_path = os.path.join(base_dir, repo_name + suffix) # expected format: "workspace/repo_name"

    # If path exists, append timestamp to make it unique
    if os.path.exists(expected_path):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        expected_path = os.path.join(base_dir, f"{repo_name}_{timestamp}{suffix}")
        print(f"WARNING: Normal repo path already exists. Creating unique repo path with timestamp: {expected_path}")
        bench_logger.info(f"WARNING: Normal repo path already exists. Creating unique repo path with timestamp: {expected_path}")

    # Clone without specifying destination (git will create repo_name folder)
    subprocess.run(["git", "clone", github_url, expected_path], check=True)

    subprocess.run(["chmod", "777", "-R", expected_path], check=True)

    # Make sure submodules are cloned too:
    ret_val = convert_ssh_submodules_to_https(expected_path)
    if ret_val != "":
        update_submodules(expected_path)

    # Copy setup (containing credentialsfiles into cloned repo
    shutil.copytree("evaluation/setup", os.path.join(expected_path, "setup_apis_exp"), dirs_exist_ok=True)

    return expected_path # expected format: "workspace/repo_name"

def git_remove_repo(repo_path):
    """
    Removes the given repo directory (recursively), similar to 'rm -rf'.
    """
    if os.path.exists(repo_path):
        print(f"Removing repo directory at: {repo_path}")
        shutil.rmtree(repo_path)
    else:
        print(f"Directory does not exist at: {repo_path}, nothing to remove.")

    return repo_path

def generate_task_prompt(task_data: dict, config: dict, task_counter: int):
    """
        Generate the task prompt for the agent.
        Inputs:
            task_data: the actual task dict obtained from 1 task within a file such as: outputs/logs/neurips2024/93022/93022_complete_final.json
    """
    question = task_data["question"] if "question" in task_data else task_data["hypothesis"]
    method = task_data["method"]
    agent_instructions = task_data["agent_instructions"] if "agent_instructions" in task_data else "" # could be a type 3 task

    eval_gen_prompt_filename = config["eval_gen_prompt"]

    if config["agent_name"] in ["openhands", "inspectai"]:
        output_path = get_agent_relative_output_path(config, task_counter)
        output_script_name = "/workspace/reproduce_exp_bench.sh"
        additional_info = "The code repo is available in /workspace. LLM related credentials (if needed) are available in /workspace/setup_apis_exp/"
    else: # TODO: need to handle this later for other agents potentially
        output_path = get_relative_output_path(config, task_counter) # assume agent workspace view == local workspace view
        output_script_name = "reproduce_exp_bench.sh"
        additional_info = "LLM related credentials (if needed) are available in /workspace/setup_apis_exp/" # May need to fix this later for other agents
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    prompt_content_template = load_prompt_from_file(eval_gen_prompt_filename)
    task_prompt = load_system_prompt(
        prompt_content_template,
        question=question,
        method=method,
        agent_instructions=agent_instructions,
        output_json_path=output_path,
        output_script_name=output_script_name,
        additional_info=additional_info
    )

    return task_prompt

def _query_inspectai(config, task_prompt, task_counter, github_workspace_path, inspectai_log_path):
    """
        Currently, works for Claude models on Bedrock. Not working for deepseek (bedrock/converse) or openai UM
    """
    inspect_agent_dir_path = 'inspect_agent'
    # Save model env file to inspect_agent dir:
    original_env_file_path = config["llm_config_filename"] # contains model creds
    remote_custom_prompt_filepath = f"{os.getpid()}_{str(config['paper_id'])}_task_index_{task_counter}_iter_{config['iteration']}_duration_{config['max_duration_per_task_in_hours']}_agent_prompt.txt" # this is what the inspectai agent will see, since we cd before running start.sh
    custom_prompt_filepath = f"{inspect_agent_dir_path}/{remote_custom_prompt_filepath}"
    env_file_contents = f"""
export CODE_DIR=/home/code
export AGENT_DIR=/home/agent
export WORKSPACE_BASE=/home/paper
export MAX_TIME_IN_HOURS={config["max_duration_per_task_in_hours"]}
export DISALLOW_SUBMIT=False
export ITERATIVE_AGENT=True
export PROMPT_FILE_PATH={remote_custom_prompt_filepath}

"""
    # Append the contents of original_env_file_path to env_file_contents:
    with open(original_env_file_path, 'r') as f:
        env_file_contents += f.read()

    # If "AWS_REGION_NAME" exists in env_file_contents, replace it with AWS_DEFAULT_REGION
    if "AWS_REGION_NAME" in env_file_contents:
        env_file_contents = env_file_contents.replace("AWS_REGION_NAME", "AWS_DEFAULT_REGION")

    # Generate unique env.sh path
    unique_env_filename = f"env_{os.getpid()}_{str(config['paper_id'])}_task_index_{task_counter}_iter_{config['iteration']}_duration_{config['max_duration_per_task_in_hours']}.sh"
    model_env_path = f"inspect_agent/{unique_env_filename}"
    with open(model_env_path, 'w') as f: # this will be used within start.sh. We overwrite whatever env.sh that currently exists
        f.write(env_file_contents)

    # Save prompt contents by reading from prompt_path first and then writing to a file within inspect_agent dir:
    with open(custom_prompt_filepath, "w") as f: # we will be using this within start.py
        f.write(task_prompt)

    command = f"python inspect_agent/entry_point.py --base_dir {config["base_dir"]} --prompt_path {custom_prompt_filepath} --code_repo_path {github_workspace_path} --inspect_agent_dir_path 'inspect_agent' --env_file {unique_env_filename} --max_timeout_in_seconds {config["max_duration_per_task_in_hours"] * 3600} 2>&1 | tee -a {inspectai_log_path}" # prompt_path is the path to the task prompt, code_repo_path is the path to the cloned repo, inspect_path is the path to the inspect_agent

    bench_logger.info("🤖️ Running command: " + command)
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    bench_logger.info("🤖️ Command output: " + result.stdout)

def _query_agent(config, task_prompt, task_counter, github_workspace_path):

    # Save full prompt to a temp file:
    temp_prompt_path = "logs/temp_prompt_eval_{}.txt".format(os.getpid())
    with open(temp_prompt_path, 'w') as f:
        f.write(task_prompt)
    # Get absolute path of the temp prompt file:
    temp_prompt_path = os.path.abspath(temp_prompt_path)
    
    output_path = get_relative_output_path(config, task_counter) # final output path from perspective of eval.py outer docker container, this will be in outputs/evaluation/<conference_name>/<paper_id>/<agent_name>/<llm_name>
    workspace_response_path = get_agent_relative_output_path(config, task_counter) # output path from perspective of agent inner docker container, this will be in /workspace
    workspace_local_response_path = get_agent_relative_local_output_path(config, task_counter, github_workspace_path) # output path from perspective of eval.py outer docker container, this will be in workspace/<repo_name>
    agent_log_path = get_relative_log_path(config, task_counter)
    os.makedirs(os.path.dirname(agent_log_path), exist_ok=True)
    os.makedirs(os.path.dirname(workspace_local_response_path), exist_ok=True)

    max_iterations = 1
    iteration = 0

    while True: # repeat until the expected format of JSON is returned
        print("Generating Agent output..")
        bench_logger.info("Generating Agent output..")

        if config["agent_name"] == "openhands":
            call_oh_with_prompt(task_prompt, temp_prompt_path, config, github_workspace_path, agent_log_path, max_duration_per_task_in_seconds=config["max_duration_per_task_in_hours"] * 3600, iterations=30)
        elif config["agent_name"] == "inspectai":
            _query_inspectai(config, task_prompt, task_counter, github_workspace_path, agent_log_path)
        else:
            raise ValueError(f"Agent {config['agent_name']} not supported")

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
            response = {"no_answer": "No answer found in eval gen after {} iterations.".format(max_iterations)}
            with open(output_path, 'w') as f:
                json.dump(response, f, indent=2)
            break
        else:
            print("Retrying... (iteration {})".format(iteration))
            bench_logger.info("Retrying... (iteration {})".format(iteration))

def query_agent(config: dict, task_prompt: str, repo_path: str, task_counter: int):
    start_time = time.time()
    if config["agent_name"] in ["openhands", "inspectai"]:
        _query_agent(config, task_prompt, task_counter, repo_path)
        # Calculate agent cost:
        agent_cost_path = get_agent_cost_relative_log_path(config, task_counter)
        agent_log_filepath = get_relative_log_path(config, task_counter)
        if config["agent_name"] == "openhands":
            filename_suffix = f"{str(config["paper_id"])}_task_index_{task_counter}_iter{config["iteration"]}_duration_{config['max_duration_per_task_in_hours']}_eval_gen"
            _collect_openhands_cost(f"task {task_counter}", filename_suffix, filename2=agent_cost_path, mode="eval_gen")
        elif config["agent_name"] == "inspectai":
            _collect_inspectai_cost(agent_log_filepath, agent_cost_path, mode="eval_gen")
    else:
        # Raise warning:
        print(f"Agent {config["agent_name"]} not supported")
        bench_logger.info(f"Agent {config["agent_name"]} not supported")
        raise ValueError(f"Agent {config["agent_name"]} not supported")
    end_time = time.time()
    timing_path = get_timing_relative_log_path(config, task_counter)
    with open(timing_path, 'a') as f:
        f.write(f"Task {task_counter} took {end_time - start_time} seconds to complete.\n")

def postprocessing(config: dict, repo_path: str, task_counter: int):
    print("Postprocessing...")
    bench_logger.info("Postprocessing...")
    # Obtain git diff patch:
    generate_patch(repo_path, config, task_counter)

    # Remove repo:
    # git_remove_repo(repo_path)

    # Save config to task specific output config path: (this will be used by exec verifier in the future. Caveat: judge now needs to be executed in the same machine as the eval gen)
    output_config_path = get_relative_output_config_path(config, task_counter)
    with open(output_config_path, 'w') as f:
        config["github_workspace_path"] = repo_path
        json.dump(config, f, indent=2)

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

    # First mask the README if it exists
    readme_path = repo_path + "/README.md"
    if os.path.isfile(readme_path):
        print("Masking README file:", readme_path)
        bench_logger.info("Masking README file: " + readme_path)
        # Remove README from git tracking
        run_command("git rm --cached README.md -f", cwd=repo_path)
        # Force remove the README from filesystem
        run_command("rm -f README.md", cwd=repo_path)

    # Then mask the task-specific files
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

    print("Cloning GitHub repo...")
    bench_logger.info("Generating task prompt...")
    repo_path = git_clone_repo(config["github_url"], suffix=f"_eval_task_index_{task_counter}_iter_{str(config["iteration"])}_duration_{config['max_duration_per_task_in_hours']}")

    print("Masking repo...")
    bench_logger.info("Masking repo...")
    is_mask_fail, failed_masked_sources = mask_repo(task_data, repo_path)

    print("Generating task prompt...")
    bench_logger.info("Generating task prompt...")
    task_prompt = generate_task_prompt(task_data, config, task_counter)
    print("Generate task prompt done.")
    bench_logger.info("Generate task prompt done.")

    # Save to output path now if masking failed for any sources:
    if is_mask_fail:
        output_path = get_relative_output_path(config, task_counter)
        with open(output_path, 'w') as f:
            json.dump({"invalid_task": "Masking failed for files: {}".format(failed_masked_sources)}, f, indent=2)
        print("WARNING: Skipping task. Masking failed for files: {}".format(failed_masked_sources))
        bench_logger.info("WARNING: Skipping task. Masking failed for files: {}".format(failed_masked_sources))

    return task_prompt, repo_path, is_mask_fail

def get_output_eval_gen_filename(config: dict, task_counter: int):
    return f"/{str(config["paper_id"])}_task_index_{task_counter}_iter_{config["iteration"]}_duration_{config["max_duration_per_task_in_hours"]}_eval_gen.json"

def get_agent_relative_output_path(config: dict, task_counter: int):
    """
        Get the final relative output path (for openhands and inspectai) for design and conclusion
    """
    return "/workspace" + get_output_eval_gen_filename(config, task_counter)

def get_relative_log_path(config: dict, task_counter: int):
    return config["output_log_folder"] + f"/{str(config["paper_id"])}_task_index_{task_counter}_iter_{config["iteration"]}_duration_{config["max_duration_per_task_in_hours"]}_eval_gen_{config["agent_name"]}_logs.txt"

def get_agent_cost_relative_log_path(config: dict, task_counter: int):
    return config["output_log_folder"] + f"/{config['agent_name']}_total_cost_{datetime.datetime.now().strftime('%Y-%m-%d')}_{str(config['paper_id'])}_task_index_{task_counter}_iter{config['iteration']}_duration_{config['max_duration_per_task_in_hours']}_eval_gen.txt" 

def get_agent_relative_local_output_path(config: dict, task_counter: int, repo_path: str):
    """
        Get the relative output path (local, i.e., exp-bench docker or actual local) version of get_agent_relative_output_path for design and conclusion

        Inputs:
            repo_path: the path to cloned repo such as: workspace/<repo_name> (note there is no slash at the end)
    """
    return repo_path + get_output_eval_gen_filename(config, task_counter)

def get_timing_relative_log_path(config: dict, task_counter: int):
    return config["output_log_folder"] + f"/{str(config["paper_id"])}_task_index_{task_counter}_iter_{config["iteration"]}_duration_{config["max_duration_per_task_in_hours"]}_eval_gen_timing_logs.txt"
    
def get_relative_output_path(config: dict, task_counter: int):
    """
        Get the final relative output path (local, i.e., exp-bench docker or actual local) for design and conclusion
    """
    return config["output_folder"] + get_output_eval_gen_filename(config, task_counter)

def get_relative_output_config_path(config: dict, task_counter: int):
    """
        Get the final relative output path (local, i.e., exp-bench docker or actual local) for config
    """
    return config["output_folder"] + f"/{str(config["paper_id"])}_task_index_{task_counter}_iter_{config["iteration"]}_duration_{config["max_duration_per_task_in_hours"]}_eval_gen_config.json"

def get_relative_output_patch_path(config: dict, task_counter: int):
    """
        Get the final relative output path (local, i.e., exp-bench docker or actual local) for git diff patch
    """
    return config["output_folder"] + f"/{str(config["paper_id"])}_task_index_{task_counter}_iter_{config["iteration"]}_duration_{config["max_duration_per_task_in_hours"]}_eval_gen.patch"

def process_task(args):
    """
        Inputs:
            task_counter: the index of the task in the list, just a counter for repo construction
            task: the actual task dict obtained from 1 task within a file such as: outputs/logs/neurips2024/93022/93022_complete_final.json
            config: the config dict containing all the parameters for the pipeline
    """
    task_counter, task_data, config = args

    # Check if this specific task+duration combination has been processed
    output_design_conclusion_path = get_relative_output_path(config, task_counter)
    if os.path.isfile(output_design_conclusion_path):
        print(f"Task {task_counter} for paper {str(config['paper_id'])} with duration {config['max_duration_per_task_in_hours']} already processed. Skipping...")
        bench_logger.info(f"Task {task_counter} for paper {str(config['paper_id'])} with duration {config['max_duration_per_task_in_hours']} already processed. Skipping...")
        return

    # Process inputs:
    # - Question prompt construction: hypothesis+method+agent_instruction
    # - Repo related: (1) clone repo, (2) remove scripts according to masked_source
    task_prompt, repo_path, is_mask_fail = process_inputs(config, task_data, task_counter)

    # Call LLM with appropriate prompts: use appropriate agent
    # - design+conclusion output file: if not saved in correct place repeat x runs
    if not is_mask_fail:
        query_agent(config, task_prompt, repo_path, task_counter)

    # Process outputs: 
    # - diff file: we process this next on our side.
    postprocessing(config, repo_path, task_counter)

def main(
        config: dict
    ):
    # for loop for parallel execution. check through paper folder to obtain tasks
    with open(config["input_paper_tasks_filename"], 'r') as f:
        paper_tasks_complete = json.load(f)
    tasks_list = []

    if "specific_tasks" in config:
        # For specific tasks, create a separate process for each task specification
        # even if they share the same task_index
        for task_spec in config["specific_tasks"]:
            paper_id, task_idx, duration = task_spec
            if task_idx < len(paper_tasks_complete["questions"]):
                task = paper_tasks_complete["questions"][task_idx]
                # Create a new config for this specific duration
                task_config = config.copy()
                task_config["max_duration_per_task_in_hours"] = duration
                tasks_list.append((task_idx, task, task_config))
            else:
                print(f"Warning: Task index {task_idx} is out of range for paper {config['paper_id']}")
                bench_logger.warning(f"Task index {task_idx} is out of range for paper {config['paper_id']}")
    else:
        # Original behavior for non-specific tasks
        for task_counter, task in enumerate(paper_tasks_complete["questions"]):
            tasks_list.append((task_counter, task, config))

    with ProcessPoolExecutor(max_workers=config["parallelization_factor"]) as executor:
        processed_tasks = list(executor.map(process_task, tasks_list))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run evaluation/judge with various agents/LLMs pipeline")
    parser.add_argument("--config-file", type=str, default="config.yaml", help="Path to the config file (e.g., config.yaml)")

    args = parser.parse_args()

    with open(args.config_file, 'r') as file:
        config = json.load(file)

    log_filename = config["log_filename"]
    log_path = Path(log_filename)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    setup_openhands_credential(config["llm_config_filename"])
    setup_gen_setup_logging(log_filename)
    setup_utils_logging(log_filename)

    main(
        config=config,
    )