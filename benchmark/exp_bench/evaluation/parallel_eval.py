"""
Details about this program:

Processes at the conference level, sending individual papers to main_eval.py for execution.

✅ Reading a JSONL file (one record per line). Example: logs/iclr2024/iclr2024_withcode_popularity_stars-100.json

✅ Multiprocessing with ProcessPoolExecutor,

✅ tqdm progress bar,

✅ Skipping records that already have a completed output file (based on the qid),

✅ Limiting to the first N tasks using max_papers.

Logic:
1. Read JSONL line by line, using parallel processing. Only read N lines indicated by max_papers.

2. Check if setup gen is done? Check if ID has been fully processed (i.e., outputs/logs/iclr2024/{task_id}_complete_final.json exists and is non-empty).
- TODO: later check if depth calc is done. 
- if yes, skip all

3. Check if hypo gen is done? Check if corresponding input hypo gen folder or file already exists by ID. 
- if yes, skip to setup gen
- if no, generate config for hypo gen, then run hypo gen. 

4. Run setup gen. 
- generate config for setup gen, then run setup gen.
"""
import json
import copy
import traceback
import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
from datetime import datetime 
import os
import re
import sys
from utils import get_relative_output_path_eval
from helper.utils import print_exception_and_traceback, setup_utils_logging
from helper.logger import init_logger
def setup_parallel_logging(log_filename: str):
    global bench_logger 
    bench_logger = init_logger(log_filename)

def parse_args():
    parser = argparse.ArgumentParser(description="Process input arguments for the script.")
    
    parser.add_argument(
        "--task_config",
        type=str, 
        default="evaluation/configs/parallel_eval_gen_config_template.json",
        help="Experimental Task configuration file."
    )

    return parser.parse_args()

def extract_model_identifier(env_file_path):
    with open(env_file_path, "r") as f:
        for line in f:
            match = re.match(r'export\s+MODEL="([^"]+)"', line)
            if match:
                raw_model = match.group(1)
                cleaned_model = (
                    raw_model.replace("/", "-")
                             .replace(":", "-")
                             .replace(".", "-")
                )
                return cleaned_model
    return None

def convert_to_paper_dir_path(task_config: dict, paper_id, keep_logs_prefix=False):
    """
    Converts:
      input: "outputs/logs/neurips2024/", 96264
    Into:
      - if keep_logs_prefix=False:
          "outputs/evaluation/neurips2024/96264/<agent_name>/<llm_name>"
      - if keep_logs_prefix=True: # this is used for logging
          "outputs/evaluation/logs/neurips2024/96264/<agent_name>/<llm_name>"
    """
    base = Path(task_config["input_conference_tasks_folder"]).resolve()

    if "logs" not in base.parts:
        raise ValueError("Base directory must contain 'logs'")

    parts = list(base.parts)
    logs_index = parts.index("logs")

    if keep_logs_prefix:
        # Insert 'evaluation' before 'logs'
        parts.insert(logs_index, "evaluation")
    else:
        # Replace 'logs' with 'evaluation'
        parts[logs_index] = "evaluation"

    model_name = extract_model_identifier(task_config["llm_config_filename"])
    if model_name is None:
        raise ValueError(f"Model name could not be extracted from: {task_config['llm_config_filename']}")

    abs_path = Path(*parts) / str(paper_id) / task_config["agent_name"] / model_name

    # Make it relative to current working directory
    try:
        relative_path = abs_path.relative_to(Path.cwd())
    except ValueError:
        # If not under cwd, fallback to relative to project root (optional: adjust root as needed)
        project_root = Path(__file__).resolve().parent.parent
        relative_path = abs_path.relative_to(project_root)

    return str(relative_path)

def get_task_count(task_config: dict):
    """
    Get the task count from the task_config; this refers to a specific paper.
    This is the same as used in eval.py: should convert this to a common utils in future.
    """
    with open(task_config["input_paper_tasks_filename"], 'r') as f:
        paper_tasks_complete = json.load(f)
    tasks_list = []
    for task_counter, task in enumerate(paper_tasks_complete["questions"]):
        tasks_list.append((task_counter, task, task_config))
    return len(tasks_list)

def check_if_eval_is_done(task_config: dict, mode: str):
    """
        If mode == "generate": only return true if all tasks are done.
        If mode == "judge": only return true if all tasks are done and all iterations are done.
    """
    task_counts = get_task_count(task_config)
    iterations = task_config["iterations"] # number of iterations required
    paper_id = task_config["paper_id"]

    # Check if all iterations for all tasks for the required paper for the requested agent+llm combo are done
    for task_counter in range(task_counts):
        for iteration in range(1, iterations + 1):
            output_file = get_relative_output_path_eval(task_config, task_counter, iteration, mode)
            print(f"Checking if {output_file} exists")
            if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                print(f"❌ {output_file} does not exist or is empty")
                return False
            if mode == "judge":
                # Open file and check if "execution_success" exists:
                with open(output_file, 'r') as f:
                    data = json.load(f)
                    if "execution_success" not in data:
                        return False
    return True

def is_no_eval_gen_is_done(task_config: dict):
    """
    Checks if there exist at least one eval gen file that is done.
    if none found, return True. 
    """
    task_counts = get_task_count(task_config)
    for task_counter in range(task_counts):
        for iteration in range(1, task_config["iterations"] + 1):
            output_file = get_relative_output_path_eval(task_config, task_counter, iteration, "generate")
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                return False
    return True


def run_eval(config_filename):
    # Step 1: Generate hypothesis
    cmd1 = [
        "python3", "evaluation/main_eval.py",
        f"--task_config={config_filename}",
    ]
    subprocess.run(cmd1, check=True)

def run_pipeline(config_filename: str, record: dict):
    try:
        print(f"Processing record: {record.get("paper_id", "unknown")}")
        bench_logger.info(f"Processing record: {record.get("paper_id", "unknown")}")
        # Check if setup gen is done: complete_all exists:
        if check_if_eval_is_done(record, mode="generate"):
            if record["mode"] == "generate": # if we are trying to do eval gen, we skip if eval gen is done
                print(f"✅ Skipping record {record.get("paper_id", "unknown")} as eval {record["mode"]} is already done.")
                bench_logger.info(f"✅ Skipping record {record.get("paper_id", "unknown")} as eval {record["mode"]} is already done.")
                return
        else:
            if record["mode"] == "judge" and is_no_eval_gen_is_done(record) : # if we are trying to do eval judge, we skip if no eval gen at all is done. if at least one eval gen is done, we do not skip.
                print(f"✅ Skipping record {record.get("paper_id", "unknown")} as eval gen is NOT done yet.")
                bench_logger.info(f"✅ Skipping record {record.get("paper_id", "unknown")} as eval gen is NOT done yet.")
                return
        if check_if_eval_is_done(record, mode="judge"):
            if record["mode"] == "judge": # if we are trying to do eval judge, we skip if eval judge is done
                print(f"✅ Skipping record {record.get("paper_id", "unknown")} as eval {record["mode"]} is already done.")
                bench_logger.info(f"✅ Skipping record {record.get("paper_id", "unknown")} as eval {record["mode"]} is already done.")
                return
        
        run_eval(config_filename)

        return {"paper_id": record.get("paper_id", "unknown"), "status": "success"}

    except subprocess.CalledProcessError as e:
        print_exception_and_traceback(e, prefix="❌ Error in subprocess")
        return {"paper_id": record.get("paper_id", "unknown"), "status": "failed", "error": f"Subprocess error: {e}"}
    except Exception as e:
        print_exception_and_traceback(e, prefix="❌ Error processing record")
        return {"paper_id": record.get("paper_id", "unknown"), "status": "failed", "error": str(e)}

def get_paper_details(paper_id: int, input_conference_paper_details_filename: str, task_config: dict):
    """
    Get paper details from the JSONL file.
    """
    with open(input_conference_paper_details_filename, 'r') as f:
        for line in f:
            record = json.loads(line)
            if record["id"] == str(paper_id):
                pdf_url = record["pdf_url"]
                github_url = record["code_url"] if record["code_url"] else record["reproduce_eval"]["code"]
                task_config["pdf_url"] = pdf_url
                task_config["github_url"] = github_url
                return task_config
    raise ValueError(f"Paper ID {paper_id} not found in {input_conference_paper_details_filename}")

def create_main_task_config(task_config: dict, paper_id: int, paper_path: str):
    """
    Creates a main task config for the parallel processing.
    """
    copied_task_config = copy.deepcopy(task_config)
    copied_task_config["paper_id"] = paper_id
    copied_task_config["input_paper_tasks_folder"] = paper_path
    copied_task_config["input_paper_tasks_filename"] = copied_task_config["input_paper_tasks_folder"] + "/" + str(copied_task_config["paper_id"]) + "_complete_final.json"
    copied_task_config["output_folder"] = convert_to_paper_dir_path(task_config, paper_id) # "output_folder": "output/evaluation/neurips2024/96264/<agent_name>/<llm_name>"
    copied_task_config["output_log_folder"] = convert_to_paper_dir_path(task_config, paper_id, keep_logs_prefix=True) # "output_log_folder": "output/evaluation/logs/neurips2024/96264/<agent_name>/<llm_name>"
    os.makedirs(copied_task_config["output_folder"], exist_ok=True)
    os.makedirs(copied_task_config["output_log_folder"], exist_ok=True)
    copied_task_config["output_eval_gen_filename"] = copied_task_config["output_folder"] + f"/{str(paper_id)}_eval_gen.json" # won't be used
    copied_task_config["output_eval_judge_filename"] = copied_task_config["output_folder"] + f"/{str(paper_id)}_eval_judge.json" # won't be used
    copied_task_config = get_paper_details(paper_id, copied_task_config["input_conference_paper_details_filename"], copied_task_config)
    config_filename = f"outputs/evaluation/configs/main_{str(paper_id)}_{task_config["mode"]}_config.json"
    os.makedirs(os.path.dirname(config_filename), exist_ok=True)
    with open(config_filename, 'w') as f:
        json.dump(copied_task_config, f, indent=2)
    return config_filename, copied_task_config

def process_tasks_file(task_config: dict):
    input_conference_tasks_folder = task_config["input_conference_tasks_folder"]
    max_papers = task_config["max_papers"] 
    configs = []

    # Enumerate through the dirs within input_conference_tasks_folder: 
    parent_dir = Path(input_conference_tasks_folder)
    for subdir in parent_dir.iterdir():
        if subdir.is_dir() and subdir.name.isdigit():
            try:
                paper_path = str(subdir.resolve().relative_to(Path.cwd()))
                print(f"Processing paper path: {paper_path}")
                paper_id = int(subdir.name)

                # if paper_id != 17595:
                #     continue

                # Skip if paper_id is not in specific_tasks (if specific_tasks is provided)
                if "specific_tasks" in task_config:
                    paper_ids = [task[0] for task in task_config["specific_tasks"]]
                    if str(paper_id) not in paper_ids:
                        print(f"Skipping paper {paper_id} as it's not in specific_tasks")
                        continue

                main_task_config_filename, main_task_config = create_main_task_config(task_config, paper_id, paper_path)
                
                # Pass through specific tasks with their durations if present
                if "specific_tasks" in task_config:
                    main_task_config["specific_tasks"] = [
                        task for task in task_config["specific_tasks"]
                        if task[0] == str(paper_id)
                    ]

                if not main_task_config["pdf_url"] or not main_task_config["github_url"]:
                    print(f"❌ Skipping record {paper_id} due to missing pdf_url or code_url.")
                    bench_logger.info(f"❌ Skipping record {paper_id} due to missing pdf_url or code_url.")
                    continue
                if not os.path.exists(main_task_config["input_paper_tasks_filename"]):
                    print(f"❌ Skipping record {paper_id} due to missing input_paper_tasks_filename.")
                    bench_logger.info(f"❌ Skipping record {paper_id} due to missing input_paper_tasks_filename.")
                    continue
                configs.append((main_task_config_filename, main_task_config))
                if max_papers is not None and len(configs) >= max_papers:
                    break
            except Exception as e:
                print_exception_and_traceback(e, prefix=f"❌ Error processing paper dir {subdir}")
            
    print(f"⚙️  Starting processing for {len(configs)} tasks...")
    bench_logger.info(f"⚙️  Starting processing for {len(configs)} tasks...")

    for record in tqdm(configs, desc="Processing tasks"):
        run_pipeline(record[0], record[1])

# 🧪 Example usage
def main():
    args = parse_args()
    config_file = args.task_config
    try:
        with open(config_file, 'r') as f:
            task_config = json.load(f)
    except Exception as e:
        print(f"Error reading config file: {e}")
        return

    print(f"EXP-Bench eval generation parallel pipeline is running with the following configuration: {task_config}")

    unique_id = datetime.now().strftime("%Y%m%d%H%M%S")
    log_filename = f"outputs/evaluation/logs/parallel_pipeline_runner/{unique_id}_{task_config["mode"]}.log"
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    setup_parallel_logging(log_filename)
    setup_utils_logging(log_filename)
    process_tasks_file(task_config)

if __name__ == "__main__":
    main()