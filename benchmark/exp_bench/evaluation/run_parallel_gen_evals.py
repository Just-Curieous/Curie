#!/usr/bin/env python3

import json
import os
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union
import tempfile

def create_config(
    llm_config_filename: str,
    conference_params: Tuple[str, str],  # (tasks_folder, paper_details)
    agent_name: str,
    base_config_path: str = "evaluation/configs/parallel_eval_gen_config_template_inspect_agent.json",
    max_papers: int = 50,
    parallelization_factor: int = 8,
    max_duration_per_task_in_hours: float = 0.5,
) -> Dict[str, Any]:
    """Create a configuration dictionary based on the template and provided parameters."""
    with open(base_config_path, 'r') as f:
        config = json.load(f)
    
    tasks_folder, paper_details = conference_params
    
    # Update the specified parameters
    config["llm_config_filename"] = llm_config_filename
    config["input_conference_tasks_folder"] = tasks_folder
    config["input_conference_paper_details_filename"] = paper_details
    config["agent_name"] = agent_name
    config["max_papers"] = max_papers
    config["parallelization_factor"] = parallelization_factor
    config["max_duration_per_task_in_hours"] = max_duration_per_task_in_hours
    config["mode"] = "generate"
    
    return config

def run_evaluation(config: Dict[str, Any], temp_dir: str) -> None:
    """Run a single evaluation with the given configuration."""
    # Create a temporary config file
    config_path = os.path.join(temp_dir, "temp_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Run the parallel evaluation script
    cmd = ["python", "evaluation/parallel_eval.py", f"--task_config={config_path}"]
    subprocess.run(cmd, check=True)

def main(
    max_duration_per_task_in_hours: float = 0.5,
    specific_tasks: List[Tuple[str, str, int, float]] = None  # List of (conference_name, paper_id, task_index, duration_hours)
):
    # Example parameter lists - modify these according to your needs
    llm_configs = [
        # "evaluation/setup/env-amazon-nova-pro.sh",
        # "evaluation/setup/env-openhands-o3-mini.sh",
        # "evaluation/setup/env-deepseek-r1.sh",
        "evaluation/setup/env-claude-haiku-35.sh",
        # "evaluation/setup/env-claude-sonnet-37.sh",
        # Add more LLM configs as needed
    ]
    
    # Each tuple contains (tasks_folder, paper_details)
    all_conference_params = [
        ("outputs/logs/iclr2024/", "logs/iclr2024/iclr2024_withcode_popularity_stars-100.json"),
        ("outputs/logs/neurips2024/", "logs/neurips2024/neurips_abs_2024_withcode_popularity_stars-100.json"),
        # Add more conference parameter pairs as needed
    ]
    
    agent_names = [
        # "openhands",
        "inspectai",
        # Add more agent names as needed
    ]

    # Filter conference_params based on specific_tasks if provided
    if specific_tasks:
        # Extract unique conference names from specific tasks
        target_conferences = set(task[0] for task in specific_tasks)
        # Filter conference_params to only include those in target_conferences
        conference_params = [
            params for params in all_conference_params 
            if any(conf in params[0] for conf in target_conferences)
        ]
    else:
        conference_params = all_conference_params

    # Create a temporary directory for config files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Run evaluations for each combination of parameters
        for llm_config in llm_configs:
            for conf_params in conference_params:
                for agent_name in agent_names:
                    print(f"\nRunning evaluation with:")
                    print(f"LLM Config: {llm_config}")
                    print(f"Tasks Folder: {conf_params[0]}")
                    print(f"Paper Details: {conf_params[1]}")
                    print(f"Agent Name: {agent_name}")
                    if not specific_tasks:
                        print(f"Max Duration per Task: {max_duration_per_task_in_hours} hours")
                    else:
                        print("Using per-task durations from specific tasks")
                        print(f"Specific Tasks: {specific_tasks}")
                    
                    config = create_config(
                        llm_config_filename=llm_config,
                        conference_params=conf_params,
                        agent_name=agent_name,
                        max_duration_per_task_in_hours=max_duration_per_task_in_hours
                    )

                    # Add specific tasks to config if provided
                    if specific_tasks:
                        # Filter tasks for current conference
                        conf_name = next(conf for conf in target_conferences if conf in conf_params[0])
                        current_conf_tasks = [
                            (paper_id, task_idx, duration) 
                            for (task_conf, paper_id, task_idx, duration) in specific_tasks 
                            if task_conf == conf_name
                        ]
                        if current_conf_tasks:
                            config["specific_tasks"] = current_conf_tasks
                    
                    try:
                        run_evaluation(config, temp_dir)
                    except subprocess.CalledProcessError as e:
                        print(f"Error running evaluation: {e}")
                        continue

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run parallel generation evaluations")
    parser.add_argument("--max_duration", type=float, default=0.5,
                      help="Maximum duration per task in hours (used when specific_tasks is not provided)")
    parser.add_argument("--specific_tasks", type=str, default=None,
                      help="JSON string of list of tuples: [(conference_name, paper_id, task_index, duration_hours), ...]")
    
    args = parser.parse_args()
    
    # Parse specific tasks if provided
    specific_tasks = None
    if args.specific_tasks:
        specific_tasks = json.loads(args.specific_tasks)
        # Validate format
        if not all(len(task) == 4 for task in specific_tasks):
            raise ValueError("Each task must be a tuple of (conference_name, paper_id, task_index, duration_hours)")
    
    main(
        max_duration_per_task_in_hours=args.max_duration,
        specific_tasks=specific_tasks
    )
