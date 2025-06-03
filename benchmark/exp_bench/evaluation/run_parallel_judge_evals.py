#!/usr/bin/env python3

import json
import os
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple
import tempfile

def create_config(
    llm_config_filename: str,
    conference_params: Tuple[str, str],  # (tasks_folder, paper_details)
    agent_name: str,
    base_config_path: str = "evaluation/configs/parallel_eval_judge_config_template.json",
    do_exec_check: bool = False,
    max_duration_per_task_in_hours: float = 1,
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
    config["do_exec_check"] = do_exec_check
    config["max_duration_per_task_in_hours"] = int(max_duration_per_task_in_hours) if float(max_duration_per_task_in_hours).is_integer() else float(max_duration_per_task_in_hours)
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

def main():
    do_exec_check = True

    # Example parameter lists - modify these according to your needs
    max_durations = [0.25, 0.5, 1, 2, 4, 8]  # List of durations to evaluate

    llm_configs = [
        # "evaluation/setup/env-amazon-nova-pro.sh",
        # "evaluation/setup/env-openhands-o3-mini.sh",
        # "evaluation/setup/env-deepseek-r1.sh",
        "evaluation/setup/env-claude-haiku-35.sh",
        # "evaluation/setup/env-claude-sonnet-37.sh",
        # Add more LLM configs as needed
    ]
    
    # Each tuple contains (tasks_folder, paper_details)
    conference_params = [
        ("outputs/logs/iclr2024/", "logs/iclr2024/iclr2024_withcode_popularity_stars-100.json"),
        ("outputs/logs/neurips2024/", "logs/neurips2024/neurips_abs_2024_withcode_popularity_stars-100.json"),
        # Add more conference parameter pairs as needed
    ]
    
    agent_names = [
        # "openhands",
        "inspectai",
        # Add more agent names as needed
    ]
    
    # Create a temporary directory for config files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Run evaluations for each combination of parameters
        for llm_config in llm_configs:
            for conf_params in conference_params:
                for agent_name in agent_names:
                    for duration in max_durations:
                        print(f"\nRunning evaluation with:")
                        print(f"LLM Config: {llm_config}")
                        print(f"Tasks Folder: {conf_params[0]}")
                        print(f"Paper Details: {conf_params[1]}")
                        print(f"Agent Name: {agent_name}")
                        print(f"Max Duration (hours): {duration}")
                        
                        config = create_config(
                            llm_config_filename=llm_config,
                            conference_params=conf_params,
                            agent_name=agent_name,
                            do_exec_check=do_exec_check,
                            max_duration_per_task_in_hours=duration
                        )
                        
                        try:
                            run_evaluation(config, temp_dir)
                        except subprocess.CalledProcessError as e:
                            print(f"Error running evaluation: {e}")
                            continue

if __name__ == "__main__":
    main() 