#!/usr/bin/env python3

import json
import os
import subprocess
from pathlib import Path

def main():
    # Directly configure judge parameters
    config = {
        "input_conference_tasks_folder": "outputs/logs/neurips2024/",
        "input_conference_paper_details_filename": "logs/neurips2024/neurips_abs_2024_withcode_popularity_stars-100.json",
        "parallelization_factor": 1,
        "iterations": 1,
        "llm_config_filename": "evaluation/setup/env_llm_config.sh",
        "agent_name": "curie",  # Use curie to match generation results
        "max_duration_per_task_in_hours": 0.67,
        "max_papers": 1,
        "mode": "judge",
        "do_exec_check": True,
        "llm_judge_config_filename": "evaluation/setup/env_llm_config.sh",
        "judge_agent_name": "",
        "docker_image": "exp-bench-image",
        "dockerfile_name": "ExpDockerfile_default",
        "eval_gen_prompt": "evaluation/prompts/eval_gen_prompt.txt",
        "eval_judge_prompt": "evaluation/prompts/eval_judge_prompt.txt",
        "eval_judge_setup_prompt": "evaluation/prompts/eval_judge_setup_prompt.txt",
        "eval_judge_setup_partial_prompt": "evaluation/prompts/eval_judge_setup_partial_prompt.txt",
        "eval_judge_setup_monitor_prompt": "evaluation/prompts/eval_judge_setup_monitor_prompt.txt",
        "paper_id": 93022,
        "task_index": 1,  # Corrected to 1
        "iteration": 1,
        "llm_name": "azure/gpt-4o",
        "output_folder": "outputs/evaluation/neurips2024/93022/curie/azure/gpt-4o",
        "input_paper_tasks_filename": "outputs/logs/neurips2024/93022/93022_complete_final.json",
        "log_judge_filename": "outputs/evaluation/logs/neurips2024/93022/curie/azure/gpt-4o/93022_judge_log.txt"
    }
    
    # Create temporary config file
    config_path = "judge_config_93022.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created judge config: {config_path}")
    print("Running judge evaluation...")
    
    # Run simplified judge
    try:
        # Use new simplified judge script
        cmd = [
            "python", "evaluation/judge_direct.py", 
            "--config-file", config_path
        ]
        subprocess.run(cmd, check=True)
        print("Judge evaluation completed successfully!")
        
        # Check output file
        output_file = "outputs/evaluation/neurips2024/93022/curie/azure/gpt-4o/93022_eval_judge.json"
        if os.path.exists(output_file):
            print(f"Judge results saved to: {output_file}")
            # Display result content
            with open(output_file, 'r') as f:
                result = json.load(f)
            print("\nJudge Results:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"Warning: Expected output file not found: {output_file}")
            
    except subprocess.CalledProcessError as e:
        print(f"Error running judge: {e}")
    finally:
        # Clean up temporary files
        if os.path.exists(config_path):
            os.remove(config_path)

if __name__ == "__main__":
    main() 