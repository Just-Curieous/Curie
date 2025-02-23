# source curie/setup/env.sh
# python -m benchmark.swe_bench.run_infer --config_path curie/configs/coding_config.json

import os
import re
import json
import logging
import argparse
from pathlib import Path
from datasets import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SWEBenchRunner:
    """Runner for evaluating models on SWE-Bench tasks."""
    
    def __init__(self, dataset_name="princeton-nlp/SWE-bench_Lite", split="test", config_path="curie/configs/swe_config.json"):
        """Initialize the runner with dataset and configuration."""
        self.dataset_name = dataset_name
        self.split = split
        self.config_path = config_path
        self.workspace_dir = Path("workspace")
        self.logs_dir = Path("logs/swe_results")
        self.starter_file_dir = Path("starter_file")
        
        # Ensure directories exist
        self.workspace_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True, parents=True)

    def generate_system_prompt(self, workspace_dir_name, problem_statement, hint_text):
        """Generate the system prompt for a given problem."""
        return f'''<uploaded_files>
        /{workspace_dir_name}
        </uploaded_files>

        I've uploaded a github directory under /{workspace_dir_name}. Consider the following issue description:

        <issue_description>
        {problem_statement}
        </issue_description> 
        
        Here are more discussions about the task:
        <hint>
        {hint_text}
        </hint>

        Make sure to formulate this as an experimental plan.
        '''

    def prepare_repository(self, repo_url, commit_hash):
        """Clone and prepare repository at specified commit."""
        repo_name = repo_url.split('/')[-1]
        workspace_dir_name = self.starter_file_dir / repo_name
        
        if not workspace_dir_name.exists():
            logger.info(f"Cloning repository {repo_url} to {workspace_dir_name}")
            os.system(f"git clone {repo_url} {workspace_dir_name}")
            
        logger.info(f"Resetting to commit {commit_hash}")
        os.system(f"cd {workspace_dir_name} && git reset --hard {commit_hash}")
        
        return workspace_dir_name, repo_name

    def update_curie_config(self, repo_name):
        """Update the Curie configuration for the current task."""
        with open(self.config_path, 'r') as f:
            curie_config = json.load(f)

        curie_config['workspace_name'] = repo_name
        
        with open(self.config_path, 'w') as f:
            json.dump(curie_config, f, indent=4)
            
        logger.info(f"Updated Curie config with workspace name: {repo_name}")

    def run_curie(self, instruction_file, logfile_path):
        """Execute Curie with the given instruction file."""
        try:
            command = (
                f"python3 -m curie.main --iterations 1 "
                f"--question_file {instruction_file} "
                f"--task_config {self.config_path} > {logfile_path}"
            )
            logger.info(f"Running Curie: {command}")
            os.system(command)
            return True
        except Exception as e:
            logger.error(f"Error running Curie: {e}")
            return False

    def extract_log_path(self, logfile_path):
        """Extract Curie log file path from execution log."""
        with open(logfile_path, 'r') as f:
            lines = f.readlines()
            for log_text in lines:
                if 'Check out the log file' in log_text:
                    match = re.search(r'logs/[\w\-.]+', log_text)
                    if match:
                        return match.group(0)
        return None

    def extract_starter_dir(self, curie_logfile):
        """Extract new starter file directory from Curie log."""
        with open(curie_logfile, 'r') as f:
            lines = f.readlines()
            for log_text in lines:
                if 'Starter files from' in log_text:
                    match = re.search(r'/([^ ]+)/', log_text)
                    if match:
                        return match.group(1)
        return None

    def generate_patch(self, new_starter_dir, repo_name, issue_number, instance_id):
        """Generate patch file comparing before and after changes."""
        # Set correct permissions on files
        patch_file = os.path.abspath(f"{self.logs_dir}/curie_{repo_name}_{issue_number}_{instance_id}.patch")
        new_starter_dir_abs = os.path.abspath(new_starter_dir)
        logger.info(f"<> Generating patch file: {patch_file}")
        logger.info(f"<> Workspace directory: {new_starter_dir_abs}")
        os.system(f"sudo find {new_starter_dir_abs} -type f -exec chmod 644 {{}} +")

        # Configure git for the directory
        original_dir = os.getcwd()
        os.chdir(new_starter_dir_abs)
        os.system(f"git config --global --add safe.directory {new_starter_dir_abs}")
        os.system("git config core.fileMode false")
        os.system(f"git diff | grep -v '^diff --git' | grep -v '^index' > {patch_file}")
        
        os.chdir(original_dir)
        return patch_file

    def save_results(self, instance_id, patch_content):
        """Save results to JSONL file."""
        results = {
            "instance_id": instance_id,
            "model_patch": patch_content,
            "model_name_or_path": f"curie_{os.environ.get('MODEL', 'default')}",
        }
        
        result_filename = instance_id.split('__')[0]
        result_path = f'{self.logs_dir}/{result_filename}.jsonl'
        with open(result_path, "a") as file:
            file.write(json.dumps(results) + "\n")
        logger.info(f'Results written to: {result_path}')

    def process_task(self, task_data):
        """Process a single SWE-Bench task."""
        # Extract task details
        repo_url = task_data['repo']
        if '/' not in repo_url:
            logger.warning(f"Invalid repo URL format: {repo_url}")
            return False
            
        repo_url = f"https://github.com/{repo_url}"
        commit_hash = task_data['base_commit']
        problem_statement = task_data['problem_statement']
        hint_text = task_data['hints_text']
        instance_id = task_data['instance_id']
        issue_number = instance_id.split('-')[-1]
        
        logger.info(f"Processing task: {instance_id}")
        logger.info(f"Repository: {repo_url} at commit {commit_hash}")
        
        # Prepare repository
        workspace_dir, repo_name = self.prepare_repository(repo_url, commit_hash)
        
        # Generate system prompt
        instructions = self.generate_system_prompt(workspace_dir, problem_statement, hint_text)
        instruction_file = self.workspace_dir / f"SWE-task-{repo_name}-{issue_number}.txt"
        logfile_path = self.workspace_dir / f"SWE-log-{repo_name}-{issue_number}.txt"
        
        with open(instruction_file, 'w') as f:
            f.write(instructions)
        
        # Update Curie config and run
        self.update_curie_config(repo_name)
        success = self.run_curie(instruction_file, logfile_path)
        if not success:
            return False
            
        # Process logs
        curie_logfile = self.extract_log_path(logfile_path)
        if not curie_logfile or not os.path.exists(curie_logfile):
            logger.error(f"Failed to find Curie log file path: {curie_logfile}")
            return False
            
        # Extract the new starter file directory
        new_starter_dir = self.extract_starter_dir(curie_logfile)
        if not new_starter_dir or not os.path.exists(new_starter_dir):
            logger.error(f"Failed to find new starter file directory: {new_starter_dir}")
            return False
        
        # Generate patch
        patch_file = self.generate_patch(new_starter_dir, repo_name, issue_number, instance_id)
        
        # Read the patch file
        with open(patch_file, 'r') as f:
            patch_content = f.read()
            
        # Save results
        self.save_results(instance_id, patch_content)
        return True

    def run(self, filter_repo=None):
        """Run the SWE-Bench evaluation."""
        dataset = load_dataset(self.dataset_name, split=self.split)
        
        for idx, row in enumerate(dataset):
            try: 
                repo_url = row['repo']
                
                # Apply filter if specified
                if filter_repo and repo_url != filter_repo:
                    # logger.info(f"Skipping repo: {repo_url} (filtered)")
                    continue 

                logger.info(f"Processing {idx+1}/{len(dataset)}: {repo_url}")
                success = self.process_task(row)
                
                if not success:
                    logger.warning(f"Failed to process task for {repo_url}")
                
                # For debugging, only process one task
                # break
            except Exception as e:
                logger.error(f"Error processing task: {e}")
                continue


if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--config_path", type=str, default="curie/configs/swe_config.json")
    args = parser.parse_args()

    runner = SWEBenchRunner(dataset_name=args.dataset_name, split=args.split, config_path=args.config_path)
    # runner.run(filter_repo="pytest-dev/pytest")
    # runner.run(filter_repo="scikit-learn/scikit-learn")
    runner.run(filter_repo="matplotlib/matplotlib")

    