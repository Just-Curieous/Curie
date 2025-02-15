import os
import pandas as pd
import toml
from datasets import load_dataset
import logging
import subprocess
import re
import json

logger = logging.getLogger(__name__)

# python -m benchmark.swe_bench.run_infer

def system_prompt(workspace_dir_name, problem_statement):
    """
    Function to get the system prompt
    """
    
    instructions = f'''<uploaded_files>
        /{workspace_dir_name}
        </uploaded_files>

        I've uploaded a Python code repository in the directory /{workspace_dir_name}. Consider the following issue description:

        <issue_description>
        {problem_statement}
        </issue_description>

        Can you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?
        I've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!
        Your task is to make the minimal changes to non-test files in the /workspace directory to ensure the requirements are satisfied.

        Follow these steps to resolve the issue:
        1. First, explore the repo to familiarize yourself with its structure
        2. Create a script to reproduce the error and execute it with `python <filename.py>` using the BashTool to confirm the error
        3. Edit the source code of the repo to resolve the issue
        4. Rerun your reproduce script and confirm that the error is fixed
        5. Think about edge cases and make sure your fix handles them as well

        Make sure to formulate this as an experimental plan.
    '''
    return instructions

dataset = "princeton-nlp/SWE-bench_Lite"
split = "test"

dataset = load_dataset(dataset, split=split)

# swe_bench_tests = filter_dataset(dataset.to_pandas(), 'instance_id')
# print(swe_bench_tests.head())
# iterate over the dataset
for row in dataset:

    repo_url = row['repo']
    if repo_url != 'scikit-learn/scikit-learn':
        print(f"Skipping repo: {repo_url}")
        continue
    repo_url = f"https://github.com/{repo_url}"
    commit_hash = row['base_commit']
    problem_statement = row['problem_statement']
    instance_id = row['instance_id']
    issue_number = instance_id.split('-')[-1]
    repo_name = repo_url.split('/')[-1]

    
    # print all the details
    print(f"<> Repo URL: {repo_url}")
    print(f"<> Commit Hash: {commit_hash}")
    # print(f"<> Problem Statement: {problem_statement}")
    print(f"<> Issue Number: {issue_number}")
    print(f"<> Repo Name: {repo_name}")
    # git clone to workspace/
    workspace_dir_name = f"starter_file/{repo_name}"
    if not os.path.exists(workspace_dir_name):
        os.system(f"git clone {repo_url} {workspace_dir_name}")
    # # reset to commit 'commit_hash'
    # os.system(f"cd {workspace_dir_name} && git reset --hard {commit_hash}")

    # # system prompt
    # instructions = system_prompt(f'{workspace_dir_name}', problem_statement)
    # # write instructions to a file
    # file_name = f"workspace/SWE-task-{repo_name}-{issue_number}.txt"
    # with open(file_name, 'w') as f:
    #     f.write(instructions)
    
    # logfile_name = f"workspace/SWE-config-{repo_name}-{issue_number}.txt"
    # # load the config file curie/configs/swe_config.json 
    # with open("curie/configs/swe_config.json", 'r') as f:
    #     curie_config = json.load(f)
    # curie_config['workspace_name'] = repo_name
    # # dump the config file
    # with open("curie/configs/swe_config.json", 'w') as f:
    #     json.dump(curie_config, f, indent=4)
 
    # os.system(f"python3 -m curie.main --iterations 1 --question_file {file_name} --task_config curie/configs/swe_config.json > {logfile_name}")    
    
    # # # read the log file
    # # logfile_name='starter_file/astropy/SWE-config-astropy-12907.txt'
    # curie_logfile = None
    # with open(logfile_name, 'r') as f:
    #     lines = f.readlines()
    #     for log_text in lines:
    #         if 'Check log file' in log_text:
    #             match = re.search(r'logs/[\w\-.]+', log_text)
    #             if match:
    #                 curie_logfile = match.group(0)  # Extract the log file name
    #                 print(f'<> curie_logfile: {curie_logfile}')
    #                 break
    
    # if curie_logfile is None:
    #     print("Error: Could not find the log file")
    #     break
    # if not os.path.exists(logfile_name):
    #     print("Error: Log file does not exist")
    #     break    

    curie_logfile = 'logs/SWE-task-scikit-learn-10297_20250214223259_iter1.log'
    new_starter_file_dir = None
    with open(curie_logfile, 'r') as f:
        lines = f.readlines()
        for log_text in lines:
            
            if 'Starter files from' in log_text:
                match = re.search(r'/([^ ]+)/', log_text)
                if match:
                    new_starter_file_dir = match.group(1)  # Extracting without the first slash
                    print(f'<> new_starter_file_dir: {new_starter_file_dir}')
                    break
    if new_starter_file_dir is None or not os.path.exists(new_starter_file_dir):
        print("Error: Could not find the new starter file directory")
        break
    # git diff to get the .patch file
    patch_file = f"logs/swe_results/curie_{repo_name}_{issue_number}_{instance_id}.patch"
    command = f"git diff --no-index {new_starter_file_dir} {workspace_dir_name} > {patch_file}"
    os.system(command)
    print(f'<> Patch command: {command}')
    print(f'<> Patch file: {patch_file}')
    # Read the patch file
    with open(patch_file, 'r') as f:
        patch_content = f.read()

    results = {
        "instance_id": f'{instance_id}',
        "model_patch": f'{patch_content}',
        "model_name_or_path": f"Curie_{os.environ.get('MODEL')}",
    }

    result_path = f'logs/swe_results/{instance_id}.json'
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "a") as file:
        file.write(json.dumps(results) + "\n")
    print(f'<> Results written to: {result_path}')
    break

