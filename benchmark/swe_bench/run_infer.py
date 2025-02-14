import os
import pandas as pd
import toml
from datasets import load_dataset
import logging
import subprocess

logger = logging.getLogger(__name__)

# python -m benchmark.swe_bench.run_infer

def system_prompt(workspace_dir_name, problem_statement):
    """
    Function to get the system prompt
    """
    
    instructions = f'''<uploaded_files>
        /workspace/{workspace_dir_name}
        </uploaded_files>

        I've uploaded a Python code repository in the directory {workspace_dir_name}. Consider the following issue description:

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

        Your thinking should be thorough and detailed.
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
    repo_url = f"https://github.com/{repo_url}"
    commit_hash = row['base_commit']
    problem_statement = row['problem_statement']
    instance_id = row['instance_id']
    issue_number = instance_id.split('-')[-1]
    repo_name = repo_url.split('/')[-1]

    
    # print all the details
    print(f"<> Repo URL: {repo_url}")
    print(f"<> Commit Hash: {commit_hash}")
    print(f"<> Problem Statement: {problem_statement}")
    print(f"<> Issue Number: {issue_number}")
    print(f"<> Repo Name: {repo_name}")
    # git clone to workspace/
    workspace_dir_name = f"starter_file/{repo_name}"
    if not os.path.exists(workspace_dir_name):
        os.system(f"git clone {repo_url} {workspace_dir_name}")
    # reset to commit 'commit_hash'
    os.system(f"cd {workspace_dir_name} && git reset --hard {commit_hash}")

    # system prompt
    instructions = system_prompt(f'/{workspace_dir_name}', problem_statement)
    # write instructions to a file
    file_name = f"{workspace_dir_name}/SWE-{repo_name}-{issue_number}.txt"
    with open(file_name, 'w') as f:
        f.write(instructions)
    
    # # run curie: python3 -m curie.main --iterations 1 --question_file workspace/...txt --task_config curie/configs/base_config.json
    os.system(f"python3 -m curie.main --iterations 1 --question_file {file_name} --task_config curie/configs/base_config.json")    
    
    

    break
    

'''
Each prediction must be formatted as follows:

{
    "instance_id": "<Unique task instance ID>",
    "model_patch": "<.patch file content string>",
    "model_name_or_path": "<Model name here (i.e. SWE-Llama-13b)>",
}
'''