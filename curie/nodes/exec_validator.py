from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage

import os
import subprocess
import filecmp
import json

import model
import utils
import tool

from logger import init_logger

def setup_exec_validator_logging(log_filename: str):
    global curie_logger 
    curie_logger = init_logger(log_filename)

def exec_validator(llm_verified_wrote_list):
    # This version is meant to be called directly as a function, not wrapped within langgraph abstractions. 

    curie_logger.info("------------Execution Verifier------------")

    for item in llm_verified_wrote_list:
        try:
            control_experiment_results_filename = item["control_experiment_results_filename"]
            control_experiment_filename = item["control_experiment_filename"]
            if "verifier_log_message" in item:
                verifier_log_message = item["verifier_log_message"]
            elif "patcher_log_message" in item:
                verifier_log_message = item["patcher_log_message"]
            else:
                assert False, "Error: verifier_log_message or patcher_log_message not found in item."

            result_file_contents = []

            with open(control_experiment_results_filename, "r") as file:
                file_content = file.read()  # Read the file content
                result_file_contents.append(file_content)  # Append file content to the string
                curie_logger.info(f"ExecVerifier: Successfully read content from pre-existing {control_experiment_results_filename}.")
            
            iterations = 1
            for i in range(iterations):

                curie_logger.info("Before iteration: {}".format(i))
                # utils.print_workspace_contents()

                # Run the first iteration and rename the file
                no_error, verifier_log_message, result_file_1_content = run_control_experiment_and_rename(1, control_experiment_filename, control_experiment_results_filename)

                if not no_error:
                    item["is_correct"] = False
                    item["verifier_log_message"] = "Failure encountered while repeating the control_experiment the 1st time:\n" + verifier_log_message
                    break 
                
                result_file_contents.append(result_file_1_content)

                curie_logger.info("After iteration: {}".format(i))
                # utils.print_workspace_contents()

            # # Compare the two result files
            # is_same_result = compare_results(result_file_1, result_file_2)

            # print("After comparison:")
            # # utils.print_workspace_contents()

            results_block = "\n\n".join(
                [f"Result {i + 1}:\n{content}" for i, content in enumerate(result_file_contents)]
            )

            verifier_log_message = f'''
Here are the results from {iterations+1} separate runs of this workflow:

{results_block}
'''

            item["verifier_log_message"] = verifier_log_message

        except Exception as e:
            curie_logger.error(f"ExecVerifier: Error: {e}")
            verifier_log_message = str(e)
            item["is_correct"] = False
            item["verifier_log_message"] = verifier_log_message

    return llm_verified_wrote_list

def run_control_experiment_and_rename(iteration, control_experiment_filename, control_experiment_results_filename, timeout=30):
    """
    Runs the control_experiment.sh script and renames the results file.
    """
    no_error = True
    # verifier_log_message = "Successfully ran the workflow and renamed the results file."
    # result_file_content = "Check results file content here."

    verifier_log_message = ""
    result_file_content = ""

    attempt = 0 # may retry since encountered edge case where file exists, but then does not exist later. suspect that there are some sync errors..?
    max_retries = 3

    while attempt < max_retries:
        attempt += 1
        curie_logger.info(f"ExecVerifier: Attempt {attempt} for iteration {iteration}...")
        try:
            # Run the control_experiment.sh script
            curie_logger.info(f"ExecVerifier: Running {control_experiment_filename}, iteration {iteration}...")
            # enter the conda env
            workspace_dir = os.path.dirname(control_experiment_filename) 
            
            command = f"""
            eval "$(micromamba shell hook --shell bash)" &&
            micromamba activate {workspace_dir}/venv && 
            bash {control_experiment_filename}
            """
            # TODO: instruction to install package in micromamba
            try:
                result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=timeout)
                curie_logger.info(f"ExecVerifier: {result.stdout}")
            except subprocess.TimeoutExpired:   
                curie_logger.info(f"ExecVerifier: {timeout}s timeout for long running script {control_experiment_filename}.")
                no_error = True
                verifier_log_message = f"No error found, but timeout for {control_experiment_filename}."
                return no_error, verifier_log_message, result_file_content
            
            if result.returncode != 0:
                curie_logger.info(f"ExecVerifier: Error running {control_experiment_filename}: {result.stderr}")
                no_error = False
                verifier_log_message = f"Error running {control_experiment_filename}: {result.stderr}"
                return no_error, verifier_log_message, result_file_content

            # Check if control_group_results.txt exists
            if not os.path.exists(control_experiment_results_filename):
                curie_logger.info(f"ExecVerifier: Error: {control_experiment_results_filename} was not generated.")
                no_error = False
                verifier_log_message = f"Error: {control_experiment_filename} executed successfully but {control_experiment_results_filename} was not generated."
                return no_error, verifier_log_message, result_file_content

            with open(control_experiment_results_filename, "r") as file:
                file_content = file.read()  # Read the file content
                result_file_content += file_content  # Append file content to the string
                curie_logger.info(f"ExecVerifier: Successfully read content from {control_experiment_results_filename}.")

            break
            
        except Exception as e:
            curie_logger.info(f"ExecVerifier: Error on attempt {attempt}: {e}")
            verifier_log_message = str(e)
            # If we've exhausted all retries, re-raise the last exception
            if attempt == max_retries:
                curie_logger.info(f"ExecVerifier: All {max_retries} attempts failed.")
                no_error = False

    return no_error, verifier_log_message, result_file_content

def compare_results(file1, file2):
    """
    Compares two result files and asserts they are identical. TODO: exact comparison is probably not the best way to go about this, need to account for acceptable range.
    """
    curie_logger.info(f"Comparing {file1} and {file2}...")
    if filecmp.cmp(file1, file2, shallow=False):
        curie_logger.info("ExecVerifier: The files are identical.")
        return True
    else:
        curie_logger.info("ExecVerifier: Error: The files are not identical.")
        return False