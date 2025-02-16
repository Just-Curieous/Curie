# Currently consists of two components: a LLM-based verifier, and an execution verifier (program).
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage

import os
import subprocess
import filecmp
import sys

import model
import utils
import tool

from logger import init_logger
def setup_verifier_logging(log_filename: str):
    global curie_logger 
    curie_logger = init_logger(log_filename)

def create_LLMVerifierGraph(State, store, metadata_store):
    """ Creates a verifier graph consisting of the LLM-based verifier that checks through the experimental workflow to verify that actual data is produced. """
    system_prompt_file = "prompts/llm-verifier.txt"
    verifier_write_tool = tool.LLMVerifierWriteTool(store, metadata_store)
    store_get_tool = tool.StoreGetTool(store)
    tools = [tool.execute_shell_command, store_get_tool, verifier_write_tool]
    verifier_graph = create_VerifierGraph(State, store, metadata_store, system_prompt_file, tools, "llm_verifier")
    return verifier_graph

def create_PatchVerifierGraph(State, store, metadata_store, config_dict):
    """ Creates a patcher graph consisting of the LLM-based patcher/debugger that debugs the experimental workflow to ensure that actual data is produced. """
    system_prompt_file = "prompts/exp-patcher.txt"
    patcher_record_tool = tool.PatchVerifierWriteTool(store, metadata_store)
    patch_agent_tool = tool.PatcherAgentTool(config_dict)
    store_get_tool = tool.StoreGetTool(store)
    tools = [patch_agent_tool, tool.execute_shell_command, patcher_record_tool, store_get_tool] 
    verifier_graph = create_VerifierGraph(State, store, metadata_store, system_prompt_file, tools, "patch_verifier")
    return verifier_graph

def create_AnalyzerGraph(State, store, metadata_store):
    """ Creates a analyzer graph consisting of the LLM-based analyzer that analyses results from the experimental workflow."""
    system_prompt_file = "prompts/exp-analyzer.txt"
    patcher_record_tool = tool.AnalyzerWriteTool(store, metadata_store)
    store_get_tool = tool.StoreGetTool(store)
    tools = [tool.read_file_contents, patcher_record_tool, store_get_tool] # Only tool is code execution for now
    verifier_graph = create_VerifierGraph(State, store, metadata_store, system_prompt_file, tools, "analyzer")
    return verifier_graph

def create_ConcluderGraph(State, store, metadata_store):
    """ Creates a concluder graph consisting of the LLM-based concluder that is only activated when all results have been produced."""
    system_prompt_file = "prompts/exp-concluder.txt"
    patcher_record_tool = tool.ConcluderWriteTool(store, metadata_store)
    store_get_tool = tool.StoreGetTool(store)
    tools = [tool.read_file_contents, patcher_record_tool, store_get_tool] # Only tool is code execution for now
    verifier_graph = create_VerifierGraph(State, store, metadata_store, system_prompt_file, tools, "concluder")
    return verifier_graph

def create_VerifierGraph(State, store, metadata_store, system_prompt_file, tools, node_name):
    """ Creates a verifier graph consisting of the LLM-based verifier that checks through the experimental workflow to verify that actual data is produced. """
    # TODO: only creating one worker now. Not sure if we will create multiple workers.
    # NOTE: since there is no parallelization now we assume that there is only one control_experiment.sh that we need to look into. We can hardcode this for now. 
    verifier_builder = StateGraph(State)
    verifier_node = create_Verifier(tools, system_prompt_file, State, node_name) 

    verifier_builder.add_node(node_name, verifier_node)
    verifier_builder.add_edge(START, node_name)
    tool_node = ToolNode(tools=tools)
    verifier_builder.add_node("tools", tool_node)

    verifier_builder.add_conditional_edges(
        node_name,
        tools_condition,
    )
    verifier_builder.add_edge("tools", node_name)
    
    verifier_graph = verifier_builder.compile()
    utils.save_langgraph_graph(verifier_graph, f"../logs/misc/{node_name}_graph_image.png")

    def call_verifier_graph(state: State) -> State:
        response = verifier_graph.invoke({"messages": state["messages"][-1]}, {"recursion_limit": 20})
        return {
            "messages": [
                HumanMessage(content=response["messages"][-1].content, name=f"{node_name}_graph")
            ],
            "prev_agent": response["prev_agent"],
        }
    return call_verifier_graph

def create_Verifier(tools, system_prompt_file, State, node_name):    
    def Verifier(state: State):
        # Read from prompt file:
        with open(system_prompt_file, "r") as file:
            system_prompt = file.read()

        # system_prompt = """
        # You are an agent that will use the test_search_tool tool provided when answering questions about UMich.  
        # """
        system_message = SystemMessage(
            content=system_prompt,
        )

        # Query model and append response to chat history 
        messages = state["messages"]

        # Ensure the system prompt is included at the start of the conversation
        if not any(isinstance(msg, SystemMessage) for msg in messages):
            messages.insert(0, system_message)
        
        response = model.query_model_safe(messages, tools)
        curie_logger.info(f"<> FROM {node_name}:")
        curie_logger.info(utils.parse_langchain_llm_output(response))
        curie_logger.info("----------------- END Verifier ------------------")
        return {"messages": [response], "prev_agent": node_name}
    
    return Verifier

def exec_verifier(llm_verified_wrote_list):
    # This version is meant to be called directly as a function, not wrapped within langgraph abstractions. 

    curie_logger.info("------------Entering Exec Verifier function!!!------------")

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

    curie_logger.info("------------ Exiting Exec Verifier ------------")

    return llm_verified_wrote_list

def run_control_experiment_and_rename(iteration, control_experiment_filename, control_experiment_results_filename, timeout=1200):
    """
    Runs the control_experiment.sh script and renames the results file.
    """
    no_error = True
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
            result = subprocess.run(["bash", control_experiment_filename], capture_output=True, text=True, timeout=timeout)

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