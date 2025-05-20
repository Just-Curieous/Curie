import os
import model
from langchain_core.messages import HumanMessage, SystemMessage 
import multiprocessing as mp
from multiprocessing import Process, Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple 

# write a lab report
 
def filter_logging(log_data):
    # filtered_strings = ["=======", "openhands", 'OBSERVATION', 'ACTION',
    #                     'root root', '- lib/python3.12/site-packages']
    # filtered_log_data = []
    # pre_line = ""
    # for line in log_data:
    #     if all([string not in line for string in filtered_strings]) and line != "\n":
    #         if pre_line != line:
    #             filtered_log_data.append(line)
    #     pre_line = line
    
    # print(f"before filtering: {len(log_data)}")
    # print(f"after filtering: {len(filtered_log_data)}")
    # filtered_log = "".join(filtered_log_data)

    # return filtered_log
    stay_strings = ['- logger -']
    filtered_log_data = []
    for line in log_data:
        if any(string in line for string in stay_strings):
            filtered_log_data.append(line)
    print(f"before filtering: {len(log_data)}")
    print(f"after filtering: {len(filtered_log_data)}")
    return filtered_log_data

def summarize_logging_parallel_threads(log_file):
    """
    Parallelized version using ThreadPoolExecutor.
    Best for I/O-bound tasks like API calls.
    """
    with open(log_file, 'r') as file:
        log_data = file.readlines()
    
    filtered_log_data = filter_logging(log_data)
    content = "".join(filtered_log_data)
    
    # Create chunks with their indices to maintain order
    chunks = []
    for i in range(0, len(content), 50000):
        chunk = content[i:i + 50500]
        chunks.append((i, chunk))
    
    def process_chunk(chunk_data: Tuple[int, str]) -> Tuple[int, str]:
        """Process a single chunk and return (index, summary)"""
        i, chunk = chunk_data
        # print(f'👧 Summarizing log chunk ({i} - {i + 50000}) / {len(content)}: ')
        
        messages = [
            SystemMessage(content="Summarize one chunk of the experimentation log. Here is the log chunk: "),
            HumanMessage(content=chunk)
        ]
        response = model.query_model_safe(messages)
        return (i, response.content)
    
    # Process chunks in parallel
    summaries = [None] * len(chunks)  # Pre-allocate list to maintain order
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all tasks
        future_to_index = {executor.submit(process_chunk, chunk_data): idx 
                          for idx, chunk_data in enumerate(chunks)}
        
        # Collect results as they complete
        for future in as_completed(future_to_index):
            chunk_idx = future_to_index[future]
            try:
                original_index, summary = future.result()
                summaries[chunk_idx] = summary
                print(f"👧 Completed summarizing chunk {chunk_idx + 1}/{len(chunks)}")
            except Exception as exc:
                print(f"❌ Chunk {chunk_idx} generated an exception: {exc}")
                summaries[chunk_idx] = f"Error processing chunk: {exc}"
    
    # Join all summaries, filtering out None values
    summarize_content = "\n".join([s for s in summaries if s is not None])
    print(f"👧 Completed summarizing log file: {log_file}")
    return summarize_content


def extract_raw_results(log_file, plans):
    results = []
    raw_results = []
    fig_names = []
    log_dir = os.path.dirname(log_file)
    for i, plan in enumerate(plans):
        try:
            print(f"👦 Analyzing plan {i+1}/{len(plans)}.")
            # list out all .txt files in the workspace directory
            plan_results = ["Here is the experimental plan", f"{plan}\n",
                    "Here are the actual results of the experiments: \n"]

            workspace_dir = plan["workspace_dir"] 
            # control_group = plan["control_group"]
            # experimental_group = plan["experimental_group"]
            banned_keywords = ['Warning:']
            if workspace_dir != '' and os.path.exists(workspace_dir):
                workspace_dir = plan["workspace_dir"]
                log_files = [file for file in os.listdir(workspace_dir) if file.endswith('.txt')]
                log_files += [file for file in os.listdir(workspace_dir) if file.endswith('.log')]
                for file in log_files:
                    with open(f"{workspace_dir}/{file}", 'r') as f:
                        # remove duplicate lines in f.read() 
                        lines = f.readlines()
                        for line in lines:
                            if not any(keyword in line for keyword in banned_keywords):
                                plan_results.append(line)
                # this is for stock prediction. 
                # FIXME: need to retrive all the results more smartly later.
                results_dir = f"{workspace_dir}/results"
                if os.path.exists(results_dir):
                    for file in os.listdir(results_dir):
                        if file.endswith('.json'):
                            with open(f"{results_dir}/{file}", 'r') as f:
                                plan_results.append(f"Results from file: {file} \n")
                                plan_results.append(f.read())
            plan_results = "\n".join(plan_results)
            messages = [SystemMessage(content="Understand the experiment plan and \
                                        extract the complete raw results with the experiment setup. \
                                        No need to analyze the results. \
                                        Ignore the intermediate steps. NEVER fake the results."),
                        HumanMessage(content=plan_results)]
            response = model.query_model_safe(messages)
            results.append(response.content)
            # fig_name = plot_results(response.content, log_dir)
            # if fig_name is not None:
            #     fig_names.append(fig_name)
            #     results.append(f"Here is the figure of the results above: {fig_name}")
            raw_results.append(plan_results)
        except Exception as e:
            print(f"Error in extraction: {e}")
            continue
    
    # summarize the results
    all_results = "\n".join(results) 
    num_tries = 0
    while num_tries < 3:
        try:
            fig_name_list = plot_results(all_results, log_dir, True)
            if fig_name_list is not None:
                fig_names += fig_name_list
            break
        except Exception as e:
            print(f"Error in plotting: {e}")
            num_tries += 1
            continue
    # if fig_names is not None:
    #     all_results.append(f"Here is the aggregated figure of the results: {fig_names}")

    results_file_name = log_file.replace(".log", "_all_results.txt")
    with open(f'{results_file_name}', 'w') as file:
        file.write("\033[1;36m╔══════════════════════════╗\033[0m\n")  # Cyan bold
        file.write("\033[1;33m║     Summarized Results   ║\033[0m\n")  # Yellow bold
        file.write("\033[1;36m╚══════════════════════════╝\033[0m\n")  # Cyan bold
        file.write(all_results)
    
        file.write("\n\033[1;36m╔══════════════════════╗\033[0m\n")  # Cyan bold
        file.write("\033[1;33m║     Raw Results      ║\033[0m\n")  # Yellow bold
        file.write("\033[1;36m╚══════════════════════╝\033[0m\n")  # Cyan bold
        raw_results = "\n".join(raw_results)
        file.write(raw_results)

    return all_results, fig_names, results_file_name

import signal
from contextlib import contextmanager

class TimeoutError(Exception):
    pass

@contextmanager
def timeout(seconds):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Timed out after {seconds} seconds")
    
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Restore the old signal handler and disable the alarm
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)
 
def plot_results(summarized_results, directory, is_aggregated=False):
    if is_aggregated:
        prompt = f"""
        Write python code to visualize the results in the best format and save the generated figures to {directory}.
        You will be given the results of multiple experiment plans, select useful information to visualize.
        Use the visualization to show the comparison of different experiment plans.
        Don't use pandas or seaborn.
        The figure name list is stored in the variable 'fig_name_list'.
        The python code should return the name of the figure. 
        """
    else:
        prompt = f"""
        If the results worth visualizing, write python code to visualize the results in the best format and save it to {directory}.
        If the results are not worth visualizing, meaning the generated figure won't not usefult to the report, the python code should set fig_name=None and return None.
        Don't use pandas or seaborn.
        The figure name list is stored in the variable 'fig_name_list'.
        The python code should return the name of the figure. 
        """
    messages = [SystemMessage(content=prompt),
                HumanMessage(content=summarized_results)]
    response = model.query_model_safe(messages)
    # extract the python code from the response
    namespace = {}

    python_code = response.content.split("```python")[1].split("```")[0]
    try:
        with timeout(10):
            exec(python_code, namespace)
    except TimeoutError as e:
        print(f"Timeout / Error in plotting: {e}")
        return None
    
    # get the name of the figure from the python process
    fig_name_list = namespace['fig_name_list']
    print(f"Figure name list: {fig_name_list}")
    if fig_name_list is None:
        return None
    return [os.path.basename(fig) for fig in fig_name_list]

def generate_report(config, plans): 
    # read questions  
    log_file = '/' + config["log_filename"]
    model.setup_model_logging(log_file) 
    all_logging = [f"Here is the research questions: \n {plans[0]['question']}"]

    # Create queues to get results from parallel processes
    results_queue = Queue()
    log_summary_queue = Queue()
    
    # Start processes in parallel
    extract_process = Process(target=parallel_extract_raw_results, 
                             args=(log_file, plans, results_queue))
    summary_process = Process(target=parallel_summarize_logging,
                             args=(log_file, log_summary_queue))
    
    extract_process.start()
    summary_process.start()
    
    # Wait for processes to complete and get results
    # extract_process.join()
    # summary_process.join()
    try:
        extract_process.join(timeout=600)  # 5 minutes timeout
        summary_process.join(timeout=600)
        
        if extract_process.is_alive():
            print("Extract process timed out, terminating...")
            extract_process.terminate()
            extract_process.join(timeout=5)
            if extract_process.is_alive():
                extract_process.kill()
        
        if summary_process.is_alive():
            print("Summary process timed out, terminating...")
            summary_process.terminate()
            summary_process.join(timeout=5)
            if summary_process.is_alive():
                summary_process.kill()
            
    except Exception as e:
        print(f"Error waiting for processes: {e}")

    print(f"Joining processes")
    all_results, fig_names, results_file_name = results_queue.get()
    summarize_log_content = log_summary_queue.get()

    all_logging.append(f"Here is the visualization of the aggregated results: {fig_names}") 
    all_logging += ["Here are the summarized results of the experiments: \n"]
    all_logging.append(all_results)  

    # append the filtered_log_data to the results
    all_logging.append("Here are the summarized logs from the experiment: \n")
    all_logging.append(summarize_log_content)

    all_logging = "\n".join(all_logging)

    with open("/curie/prompts/exp-reporter.txt", "r") as file:
        system_prompt = file.read() 

    messages = [SystemMessage(content=system_prompt),
               HumanMessage(content=all_logging)]

    response = model.query_model_safe(messages)
    report_name = log_file.replace('.log', '.md') 

    with open(report_name, "w") as file:
        file.write(response.content)
    # print(f"Report saved to {report_name}")
    return report_name, results_file_name

# Parallel versions of functions to use with multiprocessing
def parallel_extract_raw_results(log_file, plans, queue):
    result = extract_raw_results(log_file, plans)
    queue.put(result)
    
def parallel_summarize_logging(log_file, queue):
    result = summarize_logging_parallel_threads(log_file)
    queue.put(result)

# Add multiprocessing guard to properly support multiprocessing 
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
 