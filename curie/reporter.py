import os
import model
from langchain_core.messages import HumanMessage, SystemMessage 
import multiprocessing as mp
from multiprocessing import Process, Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple 
import string

# write a lab report
 
def filter_logging(log_data): 
    # return filtered_log
    stay_strings = ['- logger -']
    filtered_log_data = []
    for line in log_data:
        if any(string in line for string in stay_strings):
            filtered_log_data.append(line)
    print(f"before filtering: {len(log_data)}")
    print(f"after filtering: {len(filtered_log_data)}")
    return filtered_log_data

def safe_join(data_list, separator=''):
    return separator.join(item or "" for item in data_list) 

def summarize_logging_parallel_threads(log_file):
    """
    Parallelized version using ThreadPoolExecutor.
    Best for I/O-bound tasks like API calls.
    """
    with open(log_file, 'r') as file:
        log_data = file.readlines()
    
    filtered_log_data = filter_logging(log_data)
    content = safe_join(filtered_log_data)
    
    # Create chunks with their indices to maintain order
    chunks = []
    for i in range(0, len(content), 50000):
        chunk = content[i:i + 50500]
        chunks.append((i, chunk))
    
    if len(chunks) > 10:
        chunks = chunks[:3] + chunks[-7:]
    
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
    
    with ThreadPoolExecutor(max_workers=2) as executor:
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
    summarize_content = safe_join([s for s in summaries if s is not None])
    print(f"👧 Completed summarizing log file: {log_file}")
    return summarize_content


def process_plan(plan_data):
    """Process a single plan and return the results"""
    i, plan = plan_data
    
    try:
        print(f"👦 Analyzing plan {i+1}.")
        
        # list out all .txt files in the workspace directory
        plan_results = [f"Here is the experimental plan: {plan}\n",
                "Here are the actual results of the experiments: \n"]

        workspace_dir = plan["workspace_dir"]  
        banned_keywords = ['Warning:', '\x00']


        if workspace_dir != '' and os.path.exists(workspace_dir):
            # TODO: need to retrive all the results more smartly later. 
            log_files = []
            workspace_dir_list = [plan["workspace_dir"], os.path.join(plan["workspace_dir"], "results")]

            for workspace_dir in workspace_dir_list:
                if os.path.exists(workspace_dir):
                    log_files += [os.path.join(workspace_dir, file) for file in os.listdir(workspace_dir) if file.endswith('.log')]
                    log_files += [os.path.join(workspace_dir, file) for file in os.listdir(workspace_dir) if file.endswith('.txt')]
                    log_files += [os.path.join(workspace_dir, file) for file in os.listdir(workspace_dir) if file.endswith('.json')]
            
            print(f"📃📃📃 Found log files {log_files}")
            for file in log_files:
                with open(file, 'r') as f:
                    # remove duplicate lines in f.read() 
                    lines = f.readlines()
                    for line in lines: 
                        
                        if not any(keyword in line for keyword in banned_keywords):
                            plan_results.append(''.join(c for c in line if c in string.printable))

        plan_results_str = safe_join(plan_results)
        messages = [SystemMessage(content="Understand the experiment plan and \
                                    extract the complete raw results with the experiment setup. \
                                    No need to analyze the results. \
                                    Ignore the intermediate steps. NEVER fake the results."),
                    HumanMessage(content=plan_results_str)]
        
        response = model.query_model_safe(messages)
        return (i, response.content, plan_results_str, None)  # (index, result, raw_result, error)
        
    except Exception as e: 
        return (i, None, None, e)
    
def extract_raw_results(log_file, plans):

    results = [None] * len(plans)  # Pre-allocate to maintain order
    raw_results = [None] * len(plans)

    fig_names = []
    caption_list = []
    log_dir = os.path.dirname(log_file)
    plan_data = [(i, plan) for i, plan in enumerate(plans)]

    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all tasks
        future_to_index = {executor.submit(process_plan, data): data[0] for data in plan_data}
        
        # Collect results as they complete
        for future in as_completed(future_to_index):
            index, result, raw_result, error = future.result()
            
            if error is None:
                results[index] = result
                raw_results[index] = raw_result
                 
    # summarize the results
    all_results = safe_join(results) 
    num_tries = 0
    while num_tries < 3:
        try:

            fig_name_list, caption_list = plot_results(all_results, log_dir, True)
            if fig_name_list is not None:
                fig_names += fig_name_list
            break
        except Exception as e:
            print(f"Error in plotting: {e}")
            num_tries += 1
            continue

    results_file_name = log_file.replace(".log", "_all_results.txt")
    print(f'Output results to {results_file_name}')
    with open(f'{results_file_name}', 'w') as file:
        file.write("\033[1;36m╔══════════════════════════╗\033[0m\n")  # Cyan bold
        file.write("\033[1;33m║     Summarized Results   ║\033[0m\n")  # Yellow bold
        file.write("\033[1;36m╚══════════════════════════╝\033[0m\n")  # Cyan bold
        file.write(all_results)
    
        file.write("\n\033[1;36m╔══════════════════════╗\033[0m\n")  # Cyan bold
        file.write("\033[1;33m║     Raw Results      ║\033[0m\n")  # Yellow bold
        file.write("\033[1;36m╚══════════════════════╝\033[0m\n")  # Cyan bold
        raw_results = safe_join(raw_results, '\n')
        file.write(raw_results)

    return all_results, fig_names, caption_list, results_file_name

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
    # Load the plotting template
    tmpl_path = os.path.join('prompts', 'exp-plotting.txt')
    with open(tmpl_path, 'r') as f:
        plotting_templates = f.read()
    if is_aggregated:
        prompt = f"""
        {plotting_templates}
        Write python code (Don't use pandas or seaborn.) to visualize the results in the best format and save the generated figures to {directory}.
        You will be given the results of multiple experiment plans; select useful information to visualize.
        Use the visualization to show the comparison of different experiment plans.
        Use the above templates to style the plots (Times New Roman font, bold axes labels, clear legends, grid, etc.).
        Choose appropriate plot types based on the results.
        

        The python code must:
        1. Save all figures into {directory},
        2. Append each saved figure's path to a list named `fig_name_list`,
        3. Generate a professional caption(i.e. Fig 1: Description) for each figure, append it to a list named `caption_list`, the caption should be concise and informative, describing the key insights from the figure,
        4. Return both `fig_name_list` and `caption_list`.
        """
    else:
        prompt = f"""
        {plotting_templates}
        If the results are worth visualizing, write python code (Don't use pandas or seaborn) to visualize the results in the best format and save it to {directory}.
        If the results are not worth visualizing, set `fig_name_list = None` and `caption_list = None`, and return them.
        Use the above templates to style the plots (Times New Roman font, bold axes labels, clear legends, grid, etc.).
        Choose appropriate plot types based on the results.
       

        The python code must:
        1. Save the figure(s) into {directory},
        2. Append each saved figure's path to a list named `fig_name_list`,
        3. Generate a professional caption(i.e. Fig 1: Description) for each figure, append it to a list named `caption_list`, the caption should be concise and informative, describing the key insights from the figure,
        4. Return both `fig_name_list` and `caption_list`.
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
    fig_name_list = namespace.get('fig_name_list')
    # get the name of the caption from the python process
    caption_list = namespace.get('caption_list')
    print(f"Figure name list: {fig_name_list}")
    if fig_name_list is None:
        return None, None
    return [os.path.basename(fig) for fig in fig_name_list], caption_list

 
def generate_report(config, plans): 
    # read questions  
    log_file = '/' + config["log_filename"]
    model.setup_model_logging(log_file) 
    all_logging = [f"Here is the research questions: \n {plans[0]['question']}"]

    # Run sequentially instead of in parallel
    all_results, fig_names, caption_list, results_file_name = extract_raw_results(log_file, plans)
    summarize_log_content = summarize_logging_parallel_threads(log_file)

    all_logging.append(f"Here is the visualization of the aggregated results: {fig_names}") 

    # if captions were generated, append them to the logging
    if caption_list is not None and len(caption_list) > 0:
        all_logging.append(f"Here are the captions for the figures: \n{caption_list}")
    else:
        all_logging.append("No captions were generated for the figures.")

    all_logging += ["Here are the summarized results of the experiments: \n"]
    all_logging.append(all_results)  

    # append the filtered_log_data to the results
    all_logging.append("Here are the summarized logs from the experiment: \n")
    all_logging.append(summarize_log_content)

    all_logging = safe_join(all_logging, '\n')

    with open("/curie/prompts/exp-reporter.txt", "r") as file:
        system_prompt = file.read() 

    messages = [SystemMessage(content=system_prompt),
                HumanMessage(content=all_logging)]

    response = model.query_model_safe(messages)
    report_name = log_file.replace('.log', '.md') 

    with open(report_name, "w") as file:
        file.write(response.content)
    print(f"Report saved to {report_name}")
    return report_name, results_file_name