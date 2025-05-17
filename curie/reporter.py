import os
import model
from langchain_core.messages import HumanMessage, SystemMessage 

# write a lab report
 
def filter_logging(log_data):
    filtered_strings = ["[", "INFO", "=======", "openhands", 'OBSERVATION', 'ACTION',
                        'root root']
    filtered_log_data = []
    pre_line = ""
    for line in log_data:
        if all([string not in line for string in filtered_strings]) and line != "\n":
            if pre_line != line:
                filtered_log_data.append(line)
        pre_line = line
    
    filtered_log = "".join(filtered_log_data)

    return filtered_log

def summarize_logging(content):
    summarize_content = []
    # summzrize the log content every 50k characters
    for i in range(0, len(content), 50000):

        chunk = content[i:i + 50500]
        print(f'Summarizing log chunk ({i} - {i + 50000}) / {len(content)}: ')
        messages = [SystemMessage(content="Summarize one chunk of the experimentation log:"),
                   HumanMessage(content=chunk)]
        response = model.query_model_safe(messages)
        # print(response.content)
        summarize_content.append(response.content)
    return summarize_content

def generate_report(config, plans): 
    # read questions  
    log_file = '/' + config["log_filename"]
    model.setup_model_logging(log_file) 
    all_logging = [f"Here is the research questions: \n {plans[0]['question']}"]
    results = []
    raw_results = []
    for plan in plans:
        # list out all .txt files in the workspace directory
        plan_results = ["Here is the experimental plan", f"{plan}\n",
                "Here are the actual results of the experiments: \n"]

        workspace_dir = plan["workspace_dir"] 
        # control_group = plan["control_group"]
        # experimental_group = plan["experimental_group"]
        banned_keywords = ['Warning:']
        if workspace_dir != '' and os.path.exists(workspace_dir):
            workspace_dir = plan["workspace_dir"]
            txt_files = [file for file in os.listdir(workspace_dir) if file.endswith('.txt')]

            for file in txt_files:
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
        messages = [SystemMessage(content="Extract the raw results with the corresponding experiment setup. \
                                    No need to analyze the results. \
                                    Ignore the experiment process. NEVER fake the results. \
                                    Here is the raw results: \n" ),
                    HumanMessage(content=plan_results)]
        response = model.query_model_safe(messages)
        results.append(response.content)
        raw_results.append(plan_results)
    
    # summarize the results
    all_results = "\n".join(results)
    # messages = [SystemMessage(content="Extract the results  all experimentation plan along with the results in a structured way.\n" +
    #                           "Ignore the experiment process. NEVER fake the results."
    #                           "Here is the raw results: \n" ),
    #             HumanMessage(content=results)]
    
    # response = model.query_model_safe(messages)
    # all_results = response.content

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
    
    # print(f"Aggregated results saved to {results_file_name}") 
    all_logging += ["Here are the summarized results of the experiments: \n"]
    all_logging.append(all_results) 

    with open(log_file, 'r') as file:
        log_data = file.readlines()
    filtered_log_data = filter_logging(log_data)
    filtered_log_data = "".join(filtered_log_data) 

    summarize_log_content = summarize_logging(filtered_log_data)
    summarize_log_content = "\n".join(summarize_log_content) 

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
 