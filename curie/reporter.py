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

    for plan in plans:
        # list out all .txt files in the workspace directory
        results += ["Here is the experimental plan", f"{plan}\n",
                "Here are the actual results of the experiments: \n"]

        workspace_dir = plan["workspace_dir"] 
        # control_group = plan["control_group"]
        # experimental_group = plan["experimental_group"]
        
        if workspace_dir != '' and os.path.exists(workspace_dir):
            workspace_dir = plan["workspace_dir"]
            txt_files = [file for file in os.listdir(workspace_dir) if file.endswith('.txt')]

            for file in txt_files:
                with open(f"{workspace_dir}/{file}", 'r') as f:
                    results.append(f.read())
    # summarize the results
    all_results = "\n".join(results)
    messages = [SystemMessage(content="Summarize all experimentation results in a structured way:"),
                HumanMessage(content=all_results)]
    response = model.query_model_safe(messages)
    all_results = response.content

    results_file_name = log_file.replace(".log", "_all_results.txt")
    with open(f'{results_file_name}', 'w') as file:
        file.write(all_results)
    
    # print(f"Results saved to {results_file_name}")
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
    print(f"Report saved to {report_name}")
    return report_name, results_file_name
 