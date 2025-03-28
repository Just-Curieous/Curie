import os
import model
from langchain_core.messages import HumanMessage, SystemMessage 

# write a lab report
 
def filter_logging(log_data):
    filtered_strings = ["[", "INFO", "=======", "openhands", 'OBSERVATION', 'ACTION']
    filtered_log_data = []
    for line in log_data:
        if all([string not in line for string in filtered_strings]) and line != "\n":
            filtered_log_data.append(line)
    return filtered_log_data


def generate_report(config, plan): 
    print(plan)
    # read questions  
    workspace_dir = plan["workspace_dir"]
    log_file = '/' + config["log_filename"]
    with open(log_file, 'r') as file:
        log_data = file.readlines()
    filtered_log_data = filter_logging(log_data)
    filtered_log_data = "".join(filtered_log_data)



    # list out all .txt files in the workspace directory
    results = ["Here are the experimental plan", f"{plan}\n",
            "Here are the actual results of the experiments: \n"]
    txt_files = [file for file in os.listdir(workspace_dir) if file.endswith('.txt')]

    for file in txt_files:
        with open(f"{workspace_dir}/{file}", 'r') as f:
            results.append(f.read())
    
    # append the filtered_log_data to the results
    results.append("Here are the logs from the experiment: \n")
    results.append(filtered_log_data)

    results = "\n".join(results)
    model.setup_model_logging(log_file)

    with open("/curie/prompts/exp-reporter.txt", "r") as file:
        system_prompt = file.read() 

    messages = [SystemMessage(content=system_prompt),
               HumanMessage(content=results)]

    response = model.query_model_safe(messages)

    report_name = log_file.split("/")[-1].split(".")[0]
    with open(f"/logs/{report_name}.md", "w") as file:
        file.write(response.content)
    print(f"Report saved to logs/{report_name}.md")
    return f'logs/{report_name}.md'
 