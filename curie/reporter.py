import os
import model
from langchain_core.messages import HumanMessage, SystemMessage 

# write a lab report
 
def filter_logging(log_data):
    filtered_strings = ["[", "INFO", "=======", "openhands", 'OBSERVATION', 'ACTION',
                        'root root']
    filtered_log_data = []
    for line in log_data:
        if all([string not in line for string in filtered_strings]) and line != "\n":
            filtered_log_data.append(line)
    # write_to_tmp_file = '/logs/tmp_log.txt'
    # with open(write_to_tmp_file, 'w') as file:
    #     for line in filtered_log_data:
    #         file.write(line)
    # print(f"Filtered log data saved to {write_to_tmp_file}")
    # import sys
    # sys.exit(0)

    return filtered_log_data


def generate_report(config, plans): 
    # print(plan)
    # read questions  
    results = []
    for plan in plans:
        print(f"Plan: {plan}")
        # list out all .txt files in the workspace directory
        results += ["Here is the experimental plan", f"{plan}\n",
                "Here are the actual results of the experiments: \n"]
        
        workspace_dir = plan["workspace_dir"] 
        if workspace_dir != '' and os.path.exists(workspace_dir):
            workspace_dir = plan["workspace_dir"]
            txt_files = [file for file in os.listdir(workspace_dir) if file.endswith('.txt')]

            for file in txt_files:
                with open(f"{workspace_dir}/{file}", 'r') as f:
                    results.append(f.read())
        
    log_file = '/' + config["log_filename"]
    with open(log_file, 'r') as file:
        log_data = file.readlines()
    filtered_log_data = filter_logging(log_data)
    filtered_log_data = "".join(filtered_log_data) 

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
    exp_log_dir = f"logs/{config['workspace_name'].lstrip('/').split('/')[-1]}_{config['unique_id']}_iter{config['iteration']}"
    with open(f"/{exp_log_dir}/{report_name}.md", "w") as file:
        file.write(response.content)
    print(f"Report saved to {exp_log_dir}/{report_name}.md")
    return f"{exp_log_dir}/{report_name}.md"
 