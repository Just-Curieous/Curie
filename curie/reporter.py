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


def generate_report(config): 
    # read questions  
    log_file = '/' + config["log_filename"]
    with open(log_file, 'r') as file:
        log_data = file.readlines()
    filtered_log_data = filter_logging(log_data)
    filtered_log_data = "".join(filtered_log_data)
    model.setup_model_logging(log_file)

    with open("/curie/prompts/exp-reporter.txt", "r") as file:
        system_prompt = file.read() 
    # from langchain_core.messages import BaseMessage
    messages = [SystemMessage(content=system_prompt),
               HumanMessage(content=filtered_log_data)]

    response = model.query_model_safe(messages)

    report_name = log_file.split("/")[-1].split(".")[0]
    with open(f"/logs/{report_name}.md", "w") as file:
        file.write(response.content)
    print(f"Report saved to logs/{report_name}.md")
    return f'logs/{report_name}.md'
 