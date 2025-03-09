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
    question_file = '/' + config["question_filename"] 
    with open(question_file, 'r') as file:
        questions = file.readlines() 

    log_file = '/' + config["log_filename"]
    with open(log_file, 'r') as file:
        log_data = file.readlines()
    filtered_log_data = filter_logging(log_data)
    model.setup_model_logging(log_file)

    # concat questions and log data
    report_data = "".join(questions) + "\n" + "".join(filtered_log_data)

    with open("/curie/prompts/exp-reporter.txt", "r") as file:
        system_prompt = file.read() 
    # from langchain_core.messages import BaseMessage
    messages = [SystemMessage(content=system_prompt),
               HumanMessage(content=report_data)]

    response = model.query_model_safe(messages)

    report_name = log_file.split("/")[-1].split(".")[0]
    with open(f"/logs/{report_name}.md", "w") as file:
        file.write(response.content)
    print(f"Report saved to logs/{report_name}.md")
 