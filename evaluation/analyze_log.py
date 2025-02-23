# source curie/setup/env.sh
# python -m evaluation.analyze_log --log_file logs/SWE-task-pytest-9359_20250223174300_iter1.log



from openai import AzureOpenAI
import re
import os
from collections import defaultdict
from typing import Dict, List, Tuple
import argparse
 
SYS_PTOMPT = '''
Please analyze the provided log, which documents how different agents collaborate to address specific github issue. Understand the overall workflow, agent interactions and the key challenges faced by the agents during this process. Additionally, identify the main factors that contribute to repeated iterations in solving the problems. 

Your summary should provide:
0. The github issue to address.
1. all traces about '<<<<<<<< Scheduling ... >>>>>>>>' in the log. 
2. The detailed description of the challenges of the task that the agents faced.
3. total API cost in dollars, end-to-end time spent in minutes.
'''
def analyze_log(log: str) -> str:
    """Compile explanations for analysis.""" 
    # Prepare message for GPT analysis

    messages = [
        {"role": "system", "content": SYS_PTOMPT},
        {"role": "user", "content": log}
    ]
    
    client = AzureOpenAI(
        api_key=os.getenv('AZURE_API_KEY'),
        api_version="2024-06-01",
        azure_endpoint=os.getenv('AZURE_API_BASE'),
        organization=os.getenv('ORGANIZATION')
    )
     
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        stop=None
    )

    return response.choices[0].message.content

def read_logging(log_file: str) -> str:
    with open(log_file, "r") as f:
        return f.read()

def filter_logs(log_content: str) -> str:
    # Filter logs
    cleaned_data = re.sub(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - ', '', log_content, flags=re.MULTILINE)

    return cleaned_data

def main():
    parser = argparse.ArgumentParser(description="Summarize log files.")
    parser.add_argument("--log_file", type=str, help="Path to the log file.", required=True)
    args = parser.parse_args()
    
    log_content = read_logging(args.log_file)

    summary = analyze_log(log_content)
    print(f"Summarize {args.log_file}:\n")
    print(summary)


if __name__ == "__main__":
    main()