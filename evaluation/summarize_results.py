from openai import AzureOpenAI
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple
import argparse


def analyze_explanations(explanations: List) -> str:
    """Compile explanations for analysis.""" 
    # Prepare message for GPT analysis

    messages = [
        {"role": "system", "content": "Analyze these experiment failure explanations. Concisely summarize the common mistakes and issues encountered. Order them by frequency."},
        {"role": "user", "content": str(explanations)}
    ]
    
    client = AzureOpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        azure_endpoint=os.getenv('OPENAI_API_BASE'),
        api_version="2024-02-01",
        organization=os.getenv('OPENAI_ORGANIZATION')
    )
    
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages,
        temperature=0.7,
        stop=None
    )

    return response.choices[0].message.content


def parse_log_file(content: str) -> List[Tuple[str, Dict]]:
    """
    Parse log content where each JSON object is preceded by a filename.
    Returns a list of tuples containing (filename, parsed_json).
    """
    # Split content into lines
    lines = content.strip().split('\n')
    entries = []
    current_filename = None
    current_json_str = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if line is a filename (starts with quotes and ends with .jsonl")
        if line.startswith('"') and (line.endswith('.jsonl"') or line.endswith('.log"')):
            # If we have collected JSON from previous filename, parse it
            if current_filename and current_json_str:
                try:
                    json_data = json.loads(current_json_str)
                    entries.append((current_filename, json_data))
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON for {current_filename}: {e}")
            
            # Start new entry
            current_filename = line.strip('"')
            current_json_str = ""
        else:
            # Append to current JSON string
            current_json_str += line

    # Don't forget to process the last entry
    if current_filename and current_json_str:
        try:
            json_data = json.loads(current_json_str)
            entries.append((current_filename, json_data))
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON for {current_filename}: {e}")
    
    return entries


def calculate_metrics(entries: List[Tuple[str, Dict]]) -> Dict[str, float]:
    """Calculate average pass rates for each metric."""
    metrics = defaultdict(lambda: {'pass': 0, 'total': 0})
    metrics["Overall"] = {'pass': 0, 'total': 0}
    explanation_list = []
    for _, data in entries:
        tmp_score = 0
        for key in ["Experiment Design", "Execution Setup", 
                   "Implementation Alignment", "Conclusion Correctness"]:
            if key in data:
                metrics[key]['total'] += 1
                if data[key] == "Pass":
                    metrics[key]['pass'] += 1
                    tmp_score += 1
            if tmp_score == 4: 
                metrics["Overall"]['pass'] += 1
        
        metrics["Overall"]['total'] += 1
        metrics["Average"]["total"] += 1

        if "Explanation" in data:
            explanation_list.append(data["Explanation"])
    
    metrics["Average"]["pass"] = (metrics["Experiment Design"]['pass'] + metrics["Execution Setup"]['pass'] + \
                        metrics["Implementation Alignment"]['pass'] + metrics["Conclusion Correctness"]['pass']) 
    metrics["Average"]["total"] /= 4
    return {
        key: (stats['pass'] / stats['total'] * 100) if stats['total'] > 0 else 0
        for key, stats in metrics.items()
    }, explanation_list

def analyze_one_task(file_name: str):
    print(f"=== Analysis for {file_name} ===")

    with open(file_name, 'r') as f:
        content = f.read()
    # print(f'content: {content}')
    entries = parse_log_file(content) 
    metrics, explanations = calculate_metrics(entries)
    print(metrics) 
    
    summary = analyze_explanations(explanations)
    # print(summary)

    write_to = file_name.replace(".txt", "_metrics.log")
    with open(write_to, 'a') as f:
        f.write(json.dumps(metrics, indent=2))
        f.write(json.dumps(summary, indent=2))

def main():
    parser = argparse.ArgumentParser(description='Experiment Log Verifier')
    parser.add_argument('--log_dir', type=str, default="eval_metadata/llm_judge_logs", help='Path to the experiment log file')
    args = parser.parse_args()

    # Read and process the example content
    print(os.listdir(args.log_dir))
    for file_name in os.listdir(args.log_dir):
        if file_name.endswith(".txt") and file_name.startswith("mag_cloud"):
            exp = analyze_one_task(os.path.join(args.log_dir, file_name))

    
if __name__ == "__main__":
    main()