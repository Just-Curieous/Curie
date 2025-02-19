import os 
from typing import Dict, List
import pandas as pd
from pathlib import Path
from openai import AzureOpenAI

# run this file at ~/langgraph-exp-agent$

class LLMLogEvaluator:
    """Evaluates LLM log files focusing on the latter half of reasoning patterns."""
    
    def __init__(self, log_dir: str = 'eval_metadata/logs/openhands/llm_reasoning/',
                 save_file: str =  'eval_figures/common_errors_report.txt',
                 azure_api_key: str = None,
                 azure_endpoint: str = None):
        self.log_dir = log_dir
        self.client = None
        self.save_file = save_file
        if azure_api_key and azure_endpoint:
            self.client = AzureOpenAI(
                api_key=azure_api_key,
                azure_endpoint=azure_endpoint,
                api_version="2024-02-01",
                organization='499023'
            )

    def analyze_with_gpt(self, text: str) -> str:
        """Use Azure OpenAI to analyze reasoning errors with an enriched prompt."""
        if not self.client:
            return "GPT analysis not available - no API credentials provided"
            
        messages = [
            {"role": "system", "content": """
            You are an advanced AI system tasked with analyzing the log of an experimentation agent's activities. 
            Carefully read the provided log entries and identify any experimentation-related issues the agent encountered during its experimentation process. 
 
            Provide a list of issues it encounters. Highlights the most significant patterns found. 
            Focus on actionable insights that could improve the experimentation agent ability."""},
            {"role": "user", "content": text}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model='gpt-4o-mini',
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"GPT analysis failed: {str(e)}"

    def extract_reasoning_chains(self, log_data: Dict) -> List[str]:
        """Extract reasoning chains from log data."""
        reasoning_chains = []
        
        def extract_from_dict(d):
            for value in d:
                if isinstance(value, str) and len(value) > 50:  # Likely reasoning text
                    reasoning_chains.append(value)
                elif isinstance(value, dict):
                    extract_from_dict(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            extract_from_dict(item)
        
        extract_from_dict(log_data)
        return reasoning_chains

    def analyze_file(self, file_path: Path) -> Dict:
        """Analyze the latter half of a log file and return detailed analysis."""
        # try:
        with open(file_path, 'r') as f:
            log_data = f.read()
        
        # Take only the latter half of reasoning chains
        half_point = len(log_data) // 2
        recent_chains = log_data[half_point:]
        
        if not recent_chains:
            return {
                'file': file_path.name,
                'error': "No reasoning chains found in file"
            }
        
        # Combine recent chains for GPT analysis
        combined_text = "\n".join(recent_chains)
        gpt_analysis = self.analyze_with_gpt(combined_text[:4000])  # Limit text length
        
        return {
            'file': file_path.name,
            'gpt_analysis': gpt_analysis,
            'chains_analyzed': len(recent_chains), 
        } 

    def generate_file_summaries(self) -> List[Dict]:
        """Generate summaries for each log file."""
        summaries = []
        with open(self.save_file, 'a') as report_file:
            for i, file_path in enumerate(Path(self.log_dir).glob('*.log')):
                print(f'Processing {i}th file {file_path}...')
                summary = self.analyze_file(file_path)
                summaries.append(summary)
                report_file.write(f"Summary for {file_path.name}:\n")
                report_file.write(f"{summary}\n\n")
    
        return summaries

    def format_file_summary(self, summary: Dict) -> str:
        """Format a single file's analysis summary as readable text."""
        if 'error' in summary:
            return f"\nFile: {summary['file']}\n{summary['error']}\n"
            
        report = [
            f"\nFile: {summary['file']}",
            f"Chains Analyzed: {summary['chains_analyzed']} \n",
            "GPT Analysis:",
            summary['gpt_analysis']
        ]
        
        return "\n".join(report)

    def generate_summary_report(self) -> str:
        """Generate a comprehensive report with per-file summaries."""
        summaries = self.generate_file_summaries()
        
        report = [
            "LLM Reasoning Analysis Report",
            "===========================\n",
            f"Total Files Analyzed: {len(summaries)}\n",
            "Per-File Analysis:",
            "----------------"
        ]
        
        for summary in summaries:
            report.append(self.format_file_summary(summary))
            
        return "\n".join(report)

import time
def main():
    evaluator = LLMLogEvaluator(
        azure_api_key=os.getenv('OPENAI_API_KEY'),
        azure_endpoint=os.getenv('OPENAI_API_BASE'),
        save_file=f'eval_figures/common_errors_nonbug_report-{int(time.time())}.txt'
    )
    print(evaluator.generate_summary_report())

if __name__ == "__main__":
    main()