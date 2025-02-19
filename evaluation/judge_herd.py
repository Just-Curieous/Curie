

import json
import argparse
from collections import defaultdict
from typing import List, Dict
import subprocess
import os
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class LogAnalysisResult:
    experiment_design: str
    execution_setup: str
    implementation_alignment: str
    conclusion_correctness: str
    explanations: List[str]

class LogAnalyzer:
    def __init__(self):
        self.results: List[LogAnalysisResult] = []
        
    def run_analysis_command(self, commands: List[str]) -> None:
        """Run the analysis command for each log file."""
        for cmd in commands:
            try:
                subprocess.run(cmd.split(), check=True)
                # Read the output file after each command
                self._process_output_file(cmd.split()[-1])
            except subprocess.CalledProcessError as e:
                print(f"Error running command: {cmd}")
                print(f"Error details: {e}")

    def _process_output_file(self, output_file: str) -> None:
        """Process the output file and store results."""
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
                result = LogAnalysisResult(
                    experiment_design=data["Experiment Design"],
                    execution_setup=data["Execution Setup"],
                    implementation_alignment=data["Implementation Alignment"],
                    conclusion_correctness=data["Conclusion Correctness"],
                    explanations=data["Explanation"]
                )
                self.results.append(result)
        except FileNotFoundError:
            print(f"Output file not found: {output_file}")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {output_file}")

    def generate_summary(self) -> Dict:
        """Generate a summary of all analyzed logs."""
        if not self.results:
            return {"error": "No results to analyze"}

        total_logs = len(self.results)
        metrics = {
            "Experiment Design": defaultdict(int),
            "Execution Setup": defaultdict(int),
            "Implementation Alignment": defaultdict(int),
            "Conclusion Correctness": defaultdict(int)
        }
        
        # Count occurrences of each result
        for result in self.results:
            metrics["Experiment Design"][result.experiment_design] += 1
            metrics["Execution Setup"][result.execution_setup] += 1
            metrics["Implementation Alignment"][result.implementation_alignment] += 1
            metrics["Conclusion Correctness"][result.conclusion_correctness] += 1

        # Calculate percentages
        summary = {
            "total_logs_analyzed": total_logs,
            "metrics": {}
        }

        for metric, counts in metrics.items():
            summary["metrics"][metric] = {
                result: {
                    "count": count,
                    "percentage": (count / total_logs) * 100
                }
                for result, count in counts.items()
            }

        return summary

def main():
    parser = argparse.ArgumentParser(description='Analyze multiple log files and aggregate results')
    parser.add_argument('--commands_file', type=str, required=True,
                      help='File containing list of analysis commands')
    parser.add_argument('--output_summary', type=str, required=True,
                      help='Output file for the aggregated summary')

    args = parser.parse_args()

    # Read commands from file
    with open(args.commands_file, 'r') as f:
        commands = [line.strip() for line in f if line.strip()]

    # Initialize analyzer and run analysis
    analyzer = LogAnalyzer()
    analyzer.run_analysis_command(commands)

    # Generate and save summary
    summary = analyzer.generate_summary()
    
    with open(args.output_summary, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary to console
    print("\nAnalysis Summary:")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

    