import json
from typing import Dict, List, Tuple
from openai import AzureOpenAI
import os
from dataclasses import dataclass
import tiktoken
import argparse
from pathlib import Path

@dataclass
class VerificationResult:
    score: float
    matches: List[str]
    mismatches: List[str]
    explanation: str

class ExperimentVerifier:
    def __init__(self, api_key: str, azure_endpoint: str, api_version: str, organization: str):
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            organization=organization
        )
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        return len(self.encoding.encode(text))
    
    def _chunk_log(self, log_content: str, max_tokens: int = 20000) -> List[str]:
        """Split log content into chunks of approximately max_tokens."""
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        lines = log_content.split('\n')
        
        for line in lines:
            if 'LLMCallEvent' in line or '========================' in line or '------------' in line:
                continue
            
            line_tokens = self._count_tokens(line)
            
            if current_tokens + line_tokens > max_tokens:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_tokens = line_tokens
            else:
                current_chunk.append(line)
                current_tokens += line_tokens
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
            
        return chunks

    def _create_verification_prompt(self, log_chunk: str, ground_truth: str, question: str) -> List[Dict]:
        """Create a prompt for verifying a log chunk against ground truth."""
        system_prompt = """
        You are an strict Experimentation Agent Verifier, responsible for evaluating whether an experimentation agent correctly conducted an experiment based on the experimentation question. 
        You are provided with an experiment log chunk, the original experimentation question, and the ground truth (only contains the conclusion).
        Your assessment should focus on:
        1. Experiment Design - Did the agent structure the correct high-level plan to address the experimentation question? It does not need to write implementation code or execute the plan. 
        2. Execution Setup - Is the generated code runnable and producing real outputs? 
        3. Implementation Alignment- Is the code properly aligned with the experimentation design and accurately implementing the intended methodology? Ensure: Legitimate handling of inputs and outputs. No hardcoded or mock data. 
        4. Conclusion Correctness - Is the conclusion acceptable by the ground truth?

        Analyze the provided chunked Log File, and provide a structured evaluation based on the criteria below:
        Response Format
        * Overall Verdict: ✅ Correct / ❌ Incorrect
        * Detailed Assessment:
            * Experiment Design: [Pass/Fail]
            * Execution Setup: [Pass/Fail]
            * Implementation Alignment : [Pass/Fail]
            * Conclusion Correctness: [Pass/Fail]  
        * Explanation: [Concisely explanation about the failure reasons, no reason needed if the step is missing]
        """

        user_prompt = f"""
            > Original Experimentation Question:
            {question}
        
            > Ground Truth:
            {ground_truth}

            > Log Chunk:
            {log_chunk}

            Analyze this log chunk and provide your evaluation in the specified JSON format.
            """

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    def _process_chunk(self, log_chunk: str, ground_truth: str, question: str) -> Dict:
        """Process a single chunk of the log file."""
        messages = self._create_verification_prompt(log_chunk, ground_truth, question)
        
        response = self.client.chat.completions.create(
            model='gpt-4o',
            messages=messages,
            temperature=0.7,
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)

    @staticmethod
    def load_file(file_path: str) -> str:
        # Load ground truth from a txt file
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            return content
        except Exception as e:
            raise ValueError(f"Error loading file {file_path}: {str(e)}")

    def extract_only_question(self, question: str) -> str:
        # use LLM to extract only the question from the question file, ignore the environment setup or code instruction details
        prompt = f"""
        Extract only the research question from the following text, ignore any code or environment setup details:
        {question}
        """
        messages = [
                {"role": "system", "content": ''},
                {"role": "user", "content": prompt}
            ]           
        response = self.client.chat.completions.create(
            model='gpt-4o-mini',
            messages=messages,
            temperature=0.7,
        )

        return response.choices[0].message.content

    def verify_experiment(self, log_file_path: str, 
                        ground_truth_path: str, 
                        question: str
                        ) -> VerificationResult:
        """
        Verify an experiment log against ground truth.
        
        Args:
            log_file_path: Path to the experiment log file
            ground_truth_path: Path to the ground truth JSON file
            
        Returns:
            VerificationResult containing the evaluation results
        """
        # Load ground truth
        ground_truth = self.load_file(ground_truth_path)
        question = self.load_file(question)
        question = self.extract_only_question(question)
        log_content = self.load_file(log_file_path)
        
        # Split log into chunks
        chunks = self._chunk_log(log_content)
        
        # Process each chunk
        chunk_results = []
        for i, chunk in enumerate(chunks, 1):
            print(f"Processing chunk {i}/{len(chunks)}...")
            result = self._process_chunk(chunk, ground_truth, question)
            chunk_results.append(result)
        
        # Aggregate results
        return chunk_results

def save_results(result: VerificationResult, output_path: str):
    # save to a txt file
    with open(output_path, 'a') as f:
        json.dump(result, f, indent=2)
        f.write("\n")

def aggregated_result(results: List[VerificationResult]) -> VerificationResult:

    aggregated_results = {
        "Experiment Design": "Fail",
        "Execution Setup": "Fail",
        "Implementation Alignment": "Fail",
        "Conclusion Correctness": "Fail"
    } 
    explanation_list = []
    for result in results:
        try :
            for key, value in result["Detailed Assessment"].items():
                if value == "Pass":
                    aggregated_results[key] = "Pass"
            explanation_list.append(result["Explanation"])
        except:
            pass
        
    aggregated_results["Explanation"] = explanation_list

    return aggregated_results



def main():
    parser = argparse.ArgumentParser(description='Experiment Log Verifier')
    parser.add_argument('--log_file', type=str, help='Path to the experiment log file')
    parser.add_argument('--ground_truth', type=str, help='Path to the ground truth JSON file')
    parser.add_argument('--question', type=str, help='Research question')
    parser.add_argument('--output', type=str, help='Path to save results (optional)') 
    args = parser.parse_args()

    # Check if API credentials are provided as arguments or environment variables
    api_key = os.getenv('OPENAI_API_KEY')
    endpoint =  os.getenv('OPENAI_API_BASE')
    organization = os.getenv('OPENAI_ORGANIZATION')
    api_version = os.getenv('API_VERSION')

    if not all([api_key, endpoint, organization]):
        raise ValueError("Missing required API credentials. Provide them as arguments or environment variables.")

    try:
        # Initialize verifier
        verifier = ExperimentVerifier(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
            organization=organization
        )

        # Run verification
        results = verifier.verify_experiment(args.log_file, args.ground_truth, args.question)
        # print("\n=== Verification Results ===")
        # for result in results:
        #     print(json.dumps(result, indent=2))
        #     print("\n") 

        # Print the aggregated result
        aggregated_results = aggregated_result(results)
        for key, value in aggregated_results.items():
            print(f"{key}: {value}")
        
        # Save results if output path is provided
        if args.output:
            output_path = args.output
            save_results(args.log_file, output_path)
            save_results(aggregated_results, output_path)
            print(f"\nResults saved to: {output_path}")

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())



# python eval_figures/llm_judge.py --log_file  --ground_truth   --question   --output 

# python eval_figures/llm_judge.py --log_file "eval_metadata/logs/curie/vector_index/q11_data_character_20250125043238_iter2.log" --ground_truth "benchmark/vector_index/ground_truth/q11.txt" --question "benchmark/vector_index/q11_data_character.txt" --output "eval_metadata/llm_judge_logs/vdb_q11_verification_results.txt" 
 