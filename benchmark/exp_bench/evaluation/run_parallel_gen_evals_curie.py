#!/usr/bin/env python3
import json
import os
import subprocess
import time
import re
import argparse
import shutil
import datetime
from pathlib import Path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from helper.logger import init_logger
bench_logger = init_logger("logs/curie_bench.log")
import evaluation.eval as eval_mod
eval_mod.bench_logger = bench_logger
from evaluation.eval import git_clone_repo, mask_repo, generate_task_prompt

def parse_llm_model_name(llm_config_path):
    model_keys = ["MODEL_NAME", "LLM_MODEL", "MODEL"]
    with open(llm_config_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("export "):
                parts = line[len("export "):].split("=", 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip().strip('"').strip("'")
                    if key in model_keys:
                        return value
    return "unknown-llm"

def get_standard_output_name(md_filename, paper_id):
    m = re.match(r".*?_task_index_(\d+)_iter_(\d+)_duration_([0-9.]+)", md_filename)
    if m:
        task_index, iter_num, duration = m.groups()
        return f"{paper_id}_task_index_{task_index}_iter_{iter_num}_duration_{duration}_eval_gen.json", \
               f"{paper_id}_task_index_{task_index}_iter_{iter_num}_duration_{duration}_eval_gen.patch"
    else:
        base = os.path.splitext(md_filename)[0]
        return base + ".json", base + ".patch"

def get_github_url_from_paper_details(paper_id, paper_details_path):
    """Consistent with run_parallel_gen_evals.py, find github_url, prioritize code_url, then reproduce_eval.code"""
    with open(paper_details_path, "r") as f:
        for line in f:
            record = json.loads(line)
            if str(record.get("id")) == str(paper_id):
                if record.get("code_url"):
                    return record["code_url"]
                elif record.get("reproduce_eval") and record["reproduce_eval"].get("code"):
                    return record["reproduce_eval"]["code"]
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_duration", type=float, default=0.5)
    parser.add_argument("--specific_tasks", type=str, default=None)
    parser.add_argument("--llm_config", type=str, default="evaluation/setup/env_llm_config.sh")
    args = parser.parse_args()
    specific_tasks = None
    paper_id = None
    task_index = "0"  # Default value
    if args.specific_tasks:
        specific_tasks = json.loads(args.specific_tasks)
        if specific_tasks and len(specific_tasks[0]) > 1:
            paper_id = int(specific_tasks[0][1])
            if len(specific_tasks[0]) > 2:
                task_index = str(specific_tasks[0][2])  # Extract task_index
    if paper_id is None:
        raise ValueError("Failed to parse paper_id from --specific_tasks parameter")
    llm_config_path = args.llm_config
    duration = args.max_duration

    # 1. clone+mask+copy repo
    paper_details_path = "logs/neurips2024/neurips_abs_2024_withcode_popularity_stars-100.json"
    def get_github_url_from_paper_details(paper_id, paper_details_path):
        with open(paper_details_path, "r") as f:
            for line in f:
                record = json.loads(line)
                if str(record.get("id")) == str(paper_id):
                    if record.get("code_url"):
                        return record["code_url"]
                    elif record.get("reproduce_eval") and record["reproduce_eval"].get("code"):
                        return record["reproduce_eval"]["code"]
        return None
    github_url = get_github_url_from_paper_details(paper_id, paper_details_path)
    if not github_url:
        raise ValueError(f"GitHub URL not found for paper_id={paper_id}")
    # Extract repo name
    repo_name = os.path.splitext(os.path.basename(github_url))[0]
    suffix = f"_task_index_{task_index}_iter_1_duration_{duration}"
    workspace_dir_name = f"{repo_name}{suffix}"
    # Read task_data
    paper_tasks_filename = f"outputs/logs/neurips2024/{paper_id}/{paper_id}_complete_final.json"
    with open(paper_tasks_filename, "r") as f:
        paper_tasks_complete = json.load(f)
    task_data = paper_tasks_complete["questions"][int(task_index)]
    # clone repo
    from evaluation.eval import git_clone_repo, mask_repo, generate_task_prompt
    repo_path = git_clone_repo(github_url, base_dir="workspace", suffix=suffix)
    import time
    time.sleep(1)
    if not os.path.exists(repo_path):
        print(f"[ERROR] Clone directory not found: {repo_path}")
        import sys
        sys.exit(1)
    else:
        print(f"[CHECK PASSED] Clone directory exists: {repo_path}")
    # mask
    is_mask_fail, failed_masked_sources = mask_repo(task_data, repo_path)
    if is_mask_fail:
        print(f"Masking failed for files: {failed_masked_sources}")
    # Copy to Curie root directory workspace
    dst = os.path.join("workspace", os.path.basename(repo_path))
    abs_dst = os.path.abspath(os.path.join("../../workspace", os.path.basename(repo_path)))
    os.makedirs(os.path.dirname(abs_dst), exist_ok=True)
    if os.path.exists(abs_dst):
        shutil.rmtree(abs_dst)
    shutil.copytree(repo_path, abs_dst)
    print(f"Copied masked repo to: {abs_dst}")

    # 2. Parse llm_config to generate key_dict
    def parse_llm_keys(llm_config_path):
        key_map = {
            'MODEL': None,
            'AZURE_API_VERSION': None,
            'AZURE_API_KEY': None,
            'ORGANIZATION': None,
            'AZURE_API_BASE': None
        }
        with open(llm_config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('export '):
                    parts = line[len('export '):].split('=', 1)
                    if len(parts) == 2:
                        k, v = parts[0].strip(), parts[1].strip().strip('"').strip("'")
                        if k in key_map:
                            key_map[k] = v
        return key_map
    key_dict = parse_llm_keys(llm_config_path)
    llm_name = key_dict.get('MODEL', 'llm')

    # 3. Use generate_task_prompt to generate real prompt (agent_name=local)
    config = {
        "agent_name": "local",
        "eval_gen_prompt": "evaluation/prompts/eval_gen_prompt.txt",
        "paper_id": paper_id,
        "iteration": 1,
        "max_duration_per_task_in_hours": float(duration),
        "output_folder": "outputs/tmp_prompt_gen"
    }
    task_prompt = generate_task_prompt(task_data, config, 1)

    # 4. Generate run_task.py to Curie root directory, all fields use real content
    run_task_content = f'''import curie
key_dict = {json.dumps(key_dict, ensure_ascii=False, indent=4)}

result = curie.experiment(
    api_keys=key_dict,
    question={json.dumps(task_prompt, ensure_ascii=False)},
    task_config={{
        "job_name": "default_research",
        "docker_image": "amberljc/curie:latest",
        "dockerfile_name": "ExpDockerfile_pip",
        "benchmark_specific_context": "none",
        "is_user_interrupt_allowed": False,
        "timeout": 600,
        "max_coding_iterations": 10,
        "max_global_steps": 10,
        "supervisor_system_prompt_filename": "prompts/simple/simple-supervisor.txt",
        "control_worker_system_prompt_filename": "prompts/simple/simple-control-worker.txt",
        "patcher_system_prompt_filename": "prompts/simple/simple-patcher.txt",
        "llm_verifier_system_prompt_filename": "prompts/simple/simple-llm-verifier.txt",
        "coding_prompt_filename": "prompts/simple/simple-coding.txt",
        "worker_system_prompt_filename": "prompts/simple/simple-worker.txt",
        "workspace_name": "{dst}",
        "dataset_dir": "",
        "env_requirements": "",
        "code_instructions": ""
    }}
)
'''
    run_task_path = os.path.abspath(os.path.join("../../run_task.py"))
    with open(run_task_path, "w") as f:
        f.write(run_task_content)
    print(f"Generated run_task.py to: {run_task_path}")

    # ====== Generate run_task.py and run ======
    run_task_py_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
    run_task_py_path = os.path.join(run_task_py_dir, 'run_task.py')
    if os.path.exists(run_task_py_path):
        print("Starting execution of run_task.py ...")
        process = subprocess.Popen(['python', 'run_task.py'], cwd=run_task_py_dir)
        while True:
            ret = process.poll()
            if ret is not None:
                print(f"run_task.py execution completed, return code: {ret}")
                break
            else:
                print("run_task.py still running, waiting...")
                time.sleep(30)
    else:
        print(f"run_task.py not found: {run_task_py_path}")

    # ====== Log post-processing: extract md to generate output and patch ======
    import re
    import glob
    
    # Fix path matching issue: use more flexible matching pattern
    repo_name = os.path.splitext(os.path.basename(github_url))[0]
    LOGS_GLOB = os.path.abspath(os.path.join("../../logs", f"{repo_name}*_task_index_{task_index}_iter_1_duration_{duration}*_iter1"))
    log_dirs = glob.glob(LOGS_GLOB)
    print(f"Searching log directory pattern: {LOGS_GLOB}")
    print(f"Found log directories: {log_dirs}")
    
    OUTPUT_BASE = os.path.join("outputs/evaluation/neurips2024", str(paper_id), "curie", llm_name)
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    # Use main process parameters to generate standard name directly
    iter_num = 1  # Currently only supports single iter
    def get_standard_output_name():
        return f"{paper_id}_task_index_{task_index}_iter_{iter_num}_duration_{duration}_eval_gen.json", \
               f"{paper_id}_task_index_{task_index}_iter_{iter_num}_duration_{duration}_eval_gen.patch"
    
    for log_dir in log_dirs:
        print(f"Processing log directory: {log_dir}")
        for file in os.listdir(log_dir):
            if not file.endswith(".md"):
                continue
            file_path = os.path.join(log_dir, file)
            print(f"Processing file: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Remove markdown code block markers
            if content.startswith("```markdown"):
                content = content[11:]  # Remove leading ```markdown
            if content.endswith("```"):
                content = content[:-3]  # Remove trailing ```
            
            # Fix regex: properly handle markdown format
            design_match = re.search(r"## Experiment Design\s*([\s\S]*?)(^## |^# |\Z)", content, re.MULTILINE)
            design = design_match.group(1).strip() if design_match else ""
            
            conclusion_match = re.search(r"# Conclusion and Future Work\s*([\s\S]*?)(^# |\Z)", content, re.MULTILINE)
            conclusion = conclusion_match.group(1).strip() if conclusion_match else ""
            
            print(f"Design length: {len(design)}")
            print(f"Conclusion length: {len(conclusion)}")
            json_name, patch_name = get_standard_output_name()
            out_json_path = os.path.join(OUTPUT_BASE, json_name)
            out_patch_path = os.path.join(OUTPUT_BASE, patch_name)
            if design and conclusion:
                result = {"design": design, "conclusion": conclusion}
                os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
                with open(out_json_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"Extracted: {file_path} -> {out_json_path}")
                # Generate patch (masked repo and agent repo)
                try:
                    with open(out_patch_path, "w", encoding="utf-8") as pf:
                        result = subprocess.run(["diff", "-urN", repo_path, abs_dst], stdout=pf, check=False)
                    if result.returncode == 2:
                        print(f"Failed to generate patch: diff command error, return code 2")
                    elif result.returncode == 1:
                        print(f"Generated patch: {out_patch_path} (with differences)")
                    else:
                        print(f"Generated patch: {out_patch_path} (no differences)")
                except Exception as e:
                    print(f"Failed to generate patch: {e}")
                # Generate config file
                config_name = json_name.replace(".json", "_config_config.json")
                out_config_path = os.path.join(OUTPUT_BASE, config_name)
                config_to_save = {
                    "input_conference_tasks_folder": f"outputs/logs/neurips2024",
                    "input_conference_paper_details_filename": "logs/neurips2024/neurips_abs_2024_withcode_popularity_stars-100.json",
                    "iterations": 1,
                    "parallelization_factor": 6,
                    "llm_config_filename": llm_config_path,
                    "agent_name": "curie",
                    "max_duration_per_task_in_hours": float(duration),
                    "max_papers": 3,
                    "mode": "generate",
                    "docker_image": "exp-bench-image",
                    "dockerfile_name": "ExpDockerfile_default",
                    "eval_gen_prompt": "evaluation/prompts/eval_gen_prompt.txt",
                    "eval_judge_prompt": "evaluation/prompts/eval_judge_prompt.txt",
                    "paper_id": paper_id,
                    "task_index": int(task_index),
                    "iteration": 1,
                    "llm_name": llm_name,
                    "output_folder": OUTPUT_BASE
                }
                with open(out_config_path, "w", encoding="utf-8") as f:
                    json.dump(config_to_save, f, indent=2, ensure_ascii=False)
                print(f"Generated config: {out_config_path}")
            else:
                print(f"Design/conclusion not found: {file_path}")
