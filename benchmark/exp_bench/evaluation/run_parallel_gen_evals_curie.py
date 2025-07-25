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
    """与run_parallel_gen_evals.py一致，查找github_url，优先code_url，其次reproduce_eval.code"""
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
    if args.specific_tasks:
        specific_tasks = json.loads(args.specific_tasks)
        if specific_tasks and len(specific_tasks[0]) > 1:
            paper_id = int(specific_tasks[0][1])
    if paper_id is None:
        raise ValueError("未能从--specific_tasks参数中解析出paper_id")
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
        raise ValueError(f"未找到paper_id={paper_id}的github_url")
    task_index = "1"
    # 提取repo名
    repo_name = os.path.splitext(os.path.basename(github_url))[0]
    suffix = f"_task_index_{task_index}_iter_1_duration_{duration}"
    workspace_dir_name = f"{repo_name}{suffix}"
    # 读取 task_data
    paper_tasks_filename = f"outputs/logs/neurips2024/{paper_id}/{paper_id}_complete_final.json"
    with open(paper_tasks_filename, "r") as f:
        paper_tasks_complete = json.load(f)
    task_data = paper_tasks_complete["questions"][0]
    # clone repo
    from evaluation.eval import git_clone_repo, mask_repo, generate_task_prompt
    repo_path = git_clone_repo(github_url, base_dir="workspace", suffix=suffix)
    import time
    time.sleep(1)
    if not os.path.exists(repo_path):
        print(f"[错误] clone目录未找到: {repo_path}")
        import sys
        sys.exit(1)
    else:
        print(f"[检查通过] clone目录存在: {repo_path}")
    # mask
    is_mask_fail, failed_masked_sources = mask_repo(task_data, repo_path)
    if is_mask_fail:
        print(f"Masking failed for files: {failed_masked_sources}")
    # copy到Curie根目录workspace
    dst = os.path.join("workspace", os.path.basename(repo_path))
    abs_dst = os.path.abspath(os.path.join("../../workspace", os.path.basename(repo_path)))
    os.makedirs(os.path.dirname(abs_dst), exist_ok=True)
    if os.path.exists(abs_dst):
        shutil.rmtree(abs_dst)
    shutil.copytree(repo_path, abs_dst)
    print(f"已复制mask好的repo到: {abs_dst}")

    # 2. 解析llm_config生成key_dict
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

    # 3. 用generate_task_prompt生成真实prompt（agent_name=local）
    config = {
        "agent_name": "local",
        "eval_gen_prompt": "evaluation/prompts/eval_gen_prompt.txt",
        "paper_id": paper_id,
        "iteration": 1,
        "max_duration_per_task_in_hours": float(duration),
        "output_folder": "outputs/tmp_prompt_gen"
    }
    task_prompt = generate_task_prompt(task_data, config, 1)

    # 4. 生成run_task.py到Curie根目录，所有字段用真实内容
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
    print(f"已生成 run_task.py 到: {run_task_path}")

    # ====== 生成 run_task.py 并运行 ======
    run_task_py_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
    run_task_py_path = os.path.join(run_task_py_dir, 'run_task.py')
    if os.path.exists(run_task_py_path):
        print("开始执行 run_task.py ...")
        process = subprocess.Popen(['python', 'run_task.py'], cwd=run_task_py_dir)
        while True:
            ret = process.poll()
            if ret is not None:
                print(f"run_task.py 执行完毕，返回码: {ret}")
                break
            else:
                print("run_task.py 仍在运行，等待中...")
                time.sleep(30)
    else:
        print(f"未找到 run_task.py: {run_task_py_path}")

    # ====== 日志后处理：提取md生成output和patch ======
    import re
    import glob
    log_prefix = os.path.basename(repo_path)
    LOGS_GLOB = os.path.abspath(os.path.join("../../logs", f"{log_prefix}*_iter1"))
    log_dirs = glob.glob(LOGS_GLOB)
    OUTPUT_BASE = os.path.join("outputs/evaluation/neurips2024", str(paper_id), "curie", llm_name)
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    # 直接用主流程参数生成标准名
    iter_num = 1  # 目前只支持单iter
    def get_standard_output_name():
        return f"{paper_id}_task_index_{task_index}_iter_{iter_num}_duration_{duration}_eval_gen.json", \
               f"{paper_id}_task_index_{task_index}_iter_{iter_num}_duration_{duration}_eval_gen.patch"
    for log_dir in log_dirs:
        for file in os.listdir(log_dir):
            if not file.endswith(".md"):
                continue
            file_path = os.path.join(log_dir, file)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            # 提取 Experiment Design
            design_match = re.search(r"### Experiment Design\s*([\s\S]*?)(^### |^## |\Z)", content, re.MULTILINE)
            design = design_match.group(1).strip() if design_match else ""
            # 提取 Conclusion and Future Work
            conclusion_match = re.search(r"## Conclusion and Future Work\s*([\s\S]*?)(^## |\Z)", content, re.MULTILINE)
            conclusion = conclusion_match.group(1).strip() if conclusion_match else ""
            json_name, patch_name = get_standard_output_name()
            out_json_path = os.path.join(OUTPUT_BASE, json_name)
            out_patch_path = os.path.join(OUTPUT_BASE, patch_name)
            if design and conclusion:
                result = {"design": design, "conclusion": conclusion}
                os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
                with open(out_json_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"已提取: {file_path} -> {out_json_path}")
                # 生成patch（mask后repo和agent后repo）
                try:
                    with open(out_patch_path, "w", encoding="utf-8") as pf:
                        result = subprocess.run(["diff", "-urN", repo_path, abs_dst], stdout=pf, check=False)
                    if result.returncode == 2:
                        print(f"生成patch失败: diff 命令错误，返回码2")
                    elif result.returncode == 1:
                        print(f"已生成patch: {out_patch_path}（有差异）")
                    else:
                        print(f"已生成patch: {out_patch_path}（无差异）")
                except Exception as e:
                    print(f"生成patch失败: {e}")
                # 生成config文件
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
                print(f"已生成config: {out_config_path}")
            else:
                print(f"未找到design/conclusion: {file_path}")
