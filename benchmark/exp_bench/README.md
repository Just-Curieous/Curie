# EXP-Bench Usage Instructions

## Setup docker image
```bash
docker images -q exp-bench-image | xargs -r docker rmi -f # remove any existing conflict image
docker build --progress=plain -t exp-bench-image -f ExpDockerfile_default .
docker build --no-cache --progress=plain -t exp-bench-image -f ExpDockerfile_default . # no cache version (not really needed)
# There may be some python packages missing that need to be installed. Also make sure python 3.12.4 is installed (conda is best).
conda create -n exp-bench python=3.12.4 
conda activate exp-bench
pip install -e . # this needs to be executed in the conda env too
```

# Evaluation:
Make sure you have configured `evaluation/config/parallel_eval_gen_config_template.json` with the correct parameters, e.g., llm config and agent name
1. First, generate evaluation output:
```bash
python evaluation/parallel_eval.py

python evaluation/parallel_eval.py --task_config=evaluation/configs/parallel_eval_gen_config_template_inspect_agent.json # use this for inspect-agent instead

# Parallel specific task eval:
# Explanation of list args: conf_name, paper_id, task_index, duration_in_hours
python evaluation/run_parallel_gen_evals.py \
  --max_duration 0.5 \
  --specific_tasks '[["neurips2024", "93022", 1, 0.25], ["neurips2024", "93022", 1, 0.5], ["neurips2024", "93022", 1, 1], ["neurips2024", "93022", 1, 2], ["neurips2024", "93022", 1, 4], ["neurips2024", "93022", 1, 8], ["neurips2024", "94155", 6, 8]]'
```

2. Second, generate judge output based on evaluation output and ground truths:
Make sure you have configured `evaluation/config/parallel_eval_judge_config_template.json` with the correct parameters, e.g., llm config and agent name
- NOTE: the output folder that will be judged is determined in part by the config keys "llm_config_filename": "evaluation/setup/env-claude-sonnet-37.sh",
    "agent_name": "openhands". Judge has its own keys that are used to select the LLM used for judging. 
```bash
python evaluation/parallel_eval.py --task_config=evaluation/configs/parallel_eval_judge_config_template.json

# Use this instead, it covers all agents + LLM combos + conference combos:
python evaluation/run_parallel_judge_evals.py
```