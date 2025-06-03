# EXP-Bench

EXP-Bench is our in-house, novel benchmark designed by the Curie team, to systematically evaluate AI agents on complete research experiments sourced from influential AI publications. Given a research question and incomplete starter code, EXP-Bench challenges AI agents to formulate hypotheses, design and implement experimental procedures, execute them, and analyze results. This dataset curates 461 AI research tasks from 51 top-tier AI research papers.

Link to the dataset: [HuggingFace](https://huggingface.co/datasets/Just-Curieous/EXP-Bench)

Next, we describe how to run the EXP-Bench dataset against your choice of LLM and Agent.

# EXP-Bench Evaluation Harness Usage Instructions V0

DISCLAIMER: we are still working on polishing the code here, so please expect frequent changes!

New clean docker containers are used for each instance (task+LLM+Agent) during evaluation (which consists of output generation, and judging). EXP-Bench currently supports evaluation against the following judges and agents, with plans to expand support in the future:

Agent: OpenHands and Iterative Agent
LLMs: OpenAI models, Claude Sonnet 3.7, Claude Haiku 3.5, DeepSeek R1, Amazon Nova Pro

All commands should be executed from this directory. To run the evaluation and judging pipelines, simply build the Docker image and execute the corresponding scripts provided below.

## Setup Docker image
Note: There may be a few python packages missing that need to be installed manually. Also make sure Python 3.12.4 is installed (conda is best, which we show below).
```bash
docker images -q exp-bench-image | xargs -r docker rmi -f # removes any existing conflict image
docker build --progress=plain -t exp-bench-image -f ExpDockerfile_default .
conda create -n exp-bench python=3.12.4 
conda activate exp-bench
pip install -e . # this needs to be executed in the conda env that you just activated
```

## Evaluation

### Evaluation Output Generation
These scripts will generate outputs within the folder `outputs/evaluation`. 

Make sure you have configured `evaluation/config/parallel_eval_gen_config_template.json` with the correct parameters, e.g., LLM config and Agent name.
```bash
python evaluation/parallel_eval.py
```

The following config will work for Inspect Agent, though make sure to set the other parameters correctly, e.g., LLM config.
```bash
python evaluation/parallel_eval.py --task_config=evaluation/configs/parallel_eval_gen_config_template_inspect_agent.json
```

The following is an optional script for generating evaluation output for specific tasks only:
```bash
python evaluation/run_parallel_gen_evals.py \
  --max_duration 0.5 \
  --specific_tasks '[["neurips2024", "93022", 1, 0.25], ["neurips2024", "93022", 1, 0.5], ["neurips2024", "93022", 1, 1], ["neurips2024", "93022", 1, 2], ["neurips2024", "93022", 1, 4], ["neurips2024", "93022", 1, 8], ["neurips2024", "94155", 6, 8]]'
```

### Judge Evaluation Output
Make sure you have configured `evaluation/config/parallel_eval_judge_config_template.json` with the correct parameters, e.g., LLM config and Agent name.
- The specific evalauation output folder within `outputs/evaluation` that will be judged is determined in part by the config keys "llm_config_filename" and "agent_name".
- Leave "judge_agent_name" blank. 
- Fill in "llm_judge_config_filename" with the LLM config used for judging. Currently we default to o3-mini. 
```bash
python evaluation/parallel_eval.py --task_config=evaluation/configs/parallel_eval_judge_config_template.json
```

This is an optional script that will perform judging for all evaluation output configs. Currently, you need to modify the parameters within the script manually. 
```bash
python evaluation/run_parallel_judge_evals.py
```