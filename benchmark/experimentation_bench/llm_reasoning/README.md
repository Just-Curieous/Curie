# LLM Reasoning: Large Language Monkeys
The paper [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](https://arxiv.org/abs/2407.21787) explores repeated sampling as a method to enhance reasoning performance in large language models (LLMs) by increasing inference compute.

## Common setup

Put the API key into `env.sh` under each working directory.
```bash
export MODEL="gpt-4o"
export API_VERSION="2024-06-01"
export OPENAI_API_KEY=X
export OPENAI_ORGANIZATION=Y
export OPENAI_API_BASE=Z
```

We put all starter file and our code base under `$WORKHOME` 
```bash
export WORKHOME=~/ # change this to your home repo
```

## Large language monkeys setup
```bash
cd $WORKHOME 
git clone https://github.com/AmberLJC/large_language_monkeys.git
```

### Run the ground truth by yourself

```bash
cd $WORKHOME/large_language_monkeys
pip install -r requirements.txt # you may need to fix the requirement.txt based on your workspace environment
python llmonk/generate/gsm8k.py
python llmonk/evaluate/math_datasets.py
```

### Run baseline OpenHands

We provide a way to run OpenHands in our docker, see: [Docker Setup for Benchmarking Yourself](../../README.md). Once you build a docker image using `curie/ExpDockerfile_default` and create container with it, you can connect to this container and run the following commands.

```bash
export LOG_ALL_EVENTS=true
cd /helper/OpenHands 
poetry run python -m openhands.core.main -f <(cat /benchmark/common.txt /benchmark/experimentation_bench/llm_reasoning/q1_simple_relation.txt) 2>&1 | tee -a /logs/openhands_llmmonkey_q1_logging.txt
```

