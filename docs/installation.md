# Curie

## Installation

1. Install docker: https://docs.docker.com/engine/install/ubuntu/. 
Grant permission to docker via `sudo chmod 666 /var/run/docker.sock`. Run `docker ps` to check the permission with the Docker daemon. 

2. Build the container image. Whenever changes have been made: delete the current mounted volume (after backing up necessary data, of course), and rebuild the container image.

```bash
git clone https://github.com/Just-Curieous/Curie.git
cd Curie
pip install -e .
cd curie && sudo docker build --no-cache --progress=plain -t exp-agent-image -f ExpDockerfile_default .. && cd -
```

## Quick Start

1. Put your LLM API credentials under `curie/setup/env.sh`. Example: 

```
export MODEL="gpt-4o"
export API_VERSION="2024-06-01"
export OPENAI_API_KEY= 
export OPENAI_ORGANIZATION= 
export OPENAI_API_BASE= 
```


2. Input your research problem
```bash
python3 -m curie.main -q "How does the choice of sorting algorithm impact runtime performance across different input distributions?" --task_config curie/configs/base_config.json
```
You can check the logging under `logs/research_question_<ID>.log`.

You can check the reproducible experimentation process under `workspace/research_<ID>`.

## Tutorial for Reproducing 'Large Language Monkeys' Results

The paper [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](https://arxiv.org/abs/2407.21787) explores repeated sampling as a method to enhance reasoning performance in large language models (LLMs) by increasing inference compute. 

Download the related starter files under `workspace`.
```bash
cd Curie
git submodule update --init --recursive 
```
- [ ] TODO: need update the credential for large language monkey.

As a LLM researcher, you are just curious about how does the number of generated samples per question impact the overall success? (The concrete question can be found in our benchmark `benchmark/llm_reasoning/q1_simple_relation.txt`, which specify the location of corresponding starter files.)

```bash
cd Curie
python3 -m curie.main --iterations 1 --question_file benchmark/llm_reasoning/q1_simple_relation.txt --task_config curie/configs/llm_reasoning_config.json
```

You can check the logging under `logs/q1_simple_relation_<ID>.log`.

You can check the reproducible experimentation process under `workspace/large_language_monkeys_<ID>`.



## Develop Your Customized Experimentation Agents

Config `curie/configs/base_config.json`.
- You can add your domain-specific instructions for the supervisor by replacing `supervisor_system_prompt_filename` and worker `control_worker_system_prompt_filename`
- TODO
