# Curie: Automate Rigorous Scientific Experimentation

![Documentation](https://img.shields.io/badge/docs-Just--Curieous.github.io-blue)
![Discord](https://img.shields.io/discord/discord-id?label=Discord&logo=discord&link=https://discord.gg/uCEbmG7EKU)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)

Curie is the first AI-agent framework designed for automated and rigorous scientific experimentation. 
Curie helps answer your curiosity through end-to-end experimentation automation, ensuring that every step—from hypothesis formulation to result interpretation—is conducted with precision, reliability, and reproducibility.
<p align="center">
  <img src="./docs/static/img/curie-overview.png" width="600px"/>
</p>

**Key Features**
- 🚀 Automated Experimentation – End-to-end workflow management: hypothesis formulation, experiment setup, experiment execution, result analysis and finding reflection.
- 📊 Data-Driven Insights – Systematic analysis and structured result documentation.
- 🔄 Iterative Refinement – Adapts hypotheses and re-executes experiments based on findings.
- 🔬 Broad Applicability – Supports ML research, system analysis, and scientific discovery.
- 📖 Experimentation Benchmark - Provide 46 questions from 4 Computer Science domains, based on influential papers and open-source projects (`benchmark/`).


- [ ] add some evaluation results/figures.
- [ ] add pointer to our website.
- [ ] add more tutorials and move to website.

## Table of Contents 
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Use Cases](#use-cases)
- [Tutorial](#tutorial-for-reproducing-large-language-monkeys-results)
- [Customize Your Experiment Agents](#develop-your-customized-experimentation-agents) 

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
- You can check the logging under `logs/research_question_<ID>.log`.

- You can check the reproducible experimentation process under `workspace/research_<ID>`.

## Use Cases
Curie is designed for scientific discovery across multiple domains:

- 🔬 Machine Learning & AI Research – Hyperparameter tuning, algorithm behavior and 
- 💻 System Performance Analysis – Benchmarking systems, optimizing configurations, investigating system trade-offs.
- 🧪 Algorithmic & Scientific Discovery – Validating hypotheses, automating computational simulations.
 

## Tutorial for Reproducing 'Large Language Monkeys' Results

The paper [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](https://arxiv.org/abs/2407.21787) explores repeated sampling as a method to enhance reasoning performance in large language models (LLMs) by increasing inference compute. 

Download the related starter files under `workspace`.
```bash
cd Curie
git submodule update --init --recursive 
```
- [ ] TODO: need update the credential for large language monkey.

As a LLM researcher, you are just curious about how does the number of repeatedly generated samples per question impact the overall success? (The concrete question can be found in our benchmark `benchmark/llm_reasoning/q1_simple_relation.txt`, which specify the location of corresponding starter files.)

```bash
cd Curie
python3 -m curie.main --iterations 1 --question_file benchmark/llm_reasoning/q1_simple_relation.txt --task_config curie/configs/llm_reasoning_config.json
```

- You can check the logging under `logs/q1_simple_relation_<ID>.log`.

- You can check the reproducible experimentation process under `workspace/large_language_monkeys_<ID>`.


## Customize Your Experimentation Agents

Config `curie/configs/base_config.json`.
- You can add your domain-specific instructions for the supervisor by customizing `supervisor_system_prompt_filename` and worker `control_worker_system_prompt_filename`
- TODO


## Community and Support

Join our community on [Discord](https://discord.gg/uCEbmG7EKU) to connect with other users and developers. For any issues or feature requests, please open an issue on our [GitHub Issues](https://github.com/Just-Curieous/curie/issues) page.

## License

Curie is released under the Apache 2.0 License. See `LICENSE` for more details.
