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
- 📖 Experimentation Benchmark - Provide 46 questions from 4 Computer Science domains, based on influential papers and open-source projects (`benchmark/experimentation_bench`).


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

2. Clone the repository:
```
git clone https://github.com/Just-Curieous/Curie.git
cd Curie
```

3. Put your LLM API credentials under `curie/setup/env.sh`. Example: 

```
export MODEL="azure/gpt-4o"
export AZURE_API_VERSION="2024-06-01"
export AZURE_API_KEY="your-key"
export ORGANIZATION='your-org'
export AZURE_API_BASE='your-base'
```

4. Build the container image. This will take a few minutes. Note: you may need to setup a virtual environment before running pip install.

```bash
pip install -e .
cd curie && sudo docker build --no-cache --progress=plain -t exp-agent-image -f ExpDockerfile_default .. && cd -
```

## Quick Start

1. Input your research question or problem statement (processing time: 5-10 minutes).
```bash
python3 -m curie.main -q "How does the choice of sorting algorithm impact runtime performance across different input distributions?" --task_config curie/configs/base_config.json
```
- You can monitor the logs in real-time at `logs/research_question_<ID>.log`, as they are continuously streamed.

- You can check the reproducible experimentation process under `workspace/research_<ID>`.

## Tutorial
- [How to reproduce the results in `Large Language Monkeys'. ](./docs/tutorial-large-language-monkey.md)


## Use Cases
Curie is designed for scientific discovery across multiple domains:

- 🔬 Machine Learning & AI Research – Hyperparameter tuning, algorithm behavior and 
- 💻 System Performance Analysis – Benchmarking systems, optimizing configurations, investigating system trade-offs.
- 🧪 Algorithmic & Scientific Discovery – Validating hypotheses, automating computational simulations.
 
<p align="center">
  <img src="./docs/static/img/case_study.png" width="1000px"/>
</p>


## Customize Your Experimentation Agents

Config `curie/configs/base_config.json`.
- You can add your domain-specific instructions for the supervisor by customizing `supervisor_system_prompt_filename` and worker `control_worker_system_prompt_filename`
- TODO


## Community and Support

Join our community on [Discord](https://discord.gg/uCEbmG7EKU) to connect with other users and developers. For any issues or feature requests, please open an issue on our [GitHub Issues](https://github.com/Just-Curieous/curie/issues) page.

## License

Curie is released under the Apache 2.0 License. See `LICENSE` for more details.
