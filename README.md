# Curie: Automate Rigorous Scientific Experimentation

![Documentation](https://img.shields.io/badge/docs-Just--Curieous.github.io-blue)
![Discord](https://img.shields.io/discord/discord-id?label=Discord&logo=discord&link=https://discord.gg/uCEbmG7EKU)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)

Curie is the first AI-agent framework designed for automated and rigorous scientific experimentation. As a first step in this direction, Curie focuses on the kinds of experimentation a human developer would conduct—any process involving code execution, from running simulations to testing hypotheses in real-time environments.

![Curie Overview](./docs/static/img/curie-overview.png)

## Key Features
On a high-level, Curie is composed of two types of LLM-based agents (an Architect and a host of Technicians). Sandwiched between them is Curie's **Experimental Rigor Engine** that injects rigor throughout the experimental process. It consists of 3 main modules:

- **Intra-Agent Rigor Module**: Ensures the reliability of experiments by maintaining consistency and accuracy within individual agents.
- **Inter-Agent Rigor Module**: Provides methodical control across multiple agents, ensuring systematic and controlled experimentation.
- **Experiment Knowledge Module**: Enhances the interpretability of results, making it easier to derive meaningful insights from experiments, while enabling seamless collaboration between all components during large-scale experiments. 

Curie also consists of a novel **Experimentation Benchmark** composed of 46 questions across 4 Computer Science domains,  derived from influential research papers, and widely adopted open-source projects. 
- Compared to the strongest baseline tested, we achieve a 3.4× improvement in correctly answering experimental questions.

## Codebase Overview

- **analyze_log**: Contains scripts for analyzing and summarizing experiment logs.
- **baselines**: Includes baseline models and methods for comparison.
- **benchmark**: Provides a set of benchmarks across various domains like cloud infrastructure, machine learning training, and vector indexing.
- **curie**: The core framework containing modules for experiment setup, execution, and verification.
- **docs**: Documentation files for installation and usage instructions.
- **starter_file**: Contains starter files and examples to help you get started quickly.

<!-- ## Benchmark and Evaluation

Curie has been evaluated using a novel benchmark consisting of 46 questions across four computer science domains. The results demonstrate a significant improvement in experiment design correctness, execution reproducibility, and conclusion accuracy compared to existing baselines. -->

## Installation

1. Install Docker: Follow the instructions at [Docker Installation](https://docs.docker.com/engine/install/ubuntu/).

2. Build the container image. If changes have been made, delete the current mounted volume (after backing up necessary data) and rebuild the container image.

   ```bash
   cd Curie
   git submodule update --init --recursive 
   pip install -e .
   cd curie
   sudo docker build --no-cache --progress=plain -t exp-agent-image -f ExpDockerfile_default ..
   ```

## Quick Start

### Add your LLM API keys:

Set up your environment variables for the LLM API:

```bash
export MODEL="gpt-4o"
export API_VERSION="2024-06-01"
export OPENAI_API_KEY= 
export OPENAI_ORGANIZATION= 
export OPENAI_API_BASE= 
```

### Specify your research problem

Configure `curie/configs/base_config.json` to specify your research problem.

### Start experiments:

Change the execution path to its parent and run:

```bash
cd Curie

python3 -m curie.main --iterations 1 --question_file benchmark/llm_reasoning/q1_simple_relation.txt --task_config curie/configs/llm_reasoning_config.json
```
 

## Contribute

Any contributions are welcome! Please read our `CONTRIBUTING.md` to learn how you can help improve Curie.

## Community and Support

Join our community on [Discord](https://discord.gg/yourdiscordlink) to connect with other users and developers. For any issues or feature requests, please open an issue on our [GitHub Issues](https://github.com/Just-Curieous/curie/issues) page.

## License

Curie is released under the Apache 2.0 License. See `LICENSE` for more details.
