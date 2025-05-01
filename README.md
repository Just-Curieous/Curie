# Curie: Automate Rigorous Scientific Experimentation

[![arXiv](https://img.shields.io/badge/arXiv-2502.16069-b31b1b.svg)](https://arxiv.org/abs/2502.16069)
[![Slack](https://img.shields.io/badge/Slack-Join%20Community-4A154B?logo=slack)](https://join.slack.com/t/just-curieous/shared_invite/zt-313elxhhy-hpEK5r9kX9Xv1Pfxzt9CJQ)
[![Demo](https://img.shields.io/badge/Demo-Live-green)](http://44.202.70.8:5000/)
[![Blog](https://img.shields.io/badge/Blog-Read%20More-orange)](https://www.just-curieous.com/)
[![License](https://img.shields.io/badge/license-Apache_2.0-blue)](LICENSE)


Curie is the first AI-agent framework designed for automated and rigorous scientific experimentation. 
Curie helps answer your curiosity through end-to-end experimentation automation, ensuring that every step—from hypothesis formulation to result interpretation—is conducted with precision, reliability, and reproducibility.

<p align="center">
  <img src="./docs/static/img/curie-overview.png" width="600px"/>
</p>

**Key Features**
- 🚀 Automated Experimentation – End-to-end workflow management: hypothesis formulation, experiment setup, experiment execution, result analysis and finding reflection.
- 📊 Rigor Enhancement - Built-in verification modules enforce methodical procedure, reliability and interpretability.
- 🔬 Broad Applicability – Supports ML research, system analysis, and scientific discovery.
- 📖 Experimentation Benchmark - Provide 46 questions from 4 Computer Science domains, based on influential papers and open-source projects (`benchmark/experimentation_bench`).

## Table of Contents 
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Use Cases](#use-cases)
- [Tutorial](#tutorial)
- [Customize Your Experiment Agents](#customize-your-experimentation-agents) 

## [Installation](./docs/installation.md)

1. Install docker: https://docs.docker.com/engine/install/ubuntu/. 
  - Grant permission to docker via `sudo chmod 666 /var/run/docker.sock`. 
  - Run `docker ps` to check that permission has been granted with the Docker daemon.

2. Clone the repository:
```
git clone https://github.com/Just-Curieous/Curie.git
cd Curie
```

3. Put your LLM API credentials under `curie/setup/env.sh`. Example: 

```
export MODEL="gpt-4o" 
export OPENAI_API_KEY="sk-xxx" 
```

4. Build the container image. This will take a few minutes. Note: you may need to setup a virtual environment before running pip install.

```bash
pip install -e .
docker images -q exp-agent-image | xargs -r docker rmi -f # remove any existing conflict image
cd curie && docker build --no-cache --progress=plain -t exp-agent-image -f ExpDockerfile_default .. && cd -
```

## Quick Start
Use the following command to input your research question or problem statement: `python3 -m curie.main -q "<Your research question>"`.

### **Example 1**: Understanding Sorting Algorithm Efficiency

```bash
python3 -m curie.main \
  -q "How does the choice of sorting algorithm impact runtime performance across different \
  input distributions (random, nearly sorted, reverse sorted)?" --report
```
- **Estimated runtime**: ~5 minutes
- **Sample log file**: Available [here](./docs/example_logs/research_sorting_efficiency_20250310015235.log)
- **Experiment report**: Available [here](./docs/example_logs/research_sorting_efficiency_20250310015235.md).
- **Logs and Reproducibilty**:
  - Real-time logs are streamed to the console.
  - Experiment logs and experiment report (`--report`) are stored in `logs/research_<ID>`  
  - The full experimentation process (code, script and real results) is saved in `workspace/research_<ID>/`.

### **Example 2**: How does the choice of activation function (e.g., ReLU, sigmoid, tanh) impact the model training convergence rate?

```bash
python3 -m curie.main -f benchmark/junior_ml_engineer_bench/q1_activation_func.txt --report
```
- **Sample log file**: Available [here](./docs/example_logs/mle_activation_func_20250326.log)
- **Sample report file**: Available [here](./docs/example_logs/mle_activation_func_20250326.md)

### Example 3: General Machine Learning Questions with Your Dataset

If you have a dataset but are unsure how to start training/deloying your ML models to achieve your goals, simply provide your dataset and question to Curie:
```bash
python3 -m curie.main -q 'Example: How to improve my prediction accuracy on my datastet' \
                      --task_config curie/configs/mle.json \
                      --dataset_dir <path to your dataset> \
                      --report
```  
- You can include your own starter code by adding the argument `--workspace_name <path_to_your_workspace>`.
- Check out an [example](./benchmark/mle_bench/dog-breed-identification/) from [MLE-Bench](https://github.com/openai/mle-bench).

Check out more [computational questions](./docs/quick_start.md), as well as [Machine Learning questions](/benchmark/junior_ml_engineer_bench/) and [Machine Learning Systems questions](/benchmark/junior_mlsys_engineer_bench/).


## Tutorial
- [How to let Curie work on your own starter files?](./docs/tutorial_with_your_own_starter_file.md)
- [How to reproduce the results in `Large Language Monkeys'. ](./docs/tutorial-large-language-monkey.md)


## Use Cases
Curie is designed for scientific discovery across multiple domains:

- 🔬 Machine Learning & AI Research – Hyperparameter tuning and algorithm behavior
  - [How does the optimal learning rate change with the increase of model size?](https://github.com/microsoft/mup)
  - [How does repeated sampling in LLM inference affect the quality of response?](https://arxiv.org/abs/2407.21787)
- 💻 System Performance Analysis – Benchmarking systems, optimizing configurations, investigating system trade-offs.
  - [What configurations affects the energy consumption of LLM serving?](https://ml.energy/leaderboard/?__theme=light)
  - [How does the request bursty arrival pattern affects the user experience in LLM serving?](https://arxiv.org/abs/2404.16283)
- 🧪 Algorithmic & Scientific Discovery – Validating hypotheses, automating computational simulations.

<p align="center">
  <img src="./docs/static/img/case_study.png" width="1000px"/>
</p>

## Community and Support

For any issues or feature requests, please open an issue on our [GitHub Issues](https://github.com/Just-Curieous/curie/issues) page.

## License

Curie is released under the Apache 2.0 License. See `LICENSE` for more details.
