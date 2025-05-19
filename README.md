# Curie: A Research Experimentation Agent 
<!-- # Curie: Automate Rigorous Scientific Experimentation -->

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
- 🚀 Automated Experimentation – From hypothesis formulation, experiment implementation, experiment execution, result analysis and finding reflection.
- 📊 Rigor Enhancement - Built-in verification modules enforce methodical procedure, agent reliability and reproducibility.
- 🔬 Broad Applicability – Supports ML research, system analysis, and scientific discovery.
<!-- - 📖 Experimentation Benchmark - Provide 46 questions from 4 Computer Science domains, based on influential papers and open-source projects (`benchmark/experimentation_bench`). -->

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

3. Put your [LLM API credentials](https://github.com/BerriAI/litellm) under `curie/setup/env.sh`. Example: 

```
export MODEL="gpt-4o" 
export OPENAI_API_KEY="sk-xxx" 
```

4. Build the container image. This will take a few minutes. 
```bash
pip install -e .
docker images -q exp-agent-image | xargs -r docker rmi -f # remove any existing conflict image
cd curie && docker build --no-cache --progress=plain -t exp-agent-image -f ExpDockerfile_default .. && cd -
```

## Quick Start
Use the following command to input your research question or problem statement: `python3 -m curie.main -q "<Your research question>"`.

### **Example 1**: [You Have a Single Question Needed to be Verified](./docs/quick_start.md).

Q: I want to understand the Sorting Algorithm Efficiency.

A: Simply input your question to Curie:

```bash
python3 -m curie.main \
  -q "How does the choice of sorting algorithm impact runtime performance across different \
  input distributions (random, nearly sorted, reverse sorted)?" 
```
- **Estimated runtime**: ~5 minutes
- **Auto-generated Experiment report**: Available [here](./docs/example_logs/research_sorting_efficiency_20250310015235.md).
- **Curie log**: Available [here](./docs/example_logs/research_sorting_efficiency_20250310015235.log)
- **Logs and Reproducibilty**:
  - Real-time logs are streamed to the console.
  - Experiment report are stored in `logs/research_<ID>.md`  
  - The full experimentation process (script to reproduce results, generated code and experiment results) is saved in `workspace/research_<ID>/`.

### Example 2: You Have a Dataset and Want to Gain Insight from It

Q: I have a dataset and some starter code,and I want to train/deloy ML models to achieve my goals

A: Simply provide your dataset, codebase and question to Curie:

```bash
python3 -m curie.main -q '[Example]: How to improve my prediction accuracy on my datastet. \
                      Checkout <paper.pdf> for the background information.' \
                      --task_config curie/configs/mle.json \
                      --dataset_dir <abs_path_to_your_dataset> \
                      --workspace_name <abs_path_to_your_codebase [optional]> 
```  
- Check out some [examples](./benchmark/mle_bench/dog-breed-identification/) from [MLE-Bench](https://github.com/openai/mle-bench).
  - [Predict the dog breed](./benchmark/mle_bench/dog-breed-identification/)
  - [Identify melanoma in images of skin lesions](./benchmark/mle_bench/siim-isic-melanoma-classification/)
  - [Predict the severity level of diabetic retinopathy based on retinal images](./benchmark/mle_bench/aptos2019-blindness-detection/)



Check out more **Machine Learning Use Cases** [here](https://github.com/Just-Curieous/Curie-Use-Cases). 



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

## Contact Us

Have questions or need assistance with Curie? We're here to help - [schedule a meeting with our team](https://calendly.com/amberljc/30min)
