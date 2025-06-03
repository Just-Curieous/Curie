# Curie: A Research Experimentation Agent 
<!-- # Curie: Automate Rigorous Scientific Experimentation -->

[![arXiv](https://img.shields.io/badge/arXiv-2502.16069-b31b1b.svg)](https://arxiv.org/abs/2502.16069)
[![License](https://img.shields.io/badge/license-Apache_2.0-blue)](LICENSE)
[![PyPI](https://img.shields.io/badge/PyPI-Install-blue)](https://pypi.org/project/curie-ai/)
[![Slack](https://img.shields.io/badge/Slack-Join%20Community-4A154B?logo=slack)](https://join.slack.com/t/just-curieous/shared_invite/zt-313elxhhy-hpEK5r9kX9Xv1Pfxzt9CJQ)
[![Demo](https://img.shields.io/badge/Demo-Live-green)](http://44.202.70.8:5000/)
[![Blog](https://img.shields.io/badge/Blog-Read%20More-orange)](https://www.just-curieous.com/)


Curie is the first AI-agent framework designed for automated and rigorous scientific experimentation. 
Curie helps answer your curiosity through end-to-end experimentation automation, ensuring that every step—from hypothesis formulation to result interpretation—is conducted with precision, reliability, and reproducibility.
Our mission is to empower scientists to move research at the speed of thought.

<p align="center">
  <img src="./docs/static/img/curie-overview.png" width="600px"/>
</p>

**Key Features**
- 🚀 Automated Experimentation – From hypothesis formulation, experiment implementation, experiment execution, result analysis and finding reflection.
- 📊 Rigor Enhancement - Built-in verification modules enforce methodical procedure, agent reliability and reproducibility.
- 🔬 Broad Applicability – Supports ML research, system analysis, and scientific discovery.


## Table of Contents 
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Tutorial](#tutorial)
- [Demo](#demo)

## Installation
**Prerequisite: Install Docker**
   ```bash
   # Install Docker from https://docs.docker.com/engine/install/ubuntu/
   sudo chmod 666 /var/run/docker.sock
   docker ps  # Verify Docker installation
   ```

#### Option 1: Quick Install via `pip` 
```bash
pip install curie-ai
```

#### Option 2: Manual [Installation](./docs/installation.md) for Developers
<!-- 1. Clone and setup:
   ```bash
   git clone https://github.com/Just-Curieous/Curie.git
   cd Curie
   ```

2. Configure API credentials in `curie/setup/env.sh`:
   ```bash
   export MODEL="claude-3-7-sonnet-20250219" 
   export ANTHROPIC_API_KEY="your-anthropic-key"
   ```

3. Build container:
   ```bash
   pip install -e .
   docker images -q exp-agent-image | xargs -r docker rmi -f
   cd curie && docker build --no-cache --progress=plain -t exp-agent-image -f ExpDockerfile_default .. && cd -
   ``` -->

## Quick Start
*It's recommended to use `tmux` or a similar terminal multiplexer before running Curie, as experiments can take several minutes depending on the task and budget.*


### (Simple) Example 1: [You Have a Single Question that Needs to be Verified](./docs/quick_start.md).

👩‍🎓: I want to understand the Sorting Algorithm Efficiency.

```python
import curie
# Set up your API keys, refer to curie/setup/env.sh.example
key_dict = {
    "MODEL": "claude-3-7-sonnet-20250219",
    "ANTHROPIC_API_KEY": "your-anthropic-key"
}

result = curie.experiment(api_keys=key_dict, question="How does the choice of sorting algorithm impact runtime performance across different input distributions?")
```
- **Auto-generated Experiment report**: Available [ `logs/research_<ID>.md`](./docs/example_logs/sorting_example/research_1747978647_20250523013727_iter1.md).
- **Reproducibilty and Logs**:
  - The full experimentation process (script to reproduce results, generated code and experiment results) is saved in `workspace/research_<ID>/`.
  - Real-time logs are streamed to the console and stored in file `research_*.log`.

### Example 2: Find Optimal ML Strategies for Noisy Cancer Data.
👩‍🎓: I want to find the most robust ML methods for my noisy data.

```python 
import curie
key_dict = {
    "MODEL": "claude-3-7-sonnet-20250219",
    "ANTHROPIC_API_KEY": "your-anthropic-key"
}

result = curie.experiment(api_keys=key_dict, question="Are ensemble methods (e.g., Random Forests, Gradient Boosting) more robust to added noise in the Breast Cancer Wisconsin dataset compared to linear models like Logistic Regression for a binary classification task?")
```


### (Advanced) Example 3: You Have a Dataset and Want to Gain Insight from It

👨‍🎓: I have a dataset and some starter code,and I want to train/deloy ML models to achieve specific goals.

```python 
result = curie.experiment(
    api_keys=key_dict,
    question="E.g. How to improve my prediction accuracy on my dataset.",
    workspace_name="[optional] ABS_PATH_STARTERCODE_DIR",
    dataset_dir="ABS_PATH_DATASET_DIR"
)
```  
- Check out some [examples](./benchmark/mle_bench/) from [MLE-Bench](https://github.com/openai/mle-bench).
  - [Predict the dog breed](./benchmark/mle_bench/dog-breed-identification/)
  - [Identify melanoma in images of skin lesions](./benchmark/mle_bench/siim-isic-melanoma-classification/)
  - [Predict the severity level of diabetic retinopathy based on retinal images](./benchmark/mle_bench/aptos2019-blindness-detection/)
  - [Histopathologic Cancer Detection](./benchmark/mle_bench/histopathologic-cancer-detection/)
  - [Predict the stock price ranking](https://github.com/Just-Curieous/Curie-Use-Cases/tree/main/stock_prediction)
- **Sample auto-generated experiment [report](./benchmark/mle_bench/aptos2019-blindness-detection/report.pdf)**:


<!-- First row with 3 images -->
<p align="center">
<img src="benchmark/mle_bench/aptos2019-blindness-detection/report-fig/output-1.png" width="23%"/>
<img src="benchmark/mle_bench/aptos2019-blindness-detection/report-fig/output-2.png" width="23%"/>
<img src="benchmark/mle_bench/aptos2019-blindness-detection/report-fig/output-3.png" width="23%"/>
<img src="benchmark/mle_bench/aptos2019-blindness-detection/report-fig/output-4.png" width="23%"/>
</p> 

Check out more **Machine Learning Use Cases** [here](https://github.com/Just-Curieous/Curie-Use-Cases). 

## Tutorial 
(Incoming...)

## Demo Video 

<div align="center">

[![Demo Video](https://img.youtube.com/vi/Qn_T5mm2OP4/0.jpg)](https://www.youtube.com/watch?v=Qn_T5mm2OP4)

</div>

<p align="center">
  <em>Curie Overview & Demo.</em>
</p>

## Community and Support

- [GitHub Issues](https://github.com/Just-Curieous/curie/issues) - Report bugs or request features
- [Schedule a Meeting with Us](https://calendly.com/amberljc/30min) - Get help from our team
- [Join our Slack Community](https://join.slack.com/t/just-curieous/shared_invite/zt-313elxhhy-hpEK5r9kX9Xv1Pfxzt9CJQ)

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.