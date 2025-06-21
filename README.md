# Curie: A Research Experimentation Agent 
<!-- # Curie: Automate Rigorous Scientific Experimentation -->

[![arXiv](https://img.shields.io/badge/arXiv-2502.16069-b31b1b.svg)](https://arxiv.org/abs/2502.16069)
[![License](https://img.shields.io/badge/license-Apache_2.0-blue)](LICENSE)
[![PyPI](https://img.shields.io/badge/PyPI-Install-blue)](https://pypi.org/project/curie-ai/)
[![Slack](https://img.shields.io/badge/Slack-Join%20Community-4A154B?logo=slack)](https://join.slack.com/t/just-curieous/shared_invite/zt-37iz7pnjo-TIrzg9aBwYTCoaMTl~X1hA)
[![Demo](https://img.shields.io/badge/Demo-Live-green)](http://44.202.70.8:5000/)
[![Blog](https://img.shields.io/badge/Blog-Read%20More-orange)](https://www.just-curieous.com/)


Curie is the first AI-agent framework designed for automated and rigorous scientific experimentation. 
Curie helps answer your curiosity through end-to-end experimentation automation, ensuring that every step—from hypothesis formulation to result interpretation—is conducted with precision, reliability, and reproducibility.
Our mission is to empower scientists to move research at the speed of thought.

<p align="center">
  <!-- <img src="./docs/static/img/curie-overview.png" width="600px"/> -->
  <img src="./docs/static/img/research-lifecyle.png" width="600px"/>
</p> 
<p align="center">Curie’s Role in the Scientific Research Lifecycle</p>


## 🗞️ News
- **[2025/06]** We published **EXP-Bench**: Can AI Conduct AI Research Experiments? → [📄 Paper](https://arxiv.org/abs/2505.24785) | [🗂️ Dataset](https://huggingface.co/datasets/Just-Curieous/EXP-Bench) | [📰 Blog](https://www.just-curieous.com/machine-learning/research/2025-06-11-exp-bench-can-ai-conduct-ai-research-experiments.html)
- **[2025/05]** We launched an **AutoML feature** to help researcher find the optimal ML solution → [📢 Blog](https://www.just-curieous.com/machine-learning/research/2025-05-27-automl-co-scientist.html)
- **[2025/02]** We published **Curie**: Toward Rigorous and Automated Scientific Experimentation with AI Agents → [📄 Paper](https://arxiv.org/abs/2502.16069) | [📰 Blog](https://www.just-curieous.com/)


## Key Features
- 🚀 Automated Experimentation – From hypothesis formulation, experiment implementation, experiment execution, result analysis and finding reflection.
- 📊 Rigor Enhancement - Built-in verification modules enforce methodical procedure, agent reliability and reproducibility.
- 🔬 Broad Applicability – Supports [**ML Engineering**](https://www.just-curieous.com/machine-learning/research/2025-05-27-automl-co-scientist.html), [system analysis](./docs/example_logs/sorting_example/research_1748830453_20250602021413_iter1.md), and scientific discovery.
- 💻 Use Your Starter Code – Supports working on arbitrary user's starter code.
- 📂 Bring Your Own Dataset – Supports working on arbitrary user's datasets.
- 🧾 Automatic, Insightful Reporting - See a sample report [here](./benchmark/mle_bench/histopathologic-cancer-detection/histopathologic-cancer-detection_20250519225201_iter1.md)


## Table of Contents 
- [⚙️ Installation](#-installation)
- [⚡ Quick Start](#-quick-start)
- [📚 Tutorial](#-tutorial)
- [🎬 Demo](#-demo-video)


## ⚙️ Installation
**Prerequisite: Install Docker** from [here](https://docs.docker.com/engine/install/ubuntu/)
   ```bash
   sudo chmod 666 /var/run/docker.sock
   docker ps  # Verify Docker installation
   ```

#### Option 1: Quick Install via `pip` 
```bash
pip install curie-ai
```

#### Option 2: Manual [Installation](./docs/installation.md) for Developers


## ⚡ Quick Start
- *It's recommended to use `tmux` or a similar terminal multiplexer before running Curie, as experiments can take several minutes depending on the task and budget.*

- *Do not use Jupyter Notebook.*

### (Simple) Example 1: You Have a Single Question that Needs to be Verified.

👩‍🎓: I want to understand the Sorting Algorithm Efficiency.

```python
import curie
# Set up your API keys, refer to curie/setup/env.sh.example
key_dict = {
    "MODEL": "claude-3-7-sonnet-20250219",
    "ANTHROPIC_API_KEY": "your-anthropic-key"
}

result = curie.experiment(api_keys=key_dict, 
                          question="How does the choice of sorting algorithm impact runtime performance across different input distributions?",
                          max_global_steps=10)

```
* 🧾 **Auto-Generated Experiment Report**: [`logs/research_<ID>.md`](./docs/example_logs/sorting_example/research_1748830453_20250602021413_iter1.md).

- 📊 Experiment Result **Notebook**: `logs/research_*_all_results.txt`.

- 🪵 The **Experimentation Process** (generated *script* generated *code* to reproduce experiment *results*): `workspace/research_<ID>/`.


### Example 2: Find Optimal ML Strategies for Noisy Cancer Data.
👩‍🎓: I want to find the most robust ML methods for my noisy data.

```python 
result = curie.experiment(api_keys=key_dict, 
                          question="Are ensemble methods (e.g., Random Forests, Gradient Boosting) more robust to added noise in the Breast Cancer Wisconsin dataset compared to linear models like Logistic Regression for a binary classification task?")
```

- 🧾 [Auto-generated Experiment **Report**](./docs/example_logs/noise_example/default_research_1748932907_20250603064147_iter1.md)

- 📊 [Experiment Result **Notebook**](./docs/example_logs/noise_example/default_research_1748932907_20250603064147_iter1_all_results.txt)

### (Advanced) Example 3: You Have a Dataset and Want to Gain Insight from It

👨‍🎓: I have a dataset and some starter code,and I want to train/deloy ML models to achieve specific goals. (*GPU is recommended for ML training tasks.*)

```python 
result = curie.experiment(
    api_keys=key_dict,
    question="E.g. How to improve my prediction accuracy on my dataset.",
    dataset_dir="/abs/path/to/your/dataset",
    codebase_dir="[Optional] /abs/path/to/your/code",
    env_requirements="[Optional] /abs/path/to/requirements.txt",
)
```  

<p align="center">
  <img src="./docs/static/img/exp-bench-mle-curie.drawio.png" width="600px"/>
</p> 
<p align="center">Curie AutoML Feature Overview.</p>

- Check out how Curie is able to find optimal ML soltuions in these [examples](./benchmark/mle_bench/) from [MLE-Bench](https://github.com/openai/mle-bench).
  - [Predict the dog breed](./benchmark/mle_bench/dog-breed-identification/)
  - [Identify melanoma in images of skin lesions](./benchmark/mle_bench/siim-isic-melanoma-classification/)
  - [Predict the severity level of diabetic retinopathy based on retinal images](./benchmark/mle_bench/aptos2019-blindness-detection/)
  - [Histopathologic Cancer Detection](./benchmark/mle_bench/histopathologic-cancer-detection/)
  - [Predict the stock price ranking](https://github.com/Just-Curieous/Curie-Use-Cases/tree/main/stock_prediction)
- **Sample Curie-Generated Experiment [Report](./benchmark/mle_bench/aptos2019-blindness-detection/report.pdf)**:

<p align="center">
<img src="benchmark/mle_bench/aptos2019-blindness-detection/report-fig/output-1.png" width="23%"/>
<img src="benchmark/mle_bench/aptos2019-blindness-detection/report-fig/output-2.png" width="23%"/>
<img src="benchmark/mle_bench/aptos2019-blindness-detection/report-fig/output-3.png" width="23%"/>
<img src="benchmark/mle_bench/aptos2019-blindness-detection/report-fig/output-4.png" width="23%"/>
</p> 

Check out more **Machine Learning Use Cases** [here](https://github.com/Just-Curieous/Curie-Use-Cases). 

## 📚 Tutorial 
- [How to use Curie to answer your curiosity (with a concrete example)?](./docs/tutorial-beginer.md)

## 🎬 Demo Video 

<div align="center">

[![Demo Video](https://img.youtube.com/vi/Qn_T5mm2OP4/0.jpg)](https://www.youtube.com/watch?v=Qn_T5mm2OP4)

</div>

<p align="center">
  <em>Curie Overview & Demo.</em>
</p>



## 📜 Citation  
If you use Curie in a research paper, please cite our work:

```bib
@article{kon2025expbenchaiconductai,
      title={EXP-Bench: Can AI Conduct AI Research Experiments?}, 
      author={Patrick Tser Jern Kon and Jiachen Liu and Xinyi Zhu and Qiuyi Ding and Jingjia Peng and Jiarong Xing and Yibo Huang and Yiming Qiu and Jayanth Srinivasa and Myungjin Lee and Mosharaf Chowdhury and Matei Zaharia and Ang Chen},
      journal={arXiv preprint 2505.24785}
      year={2025},
}
```

```bib
@article{kon2025curie,
  title={Curie: Toward rigorous and automated scientific experimentation with ai agents},
  author={Kon, Patrick Tser Jern and Liu, Jiachen and Ding, Qiuyi and Qiu, Yiming and Yang, Zhenning and Huang, Yibo and Srinivasa, Jayanth and Lee, Myungjin and Chowdhury, Mosharaf and Chen, Ang},
  journal={arXiv preprint arXiv:2502.16069},
  year={2025}
}
```


## Community and Support

- [GitHub Issues](https://github.com/Just-Curieous/curie/issues) - Report bugs or request features
- [Schedule a Meeting with Us](https://calendly.com/amberljc/30min) - Get help from our team
- [Join our Slack Community](https://join.slack.com/t/just-curieous/shared_invite/zt-313elxhhy-hpEK5r9kX9Xv1Pfxzt9CJQ)

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.