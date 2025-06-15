# Curie Tutorial for Beginners:

Welcome to Curie! This tutorial will walk you through the complete process of using Curie for automated machine learning (ML) experimentation. 
We'll cover everything from setup to running experiments and analyzing results.

## ⚙️ Installation

1. First, install Docker from [here](https://docs.docker.com/engine/install/ubuntu/) if you haven't already.
   ```bash
   sudo chmod 666 /var/run/docker.sock
   docker ps  # Verify Docker installation
   ```
2. Install Curie using pip:
    ```bash
    pip install curie-ai
    ```

3. Verify the installation:
    ```bash
    python -c "import curie; print(curie.__version__)"
    ```

## 🔑 Setting Up API Keys
We support all kinds of API key providers, but please let us know if you encounter API setup issues [here](https://github.com/Just-Curieous/Curie/issues).
```python
key_dict = {
    "MODEL": "claude-3-7-sonnet-20250219",
    "ANTHROPIC_API_KEY": "your-anthropic-key",
    # "MODEL": 'openai/gpt-4o-mini',
    # OPENAI_API_KEY: "your-openai-key",
}
```

## 📊 Get your dataset ready

Here we just download MNIST dataset to `/data` as an example:
```
sudo mkdir -p /data 
wget https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz
wget https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz
wget https://raw.githubusercontent.com/fgnt/mnist/master/t10k-images-idx3-ubyte.gz
wget https://raw.githubusercontent.com/fgnt/mnist/master/t10k-labels-idx1-ubyte.gz 
sudo mv *gz /data
sudo gunzip /data/*.gz 
```

## ❓ Get your question ready

You should provide **as many details as** you can to your question. 
```python
import curie
result = curie.experiment(
    api_keys=key_dict,
    question="What is the best model among Logistic Regression, MLP, and CNN for my MNIST dataset?",
    dataset_dir="/data" # absolute path to the dataset
)
```


## 🚀 (Optional) More Advanced Usage

1. **Work with your starter code**: 

    - *Starter code*: You can prepare the starter code to let Curie work on top of. This is very important if your dataset needs specialize data loader.

    ```bash
    /abs/path/starter_code/
    ├── train.py # Python script for training
    └── description.md # instructions that highlight how to run your experiments. 
                       # Please explicitly name it as `description.md` or `README.md`
    ```

    ```python
    import curie
    result = curie.experiment(
        api_keys=key_dict,
        question="Among Logistic Regression, MLP, and CNN, which model achieves the highest prediction accuracy on my MNIST dataset?,
        dataset_dir="`data_loader.py` will assit you load the dataset.",
        codebase_dir="/abs/path/starter_code/", # Change this to the path of your starter code
        code_instructions="",
        max_global_steps=50, # control your compute budget
    )
    ```

2. **Provide with your research paper**: 
To provide more context for Curie, you can mention the necessary paper (`txt`, `pdf`, ...) in the question. 
Please put your paper under the same directory of your starter code. 
- *If you are using `AWS bedrock` API, please give permission to model `'amazon.titan-embed-text-v2:0'`*
    ```bash
    /abs/path/starter_code/
    ├── train.py # Python script for training
    └── paper.pdf # Research paper detailing the approach
    ```

    ```python
    import curie
    result = curie.experiment(
        api_keys=key_dict,
        question="Refer to the evaluation setup in `paper.pdf`. Among Logistic Regression, MLP, and CNN, which model achieves the highest prediction accuracy on my MNIST dataset?",
        dataset_dir="/data",
        codebase_dir="/abs/path/starter_code", # Change this to the path of your starter code
        max_global_steps=50, # control your compute budget
    )
    ```

3. **Provide with your own complex environment**
You can provide your own environment by providing an environment requirements file or pre-configuring a `micromamba`/`miniconda`. This allows you to specify exact package versions and dependencies needed for your research. This is important to save time for Curie to figure out the dependencies by herself.

    - **Option 1**: Put your environment requirements file `requirements.txt` under the `codebase_dir`:
        ```bash
        /abs/path/starter_code/
        ├── train.py # Python script for training
        └── requirements.txt # including the `package==version`
        ```
        Or you can specify separately:
        ```python
        result = curie.experiment(api_keys=key_dict, 
                                question="How does the choice of sorting algorithm impact runtime performance across different input distributions?", 
                                env_requirements='/abs/path/requirements.txt')
        ```
    - **Option 2**: You can pre-configure your environment and name it as `venv` and put under your starter_code:
        ```bash
        starter_code/
        ├── venv/ # exactly named as `venv`  
        └── ... # the rest of your codebase
        ```

4. **Generate a experiment report in the middle of Curie's experimentation process**

If you’d like to monitor progress partway through Curie’s experimentation—or if the experiment wasn’t run end-to-end—you can still generate a report from the available data:

```python
curie.generate_report(api_keys=key_dict,
                    log_dir='/abs/path/logs/research_20250605231023_iter1/',
                    workspace_dir='/abs/path/workspace/')
```

5. **Customize the agent to your workload.**
Each agent and experiment stage is coupled with a system prompt, which you can fine-tune in order to let Curie understand your context better. 

    ```python
    import curie

    task_config = {
        "supervisor_system_prompt_filename": "/home/ubuntu/prompt.txt", #  
        # "control_worker_system_prompt_filename": "/path/to/your/new/prompt",
        # "patcher_system_prompt_filename": "/path/to/your/new/prompt",
        # "llm_verifier_system_prompt_filename": "/path/to/your/new/prompt", 
        # "coding_prompt_filename": "/path/to/your/new/prompt", 
        # "worker_system_prompt_filename": "/path/to/your/new/prompt", 
    }
    result = curie.experiment(
        api_keys=key_dict,
        question="Among Logistic Regression, MLP, and CNN, which model achieves the highest prediction accuracy on my MNIST dataset?",
        dataset_dir="/data",
        max_global_steps=50, # control your compute budget
        task_config=task_config,
    )
    ```