# Development

## Curie Usage

Curie provides flexible ways to initiate and manage your research experiments.

### 1. Using the Python API

You can directly integrate Curie into your Python scripts. This is useful for complex workflows, custom pre/post-processing, or when embedding Curie within a larger application.

```python
import curie

# Set up your API keys
key_dict = {
    "MODEL": "claude-3-7-sonnet-20250219",
    "ANTHROPIC_API_KEY": "your-anthropic-key"
}

# Run experiments to understand the Sorting Algorithm Efficiency.
result = curie.experiment(
    api_keys=key_dict,
    question="E.g. How to improve my prediction accuracy on my dataset.",
    workspace_name="[optional] abs_path_to_codebase_dir",
    dataset_dir="abs_path_to_dataset_dir"
)
``` 
### 2. Using the Command Line Interface (CLI)

The CLI is a convenient way to run experiments directly from your terminal, suitable for quick tests or integration into shell scripts.

- First configure API credentials in `curie/setup/env.sh`:
```bash
export MODEL="claude-3-7-sonnet-20250219" 
export ANTHROPIC_API_KEY="your-anthropic-key"
```

- Run Curie on your task

```bash
cd Curie
python3 -m curie.main \
  --question 'How can I improve the prediction accuracy of a classification model on my provided dataset?' \
  --task_config curie/configs/mle.json \
  --dataset_dir /absolute/path/to/your/dataset \
  --workspace_name /absolute/path/to/your/project/codebase # This is optional
```