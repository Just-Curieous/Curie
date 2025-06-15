import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import io
import json
import re
import ast
import os
import re
import toml
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any

def get_all_price_per_1k_tokens() -> Dict[str, Dict[str, float]]:
    return {
        "gpt-4o": {"input": 0.0025, "output": 0.01}, 
        "gpt-4o-mini": {"input": 0.00015, "output": 0.000075},
        "anthropic.claude-3-7-sonnet-20250219-v1:0": {"input": 0.003, "output": 0.015},
    }

def get_model_context_length() -> int:
    """Get the context length for the current model."""
    # FIXME: add more models as needed
    context_length_dict = {
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "anthropic.claude-3-7-sonnet-20250219-v1:0": 200000,
    }
    model_name = get_model_name()
    return context_length_dict.get(model_name, 30000)

def get_input_price_per_token() -> float:
    """Get the price per token for input text."""
    model_name = get_model_name()
    return get_all_price_per_1k_tokens()[model_name]["input"] / 1000

def get_output_price_per_token() -> float:
    """Get the price per token for output text."""
    model_name = get_model_name()
    return get_all_price_per_1k_tokens()[model_name]["output"] / 1000

def get_model_name() -> str:
    """Strip provider prefix if present (e.g., "openai/gpt-4" -> "gpt-4")"""
    current_model = os.environ.get("MODEL", "gpt-4o")
    # Strip provider prefix if present (e.g., "openai/gpt-4" -> "gpt-4")
    model_name = current_model.split('/')[-1]
    if "claude" in model_name and "us." in model_name: # example: "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        # Remove "us." prefix:
        model_name = model_name.split("us.")[1]
    return model_name

def extract_plan_id(prompt: str) -> str:
    """
    Extracts the plan ID from the given prompt.

    Args:
        prompt (str): The input text containing a plan ID.

    Returns:
        str: The extracted plan ID if found, else an empty string.
    """
    # Regular expression to match UUID-like patterns (plan_id format)
    pattern = r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"

    # Search for the pattern in the prompt
    match = re.search(pattern, prompt)

    if match:
        return True
    else:
        return False
    
    # # Return the matched plan ID if found, else return an empty string
    # return match.group(0) if match else ""

def extract_partition_name(prompt: str) -> str:
    """
    Extracts the partition name from a given prompt.

    Args:
        prompt (str): The input text that may contain a partition name.

    Returns:
        str: The extracted partition name if found, else an empty string.
    """
    # Regular expression to match partition names (e.g., 'partition_1')
    pattern = r"['\"]?(partition_\d+)['\"]?"
    
    # Search for the pattern in the prompt
    match = re.search(pattern, prompt)

    if match:
        return True
    else:
        return False
    
    # Return the matched partition name if found, else return an empty string
    # return match.group(1) if match else ""

def extract_workspace_dir(text: str) -> str:
    """
    Extracts the directory name that appears after '/workspace/' in the given text,
    excluding any surrounding single quotes.

    Args:
        text (str): The input string containing the workspace path.

    Returns:
        str: The directory name after '/workspace/', or an empty string if not found.
    """
    match = re.search(r"/workspace/([^\s'/]+)", text)  # Excludes single quotes
    if match:
        return True
    else:
        return False
    # return match.group(1) if match else ""

def check_file_exists(file_path: str) -> bool:
    """
    Checks if a file exists at the given path.

    Args:
        file_path (str): The path to the file.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    return os.path.isfile(file_path)

def print_workspace_contents():
    workspace_dir = "/workspace"

    if os.path.exists(workspace_dir):
        print(f"Contents of {workspace_dir}:")
        for root, dirs, files in os.walk(workspace_dir):
            print(f"Root: {root}")
            if dirs:
                print(f"Directories: {dirs}")
            if files:
                print(f"Files: {files}")
            print("-" * 40)  # Separator for clarity
    else:
        print(f"Directory {workspace_dir} does not exist.")

def save_langgraph_graph(graph, dst_filename) -> None:
    try:
        # Generate the graph as a PNG binary
        graph_png = graph.get_graph().draw_mermaid_png()
        # Convert binary data to an image using Matplotlib
        img = mpimg.imread(io.BytesIO(graph_png), format='png')
        plt.imshow(img)
        plt.axis('off')  # Hide axes for a cleaner display
        plt.savefig(dst_filename, dpi=300, bbox_inches='tight')  # Save the image with high quality
        plt.close()  # Close the plot to avoid overlapping when running multiple saves
    except Exception as e:
        print(f"Error displaying graph with Matplotlib: {e}")

def pretty_json(obj):
    return json.dumps(obj, sort_keys=True, indent=4, default=str)

def parse_nested(value):
    """
    Parse a value that could be a nested structure (dict, list, etc.).
    Uses ast.literal_eval for safe evaluation of Python-like literals.
    """
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value  # Return raw value if parsing fails


def extract_key_value_pairs(input_string):
    """
    Extracts all top-level key-value pairs from the input string.
    Handles nested dictionaries and lists recursively.
    """
    # Split input by top-level keys (matches <key>=<value>)
    pattern = r"(\w+)=((?:'[^']*'|\{.*?\}|\[.*?\]))"
    matches = re.finditer(pattern, input_string, re.DOTALL)

    result = {}
    last_end = 0

    for match in matches:
        key, value = match.groups()
        key = key.strip()
        value = value.strip()

        # Check for truncated values (e.g., nested dictionaries) and expand them
        if value.startswith("'") and value.endswith("'"):
            # Simple string value
            parsed_value = value[1:-1]
        elif value.startswith("{") or value.startswith("["):
            # Potential nested structure
            # Capture everything between matched braces or brackets
            balance_count = 0
            start = match.start(2)  # Start of value in the string
            for i, char in enumerate(input_string[start:], start=start):
                if char == '{' or char == '[':
                    balance_count += 1
                elif char == '}' or char == ']':
                    balance_count -= 1
                if balance_count == 0:
                    # Found the complete nested structure
                    full_value = input_string[start:i + 1]
                    parsed_value = parse_nested(full_value)
                    last_end = i + 1
                    break
        else:
            # Fallback for unstructured or invalid values
            parsed_value = value

        result[key] = parsed_value

    return result

def parse_langchain_llm_output(input_string):
    try: # https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.AIMessage.html#langchain_core.messages.ai.AIMessage.pretty_print
        return input_string.pretty_print()
    except Exception as e:
        try: 
            structured_data = extract_key_value_pairs(input_string)
            return json.dumps(structured_data, indent=4)
        except Exception as e:
            return f"Error parsing LangChain LLM output. \nError: {e}. \nRaw output: {input_string}"

def parse_env_string(env_string):
    """Parse environment string and return a dictionary of key-value pairs."""
    env_vars = {}
    for line in env_string.splitlines():
        # Skip empty lines or comment-only lines
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        # Remove 'export' if present
        if line.startswith('export'):
            line = line.replace('export', '', 1).strip()
            
        # Split on first '=' and handle inline comments
        if '=' in line:
            # Split on first '#' to remove comments
            line_without_comment = line.split('#')[0].strip()
            
            # Now split on first '=' to get key-value
            key, value = line_without_comment.split('=', 1)
            
            # Clean up key and value
            key = key.strip()
            value = value.strip().strip('"\'')
            
            if key:  # Only add if key is not empty
                env_vars[key] = value
                
    return env_vars

def categorize_variables(env_vars):
    """Categorize environment variables into config sections."""
    has_gpu = shutil.which("nvidia-smi") is not None and subprocess.call(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0

    config = {
        'core': {
            'file_store': 'local',
            'jwt_secret': 'secretpass',
            'max_iterations': 30,
        },
        'llm': {
            'input_cost_per_token': get_input_price_per_token(),
            'output_cost_per_token': get_output_price_per_token(),
            'log_completions': True,
            'log_completions_folder': '../logs/openhands', 
        },
        'sandbox':{
            'enable_gpu': has_gpu,
        }
    }

    # Define patterns for categorization
    patterns = {
        'core': [
            r'FILE_STORE',
            r'DATABASE',
            r'PORT',
            r'HOST'
        ], 
        'llm': [
            r'.*_API_BASE',
            r'.*_API_VERSION',
            r'.*MODEL',
            r'EMBEDDING',
            r'DEPLOYMENT',
            r'.*_SECRET',
            r'.*_KEY',
            r'.*_TOKEN'
        ]
    }

    for key, value in env_vars.items():
        # Check each pattern category
        for section, pattern_list in patterns.items():
            if any(re.match(pattern, key, re.IGNORECASE) for pattern in pattern_list):
                # Convert key to lowercase for consistency
                config_key = key.lower()

                # Standardize naming for common keys
                if '_API_KEY' in key:
                    config_key = 'api_key'
                elif '_API_BASE' in key or '_API_URL' in key:
                    config_key = 'base_url'
                elif '_API_VERSION' in key:
                    config_key = 'api_version'
                elif 'ORGANIZATION' in key:
                    continue  # Skip organization key

                config[section][config_key] = value
                break

    # Remove empty sections
    return {k: v for k, v in config.items() if v}

def setup_openhands_credential():
    """Convert an environment string to a TOML configuration."""
    try:
        with open("setup/env.sh", "r") as f:
            env_string = f.read()
        
        env_vars = parse_env_string(env_string)
        config = categorize_variables(env_vars)
        
        # Ensure directory exists
        output_path = Path("../workspace/config.toml")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # write to config.toml
        with open(output_path, "w") as f:
            f.write(toml.dumps(config))
            
        print(f'Set up OpenHands credentials in workspace/config.toml')
        return toml.dumps(config)  # Returns TOML as a string
        
    except Exception as e:
        print(f"Error setting up credentials: {str(e)}")
        raise

def load_system_prompt(prompt_path, **kwargs):
    with open(prompt_path, "r") as f:
        template = f.read()
    return template.format(**kwargs)
    