import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import json
import datetime
import re
import traceback
import requests
import shutil
import subprocess
from pathlib import Path
import toml
from typing import List, Dict, Any

from helper.logger import init_logger
def setup_utils_logging(log_filename: str):
    global bench_logger 
    bench_logger = init_logger(log_filename)

def get_messages_needed(msg: str):
    """Get the number of messages needed to fit within the context length."""
    token_counter = TokenCounter()
    context_length = get_model_context_length()
    max_tokens = context_length - 1000  # Reserve tokens for response
    # Get number of distinct messages needed to not exceed context length:
    # 1. Count the number of tokens in the message
    # 2. Calculate the number of messages needed to fit within the context length
    # 3. Return the number of messages needed
    num_tokens = token_counter.count_output_tokens(msg)
    print(f"Number of tokens in message: {num_tokens}")
    print(f"Max tokens: {max_tokens}")
    num_messages = (num_tokens // max_tokens) + 1
    # print(f"Number of messages needed: {num_messages}")
    return num_messages

def breakdown_message(msg: str, num_messages: int):
    """Break down msg into a list of smaller msg based on num_messages."""

    token_counter = TokenCounter()
    context_length = get_model_context_length()
    max_tokens = context_length - 10000  # Reserve tokens for response
    # Split the message into smaller messages
    chunks = text_splitter_by_tokens(msg, max_tokens, token_counter)
    # messages = []
    # for i in range(num_messages):
    #     start_index = i * max_tokens
    #     if i < num_messages - 1:
    #         # Ensure we don't exceed the message length
    #         end_index = (i + 1) * max_tokens
    #         messages.append(msg[start_index:end_index])
    #     else:
    #         # Last message can be the remainder
    #         messages.append(msg[start_index:])
    return chunks

def get_model_context_length() -> int:
    """Get the context length for the current model."""
    # FIXME: add more models as needed
    context_length_dict = {
        "gpt-4o": 128000,
        "o3-mini": 200000,
        "gpt-4o-mini": 128000,
        "anthropic.claude-3-7-sonnet-20250219-v1:0": 200000,
        "anthropic.claude-3-5-haiku-20241022-v1:0": 200000,
        "deepseek.r1-v1:0": 64000,
        "amazon.nova-pro-v1:0": 300000,
        "gemini-2.5-pro-preview-03-25": 200000,
    }
    model_name = get_model_name()
    return context_length_dict.get(model_name, 30000)

class TokenCounter:
    # Pricing per 1k tokens (as of March 2024)
    # PRICE_PER_1K_TOKENS = utils.get_all_price_per_1k_tokens()

    # Class-level variables to track accumulated usage across all instances
    _accumulated_tokens = {"input": 0, "output": 0}
    _accumulated_cost = {"input": 0.0, "output": 0.0, "tool_cost": 0.0}

    def __init__(self):
        # Strip provider prefix if present (e.g., "openai/gpt-4" -> "gpt-4")
        self.model_name = get_model_name()
        
        # try:
        #     self.encoding = tiktoken.encoding_for_model(self.model_name)
        # except KeyError:
        #     # Fall back to cl100k_base for models not in tiktoken
        #     self.encoding = tiktoken.get_encoding("cl100k_base")

    @classmethod
    def get_accumulated_stats(cls) -> Dict[str, Dict[str, float]]:
        """Get accumulated token usage and costs across all instances."""
        return {
            "tokens": dict(cls._accumulated_tokens),
            "costs": dict(cls._accumulated_cost),
            "total_cost": sum(cls._accumulated_cost.values())
        }

    def count_output_tokens(self, string: str) -> int:
        """
        MetaGPT anthropic client token counter does not work for anthropic>=0.39.0: https://github.com/geekan/MetaGPT/blob/main/metagpt/utils/token_counter.py#L479C1-L480C1
        Use simple tokenizer instead, since that is what langchain_aws is doing: 
            - https://github.com/langchain-ai/langchain-aws/commit/6355b0ff44c92b594ab8c3a5c50ac726904d716d
            - https://github.com/langchain-ai/langchain-aws/issues/314
            - https://python.langchain.com/api_reference/_modules/langchain_core/language_models/base.html#BaseLanguageModel

        Returns the number of tokens in a text string.

        Args:
            string (str): The text string.

        Returns:
            int: The number of tokens in the text string.
        """
        # if "claude" in self.model_name:
        #     vo = anthropic.Client()
        #     num_tokens = vo.count_tokens(string)
        #     return num_tokens
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            bench_logger.debug(f"Warning: model {self.model_name} not found in tiktoken. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        try:
            token_count = len(encoding.encode(string))
            return token_count
        except Exception as e:
            bench_logger.error(f"Error in token counting: {e}")
            return 0

def text_splitter_by_tokens(text: str, chunk_size: int, token_counter: TokenCounter) -> List[str]:
    """Split text based on token count instead of characters."""
    if isinstance(text, list):
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=100,
        length_function=lambda x: token_counter.count_output_tokens(x),
        is_separator_regex=False,
    )
    return text_splitter.split_text(text)

def print_exception_and_traceback(exc: Exception, prefix: str = ""):
    print(f"{prefix}: {exc}")

    # Print full traceback (if this is a Python-side error)
    traceback.print_exc()
    traceback_str = traceback.format_exc()

    # Log error and traceback
    bench_logger.error(f"Exception: {exc}")
    bench_logger.error(f"Traceback: {traceback_str}")

    # Log subprocess info if available
    for attr in ['cmd', 'stdout', 'stderr']:
        if hasattr(exc, attr):
            value = getattr(exc, attr)
            if value:  # skip if None or empty
                bench_logger.error(f"{attr.upper()}:\n{value}")
                print(f"{attr.upper()}:\n{value}")  # optionally print to console

# def print_exception_and_traceback(msg: str):
#     print(msg)
#     traceback.print_exc()
#     traceback_str = traceback.format_exc()
#     # log traceback_str
#     bench_logger.error(msg)
#     bench_logger.error(f"Traceback: {traceback_str}")

def escape_invalid_json_control_chars(raw_text):
    """
    Escapes unescaped newlines and other control characters inside double-quoted strings.
    """
    # Match double-quoted strings (not escaped), and escape control characters inside
    def escape_inside_quotes(match):
        content = match.group(0)
        fixed = content.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
        return fixed

    # This regex matches JSON-style string literals
    pattern = r'"(?:[^"\\]|\\.)*"'
    return re.sub(pattern, escape_inside_quotes, raw_text)

def safe_json_load(output_path: str) -> Any:
    with open(output_path, 'r') as f:
        try:
            response = json.load(f)
        except json.decoder.JSONDecodeError as e:
            print(f"❌ JSONDecodeError caught: {e}")
            bench_logger.info(f"❌ JSONDecodeError caught: {e}")
            with open(output_path, "r") as f:
                raw_text = f.read()
            print("Attempting to repair...")
            bench_logger.info("Attempting to repair...")
            fixed_json = escape_invalid_json_control_chars(raw_text)
            try:
                response = json.loads(fixed_json)
            except json.decoder.JSONDecodeError as e2:
                print(f"❌ Second JSONDecodeError caught during repair: {e2}")
                bench_logger.error(f"❌ Second JSONDecodeError caught during repair: {e2}")
                # As a last resort, return a dict with the error message
                return {"no_answer": f"JSONDecodeError after repair: {e2}"}
            print("Repair successful.. saving to file")
            bench_logger.info("Repair successful.. saving to file")
            with open(output_path, "w") as f:
                f.write(fixed_json)
        return response

def call_oh_with_prompt(full_prompt: str, temp_prompt_path: str, config: dict, github_workspace_path: str, oh_log_path: str, max_duration_per_task_in_seconds: int = 3600, iterations: int = 40):
    print("Calling OH with prompt:\n", full_prompt)
    bench_logger.info("Calling OH with prompt:\n" + full_prompt)
    sudo_available = shutil.which("sudo") is not None
    # print("Sudo is available:", sudo_available)
    chmod_cmd = f"{'sudo ' if sudo_available else ''}chmod 777 -R {github_workspace_path}"
    # Compose command to run OpenHands. we use more iterations here since there may be a need to execute code too.
    command = f"""
    export LOG_ALL_EVENTS=true && \
    {chmod_cmd} && \
    export WORKSPACE_BASE={config["base_dir"] + "/" + github_workspace_path} && \
    export SANDBOX_TIMEOUT=600 && \
    /root/.cache/pypoetry/virtualenvs/openhands-ai-*-py3.12/bin/python \
    -m openhands.core.main \
    -f {temp_prompt_path} \
    --config-file "workspace/config.toml" \
    --max-iterations {iterations} \
    2>&1 | tee -a {oh_log_path}
    """

    # https://alexandra-zaharia.github.io/posts/kill-subprocess-and-its-children-on-timeout-python/
    # Run the command in a shell
    import signal

    try:
        p = subprocess.Popen(
            command,
            shell=True,
            executable="/bin/bash",
            start_new_session=True  # ensures a new process group
        )
        p.wait(timeout=max_duration_per_task_in_seconds)
    except subprocess.TimeoutExpired:
        print(f"Timeout expired for command, killing process group...", flush=True)
        bench_logger.error("Timeout expired for command, killing process group...")
        try:
            os.killpg(p.pid, signal.SIGTERM)  # send SIGTERM to the process group
        except Exception as e:
            print(f"Failed to kill process group: {e}")
            bench_logger.error(f"Failed to kill process group: {e}")
        raise

def get_all_price_per_1k_tokens() -> Dict[str, Dict[str, float]]:
    return {
        "gpt-4o": {"input": 0.00375, "output": 0.015}, 
        "gpt-4o-mini": {"input": 0.00015, "output": 0.000075},
        "o3-mini": {"input": 0.0011, "output": 0.0044},
        "anthropic.claude-3-7-sonnet-20250219-v1:0": {"input": 0.003, "output": 0.015},
        "anthropic.claude-3-5-haiku-20241022-v1:0": {"input": 0.0008, "output": 0.004},
        "deepseek.r1-v1:0": {"input": 0.00135, "output": 0.0054},
        "amazon.nova-pro-v1:0": {"input": 0.0008, "output": 0.0032},
        "gemini-2.5-pro-preview-03-25": {"input": 0.0000, "output": 0.0000},
    }

def get_first_n_tokens(markdown_content, max_tokens=100000):
    # Load GPT-4 tokenizer (or another appropriate model)
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    
    # Tokenize the content
    tokens = tokenizer.encode(markdown_content)

    # Get only the first `max_tokens`
    first_tokens = tokens[:max_tokens]
    
    # Decode back to text
    truncated_markdown = tokenizer.decode(first_tokens)
    
    return truncated_markdown

def load_system_prompt(prompt_template, **kwargs):
    return prompt_template.format(**kwargs)

def load_prompt_from_file(prompt_filename=None):
    """
    Load a prompt from a file
    
    Args:
        prompt_filename (str): Path to the prompt file
        
    Returns:
        str: The prompt text from the file
    """
    
    if not os.path.exists(prompt_filename):
        raise FileNotFoundError(f"Prompt file not found: {prompt_filename}")
    
    try:
        with open(prompt_filename, 'r') as f:
            prompt_text = f.read()
        print(f"Loaded prompt from {prompt_filename}")
        return prompt_text
    except Exception as e:
        raise Exception(f"Error reading prompt file: {str(e)}")

def update_tool_costs(cost, phase, paper_id, filename="", mode="setup_gen"):
    # Write the cost to a file, using timestamp, but keep using the same file for a given day:
    if mode == "setup_gen":
        cost_file = f"./outputs/logs/openhands_total_cost_{datetime.datetime.now().strftime('%Y-%m-%d')}_{paper_id}.txt"
    else:
        cost_file = filename
    print("Writing cost to file: " + cost_file)
    bench_logger.info(f"Writing cost to file: {cost_file}")
    if os.path.exists(cost_file):
        with open(cost_file, "a") as f:
            f.write(f"Total cost for paper_ID {paper_id} phase {phase}: {cost}\n")
    else:
        # If the file does not exist, create it and write the cost
        with open(cost_file, "w") as f:
            f.write(f"Total cost for paper_ID {paper_id} phase {phase}: {cost}\n")

def _collect_openhands_cost(phase, paper_id, filename2="", mode="setup_gen"):
    total_cost = 0
    # read all openhands log json files under ../logs
    for filename in os.listdir("./logs/openhands"):
        if filename.endswith(".json"):
            remove_flag = False
            with open(f"./logs/openhands/{filename}", "r") as f:
                data = json.load(f)
                if "cost" in data:
                    remove_flag = True
                    total_cost += data["cost"]
            if remove_flag:
                os.remove(f"./logs/openhands/{filename}")
    print(f"$$$$ Total cost of OpenHands: {total_cost} $$$$") 
    bench_logger.info(f"$$$$ Total cost of OpenHands: {total_cost} $$$$")
    update_tool_costs(total_cost, phase, paper_id, filename=filename2, mode=mode)

def extract_tokens(log_str): 
    # Pattern to match "input=X, output=Y, total=Z"
    pattern = r'input=(\d+),\s*[\r\n]*\s*output=(\d+),\s*[\r\n]*\s*total=(\d+)'

    # Find all matches in the log string
    matches = re.findall(pattern, log_str, flags=re.DOTALL)
    
    if not matches:
        return None  # or return {'input': 0, 'output': 0, 'total': 0}

    # Take only the last match
    input_count, output_count, _ = matches[-1]
    
    return {
        'input': int(input_count),
        'output': int(output_count),
        'total': int(input_count) + int(output_count)
    }

def _collect_inspectai_cost(agent_log_filepath, agent_cost_filepath, mode="setup_gen"):
    with open(agent_log_filepath, "r") as f:
        log_str = f.read()
    tokens = extract_tokens(log_str)
    total_cost = tokens["input"] * get_input_price_per_token() + tokens["output"] * get_output_price_per_token()
    # Write to cost file:
    with open(agent_cost_filepath, "w") as f:
        f.write(f"Total cost: {total_cost}\n")
    print(f"Total cost: {total_cost}")
    bench_logger.info(f"Total cost: {total_cost}")

def get_model_name() -> str:
    """Strip provider prefix if present (e.g., "openai/gpt-4" -> "gpt-4")"""
    current_model = os.environ.get("MODEL", "gpt-4o")
    # Strip provider prefix if present (e.g., "openai/gpt-4" -> "gpt-4")
    model_name = current_model.split('/')[-1]
    if ("claude" in model_name or "amazon" in model_name or "deepseek" in model_name) and "us." in model_name: # example: "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        # Remove "us." prefix:
        model_name = model_name.split("us.")[1]
    return model_name

def get_input_price_per_token() -> float:
    """Get the price per token for input text."""
    model_name = get_model_name()
    return get_all_price_per_1k_tokens()[model_name]["input"] / 1000

def get_output_price_per_token() -> float:
    """Get the price per token for output text."""
    model_name = get_model_name()
    return get_all_price_per_1k_tokens()[model_name]["output"] / 1000

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
            'log_completions_folder': './logs/openhands', 
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

def setup_openhands_credential(llm_config_filename: str = "setup/env-openhands.sh"):
    """Convert an environment string to a TOML configuration."""
    try:
        with open(llm_config_filename, "r") as f:
            env_string = f.read()
        
        env_vars = parse_env_string(env_string)
        config = categorize_variables(env_vars)
        
        # Ensure directory exists
        output_path = Path("./workspace/config.toml")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # write to config.toml
        with open(output_path, "w") as f:
            f.write(toml.dumps(config))
            
        print(f'Set up OpenHands credentials in workspace/config.toml')
        return toml.dumps(config)  # Returns TOML as a string
        
    except Exception as e:
        print(f"Error setting up credentials: {str(e)}")
        raise

def get_file_from_github(repo_url, file_path, branches=("main", "master")):
    # Normalize URL
    repo_url = repo_url.rstrip("/")
    if repo_url.endswith(".git"):
        repo_url = repo_url[:-4]

    parts = repo_url.split("/")
    if len(parts) < 2:
        raise ValueError("Invalid GitHub repo URL")

    # Strip /workspace from file_path:
    if file_path.startswith("/workspace"):
        file_path = file_path[len("/workspace"):]

    user, repo = parts[-2], parts[-1]

    for branch in branches:
        raw_url = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{file_path}"
        response = requests.get(raw_url)
        if response.status_code == 200:
            return response.text

    raise Exception(f"File not found in branches {branches} for repo {repo_url}")

def concatenate_setup_scripts(repo_url: str, setup_scripts: List[str]) -> str:
    """
    Concatenate setup scripts from a GitHub repository into a single string.
    
    Args:
        repo_url (str): The URL of the GitHub repository.
        setup_scripts (List[str]): A list of file paths to the setup scripts.
        
    Returns:
        str: A single string containing the concatenated setup scripts. It will be formatted like such:
        ```
        name_of_source_file_1:

        #!/bin/bash
        # ...
        # ...
        # ...

        name_of_source_file_2:

        #!/bin/bash
        # ...
        # ...
        # ...
        ```
    """

    concatenated_setup_scripts = ""
    for script in setup_scripts:
        concatenated_setup_scripts += f"{script}:\n\n"
        concatenated_setup_scripts += get_file_from_github(repo_url, script) + "\n\n"
    return concatenated_setup_scripts