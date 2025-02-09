import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import io
import json
import re
import ast
import os

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

