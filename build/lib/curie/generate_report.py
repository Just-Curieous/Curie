import subprocess
import tempfile
import os
import json
import argparse
from typing import Optional, Dict, Any


def write_api_keys_to_env(api_keys: Dict[str, str]) -> None:
    """Write API keys to env.sh file."""
    env_path = os.path.join(os.getcwd(), '.setup', 'env.sh')
    os.makedirs(os.path.dirname(env_path), exist_ok=True) 
    
    with open(env_path, 'w') as f:
        for key, value in api_keys.items():
            print(f"Writing {key} to {env_path}")
            f.write(f'export {key}="{value}"\n')


def generate_report(log_dir: str,
                    api_keys: Dict[str, str],
                    workspace_dir: Optional[str] = None):
    """
    Run a Docker container with exp-agent-image, activate micromamba environment,
    and execute the report generation code.
    
    Args:
        log_dir (str): Path to the input directory containing the JSON config file
        api_keys (Optional[Dict[str, str]]): Dictionary of API keys to be written to env.sh
    
    Returns:
        tuple: (return_code, stdout, stderr)
    """
    # Write API keys to env file if provided
    if api_keys:
        write_api_keys_to_env(api_keys)

    # check if the input_dir_path is a valid directory
    if not os.path.exists(log_dir):
        raise ValueError(f"Input directory {log_dir} does not exist")
    # check if json file exists
    json_files = [f for f in os.listdir(log_dir) if f.endswith('.json') and 'config' in f]
    if len(json_files) == 0:
        raise ValueError(f"No JSON file found in {log_dir}. Please input the correct log directory.")
    config_file = json_files[0]
    config_file = os.path.basename(config_file)

    # Python code to execute inside the container
    python_code = '''
import json, os
from reporter import generate_report 
with open('/tmp_logs/{}', 'r') as file:
    config = json.load(file)

exp_plan_filename = config['exp_plan_filename'].split("/")[-1].replace(".txt", ".json")
dirname = config['log_filename'].split("/")[:-1]


with open('/tmp_logs/' + exp_plan_filename, 'r') as file:
    workspace_dir_list = []
    plans = []
    for line in file.readlines():
        if line == '\\n':
            continue
        plan = json.loads(line)
        
        plans.append(plan)
        workspace_dir = plan['workspace_dir'].replace('/', '', 1)
        workspace_dir_list.append(workspace_dir)
        
report_filename = generate_report(config, plans)

'''.format(config_file.rstrip('/'))

    # Create a temporary Python file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
        temp_file.write(python_code)
        temp_python_file = temp_file.name
    
    path_name = os.path.abspath(log_dir).split('/')[-1]
    print(f'✍️ Report will be saved to {log_dir}. Please wait 2~5 minutes for the report to be generated...')
    
    api_key_dir = os.path.join(os.getcwd(), '.setup')
    workspace_dir = workspace_dir if workspace_dir else os.path.join(os.getcwd(), 'workspace')
    print(f'cwd: {os.getcwd()}')
    try:
        # Docker command to run the container
        docker_cmd = [
            'docker', 'run',
            '--rm',  # Remove container after execution
            '-v', f'{log_dir}:/tmp_logs',  # Mount input directory
            '-v', f'{temp_python_file}:/curie/script.py',  # Mount the Python script
            '-v', f'{api_key_dir}:/curie/setup/',  # Mount API keys directory
            # '-v', f'{os.path.join(os.getcwd(), 'curie')}:/curie/',
            '-v', f'{workspace_dir}:/workspace/',
            'amberljc/curie:latest',
            'bash', '-c',
            f'mkdir -p /logs/{path_name} && '
            f'touch /logs/{path_name}/{config_file.replace(".json", ".log")} && '
            'source /curie/setup/env.sh && '
            'cd /curie && '
            '''eval "$(micromamba shell hook --shell bash)" && '''
            'micromamba activate /opt/micromamba/envs/curie && python script.py &&'
            f'mv /logs/{path_name}/* /tmp_logs'
        ]
        
        # Execute the Docker command
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=1200 
        )
        
        return result.returncode, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        return -1, "", "Docker command timed out after 5 minutes"
    except Exception as e:
        return -1, "", f"Error running Docker command: {str(e)}"
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_python_file)
        except:
            pass


def main():
    """Command-line interface for generating reports."""
    parser = argparse.ArgumentParser(description='Generate experiment reports from Curie logs.')
    parser.add_argument('--log-dir', '-i',  
                      help='Path to the corresponding log directory.')
    parser.add_argument('--api-keys', '-k',
                      type=json.loads,
                      help='JSON string containing API keys')
    parser.add_argument('--workspace-dir', '-w',
                      help='Path to the workspace directory.')
    
    args = parser.parse_args()
    
    return_code, stdout, stderr = generate_report(args.log_dir, args.api_keys, args.workspace_dir)
    
    print(f"Return code: {return_code}")
    print(f"STDOUT:\n{stdout}")
    if stderr:
        print(f"STDERR:\n{stderr}")
    
    return return_code


if __name__ == "__main__":
    main()