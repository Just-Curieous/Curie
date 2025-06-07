import subprocess
import tempfile
import os
import json
import argparse


def generate_report(input_dir_path="/home/amberljc/dev/Curie/logs/istar_20250605231023_iter1/"):
    """
    Run a Docker container with exp-agent-image, activate micromamba environment,
    and execute the report generation code.
    
    Args:
        input_dir_path (str): Path to the input directory containing the JSON config file
    
    Returns:
        tuple: (return_code, stdout, stderr)
    """
    # check if the input_dir_path is a valid directory
    if not os.path.exists(input_dir_path):
        raise ValueError(f"Input directory {input_dir_path} does not exist")
    # check if json file exists
    json_files = [f for f in os.listdir(input_dir_path) if f.endswith('.json') and 'config' in f]
    if len(json_files) == 0:
        raise ValueError(f"No JSON file found in {input_dir_path}. Please input the correct log directory.")
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
    
    path_name = os.path.abspath(input_dir_path).split('/')[-1]
    print(f'Report will be saved to {input_dir_path}. Please wait 2~5 minutes for the report to be generated...')
    
    
    try:
        # Docker command to run the container
        docker_cmd = [
            'docker', 'run',
            '--rm',  # Remove container after execution
            '-v', f'{input_dir_path}:/tmp_logs',  # Mount input directory
            '-v', f'{temp_python_file}:/curie/script.py',  # Mount the Python script
            'curie-pip-image',
            'bash', '-c',
            f'mkdir -p /logs/{path_name} && '
            f'touch /logs/{path_name}/{config_file.replace(".json", ".log")} && '
            'cd /curie && '
            '''eval "$(micromamba shell hook --shell bash)" && '''
            'micromamba activate /opt/micromamba/envs/curie && python script.py'
        ]
        
        print(f"Running Docker command: {' '.join(docker_cmd)}")
        
        # Execute the Docker command
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=1000 
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
    parser.add_argument('--input-dir', '-i', 
                      default="/home/amberljc/dev/Curie/logs/istar_20250605231023_iter1/",
                      help='Path to the corresponding log directory.')
    
    args = parser.parse_args()
    
    return_code, stdout, stderr = generate_report(args.input_dir)
    
    print(f"Return code: {return_code}")
    print(f"STDOUT:\n{stdout}")
    if stderr:
        print(f"STDERR:\n{stderr}")
    
    return return_code


if __name__ == "__main__":
    main()