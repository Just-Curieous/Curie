import subprocess
import time
import os
import shutil  # Import for deleting directories
import psutil


def clear_workspace():
    """
    Deletes all contents of the /workspace directory but keeps the directory itself.
    """
    workspace_path = "/workspace"
    if os.path.exists(workspace_path):
        try:
            # Iterate over all contents in the directory
            for item in os.listdir(workspace_path):
                item_path = os.path.join(workspace_path, item)
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)  # Remove files or symlinks
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)  # Remove directories
            print(f"Cleared all contents of the directory: {workspace_path}")
        except Exception as e:
            print(f"Error clearing contents of {workspace_path}: {e}")
    else:
        print(f"Directory {workspace_path} does not exist. Nothing to clear.")

def run_command_x_times(base_command, log_prefix, config_file, x, timeout):
    """
    Runs a shell command X number of times with dynamic log file names and a fixed configuration file.
    Skips to the next iteration if the log file does not receive new lines within the specified timeout.

    :param base_command: The base command to execute (excluding the log file and config file).
    :param log_prefix: The prefix for the log file names.
    :param config_file: The configuration file to pass as the second argument.
    :param x: The number of times to run the command.
    :param timeout: The duration (in seconds) to wait for new lines in the log file.
    """
    for i in range(1, x + 1):  # Iteration numbers start from 1
        # log_file = f"{log_prefix}{i}.log"  # Generate log file name
        command = f"{base_command}"  # Complete command
        print(f"Running command iteration {i}: {command}")
        
        # Start the command
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Monitor the log file for changes
        start_time = time.time()
        last_size = 0
        
        while True:
            if time.time() - start_time > timeout:
                print(f"No new lines detected in for {timeout} seconds. Moving to the next iteration.")
                process.kill()
                process.wait()  # Ensure the process is completely terminated
                print("Process forcefully terminated.")
                break  # Exit the loop and move to the next iteration
            
            time.sleep(1)  # Check the file every second
            
            # Check if the process has completed
            if process.poll() is not None:
                print(f"Command finished for iteration {i}.")
                break

        # Verify that process has closed:
        # Get the process ID (PID)
        pid = process.pid

        # Check if the process is still running
        if psutil.pid_exists(pid):
            print(f"Process with PID {pid} is still running...")
        else:
            print(f"Process with PID {pid} has terminated.")
        
        # # Clear the /workspace folder. This is used by the control and experimental workers to save their files. 
        # clear_workspace()

if __name__ == "__main__":
    # Base command without log file or config file
    base_command = "sleep 10"
    
    # Log file prefix
    log_prefix = "logs/log-temp"
    
    # Configuration file
    config_file = "configs/cloud_config.json"  # Replace with your desired config file
    
    # Number of times to execute the command
    iterations = 10  # Replace with your desired number
    
    # Timeout in seconds for monitoring log file
    timeout = 9  # Replace with your desired timeout duration
    
    # Run the command
    run_command_x_times(base_command, log_prefix, config_file, iterations, timeout)
