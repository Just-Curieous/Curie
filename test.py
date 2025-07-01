import curie
# Set up your API keys, refer to curie/setup/env.sh.example
key_dict = {
    "MODEL": "azure/gpt-4o",
    "AZURE_API_VERSION": "2025-01-01-preview",
    "AZURE_API_KEY": "bdef9bf150194da5b8ad860046aff989",
    "ORGANIZATION": '327403',
    "AZURE_API_BASE": 'https://api.umgpt.umich.edu/azure-openai-api'
}

result = curie.experiment(api_keys=key_dict, 
                          question="How does the choice of sorting algorithm impact runtime performance across different input distributions?",
                          task_config = {
                            "job_name": "default_research",
                            "docker_image": "amberljc/curie:latest",
                            "dockerfile_name": "ExpDockerfile_pip", 
                            "benchmark_specific_context": "none",
                            "is_user_interrupt_allowed": False,
                            "timeout": 600,
                            "max_coding_iterations": 25,
                            "max_global_steps": 20,
                            #/Users/crakeee/goodgood/iac-project/Curie/
                            "supervisor_system_prompt_filename": "prompts/simple/simple-supervisor.txt",
                            "control_worker_system_prompt_filename": "prompts/simple/simple-control-worker.txt",
                            "patcher_system_prompt_filename": "prompts/simple/simple-patcher.txt",
                            "llm_verifier_system_prompt_filename": "prompts/simple/simple-llm-verifier.txt",
                            "coding_prompt_filename": "prompts/simple/simple-coding.txt",
                            "worker_system_prompt_filename": "prompts/simple/simple-worker.txt",
                            "workspace_name": "", # to be filled up by the user
                            "dataset_dir": "", # to be filled up by the user
                            "env_requirements": "", # to be filled up by the user
                            "code_instructions": "", # to be filled up by the user
                            }
                        )