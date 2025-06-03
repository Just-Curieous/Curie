# Agent for AI Research

This project reproduces the iterative agent framework introduced in the paper [PaperBench](https://openai.com/index/paperbench/), providing a practical implementation for researchers and developers to explore and extend.

 
## Installation 
```bash
git clone https://github.com/Just-Curieous/inspect-agent.git
cd inspect-agent/
docker build --platform=linux/amd64 -t pb-env -f Dockerfile.base .
```

## Run the agent on your research tasks
0. Build the docker.
1. Copy `env.sh.example` to `env.sh`. 
2. Replace your system prompt under `instructions.txt`  
3. Run Inspect AI Agent with your code base and questions: `python entry_point.py --json_path <path_to_research_paper>  --code_repo_path <path_to_code_repo> --inspect_path $(PWD)
   For example: 
```bash
python entry_point.py --json_path /home/ubuntu/Benchmark-Construction/logs/neurips2024/95262.json --code_repo_path /home/ubuntu/Benchmark-Construction/logs/neurips2024/MoE-Jetpack --inspect_path /home/ubuntu/inspect-agent
```

## Manual Setup
### Setup experiment
```bash
cd inspect-agent/; docker run -it --name my-pb-env -v $(pwd):/workspace -v /:/all pb-env 
docker exec -it my-pb-env bash
```

Copy `env.sh.example` to `env.sh`. 
And configure:
- Your model and API key.
- Directory to your code and paper/questions

Remember to 
- Put your system prompt under `instructions.txt`  
- Put your code repo under `$CODE_DIR`.

### Start the agent

```
cd /workspace 
bash start.sh <PATH_TO_CODE> <PATH_TO_PAPER>
```



