# Curie

## Installation

1. Install docker: https://docs.docker.com/engine/install/ubuntu/

2. Build the container image. Whenever changes have been made: delete the current mounted volume (after backing up necessary data, of course), and rebuild the container image.

```bash
cd Curie
git submodule update --init --recursive 
pip install -e .
cd curie
sudo docker build --no-cache --progress=plain -t exp-agent-image -f ExpDockerfile_default ..
```

## Quick Start

### Add your LLM API keys: 
- [ ] support litellm
- [ ] add a config.toml to include all information

```
export MODEL="gpt-4o"
export API_VERSION="2024-06-01"
export OPENAI_API_KEY= 
export OPENAI_ORGANIZATION= 
export OPENAI_API_BASE= 
```

### Specify your research problem
Config `curie/configs/base_config.json` 
- [ ] Provide instructions

### Start experiments:
- [ ] change the execution path to its parent
```bash
cd Curie

python3 -m curie.main --iterations 1 --question_file benchmark/llm_reasoning/q1_simple_relation.txt --task_config curie/configs/llm_reasoning_config.json
```

To clean up the dockers
```bash
docker ps -a --format "{{.Names}}" | grep '^openhands-runtime' | xargs -r docker rm -f
```

- [ ] include the logging path
- [ ] enable streaming output



