# LLM Reasoning: Impact of CoT Reasoning Steps

The paper [The Impact of Reasoning Step Length on Large Language Models](https://arxiv.org/abs/2401.04925) explores the correlation between the effectiveness of CoT and the length of reasoning steps in prompts, namely would increasing reasoning steps of CoT improve the LLM performance.

## Common Setup

Put the API key into `env.sh` under each working directory.

```bash
export MODEL="gpt-4o"
export API_VERSION="2024-06-01"
export OPENAI_API_KEY=X
export OPENAI_ORGANIZATION=Y
export OPENAI_API_BASE=Z
```

We put all starter file and our code base under `$WORKHOME`

```bash
export WORKHOME=~/ # change this to your home repo
```

## Impact of CoT Reasoning Steps Setup

```bash
cd $WORKHOME 
git clone https://github.com/AE-W/The-Impact-of-Reasoning-Step-Length-on-Large-Language-Models.git
```

This experiment requires a CUDA-supported machine. We recommend using our docker image to set up your experiment environment. You can build this image by running the following commands.

```bash
# bin/bash
docker images -q exp-agent-cotsteps-image | xargs -r docker rmi -f # remove any existing conflict image
cd curie && docker build --no-cache --progress=plain -t exp-agent-cotsteps-image -f ExpDockerfile_CotSteps .. && cd -
docker run -v /var/run/docker.sock:/var/run/docker.sock \
        -v $(pwd)/curie:/curie:ro \
        -v $(pwd)/benchmark:/benchmark:ro \
        -v $(pwd)/logs:/logs \
        -v $(pwd)/starter_file:/starter_file:ro \
        -v $(pwd)/workspace:/workspace \
        --network=host -d --name exp-agent-cotsteps-container-test exp-agent-cotsteps-image
```

