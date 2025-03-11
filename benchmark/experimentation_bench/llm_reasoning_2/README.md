# LLM Reasoning: CoT Reasoning Steps

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

## CoT Reasoning Steps Setup

```bash
cd $WORKHOME 
git clone https://github.com/AE-W/The-Impact-of-Reasoning-Step-Length-on-Large-Language-Models
```

