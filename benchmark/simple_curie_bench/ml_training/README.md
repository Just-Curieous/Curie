# ML Training Benchmark
- CPU only

## Setup

```
export WORKHOME=~/langgraph-exp-agent/starter_file # change this to your home repo
cd $WORKHOME 
git clone https://github.com/patrickkon/MLAgentBench.git
```

- Setup Kaggle. Ensure that you have ".kaggle/kaggle.json" with your API credentials in the MLAgentBench root folder.
```
{"username":"X","key":"Y"}
```

Test on Openhand:

```
source myenv/bin/activate
export LOG_ALL_EVENTS=true
cd ~/OpenHands 
poetry run python -m openhands.core.main -f  <(cat ~/langgraph-exp-agent/benchmark/common.txt ~/langgraph-exp-agent/benchmark/ml_training/q1_housing_price.txt) 2>&1 | tee -a ~/langgraph-exp-agent/logs/openhands/ml_training/test_logging.txt
```