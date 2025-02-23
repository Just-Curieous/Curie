# [SWE-bench](https://www.swebench.com/)

To evaluate results on SWE-bench, run:

```bash
python -m swebench.harness.run_evaluation \ 
    --dataset_name princeton-nlp/SWE-bench_Lite \ 
    --max_workers 1 \ 
    --run_id validate-curie-test \ 
    --split test \ 
    --predictions_path ~/Curie/logs/swe_results/matplotlib.jsonl
```