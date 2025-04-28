# Dog Breed Identification
> We assume you have install `mlebench` following instructions under the MLE Bench [repo](`https://github.com/openai/mle-bench/tree/main`).


Given a set of dog images (training set with labels and a test set without labels), predict the breed of each dog. There are **120 possible breeds**.  

## Download Dataset  

```bash
mlebench prepare -c dog-breed-identification
```

## Run Curie

```bash
python3 -m curie.main -f benchmark/mle_bench/dog-breed-identification/dog-breed-identification-question.txt   --task_config curie/configs/mle_dog_config.json  --report
```

## Curie's Results
- Detailed question: `dog-breed-identification-question.txt`
- **Estimated runtime**: 2.2h  (Model training is time-consuming.)
- **Estimated cost**: $28 
- **Sample report file**: Available [here](./dog-breed-identification-question_20250427163751_iter1.md) 
<!-- - **Sample log fil**e: Available [here](/docs/example_logs/mle_activation_func_20250326.log) -->
