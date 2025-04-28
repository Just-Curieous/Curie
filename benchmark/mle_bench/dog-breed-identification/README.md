# Dog Breed Identification
> Prerequisites: Ensure you have install `mlebench` following instructions under the MLE Bench [repo](`https://github.com/openai/mle-bench/tree/main`).

Given a set of dog images (training set with labels and a test set without labels), predict the breed of each dog. There are **120 possible breeds**.  

## Download Dataset  

```bash
mlebench prepare -c dog-breed-identification
```

## Run Curie
- Update the configuration: Open `curie/configs/mle_dog_config.json` and verify the paths to the dataset and starter code.
- Execute Curie. 
```bash
python3 -m curie.main -f benchmark/mle_bench/dog-breed-identification/dog-breed-identification-question.txt --task_config curie/configs/mle_dog_config.json  --report
``` 


## Curie's Results 
- Detailed question: `dog-breed-identification-question.txt`
- **Estimated runtime**: ~2.2h  (Model training is time-consuming.)
- **Estimated cost**: $28 
- **Auto generated experiment report**: Available [here](./dog-breed-identification-question_20250427163751_iter1.md) 
- **Curie log file**: Available [here](./dog-breed-identification-question_20250427163751_iter1.log)


> Summary of Results

| Experiment | Model      | Fine-tuning Strategy | Augmentation | Epochs | Validation Accuracy |
|------------|------------|----------------------|--------------|--------|---------------------|
| 1          | ResNet50   | Last layer           | Basic        | 30     | 82.65%              |
| 2          | EfficientNetB4 | Last layer       | Standard     | 30     | 75.22%              |
| 3          | ResNet50   | Last layer           | Basic        | 30     | 12.99%*             |
| 4          | ResNet50   | Last layer           | Enhanced     | 30     | 76.79%              |
| 5          | ResNet50   | Last layer           | Basic        | 30     | 75.32%              |
| 6          | ResNet50   | Last layer           | Basic        | 3      | 76.79%              |

*Note: The unusually low accuracy in Experiment 3 suggests implementation issues rather than a fundamental limitation of the resolution.
