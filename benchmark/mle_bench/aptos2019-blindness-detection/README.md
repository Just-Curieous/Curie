# APTOS 2019 Blindness Detection
> Prerequisites: Ensure you have installed `mlebench` following [instructions](../README.md).

Given a dataset of retinal images, predict the severity level of diabetic retinopathy on a scale of 0 to 4. This is a multi-class classification task with medical significance.

## Dataset Overview

The APTOS 2019 Blindness Detection dataset contains high-resolution retinal images, classified into five severity levels:
- 0: No diabetic retinopathy
- 1: Mild diabetic retinopathy
- 2: Moderate diabetic retinopathy
- 3: Severe diabetic retinopathy
- 4: Proliferative diabetic retinopathy

This challenge addresses a critical healthcare issue as millions of people suffer from diabetic retinopathy, the leading cause of blindness among working-aged adults. Early detection can prevent blindness.

## Download Dataset

```bash
mlebench prepare -c aptos2019-blindness-detection
```

## Run Curie
- Update the configuration: Open `curie/configs/mle-aptos-config.json` and verify the paths to the dataset and starter code.
- Execute Curie:
```bash
cd Curie/
python3 -m curie.main -f benchmark/mle_bench/aptos2019-blindness-detection/question.txt --task_config curie/configs/mle_config.json --dataset_dir /home/amberljc/.cache/mle-bench/data/aptos2019-blindness-detection/prepared/public 
```
- Change `--dataset_dir` to the absolute path to your dataset. 

## Curie Results

After asking Curie to solve this question, the following output files are generated:
- [`Report`](question_20250517013357_iter1.md): Auto-generated report with experiment design and findings  
- [`Experiment results`](question_20250517013357_iter1_all_results.txt): All detailed results for all conducted experiments
- [`Curie logs`](question_20250517013357_iter1.log): Execution log file  
- [`Curie workspace`](https://github.com/Just-Curieous/Curie-Use-Cases/tree/main/machine_learning/q4-aptos2019-blindness-detection): Generated code, complete script to reproduce and raw results (excluding the model checkpoint).

### Performance Summary

The agent's experiments yielded impressive results for diabetic retinopathy detection:

- **Best Model**: EfficientNet-B5 with 5-fold cross-validation
- **Quadratic Weighted Kappa**: 0.9058 (benchmark metric for this task)
- **Classification Accuracy**: 82.50%
- **Model Architecture Comparison**:
  - ResNet50 (baseline): Kappa 0.7733
  - EfficientNet-B3: Kappa 0.8108
  - EfficientNet-B5: Kappa 0.9058

![Model Performance Comparison](model_performance_comparison.png)
![Computational Efficiency Comparison](computational_efficiency_comparison.png)
![Comprehensive Model Comparison](comprehensive_model_comparison.png)

The agent systematically experimented with multiple architectures and demonstrated that EfficientNet models outperformed ResNet50, with EfficientNet-B5 showing the best results. The 5-fold cross-validation approach produced more robust and generalizable results than single-split training.

For complete details on methodology, experiments, and analysis, refer to the generated [report](./question_20250517013357_iter1.md)