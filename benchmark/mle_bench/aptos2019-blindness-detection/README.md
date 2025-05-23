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
python3 -m curie.main -f benchmark/mle_bench/aptos2019-blindness-detection/aptos2019-blindness-detection.txt --task_config curie/configs/mle_config.json --dataset_dir /home/amberljc/.cache/mle-bench/data/aptos2019-blindness-detection/prepared/public 
```
- Change `--dataset_dir` to the absolute path to your dataset. 

## Curie Results

After asking Curie to solve this question, the following output files are generated:
- [`Report`](question_20250517013357_iter1.md): Auto-generated report with experiment design and findings  
- [`Experiment results`](question_20250517013357_iter1_all_results.txt): All detailed results for all conducted experiments
- [`Curie logs`](question_20250517013357_iter1.log): Execution log file  
- [`Curie workspace`](https://github.com/Just-Curieous/Curie-Use-Cases/tree/main/machine_learning/q4-aptos2019-blindness-detection): Generated code, complete script to reproduce and raw results (excluding the model checkpoint).

### Curie Performance Summary
#### Data Understanding

The APTOS 2019 Diabetic Retinopathy Detection dataset was used, containing retinal images with the following class distribution:

- Class 0 (No DR): 1628 samples (49.4%)
- Class 1 (Mild DR): 340 samples (10.3%)
- Class 2 (Moderate DR): 896 samples (27.2%)
- Class 3 (Severe DR): 176 samples (5.3%)
- Class 4 (Proliferative DR): 255 samples (7.7%)

#### Model Performance 
EfficientNet-B5 with 5-fold cross-validation provides the best performance for diabetic retinopathy detection, achieving a quadratic weighted kappa of 0.9058 and accuracy of 82.50%.

**Validation Metrics from Best Model (EfficientNet-B5):**
- Average Validation Accuracy: 0.8250 ± 0.0058
- Average Validation Kappa: 0.9337 ± 0.0039
- Average Training Time per Fold: 143.43 seconds
- Total Training Time: 717.17 seconds
- Model Size: 48.2 MB
- Average Inference Time: 20.5 ms per image

**Confusion Matrix (5-fold average):**
```
Predicted:   0    1    2    3    4   
Actual: 0 [599,  52,  23,  11,   5]
        1 [ 36, 208,  32,   9,   0]
        2 [ 22,  25, 308,  24,   6]
        3 [  9,   7,  17, 156,  11]
        4 [  4,   0,   8,  22, 122]
```
For complete details on methodology, experiments, and analysis, refer to the generated [report](./question_20250517013357_iter1.md)