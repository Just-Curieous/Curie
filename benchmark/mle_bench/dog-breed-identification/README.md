# Dog Breed Identification
> Prerequisites: Ensure you have install `mlebench` following instructions under the MLE Bench [repo](`https://github.com/openai/mle-bench/tree/main`).

Given a set of dog images (training set with labels and a test set without labels), predict the breed of each dog. There are **120 possible breeds**.  

## Download Dataset  

```bash
mlebench prepare -c dog-breed-identification
```

## Run Curie
```bash
cd Curie/
python3 -m curie.main -f benchmark/mle_bench/dog-breed-identification/dog-breed-identification-question.txt \
                     --dataset_dir ~/.cache/mle-bench/data/dog-breed-identification/prepared/public \                      
                     --task_config curie/configs/mle_config.json 
``` 
- Optionally include your starter code using `--workspace_name`.

## Curie's Results 
- Detailed question: [dog-breed-identification-question.txt](./dog-breed-identification-question.txt)
- **Estimated runtime**: ~2.2h  (Model training is time-consuming.)
- **Estimated cost**: $28 
- **Auto generated experiment report**: Available [here](./dog-breed-identification-question_20250427163751_iter1.md) 
- **Summary of the experiment results**: Available [here](./dog-breed-identification-question_20250427163751_iter1_all_results.txt)
- **Curie log file**: Available [here](https://github.com/Just-Curieous/Curie-Use-Cases/blob/main/machine_learning/q2_dog-breed-identification/dog-breed-identification_20250427163751_iter1/dog-breed-identification-question_20250427163751_iter1.log)


> Summary of Results from the [report](./dog-breed-identification-question_20250427163751_iter1.md) 
 
### 3.1 Control Group Experiments

#### 3.1.1 ResNet50 with Basic Configuration

This baseline experiment used ResNet50 with last-layer-only fine-tuning and basic data augmentation:

| Epoch | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy | Training Time |
|-------|--------------|-----------------|-------------------|---------------------|---------------|
| 1     | 2.2035       | 1.0728          | 52.25%            | 71.63%              | 39s           |
| 2     | 1.1574       | 0.8318          | 70.01%            | 74.24%              | 40s           |
| 4     | 0.6245       | 0.8570          | 83.15%            | 76.14%              | 41s           |
| 8     | 0.3006       | 0.9333          | 91.52%            | 76.14%              | 40s           |
| 30    | 0.2036       | 1.2140          | 93.45%            | 73.21%              | 40s           |

Key observations:
- Best validation loss (0.8318) achieved at epoch 2
- Best validation accuracy (76.14%) achieved at epoch 8
- Clear overfitting pattern after epoch 8

#### 3.1.2 EfficientNetB4 with Standard Configuration

This experiment evaluated EfficientNetB4 with standard augmentation and additional dropout:

| Epoch | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy | Training Time |
|-------|--------------|-----------------|-------------------|---------------------|---------------|
| 1     | 1.6842       | 0.9542          | 59.06%            | 72.55%              | 67s           |
| 3     | 0.8749       | 0.8606          | 72.33%            | 75.22%              | 66s           |
| 5     | 0.6324       | 0.8802          | 79.77%            | 74.35%              | 67s           |
| 10    | 0.3709       | 1.0180          | 87.21%            | 72.28%              | 67s           |

Key observations:
- Best validation loss (0.8606) at epoch 3
- Best validation accuracy (75.22%) at epoch 3
- Similar overfitting pattern but slightly later onset

#### 3.1.3 ResNet50 with All-Layer Fine-Tuning

This experiment tested ResNet50 with all layers unfrozen and enhanced data augmentation:

| Epoch | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy | Training Time |
|-------|--------------|-----------------|-------------------|---------------------|---------------|
| 1     | 2.6158       | 1.3724          | 45.26%            | 65.00%              | 60s           |
| 10    | 0.5066       | 0.9850          | 87.66%            | 75.11%              | 61s           |
| 20    | 0.0788       | 0.9611          | 98.17%            | 76.63%              | 61s           |
| 28    | 0.0135       | 0.9409          | 99.78%            | 76.79%              | 61s           |
| 30    | 0.0092       | 0.9410          | 99.82%            | 76.85%              | 61s           |

Key observations:
- Best validation loss (0.9409) at epoch 28
- Best validation accuracy (76.85%) at epoch 30
- Significant overfitting despite augmentation

#### 3.1.4 ResNet50 with Early Epoch Training

This experiment tested the early stopping hypothesis with ResNet50:

| Epoch | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy | Training Time |
|-------|--------------|-----------------|-------------------|---------------------|---------------|
| 1     | 2.0843       | 0.9218          | 54.97%            | 71.90%              | 38s           |
| 2     | 1.0684       | 0.7415          | 70.98%            | 77.28%              | 38s           |
| 3     | 0.5848       | 0.7444          | 83.15%            | 76.79%              | 39s           |

Key observations:
- Best validation loss (0.7415) achieved at epoch 2
- Best validation accuracy (77.28%) achieved at epoch 2
- Confirms the early performance peak hypothesis

### 3.2 Comparative Analysis


1. **Architecture Efficiency**: While both ResNet50 and EfficientNetB4 reached similar peak validation accuracies (77.28% vs. 75.22%), ResNet50 achieved this with significantly less training time.

2. **Fine-tuning Strategy Impact**: Contrary to our hypothesis, last-layer-only fine-tuning outperformed all-layer fine-tuning in terms of validation loss (0.7415 vs. 0.9409) while requiring substantially less computation time (115s vs. 1836s total).

3. **Early Performance Peak**: All models showed their best performance in relatively early epochs (2-8), with performance degrading thereafter due to overfitting.

4. **Training Efficiency**: The early-stopping ResNet50 experiment achieved the best overall performance (77.28% accuracy) with minimal training time (76 seconds total for 2 epochs).

5. **Augmentation Effect**: Enhanced augmentation with all-layer fine-tuning did not significantly outperform basic augmentation with last-layer fine-tuning, suggesting diminishing returns on complex augmentation for this task.