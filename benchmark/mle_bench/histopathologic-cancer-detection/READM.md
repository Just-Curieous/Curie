# Histopathologic Cancer Detection
> Prerequisites: Ensure you have installed `mlebench` following instructions under the MLE Bench [repo](https://github.com/openai/mle-bench/tree/main).

## Overview

Histopathologic cancer detection is a critical task in medical image analysis that involves identifying cancerous cells in microscopic images of tissue samples. Early and accurate detection of cancer cells can significantly improve patient outcomes and treatment planning. This task focuses on developing machine learning models to automate the detection of cancer cells in histopathology images, which can help pathologists work more efficiently and reduce human error.

In this task, you will develop a model to identify cancer cells in histopathology images. The challenge involves analyzing microscopic images of tissue samples and determining whether they contain cancerous cells. This is a binary classification task where each image needs to be classified as either containing cancer cells (positive) or not (negative).

## Download Dataset

```bash
mlebench prepare -c histopathologic-cancer-detection
```
  
## Run Curie

1. Update the configuration: Open `curie/configs/mle-histopathologic-cancer.json` and verify the paths to the dataset and starter code.

2. Execute Curie:
```bash
cd Curie/
python3 -m curie.main -f benchmark/mle_bench/histopathologic-cancer-detection/histopathologic-cancer-detection.txt --task_config curie/configs/mle_config.json --dataset_dir /home/amberljc/.cache/mle-bench/data/histopathologic-cancer-detection/prepared/public
```

3. Change `--dataset_dir` to the absolute path to your dataset.

## Dataset

The dataset consists of histopathology images from the PatchCamelyon (PCam) benchmark dataset, which is derived from the Camelyon16 challenge. The images are 96x96 pixel patches extracted from larger whole-slide images of lymph node sections.

- **Images:** Available in PNG format (96x96 pixels)
- **Metadata:**
  - `id`: Unique identifier for each image
  - `label`: Binary classification label (0 = negative, 1 = positive for cancer)

### Dataset Characteristics

- **Image Size:** 96x96 pixels
- **Color Channels:** RGB
- **Total Images:** ~220,000 training images
- **Class Distribution:** Approximately balanced between positive and negative cases
- **Image Source:** Lymph node sections from whole-slide images

### Dataset Challenges

- **Small Image Size:** The 96x96 pixel patches require efficient feature extraction
- **Complex Patterns:** Cancer cells can appear in various forms and patterns
- **Image Quality:** Variations in staining and tissue preparation can affect image appearance
- **Computational Efficiency:** Large dataset size requires efficient training approaches

## Evaluation Metrics

The model will be evaluated using the following metrics:
- Area Under the ROC Curve (AUC-ROC)
- Accuracy
- Precision
- Recall
- F1-Score

## Best Practices

1. **Data Preprocessing:**
   - Normalize pixel values
   - Apply appropriate augmentations
   - Handle class imbalance if present

2. **Model Selection:**
   - Consider CNN architectures suitable for small image classification
   - Experiment with transfer learning from pre-trained models
   - Optimize for both accuracy and computational efficiency

3. **Training Strategy:**
   - Use appropriate learning rates and optimization techniques
   - Implement early stopping to prevent overfitting
   - Consider cross-validation for robust evaluation

4. **Inference:**
   - Optimize model for fast inference
   - Consider model quantization for deployment
   - Ensure reproducible results
