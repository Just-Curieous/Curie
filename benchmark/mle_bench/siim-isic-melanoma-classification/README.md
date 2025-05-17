# SIIM-ISIC Melanoma Classification
> Prerequisites: Ensure you have installed `mlebench` following instructions under the MLE Bench [repo](https://github.com/openai/mle-bench/tree/main).

Given a dataset of skin lesion images, predict whether the lesion is malignant (melanoma) or benign. This is a binary classification task with significant class imbalance.

## Download Dataset


```bash
mlebench prepare -c siim-isic-melanoma-classification

```

## Run Curie
- Update the configuration: Open `curie/configs/mle-siim-isic-melanoma.json` and verify the paths to the dataset and starter code.
- Execute Curie. 
```bash
cd Curie/
python3 -m curie.main -f benchmark/mle_bench/siim-isic-melanoma-classification/question.txt --task_config curie/configs/mle_config.json --dataset_dir  /home/amberljc/.cache/mle-bench/data/siim-isic-melanoma-classification/prepared/public/ --report
``` 
- Change `--dataset_dir` to the absolute path to your dataset. 