# More Quick Start Questions

Input your research question or problem statement: `python3 -m curie.main -q "<Your research question>"`.

## **Example 1**: Understand the sorting algorithm efficiency.

```bash
python3 -m curie.main \
  -q "How does the choice of sorting algorithm impact runtime performance across different \
  input distributions (random, nearly sorted, reverse sorted)?"
``` 
- **Auto-generated Experiment report**: Available [ `logs/research_<ID>.md`](../docs/example_logs/sorting_example/research_1747978647_20250523013727_iter1.md).
- **Logs and Reproducibilty**:
  - The full experimentation process (script to reproduce results, generated code and experiment results) is saved in `workspace/research_<ID>/`.
  - Real-time logs are streamed to the console and stored in file `research_*.log`.


## **Example 2**: Find good ML strategies for noisy data.

```bash
python3 -m curie.main \
  -q "Are ensemble methods (e.g., Random Forests, Gradient Boosting) more robust to added noise \
  in the Breast Cancer Wisconsin dataset compared to linear models like Logistic Regression \
  for a binary classification task?"
```

- **Estimated runtime**: <5 minutes
- **Estimated cost**: $0.55
- **Sample log file**: Available [here](./docs/example_logs/research_noise_robustness_20250309.log)

<!-- 
## **Example 3**: Optimize feature selection for classification tasks.
- *Basic question*: whether feature selection helps the model performace.
```bash
python3 -m curie.main \
  -q "In the Wine dataset (which classifies wine cultivars based on chemical properties), \
  does using a genetic algorithm for feature selection improve model classification performance \
  in terms of accuracy when compared to using the full feature set? Specifically, does combining \
  the selected features with an ensemble classifier (e.g., Random Forest) lead to higher accuracy?"
```

- *More advanced question*: Find the optimal feature selection.

```bash
python3 -m curie.main \
  -q "For the Wine dataset (identifying wine cultivars using chemical properties), when using \ 
  an ensemble classifier (e.g., Random Forest), what is the best subset of features that will create a simpler, \
  more interpretable model that outperforms models  built on the full feature set. "
``` -->

## Expample 3: Evaluating CNN Performance on MNIST with Varying Batch Sizes (GPU Preferred)

```bash
python3 -m curie.main \
-q "How does the batch size affect model performance on the MNIST dataset, in terms of test accuracy and training speed? \
Setup: \
Please use PyTorch to code and GPU to train models. \
Use a CNN with 1 convolutional layers: each layer has 32 filters, a 3x3 kernel, ReLU activation, followed by a 2x2 max-pooling layer. \
Then flatten and connect to a fully connected layer for classification. \
Evaluate the model using different batch sizes: 16, 64, and 256. \
Training Details: \
Use the Adam optimizer with a learning rate of 0.001. \
Train each model for 20 epochs. \
Use the standard MNIST train/test split (60,000 training images, 10,000 test images). \
Metrics: \
Record the test accuracy after training. \
Measure the average training time per epoch (in seconds)." 
```


- **Estimated runtime**: ~20 minutes (Model training is time-consuming.)
- **Estimated cost**: $1.5
- **Sample log fil**e: Available [here](./docs/example_logs/research_batch_size_20250324.log)
- **Sample report file**: Available [here](./docs/example_logs/research_batch_size_20250324.md)