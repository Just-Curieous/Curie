# More Quick Start Questions

Input your research question or problem statement: `python3 -m curie.main -q "<Your research question>"`.

## **Example 1**: Understand the sorting algorithm efficiency.

```bash
python3 -m curie.main \
  -q "How does the choice of sorting algorithm impact runtime performance across different \
  input distributions (random, nearly sorted, reverse sorted)?"
``` 
- Expected processing time: ~10 minutes.
- While the logs are continuously streamed, you can also check the logs at `logs/research_question_<ID>_verbose.log`.
- You can check the reproducible experimentation process under `workspace/research_<ID>/`.

## **Example 2**: Find good ML strategies for noisy data.

```bash
python3 -m curie.main \
  -q "For binary classification task for breast cancer Wisconsin dataset, ensemble methods \
  (e.g., Random Forests, Gradient Boosting) are more robust to added noise in the Breast Cancer \ 
  dataset compared to linear models like logistic regression."
```

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
```