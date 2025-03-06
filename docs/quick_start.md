# More Quick Start Questions

Input your research question or problem statement: `python3 -m curie.main -q "<Your research question>"`.

## **Example 1**: Understand the sorting algorithm efficiency.

```bash
python3 -m curie.main \
  -q "How does the choice of sorting algorithm impact runtime performance 
  across different input distributions (random, nearly sorted, reverse sorted)?"
``` 
- Expected processing time: ~10 minutes.
- While the logs are continuously streamed, you can also check the logs at `logs/research_question_<ID>_verbose.log`.
- You can check the reproducible experimentation process under `workspace/research_<ID>/`.

## **Example 2**: Find good ML strategies for noisy data.

```bash
python3 -m curie.main \
  -q "For binary classification task for breast cancer Wisconsin dataset, \
  ensemble methods (e.g., Random Forests, Gradient Boosting) are more \
  robust to added noise in the Breast Cancer dataset compared to linear models \
  like logistic regression."
```

<!-- 2025-03-05 21:07:44 - logger - construct_workflow_graph.py - INFO - Event value: The experiment is now concluded based on the available data. Gradient Boosting generally outperforms both Random Forest and Logistic Regression in most noise scenarios, although at high noise levels, models perform similarly. While future exploration could yield deeper insights, especially around extreme noise conditions or further model variants, these results provide a reasonable comprehension of ensemble methods' robustness in this context. -->

## **Example 3**: Optimize feature selection for classification tasks.
- *Basic question*: whether feature selection helps the model performace.
```bash
python3 -m curie.main \
  -q "For the Wine dataset (identifying wine cultivars using chemical properties), \
  using a genetic algorithm to select a subset of features will create a simpler, \
  more interpretable model that, when combined with an ensemble classifier \
  (e.g., Random Forest), outperforms models built on the full feature set. "
```

- *More advanced question*: Find the optimal feature selection.

```bash
python3 -m curie.main \
  -q "For the Wine dataset (identifying wine cultivars using chemical properties), when using 
  an ensemble classifier (e.g., Random Forest), what is the best subset of features that will create a simpler, more interpretable model that outperforms models  built on the full feature set. "
```

- [] We will include the conclusion and loggings from Curie soon.

