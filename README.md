# Machine Learning Labs

Lab notebooks from the Machine Learning module of my MSc in Artificial Intelligence at National College of Ireland (NCI), Dublin. Each lab is a self-contained Jupyter notebook covering one technique, from data preparation through to evaluation and written analysis.

## Labs

| Lab | Topic | Notebook |
|-----|-------|----------|
| Lab 3 | Ensemble learning for heart disease prediction | [Ensemble_Heart_Disease_Lab_COMPLETED.ipynb](Lab%203/Ensemble_Heart_Disease_Lab_COMPLETED.ipynb) |

## Lab 3 — Ensemble Learning for Heart Disease Prediction

This lab compares single classifiers with ensemble methods on a binary heart disease prediction task. It focuses on what matters in a medical screening setting: catching true cases (recall), not just overall accuracy.

**Dataset:** [Heart Disease (oktayrdeki) on Kaggle](https://www.kaggle.com/datasets/oktayrdeki/heart-disease), downloaded automatically with `kagglehub`. The dataset is not stored in this repo.

**What the notebook covers**

1. Data inspection, target detection and handling of missing values and duplicates
2. A leakage-safe preprocessing pipeline: imputation, scaling and one-hot encoding inside each model's `Pipeline`
3. Baselines and single models: Dummy, Logistic Regression, Decision Tree, kNN
4. A manual bootstrap experiment showing the ~63.2% unique-sample property behind bagging
5. Ensembles: Bagging, Random Forest, Extra Trees, AdaBoost, Gradient Boosting, Soft Voting and Stacking
6. 5-fold stratified cross-validation for reliable model comparison
7. Ensemble diversity analysis using pairwise disagreement and prediction correlation
8. Feature importance, comparing impurity-based importance with permutation importance
9. Decision-threshold tuning for clinical sensitivity (F2-optimised on out-of-fold predictions, so the test set stays untouched)
10. Random Forest hyperparameter tuning with `RandomizedSearchCV`
11. Manual error analysis of false negatives and false positives

**Key findings:** _to be added after the final run._

## Running the notebooks

```bash
pip install numpy pandas matplotlib scikit-learn kagglehub jupyter
jupyter notebook
```

The notebooks were developed with Python 3 and scikit-learn 1.x.

## Author

**Sourav Arun Mahajan**, MSc in Artificial Intelligence, National College of Ireland
GitHub: [@X2RSaM](https://github.com/X2RSaM)