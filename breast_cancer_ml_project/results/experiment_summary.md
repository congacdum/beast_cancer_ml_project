# Experiment Summary

## Dataset

- Dataset: `data/breast_cancer_wisconsin.csv`
- Samples: 569
- Features before encoding: 30
- Features after encoding: 30
- Target column: `diagnosis`
- Target mapping:
  - Benign = 0
  - Malignant = 1

## Data Quality

- Quality score: 85/100
- Duplicate rows: 0
- Suspected outliers: 172
- Class balance: {'0': 357, '1': 212}
- Class imbalance detected: False

### Missing / Cleaning Decisions

- No missing values required imputation.

### Highly Correlated Features

- mean radius vs mean perimeter: 0.9979
- mean radius vs mean area: 0.9874
- mean perimeter vs mean area: 0.9865
- radius error vs perimeter error: 0.9728
- radius error vs area error: 0.9518
- mean radius vs worst radius: 0.9695
- mean perimeter vs worst radius: 0.9695
- mean area vs worst radius: 0.9627
- mean radius vs worst perimeter: 0.9651
- mean perimeter vs worst perimeter: 0.9704

Outliers and highly correlated features are reported for review and are not deleted automatically.

## Train/Test Split

- Test size: 0.2
- Random state: 42
- Stratify: `y`
- Scaling: model pipelines fit `StandardScaler` on training folds/train set only.

## Models

- Logistic Regression
- Decision Tree
- Random Forest
- SVM
- KNN
- Naive Bayes

## Hold-out Metric Table

| Model | Accuracy | Precision (M) | Recall (M) | F1-score (M) | Specificity (B) | MCC | ROC-AUC | PR-AUC | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 1.0000 | 0.9442 | 0.9954 | 0.9929 | 72 | 0 | 3 | 39 |
| SVM | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 1.0000 | 0.9442 | 0.9947 | 0.9927 | 72 | 0 | 3 | 39 |
| Logistic Regression | 0.9649 | 0.9750 | 0.9286 | 0.9512 | 0.9861 | 0.9245 | 0.9960 | 0.9943 | 71 | 1 | 3 | 39 |
| KNN | 0.9561 | 0.9744 | 0.9048 | 0.9383 | 0.9861 | 0.9058 | 0.9825 | 0.9754 | 71 | 1 | 4 | 38 |
| Naive Bayes | 0.9386 | 1.0000 | 0.8333 | 0.9091 | 1.0000 | 0.8715 | 0.9934 | 0.9890 | 72 | 0 | 7 | 35 |
| Decision Tree | 0.9035 | 0.9189 | 0.8095 | 0.8608 | 0.9583 | 0.7911 | 0.9153 | 0.8781 | 69 | 3 | 8 | 34 |

## 5-Fold Stratified Cross-Validation

| Model | Accuracy Mean | Accuracy Std | Precision (M) Mean | Precision (M) Std | Recall (M) Mean | Recall (M) Std | F1-score (M) Mean | F1-score (M) Std | MCC Mean | MCC Std | ROC-AUC Mean | ROC-AUC Std | PR-AUC Mean | PR-AUC Std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9736 | 0.0149 | 0.9771 | 0.0280 | 0.9529 | 0.0399 | 0.9640 | 0.0207 | 0.9444 | 0.0313 | 0.9958 | 0.0047 | 0.9949 | 0.0046 |
| Decision Tree | 0.9253 | 0.0306 | 0.9153 | 0.0428 | 0.8824 | 0.0526 | 0.8979 | 0.0418 | 0.8401 | 0.0658 | 0.9455 | 0.0340 | 0.9020 | 0.0629 |
| Random Forest | 0.9582 | 0.0235 | 0.9534 | 0.0365 | 0.9353 | 0.0471 | 0.9434 | 0.0320 | 0.9115 | 0.0498 | 0.9875 | 0.0071 | 0.9860 | 0.0071 |
| SVM | 0.9714 | 0.0054 | 0.9769 | 0.0210 | 0.9471 | 0.0343 | 0.9610 | 0.0082 | 0.9397 | 0.0116 | 0.9949 | 0.0050 | 0.9939 | 0.0047 |
| KNN | 0.9692 | 0.0082 | 0.9879 | 0.0149 | 0.9294 | 0.0235 | 0.9575 | 0.0116 | 0.9348 | 0.0176 | 0.9860 | 0.0142 | 0.9812 | 0.0156 |
| Naive Bayes | 0.9407 | 0.0266 | 0.9402 | 0.0418 | 0.9000 | 0.0546 | 0.9185 | 0.0361 | 0.8737 | 0.0571 | 0.9878 | 0.0075 | 0.9820 | 0.0117 |

## Hyperparameter Tuning

GridSearchCV was run for Logistic Regression, Decision Tree, Random Forest, SVM, and KNN
using Recall (M) as the tuning score. These tuning results are saved as experiment evidence
and do not silently replace the fixed baseline models.

| Model | Best CV Recall (M) | Best Params | Candidates | Scoring |
|---|---:|---|---:|---|
| Logistic Regression | 0.9647 | {'model__C': 0.1, 'model__class_weight': 'balanced'} | 6 | recall |
| Decision Tree | 0.8886 | {'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 2} | 24 | recall |
| Random Forest | 0.9296 | {'class_weight': None, 'max_depth': None, 'n_estimators': 300} | 12 | recall |
| SVM | 0.9472 | {'model__C': 2.0, 'model__class_weight': 'balanced', 'model__gamma': 'scale'} | 12 | recall |
| KNN | 0.9295 | {'model__n_neighbors': 3, 'model__weights': 'uniform'} | 8 | recall |

## Best Model

- Best model: Random Forest
- Selection rule: highest Recall (M), then lowest FN, then F1-score (M), then ROC-AUC, then Accuracy, then simpler/more interpretable model
- Decision threshold used by prediction artifacts: 0.30
- Note: cross-validation is run on the training split only. The final deployment model is
  selected from the untouched hold-out evaluation table and does not replace the fixed final
  hold-out selection rule.

## Confusion Matrix of Best Model

- TN: 72
- FP: 0
- FN: 3
- TP: 39

## Threshold Optimization

Threshold optimization is computed on the training split only. The hold-out test split remains
reserved for final model evaluation and model selection reporting.

| Threshold | Accuracy | Precision (M) | Recall (M) | F1-score (M) | Specificity (B) | MCC | ROC-AUC | PR-AUC | TN | FP | FN | TP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.3000 | 0.9912 | 0.9770 | 1.0000 | 0.9884 | 0.9860 | 0.9815 | 0.9999 | 0.9999 | 281 | 4 | 0 | 170 |
| 0.3500 | 0.9912 | 0.9770 | 1.0000 | 0.9884 | 0.9860 | 0.9815 | 0.9999 | 0.9999 | 281 | 4 | 0 | 170 |
| 0.4000 | 0.9934 | 0.9941 | 0.9882 | 0.9912 | 0.9965 | 0.9859 | 0.9999 | 0.9999 | 284 | 1 | 2 | 168 |
| 0.4500 | 0.9934 | 1.0000 | 0.9824 | 0.9911 | 1.0000 | 0.9860 | 0.9999 | 0.9999 | 285 | 0 | 3 | 167 |
| 0.5000 | 0.9934 | 1.0000 | 0.9824 | 0.9911 | 1.0000 | 0.9860 | 0.9999 | 0.9999 | 285 | 0 | 3 | 167 |
| 0.5500 | 0.9934 | 1.0000 | 0.9824 | 0.9911 | 1.0000 | 0.9860 | 0.9999 | 0.9999 | 285 | 0 | 3 | 167 |
| 0.6000 | 0.9912 | 1.0000 | 0.9765 | 0.9881 | 1.0000 | 0.9813 | 0.9999 | 0.9999 | 285 | 0 | 4 | 166 |

## Explainability

- SHAP summary: created
- Explainability report: `reports/model_explainability.md`

## Figures

- `reports/figures/confusion_matrix_logistic_regression.png`
- `reports/figures/roc_curve_logistic_regression.png`
- `reports/figures/confusion_matrix_decision_tree.png`
- `reports/figures/roc_curve_decision_tree.png`
- `reports/figures/confusion_matrix_random_forest.png`
- `reports/figures/roc_curve_random_forest.png`
- `reports/figures/confusion_matrix_svm.png`
- `reports/figures/roc_curve_svm.png`
- `reports/figures/confusion_matrix_knn.png`
- `reports/figures/roc_curve_knn.png`
- `reports/figures/confusion_matrix_naive_bayes.png`
- `reports/figures/roc_curve_naive_bayes.png`
- `reports/figures/threshold_optimization.png`
- `reports/figures/feature_importance.png`
- `reports/figures/shap_summary.png`
- `reports/figures/target_distribution.png`
- `reports/figures/correlation_heatmap.png`
- `reports/figures/confusion_matrix_best_model.png`
- `reports/figures/model_metric_comparison.png`
- `reports/figures/roc_curve.png`
- `reports/figures/decision_tree.png`

## Model / Artifact Files

- `models/best_model.joblib`
- `models/feature_names.joblib`
- `models/label_mapping.joblib`
- `models/decision_threshold.joblib`
- `models/logistic_regression.joblib`
- `models/decision_tree.joblib`
- `models/random_forest.joblib`
- `models/svm.joblib`
- `models/knn.joblib`
- `models/naive_bayes.joblib`

## Figure Generation Errors

- None

## Limitations

- No nested cross-validation yet.
- Hyperparameter tuning grid is intentionally small for classroom/demo runtime.
- Pytest covers core preprocessing, validation, and prediction paths; UI tests are still manual.
- SHAP can be skipped if the optional package/runtime cannot support it.

## Future Work

- Add 5-fold nested cross-validation for tuned models.
- Add broader GridSearchCV/RandomizedSearchCV.
- Expand pytest coverage for training report generation and Streamlit utility flows.
- Add SHAP/LIME explanations in the Streamlit prediction page.

## Medical Warning

This model is for learning/demo purposes only. Results do not replace medical advice,
clinical tests, or a doctor's diagnosis.
