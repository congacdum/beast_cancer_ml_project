# Best Model Summary

## Best Model

- Best model: Random Forest
- Selection rule: highest Recall (M), then lowest FN, then F1-score (M), then ROC-AUC, then Accuracy, then simpler/more interpretable model
- Reason: Random Forest was selected by the medical-priority rule: Recall (M)=0.9286, FN=3, F1-score (M)=0.9630, ROC-AUC=0.9954, Accuracy=0.9737. Recall for Malignant is prioritized because False Negative cases are more dangerous than False Positive cases in breast cancer screening.

## Metrics at Default Model Decision Rule

| Metric | Value |
|---|---:|
| Accuracy | 0.9737 |
| Precision (M) | 1.0000 |
| Recall (M) | 0.9286 |
| F1-score (M) | 0.9630 |
| Specificity (B) | 1.0000 |
| MCC | 0.9442 |
| ROC-AUC | 0.9954 |
| PR-AUC | 0.9929 |

## Confusion Matrix

- TN: 72
- FP: 0
- FN: 3
- TP: 39

## Threshold Optimization

Optimized decision threshold on the training split: 0.30. At this threshold, Recall (M)=1.0000, FN=0, FP=4.

## False Negative Note

FN is the number of malignant cases predicted as benign. In medical screening, FN is more
dangerous than FP because it can delay follow-up examination or treatment. This is why
model selection prioritizes Recall (M) and FN before general Accuracy.

## Medical Warning

This model is for learning/demo purposes only. It does not replace a doctor, clinical
testing, or formal medical diagnosis.
