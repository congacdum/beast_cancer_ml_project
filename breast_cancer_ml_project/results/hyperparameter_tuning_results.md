| Model | Best CV Recall (M) | Best Params | Candidates | Scoring |
|---|---:|---|---:|---|
| Logistic Regression | 0.9647 | {'model__C': 0.1, 'model__class_weight': 'balanced'} | 6 | recall |
| Decision Tree | 0.8886 | {'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 2} | 24 | recall |
| Random Forest | 0.9296 | {'class_weight': None, 'max_depth': None, 'n_estimators': 300} | 12 | recall |
| SVM | 0.9472 | {'model__C': 2.0, 'model__class_weight': 'balanced', 'model__gamma': 'scale'} | 12 | recall |
| KNN | 0.9295 | {'model__n_neighbors': 3, 'model__weights': 'uniform'} | 8 | recall |
