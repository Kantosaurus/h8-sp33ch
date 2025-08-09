# Machine Learning Models Performance

## Performance Comparison Table

| Model | Model Name | Accuracy | AUC | Recall | Prec. | F1 | Kappa | MCC | TT (Sec) |
|-------|------------|----------|-----|--------|-------|----|----|-----|----------|
| et | Extra Trees Classifier | 0.8384 | 0.8997 | 0.7795 | 0.8855 | 0.8290 | 0.6769 | 0.6820 | 2.2820 |
| rf | Random Forest Classifier | 0.8349 | 0.8950 | 0.8002 | 0.8618 | 0.8297 | 0.6698 | 0.6718 | 8.9130 |
| xgboost | Extreme Gradient Boosting | 0.8144 | 0.8774 | 0.8116 | 0.8182 | 0.8148 | 0.6289 | 0.6290 | 4.9650 |
| catboost | CatBoost Classifier | 0.7899 | 0.8599 | 0.7477 | 0.8191 | 0.7816 | 0.5800 | 0.5824 | 54.2000 |
| qda | Quadratic Discriminant Analysis | 0.7870 | 0.8520 | 0.7582 | 0.8067 | 0.7816 | 0.5742 | 0.5754 | 0.3320 |
| lightgbm | Light Gradient Boosting Machine | 0.7843 | 0.8561 | 0.7486 | 0.8085 | 0.7773 | 0.5688 | 0.5705 | 9.2390 |
| dt | Decision Tree Classifier | 0.7567 | 0.7567 | 0.8125 | 0.7328 | 0.7705 | 0.5130 | 0.5163 | 2.6700 |
| lda | Linear Discriminant Analysis | 0.7460 | 0.8164 | 0.7038 | 0.7712 | 0.7358 | 0.4923 | 0.4944 | 0.5210 |
| gbc | Gradient Boosting Classifier | 0.7288 | 0.8040 | 0.6373 | 0.7829 | 0.7025 | 0.4582 | 0.4665 | 51.5740 |
| ridge | Ridge Classifier | 0.7139 | 0.0000 | 0.6506 | 0.7477 | 0.6956 | 0.4282 | 0.4320 | 0.1100 |
| knn | K Neighbors Classifier | - | - | - | - | - | - | - | - |

## List of Models

- Extra Trees Classifier (et)
- Random Forest Classifier (rf)
- Extreme Gradient Boosting (xgboost)
- CatBoost Classifier (catboost)
- Quadratic Discriminant Analysis (qda)
- Light Gradient Boosting Machine (lightgbm)
- Decision Tree Classifier (dt)
- Linear Discriminant Analysis (lda)
- Gradient Boosting Classifier (gbc)
- Ridge Classifier (ridge)
- K Neighbors Classifier (knn)

## Performance Summary

The models are ranked by accuracy:

1. **Extra Trees Classifier** - Best overall performance with 83.84% accuracy
2. **Random Forest Classifier** - Close second with 83.49% accuracy
3. **Extreme Gradient Boosting** - Third with 81.44% accuracy

The Extra Trees and Random Forest classifiers show the best balance of accuracy, precision, and recall, while also maintaining reasonable training times compared to ensemble methods like CatBoost and Gradient Boosting Classifier.
