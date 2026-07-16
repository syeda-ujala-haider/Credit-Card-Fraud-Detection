# Credit Card Fraud Detection

Comparing Undersampling vs SMOTE, and Logistic Regression vs 
Random Forest on a highly imbalanced fraud dataset.

## Dataset
284,807 transactions | 492 fraud (0.17%)  
Source: Kaggle Credit Card Fraud Detection
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
## Results
<img width="1664" height="577" alt="image" src="https://github.com/user-attachments/assets/225bb41e-a0e4-45b2-b0c8-e2daec9299a7" />

<img width="827" height="289" alt="image" src="https://github.com/user-attachments/assets/e59d5c6d-9d32-46aa-a9a2-e045cbd68f71" />


### Undersampling
| Model | Accuracy | Fraud Recall | F1 |
|---|---|---|---|
| Logistic Regression | 93% | 89% | 0.93 |
| Random Forest | 92% | 87% | 0.91 |

### SMOTE (fixed — split before resampling)
| Model               | Recall | Precision | F1   |
| ------------------- | ------ | --------- | ---- |
| Logistic Regression | 89.8%  | 13.4%     | 0.23 |
| Random Forest       | 82.7%  | 83.5%     | 0.83 |

- Undersampling: LR beat RF (small dataset, RF couldn't shine)
- SMOTE (initial attempt): RF showed 100% recall — turned out to be data 
  leakage from resampling before train/test split
- SMOTE (fixed): RF genuinely outperforms LR — 82.7% recall, 83.5% precision 
  vs LR's 89.8% recall but only 13.4% precision (too many false alarms)
- Lesson learned: always split data before applying SMOTE/oversampling

## What's Next
- [ ] Test on truly unseen real fraud cases
- [ ] Try cross-validation to verify RF score
- [ ] Explore XGBoost

## Stack
Python · scikit-learn · imbalanced-learn · pandas · seaborn
