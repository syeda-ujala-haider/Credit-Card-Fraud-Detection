# Credit Card Fraud Detection

Comparing Undersampling vs SMOTE, and Logistic Regression vs 
Random Forest on a highly imbalanced fraud dataset.

## Dataset
284,807 transactions | 492 fraud (0.17%)  
Source: Kaggle Credit Card Fraud Detection
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
## Results
<img width="1664" height="577" alt="image" src="https://github.com/user-attachments/assets/225bb41e-a0e4-45b2-b0c8-e2daec9299a7" />

<img width="1670" height="565" alt="image" src="https://github.com/user-attachments/assets/e0f624f9-80c5-4bbf-94b5-4842e07e7b6a" />

### Undersampling
| Model | Accuracy | Fraud Recall | F1 |
|---|---|---|---|
| Logistic Regression | 93% | 89% | 0.93 |
| Random Forest | 92% | 87% | 0.91 |

### SMOTE
| Model | Accuracy | Fraud Recall | F1 |
|---|---|---|---|
| Logistic Regression | 98% | 97% | 0.98 |
| Random Forest | 100% | 100% | 1.00 |

## Key Findings
- Undersampling: LR beat RF (small dataset, RF couldn't shine)
- SMOTE: RF dominated with perfect recall
- RF's 100% score likely reflects overfitting on synthetic 
  SMOTE data — real-world testing needed

## What's Next
- [ ] Test on truly unseen real fraud cases
- [ ] Try cross-validation to verify RF score
- [ ] Explore XGBoost

## Stack
Python · scikit-learn · imbalanced-learn · pandas · seaborn
