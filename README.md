# Credit Card Fraud Detection

Comparing Logistic Regression vs Random Forest on a highly 
imbalanced fraud dataset — with a result I didn't expect.

## The Problem
284,807 transactions. Only 492 are fraud (0.17%).
Accuracy alone is useless here — a model predicting 
"not fraud" every time scores 99.8%. Meaningless.

## Approach
- Handled class imbalance via **undersampling**
- Feature scaling with **StandardScaler**
- Compared **Logistic Regression** vs **Random Forest**
- Dataset link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
  <img width="1661" height="572" alt="image" src="https://github.com/user-attachments/assets/eb4ff751-acad-42e9-9a7f-4cba6f7e827e" />


## Results

| Model               | Accuracy | Fraud Recall | F1  |
|---------------------|----------|--------------|-----|
| Logistic Regression | 93%      | 89%          | 0.93|
| Random Forest       | 92%      | 87%          | 0.91|

**LR outperformed RF** — hypothesis: undersampled dataset 
(~984 samples) too small for RF to leverage its strength.

## Key Takeaway
Model complexity ≠ better results.
Data size and quality determine which model wins.

## What's Next
- [ ] SMOTE instead of undersampling
- [ ] Rerun RF comparison with full data
- [ ] Add ROC-AUC curve

## Stack
Python · scikit-learn · pandas · NumPy · seaborn
