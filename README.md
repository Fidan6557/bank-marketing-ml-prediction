# Bank Marketing ML Prediction

## Project Overview

This project focuses on predicting whether a bank client is likely to subscribe to a term deposit using machine learning.

The dataset is based on a direct marketing campaign of a Portuguese banking institution, where clients were contacted through phone calls. The main objective is to build a realistic machine learning pipeline that can help marketing teams prioritize customers who are more likely to respond positively to a campaign.

This project is designed as a **leakage-aware and imbalance-aware classification workflow**. Instead of only trying to maximize accuracy, the project focuses on realistic model evaluation, minority class performance, threshold tuning, and practical business interpretation.

---

## Business Problem

In bank marketing campaigns, contacting every customer randomly can be inefficient and expensive. A machine learning model can help identify customers with a higher probability of subscribing to a term deposit.

The model can support the bank in answering the following question:

> Which customers should be prioritized before making the next marketing call?

This makes the model useful as a **customer prioritization tool**, not as a perfect prediction system.

---

## Dataset

The project uses the Bank Marketing dataset, which contains client information, campaign-related attributes, and macroeconomic indicators.

Typical feature groups include:

- Client demographic information
- Contact and campaign information
- Previous campaign outcomes
- Macroeconomic indicators
- Target variable: whether the client subscribed to a term deposit

The target variable is binary:

| Class | Meaning |
|---|---|
| `0` | Client did not subscribe |
| `1` | Client subscribed |

---

## Key Machine Learning Challenge

The dataset is highly imbalanced. Most clients do not subscribe to the term deposit, while only a small percentage of clients belong to the positive class.

Because of this, accuracy alone is not a reliable metric. A model can achieve high accuracy by mostly predicting the majority class, but still perform poorly on the customers who actually subscribe.

For this reason, the project evaluates the model using metrics such as:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Balanced accuracy
- Confusion matrix

PR-AUC is especially important because it better reflects model performance on the minority class in imbalanced classification problems.

---

## Data Leakage Handling

One of the most important decisions in this project is the treatment of the `duration` feature.

The `duration` feature represents the length of the phone call. It is highly predictive because longer calls are often associated with more interested customers.

However, `duration` is only known **after the phone call ends**. If the goal is to predict whether a customer is likely to subscribe **before calling them**, then this feature is not available at prediction time.

Therefore, the final deployable model excludes `duration` to avoid data leakage.

A separate leakage experiment with `duration` is included only to show how much performance increases when post-call information is used.

---

## Project Workflow

The project follows a structured machine learning workflow:

1. Load and inspect the dataset
2. Perform exploratory data analysis
3. Analyze class imbalance
4. Clean and prepare the data
5. Apply feature engineering
6. Remove leakage-prone features from the final model
7. Split the data into train, validation, and test sets
8. Build preprocessing pipelines
9. Train multiple classification models
10. Compare models using validation metrics
11. Tune hyperparameters for Random Forest
12. Select the final model using validation PR-AUC
13. Tune the classification threshold on the validation set
14. Evaluate the final model on the test set
15. Analyze confusion matrix, ROC curve, PR curve, and feature importance
16. Compare the realistic model with a leakage experiment

---

## Feature Engineering

Several feature engineering steps were applied to improve model learning and handle special values properly.

Important engineered features include:

| Feature | Description |
|---|---|
| `was_contacted_before` | Indicates whether the client was contacted in a previous campaign |
| `pdays_clean` | Cleans the special `pdays = 999` value |
| `campaign_log` | Log-transformed campaign contact count |
| `previous_log` | Log-transformed previous contact count |
| `age_group` | Groups clients into age categories |
| `had_previous_campaign` | Indicates whether the client had any previous campaign interaction |
| `contact_intensity` | Combines current and previous contact frequency |
| `high_campaign_contact` | Flags clients contacted many times |
| `season` | Groups campaign months into seasons |
| `economic_pressure` | Combines macroeconomic indicators |
| `employment_euribor_interaction` | Captures interaction between employment and interest rate indicators |

These transformations help the models capture non-linear patterns and domain-specific signals.

---

## Models Used

Several machine learning models were trained and compared:

- Logistic Regression
- Random Forest
- Extra Trees
- Gradient Boosting
- XGBoost
- Tuned Random Forest

The final model was selected based on validation performance, especially PR-AUC, because the dataset is imbalanced.

---

## Final Deployable Model

The final deployable model excludes the `duration` feature and represents a realistic pre-call prediction scenario.

### Final Model Performance

| Model | Threshold | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tuned Random Forest without duration | 0.5375 | 0.876 | 0.462 | 0.615 | 0.528 | 0.816 | 0.483 |

### Interpretation

The final model achieves a reasonable balance between precision and recall for an imbalanced classification problem.

The recall value shows that the model is able to identify a meaningful portion of customers who are likely to subscribe. The precision value shows that not every predicted positive customer will actually subscribe, but the model still provides a more targeted strategy than random selection.

This makes the model useful for campaign prioritization.

---

## Leakage Experiment With Duration

A separate experiment was conducted using the `duration` feature.

This experiment shows how model performance increases when post-call information is included. However, this version is not suitable for real-world pre-call prediction.

### Leakage Experiment Performance

| Model | Threshold | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gradient Boosting with duration | 0.5000 | 0.917 | 0.659 | 0.545 | 0.597 | 0.949 | 0.659 |

### Why This Is Not the Final Model

The model with `duration` performs better because call duration is strongly related to the final outcome. However, this information is only available after the phone call has already happened.

Using this feature for pre-call prediction would create data leakage and produce unrealistic performance estimates.

For this reason, the final deployable model is the model without `duration`.

---

## Final Model vs Leakage Experiment

| Setup | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC | Usage |
|---|---:|---:|---:|---:|---:|---:|---|
| Without duration | 0.876 | 0.462 | 0.615 | 0.528 | 0.816 | 0.483 | Realistic pre-call prediction |
| With duration | 0.917 | 0.659 | 0.545 | 0.597 | 0.949 | 0.659 | Leakage experiment only |

The comparison demonstrates the importance of leakage-aware modeling. Higher scores are not always better if the model uses information that would not be available in real-world deployment.

---

## Evaluation Strategy

The dataset was split into three parts:

- Training set: used to train the models
- Validation set: used for model selection and threshold tuning
- Test set: used only once for final unbiased evaluation

This approach helps avoid overfitting to the test set and provides a more realistic estimate of model performance.

---

## Threshold Tuning

The default classification threshold of `0.5` is not always optimal for imbalanced datasets.

In this project, the classification threshold was tuned on the validation set to improve the balance between precision and recall.

This is important because, in a marketing campaign, the business may prefer higher recall to identify more potential subscribers, even if it means contacting some customers who will not subscribe.

---

## Feature Importance

Feature importance analysis showed that macroeconomic indicators and previous campaign outcomes had strong influence on the model predictions.

Important features included:

- `nr.employed`
- `poutcome_success`
- `euribor3m`
- `was_contacted_before`
- `cons.conf.idx`
- `age`
- `month`
- `pdays_clean`

This suggests that both customer-level behavior and broader economic conditions influence the likelihood of term deposit subscription.

---

## Business Interpretation

The model should be used as a decision-support tool for marketing prioritization.

Instead of contacting customers randomly, the bank can rank customers by predicted probability and prioritize those with higher subscription likelihood.

Possible business benefits include:

- More efficient campaign targeting
- Better use of call center resources
- Reduced unnecessary customer contact
- Improved focus on high-potential customers
- More data-driven marketing decisions

However, the model should not be treated as a fully automated decision-making system. Human review, business constraints, and campaign strategy should still be considered.

---

## Limitations

This project has several important limitations:

1. The final model does not use `duration`, so performance is lower but more realistic.
2. The dataset is imbalanced, making positive class prediction more difficult.
3. The model relies strongly on macroeconomic indicators, which may change over time.
4. Real production deployment would require monitoring for data drift.
5. Business cost assumptions are not directly included in the current model.
6. Further validation would be needed before using the model in a real banking environment.

---

## Possible Future Improvements

Future work could include:

- Profit-based threshold optimization
- Model calibration
- SHAP-based explainability
- More advanced gradient boosting models
- Time-based validation
- Monitoring for data drift
- Deployment as an API
- Dashboard for campaign managers
- Customer segmentation before modeling
- Cost-sensitive learning based on marketing campaign costs

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Joblib
- Jupyter Notebook / Google Colab

---

## Repository Structure

```text
bank-marketing-ml-prediction/
│
├── README.md
├── requirements.txt
└── bank_marketing_ml_project.ipynb
