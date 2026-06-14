# Credit Risk Modelling

## Introduction 
This project delivers a comprehensive Credit Risk Modelling framework developed to
assess and manage the default risk for retail loan applicants using historical Loan data.
The model focuses on estimating the Probability of Default (PD) — a key component
of credit risk through a Logistic Regression approach offering both interpretability and
performance.
At its core, the system utilizes Logistic Regression to predict whether a borrower will default
based on features such as loan amount, interest rate, annual income, employment length,
credit history and loan purpose. The dataset covering applications from 2007 to 2014
undergoes extensive preprocessing: handling of missing values, outlier treatment and trans-
formation of both continuous (e.g., annual income, installment) and categorical (e.g., home
ownership, loan purpose) variables using techniques such as One-Hot Encoding.

## Data Extraction and Data Preprocessing 
Data is been used from Kaggle. It is the data of 7 years (2007 - 2014) based on Australia and further also used synthetic data.
The system applies domain-informed preprocessing techniques:
• Imputation of missing values using mean and replacing missing values directly with 0.
• Feature encoding i.e. One Hot Encoding was applied to nominal categorical variables.
• Weight of Evidence (WoE) transformation for categorical variables, ensuring monotonicity with the target variable, a critical requirement in risk scorecards
• IV (Information Value) used for feature selection to retain only predictive variables

This model is aligned with Basel II guideliness supporting the Expected Loss (ED) = PD * LGD * EAD, while for our model we primarly focuses on calculating PD but it also supports future extension for LGD and EAD

## Logistic Regression and Evaluation Metrics
The core model used is Logistic Regression chosen for its explainability, ease of interpretation and widespread regulatory acceptance. The logistic model outputs:
• Predicted Probability of Default (PD) for each borrower
• Scorecards, mapping borrower characteristics to risk scores using model coefficients and WoE bins

To validate the performance and discriminatory power of the model, multiple evaluation
metrics were used:
• AUC-ROC Curve: Measures model’s ability to distinguish between defaulters and non-defaulters.
• KS Statistic: Assesses the separation between the two classes.
• Confusion Matrix, F1 Score, Precision, Recall: Capture the model’s classification performance.
• Gini Coefficient: Quantifies inequality in score distribution.
• Score Distribution and Plots: Visual checks for model calibration and stability.

## Scorecard Generation 
After training, the logistic regression coefficients and WoE-transformed variables were used to generate a credit scorecard. This enables the PD model to produce credit scores that can be easily interpreted and applied in decision-making. Each borrower is assigned a score based on their binned features, which aligns with financial institution practices. The Credit Risk Modelling combines best practices from risk management, machine learning and software engineering to provide accurate, explainable and actionable credit assessments.
By integrating regulatory standards like Basel II, leveraging explainable models like Logistic Regression and incorporating techniques such as WoE and Information Value, this platform represents a strategic advancement in the field of credit analytics.

## Coding 
There is src folder which contains all feature engineering and preprocessing. for app.py there are different files for the preprocessing of the input features in the real time. Based on the WOE values of all fine bins we merged those bins having similar woe value and are monotonic and basically from every actual features of the data we get to the point where we are representing every situation in yes or no (1 or 0) format.
In the models folder present in the src folder we train our model also doing the evaluation part of the model for model we created a modification in the actual Logistic Regression so like with model.p_values we get the p_values for evry features we used as an input to the model so by doing that we rejected some features having p_values >0.05 becasue those features are not statistically significant for our model. In rejecting those features also there is a catch basically if that features (combination of some small bins) has a p_value > 0.05 in that case we cant directly reject, what we did that we checked for the majority like lets say if there are 5 features but all of them belong to the same category so if 4 of them are statistically insignificant we can reject that category as a whole but if there are more features significant than insignificant then we kept that whole category. so in short if I have to reject either I have to reject the whole category or I will not reject anything 

### MLFlow and Fastapi
We also implemented this model in MLflow for future experimenting and furthur improvement 
        to see the mlflow part in your terminal run "python main.py"
        then type "mlflow ui"
For creating the ui we used FastApi for better and fast results (also tried streamlit)
        to run this type "python app.py" + download all the html templates

## Steps to run in bash
1) python -m venv .venv
2) pip install requirements.txt
3) python main.py (in the file if you have to train do mode = "train", else mode = "mlflow"), the datadet will be needed to run ask the author for the dataset
4) python app.py (to run the frontend done in FastAPI)
