# Overview (The user interface components are located in the templates folder.)
To make the developed credit risk prediction model easily accessible, a RESTful API was
implemented using the FastAPI framework. This API allows users or external systems
to send applicant data and receive a predicted probability of loan default. In addition
to predictions, the API provides Exploratory Data Analysis (EDA) endpoints to generate
summary statistics and plots for the underlying loan dataset.
The application serves as a bridge between the trained machine learning model and a user
friendly dashboard interface, enabling real-time prediction and interactive analysis. The
dashboard supports three key functionalities: Single Input Prediction, Batch Predic
tion, and EDA Exploration.
# Single Input Prediction
In the Single Input tab, users can manually enter applicant details via a structured form.
Below is an example of the input fields filled by the user:
{
"Saving Account Cash Amount": 5000,
"Avg. Current and Saving Account Balance": 12000,
"Vintage of CASA Account": 36,
"Income": 45000,
"Max EMI": 8000,
"Number of Loans": 2,
"Months Since Spending": 4,
"Min Transaction Value": 300,
"Minimum Monthly Average Balance": 10000,
"Source": "Referral",
"Loan Amount": 250000,
"Number of Inquiries": 1,
"Number of Dependents": 2
}
Upon submission, the API processes the input and returns the following output:
{
}
"Predicted Probability of Default": 0.2209,
"Risk Classification": "Low Risk"

# Batch Prediction 
The Batch Prediction tab allows users to upload a dataset containing multiple applicants.
The API processes the batch and returns a list of predicted probabilities and classifications.

# EDA
The EDA tab provides interactive tools to explore the dataset, including:
• Summary statistics and missing value analysis
• Lists of numerical and categorical features
• Univariate and bivariate plots for selected feature
