from fastapi import APIRouter, HTTPException
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import numpy as np

router = APIRouter()

# ---------------------------
# Helpers
# ---------------------------

def get_eda_dataframe():
    """Load and return EDA dataframe."""
    raw_data_file = 'data/raw/Synthetic_data_1.csv'
    
    if not os.path.exists(raw_data_file):
        raise HTTPException(status_code=500, detail="Raw data not found.")
    
    return pd.read_csv(raw_data_file, index_col=0)

def create_and_encode_plot(plot_func, **kwargs):
    """Helper to create a plot and encode it to base64."""
    buffer = BytesIO()
    plt.figure(figsize=(10, 6))
    plot_func(**kwargs)
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    plt.close('all')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# ---------------------------
# Routes
# ---------------------------

@router.get("/overview")
async def get_eda_overview():
    df = get_eda_dataframe()
    summary_stats = df.describe().replace([np.inf, -np.inf, np.nan], None).to_dict()
    missing_values = (df.isnull().sum() / len(df) * 100).to_dict()
    numeric_df = df.select_dtypes(include=[np.number])

    correlation_matrix = {}
    if not numeric_df.empty:
        correlation_matrix = numeric_df.corr().round(2).replace([np.inf, -np.inf, np.nan], None).to_dict()

    return {
        "summary_statistics": summary_stats,
        "missing_values": {k: v for k, v in missing_values.items() if v > 0},
        "correlation_matrix": correlation_matrix,
    }

@router.get("/features")
async def get_eda_features():
    df = get_eda_dataframe()
    numerical = df.select_dtypes(include=np.number).columns.tolist()
    categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
    target_col = 'good_bad'
    if target_col in numerical: numerical.remove(target_col)
    if target_col in categorical: categorical.remove(target_col)

    return {"numerical": numerical, "categorical": categorical}

@router.get("/univariate_plot")
async def get_univariate_plot(feature_name: str):
    df = get_eda_dataframe()
    if feature_name not in df.columns:
        raise HTTPException(status_code=404, detail="Feature not found.")
    if df[feature_name].dtype == 'object' or df[feature_name].nunique() < 20:
        plot = create_and_encode_plot(
            sns.countplot, x=feature_name, data=df, order=df[feature_name].value_counts().index
        )
    else:
        plot = create_and_encode_plot(sns.histplot, data=df, x=feature_name, kde=True)
    return {"plot": plot}

@router.get("/bivariate_plot")
async def get_bivariate_plot(feature_name: str):
    df = get_eda_dataframe()
    target_col = "good_bad"
    if feature_name not in df.columns or target_col not in df.columns:
        raise HTTPException(status_code=404, detail="Feature/target not found.")
    if df[feature_name].dtype == 'object' or df[feature_name].nunique() < 20:
        plot = create_and_encode_plot(sns.countplot, x=feature_name, hue=target_col, data=df)
    else:
        plot = create_and_encode_plot(sns.boxplot, x=target_col, y=feature_name, data=df)
    return {"plot": plot}
