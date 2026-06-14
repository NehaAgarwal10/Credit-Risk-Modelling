from fastapi import APIRouter, HTTPException
import pandas as pd
import joblib
import os
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from typing import List, Dict, Annotated
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from src.pipeline.predict_pipeline import transform_woe

router = APIRouter()

class SingleInput(BaseModel):
    SA_AMT_C_CASH: Annotated[float,Field(ge=0)]
    MEAN_TRV: Annotated[float, Field(ge=0)]
    AVG_CASA_BAL_AMT : Annotated[float, Field(ge = 0)]
    MIN_CASA_MDAB_AMT :  Annotated[float, Field(ge = 0)]
    VINTAGE_CASA : Annotated[float,Field(ge = 0)]
    income : Annotated[int, Field(ge = 0)]
    credit_score : Annotated[int, Field(ge = 300,le = 850)]
    loan_amount : Annotated[int, Field(ge = 0)]
    number_of_loans : int
    max_dpd_of_tradelines : int
    dti_ratio :  Annotated[float, Field(ge = 0)]
    product :  str
    months_since_opened : int
    inquiries :  int

def process_data(input_data: pd.DataFrame, is_batch: bool = False):
    """
    Handles preprocessing and feature alignment for prediction.
    """
    woe_transformed = transform_woe(input_data)

    return woe_transformed

# Load model once
MODEL_PATH = os.path.join("models", "Logistic_with_Synthetic.pkl")
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError as e:
    raise RuntimeError(f"Could not load model: {e}")

@router.post("/predict_single")
async def predict_single(item: SingleInput):
    try:
        input_df = pd.DataFrame([item.model_dump()])
        woe_input = process_data(input_df)
        probability = model.predict_proba(woe_input)[0][1]
        return {"predicted_probability_of_default": probability}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/predict_batch")
async def predict_batch(items: list[SingleInput]):
    try:
        input_df = pd.DataFrame([item.model_dump() for item in items])
        woe_input = process_data(input_df)
        probabilities = model.predict_proba(woe_input)[:, 1]
        return {"predicted_probabilities": probabilities.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
