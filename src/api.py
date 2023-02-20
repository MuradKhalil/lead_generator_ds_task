import pandas as pd
from fastapi import FastAPI, File, UploadFile
from joblib import load

from custom_transformers import SelectX

app = FastAPI(title="ML API")

@app.on_event("startup")
def load_model():
    app.model = load("../models/model.joblib")

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    id = df['fakeID']
    preds = pd.Series(app.model.predict(df), name='prediction')
    output = pd.concat([id, preds], axis=1)
    output = output.to_json(orient='records')
    output = eval(output)
    return {"prediction": output}