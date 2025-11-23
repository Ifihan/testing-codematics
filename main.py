import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
from typing import List

app = FastAPI()

# Load model and metadata
try:
    with open('iris_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model_info.json', 'r') as f:
        model_info = json.load(f)
    feature_names = model_info['feature_names']
    target_names = model_info['target_names']
except FileNotFoundError:
    # Train a simple model if the pickled model is not found
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names.tolist()

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Create dummy model_info
    model_info = {
        'feature_names': feature_names,
        'target_names': target_names,
        'accuracy': 0.95,
        'n_samples': len(X),
        'n_features': X.shape[1]
    }


    # Save the model and metadata for future use
    with open('iris_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)

except Exception as e:
    raise RuntimeError(f"Error loading model or metadata: {e}")


# Pydantic model for input data
class IrisData(BaseModel):
    sepal_length_cm: float
    sepal_width_cm: float
    petal_length_cm: float
    petal_width_cm: float

    @classmethod
    def from_list(cls, data: List[float]):
        if len(data) != 4:
            raise ValueError("Input list must have 4 elements.")
        return cls(
            sepal_length_cm=data[0],
            sepal_width_cm=data[1],
            petal_length_cm=data[2],
            petal_width_cm=data[3]
        )


@app.post("/predict")
async def predict(data: IrisData):
    try:
        # Convert input data to numpy array
        input_data = np.array([
            data.sepal_length_cm,
            data.sepal_width_cm,
            data.petal_length_cm,
            data.petal_width_cm
        ]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0].tolist()

        # Return prediction and probabilities
        return {
            "predicted_class": target_names[prediction],
            "probabilities": dict(zip(target_names, probabilities))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Auto-generated startup code
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
