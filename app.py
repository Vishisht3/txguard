from fastapi import FastAPI
from pydantic import BaseModel
import torch
import time
from typing import List
import model_builder

app = FastAPI(title="TxGuard AI Risk Manager API")

device = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_FEATURES = 29

model = model_builder.RiskScorerModel(input_features=INPUT_FEATURES)
MODEL_PATH = "/content/drive/MyDrive/Colab Notebooks/txguard/models/txguard_model.pth"
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

class TransactionPayload(BaseModel):
    features: List[float]

@app.post("/predict-risk")
def predict_risk(payload: TransactionPayload):
    start_time = time.time()
    
    if len(payload.features) != INPUT_FEATURES:
        return {"error": f"Expected {INPUT_FEATURES} features, got {len(payload.features)}"}

    x_tensor = torch.tensor([payload.features], dtype=torch.float32).to(device)
    
    with torch.inference_mode():
        raw_logit = model(x_tensor).squeeze()
        risk_score = torch.sigmoid(raw_logit).item()

    latency_ms = (time.time() - start_time) * 1000
    action = "FLAG_AND_GATE" if risk_score >= 0.85 else "ALLOW"

    return {
        "risk_score": round(risk_score, 4),
        "action": action,
        "latency_ms": round(latency_ms, 2),
        "policy_applied": "PRECISION_THETA_0.85"
    }
