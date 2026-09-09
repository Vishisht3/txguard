from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import torch
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List
import os
import model_builder

app = FastAPI(
    title="TxGuard AI Risk Manager API",
    description="Concurrent, Async Microservice for Real-Time Fraud Inferences."
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_FEATURES = 29
MODEL_PATH = "models/txguard_model.pth"

# Thread pool executor for offloading non-blocking CPU matrix computations
executor = ThreadPoolExecutor(max_workers=4)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Missing model state dictionary: {MODEL_PATH}")

model = model_builder.RiskScorerModel(input_features=INPUT_FEATURES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

class TransactionPayload(BaseModel):
    features: List[float] = Field(..., description="Array of 29 scaled transaction features.")

def synchronous_inference(features: List[float]) -> float:
    """CPU-bound prediction logic executed inside worker thread pool."""
    x_tensor = torch.tensor([features], dtype=torch.float32).to(device)
    with torch.inference_mode():
        raw_logit = model(x_tensor).squeeze(-1)
        risk_score = torch.sigmoid(raw_logit).item()
    return risk_score

@app.post("/predict-risk", status_code=status.HTTP_200_OK)
async def predict_risk(payload: TransactionPayload):
    start_time = time.time()
    
    if len(payload.features) != INPUT_FEATURES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Dimension Mismatch: Expected {INPUT_FEATURES} features, got {len(payload.features)}"
        )

    try:
        # Offload sync PyTorch computation to executor thread pool to keep asyncio event loop non-blocking
        loop = asyncio.get_event_loop()
        risk_score = await loop.run_in_executor(executor, synchronous_inference, payload.features)

        latency_ms = (time.time() - start_time) * 1000
        action = "FLAG_AND_GATE" if risk_score >= 0.85 else "ALLOW"

        return {
            "status": "success",
            "risk_score": round(risk_score, 4),
            "action": action,
            "latency_ms": round(latency_ms, 2),
            "policy_applied": "PRECISION_THETA_0.85"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference Engine Failure: {str(e)}"
        )
