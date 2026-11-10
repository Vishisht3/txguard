import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
import torch

from txguard.config import INDEX_HTML_PATH, MODEL_PATH, STATIC_DIR
from txguard.main import app
from txguard.ml.engine import CostSensitiveFocalLoss
from txguard.ml.model import RiskScorerModel
from txguard.services import catalog_service
from txguard.services.risk_service import TRANSACTION_LEDGER, extract_paypal_features, risk_orchestrator

client = TestClient(app)


def test_zero_state_boot_and_empty_initial_ledger():
    risk_orchestrator.ledger.clear()
    assert TRANSACTION_LEDGER == []
    tx_res = client.get("/api/transactions")
    assert tx_res.status_code == 200
    payload = tx_res.json()
    assert payload["transactions"] == []
    assert payload["metrics"]["total_orders"] == 0
    assert payload["metrics"]["captured_volume_usd"] == 0.0
    assert payload["metrics"]["fraud_prevented_usd"] == 0.0
    assert payload["metrics"]["step_up_count"] == 0
    assert payload["metrics"]["avg_latency_ms"] == 0.0


def test_root_paths_and_static_assets():
    assert MODEL_PATH.is_absolute()
    assert INDEX_HTML_PATH.is_absolute()
    assert (STATIC_DIR / "styles.css").exists()
    assert (STATIC_DIR / "app.js").exists()

    index_res = client.get("/")
    assert index_res.status_code == 200
    assert "/static/styles.css" in index_res.text
    assert "/static/app.js" in index_res.text

    css_res = client.get("/static/styles.css")
    assert css_res.status_code == 200

    js_res = client.get("/static/app.js")
    assert js_res.status_code == 200


def test_health_and_config_endpoints():
    health_res = client.get("/health")
    assert health_res.status_code == 200
    data = health_res.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

    cfg_res = client.get("/api/config")
    assert cfg_res.status_code == 200
    cfg = cfg_res.json()
    assert "thresholds" in cfg
    assert cfg["thresholds"]["step_up_2fa"] == 0.45
    assert cfg["thresholds"]["auto_void"] == 0.85


def test_predict_risk_endpoint_and_validation():
    legit_features, _ = extract_paypal_features({
        "amount": 120.0,
        "velocity_1h": 1,
        "account_age_days": 900,
        "country_mismatch": False,
        "ip_proxy_detected": False,
        "device_trust_score": 0.96,
    })
    res = client.post("/predict-risk", json={"features": legit_features})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "success"
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["action"] == "ALLOW"

    bad_res = client.post("/predict-risk", json={"features": [0.1, 0.2]})
    assert bad_res.status_code == 422


def test_paypal_orders_v2_three_tier_cascading_gate():
    low_res = client.post(
        "/api/paypal/orders/create-and-evaluate",
        json={
            "product_id": "CH3-SONY-XM5",
            "amount": 348.0,
            "payer_email": "verified@buyer.com",
            "payer_country": "US",
            "shipping_country": "US",
            "velocity_1h": 1,
            "account_age_days": 850,
            "country_mismatch": False,
            "ip_proxy_detected": False,
            "device_trust_score": 0.96,
        },
    )
    assert low_res.status_code == 201
    low_order = low_res.json()["order"]
    assert low_order["paypal_status"] == "CAPTURED"
    assert low_order["decision"] == "AUTO_CAPTURE"
    assert low_order["capture_id"] is not None

    high_res = client.post(
        "/api/paypal/orders/create-and-evaluate",
        json={
            "product_id": "CH3-NVIDIA-RTX5090",
            "amount": 2499.0,
            "payer_email": "bot99@anon.xyz",
            "payer_country": "US",
            "shipping_country": "RO",
            "velocity_1h": 8,
            "account_age_days": 1,
            "country_mismatch": True,
            "ip_proxy_detected": True,
            "device_trust_score": 0.12,
        },
    )
    assert high_res.status_code == 201
    high_order = high_res.json()["order"]
    assert high_order["paypal_status"] == "VOIDED"
    assert high_order["decision"] == "AUTO_VOID"


def test_step_up_2fa_verification_and_manual_actions():
    res = client.post(
        "/api/paypal/orders/create-and-evaluate",
        json={
            "product_id": "CH3-APPLE-IPAD-M4",
            "amount": 1299.0,
            "payer_email": "jordan.lee@fastmail.io",
            "payer_country": "US",
            "shipping_country": "CA",
            "velocity_1h": 3,
            "account_age_days": 25,
            "country_mismatch": True,
            "ip_proxy_detected": False,
            "device_trust_score": 0.58,
        },
    )
    assert res.status_code == 201
    order = res.json()["order"]
    order_id = order["order_id"]

    step_res = client.post(
        f"/api/paypal/orders/{order_id}/step-up",
        json={"buyer_response": "849201 - Confirming travel purchase", "otp_code": "849201"},
    )
    assert step_res.status_code == 200
    step_data = step_res.json()
    assert step_data["verdict"]["verified"] is True
    assert step_data["order"]["paypal_status"] == "CAPTURED"


def test_agentic_commerce_and_channel3_and_webhooks():
    catalog = catalog_service.get_catalog()
    assert len(catalog) >= 5

    cat_res = client.get("/api/channel3/products?query=Sony")
    assert cat_res.status_code == 200
    assert cat_res.json()["count"] >= 1

    agent_res = client.post(
        "/api/agent/chat",
        json={"prompt": "Buy the Sony XM5 headphones via PayPal Sandbox and run TxGuard protection"},
    )
    assert agent_res.status_code == 200
    agent_body = agent_res.json()["agent_response"]
    assert len(agent_body["tool_calls"]) >= 1

    wh_res = client.post(
        "/webhooks/paypal",
        json={
            "id": "WH-TEST-1",
            "event_type": "PAYMENT.AUTHORIZATION.CREATED",
            "resource": {"id": "0VF123456"},
        },
    )
    assert wh_res.status_code == 200
    assert wh_res.json()["status"] == "webhook_processed"


def test_pytorch_model_and_focal_loss():
    model = RiskScorerModel(input_features=29, hidden_units=32)
    x = torch.randn(8, 29)
    y = torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
    logits = model(x).squeeze(-1)
    loss_fn = CostSensitiveFocalLoss(alpha=0.75, gamma=4.0, fn_weight=8.0)
    loss = loss_fn(logits, y)
    assert loss.item() > 0.0
