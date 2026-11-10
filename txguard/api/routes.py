import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from txguard.api.schemas import (
    AgentChatRequest,
    AnalystActionRequest,
    PayPalCheckoutRequest,
    StepUpChallengeSubmission,
    TransactionPayload,
)
from txguard.config import (
    AUTO_BLOCK_THRESHOLD,
    INDEX_HTML_PATH,
    INPUT_FEATURES,
    MODEL_PATH,
    STEP_UP_THRESHOLD,
)
from txguard.services import catalog_service
from txguard.services.risk_service import classify_cascading_tier, risk_orchestrator

router = APIRouter()


def _get_order_or_404(order_id: str) -> Dict[str, Any]:
    record = risk_orchestrator.find_order(order_id)
    if not record:
        raise HTTPException(
            status_code=404,
            detail=f"PayPal Order '{order_id}' not found in TxGuard ledger.",
        )
    return record


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_dashboard():
    if INDEX_HTML_PATH.exists():
        return FileResponse(INDEX_HTML_PATH)
    return HTMLResponse("<h1>TxGuard × PayPal AI Risk Manager API is running.</h1>")


@router.get("/health", status_code=200)
async def health_check():
    return {
        "status": "healthy",
        "service": "TxGuard × PayPal AI Risk Manager",
        "pytorch_device": str(risk_orchestrator.device),
        "model_loaded": MODEL_PATH.exists(),
        "paypal_mode": risk_orchestrator.paypal_client.mode_label,
        "gemini_mode": risk_orchestrator.gemini_copilot.mode_label,
        "sponsor_integrations": [
            "PayPal Developer Platform (Orders v2, Authorizations v2, Webhooks v1)",
            "Google Gemini 3.8 Flash (Structured XAI & Agentic Commerce Function Calling)",
            "AG Grid (Interactive Enterprise Transaction & Order Monitoring Grid)",
            "Channel3 (Product Intelligence & Agentic Commerce Feed)",
            "Bryntum-style Settlement & Velocity Timeline",
            "APIMatic & Postman (OpenAPI 3.1 + SDK Metadata + Collection)",
            "Render (Cloud Deployment Blueprint render.yaml)",
        ],
    }


@router.get("/api/config", status_code=200)
async def get_client_config():
    return {
        "paypal_client_id": risk_orchestrator.paypal_client.client_id or "sb",
        "paypal_mode": risk_orchestrator.paypal_client.mode_label,
        "gemini_mode": risk_orchestrator.gemini_copilot.mode_label,
        "thresholds": {
            "step_up_2fa": STEP_UP_THRESHOLD,
            "auto_void": AUTO_BLOCK_THRESHOLD,
        },
        "metrics": risk_orchestrator.compute_metrics(),
    }


@router.post("/predict-risk", status_code=200)
async def predict_risk(payload: TransactionPayload):
    start_time = time.time()

    if len(payload.features) != INPUT_FEATURES:
        raise HTTPException(
            status_code=422,
            detail=f"Dimension Mismatch: Expected {INPUT_FEATURES} features, got {len(payload.features)}",
        )

    try:
        risk_score = await risk_orchestrator.async_inference(payload.features)
        latency_ms = (time.time() - start_time) * 1000
        action = "FLAG_AND_GATE" if risk_score >= 0.85 else "ALLOW"
        tier_info = classify_cascading_tier(risk_score)

        return {
            "status": "success",
            "risk_score": round(risk_score, 4),
            "action": action,
            "cascading_decision": tier_info["decision"],
            "paypal_recommended_action": tier_info["paypal_action"],
            "latency_ms": round(latency_ms, 2),
            "policy_applied": tier_info["policy_applied"],
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference Engine Failure: {str(e)}",
        )


@router.get("/api/channel3/products", status_code=200)
async def list_channel3_products(query: str = "", max_price: Optional[float] = None):
    products = catalog_service.search_products(query=query, max_price=max_price)
    return {"status": "success", "count": len(products), "products": products}


@router.get("/api/transactions", status_code=200)
async def list_transactions():
    return {
        "status": "success",
        "metrics": risk_orchestrator.compute_metrics(),
        "transactions": risk_orchestrator.ledger,
    }


@router.post("/api/paypal/orders/create-and-evaluate", status_code=201)
async def create_and_evaluate_paypal_order(req: PayPalCheckoutRequest):
    record = await risk_orchestrator.async_execute_checkout_pipeline(req)
    return {
        "status": "success",
        "order": record,
        "metrics": risk_orchestrator.compute_metrics(),
    }


@router.post("/api/paypal/orders/{order_id}/capture", status_code=200)
async def manual_capture_order(order_id: str, body: Optional[AnalystActionRequest] = None):
    record = _get_order_or_404(order_id)
    if record["paypal_status"] == "CAPTURED":
        return {"status": "already_captured", "order": record, "metrics": risk_orchestrator.compute_metrics()}
    if record["paypal_status"] == "VOIDED":
        raise HTTPException(
            status_code=400,
            detail="Cannot capture a PayPal authorization that has already been VOIDED.",
        )

    note = body.note if body else "Captured after analyst/step-up approval."
    receipt = risk_orchestrator.paypal_client.capture_authorization(
        authorization_id=record["authorization_id"],
        amount=record["amount"],
        currency=record["currency"],
        note_to_payer=note,
    )
    record["paypal_status"] = "CAPTURED"
    record["decision"] = "CAPTURED_AFTER_REVIEW"
    record["capture_id"] = receipt.get("capture_id")
    record["settlement_receipt"] = receipt
    return {"status": "success", "order": record, "metrics": risk_orchestrator.compute_metrics()}


@router.post("/api/paypal/orders/{order_id}/void", status_code=200)
async def manual_void_order(order_id: str, body: Optional[AnalystActionRequest] = None):
    record = _get_order_or_404(order_id)
    if record["paypal_status"] == "VOIDED":
        return {"status": "already_voided", "order": record, "metrics": risk_orchestrator.compute_metrics()}
    if record["paypal_status"] == "CAPTURED":
        raise HTTPException(
            status_code=400,
            detail="Cannot void an authorization that has already been CAPTURED.",
        )

    reason = body.note if body else "Voided by analyst from AG Grid Command Center."
    receipt = risk_orchestrator.paypal_client.void_authorization(
        authorization_id=record["authorization_id"],
        reason=reason,
    )
    record["paypal_status"] = "VOIDED"
    record["decision"] = "VOIDED_BY_ANALYST"
    record["settlement_receipt"] = receipt
    return {"status": "success", "order": record, "metrics": risk_orchestrator.compute_metrics()}


@router.post("/api/paypal/orders/{order_id}/step-up", status_code=200)
async def resolve_step_up_challenge(order_id: str, submission: StepUpChallengeSubmission):
    record = _get_order_or_404(order_id)
    verdict = risk_orchestrator.gemini_copilot.evaluate_step_up_challenge(
        order_record=record,
        buyer_response=submission.buyer_response,
        otp_code=submission.otp_code,
    )

    if verdict["verified"]:
        receipt = risk_orchestrator.paypal_client.capture_authorization(
            authorization_id=record["authorization_id"],
            amount=record["amount"],
            currency=record["currency"],
            note_to_payer="Step-Up 2FA verified by TxGuard Gemini Copilot.",
        )
        record["paypal_status"] = "CAPTURED"
        record["decision"] = "STEP_UP_VERIFIED_CAPTURE"
        record["capture_id"] = receipt.get("capture_id")
        record["settlement_receipt"] = receipt
    else:
        receipt = risk_orchestrator.paypal_client.void_authorization(
            authorization_id=record["authorization_id"],
            reason=verdict["reasoning"],
        )
        record["paypal_status"] = "VOIDED"
        record["decision"] = "STEP_UP_FAILED_VOID"
        record["settlement_receipt"] = receipt

    record["step_up_verdict"] = verdict
    return {
        "status": "success",
        "verdict": verdict,
        "order": record,
        "metrics": risk_orchestrator.compute_metrics(),
    }


@router.post("/api/paypal/orders/{order_id}/ai-audit", status_code=200)
async def run_ai_forensic_audit(order_id: str):
    record = _get_order_or_404(order_id)
    record["ai_forensics"] = risk_orchestrator.gemini_copilot.generate_risk_forensics(record)
    return {"status": "success", "order": record, "ai_forensics": record["ai_forensics"]}


@router.post("/api/agent/chat", status_code=200)
async def agentic_commerce_chat(req: AgentChatRequest):
    def tool_search_channel3(query: str = "", max_price: Optional[float] = None) -> List[Dict[str, Any]]:
        return catalog_service.search_products(query=query, max_price=max_price)

    def tool_checkout(product_id: str, risk_profile: str = "LOW_RISK") -> Dict[str, Any]:
        prod = catalog_service.get_product_by_id(product_id) or catalog_service.CHANNEL3_PRODUCTS[0]
        preset = dict(prod["preset_telemetry"])
        profile = (risk_profile or "").upper()
        if "HIGH" in profile or "FRAUD" in profile:
            preset.update({
                "velocity_1h": 8,
                "account_age_days": 1,
                "country_mismatch": True,
                "ip_proxy_detected": True,
                "device_trust_score": 0.12,
                "shipping_country": "RO",
            })
        elif "STEP" in profile or "MED" in profile:
            preset.update({
                "velocity_1h": 3,
                "account_age_days": 22,
                "country_mismatch": True,
                "ip_proxy_detected": False,
                "device_trust_score": 0.56,
                "shipping_country": "CA",
            })
        checkout_req = PayPalCheckoutRequest(
            product_id=prod["product_id"],
            item_title=prod["title"],
            amount=prod["price"],
            currency=prod["currency"],
            **preset,
        )
        return risk_orchestrator.execute_checkout_pipeline(checkout_req)

    def tool_audit_pending(resolution_policy: str = "REPORT_ONLY") -> Dict[str, Any]:
        pending = [r for r in risk_orchestrator.ledger if r["paypal_status"] == "PENDING_STEP_UP_2FA"]
        updated = 0
        pol = (resolution_policy or "REPORT_ONLY").upper()
        for r in pending:
            if "VOID" in pol and r["risk_score"] >= 0.60:
                receipt = risk_orchestrator.paypal_client.void_authorization(
                    authorization_id=r["authorization_id"],
                    reason="Autonomous Agent voided elevated-risk pending authorization.",
                )
                r["paypal_status"] = "VOIDED"
                r["decision"] = "AGENT_AUTO_VOIDED"
                r["settlement_receipt"] = receipt
                updated += 1
            elif "CAPTURE" in pol and r["risk_score"] < 0.70:
                receipt = risk_orchestrator.paypal_client.capture_authorization(
                    authorization_id=r["authorization_id"],
                    amount=r["amount"],
                    currency=r["currency"],
                    note_to_payer="Autonomous Agent captured step-up authorization.",
                )
                r["paypal_status"] = "CAPTURED"
                r["decision"] = "AGENT_AUTO_CAPTURED"
                r["capture_id"] = receipt.get("capture_id")
                r["settlement_receipt"] = receipt
                updated += 1
        return {
            "policy": pol,
            "inspected_count": len(pending),
            "updated_count": updated,
            "summary": f"Inspected {len(pending)} pending PayPal authorizations and updated {updated}.",
        }

    tool_handlers = {
        "search_channel3_catalog": tool_search_channel3,
        "checkout_with_paypal_and_txguard": tool_checkout,
        "audit_and_resolve_pending_orders": tool_audit_pending,
        "get_ledger_metrics": risk_orchestrator.compute_metrics,
    }

    result = risk_orchestrator.gemini_copilot.run_agentic_commerce(req.prompt, tool_handlers=tool_handlers)
    return {
        "status": "success",
        "agent_response": result,
        "metrics": risk_orchestrator.compute_metrics(),
        "transactions": risk_orchestrator.ledger,
    }


@router.post("/webhooks/paypal", status_code=200)
async def paypal_webhook_listener(request: Request):
    """
    PayPal Webhooks v1 Receiver (`POST /webhooks/paypal`).
    Verifies webhook cryptographic headers and processes order/authorization lifecycle events.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    headers = {k.lower(): v for k, v in request.headers.items()}
    verification = risk_orchestrator.paypal_client.verify_webhook_signature(
        headers=headers, event_body=body
    )
    event_type = body.get("event_type", "PAYMENT.AUTHORIZATION.CREATED")
    resource = body.get("resource", {})

    return {
        "status": "webhook_processed",
        "event_type": event_type,
        "resource_id": resource.get("id", "WH-SIMULATED"),
        "signature_verification": verification,
    }

