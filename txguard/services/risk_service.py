import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid

import torch

from txguard.api.schemas import PayPalCheckoutRequest
from txguard.config import (
    AUTO_BLOCK_THRESHOLD,
    INPUT_FEATURES,
    MODEL_PATH,
    STEP_UP_THRESHOLD,
)
from txguard.ml.trainer import bootstrap_calibrated_model
from txguard.services import catalog_service
from txguard.services.ai_copilot import GeminiRiskCopilot
from txguard.services.paypal_service import PayPalSandboxService


def extract_paypal_features(order_context: Dict[str, Any]) -> Tuple[List[float], List[Dict[str, Any]]]:
    """
    Maps a PayPal Orders v2 context (amount, payer telemetry, shipping/billing match, velocity,
    account age, IP proxy status) into the 29 scaled numerical features expected by RiskScorerModel,
    and returns human-readable feature attributions for Explainable AI (XAI).
    """
    amount = float(order_context.get("amount", 89.99))
    velocity_1h = int(order_context.get("velocity_1h", 1))
    account_age_days = int(order_context.get("account_age_days", 420))
    payer_country = str(order_context.get("payer_country", "US")).upper()
    shipping_country = str(order_context.get("shipping_country", "US")).upper()
    country_mismatch = bool(order_context.get("country_mismatch", False) or (payer_country != shipping_country))
    ip_proxy_detected = bool(order_context.get("ip_proxy_detected", False))
    device_trust_score = float(order_context.get("device_trust_score", 0.92))
    is_digital_goods = bool(order_context.get("is_digital_goods", False))
    email_domain_risk = float(order_context.get("email_domain_risk", 0.05))

    features = [0.0] * 29
    features[0] = 0.55   # V1: Behavioral consistency
    features[1] = -0.25  # V2: Short-window velocity
    features[3] = -0.45  # V4: Geo/address consistency
    features[10] = -0.30 # V11: Network anonymity indicator
    features[11] = 0.45  # V12: Account tenure & history
    features[13] = 0.50  # V14: Device fingerprint trust
    features[16] = 0.25  # V17: Merchant category risk

    attributions: List[Dict[str, Any]] = []

    if velocity_1h >= 5:
        features[0] -= 2.4
        features[1] += 2.3
        attributions.append({
            "feature": "1h Velocity",
            "value": f"{velocity_1h} tx/hr",
            "impact": "HIGH_RISK",
            "signal": f"High Velocity ({velocity_1h}/hr)",
            "detail": f"High Velocity ({velocity_1h}/hr)",
        })
    elif velocity_1h >= 3:
        features[0] -= 1.15
        features[1] += 1.05
        attributions.append({
            "feature": "1h Velocity",
            "value": f"{velocity_1h} tx/hr",
            "impact": "MEDIUM_RISK",
            "signal": f"Elevated Velocity ({velocity_1h}/hr)",
            "detail": f"Elevated Velocity ({velocity_1h}/hr)",
        })
    else:
        attributions.append({
            "feature": "1h Velocity",
            "value": f"{velocity_1h} tx/hr",
            "impact": "SAFE",
            "signal": f"Normal Velocity ({velocity_1h}/hr)",
            "detail": f"Normal Velocity ({velocity_1h}/hr)",
        })

    if country_mismatch:
        features[3] += 2.6
        route_str = f"{payer_country} → {shipping_country}" if payer_country != shipping_country else "Billing ≠ Shipping"
        attributions.append({
            "feature": "Region Match",
            "value": f"Mismatch ({route_str})",
            "impact": "HIGH_RISK",
            "signal": f"Cross-Border Mismatch ({route_str})",
            "detail": f"Cross-Border Mismatch ({route_str})",
        })
    else:
        attributions.append({
            "feature": "Region Match",
            "value": f"Match ({payer_country} → {shipping_country})",
            "impact": "SAFE",
            "signal": f"Region Match ({payer_country})",
            "detail": f"Region Match ({payer_country})",
        })

    if ip_proxy_detected:
        features[10] += 2.5
        attributions.append({
            "feature": "Network Origin",
            "value": "Proxy / Datacenter",
            "impact": "HIGH_RISK",
            "signal": "Proxy / VPN Detected",
            "detail": "Proxy / VPN Detected",
        })

    age_unit = "day" if account_age_days == 1 else "days"
    if account_age_days < 7:
        features[11] -= 2.6
        attributions.append({
            "feature": "Account Age",
            "value": f"{account_age_days} {age_unit}",
            "impact": "HIGH_RISK",
            "signal": f"New Account ({account_age_days}d)",
            "detail": f"New Account ({account_age_days}d)",
        })
    elif account_age_days < 45:
        features[11] -= 1.25
        attributions.append({
            "feature": "Account Age",
            "value": f"{account_age_days} {age_unit}",
            "impact": "MEDIUM_RISK",
            "signal": f"Young Account ({account_age_days}d)",
            "detail": f"Young Account ({account_age_days}d)",
        })
    else:
        attributions.append({
            "feature": "Account Age",
            "value": f"{account_age_days} {age_unit}",
            "impact": "SAFE",
            "signal": f"Established Account ({account_age_days}d)",
            "detail": f"Established Account ({account_age_days}d)",
        })

    if device_trust_score < 0.35:
        features[13] -= 2.9
        features[16] -= 2.1
        attributions.append({
            "feature": "Device Trust",
            "value": f"{device_trust_score:.2f} / 1.00",
            "impact": "HIGH_RISK",
            "signal": f"Untrusted Device ({device_trust_score:.2f})",
            "detail": f"Untrusted Device ({device_trust_score:.2f})",
        })
    elif device_trust_score < 0.70:
        features[13] -= 1.35
        features[16] -= 0.75
        attributions.append({
            "feature": "Device Trust",
            "value": f"{device_trust_score:.2f} / 1.00",
            "impact": "MEDIUM_RISK",
            "signal": f"Unverified Device ({device_trust_score:.2f})",
            "detail": f"Unverified Device ({device_trust_score:.2f})",
        })
    else:
        attributions.append({
            "feature": "Device Trust",
            "value": f"{device_trust_score:.2f} / 1.00",
            "impact": "SAFE",
            "signal": f"Trusted Device ({device_trust_score:.2f})",
            "detail": f"Trusted Device ({device_trust_score:.2f})",
        })

    if is_digital_goods:
        features[16] -= 0.7
    if email_domain_risk > 0.5:
        features[0] -= 0.9
        features[11] -= 0.8

    scaled_amount = (math.log1p(max(amount, 0.01)) - 4.0) / 1.5
    features[28] = round(float(scaled_amount), 4)
    if amount >= 1000:
        attributions.append({
            "feature": "Order Value",
            "value": f"${amount:,.2f} USD",
            "impact": "MEDIUM_RISK" if amount < 2000 else "HIGH_RISK",
            "signal": f"High Ticket (${amount:,.0f})",
            "detail": f"High Ticket (${amount:,.0f})",
        })

    features = [round(float(x), 4) for x in features]
    return features, attributions


def classify_cascading_tier(risk_score: float) -> Dict[str, str]:
    """Maps a continuous risk probability to TxGuard's Two-Tier Cascading PayPal Gate."""
    if risk_score >= AUTO_BLOCK_THRESHOLD:
        return {
            "action": "FLAG_AND_GATE",
            "decision": "AUTO_VOID",
            "paypal_action": "VOID_AUTHORIZATION",
            "policy_applied": "Auto-Void (≥ 0.85)",
        }
    if risk_score >= STEP_UP_THRESHOLD:
        return {
            "action": "STEP_UP_2FA",
            "decision": "PENDING_STEP_UP_2FA",
            "paypal_action": "HOLD_AUTHORIZATION_FOR_2FA",
            "policy_applied": "2FA Hold (0.45–0.85)",
        }
    return {
        "action": "ALLOW",
        "decision": "AUTO_CAPTURE",
        "paypal_action": "CAPTURE_AUTHORIZATION",
        "policy_applied": "Auto-Capture (< 0.45)",
    }


TRANSACTION_LEDGER: List[Dict[str, Any]] = []


class RiskOrchestrationService:
    """
    Core domain service orchestrating PyTorch Residual MLP inference, PayPal Orders/Authorizations v2,
    Gemini 3.8 Flash XAI forensics, and the stateful transaction ledger.
    """

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.model = bootstrap_calibrated_model(model_path=MODEL_PATH, input_features=INPUT_FEATURES)
        self.model.to(self.device)
        self.model.eval()

        self.paypal_client = PayPalSandboxService()
        self.gemini_copilot = GeminiRiskCopilot()
        self.ledger: List[Dict[str, Any]] = TRANSACTION_LEDGER

    def synchronous_inference(self, features: List[float]) -> float:
        """CPU/GPU-bound prediction logic executed inside worker thread pool."""
        x_tensor = torch.tensor([features], dtype=torch.float32).to(self.device)
        with torch.inference_mode():
            raw_logit = self.model(x_tensor).squeeze(-1)
            risk_score = torch.sigmoid(raw_logit).item()
        return float(risk_score)

    async def async_inference(self, features: List[float]) -> float:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self.synchronous_inference, features)

    def execute_checkout_pipeline(self, req: PayPalCheckoutRequest) -> Dict[str, Any]:
        """Executes the full PayPal Orders v2 + TxGuard PyTorch + Gemini XAI + Cascading Gate flow."""
        t0 = time.time()

        prod = catalog_service.get_product_by_id(req.product_id or "") if req.product_id else None
        raw_title = req.item_title or (prod["title"] if prod else "Merchant Order Item")
        item_title = raw_title.split(" ($")[0].strip()
        amount = float(req.amount if req.amount is not None else (prod["price"] if prod else 149.00))
        merchant_name = prod["merchant_name"] if prod else "TxGuard Protected Merchant"

        order_context = {
            "amount": amount,
            "velocity_1h": req.velocity_1h,
            "account_age_days": req.account_age_days,
            "payer_country": req.payer_country,
            "shipping_country": req.shipping_country,
            "country_mismatch": req.country_mismatch or (req.payer_country != req.shipping_country),
            "ip_proxy_detected": req.ip_proxy_detected,
            "device_trust_score": req.device_trust_score,
            "is_digital_goods": req.is_digital_goods,
            "email_domain_risk": req.email_domain_risk,
        }

        created_order = self.paypal_client.create_order(
            amount=amount,
            currency=req.currency,
            item_title=item_title,
            merchant_name=merchant_name,
            payer_email=req.payer_email,
            payer_name=req.payer_name,
            payer_country=req.payer_country,
            shipping_country=req.shipping_country,
        )
        order_id = created_order["id"]

        auth_res = self.paypal_client.authorize_order(
            order_id=order_id, amount=amount, currency=req.currency
        )
        authorization_id = auth_res["authorization_id"]

        features, attributions = extract_paypal_features(order_context)
        risk_score = self.synchronous_inference(features)
        tier_info = classify_cascading_tier(risk_score)
        decision = tier_info["decision"]

        settlement_receipt: Dict[str, Any] = {}
        paypal_status = "AUTHORIZED_PENDING_2FA"
        capture_id: Optional[str] = None

        if decision == "AUTO_CAPTURE":
            settlement_receipt = self.paypal_client.capture_authorization(
                authorization_id=authorization_id,
                amount=amount,
                currency=req.currency,
                note_to_payer=f"Auto-captured by TxGuard AI (risk={risk_score:.3f}).",
            )
            paypal_status = "CAPTURED"
            capture_id = settlement_receipt.get("capture_id")
        elif decision == "AUTO_VOID":
            settlement_receipt = self.paypal_client.void_authorization(
                authorization_id=authorization_id,
                reason=f"Auto-voided by TxGuard AI (risk={risk_score:.3f} >= {AUTO_BLOCK_THRESHOLD}).",
            )
            paypal_status = "VOIDED"
        else:
            settlement_receipt = {
                "authorization_id": authorization_id,
                "status": "HELD_FOR_STEP_UP_2FA",
                "otp_demo_code": "849201",
                "note": "Authorization held on PayPal Sandbox awaiting buyer Step-Up 2FA verification.",
            }
            paypal_status = "PENDING_STEP_UP_2FA"

        latency_ms = round((time.time() - t0) * 1000, 2)
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        record: Dict[str, Any] = {
            "tx_id": f"TXG-{uuid.uuid4().hex[:6].upper()}",
            "timestamp": now_iso,
            "order_id": order_id,
            "authorization_id": authorization_id,
            "capture_id": capture_id,
            "product_id": req.product_id or "CUSTOM",
            "item_title": item_title,
            "merchant_name": merchant_name,
            "amount": round(amount, 2),
            "currency": req.currency,
            "payer_email": req.payer_email,
            "payer_name": req.payer_name,
            "payer_country": req.payer_country,
            "shipping_country": req.shipping_country,
            "velocity_1h": req.velocity_1h,
            "account_age_days": req.account_age_days,
            "country_mismatch": order_context["country_mismatch"],
            "ip_proxy_detected": req.ip_proxy_detected,
            "device_trust_score": req.device_trust_score,
            "risk_score": round(risk_score, 4),
            "action": tier_info["action"],
            "decision": decision,
            "paypal_status": paypal_status,
            "policy_applied": tier_info["policy_applied"],
            "latency_ms": latency_ms,
            "api_mode": created_order.get("api_mode", self.paypal_client.mode_label),
            "attributions": attributions,
            "paypal_links": created_order.get("links", []),
            "settlement_receipt": settlement_receipt,
        }

        record["ai_forensics"] = self.gemini_copilot.generate_risk_forensics(record)
        self.ledger.insert(0, record)
        return record

    async def async_execute_checkout_pipeline(self, req: PayPalCheckoutRequest) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self.execute_checkout_pipeline, req)

    def compute_metrics(self) -> Dict[str, Any]:
        total = len(self.ledger)
        captured = [r for r in self.ledger if r["paypal_status"] == "CAPTURED"]
        voided = [r for r in self.ledger if r["paypal_status"] == "VOIDED"]
        step_up = [r for r in self.ledger if r["paypal_status"] == "PENDING_STEP_UP_2FA"]

        captured_vol = sum(r["amount"] for r in captured)
        fraud_prevented = sum(r["amount"] + 15.0 for r in voided)
        avg_latency = (
            round(sum(r["latency_ms"] for r in self.ledger) / total, 2) if total else 0.0
        )

        return {
            "total_orders": total,
            "auto_captured_count": len(captured),
            "step_up_count": len(step_up),
            "auto_voided_count": len(voided),
            "captured_volume_usd": round(captured_vol, 2),
            "fraud_prevented_usd": round(fraud_prevented, 2),
            "avg_latency_ms": avg_latency,
            "paypal_mode": self.paypal_client.mode_label,
            "gemini_mode": self.gemini_copilot.mode_label,
        }

    def find_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        for record in self.ledger:
            if record["order_id"] == order_id or record["tx_id"] == order_id:
                return record
        return None


risk_orchestrator = RiskOrchestrationService()

