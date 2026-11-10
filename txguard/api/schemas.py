from typing import List, Optional
from pydantic import BaseModel, Field


class TransactionPayload(BaseModel):
    features: List[float] = Field(..., description="Array of 29 scaled transaction features.")


class PayPalCheckoutRequest(BaseModel):
    product_id: Optional[str] = Field(
        default="CH3-SONY-XM5",
        description="Channel3 catalog product ID (or custom item if omitted).",
    )
    item_title: Optional[str] = Field(default=None, description="Custom item title override.")
    amount: Optional[float] = Field(default=None, description="Order amount in USD.")
    currency: str = Field(default="USD", description="3-letter ISO currency code.")
    payer_email: str = Field(default="alex.morgan@verified-buyer.com", description="PayPal buyer email.")
    payer_name: str = Field(default="Alex Morgan", description="PayPal buyer full name.")
    payer_country: str = Field(default="US", description="2-letter buyer residency country.")
    shipping_country: str = Field(default="US", description="2-letter shipping destination country.")
    velocity_1h: int = Field(default=1, ge=1, le=50, description="Orders attempted by buyer in past 60m.")
    account_age_days: int = Field(default=620, ge=0, description="Age of PayPal buyer account in days.")
    country_mismatch: bool = Field(default=False, description="True if billing/shipping countries differ.")
    ip_proxy_detected: bool = Field(default=False, description="True if VPN/Tor/Proxy IP detected.")
    device_trust_score: float = Field(default=0.94, ge=0.0, le=1.0, description="Device telemetry score [0, 1].")
    is_digital_goods: bool = Field(default=False, description="True for instant digital delivery items.")
    email_domain_risk: float = Field(default=0.04, ge=0.0, le=1.0, description="Disposable email risk [0, 1].")


class StepUpChallengeSubmission(BaseModel):
    buyer_response: str = Field(..., description="Buyer's OTP code or contextual confirmation response.")
    otp_code: str = Field(default="849201", description="Expected OTP code for validation.")


class AnalystActionRequest(BaseModel):
    note: str = Field(default="Manual action executed from AG Grid Command Center.")


class AgentChatRequest(BaseModel):
    prompt: str = Field(..., description="Natural-language instruction for the Agentic Commerce Copilot.")


class RiskForensicsResponse(BaseModel):
    """Structured output schema for Gemini 3.8 Flash PayPal transaction risk forensics."""
    executive_summary: str = Field(
        description="Concise 2-sentence explanation of why this PayPal order received its risk score and decision."
    )
    risk_tier: str = Field(
        description="Tier classification: LOW_RISK_AUTO_CAPTURE, MEDIUM_RISK_STEP_UP_2FA, or HIGH_RISK_AUTO_VOID."
    )
    recommended_paypal_api_call: str = Field(
        description="Exact PayPal Orders/Payments v2 API endpoint action recommended."
    )
    key_signals: List[str] = Field(
        description="Top 3 forensic signals driving the risk score."
    )
    step_up_challenge_question: str = Field(
        description="Contextual verification challenge question if step-up 2FA is needed, or 'None required'."
    )
    chargeback_exposure_usd: float = Field(
        description="Estimated financial chargeback exposure prevented or managed in USD."
    )


class StepUpVerificationVerdict(BaseModel):
    """Structured output schema for evaluating a buyer's Step-Up 2FA response."""
    verified: bool = Field(description="True if buyer's verification response is credible and passes 2FA.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0.")
    reasoning: str = Field(description="Explanation of why the verification succeeded or failed.")
    paypal_action: str = Field(description="CAPTURE_AUTHORIZATION or VOID_AUTHORIZATION.")

