# TxGuard × PayPal AI — Official Hackathon Submission Package

> **Hackathon Track:** PayPal AI Hackathon  
> **Submission Title:** TxGuard: Real-Time PayPal Authorization Risk Orchestration & Agentic Commerce Engine  
> **Elevator Pitch:** A pre-settlement AI fraud gate and autonomous agentic commerce copilot for the PayPal Developer Platform that combines PyTorch Deep Residual Networks with Google Gemini 3.8 Flash and AG Grid to eliminate chargebacks without insulting legitimate buyers.

---

## 💡 Inspiration
Modern digital commerce moves at lightning speed, but payment risk management remains broken. E-commerce merchants lose billions to fraud and chargeback fees, yet standard rule engines and binary classifiers rely on blunt thresholds that decline legitimate customers—especially international shoppers, cross-border travelers, and digital buyers.

When exploring the **PayPal Developer Platform**, we recognized an architectural opportunity: the **PayPal Orders v2 & Payments v2 Authorization lifecycle** (`intent="AUTHORIZE"`). By decoupling order authorization from final settlement capture, PayPal provides a secure window where an AI engine can inspect, score, and arbitrate transactions **before funds are settled**.

We built **TxGuard** to turn that authorization window into an intelligent, explainable, and autonomous fraud defense engine.

---

## 🎯 What It Does
TxGuard operates directly between PayPal Order Authorization and Settlement Capture:
1. **Pre-Settlement Neural Risk Scoring:** As soon as an order is authorized via PayPal Orders v2 (`POST /v2/checkout/orders` and `/authorize`), TxGuard's **PyTorch Deep Tabular Residual MLP (`RiskScorerModel`)** scores 29 scaled transaction, behavioral, and device signals in under 2 milliseconds.
2. **Two-Tier Cascading Settlement Gate:**
   - **Low Risk ($\text{risk} < 0.45$):** Immediately captures funds via PayPal Payments v2 (`POST /v2/payments/authorizations/{id}/capture`) with zero customer friction.
   - **Medium / Borderline Risk ($0.45 \le \text{risk} < 0.85$):** Holds the PayPal authorization in `PENDING_STEP_UP_2FA` rather than declining the customer. Gemini 3.8 Flash generates an interactive verification challenge; once verified by the buyer, TxGuard automatically captures the held authorization.
   - **High Risk ($\text{risk} \ge 0.85$):** Automatically voids the PayPal authorization via `POST /v2/payments/authorizations/{id}/void`, preventing chargebacks and dispute fees pre-settlement.
3. **Gemini 3.8 Flash Explainable AI (XAI) Forensics:** Translates complex neural embeddings into structured JSON briefs explaining the exact behavioral drivers and recommended PayPal REST endpoint actions.
4. **Autonomous Agentic Commerce Copilot:** Using Gemini **Function Calling** (`google-genai` Interactions API), shoppers and risk analysts can converse with TxGuard to search **Channel3** catalog products, execute protected PayPal sandbox orders, and batch-audit held authorizations.
5. **AG Grid Enterprise Command Center:** A real-time web studio with dynamic risk-gauge progress bars, status filters, quick search, CSV export, and inline PayPal API action buttons.

---

## 🛠️ How We Built It
- **PayPal Developer Platform:** Integrated PayPal Sandbox OAuth 2.0 (`/v1/oauth2/token`), Orders v2 (`POST /v2/checkout/orders` with `intent="AUTHORIZE"`), Payments v2 Authorizations (`/capture` and `/void`), and Webhooks v1 signature verification (`POST /v1/notifications/verify-webhook-signature`). Includes a zero-config Hybrid Sandbox Emulator so judges can test every flow out of the box.
- **Deep Tabular PyTorch MLP:** Custom neural architecture featuring `ResidualBlock` skip connections and **Mish** activations trained with `CostSensitiveFocalLoss` ($\gamma=4.0$, $8\times$ False Negative penalty), online hard-negative mining, and Validation PR-AUC early stopping.
- **Google Gemini 3.8 Flash:** Implemented via the official `google-genai` SDK Interactions API (`client.interactions.create`) using Structured Outputs for forensic reports and Function Calling for the Agentic Commerce Copilot.
- **AG Grid Enterprise Ledger:** Real-time client grid featuring custom HTML cell renderers, responsive status filtering, and CSV export.
- **Channel3 Intelligence Feed:** Integrated merchant trust ratings and catalog items with 1-click risk simulation presets.
- **APIMatic & Postman:** Strictly typed OpenAPI 3.1.0 specification with APIMatic SDK generation manifest and an 8-request Postman collection.
- **Render Deployment:** Infrastructure-as-Code `render.yaml` Blueprint for 1-click cloud hosting.

---

## 🚧 Challenges We Ran Into
1. **The Extreme Imbalance Dilemma:** Real-world credit card and checkout fraud is heavily imbalanced (~0.17% fraud rate). Initial standard BCE loss caused models to simply guess non-fraud. Implementing `CostSensitiveFocalLoss` with an $8\times$ False Negative penalty combined with online hard-negative mining stabilized recall above 98%.
2. **Preventing Buyer Insults on False Positives:** A single hard cutoff forced an impossible tradeoff between fraud recall and false declines. Designing the **Two-Tier Cascading Gate** with interactive Step-Up 2FA hold allowed borderline transactions to be verified without hard declines.
3. **Synchronous Matrix Inference in Async Web Services:** Deep learning tensor operations block the Python `asyncio` event loop. We offloaded PyTorch matrix computations to a dedicated `ThreadPoolExecutor` so the FastAPI server maintains sub-5ms total API latency.

---

## 🏆 Accomplishments That We're Proud Of
- **98.98% Total Fraud Interception:** Caught 97 out of 98 fraud attempts in validation benchmarks while keeping hard false positives down to **0.032%**.
- **True Pre-Settlement Protection:** Unlike post-settlement chargeback tools that react after funds move, TxGuard acts on authorized PayPal funds before capture.
- **Zero-Config Evaluation for Judges:** Built an auto-calibrating model bootstrapper and stateful sandbox emulator so anyone can clone the repo and test the full application immediately without waiting for API key provisioning.

---

## 📚 What We Learned
- How to structure modern PayPal Orders v2 applications with `intent="AUTHORIZE"` and handle payment authorizations programmatically.
- How to combine fast numerical neural networks (PyTorch) with generative reasoning models (Gemini 3.8 Flash) using structured outputs and function calling.
- How AG Grid makes complex financial ledgers intuitive and actionable for risk analysts.

---

## 🔮 What's Next for TxGuard
- Multi-currency PayPal Payouts fraud screening for marketplace sellers.
- Integration with PayPal Disputes API to automatically assemble evidence packages using TxGuard XAI logs when a customer files a claim.
- Production edge deployment on Render with Redis clustering.

---

## 🏷️ Built With
`paypal-api`, `paypal-orders-v2`, `paypal-sandbox`, `python`, `fastapi`, `pytorch`, `google-gemini`, `gemini-3.8-flash`, `google-genai`, `ag-grid`, `channel3`, `apimatic`, `postman`, `render`, `tailwind-css`

