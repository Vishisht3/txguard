# APIMatic SDK & Developer Portal Integration (`TxGuard × PayPal AI`)

TxGuard exposes a strictly typed **OpenAPI 3.1.0** specification at `/openapi.json` designed for zero-friction SDK generation and API validation via **[APIMatic](https://www.apimatic.io/)**.

## Quick Workflow with APIMatic

1. **Export OpenAPI 3.1 Spec**:
   Start the server (`uvicorn app:app --port 8000`) and download `http://localhost:8000/openapi.json` (or use the pre-exported `docs/openapi.json`).
2. **Validate & Transform with APIMatic CLI / Dashboard**:
   Import `docs/openapi.json` and `docs/apimatic/APIMatic-Manifest.json` into the APIMatic Dashboard or CLI:
   ```bash
   apimatic api:validate --file docs/openapi.json
   apimatic sdk:generate --file docs/openapi.json --language python
   apimatic portal:serve --folder ./docs/apimatic
   ```
3. **Generated SDK Capabilities**:
   - Strongly typed client methods for `create_and_evaluate_paypal_order`, `resolve_step_up_challenge`, `agentic_commerce_chat`, and `predict_risk`.
   - Automatic OAuth/Bearer header handling and exponential backoff retries for high-throughput merchant checkouts.

