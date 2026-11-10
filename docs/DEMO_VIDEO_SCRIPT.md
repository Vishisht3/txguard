# TxGuard × PayPal AI — 3-Minute Demo Video Script & Storyboard

> **Target Duration:** 2 minutes 45 seconds (Under the strict 3-minute hackathon limit)  
> **Platform Requirements:** Shows the actual functioning application running locally on `http://localhost:8000`.

---

## ⏱️ Video Timeline & Scene Breakdown

### Scene 1: Zero-State Workbench & Problem Statement (0:00 – 0:30)
- **On Screen:** TxGuard Workbench (`http://localhost:8000`) booting in a clean zero-state RocqIDE desktop layout (`#2F6F6D` header, `#FAF8EA` cream parchment panes, `IBM Plex Sans` & `IBM Plex Mono` typography, `0 Screened` metrics).
- **Voiceover:**
  > *"Every year, e-commerce merchants lose billions to fraud and chargeback fees, while rigid binary decline rules insult legitimate buyers.*
  > *Welcome to **TxGuard**—a pre-settlement payment risk workbench built on the **PayPal Developer Platform**.*
  > *Instead of reacting after money moves, TxGuard operates directly inside the **PayPal Orders v2 authorization window**—evaluating every order in sub-millisecond latency to execute a Two-Tier Cascading Settlement Gate."*

---

### Scene 2: PayPal Orders v2 Authorization & Cascading Risk Gate (0:30 – 1:15)
- **On Screen:** **Command Center** tab — **Order Authorization** pane on the left, **Forensic Inspector** on the right, and **Transaction Ledger** below.
- **Action 1:** Select the Sony WH-1000XM5 item from the catalog dropdown, enter a verified buyer email (`buyer@verified.com`), and click **Authorize Order**.
  - Show the button transition (`Authorizing...`), the bottom status bar report round-trip completion (`Completed in 14ms`), the **Forensic Inspector** populate with `CAPTURED` (`#B7F0B1` mint-sage pill), and the new row appear in the **Transaction Ledger**.
- **Voiceover:**
  > *"Let's authorize a legitimate domestic order. When we click 'Authorize Order', TxGuard calls PayPal Orders v2 with `intent='AUTHORIZE'`, scores 29 telemetry features, and immediately captures the authorization via PayPal Payments v2 (`CAPTURED`)."*
- **Action 2:** Check **Country Mismatch** and **Proxy / VPN IP**, set 1h Velocity to `8`, and click **Authorize Order**.
  - Show the row appear in the ledger with a `VOIDED` (`#FECACA` soft rose) status pill and `$2,514.00 Prevented` in the metrics strip.
- **Voiceover:**
  > *"Now let's test an automated proxy attack. Risk spikes above the 0.85 threshold, so TxGuard immediately voids the PayPal authorization pre-settlement—preventing chargebacks and dispute fees."*

---

### Scene 3: Step-Up 2FA Hold & Contextual Resolution (1:15 – 1:55)
- **On Screen:** Submit a borderline cross-border order (or click **Simulate Traffic** in the header to stream live orders).
  - Select a `STEP-UP 2FA` (`#FDE68A` warm amber) row in the **Transaction Ledger**.
  - In the **Forensic Inspector**, show the **Step-Up 2FA Hold Verification** panel, enter OTP `849201`, and click **Verify 2FA** (or use the contextual **Capture** / **Void** buttons directly in the ledger row).
- **Voiceover:**
  > *"When a legitimate customer shops while traveling, standard systems decline them. TxGuard holds the PayPal authorization in `PENDING_STEP_UP_2FA`. Once the buyer confirms the challenge or OTP `849201`, clicking 'Verify 2FA' automatically captures the held PayPal authorization."*

---

### Scene 4: Agent Studio & PayPal Webhooks v1 (1:55 – 2:45)
- **On Screen:** Switch to the **Agent Studio** tab, run a natural-language prompt (`"Audit pending orders and summarize ledger metrics"`), then switch to the **Webhooks** tab and click **Send Webhook** for `PAYMENT.AUTHORIZATION.CREATED`. Finish by clicking **Export CSV** in the top header.
- **Voiceover:**
  > *"In **Agent Studio**, analysts can instruct Gemini 3.8 Flash via function calling to search catalog items, execute protected PayPal checkouts, or batch-audit held authorizations. And in the **Webhooks** tab, we can dispatch and verify signed PayPal Webhooks v1 events and export the full ledger to CSV. Thank you!"*
