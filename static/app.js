/**
 * TxGuard — Frontend Controller (ES6+)
 * CoqIDE / RocqIDE GTK3 Workbench • Zero-State Boot • PayPal Orders v2
 */

const state = {
  catalog: [],
  ledger: [],
  selectedOrderId: null,
};

const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

/* ==========================================================================
   1. Centralized API Client & Status Bar Controller
   ========================================================================== */
function updateStatusBar(message) {
  const msgEl = document.getElementById("status-message");
  if (msgEl && message) {
    msgEl.textContent = message;
  }
  const countEl = document.getElementById("status-ledger-count");
  if (countEl) {
    const n = state.ledger.length;
    countEl.textContent = `Ledger: ${n} ${n === 1 ? "Order" : "Orders"}`;
  }
  const exportBtn = document.getElementById("btn-export");
  if (exportBtn) {
    exportBtn.disabled = state.ledger.length === 0;
  }
}

const apiClient = {
  async request(url, options = {}, statusMessage = "Processing request...") {
    const t0 = performance.now();
    updateStatusBar(statusMessage);

    try {
      const response = await fetch(url, {
        headers: { "Content-Type": "application/json", ...(options.headers || {}) },
        ...options,
      });

      const elapsedMs = Math.round(performance.now() - t0);
      let payload = null;
      try {
        payload = await response.json();
      } catch (_err) {
        payload = null;
      }

      if (!response.ok) {
        const detail =
          (payload && (payload.detail || payload.message)) ||
          `HTTP ${response.status} (${response.statusText})`;
        updateStatusBar(`Error (${response.status}) • ${elapsedMs}ms`);
        throw new Error(detail);
      }

      updateStatusBar(`Completed in ${elapsedMs}ms`);
      return payload;
    } catch (error) {
      if (!error.message.includes("HTTP")) {
        updateStatusBar(`Request failed: ${error.message}`);
      }
      throw error;
    }
  },

  getCatalog() {
    return this.request("/api/channel3/products", { method: "GET" }, "Loading catalog...");
  },

  getTransactions() {
    return this.request("/api/transactions", { method: "GET" }, "Syncing transaction ledger...");
  },

  authorizeOrder(body) {
    return this.request(
      "/api/paypal/orders/create-and-evaluate",
      { method: "POST", body: JSON.stringify(body) },
      "Processing PayPal Order..."
    );
  },

  captureOrder(orderId, note) {
    return this.request(
      `/api/paypal/orders/${encodeURIComponent(orderId)}/capture`,
      { method: "POST", body: JSON.stringify({ note }) },
      `Capturing authorization for ${orderId}...`
    );
  },

  voidOrder(orderId, note) {
    return this.request(
      `/api/paypal/orders/${encodeURIComponent(orderId)}/void`,
      { method: "POST", body: JSON.stringify({ note }) },
      `Voiding authorization for ${orderId}...`
    );
  },

  verifyStepUp(orderId, buyerResponse, otpCode = "849201") {
    return this.request(
      `/api/paypal/orders/${encodeURIComponent(orderId)}/step-up`,
      {
        method: "POST",
        body: JSON.stringify({ buyer_response: buyerResponse, otp_code: otpCode }),
      },
      `Verifying Step-Up 2FA for ${orderId}...`
    );
  },

  runAgent(prompt) {
    return this.request(
      "/api/agent/chat",
      { method: "POST", body: JSON.stringify({ prompt }) },
      "Running Agent Studio..."
    );
  },

  sendWebhook(eventType, orderId) {
    return this.request(
      "/webhooks/paypal",
      {
        method: "POST",
        body: JSON.stringify({
          event_type: eventType,
          resource: { id: orderId, status: "COMPLETED" },
        }),
      },
      `Dispatching webhook ${eventType}...`
    );
  },
};

/* ==========================================================================
   2. Utility & Formatting Helpers
   ========================================================================== */
function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function cleanItemTitle(rawTitle) {
  return String(rawTitle || "Merchant Order Item")
    .replace(/\s*\(\$[0-9,.]+[^)]*\)\s*/g, "")
    .trim();
}

function formatPolicyOutcome(record) {
  const score = Number(record?.risk_score || 0);
  if (score >= 0.85 || record?.decision === "AUTO_VOID") {
    return "Auto-Void (≥ 0.85)";
  }
  if (score >= 0.45 || record?.decision === "PENDING_STEP_UP_2FA") {
    return "2FA Hold (0.45–0.85)";
  }
  return "Auto-Capture (< 0.45)";
}

function buildTerseSignals(record) {
  const attrs = record?.attributions || [];
  const signals = [];

  for (const a of attrs) {
    if (a.signal) {
      signals.push(a.signal);
    } else if (a.detail && a.detail.split(" ").length <= 5) {
      signals.push(a.detail);
    }
  }

  if (signals.length > 0) {
    return signals;
  }

  const fallback = [];
  const vel = Number(record?.velocity_1h || 1);
  if (vel >= 5) fallback.push(`High Velocity (${vel}/hr)`);
  else if (vel >= 3) fallback.push(`Elevated Velocity (${vel}/hr)`);
  else fallback.push(`Normal Velocity (${vel}/hr)`);

  if (record?.country_mismatch) {
    fallback.push(
      `Cross-Border Mismatch (${record.payer_country || "US"} → ${record.shipping_country || "US"})`
    );
  } else {
    fallback.push(`Region Match (${record?.payer_country || "US"})`);
  }

  if (record?.ip_proxy_detected) {
    fallback.push("Proxy / VPN Detected");
  }

  const age = Number(record?.account_age_days ?? 365);
  if (age < 7) fallback.push(`New Account (${age}d)`);
  else if (age < 45) fallback.push(`Young Account (${age}d)`);
  else fallback.push(`Established Account (${age}d)`);

  return fallback;
}

function renderStatusBadge(status) {
  if (status === "CAPTURED") {
    return `<span class="badge badge-approved">CAPTURED</span>`;
  }
  if (status === "PENDING_STEP_UP_2FA" || status === "AUTHORIZED_PENDING_2FA") {
    return `<span class="badge badge-2fa">2FA HOLD</span>`;
  }
  if (status === "VOIDED") {
    return `<span class="badge badge-rejected">VOIDED</span>`;
  }
  return `<span class="badge">${escapeHtml(status || "UNKNOWN")}</span>`;
}

function matchesStatusFilter(paypalStatus, filterVal) {
  if (!filterVal || filterVal === "ALL") return true;
  if (filterVal === "APPROVED" || filterVal === "CAPTURED") {
    return paypalStatus === "CAPTURED";
  }
  if (filterVal === "CHALLENGE_2FA" || filterVal === "PENDING_STEP_UP_2FA") {
    return paypalStatus === "PENDING_STEP_UP_2FA" || paypalStatus === "AUTHORIZED_PENDING_2FA";
  }
  if (filterVal === "REJECTED" || filterVal === "VOIDED") {
    return paypalStatus === "VOIDED";
  }
  return paypalStatus === filterVal;
}

async function withButtonLoading(buttonEl, loadingText, asyncFn) {
  if (!buttonEl) return asyncFn();
  const originalText = buttonEl.textContent;
  buttonEl.disabled = true;
  buttonEl.textContent = loadingText;
  try {
    return await asyncFn();
  } finally {
    buttonEl.disabled = false;
    buttonEl.textContent = originalText;
  }
}

/* ==========================================================================
   3. Navigation & Metrics Strip
   ========================================================================== */
function switchTab(targetTab) {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === targetTab);
  });

  document.querySelectorAll(".tab-pane").forEach((pane) => {
    pane.classList.toggle("active", pane.id === `tab-${targetTab}`);
  });
}

function renderMetrics(metrics) {
  if (!metrics) return;
  const total = Number(metrics.total_orders || 0);
  const captured = Number(metrics.captured_volume_usd || 0);
  const prevented = Number(metrics.fraud_prevented_usd || 0);
  const holds = Number(metrics.step_up_count || 0);
  const latency = Number(metrics.avg_latency_ms || 0);

  const elScreened = document.getElementById("stat-screened");
  const elCaptured = document.getElementById("stat-captured");
  const elPrevented = document.getElementById("stat-prevented");
  const elHolds = document.getElementById("stat-holds");
  const elLatency = document.getElementById("stat-latency");

  if (elScreened) elScreened.textContent = String(total);
  if (elCaptured) elCaptured.textContent = `$${captured.toFixed(2)}`;
  if (elPrevented) elPrevented.textContent = `$${prevented.toFixed(2)}`;
  if (elHolds) elHolds.textContent = String(holds);
  if (elLatency) elLatency.textContent = total > 0 ? `${latency.toFixed(1)} ms` : "—";
}

/* ==========================================================================
   4. Catalog & Order Authorization Form
   ========================================================================== */
async function initCatalogSelect() {
  try {
    const data = await apiClient.getCatalog();
    state.catalog = data.products || [];
    const selectEl = document.getElementById("catalog-select");
    if (!selectEl) return;

    selectEl.innerHTML =
      `<option value="">Select catalog item or enter custom item</option>` +
      state.catalog
        .map(
          (item) =>
            `<option value="${escapeHtml(item.product_id)}">${escapeHtml(
              cleanItemTitle(item.title)
            )} — $${Number(item.price).toFixed(2)}</option>`
        )
        .join("");
  } catch (err) {
    updateStatusBar(`Failed to load catalog: ${err.message}`);
  }
}

function handleCatalogSelectChange() {
  const selectEl = document.getElementById("catalog-select");
  const productId = selectEl ? selectEl.value : "";
  if (!productId) return;

  const product = state.catalog.find((p) => p.product_id === productId);
  if (!product) return;

  document.getElementById("item-title").value = cleanItemTitle(product.title);
  document.getElementById("amount").value = Number(product.price).toFixed(2);
}

function clearOrderForm() {
  const form = document.getElementById("order-form");
  if (form) form.reset();
  document.querySelectorAll("#order-form input").forEach((input) => {
    input.classList.remove("invalid");
  });
  const errBanner = document.getElementById("order-form-error");
  if (errBanner) errBanner.remove();
}

function showOrderFormError(message) {
  let banner = document.getElementById("order-form-error");
  const form = document.getElementById("order-form");
  if (!form) return;

  if (!message) {
    if (banner) banner.remove();
    return;
  }

  if (!banner) {
    banner = document.createElement("div");
    banner.id = "order-form-error";
    banner.className = "alert-banner";
    form.prepend(banner);
  }
  banner.textContent = message;
}

function validateAndReadOrderForm() {
  const titleEl = document.getElementById("item-title");
  const amountEl = document.getElementById("amount");
  const emailEl = document.getElementById("buyer-email");

  [titleEl, amountEl, emailEl].forEach((el) => el?.classList.remove("invalid"));

  const rawTitle = (titleEl?.value || "").trim();
  const itemTitle = cleanItemTitle(rawTitle);
  const amount = parseFloat(amountEl?.value || "");
  const payerEmail = (emailEl?.value || "").trim();

  if (!rawTitle) {
    titleEl?.classList.add("invalid");
    throw new Error("Enter a non-empty Item Title.");
  }
  if (!Number.isFinite(amount) || amount <= 0) {
    amountEl?.classList.add("invalid");
    throw new Error("Enter a positive USD Amount greater than $0.00.");
  }
  if (!EMAIL_REGEX.test(payerEmail)) {
    emailEl?.classList.add("invalid");
    throw new Error("Enter a valid Buyer Email address (e.g., buyer@domain.com).");
  }

  const productId = document.getElementById("catalog-select")?.value.trim() || null;
  const payerName = document.getElementById("buyer-name")?.value.trim() || "PayPal Buyer";
  const payerCountry = (document.getElementById("billing-country")?.value.trim() || "US").toUpperCase();
  const shippingCountry = (document.getElementById("shipping-country")?.value.trim() || "US").toUpperCase();

  const velocityVal = parseInt(document.getElementById("velocity-1h")?.value || "1", 10);
  const velocity1h = Number.isFinite(velocityVal) && velocityVal >= 1 ? Math.min(50, velocityVal) : 1;

  const ageVal = parseInt(document.getElementById("account-age")?.value || "365", 10);
  const accountAgeDays = Number.isFinite(ageVal) && ageVal >= 0 ? ageVal : 365;

  const trustVal = parseFloat(document.getElementById("device-trust")?.value || "0.90");
  const deviceTrustScore = Number.isFinite(trustVal) ? Math.min(1, Math.max(0, trustVal)) : 0.9;

  const emailRiskVal = parseFloat(document.getElementById("email-risk")?.value || "0.05");
  const emailDomainRisk = Number.isFinite(emailRiskVal) ? Math.min(1, Math.max(0, emailRiskVal)) : 0.05;

  const countryMismatch =
    Boolean(document.getElementById("country-mismatch")?.checked) ||
    payerCountry !== shippingCountry;
  const ipProxyDetected = Boolean(document.getElementById("proxy-ip")?.checked);
  const isDigitalGoods = Boolean(document.getElementById("digital-goods")?.checked);

  return {
    product_id: productId,
    item_title: itemTitle,
    amount,
    currency: "USD",
    payer_email: payerEmail,
    payer_name: payerName,
    payer_country: payerCountry,
    shipping_country: shippingCountry,
    velocity_1h: velocity1h,
    account_age_days: accountAgeDays,
    country_mismatch: countryMismatch,
    ip_proxy_detected: ipProxyDetected,
    device_trust_score: deviceTrustScore,
    is_digital_goods: isDigitalGoods,
    email_domain_risk: emailDomainRisk,
  };
}

async function handleAuthorizeSubmit(event) {
  event.preventDefault();
  showOrderFormError(null);

  let payload;
  try {
    payload = validateAndReadOrderForm();
  } catch (validationErr) {
    showOrderFormError(validationErr.message);
    updateStatusBar("Validation failed");
    return;
  }

  const submitBtn = document.getElementById("btn-authorize");
  await withButtonLoading(submitBtn, "Authorizing...", async () => {
    try {
      const data = await apiClient.authorizeOrder(payload);
      if (data.order) {
        state.selectedOrderId = data.order.order_id;
        await syncLedgerState();
        renderForensicInspector(data.order);
      }
    } catch (apiErr) {
      showOrderFormError(`Authorization failed: ${apiErr.message}`);
    }
  });
}

/* ==========================================================================
   5. Forensic Inspector & Step-Up 2FA Hold Resolution
   ========================================================================== */
function selectOrderForInspection(orderId) {
  state.selectedOrderId = orderId;
  const order = state.ledger.find((r) => r.order_id === orderId) || null;
  renderForensicInspector(order);
  renderLedgerTable();
}

function renderForensicInspector(record) {
  const container = document.getElementById("forensic-body");
  if (!container) return;

  if (!record) {
    container.innerHTML = `<div class="empty-state-box">No order selected.</div>`;
    return;
  }

  const itemClean = cleanItemTitle(record.item_title);
  const amountFormatted = `$${Number(record.amount || 0).toFixed(2)} ${record.currency || "USD"}`;
  const buyerFormatted = `${record.payer_email} (${record.payer_country} → ${record.shipping_country})`;
  const scoreFormatted = Number(record.risk_score || 0).toFixed(4);
  const policyFormatted = formatPolicyOutcome(record);
  const apiCall = record.ai_forensics?.recommended_paypal_api_call || "—";

  const terseSignals = buildTerseSignals(record);
  const signalsHtml = terseSignals.map((sig) => `<div>• ${escapeHtml(sig)}</div>`).join("");

  const attributions = record.attributions || [];
  const telemetryHtml = attributions
    .map(
      (attr) => `
        <div class="kv-row">
          <span class="kv-key">${escapeHtml(attr.feature)}</span>
          <span class="kv-val">${escapeHtml(attr.value)}</span>
        </div>
      `
    )
    .join("");

  const isPending2FA =
    record.paypal_status === "PENDING_STEP_UP_2FA" ||
    record.paypal_status === "AUTHORIZED_PENDING_2FA";

  const stepUpHtml = isPending2FA
    ? `
      <div class="stepup-box">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <strong>Step-Up 2FA Hold Verification</strong>
          <span style="font-family: var(--font-mono); font-size: 11.5px;">Expected OTP: 849201</span>
        </div>
        <div style="font-size: 12px;">
          Confirm OTP 849201 for ${escapeHtml(record.payer_email)} (${escapeHtml(
            record.payer_country
          )} → ${escapeHtml(record.shipping_country)}).
        </div>
        <div class="form-actions">
          <input type="text" id="stepup-otp-input" style="flex: 1;" placeholder="Enter OTP (849201) or buyer confirmation..." />
          <button type="button" class="btn btn-primary" id="btn-verify-stepup">Verify 2FA</button>
        </div>
      </div>
    `
    : "";

  container.innerHTML = `
    <div id="inspector-alert-slot"></div>
    <div class="inspector-grid">
      <fieldset class="inspector-fieldset">
        <legend>Order Context</legend>
        <div class="kv-row">
          <span class="kv-key">Order ID</span>
          <span class="kv-val">${escapeHtml(record.order_id)}</span>
        </div>
        <div class="kv-row">
          <span class="kv-key">Authorization ID</span>
          <span class="kv-val">${escapeHtml(record.authorization_id || "—")}</span>
        </div>
        <div class="kv-row">
          <span class="kv-key">Item</span>
          <span class="kv-val">${escapeHtml(itemClean)}</span>
        </div>
        <div class="kv-row">
          <span class="kv-key">Amount</span>
          <span class="kv-val">${escapeHtml(amountFormatted)}</span>
        </div>
        <div class="kv-row">
          <span class="kv-key">Buyer</span>
          <span class="kv-val">${escapeHtml(buyerFormatted)}</span>
        </div>
        <div class="kv-row">
          <span class="kv-key">Risk Score</span>
          <span class="kv-val">${escapeHtml(scoreFormatted)}</span>
        </div>
        <div class="kv-row">
          <span class="kv-key">PayPal Status</span>
          <span>${renderStatusBadge(record.paypal_status)}</span>
        </div>
        <div class="kv-row">
          <span class="kv-key">Policy Gate</span>
          <span class="kv-val">${escapeHtml(policyFormatted)}</span>
        </div>
      </fieldset>

      <fieldset class="inspector-fieldset">
        <legend>Triggered Signals &amp; Telemetry</legend>
        <div class="kv-row">
          <span class="kv-key">REST Endpoint</span>
          <span class="kv-val">${escapeHtml(apiCall)}</span>
        </div>
        <div class="signal-list">${signalsHtml}</div>
        <div>${telemetryHtml}</div>
      </fieldset>
    </div>
    ${stepUpHtml}
  `;

  const verifyBtn = document.getElementById("btn-verify-stepup");
  if (verifyBtn) {
    verifyBtn.addEventListener("click", handleVerifyStepUp);
  }
}

async function handleVerifyStepUp() {
  if (!state.selectedOrderId) return;
  const otpInput = document.getElementById("stepup-otp-input");
  const buyerResponse = (otpInput?.value || "").trim();
  const alertSlot = document.getElementById("inspector-alert-slot");

  if (!buyerResponse) {
    if (alertSlot) {
      alertSlot.innerHTML = `<div class="alert-banner">Enter OTP code (849201) or buyer confirmation response.</div>`;
    }
    return;
  }

  const verifyBtn = document.getElementById("btn-verify-stepup");
  await withButtonLoading(verifyBtn, "Verifying...", async () => {
    try {
      const data = await apiClient.verifyStepUp(state.selectedOrderId, buyerResponse, "849201");
      await syncLedgerState();
      if (data.order) {
        renderForensicInspector(data.order);
      }
    } catch (err) {
      if (alertSlot) {
        alertSlot.innerHTML = `<div class="alert-banner">2FA Verification failed: ${escapeHtml(
          err.message
        )}</div>`;
      }
    }
  });
}

/* ==========================================================================
   6. Transaction Ledger & Contextual Row Actions
   ========================================================================== */
async function syncLedgerState() {
  const data = await apiClient.getTransactions();
  state.ledger = data.transactions || [];
  renderMetrics(data.metrics);
  renderLedgerTable();

  if (state.selectedOrderId) {
    const current = state.ledger.find((r) => r.order_id === state.selectedOrderId);
    if (current) {
      renderForensicInspector(current);
    }
  }
  updateStatusBar();
}

function renderLedgerTable() {
  const ledgerBody = document.getElementById("ledger-body");
  if (!ledgerBody) return;

  if (state.ledger.length === 0) {
    ledgerBody.innerHTML = `<div class="empty-state-box">No transactions recorded yet.</div>`;
    return;
  }

  const query = (document.getElementById("ledger-search")?.value || "").trim().toLowerCase();
  const statusFilter = document.getElementById("ledger-status-filter")?.value || "ALL";

  const filtered = state.ledger.filter((row) => {
    if (!matchesStatusFilter(row.paypal_status, statusFilter)) {
      return false;
    }
    if (!query) return true;
    return (
      String(row.order_id || "").toLowerCase().includes(query) ||
      String(row.item_title || "").toLowerCase().includes(query) ||
      String(row.payer_email || "").toLowerCase().includes(query)
    );
  });

  if (filtered.length === 0) {
    ledgerBody.innerHTML = `<div class="empty-state-box">No orders match the current filter.</div>`;
    return;
  }

  const rowsHtml = filtered
    .map((row) => {
      const timeStr = (row.timestamp || "").split("T")[1]?.replace("Z", "") || row.timestamp || "—";
      const isSelected = row.order_id === state.selectedOrderId;
      const isPending =
        row.paypal_status === "PENDING_STEP_UP_2FA" ||
        row.paypal_status === "AUTHORIZED_PENDING_2FA";
      const itemClean = cleanItemTitle(row.item_title);

      const actionsHtml = isPending
        ? `<div class="row-action-group">
             <button type="button" class="btn btn-primary" data-action="capture" data-order-id="${escapeHtml(
               row.order_id
             )}">Capture</button>
             <button type="button" class="btn btn-stop" data-action="void" data-order-id="${escapeHtml(
               row.order_id
             )}">Void</button>
           </div>`
        : `<span>&mdash;</span>`;

      return `
        <tr class="${isSelected ? "selected" : ""}" data-row-order-id="${escapeHtml(row.order_id)}">
          <td class="col-time">${escapeHtml(timeStr)}</td>
          <td class="col-order" title="${escapeHtml(row.order_id)}">${escapeHtml(row.order_id)}</td>
          <td class="col-item" title="${escapeHtml(itemClean)}">${escapeHtml(itemClean)}</td>
          <td class="col-buyer" title="${escapeHtml(row.payer_email)}">${escapeHtml(row.payer_email)}</td>
          <td class="col-amount">$${Number(row.amount || 0).toFixed(2)}</td>
          <td class="col-score">${Number(row.risk_score || 0).toFixed(4)}</td>
          <td class="col-status">${renderStatusBadge(row.paypal_status)}</td>
          <td class="col-actions">${actionsHtml}</td>
        </tr>
      `;
    })
    .join("");

  ledgerBody.innerHTML = `
    <table class="gtk-table">
      <thead>
        <tr>
          <th class="col-time">Time</th>
          <th class="col-order">Order ID</th>
          <th class="col-item">Item</th>
          <th class="col-buyer">Buyer</th>
          <th class="col-amount">Amount</th>
          <th class="col-score">Risk</th>
          <th class="col-status">Status</th>
          <th class="col-actions">Actions</th>
        </tr>
      </thead>
      <tbody>${rowsHtml}</tbody>
    </table>
  `;
}

async function handleLedgerBodyClick(event) {
  const actionBtn = event.target.closest("button[data-action]");
  if (actionBtn) {
    event.stopPropagation();
    const action = actionBtn.dataset.action;
    const orderId = actionBtn.dataset.orderId;
    if (!orderId) return;

    if (action === "capture") {
      await withButtonLoading(actionBtn, "Capturing...", async () => {
        try {
          const res = await apiClient.captureOrder(orderId, "Captured from ledger action.");
          state.selectedOrderId = orderId;
          await syncLedgerState();
          if (res.order) renderForensicInspector(res.order);
        } catch (err) {
          updateStatusBar(`Capture failed: ${err.message}`);
        }
      });
    } else if (action === "void") {
      await withButtonLoading(actionBtn, "Voiding...", async () => {
        try {
          const res = await apiClient.voidOrder(orderId, "Voided from ledger action.");
          state.selectedOrderId = orderId;
          await syncLedgerState();
          if (res.order) renderForensicInspector(res.order);
        } catch (err) {
          updateStatusBar(`Void failed: ${err.message}`);
        }
      });
    }
    return;
  }

  const rowEl = event.target.closest("tr[data-row-order-id]");
  if (rowEl && rowEl.dataset.rowOrderId) {
    selectOrderForInspection(rowEl.dataset.rowOrderId);
  }
}

/* ==========================================================================
   7. CSV Export
   ========================================================================== */
function exportLedgerCsv() {
  if (!state.ledger.length) return;
  const headers = [
    "tx_id",
    "timestamp",
    "order_id",
    "authorization_id",
    "item_title",
    "amount",
    "currency",
    "payer_email",
    "risk_score",
    "paypal_status",
    "policy_applied",
  ];
  const rows = state.ledger.map((r) =>
    headers.map((h) => `"${String(r[h] ?? "").replace(/"/g, '""')}"`).join(",")
  );
  const csv = [headers.join(","), ...rows].join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "txguard_paypal_ledger.csv";
  link.click();
  URL.revokeObjectURL(url);
  updateStatusBar(`Exported ${state.ledger.length} orders to CSV`);
}

/* ==========================================================================
   8. Agent Studio & Webhooks
   ========================================================================== */
async function handleRunAgent() {
  const inputEl = document.getElementById("agent-query-input");
  const prompt = (inputEl?.value || "").trim();
  if (!prompt) {
    inputEl?.classList.add("invalid");
    updateStatusBar("Enter an instruction for the agent");
    return;
  }
  inputEl?.classList.remove("invalid");

  const logEl = document.getElementById("agent-log");
  if (!logEl) return;

  const emptyBox = logEl.querySelector(".empty-state-box");
  if (emptyBox) emptyBox.remove();

  const userDiv = document.createElement("div");
  userDiv.className = "agent-entry user";
  userDiv.textContent = prompt;
  logEl.appendChild(userDiv);
  if (inputEl) inputEl.value = "";

  const runBtn = document.getElementById("btn-run-agent");
  await withButtonLoading(runBtn, "Running...", async () => {
    try {
      const data = await apiClient.runAgent(prompt);
      const replyText =
        data.agent_response?.reply || data.reply || "Completed agent instruction.";
      const assistantDiv = document.createElement("div");
      assistantDiv.className = "agent-entry assistant";
      assistantDiv.textContent = replyText;
      logEl.appendChild(assistantDiv);
      logEl.scrollTop = logEl.scrollHeight;
      await syncLedgerState();
    } catch (err) {
      const errDiv = document.createElement("div");
      errDiv.className = "alert-banner";
      errDiv.textContent = `Agent Error: ${err.message}`;
      logEl.appendChild(errDiv);
    }
  });
}

function clearAgentLog() {
  const logEl = document.getElementById("agent-log");
  if (!logEl) return;
  logEl.innerHTML = `
    <div class="empty-state-box">
      Enter an instruction to query catalog items, run a PayPal checkout, or audit pending orders.
    </div>
  `;
  updateStatusBar("Agent log cleared");
}

async function handleSendWebhook() {
  const eventSelect = document.getElementById("webhook-event-select");
  const orderInput = document.getElementById("webhook-order-id");
  const logArea = document.getElementById("webhook-log");

  const eventType = eventSelect?.value || "PAYMENT.AUTHORIZATION.CREATED";
  const orderId = (orderInput?.value || "").trim();

  if (!orderId) {
    orderInput?.classList.add("invalid");
    updateStatusBar("Enter a Resource Order ID to dispatch webhook");
    return;
  }
  orderInput?.classList.remove("invalid");

  const sendBtn = document.getElementById("btn-send-webhook");
  await withButtonLoading(sendBtn, "Sending...", async () => {
    try {
      const res = await apiClient.sendWebhook(eventType, orderId);
      if (logArea) {
        logArea.textContent = JSON.stringify(res, null, 2);
      }
    } catch (err) {
      if (logArea) {
        logArea.innerHTML = `<div class="alert-banner">Webhook Error: ${escapeHtml(
          err.message
        )}</div>`;
      }
    }
  });
}

/* ==========================================================================
   9. DOM Initialization & Event Binding
   ========================================================================== */
document.addEventListener("DOMContentLoaded", async () => {
  // Tabs
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });

  // Top Header Export CSV
  document.getElementById("btn-export")?.addEventListener("click", exportLedgerCsv);

  // Order Authorization Form
  document
    .getElementById("catalog-select")
    ?.addEventListener("change", handleCatalogSelectChange);
  document.getElementById("order-form")?.addEventListener("submit", handleAuthorizeSubmit);
  document.getElementById("btn-clear-order")?.addEventListener("click", clearOrderForm);

  // Transaction Ledger Filters & Row Actions
  document.getElementById("ledger-search")?.addEventListener("input", renderLedgerTable);
  document
    .getElementById("ledger-status-filter")
    ?.addEventListener("change", renderLedgerTable);
  document.getElementById("ledger-body")?.addEventListener("click", handleLedgerBodyClick);

  // Agent Studio
  document.getElementById("btn-run-agent")?.addEventListener("click", handleRunAgent);
  document.getElementById("agent-query-input")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleRunAgent();
    }
  });
  document.getElementById("btn-clear-agent-log")?.addEventListener("click", clearAgentLog);

  // Webhooks
  document.getElementById("btn-send-webhook")?.addEventListener("click", handleSendWebhook);

  // Boot in zero-state
  await initCatalogSelect();
  await syncLedgerState();
  updateStatusBar("Ready");
});
