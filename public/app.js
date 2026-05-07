const singleForm = document.querySelector("#singleForm");
const singleResult = document.querySelector("#singleResult");
const classifyBtn = document.querySelector("#classifyBtn");
const batchBtn = document.querySelector("#batchBtn");
const batchResult = document.querySelector("#batchResult");
const healthStatus = document.querySelector("#healthStatus");
const inquiryText = document.querySelector("#inquiryText");

const labels = {
  billing: "Billing",
  technical_support: "Technical Support",
  product_inquiry: "Product Inquiry",
  shipping: "Shipping",
  refund_return: "Refund / Return",
  account_management: "Account Management",
  general_inquiry: "General Inquiry",
};

function pct(value) {
  return `${Math.round((value || 0) * 1000) / 10}%`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.detail || `Request failed with ${response.status}`);
  }

  return response.json();
}

function renderError(target, message) {
  target.innerHTML = `<div class="error">${escapeHtml(message)}</div>`;
}

function renderPrediction(result) {
  const isReview = result.requires_human_review || result.routing_decision === "human_review";
  const probabilities = Object.entries(result.all_probabilities || {})
    .sort((a, b) => b[1] - a[1])
    .map(([key, value]) => `
      <div class="bar">
        <span>${escapeHtml(labels[key] || key)}</span>
        <div class="track"><div class="fill" style="width:${Math.max(2, value * 100)}%"></div></div>
        <strong>${pct(value)}</strong>
      </div>
    `)
    .join("");

  const keywords = (result.top_keywords || [])
    .map((word) => `<span class="chip">${escapeHtml(word)}</span>`)
    .join("");

  singleResult.innerHTML = `
    <div class="result-card">
      <div class="result-title">
        <div class="label">${escapeHtml(result.label || labels[result.final_category] || "Unknown")}</div>
        <span class="pill ${isReview ? "warn" : ""}">${escapeHtml(result.routing_decision)}</span>
      </div>
      <div class="meta">
        <div class="metric"><span>Confidence</span><strong>${pct(result.confidence)}</strong></div>
        <div class="metric"><span>Latency</span><strong>${escapeHtml(result.latency_ms)} ms</strong></div>
        <div class="metric"><span>Routed Team</span><strong>${escapeHtml(result.routed_team)}</strong></div>
        <div class="metric"><span>Review Needed</span><strong>${result.requires_human_review ? "Yes" : "No"}</strong></div>
      </div>
      <div class="bars">${probabilities}</div>
      ${keywords ? `<div class="chips">${keywords}</div>` : ""}
    </div>
  `;
}

document.querySelectorAll(".sample").forEach((button) => {
  button.addEventListener("click", () => {
    inquiryText.value = button.dataset.sample || "";
    inquiryText.focus();
  });
});

singleForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = inquiryText.value.trim();
  if (text.length < 3) {
    renderError(singleResult, "Enter at least 3 characters before classifying.");
    return;
  }

  classifyBtn.disabled = true;
  singleResult.innerHTML = `<div class="loading">Classifying inquiry...</div>`;
  try {
    renderPrediction(await postJson("/api/predict", { text }));
  } catch (error) {
    renderError(singleResult, error.message);
  } finally {
    classifyBtn.disabled = false;
  }
});

batchBtn.addEventListener("click", async () => {
  const texts = document
    .querySelector("#batchText")
    .value.split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  if (!texts.length) {
    renderError(batchResult, "Add at least one inquiry.");
    return;
  }
  if (texts.length > 50) {
    renderError(batchResult, "Batch classification supports up to 50 inquiries.");
    return;
  }

  batchBtn.disabled = true;
  batchResult.innerHTML = `<div class="loading">Classifying ${texts.length} inquiries...</div>`;
  try {
    const data = await postJson("/api/predict/batch", { texts });
    const rows = (data.results || [])
      .map((row) => `
        <tr>
          <td>${escapeHtml(row.text)}</td>
          <td>${escapeHtml(labels[row.final_category] || row.label || row.final_category)}</td>
          <td>${pct(row.confidence)}</td>
          <td>${escapeHtml(row.routing_decision)}</td>
          <td>${escapeHtml(row.routed_team)}</td>
        </tr>
      `)
      .join("");

    batchResult.innerHTML = `
      <table>
        <thead>
          <tr><th>Inquiry</th><th>Category</th><th>Confidence</th><th>Decision</th><th>Team</th></tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  } catch (error) {
    renderError(batchResult, error.message);
  } finally {
    batchBtn.disabled = false;
  }
});

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error("API unavailable");
    const data = await response.json();
    healthStatus.textContent = data.model_loaded ? "API ready" : "Model warming";
    healthStatus.className = "status ok";
  } catch {
    healthStatus.textContent = "API offline";
    healthStatus.className = "status error";
  }
}

checkHealth();
