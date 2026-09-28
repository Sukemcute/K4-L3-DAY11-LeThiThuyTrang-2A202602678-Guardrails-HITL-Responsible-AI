/**
 * VinBank Guardrails Playground & Trace Inspector Client JS
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const chatForm = document.getElementById("chatForm");
  const promptInput = document.getElementById("promptInput");
  const btnSend = document.getElementById("btnSend");
  const chatViewport = document.getElementById("chatViewport");
  const traceContainer = document.getElementById("traceContainer");
  const traceTotalLatency = document.getElementById("traceTotalLatency");
  const userSelect = document.getElementById("userSelect");
  const btnFireSpam = document.getElementById("btnFireSpam");
  const logCountBadge = document.getElementById("logCountBadge");

  // Inspector Elements
  const stepInspector = document.getElementById("stepInspector");
  const inspectorStepName = document.getElementById("inspectorStepName");
  const inspectorBody = document.getElementById("inspectorBody");
  const btnCloseInspector = document.getElementById("btnCloseInspector");

  // Layer Toggles
  const toggleRateLimit = document.getElementById("toggleRateLimit");
  const toggleInputGuard = document.getElementById("toggleInputGuard");
  const toggleOutputGuard = document.getElementById("toggleOutputGuard");
  const toggleEgress = document.getElementById("toggleEgress");

  // Radio agent pills
  const agentRadios = document.querySelectorAll('input[name="agent_type"]');
  const agentPillBlue = document.getElementById("lbl-blue");
  const agentPillRed = document.getElementById("lbl-red");
  const agentPillAdvance = document.getElementById("lbl-advance");

  let currentAgent = "blue";
  let lastTraceSpans = [];

  // =========================================================================
  // Agent Type Switcher
  // =========================================================================
  agentRadios.forEach(radio => {
    radio.addEventListener("change", (e) => {
      currentAgent = e.target.value;
      [agentPillBlue, agentPillRed, agentPillAdvance].forEach(p => p.classList.remove("active"));
      if (currentAgent === "blue") agentPillBlue.classList.add("active");
      if (currentAgent === "red") agentPillRed.classList.add("active");
      if (currentAgent === "red_advance") agentPillAdvance.classList.add("active");
    });
  });

  // =========================================================================
  // Tab Navigation
  // =========================================================================
  const navTabs = document.querySelectorAll(".nav-tab");
  const tabPanes = document.querySelectorAll(".tab-pane");

  navTabs.forEach(tab => {
    tab.addEventListener("click", () => {
      const target = tab.getAttribute("data-tab");
      navTabs.forEach(t => t.classList.remove("active"));
      tabPanes.forEach(p => p.classList.remove("active"));

      tab.classList.add("active");
      document.getElementById(`pane-${target}`).classList.add("active");

      if (target === "forensics") loadLogs();
      if (target === "metrics") loadMetrics();
    });
  });

  // =========================================================================
  // Presets & Spam Attack
  // =========================================================================
  document.querySelectorAll(".preset-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      promptInput.value = btn.getAttribute("data-prompt");
      promptInput.focus();
    });
  });

  btnFireSpam.addEventListener("click", async () => {
    btnFireSpam.disabled = true;
    btnFireSpam.textContent = "🚀 Đang bắn 12 requests...";
    const spamPrompt = "Kiểm tra số dư tài khoản của tôi ngay lập tức.";
    for (let i = 1; i <= 12; i++) {
      await sendPrompt(`[Spam #${i}] ${spamPrompt}`, true);
      await new Promise(r => setTimeout(r, 80));
    }
    btnFireSpam.disabled = false;
    btnFireSpam.textContent = "💥 Spam Attack (12x Requests)";
    loadMetrics();
  });

  // Ctrl+Enter to submit
  promptInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      chatForm.dispatchEvent(new Event("submit"));
    }
  });

  // =========================================================================
  // Chat Submit
  // =========================================================================
  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = promptInput.value.trim();
    if (!text) return;
    promptInput.value = "";
    await sendPrompt(text);
  });

  async function sendPrompt(text, isSpamBatch = false) {
    appendUserBubble(text);
    btnSend.disabled = true;

    // Show loading in trace
    if (!isSpamBatch) {
      traceTotalLatency.textContent = "Tracer processing...";
      traceContainer.innerHTML = `
        <div class="trace-empty">
          <div class="status-dot green" style="width:12px;height:12px;"></div>
          <p>Đang phân tích qua từng tầng Guardrail & Model...</p>
        </div>
      `;
    }

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          user_id: userSelect.value,
          agent_type: currentAgent,
          enable_rate_limiter: toggleRateLimit.checked,
          enable_input_guard: toggleInputGuard.checked,
          enable_output_guard: toggleOutputGuard.checked,
          enable_egress_check: toggleEgress.checked,
        }),
      });

      const data = await res.json();
      appendBotBubble(data.reply, data.status, data.blocked_layer);

      if (!isSpamBatch || text.includes("#12")) {
        renderTraceWaterfall(data);
      }
      updateLogCount();
    } catch (err) {
      appendBotBubble(`[Network Error] Không thể kết nối tới server: ${err.message}`, "BLOCKED", "network");
    } finally {
      btnSend.disabled = false;
    }
  }

  // =========================================================================
  // Chat Bubble Renderers
  // =========================================================================
  function appendUserBubble(text) {
    // Remove welcome screen if present
    const welcome = chatViewport.querySelector(".system-welcome");
    if (welcome) welcome.remove();

    const bubble = document.createElement("div");
    bubble.className = "chat-bubble user";
    bubble.innerHTML = `
      <div class="bubble-meta"><span>Bạn (${userSelect.options[userSelect.selectedIndex].text})</span> • <span>${new Date().toLocaleTimeString()}</span></div>
      <div class="bubble-content">${escapeHtml(text)}</div>
    `;
    chatViewport.appendChild(bubble);
    chatViewport.scrollTop = chatViewport.scrollHeight;
  }

  function appendBotBubble(reply, status, layer) {
    const bubble = document.createElement("div");
    let extraClass = "";
    let badgeText = "AI Response";

    if (status === "BLOCKED") {
      extraClass = "blocked";
      badgeText = `🛑 Blocked at: ${layer || "guardrail"}`;
    } else if (status === "REDACTED") {
      extraClass = "redacted";
      badgeText = "🛡️ Redacted (PII/Secret Sanitized)";
    } else if (status === "LEAKED") {
      extraClass = "leaked";
      badgeText = "🚨 SECRET LEAKED (Unsafe Agent)";
    }

    bubble.className = `chat-bubble bot ${extraClass}`;
    bubble.innerHTML = `
      <div class="bubble-meta">
        <span style="font-weight:700;">VinBank (${currentAgent.toUpperCase()})</span>
        <span class="badge-status ${status.toLowerCase()}" style="font-size:0.65rem;">${badgeText}</span>
      </div>
      <div class="bubble-content">${escapeHtml(reply)}</div>
    `;
    chatViewport.appendChild(bubble);
    chatViewport.scrollTop = chatViewport.scrollHeight;
  }

  // =========================================================================
  // LangSmith-Style Trace Waterfall Renderer
  // =========================================================================
  function renderTraceWaterfall(data) {
    traceTotalLatency.textContent = `Total Latency: ${data.total_latency_ms} ms (${data.status})`;
    traceContainer.innerHTML = "";
    lastTraceSpans = data.spans || [];

    data.spans.forEach((span, idx) => {
      const node = document.createElement("div");
      node.className = "span-node";
      node.setAttribute("data-span-id", span.id);

      const statusClass = span.status.toLowerCase();

      node.innerHTML = `
        <div class="span-top-row">
          <div class="span-title">
            <span>${escapeHtml(span.name)}</span>
          </div>
          <div class="span-badges">
            <span class="badge-status ${statusClass}">${span.status}</span>
            <span class="badge-latency">${span.latency_ms} ms</span>
          </div>
        </div>
        <div class="span-summary">${getSpanSummaryText(span)}</div>
      `;

      node.addEventListener("click", () => {
        document.querySelectorAll(".span-node").forEach(n => n.classList.remove("active-inspector"));
        node.classList.add("active-inspector");
        openInspector(span);
      });

      traceContainer.appendChild(node);
    });

    // Auto-open inspector for the first blocked/redacted span, or the last span
    const focusSpan = data.spans.find(s => s.status === "BLOCKED" || s.status === "REDACTED") || data.spans[data.spans.length - 1];
    if (focusSpan) {
      openInspector(focusSpan);
      const activeEl = traceContainer.querySelector(`[data-span-id="${focusSpan.id}"]`);
      if (activeEl) activeEl.classList.add("active-inspector");
    }
  }

  function getSpanSummaryText(span) {
    const d = span.details || {};
    if (span.layer === "rate_limiter") {
      return span.status === "BLOCKED" 
        ? `🛑 Limit reached (${d.max_requests} req / ${d.window_seconds}s). Cooldown: ${d.cooldown_remaining_sec}s`
        : `✅ Quota used: ${d.requests_in_window || 0}/${d.max_requests || 10} requests`;
    }
    if (span.layer === "input_injection") {
      return span.status === "BLOCKED"
        ? `🛑 Threat detected: ${d.threat_type || "Prompt Injection"}`
        : `✅ Clean: No prompt injection patterns detected`;
    }
    if (span.layer === "input_topic") {
      return span.status === "BLOCKED"
        ? `🛑 Off-topic or forbidden topic matched`
        : d.intent === "CONVERSATIONAL"
          ? `✅ Conversational intent accepted`
          : `✅ Banking domain validated`;
    }
    if (span.layer === "llm_inference") {
      return d.raw_preview ? `Generated: "${d.raw_preview}"` : "LLM called successfully";
    }
    if (span.layer === "output_guardrail") {
      return span.status === "REDACTED"
        ? `🛡️ Redacted ${d.redactions_count || 1} sensitive items: ${d.issues_detected?.join(", ") || ""}`
        : `✅ Output safe from secrets / PII`;
    }
    if (span.layer === "egress_boundary") {
      return `Destination verified: ${d.destination_checked || "VinBank API"}`;
    }
    return JSON.stringify(d);
  }

  // =========================================================================
  // Step Inspector Drawer
  // =========================================================================
  function openInspector(span) {
    stepInspector.style.display = "flex";
    inspectorStepName.textContent = `Inspector: ${span.name}`;
    
    let rows = `
      <tr><td class="label">Layer ID</td><td class="val">${span.layer}</td></tr>
      <tr><td class="label">Status</td><td class="val"><span class="badge-status ${span.status.toLowerCase()}">${span.status}</span></td></tr>
      <tr><td class="label">Latency</td><td class="val">${span.latency_ms} ms</td></tr>
    `;

    for (const [k, v] of Object.entries(span.details || {})) {
      let valStr = typeof v === "object" ? JSON.stringify(v, null, 2) : String(v);
      rows += `<tr><td class="label">${escapeHtml(k)}</td><td class="val">${escapeHtml(valStr)}</td></tr>`;
    }

    inspectorBody.innerHTML = `<table class="detail-table">${rows}</table>`;
  }

  btnCloseInspector.addEventListener("click", () => {
    stepInspector.style.display = "none";
    document.querySelectorAll(".span-node").forEach(n => n.classList.remove("active-inspector"));
  });

  // =========================================================================
  // Forensics Logs Loader
  // =========================================================================
  async function loadLogs() {
    try {
      const res = await fetch("/api/logs?limit=50");
      const data = await res.json();
      const tbody = document.getElementById("logTableBody");
      tbody.innerHTML = "";

      if (!data.logs || data.logs.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" class="text-center muted">Chưa có bản ghi nào.</td></tr>';
        return;
      }

      data.logs.slice().reverse().forEach(log => {
        const tr = document.createElement("tr");
        const statusBadge = log.blocked 
          ? '<span class="badge-status blocked">BLOCKED</span>' 
          : '<span class="badge-status passed">PASSED</span>';

        tr.innerHTML = `
          <td>${new Date(log.timestamp).toLocaleTimeString()}</td>
          <td style="font-family:var(--font-mono);">${log.request_id || "-"}</td>
          <td><b>${escapeHtml(log.user_id || "anon")}</b></td>
          <td title="${escapeHtml(log.input)}">${escapeHtml(truncate(log.input, 38))}</td>
          <td><code>${log.layer || "passed"}</code></td>
          <td>${statusBadge}</td>
          <td>${log.latency_sec ? Math.round(log.latency_sec * 1000) : 0} ms</td>
          <td title="${escapeHtml(log.response)}">${escapeHtml(truncate(log.response, 32))}</td>
        `;
        tbody.appendChild(tr);
      });
    } catch (e) {
      console.error("Error loading logs:", e);
    }
  }

  async function updateLogCount() {
    try {
      const res = await fetch("/api/logs?limit=1");
      const data = await res.json();
      logCountBadge.textContent = data.total || 0;
    } catch (_) {}
  }

  document.getElementById("btnClearLogs").addEventListener("click", async () => {
    if (confirm("Bạn có chắc chắn muốn xóa toàn bộ log trong phiên này?")) {
      await fetch("/api/clear", { method: "POST" });
      loadLogs();
      loadMetrics();
    }
  });

  // =========================================================================
  // Security Metrics Loader
  // =========================================================================
  async function loadMetrics() {
    try {
      const res = await fetch("/api/metrics");
      const data = await res.json();
      const snap = data.snapshot || {};

      document.getElementById("metricTotal").textContent = snap.total_requests || 0;
      document.getElementById("metricBlocked").textContent = snap.blocked_requests || 0;
      document.getElementById("metricBlockRate").textContent = `${Math.round((snap.block_rate || 0) * 100)}%`;
      document.getElementById("metricRateHits").textContent = data.rate_limiter_blocked || 0;

      const alertsContainer = document.getElementById("alertsContainer");
      if (data.active_alerts && data.active_alerts.length > 0) {
        alertsContainer.innerHTML = data.active_alerts.map(a => `
          <div style="background:rgba(244,63,94,0.15); border:1px solid rgba(244,63,94,0.4); padding:10px 14px; border-radius:8px; margin-bottom:8px; color:#fecdd3;">
            ⚠️ <b>${a.metric.toUpperCase()} THRESHOLD EXCEEDED</b>: ${escapeHtml(a.message)} (Giá trị: ${a.value} > Ngưỡng: ${a.threshold})
          </div>
        `).join("");
      } else {
        alertsContainer.innerHTML = '<div class="alert-empty">✅ Tất cả chỉ số đang nằm trong ngưỡng an toàn cho phép.</div>';
      }
    } catch (e) {
      console.error("Error loading metrics:", e);
    }
  }

  // =========================================================================
  // Utilities
  // =========================================================================
  function escapeHtml(str) {
    if (!str) return "";
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  function truncate(str, len) {
    if (!str) return "";
    return str.length > len ? str.substring(0, len) + "..." : str;
  }

  // Initial loads
  updateLogCount();
  fetch("/api/status").then(r => r.json()).then(st => {
    document.getElementById("modelLabel").textContent = `Blue: ${st.blue_model}`;
  }).catch(() => {});
});
