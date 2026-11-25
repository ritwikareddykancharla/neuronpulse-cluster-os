/* NeuronPulse Frontend Controller — Stateless browser memory + location.hash */
(() => {
  const state = {
    cluster: null,
    profiler: null,
    selectedRank: 25,
    activeTab: window.location.hash === "#profiler" ? "profiler" : "topology",
    profParams: {
      tp_degree: 8,
      precision: "FP8",
      batch_size: 16,
      nki_flash_fusion: true,
      seq_len: 4096,
    },
  };

  async function api(path, method = "GET", body = null) {
    const opts = { method, headers: { "Content-Type": "application/json" } };
    if (body !== null) opts.body = JSON.stringify(body);
    const res = await fetch(path, opts);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  function escapeHtml(str) {
    return String(str || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function formatMarkdown(md) {
    let html = escapeHtml(md);
    html = html.replace(/```(?:bash|sh|python)?\n([\s\S]*?)```/g, "<pre><code>$1</code></pre>");
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/^### (.*)$/gm, "<h4 style='margin:8px 0 4px;color:#5eead4;'>$1</h4>");
    html = html.replace(/^- (.*)$/gm, "<div style='margin:3px 0;'>&bull; $1</div>");
    html = html.replace(/\n\n/g, "<br/>");
    return html;
  }

  function switchTab(tab) {
    state.activeTab = tab;
    window.location.hash = tab === "profiler" ? "#profiler" : "#topology";
    document.getElementById("tab-topology").classList.toggle("active", tab === "topology");
    document.getElementById("tab-profiler").classList.toggle("active", tab === "profiler");
    document.getElementById("panel-topology").classList.toggle("hidden", tab !== "topology");
    document.getElementById("panel-profiler").classList.toggle("hidden", tab !== "profiler");
  }

  function renderCluster() {
    const c = state.cluster;
    if (!c) return;
    const sum = c.summary;

    const pill = document.getElementById("cluster-status-pill");
    if (c.active_fault_id) {
      pill.textContent = `IMPAIRED · ${c.active_fault.hardware_code}`;
      pill.classList.add("impaired");
    } else if (c.cordoned_nodes.length > 0) {
      pill.textContent = `64/64 HEALTHY · WARM SPARE ACTIVE (${c.warm_spare_node})`;
      pill.classList.remove("impaired");
    } else {
      pill.textContent = "64/64 CORES HEALTHY";
      pill.classList.remove("impaired");
    }

    document.getElementById("kpi-cores").textContent = `${sum.healthy_cores} / ${sum.total_neuron_cores}`;
    document.getElementById("kpi-nodes-sub").textContent =
      c.cordoned_nodes.length > 0 ? `Cordoned: ${c.cordoned_nodes.join(", ")}` : "4x trn2.48xlarge Active";
    document.getElementById("kpi-hbm-bw").textContent = `${sum.avg_hbm3e_bw_gbps.toLocaleString()} GB/s`;
    document.getElementById("kpi-ring-p99").textContent = `${sum.collective_ring_p99_ms} ms`;
    document.getElementById("kpi-flops").textContent = `${sum.avg_flops_util_pct}%`;
    document.getElementById("kpi-temp-sub").textContent = `Peak Die: ${sum.peak_temp_c}\u00B0C · KV Frag: ${sum.avg_kv_frag_pct}%`;
    document.getElementById("kpi-spare").textContent = c.warm_spare_node;
    document.getElementById("kpi-spare-sub").textContent = c.spare_state;

    // Render EFA Ring
    const ringEl = document.getElementById("efa-ring-list");
    ringEl.innerHTML = c.efa_links
      .map(
        (lk) => `
      <div class="efa-item ${lk.status !== "HEALTHY" ? "degraded" : ""}">
        <div class="efa-top">
          <span>${escapeHtml(lk.ring_hop)}</span>
          <span style="color:${lk.status === "HEALTHY" ? "#34d399" : "#fca5a5"}">${lk.p99_allreduce_ms} ms p99</span>
        </div>
        <div class="efa-meta">
          BW: ${lk.bandwidth_gbps} Gbps &middot; SRD Retrans: ${lk.srd_retransmit_pct}% &middot; Rx: ${lk.optical_rx_dbm} dBm
        </div>
      </div>`
      )
      .join("");

    // Render Cross-Node Error Correlator Banner
    const corrEl = document.getElementById("correlator-banner");
    const corr = c.correlation;
    if (c.active_fault_id && corr && corr.root_cause) {
      const rc = corr.root_cause;
      corrEl.className = "correlator-banner fault";
      corrEl.innerHTML = `
        <div class="corr-head">
          <div>
            <div class="card-kicker" style="color:#fca5a5;">CROSS-NODE CAUSAL ERROR CORRELATOR &middot; 100% ROOT-RANK ISOLATION</div>
            <div class="corr-title">${escapeHtml(c.active_fault.title)}</div>
          </div>
          <button type="button" class="btn-repair-cta" id="inline-repair-btn">
            &#9889; Execute Automated UltraServer Repair (${corr.requires_node_replacement ? "74s Spare Swap vs 42m Reboot" : "4.4s Online KV Compaction"})
          </button>
        </div>
        <div class="corr-metrics">
          <span class="corr-chip"><strong>Root Rank Isolated:</strong> <code>${rc.node_id}:${rc.core_id}</code> (Rank ${rc.global_rank})</span>
          <span class="corr-chip"><strong>Alert Noise Compression:</strong> ${corr.raw_alert_count} raw rank alarms &rarr; <strong>1 causal root fault</strong> (${corr.suppressed_symptom_count} ring stalls suppressed)</span>
          <span class="corr-chip"><strong>Cascade Lead Window:</strong> +${corr.cascade_propagation_window_ms} ms before peer timeouts</span>
        </div>
        <div class="corr-cli">$ ${escapeHtml(corr.recommended_cli)}</div>
      `;
      document.getElementById("inline-repair-btn").addEventListener("click", triggerAutomatedRepair);
    } else {
      corrEl.className = "correlator-banner healthy";
      corrEl.innerHTML = `
        <div class="corr-head">
          <div>
            <div class="card-kicker">CROSS-NODE CAUSAL ERROR CORRELATOR</div>
            <div class="corr-title">All 64 NeuronCores Synchronized &middot; Zero Uncorrectable ECC or EFA SRD Anomalies</div>
          </div>
          <span class="mini-tag">Active Nodes: ${c.active_nodes.join(" · ")}</span>
        </div>
      `;
    }

    // Render Repair Timeline Box if last_repair present and no active fault
    const repBox = document.getElementById("repair-timeline-box");
    if (c.last_repair && !c.active_fault_id) {
      const rep = c.last_repair;
      repBox.classList.remove("hidden");
      repBox.innerHTML = `
        <div class="corr-head">
          <div>
            <div class="card-kicker">AUTOMATED ULTRASERVER RECOVERY VERIFIED</div>
            <div class="corr-title">
              Recovered in <span style="color:#34d399;">${rep.automated_recovery_sec} sec</span> vs ${rep.manual_reboot_label} manual baseline
              (<strong>${rep.mttr_speedup_x}&times; faster MTTR</strong>) &middot; Replacement: <code>${rep.replacement_node}</code>
            </div>
          </div>
        </div>
        <div class="repair-steps">
          ${rep.steps
            .map(
              (s) => `
            <div class="repair-step-card">
              <div class="repair-step-top">
                <span>${s.step}. ${s.phase}</span>
                <span>${s.duration_sec}s</span>
              </div>
              <div style="color:#cbd5e1;">${escapeHtml(s.detail)}</div>
            </div>`
            )
            .join("")}
        </div>
      `;
    } else {
      repBox.classList.add("hidden");
    }

    // Render 4 Nodes x 16 NeuronCores Grid
    const nodesGrid = document.getElementById("nodes-grid");
    const grouped = {};
    c.active_nodes.forEach((n) => (grouped[n] = []));
    c.cores.forEach((core) => {
      if (!grouped[core.node_id]) grouped[core.node_id] = [];
      grouped[core.node_id].push(core);
    });

    nodesGrid.innerHTML = c.active_nodes
      .map((nodeId) => {
        const cores = grouped[nodeId] || [];
        const isSpare = nodeId === c.warm_spare_node;
        return `
        <div class="node-box ${isSpare ? "spare-node" : ""}">
          <div class="node-head">
            <span class="node-name">${nodeId} ${isSpare ? "&#9733; PROMOTED WARM SPARE" : ""}</span>
            <span class="mini-tag">16 NeuronCores &middot; 768 GB HBM3e</span>
          </div>
          <div class="cores-16-grid">
            ${cores
              .map(
                (nc) => `
              <button type="button" class="nc-tile ${nc.status} ${nc.global_rank === state.selectedRank ? "selected" : ""}" data-rank="${nc.global_rank}">
                <div class="nc-id">
                  <span>${nc.core_id}</span>
                  <span>R${nc.global_rank}</span>
                </div>
                <div class="nc-metric">${nc.hbm3e_bw_gbps} GB/s &middot; ${nc.temp_c}&deg;</div>
              </button>`
              )
              .join("")}
          </div>
        </div>`;
      })
      .join("");

    nodesGrid.querySelectorAll(".nc-tile").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.selectedRank = Number(btn.dataset.rank);
        renderCluster();
      });
    });

    renderCoreInspector();
  }

  function renderCoreInspector() {
    const c = state.cluster;
    if (!c) return;
    const core = c.cores.find((x) => x.global_rank === state.selectedRank) || c.cores[0];
    const el = document.getElementById("core-inspector");
    el.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
        <strong>Selected Hardware Slice: <code>${core.node_id}:${core.core_id}</code> (Global Rank ${core.global_rank} &middot; EFA ${core.efa_nic})</strong>
        <span class="mini-tag">Status: ${core.status} &middot; Code: ${core.error_code}</span>
      </div>
      <div class="insp-grid">
        <div class="insp-cell"><span>HBM3e Bandwidth</span><strong>${core.hbm3e_bw_gbps} GB/s</strong></div>
        <div class="insp-cell"><span>HBM3e Used / Cap</span><strong>${core.hbm_used_gb} / ${core.hbm_total_gb} GB</strong></div>
        <div class="insp-cell"><span>TensorEngine MFU</span><strong>${core.flops_util_pct}%</strong></div>
        <div class="insp-cell"><span>All-Reduce Ring p99</span><strong>${core.collective_p99_ms} ms</strong></div>
        <div class="insp-cell"><span>KV External Frag</span><strong>${core.kv_frag_pct}%</strong></div>
        <div class="insp-cell"><span>ECC (SBE / DBE)</span><strong>${core.ecc_sbe} / ${core.ecc_dbe}</strong></div>
      </div>
    `;
  }

  function renderProfiler() {
    const p = state.profiler;
    if (!p) return;
    const sel = p.selected;
    const delta = p.vs_bf16_unfused_baseline;

    document.getElementById("profiler-kpis").innerHTML = `
      <div class="kpi-card">
        <span class="kpi-label">Prefill TTFT</span>
        <strong class="kpi-value">${sel.fits_in_hbm ? sel.ttft_ms + " ms" : "HBM OOM"}</strong>
        <span class="kpi-sub">${delta.ttft_reduction_pct >= 0 ? "-" + delta.ttft_reduction_pct + "% vs BF16 Unfused" : "Unfused / Higher Latency"}</span>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">64-Core Throughput</span>
        <strong class="kpi-value">${sel.fits_in_hbm ? sel.ultraserver_throughput_tok_s.toLocaleString() + " tok/s" : "0 tok/s (OOM)"}</strong>
        <span class="kpi-sub">${sel.replicas_on_64_cores}x TP=${sel.tp_degree} Replicas (${delta.throughput_speedup_x}&times; speedup)</span>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">Decode TPOT</span>
        <strong class="kpi-value">${sel.tpot_ms} ms/tok</strong>
        <span class="kpi-sub">Replica: ${sel.replica_throughput_tok_s} tok/s</span>
      </div>
      <div class="kpi-card">
        <span class="kpi-label">Per-Core HBM3e Footprint</span>
        <strong class="kpi-value" style="color:${sel.fits_in_hbm ? "#6ee7b7" : "#fca5a5"}">${sel.hbm_used_per_core_gb} / ${sel.hbm_capacity_per_core_gb} GB</strong>
        <span class="kpi-sub">${sel.fits_in_hbm ? sel.hbm_utilization_pct + "% HBM3e Util" : "EXCEEDS 48GB CORE SLICE"}</span>
      </div>
    `;

    const bars = p.comparison_bars;
    const maxTtft = Math.max(...bars.map((b) => b.ttft_ms), 1);
    const maxTput = Math.max(...bars.map((b) => b.ultraserver_throughput_tok_s), 1);

    document.getElementById("ttft-bars").innerHTML = bars
      .map((b, idx) => {
        const pct = Math.min(100, Math.round((b.ttft_ms / maxTtft) * 100));
        const isAct = idx === bars.length - 1;
        return `
        <div class="bar-row">
          <div class="bar-label-line">
            <span>${escapeHtml(b.label)}</span>
            <strong>${b.fits_in_hbm ? b.ttft_ms + " ms" : "OOM (>48 GB)"}</strong>
          </div>
          <div class="bar-track">
            <div class="bar-fill ttft ${isAct ? "active-highlight" : ""}" style="width:${pct}%"></div>
          </div>
        </div>`;
      })
      .join("");

    document.getElementById("tput-bars").innerHTML = bars
      .map((b, idx) => {
        const pct = Math.min(100, Math.round((b.ultraserver_throughput_tok_s / maxTput) * 100));
        const isAct = idx === bars.length - 1;
        return `
        <div class="bar-row">
          <div class="bar-label-line">
            <span>${escapeHtml(b.label)}</span>
            <strong>${b.ultraserver_throughput_tok_s.toLocaleString()} tok/s</strong>
          </div>
          <div class="bar-track">
            <div class="bar-fill tput ${isAct ? "active-highlight" : ""}" style="width:${pct}%"></div>
          </div>
        </div>`;
      })
      .join("");

    const mb = sel.memory_breakdown_gb;
    document.getElementById("profiler-footer-card").innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:8px;">
        <strong>Memory Breakdown per NeuronCore: Weights <code>${mb.weights_per_core_gb} GB</code> + Paged KV <code>${mb.kv_cache_per_core_gb} GB</code> + NKI Scratch <code>${mb.activations_scratch_gb} GB</code></strong>
        <span class="mini-tag">Primary Limiter: ${sel.bottleneck}</span>
      </div>
      <div class="corr-cli">export NEURON_CC_FLAGS="${escapeHtml(sel.neuron_cc_flags)}"</div>
    `;
  }

  async function injectFault(faultId, focusRank) {
    switchTab("topology");
    state.selectedRank = focusRank;
    const data = await api("/api/faults/inject", "POST", { fault_id: faultId });
    state.cluster = data.cluster;
    renderCluster();
  }

  async function triggerAutomatedRepair() {
    switchTab("topology");
    if (!state.cluster || !state.cluster.active_fault_id) {
      await injectFault("fault_hbm_ecc", 25);
    }
    const data = await api("/api/repair/execute", "POST", {});
    state.cluster = data.snapshot;
    renderCluster();
  }

  async function updateProfiler() {
    const data = await api("/api/profiler/simulate", "POST", state.profParams);
    state.profiler = data.profiler;
    renderProfiler();
  }

  async function askCopilot(question) {
    const outEl = document.getElementById("copilot-output");
    const btn = document.getElementById("copilot-submit-btn");
    btn.disabled = true;
    btn.textContent = "Querying Amazon Bedrock Nova...";
    outEl.innerHTML = `<div class="copilot-placeholder">Synthesizing live telemetry &amp; compiler flags via <code>us.amazon.nova-lite-v1:0</code>...</div>`;
    try {
      const res = await api("/api/copilot/ask", "POST", { question });
      outEl.innerHTML = `
        <div class="copilot-meta">
          <span>${escapeHtml(res.engine)}</span>
          <span>${res.latency_ms} ms &middot; ${res.grounded_on_fault}</span>
        </div>
        <div class="copilot-body">${formatMarkdown(res.answer)}</div>
      `;
    } catch (err) {
      outEl.innerHTML = `<div style="color:#fca5a5;">Copilot request failed: ${escapeHtml(err.message)}</div>`;
    } finally {
      btn.disabled = false;
      btn.textContent = "Ask Amazon Bedrock Nova \u2192";
    }
  }

  function bindEvents() {
    document.getElementById("tab-topology").addEventListener("click", () => switchTab("topology"));
    document.getElementById("tab-profiler").addEventListener("click", () => switchTab("profiler"));

    // 1-Click Guided Workflows
    document.getElementById("wf-ecc-btn").addEventListener("click", () => injectFault("fault_hbm_ecc", 25));
    document.getElementById("wf-repair-btn").addEventListener("click", triggerAutomatedRepair);
    document.getElementById("wf-efa-btn").addEventListener("click", () => injectFault("fault_efa_straggler", 56));
    document.getElementById("wf-kv-btn").addEventListener("click", () => injectFault("fault_kv_frag", 3));
    document.getElementById("wf-nki-btn").addEventListener("click", async () => {
      switchTab("profiler");
      state.profParams = { tp_degree: 8, precision: "FP8", batch_size: 16, nki_flash_fusion: true, seq_len: 4096 };
      syncProfilerControls();
      await updateProfiler();
      await askCopilot("Give me exact NEURON_CC_FLAGS and NKI FlashAttention speedup analysis for Llama-3.1-70B at FP8 TP=8.");
    });

    document.getElementById("reset-cluster-btn").addEventListener("click", async () => {
      const res = await api("/api/reset", "POST", {});
      state.cluster = res.cluster;
      state.profiler = res.profiler;
      renderCluster();
      renderProfiler();
    });

    // Profiler controls
    document.querySelectorAll("#tp-seg .seg-btn").forEach((b) => {
      b.addEventListener("click", () => {
        state.profParams.tp_degree = Number(b.dataset.tp);
        syncProfilerControls();
        updateProfiler();
      });
    });

    document.querySelectorAll("#prec-seg .seg-btn").forEach((b) => {
      b.addEventListener("click", () => {
        state.profParams.precision = b.dataset.prec;
        syncProfilerControls();
        updateProfiler();
      });
    });

    const slider = document.getElementById("batch-slider");
    slider.addEventListener("input", () => {
      document.getElementById("batch-val-label").textContent = slider.value;
    });
    slider.addEventListener("change", () => {
      state.profParams.batch_size = Number(slider.value);
      updateProfiler();
    });

    document.getElementById("nki-toggle-btn").addEventListener("click", () => {
      state.profParams.nki_flash_fusion = !state.profParams.nki_flash_fusion;
      syncProfilerControls();
      updateProfiler();
    });

    // Copilot
    document.querySelectorAll(".chip-btn").forEach((chip) => {
      chip.addEventListener("click", () => {
        const q = chip.dataset.q;
        document.getElementById("copilot-input").value = q;
        askCopilot(q);
      });
    });

    document.getElementById("copilot-form").addEventListener("submit", (e) => {
      e.preventDefault();
      const q = document.getElementById("copilot-input").value.trim();
      if (q) askCopilot(q);
    });

    // Architecture modal
    const modal = document.getElementById("arch-modal");
    document.getElementById("arch-modal-btn").addEventListener("click", () => modal.classList.remove("hidden"));
    document.getElementById("arch-close-btn").addEventListener("click", () => modal.classList.add("hidden"));
    modal.addEventListener("click", (e) => {
      if (e.target === modal) modal.classList.add("hidden");
    });
  }

  function syncProfilerControls() {
    document.querySelectorAll("#tp-seg .seg-btn").forEach((b) => {
      b.classList.toggle("active", Number(b.dataset.tp) === state.profParams.tp_degree);
    });
    document.querySelectorAll("#prec-seg .seg-btn").forEach((b) => {
      b.classList.toggle("active", b.dataset.prec === state.profParams.precision);
    });
    document.getElementById("batch-slider").value = String(state.profParams.batch_size);
    document.getElementById("batch-val-label").textContent = String(state.profParams.batch_size);
    const nkiBtn = document.getElementById("nki-toggle-btn");
    nkiBtn.classList.toggle("on", state.profParams.nki_flash_fusion);
    nkiBtn.textContent = state.profParams.nki_flash_fusion
      ? "NKI Kernel Fusion: ENABLED"
      : "NKI Kernel Fusion: DISABLED (Unfused Baseline)";
  }

  async function init() {
    bindEvents();
    switchTab(state.activeTab);
    const initial = await api("/api/state");
    state.cluster = initial.cluster;
    state.profiler = initial.profiler;
    renderCluster();
    renderProfiler();
  }

  init();
})();
