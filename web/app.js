/* NeuronPulse Blueprint UI Controller — Stateless browser memory + URL hash (zero web storage) */
(() => {
  const root = document.documentElement;

  const state = {
    cluster: null,
    profiler: null,
    selectedRank: 25,
    activeTab: "topology",
    profParams: {
      tp_degree: 8,
      precision: "FP8",
      batch_size: 16,
      nki_flash_fusion: true,
      seq_len: 4096,
    },
  };

  /* Theme toggle without web storage */
  const themeBtn = document.querySelector(".theme");
  if (themeBtn) {
    themeBtn.addEventListener("click", () => {
      const next = root.dataset.theme === "dark" ? "light" : "dark";
      root.dataset.theme = next;
      history.replaceState(null, "", `#theme=${next}`);
    });
  }

  /* Reading progress bar */
  function updateReadingProgress() {
    const body = document.querySelector(".post-body");
    const fill = document.getElementById("side-bar-fill");
    const pct = document.getElementById("side-pct");
    if (!body || !fill || !pct) return;
    const r = body.getBoundingClientRect();
    const p = Math.min(1, Math.max(0, (window.innerHeight * 0.45 - r.top) / Math.max(r.height, 1)));
    fill.style.transform = `scaleX(${p})`;
    pct.textContent = `${Math.round(p * 100)}%`;
  }
  window.addEventListener("scroll", updateReadingProgress, { passive: true });

  async function api(path, method = "GET", body = null) {
    const opts = { method, headers: { "Content-Type": "application/json" } };
    if (body !== null) opts.body = JSON.stringify(body);
    const res = await fetch(path, opts);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  function esc(str) {
    return String(str || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function formatMd(md) {
    let h = esc(md);
    h = h.replace(/```(?:bash|sh|python)?\n([\s\S]*?)```/g, "<pre><code>$1</code></pre>");
    h = h.replace(/`([^`]+)`/g, "<code>$1</code>");
    h = h.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    h = h.replace(/^### (.*)$/gm, "<strong style='display:block;margin:8px 0 4px;font-family:var(--display);color:var(--ink);'>$1</strong>");
    h = h.replace(/^- (.*)$/gm, "<div style='margin:4px 0;'>&bull; $1</div>");
    h = h.replace(/\n\n/g, "<br>");
    return h;
  }

  function switchTab(tab) {
    state.activeTab = tab;
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
      pill.className = "st-tag bad";
    } else if (c.cordoned_nodes.length > 0) {
      pill.textContent = `64/64 HEALTHY · SPARE ${c.warm_spare_node} ACTIVE`;
      pill.className = "st-tag ok";
    } else {
      pill.textContent = "64/64 NEURONCORES HEALTHY";
      pill.className = "st-tag ok";
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

    // Cross-node correlator callout
    const corrEl = document.getElementById("correlator-banner");
    const corr = c.correlation;
    if (c.active_fault_id && corr && corr.root_cause) {
      const rc = corr.root_cause;
      corrEl.className = "corr-box fault";
      corrEl.innerHTML = `
        <div class="corr-top">
          <div>
            <div class="corr-k">Cross-Node Causal Error Correlator &middot; 100% Root-Rank Isolation</div>
            <div class="corr-title">${esc(c.active_fault.title)}</div>
          </div>
          <button type="button" class="btn signal" id="inline-repair-btn">
            Execute Automated Repair (${corr.requires_node_replacement ? "74s Spare Swap vs. 42m" : "4.4s Online KV Compaction"}) &rarr;
          </button>
        </div>
        <div class="corr-chips">
          <span><b>Root Isolated:</b> <code>${rc.node_id}:${rc.core_id}</code> (Rank ${rc.global_rank})</span>
          <span><b>Noise Reduction:</b> ${corr.raw_alert_count} rank alarms &rarr; 1 root cause (${corr.suppressed_symptom_count} ring stalls suppressed)</span>
          <span><b>Lead Window:</b> +${corr.cascade_propagation_window_ms} ms</span>
        </div>
        <div class="cli-strip">$ ${esc(corr.recommended_cli)}</div>
      `;
      document.getElementById("inline-repair-btn").addEventListener("click", triggerAutomatedRepair);
    } else {
      corrEl.className = "corr-box ok";
      corrEl.innerHTML = `
        <div class="corr-top">
          <div>
            <div class="corr-k">Cross-Node Causal Error Correlator</div>
            <div class="corr-title" style="margin-bottom:0;">All 64 NeuronCores Synchronized &middot; Zero Uncorrectable ECC or EFA SRD Anomalies</div>
          </div>
          <span style="font:400 10.5px var(--mono);color:var(--soft);">Nodes: ${c.active_nodes.join(" · ")}</span>
        </div>
      `;
    }

    // Repair timeline strip
    const repBox = document.getElementById("repair-timeline-box");
    if (c.last_repair && !c.active_fault_id) {
      const rep = c.last_repair;
      repBox.classList.remove("hidden");
      repBox.innerHTML = `
        <div class="corr-k" style="color:var(--ok);">Automated UltraServer Recovery Verified</div>
        <div class="corr-title" style="margin-bottom:4px;">
          Recovered in <strong>${rep.automated_recovery_sec} sec</strong> vs. ${rep.manual_reboot_label} manual baseline
          (<strong>${rep.mttr_speedup_x}&times; faster MTTR</strong>) &middot; Active Replacement: <code>${rep.replacement_node}</code>
        </div>
        <div class="repair-grid">
          ${rep.steps
            .map(
              (s) => `
            <div class="rep-step">
              <b><span>0${s.step} ${s.phase}</span><span>${s.duration_sec}s</span></b>
              <p>${esc(s.detail)}</p>
            </div>`
            )
            .join("")}
        </div>
      `;
    } else {
      repBox.classList.add("hidden");
    }

    // 4 Nodes x 16 NeuronCores Grid
    const grouped = {};
    c.active_nodes.forEach((n) => (grouped[n] = []));
    c.cores.forEach((core) => {
      if (!grouped[core.node_id]) grouped[core.node_id] = [];
      grouped[core.node_id].push(core);
    });

    const nodesGrid = document.getElementById("nodes-grid");
    nodesGrid.innerHTML = c.active_nodes
      .map((nodeId) => {
        const cores = grouped[nodeId] || [];
        const isSpare = nodeId === c.warm_spare_node;
        return `
        <div class="trn-node ${isSpare ? "spare" : ""}">
          <div class="trn-node-h">
            <strong>${nodeId}${isSpare ? " [WARM SPARE]" : ""}</strong>
            <span style="color:var(--soft);">16 Cores &middot; 768 GB HBM3e</span>
          </div>
          <div class="nc-grid-16">
            ${cores
              .map(
                (nc) => `
              <button type="button" class="nc-cell ${nc.status} ${nc.global_rank === state.selectedRank ? "selected" : ""}" data-rank="${nc.global_rank}">
                <div class="nc-top"><span>${nc.core_id}</span><span>R${nc.global_rank}</span></div>
                <div class="nc-sub">${Math.round(nc.hbm3e_bw_gbps)}G &middot; ${nc.temp_c}&deg;</div>
              </button>`
              )
              .join("")}
          </div>
        </div>`;
      })
      .join("");

    nodesGrid.querySelectorAll(".nc-cell").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.selectedRank = Number(btn.dataset.rank);
        renderCluster();
      });
    });

    // Selected NeuronCore Inspector
    const sel = c.cores.find((x) => x.global_rank === state.selectedRank) || c.cores[0];
    document.getElementById("core-inspector").innerHTML = `
      <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;font:500 11px var(--mono);">
        <span>Selected Slice: <code>${sel.node_id}:${sel.core_id}</code> (Rank ${sel.global_rank} &middot; ${sel.efa_nic})</span>
        <span>Status: <strong>${sel.status}</strong> &middot; Trap: <code>${sel.error_code}</code></span>
      </div>
      <div class="insp-cells">
        <div class="insp-c"><small>HBM3e BW</small><strong>${sel.hbm3e_bw_gbps} GB/s</strong></div>
        <div class="insp-c"><small>HBM Used</small><strong>${sel.hbm_used_gb}/${sel.hbm_total_gb} GB</strong></div>
        <div class="insp-c"><small>MFU FLOPs</small><strong>${sel.flops_util_pct}%</strong></div>
        <div class="insp-c"><small>Ring p99</small><strong>${sel.collective_p99_ms} ms</strong></div>
        <div class="insp-c"><small>KV Frag</small><strong>${sel.kv_frag_pct}%</strong></div>
        <div class="insp-c"><small>ECC SBE/DBE</small><strong>${sel.ecc_sbe} / ${sel.ecc_dbe}</strong></div>
      </div>
    `;

    // EFAv3 SRD Collective Ring Hops
    document.getElementById("efa-ring-list").innerHTML = c.efa_links
      .map(
        (lk) => `
      <div class="efa-hop ${lk.status !== "HEALTHY" ? "bad" : ""}">
        <div class="efa-hop-h">
          <span>EFA Ring: ${esc(lk.ring_hop)}</span>
          <span style="color:${lk.status === "HEALTHY" ? "var(--ok)" : "var(--bad)"}">${lk.p99_allreduce_ms} ms p99</span>
        </div>
        <div style="color:var(--soft);">BW ${lk.bandwidth_gbps} Gbps &middot; SRD Retrans ${lk.srd_retransmit_pct}% &middot; Rx ${lk.optical_rx_dbm} dBm</div>
      </div>`
      )
      .join("");
  }

  function renderProfiler() {
    const p = state.profiler;
    if (!p) return;
    const sel = p.selected;
    const d = p.vs_bf16_unfused_baseline;

    document.getElementById("profiler-kpis").innerHTML = `
      <div class="os-kpi">
        <small>Prefill TTFT</small>
        <strong>${sel.fits_in_hbm ? sel.ttft_ms + " ms" : "HBM OOM"}</strong>
        <span>${d.ttft_reduction_pct >= 0 ? "\u2212" + d.ttft_reduction_pct + "% vs. BF16 Unfused" : "Higher Latency"}</span>
      </div>
      <div class="os-kpi">
        <small>64-Core Throughput</small>
        <strong>${sel.fits_in_hbm ? sel.ultraserver_throughput_tok_s.toLocaleString() + " tok/s" : "0 tok/s"}</strong>
        <span>${sel.replicas_on_64_cores}x TP=${sel.tp_degree} (${d.throughput_speedup_x}&times;)</span>
      </div>
      <div class="os-kpi">
        <small>Decode TPOT</small>
        <strong>${sel.tpot_ms} ms/tok</strong>
        <span>Replica: ${sel.replica_throughput_tok_s} tok/s</span>
      </div>
      <div class="os-kpi">
        <small>HBM3e / Core</small>
        <strong style="color:${sel.fits_in_hbm ? "var(--ok)" : "var(--bad)"}">${sel.hbm_used_per_core_gb} / ${sel.hbm_capacity_per_core_gb} GB</strong>
        <span>${sel.fits_in_hbm ? sel.hbm_utilization_pct + "% Utilized" : "EXCEEDS 48GB CORE"}</span>
      </div>
    `;

    const bars = p.comparison_bars;
    const maxTtft = Math.max(...bars.map((b) => b.ttft_ms), 1);
    const maxTput = Math.max(...bars.map((b) => b.ultraserver_throughput_tok_s), 1);

    document.getElementById("ttft-bars").innerHTML = bars
      .map((b, i) => {
        const w = Math.min(100, Math.round((b.ttft_ms / maxTtft) * 100));
        return `
        <div class="pbar-row">
          <div class="pbar-lbl"><span>${esc(b.label)}</span><strong>${b.fits_in_hbm ? b.ttft_ms + " ms" : "OOM"}</strong></div>
          <div class="pbar-track"><div class="pbar-fill ${i === bars.length - 1 ? "active" : ""}" style="width:${w}%"></div></div>
        </div>`;
      })
      .join("");

    document.getElementById("tput-bars").innerHTML = bars
      .map((b, i) => {
        const w = Math.min(100, Math.round((b.ultraserver_throughput_tok_s / maxTput) * 100));
        return `
        <div class="pbar-row">
          <div class="pbar-lbl"><span>${esc(b.label)}</span><strong>${b.ultraserver_throughput_tok_s.toLocaleString()} tok/s</strong></div>
          <div class="pbar-track"><div class="pbar-fill teal ${i === bars.length - 1 ? "active" : ""}" style="width:${w}%"></div></div>
        </div>`;
      })
      .join("");

    const mb = sel.memory_breakdown_gb;
    document.getElementById("profiler-footer-card").innerHTML = `
      <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;font:500 11px var(--mono);">
        <span>Per-Core HBM3e: Weights <code>${mb.weights_per_core_gb} GB</code> + KV <code>${mb.kv_cache_per_core_gb} GB</code> + NKI Scratch <code>${mb.activations_scratch_gb} GB</code></span>
        <span>Limiter: <strong>${sel.bottleneck}</strong></span>
      </div>
      <div class="cli-strip">export NEURON_CC_FLAGS="${esc(sel.neuron_cc_flags)}"</div>
    `;
  }

  async function injectFault(faultId, rank) {
    switchTab("topology");
    state.selectedRank = rank;
    const res = await api("/api/faults/inject", "POST", { fault_id: faultId });
    state.cluster = res.cluster;
    renderCluster();
  }

  async function triggerAutomatedRepair() {
    switchTab("topology");
    if (!state.cluster || !state.cluster.active_fault_id) {
      await injectFault("fault_hbm_ecc", 25);
    }
    const res = await api("/api/repair/execute", "POST", {});
    state.cluster = res.snapshot;
    renderCluster();
  }

  async function updateProfiler() {
    const res = await api("/api/profiler/simulate", "POST", state.profParams);
    state.profiler = res.profiler;
    renderProfiler();
  }

  async function askCopilot(question) {
    const out = document.getElementById("copilot-output");
    const btn = document.getElementById("copilot-submit-btn");
    btn.disabled = true;
    btn.textContent = "Querying Amazon Bedrock Nova...";
    out.innerHTML = `Synthesizing live telemetry &amp; compiler runbook via <code>us.amazon.nova-lite-v1:0</code>...`;
    try {
      const res = await api("/api/copilot/ask", "POST", { question });
      out.innerHTML = `
        <div class="cop-meta"><span>${esc(res.engine)}</span><span>${res.latency_ms} ms &middot; ${esc(res.grounded_on_fault)}</span></div>
        <div>${formatMd(res.answer)}</div>
      `;
    } catch (e) {
      out.innerHTML = `<span style="color:var(--bad);">Request error: ${esc(e.message)}</span>`;
    } finally {
      btn.disabled = false;
      btn.textContent = "Ask Amazon Bedrock Nova \u2192";
    }
  }

  function syncProfilerControls() {
    document.querySelectorAll("#tp-seg .seg-b").forEach((b) => {
      b.classList.toggle("active", Number(b.dataset.tp) === state.profParams.tp_degree);
    });
    document.querySelectorAll("#prec-seg .seg-b").forEach((b) => {
      b.classList.toggle("active", b.dataset.prec === state.profParams.precision);
    });
    document.getElementById("batch-slider").value = String(state.profParams.batch_size);
    document.getElementById("batch-val-label").textContent = String(state.profParams.batch_size);
    const nkiBtn = document.getElementById("nki-toggle-btn");
    nkiBtn.className = state.profParams.nki_flash_fusion ? "btn accent" : "btn";
    nkiBtn.textContent = state.profParams.nki_flash_fusion
      ? "NKI Kernel Fusion: ENABLED"
      : "NKI Kernel Fusion: DISABLED (Unfused)";
  }

  function bind() {
    document.getElementById("tab-topology").addEventListener("click", () => switchTab("topology"));
    document.getElementById("tab-profiler").addEventListener("click", () => switchTab("profiler"));

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

    document.querySelectorAll("#tp-seg .seg-b").forEach((b) => {
      b.addEventListener("click", () => {
        state.profParams.tp_degree = Number(b.dataset.tp);
        syncProfilerControls();
        updateProfiler();
      });
    });

    document.querySelectorAll("#prec-seg .seg-b").forEach((b) => {
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

    document.querySelectorAll(".cop-chip").forEach((c) => {
      c.addEventListener("click", () => {
        const q = c.dataset.q;
        document.getElementById("copilot-input").value = q;
        askCopilot(q);
      });
    });

    document.getElementById("copilot-form").addEventListener("submit", (e) => {
      e.preventDefault();
      const q = document.getElementById("copilot-input").value.trim();
      if (q) askCopilot(q);
    });
  }

  async function init() {
    bind();
    const initial = await api("/api/state");
    state.cluster = initial.cluster;
    state.profiler = initial.profiler;
    renderCluster();
    renderProfiler();
    updateReadingProgress();
  }

  init();
})();
