/* NeuronPulse Silicon-Die Oscilloscope Controller — Stateless browser memory */
(() => {
  const state = {
    cluster: null,
    profiler: null,
    selectedRank: 25,
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

  function esc(s) {
    return String(s || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function formatTerminal(md) {
    let h = esc(md);
    h = h.replace(/```(?:bash|sh|python)?\n([\s\S]*?)```/g, "<pre><code>$1</code></pre>");
    h = h.replace(/`([^`]+)`/g, "<code style='color:#00e5ff'>$1</code>");
    h = h.replace(/\*\*([^*]+)\*\*/g, "<strong style='color:#fff'>$1</strong>");
    h = h.replace(/^### (.*)$/gm, "<div style='color:#10f59c;font-weight:600;margin:6px 0 4px;'>$1</div>");
    h = h.replace(/^- (.*)$/gm, "<div style='margin:3px 0;'>&bull; $1</div>");
    h = h.replace(/\n\n/g, "<br>");
    return h;
  }

  function renderWafer() {
    const c = state.cluster;
    if (!c) return;
    const sum = c.summary;
    const appRoot = document.getElementById("app-root");
    const pulse = document.getElementById("cluster-pulse");
    const pulseTxt = document.getElementById("cluster-pulse-text");

    appRoot.classList.toggle("is-impaired", Boolean(c.active_fault_id));
    appRoot.classList.toggle("is-spared", !c.active_fault_id && c.cordoned_nodes.length > 0);

    if (c.active_fault_id) {
      pulse.className = "pulse-badge impaired";
      pulseTxt.textContent = `IMPAIRED // ${c.active_fault.hardware_code}`;
    } else if (c.cordoned_nodes.length > 0) {
      pulse.className = "pulse-badge spared";
      pulseTxt.textContent = `64/64 NOMINAL // ${c.warm_spare_node} ACTIVE`;
    } else {
      pulse.className = "pulse-badge";
      pulseTxt.textContent = "64/64 CORES NOMINAL";
    }

    // Highlight active sabotage switch
    document.querySelectorAll(".sw-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.fault === c.active_fault_id);
    });

    document.getElementById("kpi-cores").textContent = `${sum.healthy_cores} / ${sum.total_neuron_cores}`;
    document.getElementById("kpi-nodes-sub").textContent =
      c.cordoned_nodes.length > 0 ? `Cordoned: ${c.cordoned_nodes.join(", ")}` : "4x trn2.48xlarge Dies";
    document.getElementById("kpi-bw").textContent = `${sum.avg_hbm3e_bw_gbps.toLocaleString()} GB/s`;
    document.getElementById("kpi-p99").textContent = `${sum.collective_ring_p99_ms} ms`;
    document.getElementById("kpi-mfu").textContent = `${sum.avg_flops_util_pct}%`;
    document.getElementById("kpi-temp").textContent = `Peak Die: ${sum.peak_temp_c}\u00B0C · KV Frag: ${sum.avg_kv_frag_pct}%`;
    document.getElementById("kpi-spare").textContent = c.warm_spare_node;
    document.getElementById("kpi-spare-state").textContent = c.spare_state;

    // Cross-node causal correlator strip
    const corrEl = document.getElementById("correlator-strip");
    const corr = c.correlation;
    if (c.active_fault_id && corr && corr.root_cause) {
      const rc = corr.root_cause;
      corrEl.className = "correlator-strip fault";
      corrEl.innerHTML = `
        <div class="corr-row">
          <div>
            <div style="font:600 10.5px var(--mono);color:#ff6d00;letter-spacing:.08em;">CAUSAL CORRELATOR // 100% ROOT-RANK ISOLATION</div>
            <div class="corr-headline">${esc(c.active_fault.title)}</div>
          </div>
          <button type="button" class="btn repair" id="inline-swap-btn">
            &#9889; ${corr.requires_node_replacement ? "Cordon & Swap trn2-spare-01 (74s)" : "Compact PagedKV In-Place (4.4s)"}
          </button>
        </div>
        <div class="corr-badges">
          <span>Root Die: <strong style="color:#ff3d57">${rc.node_id}:${rc.core_id}</strong> (Rank ${rc.global_rank})</span>
          <span>Alert Compression: <strong>${corr.raw_alert_count} &rarr; 1</strong> (${corr.suppressed_symptom_count} AllReduce timeouts suppressed)</span>
          <span>Lead Delta: <strong>+${corr.cascade_propagation_window_ms} ms</strong></span>
        </div>
        <div class="term-cmd">$ ${esc(corr.recommended_cli)}</div>
      `;
      document.getElementById("inline-swap-btn").addEventListener("click", runRepair);
    } else {
      corrEl.className = "correlator-strip";
      corrEl.innerHTML = `
        <div class="corr-row">
          <span class="mono" style="font-size:12px;color:#10f59c;">&#10003; Causal Correlator Watching 64 Ranks &mdash; 0 Uncorrectable ECC Traps &middot; 0 EFA SRD Retransmit Surges</span>
          <span class="dim mono" style="font-size:11px;">Ring: ${c.active_nodes.join(" \u2192 ")}</span>
        </div>
      `;
    }

    // 5-Step Automated Repair FSM
    const fsm = document.getElementById("fsm-box");
    if (c.last_repair && !c.active_fault_id) {
      const rep = c.last_repair;
      fsm.classList.remove("hidden");
      fsm.innerHTML = `
        <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;font:600 12px var(--mono);color:#00e5ff;">
          <span>AUTOMATED RECOVERY COMPLETE: ${rep.automated_recovery_sec}s vs ${rep.manual_reboot_label} manual (${rep.mttr_speedup_x}&times; faster)</span>
          <span>Active Node: ${rep.replacement_node}</span>
        </div>
        <div class="fsm-steps">
          ${rep.steps
            .map(
              (s) => `
            <div class="fsm-step">
              <b><span>0${s.step} ${s.phase}</span><span>${s.duration_sec}s</span></b>
              <div class="dim">${esc(s.detail)}</div>
            </div>`
            )
            .join("")}
        </div>
      `;
    } else {
      fsm.classList.add("hidden");
    }

    // 4 trn2 Dies x 16 NeuronCores
    const grouped = {};
    c.active_nodes.forEach((n) => (grouped[n] = []));
    c.cores.forEach((core) => {
      if (!grouped[core.node_id]) grouped[core.node_id] = [];
      grouped[core.node_id].push(core);
    });

    const diesGrid = document.getElementById("dies-grid");
    diesGrid.innerHTML = c.active_nodes
      .map((nodeId) => {
        const cores = grouped[nodeId] || [];
        const isSpare = nodeId === c.warm_spare_node;
        return `
        <div class="die-package ${isSpare ? "spare" : ""}">
          <div class="die-head">
            <strong style="color:${isSpare ? "#00e5ff" : "#e4f0fb"}">${nodeId}${isSpare ? " // WARM SPARE" : ""}</strong>
            <span class="dim">16 Cores &middot; 768 GB HBM3e</span>
          </div>
          <div class="cores-4x4">
            ${cores
              .map(
                (nc) => `
              <button type="button" class="core-die ${nc.status} ${nc.global_rank === state.selectedRank ? "selected" : ""}" data-rank="${nc.global_rank}">
                <div class="core-id"><span>${nc.core_id}</span><span>R${nc.global_rank}</span></div>
                <div class="core-tele">${Math.round(nc.hbm3e_bw_gbps)}G &middot; ${nc.temp_c}&deg;</div>
              </button>`
              )
              .join("")}
          </div>
        </div>`;
      })
      .join("");

    diesGrid.querySelectorAll(".core-die").forEach((b) => {
      b.addEventListener("click", () => {
        state.selectedRank = Number(b.dataset.rank);
        renderWafer();
      });
    });

    // Selected Core Probe
    const sel = c.cores.find((x) => x.global_rank === state.selectedRank) || c.cores[0];
    document.getElementById("probe-drawer").innerHTML = `
      <div style="display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;font:500 11.5px var(--mono);">
        <span>CHANNEL PROBE // <strong style="color:#00e5ff">${sel.node_id}:${sel.core_id}</strong> (Global Rank ${sel.global_rank} &middot; ${sel.efa_nic})</span>
        <span>State: <strong>${sel.status}</strong> &middot; Code: <code style="color:${sel.error_code === "NONE" ? "#10f59c" : "#ff6d00"}">${sel.error_code}</code></span>
      </div>
      <div class="probe-grid">
        <div class="probe-cell"><span>HBM3e BW</span>${sel.hbm3e_bw_gbps} GB/s</div>
        <div class="probe-cell"><span>HBM Slice</span>${sel.hbm_used_gb}/${sel.hbm_total_gb} GB</div>
        <div class="probe-cell"><span>Tensor MFU</span>${sel.flops_util_pct}%</div>
        <div class="probe-cell"><span>Ring p99</span>${sel.collective_p99_ms} ms</div>
        <div class="probe-cell"><span>KV Frag</span>${sel.kv_frag_pct}%</div>
        <div class="probe-cell"><span>ECC SBE/DBE</span>${sel.ecc_sbe} / ${sel.ecc_dbe}</div>
      </div>
    `;

    // EFAv3 SRD Collective Ring
    document.getElementById("efa-grid").innerHTML = c.efa_links
      .map(
        (lk) => `
      <div class="efa-card ${lk.status !== "HEALTHY" ? "degraded" : ""}">
        <div class="efa-top">
          <span>${esc(lk.ring_hop)}</span>
          <span style="color:${lk.status === "HEALTHY" ? "#10f59c" : "#ff6d00"}">${lk.p99_allreduce_ms} ms p99</span>
        </div>
        <div class="dim">SRD ${lk.bandwidth_gbps} Gbps &middot; Retrans ${lk.srd_retransmit_pct}% &middot; Rx ${lk.optical_rx_dbm} dBm</div>
      </div>`
      )
      .join("");
  }

  function renderScope() {
    const p = state.profiler;
    if (!p) return;
    const sel = p.selected;
    const d = p.vs_bf16_unfused_baseline;

    document.getElementById("osc-speedup-tag").textContent = sel.fits_in_hbm
      ? `${d.throughput_speedup_x}\u00D7 tok/s · ${d.ttft_reduction_pct >= 0 ? "\u2212" + d.ttft_reduction_pct : "+" + Math.abs(d.ttft_reduction_pct)}% TTFT`
      : "HBM3E CAPACITY EXCEEDED";

    document.getElementById("osc-kpis").innerHTML = `
      <div class="osc-kpi">
        <small>Prefill TTFT</small>
        <strong style="color:${sel.fits_in_hbm ? "#00e5ff" : "#ff3d57"}">${sel.fits_in_hbm ? sel.ttft_ms + " ms" : "HBM OOM"}</strong>
      </div>
      <div class="osc-kpi">
        <small>64-Core Tok/s</small>
        <strong>${sel.fits_in_hbm ? sel.ultraserver_throughput_tok_s.toLocaleString() : "0"}</strong>
      </div>
      <div class="osc-kpi">
        <small>Core HBM3e</small>
        <strong style="color:${sel.fits_in_hbm ? "#10f59c" : "#ff3d57"}">${sel.hbm_used_per_core_gb}/${sel.hbm_capacity_per_core_gb}G</strong>
      </div>
    `;

    const bars = p.comparison_bars;
    const maxTtft = Math.max(...bars.map((b) => b.ttft_ms), 1);
    const maxTput = Math.max(...bars.map((b) => b.ultraserver_throughput_tok_s), 1);

    document.getElementById("ttft-bars").innerHTML = bars
      .map((b, i) => {
        const w = Math.min(100, Math.round((b.ttft_ms / maxTtft) * 100));
        return `
        <div class="tbar-row">
          <div class="tbar-lbl"><span>${esc(b.label)}</span><strong>${b.fits_in_hbm ? b.ttft_ms + "ms" : "OOM"}</strong></div>
          <div class="tbar-track"><div class="tbar-fill ${i === bars.length - 1 ? "current" : ""}" style="width:${w}%"></div></div>
        </div>`;
      })
      .join("");

    document.getElementById("tput-bars").innerHTML = bars
      .map((b, i) => {
        const w = Math.min(100, Math.round((b.ultraserver_throughput_tok_s / maxTput) * 100));
        return `
        <div class="tbar-row">
          <div class="tbar-lbl"><span>${esc(b.label)}</span><strong>${b.ultraserver_throughput_tok_s.toLocaleString()}</strong></div>
          <div class="tbar-track"><div class="tbar-fill tput ${i === bars.length - 1 ? "current" : ""}" style="width:${w}%"></div></div>
        </div>`;
      })
      .join("");

    document.getElementById("cc-flags-box").textContent = `NEURON_CC_FLAGS="${sel.neuron_cc_flags}"`;
  }

  async function injectFault(faultId, rank) {
    state.selectedRank = rank;
    const res = await api("/api/faults/inject", "POST", { fault_id: faultId });
    state.cluster = res.cluster;
    renderWafer();
  }

  async function runRepair() {
    if (!state.cluster || !state.cluster.active_fault_id) {
      await injectFault("fault_hbm_ecc", 25);
    }
    const res = await api("/api/repair/execute", "POST", {});
    state.cluster = res.snapshot;
    renderWafer();
  }

  async function updateScope() {
    const res = await api("/api/profiler/simulate", "POST", state.profParams);
    state.profiler = res.profiler;
    renderScope();
  }

  async function askNova(question) {
    const screen = document.getElementById("term-screen");
    const btn = document.getElementById("term-submit");
    btn.disabled = true;
    screen.innerHTML = `<span style="color:#00e5ff;">// Invoking us.amazon.nova-lite-v1:0 against live 64-core wafer telemetry...</span>`;
    try {
      const res = await api("/api/copilot/ask", "POST", { question });
      screen.innerHTML = `
        <div class="term-meta"><span>${esc(res.engine)}</span><span>${res.latency_ms} ms // ${esc(res.grounded_on_fault)}</span></div>
        <div>${formatTerminal(res.answer)}</div>
      `;
    } catch (e) {
      screen.innerHTML = `<span style="color:#ff3d57;">// Error: ${esc(e.message)}</span>`;
    } finally {
      btn.disabled = false;
    }
  }

  function syncScopeControls() {
    document.querySelectorAll("#tp-seg button").forEach((b) => {
      b.classList.toggle("active", Number(b.dataset.tp) === state.profParams.tp_degree);
    });
    document.querySelectorAll("#prec-seg button").forEach((b) => {
      b.classList.toggle("active", b.dataset.prec === state.profParams.precision);
    });
    const nkiBtn = document.getElementById("nki-btn");
    nkiBtn.className = state.profParams.nki_flash_fusion ? "btn cyan" : "btn";
    nkiBtn.textContent = state.profParams.nki_flash_fusion ? "NKI Fusion: ON" : "NKI Fusion: OFF";
  }

  function bind() {
    document.getElementById("sw-ecc").addEventListener("click", () => injectFault("fault_hbm_ecc", 25));
    document.getElementById("sw-efa").addEventListener("click", () => injectFault("fault_efa_straggler", 56));
    document.getElementById("sw-kv").addEventListener("click", () => injectFault("fault_kv_frag", 3));
    document.getElementById("exec-repair-btn").addEventListener("click", runRepair);

    document.getElementById("reset-btn").addEventListener("click", async () => {
      const res = await api("/api/reset", "POST", {});
      state.cluster = res.cluster;
      state.profiler = res.profiler;
      renderWafer();
      renderScope();
    });

    document.querySelectorAll("#tp-seg button").forEach((b) => {
      b.addEventListener("click", () => {
        state.profParams.tp_degree = Number(b.dataset.tp);
        syncScopeControls();
        updateScope();
      });
    });

    document.querySelectorAll("#prec-seg button").forEach((b) => {
      b.addEventListener("click", () => {
        state.profParams.precision = b.dataset.prec;
        syncScopeControls();
        updateScope();
      });
    });

    const slider = document.getElementById("batch-slider");
    slider.addEventListener("input", () => {
      document.getElementById("batch-lbl").textContent = slider.value;
    });
    slider.addEventListener("change", () => {
      state.profParams.batch_size = Number(slider.value);
      updateScope();
    });

    document.getElementById("nki-btn").addEventListener("click", () => {
      state.profParams.nki_flash_fusion = !state.profParams.nki_flash_fusion;
      syncScopeControls();
      updateScope();
    });

    document.querySelectorAll(".pchip").forEach((c) => {
      c.addEventListener("click", () => {
        const q = c.dataset.q;
        document.getElementById("term-input").value = q;
        askNova(q);
      });
    });

    document.getElementById("term-form").addEventListener("submit", (e) => {
      e.preventDefault();
      const q = document.getElementById("term-input").value.trim();
      if (q) askNova(q);
    });
  }

  async function init() {
    bind();
    const initial = await api("/api/state");
    state.cluster = initial.cluster;
    state.profiler = initial.profiler;
    renderWafer();
    renderScope();
  }

  init();
})();
