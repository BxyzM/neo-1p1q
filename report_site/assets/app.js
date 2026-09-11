/*
 * Render the static Neo1P1Q experiment dashboard.
 *
 * Author: Aritra Bal (ETP)
 * Date: 2026-09-10
 *
 * Data contract:
 *   data/index.json lists experiment summaries under `experiments`; each item
 *   has id, status, run_count, successful_runs, mean_auc, std_auc, total_jets,
 *   loss, device, and mode.
 *   data/<id>.json provides `summary`, `validation`, `roc`, `epoch_times`,
 *   `compile_times`, `circuit_diagram`, `score_distribution`, `runs`, `config`, and `notices`.
 *   Missing values are displayed rather than inferred. `score_distribution.runs` is shown per seed
 *   only -- it has no aggregate view. `compile_times` (jax-only) is
 *   `{train?: {mean, std, n}, val?: {mean, std, n}}`, each key absent when no run in the experiment
 *   has that measurement; shown as plain stat cards, not a chart. `circuit_diagram` is a root-relative
 *   PNG path (one per experiment, structure only, not trained values) or null. Each entry in `runs`
 *   carries `aux_weights: {scale_factor, bias, hamiltonian_coeffs}` (or `{}`); hamiltonian_coeffs
 *   feeds the per-seed Hamiltonian formula and is not displayed as raw numbers itself.
 */

"use strict";

const app = document.querySelector("#app");
const params = new URLSearchParams(window.location.search);
const experimentId = params.get("experiment");
const COLORS = [
  "#7fdb95", "#e0916a", "#7fb1e0", "#c295cc", "#e0c15c",
  "#5cc9c2", "#e0a394", "#a8c96f", "#b3a3e0", "#e08aa0",
];

const PLOT_CONFIG = {
  responsive: true,
  displaylogo: false,
  modeBarButtonsToRemove: ["lasso2d", "select2d"],
  toImageButtonOptions: { format: "png", scale: 2 },
};

document.addEventListener("DOMContentLoaded", () => {
  if (experimentId && !/^\d+$/.test(experimentId)) {
    renderError("Invalid experiment", "The experiment identifier must contain digits only.");
    return;
  }
  loadPage();
});

async function loadPage() {
  const source = experimentId ? `data/${experimentId}.json` : "data/index.json";
  try {
    const response = await fetch(source, { cache: "no-cache" });
    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }
    const data = await response.json();
    if (experimentId) {
      renderReport(data);
    } else {
      renderIndex(data);
    }
  } catch (error) {
    console.error(error);
    renderError(
      experimentId ? `Experiment ${experimentId} is unavailable` : "Reports are unavailable",
      "The report data could not be loaded. Check that the generated JSON files are present and try again.",
    );
  }
}

function renderIndex(data) {
  const experiments = Array.isArray(data.experiments) ? [...data.experiments] : [];
  experiments.sort((a, b) => Number(b.id) - Number(a.id));
  document.title = "Neo1P1Q | Experiment reports";

  if (!experiments.length) {
    app.innerHTML = `
      <section class="empty-state">
        <p class="eyebrow">Neo1P1Q</p>
        <h1>No reports yet</h1>
        <p>Generated experiment reports will appear here.</p>
      </section>`;
    return;
  }

  app.innerHTML = `
    <header>
      <p class="eyebrow">Quantum classifier study</p>
      <h1>Experiment reports</h1>
      <p class="lede">Training and inference results across reproducible random-seed ensembles.</p>
      <span class="report-count">${plural(experiments.length, "experiment")} available</span>
    </header>
    <section class="experiment-grid" aria-label="Experiments">
      ${experiments.map(experimentCard).join("")}
    </section>`;
}

function experimentCard(experiment) {
  const id = safeText(experiment.id);
  const successful = numeric(experiment.successful_runs);
  const runs = numeric(experiment.run_count);
  const status = statusInfo(experiment.status, successful, runs);
  return `
    <a class="experiment-card" href="?experiment=${encodeURIComponent(String(experiment.id))}">
      <div class="experiment-card__top">
        <span class="experiment-card__number">${id}</span>
        <span class="status ${status.className}">${safeText(status.label)}</span>
      </div>
      <div class="experiment-card__metrics">
        <span><span class="metric-label">Test AUC</span><span class="metric-value">${aucWithError(experiment.mean_auc, experiment.std_auc)}</span></span>
        <span><span class="metric-label">Runs</span><span class="metric-value">${successful === null || runs === null ? "Unavailable" : `${successful} / ${runs}`}</span></span>
        <span><span class="metric-label">Evaluated jets</span><span class="metric-value">${formatInteger(experiment.total_jets)}</span></span>
        <span><span class="metric-label">Loss</span><span class="metric-value">${safeText(displayValue(experiment.loss))}</span></span>
        <span><span class="metric-label">Device</span><span class="metric-value">${safeText(displayValue(experiment.device))}</span></span>
        <span><span class="metric-label">Mode</span><span class="metric-value">${safeText(displayValue(experiment.mode))}</span></span>
      </div>
      <span class="card-arrow" aria-hidden="true">&#8594;</span>
    </a>`;
}

function renderReport(data) {
  const id = String(data.experiment?.id ?? experimentId);
  const summary = data.summary ?? {};
  const successful = numeric(summary.successful_runs);
  const runCount = numeric(summary.run_count);
  const status = statusInfo(data.experiment?.status, successful, runCount);
  const seeds = (Array.isArray(data.runs) ? data.runs : []).map((run) => run.random_seed);
  const validSeeds = seeds.filter((seed) => seed !== null && seed !== undefined);
  const seedRange = validSeeds.length ? `${validSeeds[0]}${validSeeds.length > 1 ? `–${validSeeds.at(-1)}` : ""}` : "Unavailable";
  document.title = `Neo1P1Q ${id} | Experiment report`;

  // compile_times is jax-only and absent for autograd runs or older reports --
  // each stat card is skipped entirely rather than shown as "Unavailable".
  const compileTimes = data.compile_times ?? {};
  const compileTimeCards = [
    ["train", "Avg. circuit compilation time (training)"],
    ["val", "Avg. circuit compilation time (validation)"],
  ]
    .map(([key, label]) => {
      const value = secondsWithError(compileTimes[key]?.mean, compileTimes[key]?.std);
      return value === null ? "" : statCard(label, value);
    })
    .join("");

  app.innerHTML = `
    <a class="back-link" href="./"><span aria-hidden="true">&#8592;</span> All experiments</a>
    <header class="report-header">
      <p class="eyebrow">Experiment report</p>
      <div class="report-heading">
        <h1 class="report-title">Seed ${safeText(id)}</h1>
        <span class="status ${status.className}">${safeText(status.label)}</span>
      </div>
      <p class="report-meta">Generated ${formatDateTime(data.generated_at)}</p>
      <div class="stat-grid">
        ${statCard("Mean test AUC", aucWithError(summary.mean_auc, summary.std_auc))}
        ${statCard("Successful runs", successful === null || runCount === null ? "Unavailable" : `${successful} / ${runCount}`)}
        ${statCard("Evaluated jets", formatInteger(summary.total_jets))}
        ${statCard("Jets per run", formatInteger(summary.jets_per_run))}
        ${statCard("Random seeds", safeText(seedRange))}
        ${compileTimeCards}
      </div>
    </header>

    ${renderNotices(data.notices)}

    <section class="report-section" aria-labelledby="circuit-title">
      <div class="section-heading">
        <div><p class="eyebrow">Architecture</p><h2 id="circuit-title">Circuit</h2></div>
        <p>The variational circuit structure shared by every run in this experiment.</p>
      </div>
      ${renderCircuitDiagram(data.circuit_diagram)}
    </section>

    <section class="report-section" aria-labelledby="model-summary-title">
      <div class="section-heading">
        <div><p class="eyebrow">Architecture</p><h2 id="model-summary-title">Trained model</h2></div>
        <p>Final auxiliary weights and measurement Hamiltonian, per random seed.</p>
      </div>
      ${renderModelSummaryCard(data.runs)}
    </section>

    <section class="report-section" aria-labelledby="validation-title">
      <div class="section-heading">
        <div><p class="eyebrow">Training</p><h2 id="validation-title">Validation AUC</h2></div>
        <p>Mean validation performance by epoch. Error bars show one standard deviation across contributing runs.</p>
      </div>
      ${chartCard("Validation AUC by epoch", "validation-chart", "validation")}
    </section>

    <section class="report-section" aria-labelledby="epoch-time-title">
      <div class="section-heading">
        <div><p class="eyebrow">Training</p><h2 id="epoch-time-title">Epoch time</h2></div>
        <p>Wall-clock training time per epoch, in seconds. Error bars show one standard deviation across contributing runs.</p>
      </div>
      ${chartCard("Training time by epoch", "epoch-time-chart", "epoch_times")}
    </section>

    <section class="report-section" aria-labelledby="roc-title">
      <div class="section-heading">
        <div><p class="eyebrow">Inference</p><h2 id="roc-title">ROC curve</h2></div>
        <p>The aggregate curve includes a one-standard-deviation band. Use the toggle to inspect each random seed.</p>
      </div>
      ${chartCard("Receiver operating characteristic", "roc-chart", "roc")}
    </section>

    <section class="report-section" aria-labelledby="score-title">
      <div class="section-heading">
        <div><p class="eyebrow">Inference</p><h2 id="score-title">Classifier score distribution</h2></div>
        <p>Signal and background test-set score densities, shown independently per random seed.</p>
      </div>
      ${renderScoreDistributionCard(data.score_distribution)}
    </section>

    <section class="report-section" aria-labelledby="runs-title">
      <div class="section-heading">
        <div><p class="eyebrow">Run detail</p><h2 id="runs-title">Evaluated runs</h2></div>
      </div>
      ${renderRuns(data.runs)}
    </section>

    <section class="report-section" aria-labelledby="config-title">
      <div class="section-heading">
        <div><p class="eyebrow">Configuration</p><h2 id="config-title">Hyperparameters &amp; execution</h2></div>
      </div>
      ${renderConfig(data.config)}
      ${renderConfigVariations(data.config_variations)}
    </section>`;

  renderValidationPlot(data.validation ?? {}, "aggregate");
  renderEpochTimePlot(data.epoch_times ?? {}, "aggregate");
  renderRocPlot(data.roc ?? {}, "aggregate", summary);
  setupChartToggle("validation", (mode) => renderValidationPlot(data.validation ?? {}, mode));
  setupChartToggle("epoch_times", (mode) => renderEpochTimePlot(data.epoch_times ?? {}, mode));
  setupChartToggle("roc", (mode) => renderRocPlot(data.roc ?? {}, mode, summary));

  const modelSummaryRuns = sortedRuns(data.runs).filter((run) => run.aux_weights && Object.keys(run.aux_weights).length);
  if (modelSummaryRuns.length) {
    renderModelSummary(modelSummaryRuns, modelSummaryRuns[0].random_seed);
    setupSeedToggle("model-seed-selector", (seed) => renderModelSummary(modelSummaryRuns, seed));
  }

  const distributionRuns = sortedRuns(data.score_distribution?.runs);
  const scoreAxisTitle = data.config?.optimization?.loss === "BCE" ? "Predicted probability" : "Classifier score";
  if (distributionRuns.length) {
    renderScoreDistributionPlot(distributionRuns, distributionRuns[0].random_seed, scoreAxisTitle);
    setupSeedToggle("score-seed-selector", (seed) => renderScoreDistributionPlot(distributionRuns, seed, scoreAxisTitle));
  }
}

function statCard(label, value) {
  return `<div class="stat-card"><span class="stat-label">${safeText(label)}</span><span class="stat-value">${value}</span></div>`;
}

function chartCard(label, id, group) {
  return `
    <div class="chart-card">
      <div class="chart-toolbar">
        <strong>${safeText(label)}</strong>
        <div class="segmented" role="group" aria-label="${safeText(label)} view">
          <button type="button" data-chart="${group}" data-mode="aggregate" aria-pressed="true">Aggregate</button>
          <button type="button" data-chart="${group}" data-mode="individual" aria-pressed="false">Individual</button>
        </div>
      </div>
      <div class="chart" id="${id}" role="img" aria-label="${safeText(label)} plot"></div>
    </div>`;
}

function setupChartToggle(group, render) {
  const buttons = [...document.querySelectorAll(`[data-chart="${group}"]`)];
  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      buttons.forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
      render(button.dataset.mode);
    });
  });
}

function renderValidationPlot(validation, mode) {
  const target = document.querySelector("#validation-chart");
  if (!window.Plotly) {
    renderChartError(target, "Plotly could not be loaded.");
    return;
  }

  let traces = [];
  if (mode === "individual") {
    const runs = sortedRuns(validation.runs);
    traces = runs.map((run, index) => ({
      x: (run.points ?? []).map((point) => point.epoch),
      y: (run.points ?? []).map((point) => point.auc),
      type: "scatter",
      mode: "lines+markers",
      name: `Seed ${run.random_seed}`,
      line: { color: runColor(run.random_seed, index), width: 2 },
      marker: { size: 5 },
      hovertemplate: `Seed ${safeText(run.random_seed)}<br>Epoch %{x}<br>Validation AUC %{y:.4f}<extra></extra>`,
    })).filter((trace) => trace.x.length);
  } else {
    const points = Array.isArray(validation.aggregate) ? validation.aggregate : [];
    traces = points.length ? [{
      x: points.map((point) => point.epoch),
      y: points.map((point) => point.mean ?? point.mean_auc),
      customdata: points.map((point) => [point.std ?? point.std_auc, point.n ?? validation.run_count]),
      error_y: {
        type: "data",
        array: points.map((point) => point.std ?? point.std_auc),
        color: "rgba(127,219,149,0.55)",
        thickness: 1.2,
        width: 3,
        visible: true,
      },
      type: "scatter",
      mode: "lines+markers",
      name: "Mean AUC",
      line: { color: "#7fdb95", width: 3 },
      marker: { color: "#7fdb95", size: 7 },
      hovertemplate: "Epoch %{x}<br>Mean AUC %{y:.4f}<br>Std. dev. %{customdata[0]:.4f}<br>%{customdata[1]} contributing runs<extra></extra>",
    }] : [];
  }
  drawPlot(target, traces, {
    xaxis: { title: "Epoch", rangemode: "tozero", dtick: 1 },
    yaxis: { title: "Validation AUC", range: [0, 1] },
    showlegend: mode === "individual",
  }, "No validation history is available for this experiment.");
}

function renderEpochTimePlot(epochTimes, mode) {
  const target = document.querySelector("#epoch-time-chart");
  if (!window.Plotly) {
    renderChartError(target, "Plotly could not be loaded.");
    return;
  }

  let traces = [];
  if (mode === "individual") {
    const runs = sortedRuns(epochTimes.runs);
    traces = runs.map((run, index) => ({
      x: (run.points ?? []).map((point) => point.epoch),
      y: (run.points ?? []).map((point) => point.seconds),
      type: "scatter",
      mode: "lines+markers",
      name: `Seed ${run.random_seed}`,
      line: { color: runColor(run.random_seed, index), width: 2 },
      marker: { size: 5 },
      hovertemplate: `Seed ${safeText(run.random_seed)}<br>Epoch %{x}<br>Time %{y:.2f}s<extra></extra>`,
    })).filter((trace) => trace.x.length);
  } else {
    const points = Array.isArray(epochTimes.aggregate) ? epochTimes.aggregate : [];
    traces = points.length ? [{
      x: points.map((point) => point.epoch),
      y: points.map((point) => point.mean),
      customdata: points.map((point) => [point.std, point.n]),
      error_y: {
        type: "data",
        array: points.map((point) => point.std),
        color: "rgba(127,219,149,0.55)",
        thickness: 1.2,
        width: 3,
        visible: true,
      },
      type: "scatter",
      mode: "lines+markers",
      name: "Mean epoch time",
      line: { color: "#7fdb95", width: 3 },
      marker: { color: "#7fdb95", size: 7 },
      hovertemplate: "Epoch %{x}<br>Mean time %{y:.2f}s<br>Std. dev. %{customdata[0]:.2f}s<br>%{customdata[1]} contributing runs<extra></extra>",
    }] : [];
  }
  drawPlot(target, traces, {
    xaxis: { title: "Epoch", rangemode: "tozero", dtick: 1 },
    yaxis: { title: "Time (s)", rangemode: "tozero" },
    showlegend: mode === "individual",
  }, "No epoch timing data is available for this experiment.");
}

function renderRocPlot(roc, mode, summary) {
  const target = document.querySelector("#roc-chart");
  if (!window.Plotly) {
    renderChartError(target, "Plotly could not be loaded.");
    return;
  }

  let traces = [];
  if (mode === "individual") {
    const runs = sortedRuns(roc.runs);
    traces = runs.map((run, index) => ({
      x: (run.points ?? []).map((point) => point.fpr),
      y: (run.points ?? []).map((point) => point.tpr),
      type: "scatter",
      mode: "lines",
      name: `Seed ${run.random_seed} (AUC ${formatDecimal(run.auc)})`,
      line: { color: runColor(run.random_seed, index), width: 2 },
      hovertemplate: `Seed ${safeText(run.random_seed)}<br>FPR %{x:.4f}<br>TPR %{y:.4f}<extra></extra>`,
    })).filter((trace) => trace.x.length);
  } else {
    const points = Array.isArray(roc.aggregate) ? roc.aggregate : [];
    if (points.length) {
      const x = points.map((point) => point.fpr);
      const lower = points.map((point) => clamp(Number(point.mean_tpr) - Number(point.std_tpr), 0, 1));
      const upper = points.map((point) => clamp(Number(point.mean_tpr) + Number(point.std_tpr), 0, 1));
      traces = [
        { x, y: lower, type: "scatter", mode: "lines", line: { width: 0 }, hoverinfo: "skip", showlegend: false },
        { x, y: upper, type: "scatter", mode: "lines", line: { width: 0 }, fill: "tonexty", fillcolor: "rgba(127,219,149,0.2)", hoverinfo: "skip", name: "1 std. dev." },
        {
          x,
          y: points.map((point) => point.mean_tpr),
          customdata: points.map((point) => [point.std_tpr, point.n ?? roc.run_count]),
          type: "scatter",
          mode: "lines",
          name: "Mean ROC",
          line: { color: "#7fdb95", width: 3 },
          hovertemplate: "FPR %{x:.4f}<br>Mean TPR %{y:.4f}<br>Std. dev. %{customdata[0]:.4f}<br>%{customdata[1]} contributing runs<extra></extra>",
        },
      ];
    }
  }
  if (traces.length) {
    traces.push({
      x: [0, 1], y: [0, 1], type: "scatter", mode: "lines", name: "Random classifier",
      line: { color: "#7c848c", width: 1.4, dash: "dot" }, hoverinfo: "skip",
    });
  }
  const meanAuc = numeric(summary?.mean_auc);
  const stdAuc = numeric(summary?.std_auc);
  const annotations = traces.length && meanAuc !== null ? [{
    text: `Mean AUC = ${meanAuc.toFixed(4)}${stdAuc === null ? "" : ` ± ${stdAuc.toFixed(4)}`}`,
    xref: "paper", yref: "paper",
    x: 0.97, y: 0.03, xanchor: "right", yanchor: "bottom",
    showarrow: false,
    font: { color: "#aab3ac", size: 12 },
    bgcolor: "rgba(26,29,36,0.8)",
    bordercolor: "rgba(127,219,149,0.25)",
    borderwidth: 1,
    borderpad: 6,
  }] : [];
  drawPlot(target, traces, {
    xaxis: { title: "False positive rate", range: [0, 1], constrain: "domain" },
    yaxis: { title: "True positive rate", range: [0, 1], scaleanchor: "x", scaleratio: 1 },
    showlegend: true,
    annotations,
  }, "No ROC data is available for this experiment.");
}

function renderCircuitDiagram(path) {
  if (!path) {
    return '<div class="empty-state"><p>No circuit diagram is available for this experiment.</p></div>';
  }
  return `
    <div class="chart-card circuit-card">
      <img src="${safeText(path)}" alt="Variational circuit diagram" loading="lazy">
    </div>`;
}

function renderModelSummaryCard(runsValue) {
  const runs = sortedRuns(runsValue).filter((run) => run.aux_weights && Object.keys(run.aux_weights).length);
  if (!runs.length) {
    return '<div class="empty-state"><p>No trained model weights are available for this experiment.</p></div>';
  }
  return `
    <div class="chart-card">
      <div class="chart-toolbar">
        <strong>Final auxiliary weights &amp; Hamiltonian</strong>
        <div class="segmented" id="model-seed-selector" role="group" aria-label="Trained model seed">
          ${runs.map((run, index) => `
            <button type="button" data-seed="${safeText(run.random_seed)}" aria-pressed="${index === 0}">Seed ${safeText(run.random_seed)}</button>
          `).join("")}
        </div>
      </div>
      <div class="model-summary" id="model-summary-body"></div>
    </div>`;
}

function renderModelSummary(runs, seed) {
  const target = document.querySelector("#model-summary-body");
  if (!target) return;
  const run = runs.find((candidate) => String(candidate.random_seed) === String(seed));
  const aux = run?.aux_weights ?? {};
  const formula = formatHamiltonian(aux.hamiltonian_coeffs);
  target.innerHTML = `
    <div class="stat-grid model-summary__stats">
      ${statCard("Scale factor", formatDecimal2(aux.scale_factor))}
      ${statCard("Bias", formatDecimal2(aux.bias))}
    </div>
    <p class="hamiltonian-formula">${formula ? safeText(formula) : "Unavailable"}</p>`;
}

function renderScoreDistributionCard(scoreDistribution) {
  const runs = sortedRuns(scoreDistribution?.runs);
  if (!runs.length) {
    return '<div class="empty-state"><p>No score distribution is available for this experiment.</p></div>';
  }
  return `
    <div class="chart-card">
      <div class="chart-toolbar">
        <strong>Classifier score distribution</strong>
        <div class="segmented" id="score-seed-selector" role="group" aria-label="Classifier score distribution seed">
          ${runs.map((run, index) => `
            <button type="button" data-seed="${safeText(run.random_seed)}" aria-pressed="${index === 0}">Seed ${safeText(run.random_seed)}</button>
          `).join("")}
        </div>
      </div>
      <div class="chart" id="score-chart" role="img" aria-label="Classifier score distribution plot"></div>
    </div>`;
}

function renderScoreDistributionPlot(runs, seed, axisTitle) {
  const target = document.querySelector("#score-chart");
  if (!window.Plotly) {
    renderChartError(target, "Plotly could not be loaded.");
    return;
  }
  const run = runs.find((candidate) => String(candidate.random_seed) === String(seed));
  const centers = binCenters(run?.bin_edges);
  const width = Array.isArray(run?.bin_edges) && run.bin_edges.length > 1
    ? run.bin_edges[1] - run.bin_edges[0]
    : undefined;
  const traces = centers.length ? [
    {
      x: centers, y: run.background_density, type: "bar", name: "Background",
      marker: { color: "#e0916a" }, opacity: 0.6, width,
      hovertemplate: `${safeText(axisTitle)} %{x:.4f}<br>Background density %{y:.4f}<extra></extra>`,
    },
    {
      x: centers, y: run.signal_density, type: "bar", name: "Signal",
      marker: { color: "#7fb1e0" }, opacity: 0.6, width,
      hovertemplate: `${safeText(axisTitle)} %{x:.4f}<br>Signal density %{y:.4f}<extra></extra>`,
    },
  ] : [];
  drawPlot(target, traces, {
    xaxis: { title: axisTitle },
    yaxis: { title: "Density" },
    barmode: "overlay",
    showlegend: true,
  }, "No score distribution is available for this seed.");
}

function setupSeedToggle(containerId, render) {
  const container = document.getElementById(containerId);
  if (!container) return;
  const buttons = [...container.querySelectorAll("button")];
  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      buttons.forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
      render(button.dataset.seed);
    });
  });
}

function binCenters(edges) {
  if (!Array.isArray(edges) || edges.length < 2) return [];
  return edges.slice(0, -1).map((edge, index) => (edge + edges[index + 1]) / 2);
}

function drawPlot(target, traces, axes, emptyMessage) {
  if (!traces.length) {
    renderChartError(target, emptyMessage);
    return;
  }
  const layout = {
    ...axes,
    autosize: true,
    margin: { l: 65, r: 25, t: 30, b: 62 },
    paper_bgcolor: "#1a1d24",
    plot_bgcolor: "#1a1d24",
    font: { family: '"DM Sans", system-ui, sans-serif', color: "#aab3ac", size: 12 },
    hoverlabel: { bgcolor: "#262a32", bordercolor: "#262a32", font: { color: "#eef1ee" } },
    legend: { orientation: "h", yanchor: "bottom", y: 1.02, xanchor: "left", x: 0 },
  };
  for (const axisName of ["xaxis", "yaxis"]) {
    layout[axisName] = {
      gridcolor: "rgba(127,219,149,0.12)",
      zerolinecolor: "rgba(127,219,149,0.22)",
      fixedrange: false,
      ...layout[axisName],
    };
  }
  window.Plotly.react(target, traces, layout, PLOT_CONFIG);
}

function renderChartError(target, message) {
  if (window.Plotly && target.data) {
    window.Plotly.purge(target);
  }
  target.innerHTML = `<div class="empty-state"><p>${safeText(message)}</p></div>`;
}

function renderRuns(runsValue) {
  const runs = sortedRuns(runsValue);
  if (!runs.length) {
    return '<div class="empty-state"><p>No run records are available.</p></div>';
  }
  return `
    <div class="table-card"><div class="table-scroll">
      <table>
        <thead><tr><th>Random seed</th><th>Status</th><th>Completed epochs</th><th>Stop reason</th><th>Final validation AUC</th><th>Test AUC</th><th>Evaluated jets</th><th>W&amp;B</th><th>Notices</th></tr></thead>
        <tbody>${runs.map((run) => `
          <tr>
            <td>${safeText(displayValue(run.random_seed))}</td>
            <td>${safeText(displayValue(run.status))}</td>
            <td>${safeText(displayValue(run.completed_epochs))}</td>
            <td>${safeText(displayValue(run.stop_reason))}</td>
            <td>${formatDecimal(run.final_validation_auc)}</td>
            <td>${formatDecimal(run.test_auc)}</td>
            <td>${formatInteger(run.evaluation_jets)}</td>
            <td>${safeText(displayValue(run.wandb_run_id))}</td>
            <td class="muted">${safeText(formatRunNotices(run.notices))}</td>
          </tr>`).join("")}</tbody>
      </table>
    </div></div>`;
}

function renderConfig(configValue) {
  const config = configValue && typeof configValue === "object" ? configValue : {};
  const groups = ["data", "model", "optimization", "execution"].filter((key) => config[key] && typeof config[key] === "object");
  if (!groups.length) {
    return '<div class="empty-state"><p>No saved configuration is available.</p></div>';
  }
  return `<div class="config-grid">${groups.map((group) => `
    <article class="config-card">
      <h3>${safeText(titleCase(group))}</h3>
      <dl class="config-list">
        ${Object.entries(config[group]).map(([key, value]) => `
          <div class="config-row"><dt>${safeText(titleCase(key))}</dt><dd>${safeText(configValueText(value))}</dd></div>`).join("")}
      </dl>
    </article>`).join("")}</div>`;
}

function renderConfigVariations(value) {
  if (!value || (Array.isArray(value) && !value.length)) return "";
  const entries = Array.isArray(value) ? value : Object.entries(value).map(([field, detail]) => ({ field, detail }));
  return `
    <aside class="notices" aria-label="Configuration variations">
      ${entries.map((entry) => {
        if (typeof entry === "string") return `<div class="notice notice--warning">${safeText(entry)}</div>`;
        const field = entry.field ?? entry.key ?? "Configuration";
        const detail = entry.detail ?? entry.values ?? entry.message ?? entry;
        return `<div class="notice notice--warning"><strong>${safeText(titleCase(field))}:</strong>&nbsp;${safeText(configValueText(detail))}</div>`;
      }).join("")}
    </aside>`;
}

function renderNotices(noticesValue) {
  const notices = Array.isArray(noticesValue) ? noticesValue : [];
  if (!notices.length) return "";
  return `<aside class="notices" aria-label="Experiment notices">${notices.map((notice) => {
    const level = ["warning", "error"].includes(notice.level) ? notice.level : "info";
    return `<div class="notice notice--${level}">${safeText(notice.message)}</div>`;
  }).join("")}</aside>`;
}

function renderError(title, detail) {
  app.innerHTML = `
    <section class="error-state" role="alert">
      <p class="eyebrow">Data error</p>
      <h1>${safeText(title)}</h1>
      <p>${safeText(detail)}</p>
      <a class="button-link" href="./">Return to all experiments</a>
    </section>`;
}

function statusInfo(rawStatus, successful, total) {
  const normalized = String(rawStatus ?? "").toLowerCase();
  if (["error", "failed"].includes(normalized) || (total !== null && successful === 0)) {
    return { label: rawStatus || "Failed", className: "status--error" };
  }
  if (["partial", "warning", "incomplete"].includes(normalized) || (successful !== null && total !== null && successful < total)) {
    return { label: rawStatus || "Partial", className: "status--warning" };
  }
  return { label: rawStatus || "Complete", className: "" };
}

function sortedRuns(value) {
  if (!Array.isArray(value)) return [];
  return [...value].sort((a, b) => Number(a.random_seed) - Number(b.random_seed));
}

function runColor(seed, fallbackIndex) {
  const numericSeed = Number(seed);
  const index = Number.isFinite(numericSeed) ? Math.abs(numericSeed) % COLORS.length : fallbackIndex % COLORS.length;
  return COLORS[index];
}

function numeric(value) {
  if (value === null || value === undefined || value === "") return null;
  const converted = Number(value);
  return Number.isFinite(converted) ? converted : null;
}

function formatDecimal(value) {
  const converted = numeric(value);
  return converted === null ? "Unavailable" : converted.toFixed(4);
}

function formatDecimal2(value) {
  const converted = numeric(value);
  return converted === null ? "Unavailable" : converted.toFixed(2);
}

function formatHamiltonian(coeffs) {
  if (!Array.isArray(coeffs) || !coeffs.length) return null;
  const subscriptDigits = "₀₁₂₃₄₅₆₇₈₉";
  const subscript = (n) => String(n).split("").map((digit) => subscriptDigits[Number(digit)]).join("");
  const terms = coeffs.map((raw, index) => {
    const rounded = Math.round((numeric(raw) ?? 0) * 100) / 100;
    return { magnitude: Math.abs(rounded).toFixed(2), negative: rounded < 0, index };
  });
  const body = terms.map((term, position) => {
    const symbol = `${term.magnitude} Ẑ${subscript(term.index)}`;
    if (position === 0) return term.negative ? `− ${symbol}` : symbol;
    return term.negative ? ` − ${symbol}` : ` + ${symbol}`;
  }).join("");
  return `Ĥ = ${body}`;
}

function aucWithError(mean, deviation) {
  const meanValue = numeric(mean);
  const deviationValue = numeric(deviation);
  if (meanValue === null) return "Unavailable";
  return `${meanValue.toFixed(4)}${deviationValue === null ? "" : ` <span class="muted">&plusmn; ${deviationValue.toFixed(4)}</span>`}`;
}

function secondsWithError(mean, deviation) {
  const meanValue = numeric(mean);
  const deviationValue = numeric(deviation);
  if (meanValue === null) return null;
  return `${meanValue.toFixed(2)} s${deviationValue === null ? "" : ` <span class="muted">&plusmn; ${deviationValue.toFixed(2)} s</span>`}`;
}

function formatInteger(value) {
  const converted = numeric(value);
  return converted === null ? "Unavailable" : Math.round(converted).toLocaleString("en-US");
}

function formatDateTime(value) {
  if (!value) return "Unavailable";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return safeText(value);
  return new Intl.DateTimeFormat("en-GB", {
    day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit", timeZoneName: "short",
  }).format(date);
}

function formatRunNotices(value) {
  if (!Array.isArray(value) || !value.length) return "None";
  return value.map((notice) => (
    typeof notice === "string" ? notice : notice?.message ?? "Unknown notice"
  )).join("; ");
}

function displayValue(value) {
  return value === null || value === undefined || value === "" ? "Unavailable" : value;
}

function configValueText(value) {
  if (value === null || value === undefined || value === "") return "Unavailable";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function titleCase(value) {
  return String(value).replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function clamp(value, minimum, maximum) {
  return Math.min(Math.max(Number.isFinite(value) ? value : minimum, minimum), maximum);
}

function plural(count, noun) {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

function safeText(value) {
  return String(value ?? "Unavailable")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
