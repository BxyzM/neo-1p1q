/*
 * Render the static Neo1P1Q experiment dashboard.
 *
 * Author: Aritra Bal (ETP)
 * Date: 2026-09-10
 *
 * Data contract:
 *   data/index.json lists experiment summaries under `experiments`; each item
 *   has id, status, run_count, successful_runs, mean_auc, std_auc, total_jets,
 *   loss, qubits, layers, circuit_type, operations_per_qubit, and train_n. The
 *   `?view=sweep` page groups these by every field except one swept parameter
 *   (see SWEEP_PARAMS) to plot mean AUC vs that parameter.
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
const view = params.get("view");
const COLORS = [
  "#7fdb95", "#e0916a", "#7fb1e0", "#c295cc", "#e0c15c",
  "#5cc9c2", "#e0a394", "#a8c96f", "#b3a3e0", "#e08aa0",
];

// Sweep parameters a user can hold as the single varying axis. `field` is the
// index.json key; the other two, plus `operations_per_qubit`, are held fixed
// to identify a comparable set of experiments (circuit_type is never fixed --
// it is the color/series dimension instead, for a future second circuit type).
const SWEEP_PARAMS = [
  { key: "layers", label: "Layers", field: "layers", axisTitle: "Number of layers" },
  { key: "qubits", label: "Qubits", field: "qubits", axisTitle: "Number of qubits" },
  { key: "train_n", label: "Training jets", field: "train_n", axisTitle: "Training jets (signal + background)" },
];
const FIXED_FIELDS = ["operations_per_qubit"];

const PLOT_CONFIG = {
  responsive: true,
  displaylogo: false,
  modeBarButtonsToRemove: ["lasso2d", "select2d"],
  toImageButtonOptions: { format: "png", scale: 2 },
};

document.addEventListener("DOMContentLoaded", () => {
  if (experimentId && !/^[\w-]+$/.test(experimentId)) {
    renderError("Invalid experiment", "The experiment identifier must contain only letters, digits, underscores, or hyphens.");
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
    } else if (view === "sweep") {
      renderSweepPage(data);
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
  // Numeric ids (e.g. "001") sort newest-first by value; non-numeric ids
  // (e.g. "condor_001") sort after all numeric ones, alphabetically.
  const sortKey = (id) => (/^\d+$/.test(id) ? [0, -Number(id), id] : [1, 0, id]);
  experiments.sort((a, b) => {
    const [ga, na, sa] = sortKey(a.id);
    const [gb, nb, sb] = sortKey(b.id);
    return ga - gb || na - nb || sa.localeCompare(sb);
  });
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
      <a class="button-link" href="?view=sweep">Parameter sweep &#8594;</a>
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
        <span class="experiment-card__number" style="--id-length:${id.length}">${id}</span>
        <span class="status ${status.className}">${safeText(status.label)}</span>
      </div>
      <div class="experiment-card__metrics">
        <span><span class="metric-label">Test AUC</span><span class="metric-value">${aucWithError(experiment.mean_auc, experiment.std_auc)}</span></span>
        <span><span class="metric-label">Runs</span><span class="metric-value">${successful === null || runs === null ? "Unavailable" : `${successful} / ${runs}`}</span></span>
        <span><span class="metric-label">Evaluated jets</span><span class="metric-value">${formatInteger(experiment.total_jets)}</span></span>
        <span><span class="metric-label">Loss</span><span class="metric-value">${safeText(displayValue(experiment.loss))}</span></span>
        <span><span class="metric-label">Qubits</span><span class="metric-value">${safeText(displayValue(experiment.qubits))}</span></span>
        <span><span class="metric-label">Layers</span><span class="metric-value">${safeText(displayValue(experiment.layers))}</span></span>
      </div>
      <span class="card-arrow" aria-hidden="true">&#8594;</span>
    </a>`;
}

function renderSweepPage(data) {
  const experiments = Array.isArray(data.experiments) ? data.experiments : [];
  document.title = "Neo1P1Q | Parameter sweep";
  const requested = params.get("param");
  const initialParam = SWEEP_PARAMS.some((entry) => entry.key === requested) ? requested : SWEEP_PARAMS[0].key;
  const requestedCircuits = params.get("circuits");
  // null means "no explicit filter yet" -- resolves to "every available circuit
  // type" the first time renderCircuitChips sees a model.
  let selectedTypes = requestedCircuits ? new Set(requestedCircuits.split(",").filter(Boolean)) : null;
  let currentModel = null;

  app.innerHTML = `
    <a class="back-link" href="./"><span aria-hidden="true">&#8592;</span> All experiments</a>
    <header>
      <p class="eyebrow">Quantum classifier study</p>
      <h1>Parameter sweep</h1>
      <p class="lede">Mean test AUC across experiments that vary exactly one architecture parameter, with every other reported setting held fixed.</p>
    </header>
    <section class="report-section" aria-labelledby="sweep-title">
      <div class="chart-toolbar">
        <strong id="sweep-title">Swept parameter</strong>
        <div class="segmented" id="sweep-param-selector" role="group" aria-label="Sweep parameter">
          ${SWEEP_PARAMS.map((entry) => `
            <button type="button" data-param="${entry.key}" aria-pressed="${entry.key === initialParam}">${safeText(entry.label)}</button>
          `).join("")}
        </div>
      </div>
      <div class="chart-toolbar" id="sweep-circuit-toolbar" hidden>
        <strong>Circuit type</strong>
        <div class="segmented" id="sweep-circuit-selector" role="group" aria-label="Circuit type"></div>
      </div>
      <div class="chart" id="sweep-chart" role="img" aria-label="Parameter sweep plot"></div>
      <div id="sweep-summary"></div>
    </section>`;

  // Only affects display: filters which of buildSweepModel's already-computed
  // series get drawn/listed. Never changes which experiments form "the"
  // comparable group -- that grouping happens once, inside buildSweepModel.
  function applyCircuitFilter(paramInfo) {
    const chartTarget = document.querySelector("#sweep-chart");
    const summaryTarget = document.querySelector("#sweep-summary");
    const filtered = {
      ...currentModel,
      series: selectedTypes
        ? currentModel.series.filter((series) => selectedTypes.has(series.circuitType))
        : currentModel.series,
    };
    if (!currentModel.empty && currentModel.series.length && !filtered.series.length) {
      renderChartError(chartTarget, "No circuit type selected -- choose at least one above.");
      summaryTarget.innerHTML = "";
      return;
    }
    renderSweepChart(filtered, paramInfo);
    renderSweepSummaryPanel(filtered, paramInfo);
  }

  function updateUrl(paramKey) {
    const url = new URL(window.location.href);
    url.searchParams.set("view", "sweep");
    url.searchParams.set("param", paramKey);
    const availableTypes = currentModel.series.map((series) => series.circuitType);
    const isFiltered = selectedTypes && selectedTypes.size < availableTypes.length;
    if (isFiltered) {
      url.searchParams.set("circuits", [...selectedTypes].join(","));
    } else {
      url.searchParams.delete("circuits");
    }
    history.replaceState(null, "", url);
  }

  // Rebuilds the chip row for the circuit types present in the current
  // swept-param's comparable group (this can differ per param), reconciling
  // any previously selected types against the new set.
  function renderCircuitChips(paramInfo) {
    const toolbar = document.querySelector("#sweep-circuit-toolbar");
    const selector = document.querySelector("#sweep-circuit-selector");
    const availableTypes = currentModel.series.map((series) => series.circuitType);
    if (availableTypes.length <= 1) {
      toolbar.hidden = true;
      selectedTypes = null;
      return;
    }
    toolbar.hidden = false;
    const overlap = selectedTypes ? availableTypes.filter((type) => selectedTypes.has(type)) : [];
    selectedTypes = new Set(overlap.length ? overlap : availableTypes);

    selector.innerHTML = `
      <button type="button" data-circuit="__all__" aria-pressed="${selectedTypes.size === availableTypes.length}">All</button>
      ${availableTypes.map((type) => `
        <button type="button" data-circuit="${safeText(type)}" aria-pressed="${selectedTypes.has(type)}">${safeText(titleCase(type))}</button>
      `).join("")}`;

    const chips = [...selector.querySelectorAll("button")];
    chips.forEach((chip) => {
      chip.addEventListener("click", () => {
        if (chip.dataset.circuit === "__all__") {
          selectedTypes = new Set(availableTypes);
        } else if (selectedTypes.has(chip.dataset.circuit)) {
          selectedTypes.delete(chip.dataset.circuit);
        } else {
          selectedTypes.add(chip.dataset.circuit);
        }
        chips.forEach((item) => item.setAttribute(
          "aria-pressed",
          item.dataset.circuit === "__all__"
            ? String(selectedTypes.size === availableTypes.length)
            : String(selectedTypes.has(item.dataset.circuit)),
        ));
        updateUrl(paramInfo.key);
        applyCircuitFilter(paramInfo);
      });
    });
  }

  function switchTo(paramKey) {
    const paramInfo = SWEEP_PARAMS.find((entry) => entry.key === paramKey) ?? SWEEP_PARAMS[0];
    currentModel = buildSweepModel(experiments, paramInfo.field);
    renderCircuitChips(paramInfo);
    updateUrl(paramInfo.key);
    applyCircuitFilter(paramInfo);
  }

  switchTo(initialParam);
  const buttons = [...document.querySelectorAll("#sweep-param-selector button")];
  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      buttons.forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
      switchTo(button.dataset.param);
    });
  });
}

// Groups experiments by every reported architecture field except the swept
// one (plus the always-fixed fields), picks the largest such group as "the"
// comparable sweep, and classifies every value the swept field takes on
// anywhere in the dataset as present / failed / missing within that group.
function buildSweepModel(experiments, sweptField) {
  const contextFields = SWEEP_PARAMS.map((entry) => entry.field).filter((field) => field !== sweptField).concat(FIXED_FIELDS);
  const usable = experiments.filter((experiment) => (
    numeric(experiment[sweptField]) !== null && contextFields.every((field) => numeric(experiment[field]) !== null)
  ));
  const excluded = experiments.filter((experiment) => !usable.includes(experiment));

  if (!usable.length) {
    return { empty: true, contextFields, excluded, candidateValues: [], series: [], context: null };
  }

  const contextKey = (experiment) => contextFields.map((field) => experiment[field]).join("|");
  const groups = new Map();
  for (const experiment of usable) {
    const key = contextKey(experiment);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(experiment);
  }

  let activeKey = null;
  let activeGroup = [];
  for (const [key, members] of groups) {
    const distinct = new Set(members.map((experiment) => experiment[sweptField])).size;
    const bestDistinct = new Set(activeGroup.map((experiment) => experiment[sweptField])).size;
    if (
      distinct > bestDistinct ||
      (distinct === bestDistinct && members.length > activeGroup.length) ||
      (distinct === bestDistinct && members.length === activeGroup.length && (activeKey === null || key < activeKey))
    ) {
      activeKey = key;
      activeGroup = members;
    }
  }

  const candidateValues = [...new Set(usable.map((experiment) => Number(experiment[sweptField])))].sort((a, b) => a - b);
  const circuitTypes = [...new Set(activeGroup.map((experiment) => experiment.circuit_type ?? "normal"))].sort();
  const context = Object.fromEntries(contextFields.map((field) => [field, activeGroup[0]?.[field] ?? null]));

  const series = circuitTypes.map((circuitType, index) => {
    const members = activeGroup.filter((experiment) => (experiment.circuit_type ?? "normal") === circuitType);
    const byValue = new Map();
    for (const experiment of members) {
      const value = Number(experiment[sweptField]);
      if (!byValue.has(value)) byValue.set(value, []);
      byValue.get(value).push(experiment);
    }
    const points = candidateValues.map((value) => {
      const matches = byValue.get(value) ?? [];
      if (!matches.length) return { value, state: "missing" };
      const chosen = [...matches].sort((a, b) => (numeric(b.successful_runs) ?? -1) - (numeric(a.successful_runs) ?? -1))[0];
      const auc = numeric(chosen.mean_auc);
      return {
        value,
        state: auc === null ? "failed" : "ok",
        experiment: chosen,
        duplicates: matches.length > 1 ? matches.map((m) => m.id) : null,
      };
    });
    return { circuitType, color: COLORS[index % COLORS.length], points };
  });

  return { empty: false, contextFields, excluded, candidateValues, series, context, paramField: sweptField };
}

function renderSweepChart(model, paramInfo) {
  const target = document.querySelector("#sweep-chart");
  if (!window.Plotly) {
    renderChartError(target, "Plotly could not be loaded.");
    return;
  }
  if (model.empty || !model.candidateValues.length) {
    renderChartError(target, "No experiments have enough reported configuration data to build this sweep.");
    return;
  }

  const traces = model.series.map((series) => {
    const okPoints = series.points.filter((point) => point.state === "ok");
    return {
      x: series.points.map((point) => point.value),
      y: series.points.map((point) => (point.state === "ok" ? point.experiment.mean_auc : null)),
      error_y: {
        type: "data",
        array: series.points.map((point) => (point.state === "ok" ? (numeric(point.experiment.std_auc) ?? 0) : 0)),
        color: `${series.color}88`,
        thickness: 1.2,
        width: 4,
        visible: true,
      },
      type: "scatter",
      mode: "lines+markers",
      name: model.series.length > 1 ? series.circuitType : "Mean test AUC",
      connectgaps: false,
      line: { color: series.color, width: 2 },
      marker: {
        color: series.color,
        size: 9,
        symbol: series.points.map((point) => (
          point.state === "ok" && numeric(point.experiment.successful_runs) < numeric(point.experiment.run_count)
            ? "circle-open"
            : "circle"
        )),
      },
      customdata: series.points.map((point) => [
        point.state === "ok" ? point.experiment.id : "",
        point.state === "ok" ? `${point.experiment.successful_runs}/${point.experiment.run_count}` : "",
      ]),
      hovertemplate: `${safeText(paramInfo.axisTitle)} %{x}<br>Mean AUC %{y:.4f}<br>Experiment %{customdata[0]}<br>Runs %{customdata[1]}<extra></extra>`,
      hoverinfo: okPoints.length ? "all" : "skip",
    };
  });

  drawPlot(target, traces, {
    xaxis: { title: paramInfo.axisTitle, tickvals: model.candidateValues, ticktext: model.candidateValues.map((value) => value.toLocaleString("en-US")) },
    yaxis: { title: "Mean test AUC" },
    showlegend: model.series.length > 1,
  }, "No experiments have enough reported configuration data to build this sweep.");
}

function renderSweepSummaryPanel(model, paramInfo) {
  const target = document.querySelector("#sweep-summary");
  if (!target) return;

  if (model.empty) {
    target.innerHTML = `<div class="empty-state"><p>No experiments report a complete, comparable configuration (${safeText(model.contextFields.map(titleCase).join(", "))}) to build a ${safeText(paramInfo.label.toLowerCase())} sweep.</p></div>`;
    return;
  }

  const contextLine = model.contextFields
    .map((field) => `${titleCase(field)} = ${safeText(displayValue(model.context[field]))}`)
    .join(", ");

  const notes = [];
  if (model.candidateValues.length < 2) {
    notes.push(`<li>Only one ${safeText(paramInfo.label.toLowerCase())} value (${safeText(model.candidateValues[0])}) has been run under this configuration &mdash; not enough data yet for an actual sweep.</li>`);
  }
  for (const series of model.series) {
    for (const point of series.points) {
      if (point.state === "missing") {
        notes.push(`<li><strong>${safeText(paramInfo.label)} = ${point.value}</strong>${model.series.length > 1 ? ` (${safeText(series.circuitType)})` : ""} &mdash; no experiment found with ${safeText(contextLine)}.</li>`);
      } else if (point.state === "failed") {
        notes.push(`<li><strong>${safeText(paramInfo.label)} = ${point.value}</strong>${model.series.length > 1 ? ` (${safeText(series.circuitType)})` : ""} &mdash; experiment <code>${safeText(point.experiment.id)}</code> ran but has no usable test AUC (${safeText(displayValue(point.experiment.status))}, ${safeText(point.experiment.successful_runs)}/${safeText(point.experiment.run_count)} runs succeeded).</li>`);
      } else if (point.duplicates) {
        notes.push(`<li><strong>${safeText(paramInfo.label)} = ${point.value}</strong>${model.series.length > 1 ? ` (${safeText(series.circuitType)})` : ""} &mdash; multiple experiments match (${point.duplicates.map(safeText).join(", ")}); showing <code>${safeText(point.experiment.id)}</code>, the one with the most successful runs.</li>`);
      }
    }
  }
  if (model.excluded.length) {
    notes.push(`<li>${plural(model.excluded.length, "experiment")} excluded from every sweep for missing configuration fields: ${model.excluded.map((experiment) => safeText(experiment.id)).join(", ")}.</li>`);
  }

  const totalPoints = model.series[0]?.points.length ?? 0;
  const okCount = model.series.reduce((sum, series) => sum + series.points.filter((point) => point.state === "ok").length, 0);
  const headline = totalPoints
    ? `${okCount} of ${totalPoints * model.series.length} possible ${safeText(paramInfo.label.toLowerCase())} value${totalPoints === 1 ? "" : "s"} present, holding ${safeText(contextLine)} fixed.`
    : "";

  target.innerHTML = `
    <p class="report-meta">${headline}</p>
    ${notes.length ? `<aside class="notices" aria-label="Missing sweep runs"><div class="notice notice--warning"><ul class="sweep-notes">${notes.join("")}</ul></div></aside>` : ""}`;
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
