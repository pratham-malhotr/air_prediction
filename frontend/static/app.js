"use strict";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const state = { context: null, result: null, mode: "manual", hours: 72, ready: false, busy: false, stale: false };
const fields = { "pm2.5": "pm25", TEMP: "temperature", DEWP: "dewpoint", PRES: "pressure", cbwd: "wind", Iws: "wind-speed", Ir: "rain", Is: "snow" };
const modelLabels = { xgboost: "XGBoost", ridge: "Ridge regression", persistence: "Persistence", seasonal_naive: "Daily seasonal naive" };
const hour = 3600000;
const dateFormat = (value, options) => new Intl.DateTimeFormat("en-GB", { timeZone: "Asia/Shanghai", ...options }).format(new Date(value));
const dayTime = (value) => dateFormat(value, { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", hourCycle: "h23" });
const fullTime = (value) => dateFormat(value, { day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit", hourCycle: "h23" });
const timeOnly = (value) => dateFormat(value, { hour: "2-digit", minute: "2-digit", hourCycle: "h23" });
const fmt = (value, digits = 1) => value === null || value === undefined || !Number.isFinite(Number(value)) ? "—" : Number(value).toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
const setText = (selector, value) => { $(selector).textContent = value; };

async function api(path, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 60000);
  try {
    const response = await fetch(path, { ...options, signal: controller.signal });
    let body;
    try { body = await response.json(); } catch { throw { message: response.status >= 500 ? "The forecasting service is temporarily unavailable. Please try again in a moment." : "The request could not be read. Please try again." }; }
    if (!response.ok) throw body.error || { message: "The request could not be completed." };
    return body;
  } catch (error) {
    if (error.name === "AbortError") throw { message: "The model took too long to respond. Please try again." };
    if (error instanceof TypeError) throw { message: "Could not reach the forecasting service. Check your connection and retry." };
    throw error;
  } finally { clearTimeout(timeout); }
}

function clearErrors() {
  $("#global-error").hidden = true;
  $$(".field.invalid").forEach((element) => element.classList.remove("invalid"));
  $$(".field-error").forEach((element) => { element.textContent = ""; });
  Object.values(fields).forEach((id) => $("#" + id).removeAttribute("aria-invalid"));
}

function showError(error) {
  setText("#global-error", error.message || "The forecast could not be completed. Please try again.");
  $("#global-error").hidden = false;
  for (const [key, message] of Object.entries(error.fields || {})) {
    if (!fields[key]) continue;
    const input = $("#" + fields[key]);
    input.closest(".field").classList.add("invalid");
    input.setAttribute("aria-invalid", "true");
    input.closest(".field").querySelector(".field-error").textContent = message;
    const details = input.closest("details");
    if (details) details.open = true;
  }
  const first = $(".field.invalid input, .field.invalid select");
  if (first) first.focus();
}

function setBusy(value, label = "Generating forecast") {
  state.busy = value;
  $$("#forecast-form input, #forecast-form select, #forecast-form button").forEach((element) => { element.disabled = value || !state.ready; });
  $("#reset-sample").disabled = value;
  $("#change-history").disabled = value;
  $$(".input-tab").forEach((button) => { button.disabled = value || !state.ready; });
  $("#forecast-button").classList.toggle("loading", value);
  $("#results-column").classList.toggle("busy", value);
  $("#results-column").setAttribute("aria-busy", String(value));
  setText(".button-label", value ? label : state.mode === "history" ? "Forecast from this history" : "Generate forecast");
  $("#download-forecast").disabled = value || state.stale || !state.result;
}

function markStale() {
  if (!state.result) return;
  state.stale = true;
  $("#results-column").classList.add("stale");
  setText("#result-context", "Inputs changed · generate a new forecast");
  $("#download-forecast").disabled = true;
}

function applyMode(mode, dirty = true) {
  state.mode = mode;
  $$(".input-tab").forEach((button) => {
    const active = button.dataset.mode === mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
  });
  $("#panel-manual").hidden = mode !== "manual";
  $("#panel-history").hidden = mode !== "history";
  setText(".button-label", mode === "history" ? "Forecast from this history" : "Generate forecast");
  if (state.context) renderContext();
  clearErrors();
  if (dirty) markStale();
}

function renderContext() {
  const context = state.context;
  setText("#observation-time", fullTime(context.next_observation_time));
  setText("#history-label", context.label + (context.kind === "sample" ? " history" : ""));
  setText("#history-detail", `${context.history_hours} hourly slots · ending ${fullTime(context.end)} · ${context.kind === "sample" ? "historical example" : "your readings"}`);
  const footnote = state.mode === "manual" ? (context.kind === "sample" ? "Example inputs repeat the last Beijing measurements. Edit them to explore a historical scenario." : "Your readings are added one hour after this uploaded history.") : "The forecast starts at the last hour in the selected history. Missing measurements stay unknown.";
  setText("#form-footnote span", footnote);
}

function setContext(context) {
  state.context = context;
  for (const [key, id] of Object.entries(fields)) {
    const element = $("#" + id);
    const value = context.defaults[key];
    if (key === "cbwd" && value && ![...element.options].some((option) => option.value === value)) {
      const option = document.createElement("option"); option.value = value; option.textContent = value; element.append(option);
    }
    element.value = value === null || value === undefined ? "" : String(value);
  }
  renderContext();
  if (context.kind === "upload") {
    $("#upload-status").hidden = false;
    setText("#upload-name", context.label);
    setText("#upload-detail", `${context.rows.toLocaleString()} rows checked · ${context.history_hours} recent hours loaded`);
  } else {
    $("#upload-status").hidden = true;
  }
}

function readInputs() {
  const output = {}, errors = {};
  for (const [key, id] of Object.entries(fields)) {
    const input = $("#" + id);
    if (key === "cbwd") { output[key] = input.value || null; continue; }
    if (input.validity.badInput) { errors[key] = "Enter a valid number."; continue; }
    if (!input.value.trim()) {
      output[key] = null;
      if (key === "pm2.5") errors[key] = "Enter the latest PM2.5 concentration.";
      continue;
    }
    const value = Number(input.value);
    if (!Number.isFinite(value)) errors[key] = "Enter a finite number.";
    else if (["pm2.5", "Iws", "Is", "Ir"].includes(key) && value < 0) errors[key] = "This measurement cannot be negative.";
    else if (key === "PRES" && value <= 0) errors[key] = "Pressure must be greater than zero.";
    output[key] = value;
  }
  if (Object.keys(errors).length) throw { message: "Please check the highlighted measurements.", fields: errors };
  return output;
}

async function generateForecast() {
  if (state.busy || !state.ready) return;
  clearErrors();
  try {
    const inputs = state.mode === "manual" ? readInputs() : undefined;
    setBusy(true);
    const result = await api("/api/forecast", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ history_id: state.context.id, history: state.context.history, mode: state.mode, inputs, model: $("#model").value }) });
    renderForecast(result);
    return true;
  } catch (error) { showError(error); return false; }
  finally { setBusy(false); }
}

function renderForecast(result) {
  state.result = result; state.stale = false;
  $("#results-column").classList.remove("stale");
  setText("#forecast-value", fmt(result.forecast));
  setText("#result-model", result.selected_model === "ridge" ? "Ridge" : "XGBoost");
  const change = result.change;
  setText("#change-badge", change === null ? "Latest reading unavailable" : `${change > 0 ? "↗ +" : change < 0 ? "↘ " : "→ "}${fmt(change)} µg/m³ vs latest`);
  setText("#forecast-time", `For ${fullTime(result.target_time)}`);
  setText("#result-context", `Based on ${timeOnly(result.origin_time)} readings`);
  setText("#scenario-tag", result.historical ? "Historical scenario" : "Your uploaded history");
  $("#latest-value").replaceChildren(document.createTextNode(fmt(result.latest)), Object.assign(document.createElement("small"), { textContent: " µg/m³" }));
  const other = result.selected_model === "xgboost" ? "ridge" : "xgboost";
  setText("#comparison-text", `${modelLabels[other]} predicts ${fmt(result.predictions[other])} µg/m³ for the same next hour.`);
  setText("#history-quality", `${result.data_quality.observed} of ${result.data_quality.total} recent PM2.5 readings available`);
  setText("#reference-mae", fmt(result.backtest.mae, 2));
  setText("#reference-rmse", fmt(result.backtest.rmse, 2));
  setText("#reference-count", `${Number(result.backtest.n).toLocaleString()} observed forecast hours.`);
  const warnings = $("#forecast-warnings"); warnings.replaceChildren(); warnings.hidden = !result.warnings.length;
  for (const warning of result.warnings) { const p = document.createElement("p"); p.textContent = warning; warnings.append(p); }
  $("#download-forecast").disabled = false;
  drawChart();
}

function svgElement(name, attributes = {}, text) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
  if (text !== undefined) element.textContent = text;
  return element;
}

function drawChart() {
  if (!state.result) return;
  const result = state.result, svg = $("#history-chart"); svg.replaceChildren();
  $("#chart-empty").hidden = true; $("#chart-tooltip").hidden = true;
  const origin = new Date(result.origin_time).getTime();
  const points = result.series.filter((point) => new Date(point.time).getTime() >= origin - (state.hours - 1) * hour);
  const forecast = { time: result.target_time, value: result.forecast, forecast: true };
  const allPoints = [...points, forecast];
  const left = 39, right = 16, top = 13, bottom = 35, width = 640, height = 220;
  const start = new Date(points[0].time).getTime(), end = new Date(result.target_time).getTime();
  const maximum = Math.max(10, ...allPoints.filter((point) => point.value !== null).map((point) => point.value));
  const rawStep = maximum / 4, scale = 10 ** Math.floor(Math.log10(rawStep));
  const step = [1, 2, 2.5, 5, 10].map((value) => value * scale).find((value) => value >= rawStep);
  const yMax = step * 4;
  const x = (time) => left + (new Date(time).getTime() - start) / (end - start || 1) * (width - left - right);
  const y = (value) => height - bottom - value / yMax * (height - top - bottom);
  svg.setAttribute("aria-label", `${points.length} hourly history slots and a next-hour forecast of ${fmt(result.forecast)} micrograms per cubic metre, at ${dayTime(result.target_time)} Beijing time.`);
  const defs = svgElement("defs"), gradient = svgElement("linearGradient", { id: "history-fill", x1: "0", y1: "0", x2: "0", y2: "1" });
  gradient.append(svgElement("stop", { offset: "0%", "stop-color": "#9db68c", "stop-opacity": ".18" }), svgElement("stop", { offset: "100%", "stop-color": "#9db68c", "stop-opacity": ".015" }));
  defs.append(gradient); svg.append(defs);
  for (let index = 0; index <= 4; index++) {
    const value = step * index, yPos = y(value);
    svg.append(svgElement("line", { x1: left, x2: width - right, y1: yPos, y2: yPos, stroke: "#edf0e8", "stroke-dasharray": "3 5" }));
    svg.append(svgElement("text", { x: left - 10, y: yPos + 3, "text-anchor": "end", fill: "#9eaa92", "font-size": 9 }, fmt(value, value % 1 ? 1 : 0)));
  }
  const segments = []; let segment = [];
  for (const point of points) {
    if (point.value === null || point.entered) { if (segment.length) segments.push(segment); segment = []; }
    else segment.push(point);
  }
  if (segment.length) segments.push(segment);
  for (const part of segments) {
    const path = part.map((point, index) => `${index ? "L" : "M"}${x(point.time).toFixed(2)},${y(point.value).toFixed(2)}`).join(" ");
    svg.append(svgElement("path", { d: `${path} L${x(part.at(-1).time)},${y(0)} L${x(part[0].time)},${y(0)} Z`, fill: "url(#history-fill)" }));
    svg.append(svgElement("path", { d: path, fill: "none", stroke: "#9dac91", "stroke-width": 1.8, "stroke-linejoin": "round", "stroke-linecap": "round" }));
    if (part.length === 1) svg.append(svgElement("circle", { cx: x(part[0].time), cy: y(part[0].value), r: 2, fill: "#9dac91" }));
  }
  const latest = points.at(-1);
  if (latest.entered && latest.value !== null) {
    const before = points.at(-2);
    if (before && before.value !== null) svg.append(svgElement("line", { x1: x(before.time), y1: y(before.value), x2: x(latest.time), y2: y(latest.value), stroke: "#8da17d", "stroke-width": 1.6, "stroke-dasharray": "3 3" }));
    svg.append(svgElement("circle", { cx: x(latest.time), cy: y(latest.value), r: 3, fill: "#fff", stroke: "#567f45", "stroke-width": 1.5 }));
  }
  svg.append(svgElement("rect", { x: x(latest.time), y: top, width: Math.max(0, x(forecast.time) - x(latest.time)), height: height - bottom - top, fill: "#e6efda", opacity: ".35" }));
  if (latest.value !== null) svg.append(svgElement("line", { x1: x(latest.time), y1: y(latest.value), x2: x(forecast.time), y2: y(forecast.value), stroke: "#2c684b", "stroke-width": 2, "stroke-dasharray": "3 3" }));
  svg.append(svgElement("circle", { cx: x(forecast.time), cy: y(forecast.value), r: 7, fill: "#e2edd7" }));
  svg.append(svgElement("circle", { cx: x(forecast.time), cy: y(forecast.value), r: 3, fill: "#306847" }));
  for (let index = 0; index < 5; index++) {
    const timestamp = start + (origin - start) * index / 4;
    svg.append(svgElement("text", { x: x(timestamp), y: height - 13, "text-anchor": index === 0 ? "start" : index === 4 ? "end" : "middle", fill: "#a1ad96", "font-size": 9 }, state.hours === 24 ? timeOnly(timestamp) : dateFormat(timestamp, { day: "2-digit", month: "short", hour: "2-digit", hourCycle: "h23" })));
  }
  const crosshair = svgElement("line", { y1: top, y2: height - bottom, stroke: "#91a67f", "stroke-width": 1, "stroke-dasharray": "3 3", visibility: "hidden" });
  svg.append(crosshair);
  svg.onpointermove = (event) => {
    const bounds = svg.getBoundingClientRect();
    const position = (event.clientX - bounds.left) / bounds.width * width;
    const timestamp = start + (position - left) / (width - left - right) * (end - start);
    const point = allPoints.reduce((closest, current) => Math.abs(new Date(current.time).getTime() - timestamp) < Math.abs(new Date(closest.time).getTime() - timestamp) ? current : closest);
    crosshair.setAttribute("x1", x(point.time)); crosshair.setAttribute("x2", x(point.time)); crosshair.setAttribute("visibility", "visible");
    const tooltip = $("#chart-tooltip"); tooltip.replaceChildren();
    const label = document.createElement("span"); label.textContent = `${point.forecast ? "Forecast" : point.entered ? "Entered reading" : "History"} · ${dayTime(point.time)}`;
    const value = document.createElement("strong"); value.textContent = point.value === null ? "Missing reading" : `${fmt(point.value)} µg/m³`;
    tooltip.append(label, value); tooltip.hidden = false;
    tooltip.style.left = `${Math.min(bounds.width - 75, Math.max(75, x(point.time) / width * bounds.width))}px`;
    tooltip.style.top = `${Math.max(53, (point.value === null ? 95 : y(point.value)) / height * bounds.height)}px`;
  };
  svg.onpointerleave = () => { crosshair.setAttribute("visibility", "hidden"); $("#chart-tooltip").hidden = true; };
}

function appendCell(row, text, className) {
  const cell = document.createElement("td"); cell.textContent = text; if (className) cell.className = className; row.append(cell); return cell;
}

function renderPerformance(data) {
  const metrics = Object.fromEntries(data.metrics.map((row) => [row.model, row]));
  setText("#performance-xgb", fmt(metrics.xgboost.rmse, 2)); setText("#performance-ridge", fmt(metrics.ridge.rmse, 2));
  setText("#performance-gain", fmt((metrics.persistence.rmse - metrics.xgboost.rmse) / metrics.persistence.rmse * 100, 2) + "%");
  const body = $("#metrics-body"); body.replaceChildren();
  for (const model of ["xgboost", "ridge", "persistence", "seasonal_naive"]) {
    const values = metrics[model], row = document.createElement("tr"); if (model === "xgboost") row.className = "main-model";
    const label = appendCell(row, "");
    const dot = document.createElement("span"); dot.className = "metric-model-dot"; label.append(dot, document.createTextNode(modelLabels[model]));
    if (model === "xgboost" || model === "ridge") { const badge = document.createElement("span"); badge.className = "metric-tag"; badge.textContent = model === "xgboost" ? "MAIN MODEL" : "LOWEST RMSE"; label.append(badge); }
    appendCell(row, fmt(values.mae, 3)); appendCell(row, fmt(values.rmse, 3)); appendCell(row, fmt(values.r2, 4)); body.append(row);
  }
  const quarters = $("#quarter-body"); quarters.replaceChildren();
  for (let quarter = 1; quarter <= 4; quarter++) {
    const rows = data.quarters.filter((row) => row.fold === `holdout_0${quarter}`);
    const row = document.createElement("tr"); appendCell(row, "Q" + quarter);
    const best = Math.min(...rows.filter((entry) => entry.model !== "seasonal_naive").map((entry) => entry.rmse));
    for (const model of ["persistence", "ridge", "xgboost"]) { const value = rows.find((entry) => entry.model === model).rmse; appendCell(row, fmt(value, 3), value === best ? "quarter-best" : ""); }
    quarters.append(row);
  }
  const peak = (name) => data.regimes.find((row) => row.model === name && row.pollution_regime === "above_training_p90").rmse;
  setText("#peak-explanation", `For observed concentrations at or above ${fmt(data.training_p90, 0)} µg/m³, XGBoost RMSE is ${fmt(peak("xgboost"), 3)}, compared with ${fmt(peak("persistence"), 3)} for persistence.`);
}

async function uploadHistory(file) {
  if (!file || state.busy) return;
  clearErrors();
  try {
    if (file.size > 4 * 1024 * 1024) throw { message: "This CSV is too large. Please keep uploads below 4 MB." };
    setBusy(true, "Checking your CSV");
    const body = new FormData(); body.append("file", file);
    const context = await api("/api/history", { method: "POST", body });
    setContext(context); applyMode("history");
    setBusy(false);
    if (await generateForecast()) toast("History loaded. Your forecast is ready.");
  } catch (error) { showError(error); }
  finally { $("#csv-file").value = ""; setBusy(false); }
}

async function boot(mode = "manual") {
  clearErrors();
  try {
    setBusy(true, "Loading the models");
    const data = await api("/api/bootstrap");
    state.ready = true; setContext(data.context); $("#model").value = "xgboost";
    $("#csv-file").value = "";
    applyMode(mode, false); renderPerformance(data);
    $("#connection").classList.add("ready"); $("#connection").classList.remove("failed"); setText("#connection-text", "Models connected");
    if (mode === "manual") renderForecast(data.forecast);
    else { setBusy(false); await generateForecast(); }
  } catch (error) { state.ready = false; $("#connection").classList.add("failed"); setText("#connection-text", "Models unavailable"); showError(error); }
  finally { setBusy(false); }
}

function switchView(name) {
  if (!["forecast", "performance", "about"].includes(name)) name = "forecast";
  $$(".view").forEach((view) => { view.hidden = view.id !== "view-" + name; });
  $$(".nav-item").forEach((button) => { const active = button.dataset.view === name; button.classList.toggle("active", active); if (active) button.setAttribute("aria-current", "page"); else button.removeAttribute("aria-current"); });
  setText("#breadcrumb-current", { forecast: "Forecast", performance: "Model performance", about: "About the data" }[name]);
  $("#sidebar").classList.remove("open"); $("#menu-button").setAttribute("aria-expanded", "false");
  if (location.hash !== "#" + name) history.replaceState(null, "", "#" + name);
  window.scrollTo({ top: 0, behavior: "instant" });
}

let toastTimeout;
function toast(message) { clearTimeout(toastTimeout); setText("#toast", message); $("#toast").hidden = false; toastTimeout = setTimeout(() => { $("#toast").hidden = true; }, 3500); }

function exportForecast() {
  if (!state.result || state.stale) return;
  const result = state.result;
  const escape = (value) => { const text = value === null ? "" : String(value); return '"' + (/^[=+@]/.test(text) ? "'" : "") + text.replaceAll('"', '""') + '"'; };
  const rows = [["model", "origin_time", "target_time", "pm25_ug_m3", "input_mode", "source", "historical_scenario"]];
  for (const [model, prediction] of Object.entries({ ...result.predictions, ...result.baselines })) rows.push([model, result.origin_time, result.target_time, prediction, result.mode, result.context.label, result.historical]);
  const blob = new Blob([rows.map((row) => row.map(escape).join(",")).join("\r\n") + "\r\n"], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob), anchor = document.createElement("a"); anchor.href = url; anchor.download = `aircast_forecast_${result.target_time.slice(0, 10)}.csv`; document.body.append(anchor); anchor.click(); anchor.remove(); setTimeout(() => URL.revokeObjectURL(url), 10000); toast("Forecast exported as CSV.");
}

$("#forecast-form").addEventListener("submit", (event) => { event.preventDefault(); generateForecast(); });
$$(".input-tab").forEach((button) => {
  button.addEventListener("click", () => applyMode(button.dataset.mode));
  button.addEventListener("keydown", (event) => { if (["ArrowLeft", "ArrowRight"].includes(event.key)) { event.preventDefault(); const next = button.dataset.mode === "manual" ? "history" : "manual"; applyMode(next); $("#tab-" + next).focus(); } });
});
Object.values(fields).forEach((id) => $("#" + id).addEventListener("input", () => { clearErrors(); markStale(); }));
$("#model").addEventListener("change", markStale);
$("#csv-file").addEventListener("change", (event) => uploadHistory(event.target.files[0]));
const dropzone = $("#dropzone");
for (const type of ["dragenter", "dragover"]) dropzone.addEventListener(type, (event) => { event.preventDefault(); dropzone.classList.add("dragging"); });
for (const type of ["dragleave", "drop"]) dropzone.addEventListener(type, (event) => { event.preventDefault(); dropzone.classList.remove("dragging"); });
dropzone.addEventListener("drop", (event) => uploadHistory(event.dataTransfer.files[0]));
$("#try-example").addEventListener("click", () => boot("history"));
$("#reset-sample").addEventListener("click", () => boot());
$("#change-history").addEventListener("click", () => { applyMode("history"); $("#tab-history").focus(); $("#input-heading").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth", block: "start" }); });
$("#download-forecast").addEventListener("click", exportForecast);
$$(".chart-ranges button").forEach((button) => button.addEventListener("click", () => { state.hours = Number(button.dataset.hours); $$(".chart-ranges button").forEach((entry) => { const active = entry === button; entry.classList.toggle("active", active); entry.setAttribute("aria-pressed", String(active)); }); drawChart(); }));
$$('[data-view]').forEach((button) => button.addEventListener("click", () => switchView(button.dataset.view)));
$(".brand").addEventListener("click", (event) => { event.preventDefault(); switchView("forecast"); });
$("#menu-button").addEventListener("click", () => { const open = $("#sidebar").classList.toggle("open"); $("#menu-button").setAttribute("aria-expanded", String(open)); });
document.addEventListener("click", (event) => { if (window.innerWidth <= 640 && !$("#sidebar").contains(event.target) && !$("#menu-button").contains(event.target)) { $("#sidebar").classList.remove("open"); $("#menu-button").setAttribute("aria-expanded", "false"); } });
document.addEventListener("keydown", (event) => { if (event.key === "Escape") { $("#sidebar").classList.remove("open"); $("#menu-button").setAttribute("aria-expanded", "false"); $("#chart-tooltip").hidden = true; } });
window.addEventListener("hashchange", () => switchView(location.hash.slice(1)));
switchView(location.hash.slice(1) || "forecast");
boot();
