/* app.js -- LLM Lab front-end.  Vanilla JS, no build step. */

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));

async function api(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await res.json().catch(() => ({ error: "bad response" }));
  if (!res.ok || data.error) throw new Error(data.error || ("HTTP " + res.status));
  return data;
}

async function get(path) {
  const res = await fetch(path);
  const data = await res.json().catch(() => ({ error: "bad response" }));
  if (!res.ok || data.error) throw new Error(data.error || ("HTTP " + res.status));
  return data;
}

const state = {
  corpus: "",
  modelId: "default",
  trainedId: null,
  trainedEpochs: 0,
  randomBaseline: null,
  points: [],          // [{epoch, loss}]
  polling: false,
  nextEvent: 0,
  jobId: null,
  milestones: [],
};

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pct = (x) => (x * 100).toFixed(1) + "%";
const num = (n) => Number(n).toLocaleString("en-US");

/* ------------------------------------------------------------------ boot */
async function boot() {
  wireTabs();
  wireButtons();
  await loadLessons();
  await resetCorpusToFile();
  await loadModel();
  await runNetwork();
  await runExamples();
  selectLesson("start");
}

async function loadLessons() {
  const data = await get("/api/lessons");
  const list = $("#lessonList");
  list.innerHTML = "";
  data.lessons.forEach((l, i) => {
    const li = document.createElement("li");
    li.textContent = l.title;
    li.dataset.id = l.id;
    li.addEventListener("click", () => selectLesson(l.id));
    list.appendChild(li);
  });
  window._lessons = data.lessons;

  const ex = $("#exerciseList");
  ex.innerHTML = "";
  const saved = JSON.parse(localStorage.getItem("llm-lab-exercises") || "{}");
  data.exercises.forEach((e) => {
    const li = document.createElement("li");
    li.innerHTML = `<label><input type="checkbox" data-ex="${e.id}" ${saved[e.id] ? "checked" : ""}><span>${esc(e.text)}</span></label>`;
    ex.appendChild(li);
  });
  $$("#exerciseList input").forEach((cb) => cb.addEventListener("change", () => {
    const s = JSON.parse(localStorage.getItem("llm-lab-exercises") || "{}");
    s[cb.dataset.ex] = cb.checked;
    localStorage.setItem("llm-lab-exercises", JSON.stringify(s));
  }));

  const tb = $("#compareTable tbody");
  tb.innerHTML = data.compare
    .map((r) => `<tr><td>${esc(r.mini)}</td><td>${esc(r.real)}</td></tr>`).join("");
}

function selectLesson(id) {
  const l = (window._lessons || []).find((x) => x.id === id);
  if (!l) return;
  $$("#lessonList li").forEach((li) => li.classList.toggle("active", li.dataset.id === id));
  const card = $("#lessonCard");
  const cols = (l.body && l.body.length) ? `<div>${l.body.map((p) => `<p>${esc(p)}</p>`).join("")}</div>` : "";
  card.innerHTML = `
    <div class="tag">Stage ${esc(l.title.split(".")[0])}</div>
    <h2>${esc(l.title.replace(/^\d+\.\s*/, ""))}</h2>
    <div class="tagline">${esc(l.tagline)}</div>
    ${cols}
    <div class="cols">
      <div>
        <h4>Key points</h4>
        <ul>${l.points.map((p) => `<li>${esc(p)}</li>`).join("")}</ul>
      </div>
      <div>
        <h4>Try it yourself</h4>
        <ul>${l.try_this.map((p) => `<li>${esc(p)}</li>`).join("")}</ul>
      </div>
    </div>
    ${l.code ? `<pre>${esc(l.code)}</pre>` : ""}`;
  if (l.tab) showTab(l.tab);
}

/* ---------------------------------------------------------------- tabs */
function wireTabs() {
  $$("#tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
}
function showTab(name) {
  $$("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$(".panel").forEach((p) => p.classList.toggle("active", p.id === "panel-" + name));
  if (name === "network") refreshEmbeddings();
}

/* ------------------------------------------------------- corpus panel */
async function resetCorpusToFile() {
  const c = await get("/api/corpus");
  state.corpus = c.text;
  $("#corpusText").value = c.text;
  $("#corpusStatus").textContent = "sample_text.txt \u2014 " + num(c.chars) + " characters";
}

function renderStats(s) {
  const cells = [
    ["characters", num(s.chars), ""],
    ["vocabulary", num(s.vocab_size), "accent"],
    ["examples", num(s.n_examples), ""],
    ["parameters", num(s.n_params), "accent"],
    ["random guess loss", (s.random_baseline ?? Math.log(s.vocab_size)).toFixed(4), "good"],
  ];
  $("#corpusStats").innerHTML = cells
    .map(([k, v, c]) => `<div class="stat"><div class="k">${k}</div><div class="v ${c}">${v}</div></div>`).join("");
}

function renderVocab(vocab) {
  $("#vocabTable").innerHTML = vocab
    .map((v) => `<div class="cell"><span class="ch">${esc(v.label)}</span><span class="id">= ${v.id}</span></div>`).join("");
}

function renderFreq(rows) {
  const max = Math.max(...rows.map((r) => r.count));
  $("#freqChart").innerHTML = rows.slice(0, 12).map((r) => `
    <div class="frow">
      <span class="fname">${esc(r.label)}</span>
      <span class="fbar"><i style="width:${(r.count / max * 100).toFixed(1)}%"></i></span>
      <span class="fpct">${pct(r.freq)}</span>
    </div>`).join("");
}

async function loadModel() {
  const s = await api("/api/load", { text: state.corpus });
  state.modelId = s.model_id;
  state.randomBaseline = s.random_baseline;
  renderStats(s);
  renderVocab(s.vocab);
  renderFreq(s.char_counts);
  updateEstimate(s.n_examples, s.n_params);
  return s;
}

/* ----------------------------------------------------- examples panel */
async function runExamples() {
  const probe = $("#exProbe").value.trim();
  const context = +$("#exContext").value || 3;
  try {
    const d = await api("/api/examples", {
      probe, context, count: +$("#exCount").value || 10,
    });
    const tb = $("#exTable tbody");
    tb.innerHTML = d.rows.map((r) => `
      <tr>
        <td>[${r.ids.join(", ")}]</td>
        <td>${esc(r.context_label)}</td>
        <td>${esc(r.target_label)}</td>
        <td class="wrapcell">${esc(r.whole)}</td>
      </tr>`).join("");
    $("#exSummary").textContent =
      `context = ${d.context} \u00b7 ${num(d.total)} examples from this text` +
      (probe ? "" : " (the loaded corpus)");
  } catch (e) { $("#exSummary").textContent = "Error: " + e.message; }
}

/* ------------------------------------------------------ network panel */
async function runNetwork() {
  const context = +$("#netContext").value || 3;
  const embed = +$("#netEmbed").value || 6;
  const hidden = +$("#netHidden").value || 24;
  try {
    const d = await api("/api/network", { text: state.corpus, context, embed, hidden });
    const flow = [];
    flow.push(node(`${context} chars`, "context"));
    flow.push(arrow());
    flow.push(node(`embeddings`, `${context}\u00d7${embed} = ${context * embed} numbers`));
    flow.push(arrow());
    flow.push(node(`${hidden} neurons`, "tanh"));
    flow.push(arrow());
    flow.push(node(`${d.vocab_size} scores`, "softmax"));
    flow.push(arrow());
    flow.push(node(`${d.vocab_size} probs`, "sum to 1"));
    $("#flowChart").innerHTML = flow.join("");
    $("#shapeTable tbody").innerHTML = d.shapes.map((s) => `
      <tr><td>${esc(s.name)}</td><td>${esc(s.label)}</td><td>${esc(s.cols)}</td></tr>`).join("");
    $("#netSummary").textContent =
      `${num(d.n_params)} learnable numbers \u00b7 untrained, it should score about ` +
      `${d.random_baseline.toFixed(4)} (= ln ${d.vocab_size}, random guessing)`;
    updateEstimate(null, d.n_params);
  } catch (e) { $("#netSummary").textContent = "Error: " + e.message; }
}
const node = (a, b) => `<div class="node">${esc(a)}<small>${esc(b)}</small></div>`;
const arrow = () => `<span class="arrow">&rarr;</span>`;

async function refreshEmbeddings() {
  const id = state.trainedId || state.modelId;
  const box = $("#embeddingPanel");
  try {
    const d = await api("/api/embeddings", { model_id: id });
    if (!d.trained) {
      box.innerHTML = `<p class="empty">Train a model first, then come back to see what the embeddings learned.</p>`;
      return;
    }
    box.innerHTML = d.neighbours.map((n) => `
      <div class="emb"><span class="who">${esc(n.label)}</span>
      <div>closest: ${n.neighbours.map(esc).join(", ")}</div></div>`).join("");
  } catch (e) { box.innerHTML = `<p class="empty">Error: ${esc(e.message)}</p>`; }
}

/* -------------------------------------------------------- train panel */
function wireButtons() {
  $("#resetCorpus").addEventListener("click", async () => {
    await resetCorpusToFile();
    await loadModel();
  });
  $("#loadCorpus").addEventListener("click", async () => {
    state.corpus = $("#corpusText").value;
    try {
      const s = await loadModel();
      $("#corpusStatus").textContent = `loaded \u2014 ${num(s.chars)} characters, ${num(s.vocab_size)} distinct`;
    } catch (e) { $("#corpusStatus").textContent = "Error: " + e.message; }
  });
  $("#exRun").addEventListener("click", runExamples);
  $("#exProbe").addEventListener("keydown", (e) => { if (e.key === "Enter") runExamples(); });
  $("#netRun").addEventListener("click", runNetwork);
  $("#trainBtn").addEventListener("click", startTraining);
  $("#stopBtn").addEventListener("click", () => { state.polling = false; $("#stopBtn").disabled = true; });
  $("#predRun").addEventListener("click", runPredict);
  $("#predSeed").addEventListener("keydown", (e) => { if (e.key === "Enter") runPredict(); });
  $("#genRun").addEventListener("click", runGenerate);
  $("#genTemp").addEventListener("input", () => { $("#genTempOut").textContent = (+$("#genTemp").value).toFixed(2); });
  $("#helpBtn").addEventListener("click", () => { $("#helpModal").hidden = false; });
  $("#helpClose").addEventListener("click", () => { $("#helpModal").hidden = true; });
  $("#helpModal").addEventListener("click", (e) => { if (e.target.id === "helpModal") $("#helpModal").hidden = true; });
  ["trEpochs"].forEach((id) => $("#" + id).addEventListener("input", () => updateEstimate(null, null)));
}

function updateEstimate(nExamples, nParams) {
  if (nExamples) state._nExamples = nExamples;
  if (nParams) state._nParams = nParams;
  const epochs = +$("#trEpochs").value || 1;
  const hidden = +$("#trHidden").value || 24;
  const ex = state._nExamples || 2393;
  const secs = epochs * ex * 8.5e-5 * (hidden / 24);
  const word = secs < 1 ? "under a second" : secs < 90 ? `about ${Math.round(secs)} seconds` :
    `about ${(secs / 60).toFixed(1)} minutes`;
  $("#trainEstimate").textContent =
    `${num(epochs)} epochs over ${num(ex)} examples \u00b7 roughly ${word} in this browser session ` +
    `(the maths runs in Python, one multiply at a time).`;
}

async function startTraining() {
  const payload = {
    text: state.corpus,
    context: +$("#trContext").value || 3,
    embed: +$("#trEmbed").value || 6,
    hidden: +$("#trHidden").value || 24,
    epochs: +$("#trEpochs").value || 60,
    lr: +$("#trLr").value || 5,
    seed: +$("#trSeed").value || 0,
  };
  state.points = [];
  state.milestones = [];
  state.nextEvent = 0;
  $("#milestones").innerHTML = `<p class="empty">Training&hellip;</p>`;
  $("#trainBtn").disabled = true;
  $("#progressBar").style.width = "0%";
  $("#trainStats").innerHTML = "";
  try {
    const start = await api("/api/train", payload);
    state.jobId = start.job_id;
    state.randomBaseline = start.random_baseline;
    state.polling = true;
    $("#stopBtn").disabled = false;
    $("#modelPill").textContent = "training\u2026";
    $("#modelPill").classList.remove("trained");
    pollJob();
  } catch (e) {
    $("#trainBtn").disabled = false;
    $("#milestones").innerHTML = `<p class="empty">Error: ${esc(e.message)}</p>`;
  }
}

async function pollJob() {
  if (!state.polling) return;
  try {
    const d = await get(`/api/job?id=${state.jobId}&from=${state.nextEvent}`);
    state.nextEvent = d.next;

    if (d.epochs) {
      $("#progressBar").style.width = ((d.epoch / d.epochs) * 100).toFixed(1) + "%";
    }
    if (d.loss != null) state.points.push({ epoch: d.epoch, loss: d.loss });
    drawChart();
    renderTrainStats(d);

    d.events.forEach((ev) => {
      if (ev.type === "milestone") {
        const final = ev.epoch === d.epochs;
        $("#milestones").querySelectorAll(".empty").forEach((n) => n.remove());
        const div = document.createElement("div");
        div.className = "milestone" + (final ? " final" : "");
        div.innerHTML = `
          <div class="mhead"><span>after ${num(ev.epoch)} of ${num(d.epochs)} epochs</span>
          <span>loss ${ev.loss.toFixed(4)} \u00b7 perplexity ${ev.perplexity.toFixed(2)} \u00b7 ${ev.bits.toFixed(2)} bits/char</span></div>
          <div class="mtext">${esc(ev.sample)}</div>`;
        $("#milestones").appendChild(div);
      }
      if (ev.type === "error") {
        const p = document.createElement("p");
        p.className = "empty";
        p.style.color = "var(--warn)";
        p.textContent = ev.message;
        $("#milestones").appendChild(p);
      }
      if (ev.type === "done") {
        state.trainedId = d.model_id;
        state.trainedEpochs = ev.epochs;
        $("#modelPill").textContent = `trained \u00b7 ${num(ev.epochs)} epochs`;
        $("#modelPill").classList.add("trained");
        $("#predChart").innerHTML = `<p class="empty">Model ready \u2014 press Predict.</p>`;
        $("#genOut").textContent = "Model ready \u2014 press Generate.";
        refreshEmbeddings();
        runPredict();
        runGenerate();
      }
    });

    if (d.status === "running") {
      setTimeout(pollJob, 250);
    } else {
      state.polling = false;
      $("#stopBtn").disabled = true;
      $("#trainBtn").disabled = false;
      drawChart();
    }
  } catch (e) {
    state.polling = false;
    $("#stopBtn").disabled = true;
    $("#trainBtn").disabled = false;
    $("#milestones").innerHTML = `<p class="empty">Error: ${esc(e.message)}</p>`;
  }
}

function renderTrainStats(d) {
  const cells = [
    ["epoch", d.epochs ? `${num(d.epoch)} / ${num(d.epochs)}` : "\u2014", ""],
    ["loss now", d.loss != null ? d.loss.toFixed(4) : "\u2014", "accent"],
    ["start loss", d.start_loss != null ? d.start_loss.toFixed(4) : "\u2014", ""],
    ["random guess", state.randomBaseline != null ? state.randomBaseline.toFixed(4) : "\u2014", "good"],
    ["perplexity", d.loss != null ? Math.exp(d.loss).toFixed(2) : "\u2014", ""],
    ["elapsed", (d.elapsed || 0).toFixed(1) + "s", ""],
  ];
  $("#trainStats").innerHTML = cells
    .map(([k, v, c]) => `<div class="stat"><div class="k">${k}</div><div class="v ${c}">${v}</div></div>`).join("");
}

/* loss chart ------------------------------------------------------------- */
function drawChart() {
  const cv = $("#lossChart");
  const ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  const pad = { l: 58, r: 16, t: 16, b: 34 };
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#10131c";
  ctx.fillRect(0, 0, W, H);

  if (!state.points.length) {
    ctx.fillStyle = "#8b93ab";
    ctx.font = "14px sans-serif";
    ctx.fillText("loss will appear here as the model trains", pad.l, H / 2);
    return;
  }
  const losses = state.points.map((p) => p.loss).filter((x) => isFinite(x));
  const epochs = state.points.map((p) => p.epoch);
  let ymax = Math.max(...losses, state.randomBaseline || 0);
  let ymin = Math.min(...losses);
  const padY = (ymax - ymin) * 0.08 + 1e-6;
  ymax += padY; ymin = Math.max(0, ymin - padY);
  const xmax = Math.max(...epochs), xmin = Math.min(...epochs);
  const X = (e) => pad.l + (e - xmin) / Math.max(1, xmax - xmin) * (W - pad.l - pad.r);
  const Y = (v) => pad.t + (1 - (v - ymin) / (ymax - ymin)) * (H - pad.t - pad.b);

  // grid + axis labels
  ctx.strokeStyle = "#2b3145"; ctx.lineWidth = 1;
  ctx.fillStyle = "#8b93ab"; ctx.font = "11px monospace";
  for (let i = 0; i <= 4; i++) {
    const v = ymin + (ymax - ymin) * i / 4;
    const y = Y(v);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    ctx.fillText(v.toFixed(2), 6, y + 4);
  }
  ctx.fillText(xmin, pad.l, H - 12);
  ctx.fillText("epoch " + xmax, W - pad.r - 62, H - 12);

  // random-guess baseline
  if (state.randomBaseline) {
    const by = Y(state.randomBaseline);
    ctx.save();
    ctx.setLineDash([6, 5]); ctx.strokeStyle = "#7ee0b8";
    ctx.beginPath(); ctx.moveTo(pad.l, by); ctx.lineTo(W - pad.r, by); ctx.stroke();
    ctx.restore();
    ctx.fillStyle = "#7ee0b8";
    ctx.fillText("random guessing  ln(vocab)", pad.l + 8, by - 6);
  }

  // loss line
  ctx.strokeStyle = "#6ea8fe"; ctx.lineWidth = 2;
  ctx.beginPath();
  state.points.forEach((p, i) => {
    if (!isFinite(p.loss)) return;
    const x = X(p.epoch), y = Y(p.loss);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();

  // last point dot
  const last = state.points[state.points.length - 1];
  if (isFinite(last.loss)) {
    ctx.fillStyle = "#6ea8fe";
    ctx.beginPath(); ctx.arc(X(last.epoch), Y(last.loss), 3.5, 0, 7); ctx.fill();
  }
}

/* ----------------------------------------------------- generate panel */
async function runPredict() {
  const id = state.trainedId || state.modelId;
  try {
    const d = await api("/api/predict", { model_id: id, seed: $("#predSeed").value, k: +$("#predK").value || 8 });
    const rows = d.predictions;
    const all = d.all;
    const order = all.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p);
    const topIds = new Set(rows.map((r) => r.id));
    let html = rows.map((r, idx) => `
      <div class="predrow ${idx === 0 ? "top" : ""}">
        <span class="plabel">${esc(r.label)}</span>
        <span class="pbar"><i style="width:${Math.max(0.6, r.prob * 100).toFixed(1)}%"></i></span>
        <span class="ppct">${pct(r.prob)}</span>
      </div>`).join("");
    html += `<p class="muted" style="margin-top:10px">all ${all.length} characters, by probability:</p>`;
    html += `<div class="predchart">` + order.map(({ p, i }) => `
      <div class="predrow">
        <span class="plabel" style="color:${topIds.has(i) ? "#dbe0ee" : "#8b93ab"}">${esc(chLabel(i, d))}</span>
        <span class="pbar"><i style="width:${(p * 100).toFixed(1)}%;opacity:${topIds.has(i) ? 1 : 0.45}"></i></span>
        <span class="ppct">${pct(p)}</span>
      </div>`).join("") + `</div>`;
    $("#predChart").innerHTML = html;
  } catch (e) { $("#predChart").innerHTML = `<p class="empty">Error: ${esc(e.message)}</p>`; }
}

// The server sends top-k labels; build a full label list from the vocabulary once.
let _vocabLabels = null;
function chLabel(i, d) {
  if (d.predictions.some((r) => r.id === i)) return d.predictions.find((r) => r.id === i).label;
  if (!_vocabLabels) return "#" + i;
  return _vocabLabels[i] || ("#" + i);
}

async function runGenerate() {
  const id = state.trainedId || state.modelId;
  try {
    const d = await api("/api/generate", {
      model_id: id, seed: $("#genSeed").value,
      length: +$("#genLength").value || 200,
      temperature: +$("#genTemp").value || 0.5, runs: 1,
    });
    $("#genOut").textContent = d.outputs[0];
  } catch (e) { $("#genOut").textContent = "Error: " + e.message; }
}

/* Keep a full label list available for the "all characters" view. */
(async function prefetchVocab() {
  try {
    const c = await get("/api/corpus");
    const s = await api("/api/load", { text: c.text });
    _vocabLabels = s.vocab.map((v) => v.label);
  } catch (_) { /* fine */ }
})();

boot().catch((e) => {
  document.body.insertAdjacentHTML("afterbegin",
    `<div style="padding:14px;background:#3a1f1f;color:#ef6b6b">Failed to start: ${esc(e.message)}</div>`);
});
