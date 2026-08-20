/* nerd sample-input helper - frontend.
 *
 * The table is the centre of gravity: everything else (fastq ingest,
 * pattern extraction, bulk fill, the entity wizards) is a way of filling
 * it in. Server state is authoritative -- every mutation returns the whole
 * new state and we re-render from that, so the UI can never drift out of
 * sync with what will actually be written.
 */
const $ = (id) => document.getElementById(id);

let S = null;             // last server state
let table = null;
let tokens = {};
let entitySchema = { fields: {}, lookup_fields: {} };
let queue = [];           // pending unresolved entities
let queueIndex = 0;

/* ---------------------------------------------------------------- util */

function toast(msg, kind = "info") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = msg;
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), 5000);
}

async function api(path, options = {}) {
  const resp = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data.detail || `${resp.status} ${resp.statusText}`);
  return data;
}

async function post(path, body) {
  return api(path, { method: "POST", body: JSON.stringify(body || {}) });
}

/* ---------------------------------------------------------------- panels */

document.querySelectorAll(".step").forEach((btn) => {
  btn.addEventListener("click", () => {
    const target = btn.dataset.panel;
    const panel = $(target);
    const isOpen = panel.classList.contains("open");
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("open"));
    document.querySelectorAll(".step").forEach((s) => s.classList.remove("active"));
    if (!isOpen) {
      panel.classList.add("open");
      btn.classList.add("active");
      if (table) setTimeout(() => table.redraw(true), 20);
    }
  });
});

document.querySelectorAll("[data-close]").forEach((b) =>
  b.addEventListener("click", () => b.closest("dialog").close())
);

/* ---------------------------------------------------------------- connect */

$("connectBtn").addEventListener("click", async () => {
  const project_dir = $("projectDir").value.trim();
  if (!project_dir) return toast("Enter a project folder path.", "err");
  try {
    const data = await post("/api/session/connect", {
      project_dir,
      db_path: $("dbPath").value.trim() || null,
      label: $("labelInput").value.trim() || "sample_import",
    });
    $("connectStatus").textContent = data.is_new_db ? "new database" : "connected";
    $("connectStatus").className = "pill pill-ok";
    render(data.state);
    const counts = data.db_entity_counts || {};
    toast(
      `Connected. ${counts.construct || 0} constructs, ${counts.buffer || 0} buffers already in this database.`,
      "ok"
    );
    if (data.restored_draft) toast("Picked up where you left off.", "info");
    if (!S.rows.length) openPanel("ingestPanel");
  } catch (e) {
    $("connectStatus").textContent = "failed";
    $("connectStatus").className = "pill pill-bad";
    toast(e.message, "err");
  }
});

function openPanel(id) {
  document.querySelector(`.step[data-panel="${id}"]`)?.click();
}

/* ---------------------------------------------------------------- ingest */

$("ingestFastqBtn").addEventListener("click", () => ingest(false));
$("scanLocalBtn").addEventListener("click", () => ingest(true));

async function ingest(scanLocal) {
  const fq_dir = $("fqDir").value.trim();
  if (!fq_dir) return toast("Enter the fastq folder path first.", "err");
  try {
    const data = await post("/api/ingest/fastq", {
      fq_dir,
      listing: $("fqListing").value,
      scan_local: scanLocal,
    });
    render(data.state);
    const r = data.result;
    let msg = `${r.added} sample(s) from ${r.total_files} files.`;
    if (r.unpaired.length) msg += ` ${r.unpaired.length} file(s) had no R1/R2 partner.`;
    $("ingestResult").textContent = msg + (r.unpaired.length ? ` Unpaired: ${r.unpaired.slice(0, 5).join(", ")}` : "");
    toast(msg, r.unpaired.length ? "info" : "ok");
    if (r.added) openPanel("patternPanel");
  } catch (e) { toast(e.message, "err"); }
}

$("ingestNamesBtn").addEventListener("click", async () => {
  try {
    const data = await post("/api/ingest/names", { names: $("plainNames").value });
    render(data.state);
    toast(`${data.result.added} row(s) created.`, "ok");
    openPanel("patternPanel");
  } catch (e) { toast(e.message, "err"); }
});

/* ---------------------------------------------------------------- tokens */

let caret = null;
["click", "keyup", "focus"].forEach((ev) =>
  $("patternInput").addEventListener(ev, () => { caret = $("patternInput").selectionStart; })
);
$("patternInput").addEventListener("dragover", (e) => e.preventDefault());
$("patternInput").addEventListener("drop", (e) => {
  e.preventDefault();
  insertToken(e.dataTransfer.getData("text/plain"));
});
$("patternInput").addEventListener("input", debounce(previewPattern, 350));

function insertToken(text) {
  const input = $("patternInput");
  const pos = caret ?? input.value.length;
  input.value = input.value.slice(0, pos) + text + input.value.slice(pos);
  caret = pos + text.length;
  input.focus();
  input.setSelectionRange(caret, caret);
  previewPattern();
}

async function loadTokens() {
  tokens = await api("/api/tokens");
  const palette = $("tokenPalette");
  palette.innerHTML = "";
  Object.entries(tokens).forEach(([name, spec]) => {
    const chip = document.createElement("span");
    const wild = name === "*";
    chip.className = "token-chip" + (wild ? " wild" : "");
    chip.textContent = wild ? "[*] ignore" : `[${name}]`;
    chip.title = spec.description || (spec.maps_to ? `fills ${spec.maps_to}` : "parsed for context");
    chip.draggable = true;
    chip.addEventListener("click", () => insertToken(`[${name}]`));
    chip.addEventListener("dragstart", (e) => e.dataTransfer.setData("text/plain", `[${name}]`));
    palette.appendChild(chip);
  });

  const list = $("tokenList");
  list.innerHTML = "";
  Object.entries(tokens).forEach(([name, spec]) => {
    const card = document.createElement("span");
    card.className = "token-card";
    card.innerHTML = `<b>[${name}]</b> ${spec.maps_to ? "&rarr; " + spec.maps_to : "<i>context only</i>"} `;
    const del = document.createElement("button");
    del.textContent = "×";
    del.title = "Remove token";
    del.onclick = async () => {
      try { await api(`/api/tokens/${encodeURIComponent(name)}`, { method: "DELETE" }); await loadTokens(); }
      catch (e) { toast(e.message, "err"); }
    };
    card.appendChild(del);
    list.appendChild(card);
  });
}

$("addTokenBtn").addEventListener("click", async () => {
  const name = $("newTokenName").value.trim();
  if (!name) return toast("Give the token a name.", "err");
  try {
    await post("/api/tokens", {
      name,
      maps_to: $("newTokenColumn").value || null,
      normalize: $("newTokenNormalize").value || null,
      description: "",
    });
    $("newTokenName").value = "";
    await loadTokens();
    toast(`Token [${name}] added.`, "ok");
  } catch (e) { toast(e.message, "err"); }
});

/* ---------------------------------------------------------------- pattern */

function debounce(fn, ms) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

async function previewPattern() {
  const pattern = $("patternInput").value.trim();
  const box = $("patternPreview");
  if (!pattern || !S || !S.rows.length) { box.innerHTML = ""; return; }
  try {
    const data = await post("/api/pattern/preview", { pattern });
    box.innerHTML = data.rows.map((r) => {
      const kv = Object.entries(r.values).map(([k, v]) => `${k}=<span class="preview-kv">${v}</span>`).join("  ");
      return r.matched
        ? `<div class="preview-row"><b>${r.sample_name}</b> &rarr; ${kv || "<i>nothing extracted</i>"}</div>`
        : `<div class="preview-row bad"><b>${r.sample_name}</b> &rarr; no match</div>`;
    }).join("");
    if (data.warnings?.length) box.innerHTML += `<div class="preview-row bad">${data.warnings.join(" ")}</div>`;
  } catch (e) {
    box.innerHTML = `<div class="preview-row bad">${e.message}</div>`;
  }
}

$("applyPatternBtn").addEventListener("click", async () => {
  const pattern = $("patternInput").value.trim();
  if (!pattern) return toast("Build a pattern first.", "err");
  try {
    const data = await post("/api/fill/pattern", { pattern, force: $("forcePattern").checked });
    render(data.state);
    const r = data.result;
    let msg = `${r.matched} matched, ${r.unmatched} didn't.`;
    if (r.columns_written.length) msg += ` Filled: ${r.columns_written.join(", ")}.`;
    if (r.protected_cells) msg += ` ${r.protected_cells} hand-edited cell(s) left alone.`;
    toast(msg, r.unmatched ? "info" : "ok");
  } catch (e) { toast(e.message, "err"); }
});

/* ---------------------------------------------------------------- batch */

$("batchAllBtn").addEventListener("click", () => batchFill(null));
$("batchSelBtn").addEventListener("click", () => {
  const uids = table.getSelectedData().map((r) => r.uid);
  if (!uids.length) return toast("Select some rows first.", "err");
  batchFill(uids);
});

async function batchFill(uids) {
  const column = $("batchColumn").value;
  try {
    const data = await post("/api/fill/batch", { column, value: $("batchValue").value, uids });
    render(data.state);
    toast(`${column} set on ${data.result.written} row(s).`, "ok");
  } catch (e) { toast(e.message, "err"); }
}

/* ---------------------------------------------------------------- table */

function entityFormatter(column) {
  return (cell) => {
    const value = cell.getValue() ?? "";
    if (!value) return "";
    const type = S.entity_columns[column];
    const status = (S.resolution[column] || {})[value] || "missing";
    const wrap = document.createElement("span");
    wrap.className = "ent";
    const text = document.createElement("span");
    text.className = "ent-v";
    text.textContent = value;
    const badge = document.createElement("span");
    badge.className = `ent-b ${status}`;
    badge.textContent = status === "db" ? "in db" : status === "staged" ? "new" : "+ create";
    if (status === "missing") {
      badge.onclick = (e) => { e.stopPropagation(); openEntityWizard(type, value); };
    }
    wrap.append(text, badge);
    return wrap;
  };
}

function buildColumns() {
  const cols = [{
    formatter: "rowSelection", titleFormatter: "rowSelection", hozAlign: "center",
    headerSort: false, width: 36, frozen: true,
    cellClick: (e, cell) => cell.getRow().toggleSelect(),
  }];
  // Before the first connect there is no server state yet; the table still
  // renders (with its placeholder) and picks up real columns on render().
  if (!S) return cols;
  S.columns.forEach((name) => {
    const def = {
      title: name, field: `values.${name}`, editor: "input", headerSort: false,
      minWidth: 90, resizable: true,
      cellEdited: (cell) => {
        const row = cell.getRow().getData();
        post("/api/rows/cell", { uid: row.uid, column: name, value: cell.getValue() })
          .then((d) => render(d.state))
          .catch((e) => toast(e.message, "err"));
      },
      cellFormatter: null,
    };
    if (S.entity_columns[name]) { def.formatter = entityFormatter(name); def.width = 165; }
    if (name === "sample_name") { def.frozen = true; def.width = 240; }
    if (["temperature", "reaction_time", "probe_concentration", "treated"].includes(name)) def.hozAlign = "right";
    cols.push(def);
  });
  return cols;
}

function initTable() {
  table = new Tabulator("#grid", {
    data: [], columns: buildColumns(), index: "uid",
    layout: "fitDataFill", height: "100%",
    renderVertical: "virtual",           // keeps 1000+ rows smooth
    placeholder: "No samples yet — start with “Load samples”.",
    rowFormatter: (row) => {
      const data = row.getData();
      row.getElement().classList.toggle("row-unmatched", !!data.unmatched);
      row.getCells().forEach((cell) => {
        const field = cell.getField();
        if (!field || !field.startsWith("values.")) return;
        const column = field.slice(7);
        const el = cell.getElement();
        el.classList.remove("o-fastq", "o-pattern", "o-batch", "o-manual");
        const origin = (data.origins || {})[column];
        if (origin) el.classList.add(`o-${origin}`);
      });
    },
  });
}

/* ---------------------------------------------------------------- render */

function render(state) {
  S = state;
  if (!table) initTable();
  table.setColumns(buildColumns());
  table.replaceData(S.rows);
  $("rowCount").textContent = `${S.rows.length} sample${S.rows.length === 1 ? "" : "s"}`;
  renderIssues();
  renderStaged();
  renderGroups();
  if (S.pattern && !$("patternInput").value) $("patternInput").value = S.pattern;
  if (S.fq_dir && !$("fqDir").value) $("fqDir").value = S.fq_dir;
  if (!$("batchColumn").options.length) {
    S.columns.forEach((c) => {
      $("batchColumn").add(new Option(c, c));
      $("newTokenColumn").add(new Option(c, c));
    });
    $("newTokenColumn").add(new Option("(context only)", ""), 0);
    $("newTokenColumn").selectedIndex = 0;
  }
}

const CODE_LABELS = {
  missing_required: (n) => `${n} empty required cell${n === 1 ? "" : "s"}`,
  unresolved_entity: (n) => `${n} entr${n === 1 ? "y" : "ies"} to create`,
  duplicate_sample_name: (n) => `${n} duplicate name${n === 1 ? "" : "s"}`,
  not_numeric: (n) => `${n} non-numeric value${n === 1 ? "" : "s"}`,
  missing_fastq: (n) => `${n} missing fastq file${n === 1 ? "" : "s"}`,
  missing_fq_dir: (n) => `${n} missing folder${n === 1 ? "" : "s"}`,
  remote_fq_dir: () => `cluster paths not checked here`,
  empty_sheet: () => `no rows yet`,
};

function renderIssues() {
  const bar = $("issueBar");
  bar.innerHTML = "";
  const v = S.validation;
  if (!S.rows.length) return;

  queue = v.resolution_queue || [];
  if (v.ok) {
    bar.innerHTML = `<span class="issue-chip ok">ready to create</span>`;
  } else {
    Object.entries(v.by_code).forEach(([code, n]) => {
      const chip = document.createElement("span");
      const isWarn = code === "remote_fq_dir";
      chip.className = `issue-chip ${isWarn ? "warn" : "err"}`;
      chip.textContent = (CODE_LABELS[code] || ((x) => `${x} ${code}`))(n);
      if (code === "unresolved_entity") {
        chip.classList.add("action");
        chip.title = "Click to create them one by one";
        chip.onclick = () => startQueue();
      }
      bar.appendChild(chip);
    });
  }
  $("generateBtn").disabled = !v.ok;
  $("generateBtn").title = v.ok ? "" : "Fix the flagged problems first";
}

function renderStaged() {
  const list = $("stagedList");
  list.innerHTML = "";
  let total = 0;
  Object.entries(S.staged || {}).forEach(([type, records]) => {
    records.forEach((record) => {
      total += 1;
      const key = entitySchema.lookup_fields?.[type]?.[0] || "disp_name";
      const item = document.createElement("div");
      item.className = "staged-item";
      item.innerHTML = `<span class="type">${type.replace("_", " ")}</span>
        <b>${record[key] ?? ""}</b><span class="spacer"></span>`;
      const edit = document.createElement("button");
      edit.textContent = "edit";
      edit.style.color = "var(--accent)";
      edit.onclick = () => openEntityWizard(type, record[key], record);
      const del = document.createElement("button");
      del.textContent = "remove";
      del.onclick = async () => {
        try {
          const d = await post("/api/entities/delete", { entity_type: type, value: record[key] });
          render(d.state);
        } catch (e) { toast(e.message, "err"); }
      };
      item.append(edit, del);
      list.appendChild(item);
    });
  });
  $("stagedCount").textContent = total || "";
  if (!total) list.innerHTML = `<span class="hint">Nothing new yet. Constructs, buffers and sequencing runs you create here get written alongside the samples.</span>`;
}

/* ---------------------------------------------------------------- wizard */

let currentEntity = { type: null, ntRows: null };

function startQueue() {
  if (!queue.length) return;
  queueIndex = 0;
  openEntityWizard(queue[0].entity_type, queue[0].value);
}

function openEntityWizard(type, prefill, existing) {
  currentEntity = { type, ntRows: existing?.nt_rows || null };
  const fields = entitySchema.fields[type] || [];
  const lookupField = entitySchema.lookup_fields?.[type]?.[0] || "disp_name";

  $("entityTitle").textContent = existing
    ? `Edit ${type.replace("_", " ")}`
    : `New ${type.replace("_", " ")}`;
  $("entityError").textContent = "";

  const inQueue = queue.length && !existing;
  $("entityQueueNote").textContent = inQueue
    ? `The table references “${prefill}”, which doesn't exist yet. Fill this in to create it.`
    : "";
  $("queueProgress").textContent = inQueue && queue.length > 1
    ? `${queueIndex + 1} of ${queue.length}` : "";

  const container = $("entityFields");
  container.innerHTML = "";
  fields.forEach((field) => {
    const label = document.createElement("label");
    if (field.type === "textarea") label.className = "full";
    label.textContent = field.label + (field.required ? " *" : "");
    const input = document.createElement(field.type === "textarea" ? "textarea" : "input");
    input.name = field.name;
    if (field.type === "number") input.type = "number", input.step = "0.01";
    if (field.type === "textarea") input.rows = 3;
    input.value = existing?.[field.name] ?? (field.name === lookupField ? prefill || "" : "");
    label.appendChild(input);
    container.appendChild(label);
  });

  const isConstruct = type === "construct";
  $("ntBlock").classList.toggle("hidden", !isConstruct);
  renderNtGrid(currentEntity.ntRows || []);
  $("entityModal").showModal();
}

function renderNtGrid(rows) {
  currentEntity.ntRows = rows;
  const body = document.querySelector("#ntGrid tbody");
  body.innerHTML = "";
  rows.forEach((row, index) => {
    const tr = document.createElement("tr");
    ["site", "base", "base_region"].forEach((field) => {
      const td = document.createElement("td");
      const input = document.createElement("input");
      input.value = row[field] ?? "";
      input.addEventListener("change", () => { rows[index][field] = input.value; });
      td.appendChild(input);
      tr.appendChild(td);
    });
    body.appendChild(tr);
  });
  $("ntSummary").textContent = rows.length
    ? `${rows.length} positions (${rows[0].site} … ${rows[rows.length - 1].site})`
    : "no numbering yet — it will default to 1-based";
}

$("ntDefaultBtn").addEventListener("click", async () => {
  const sequence = $("entityForm").querySelector('[name="sequence"]')?.value.trim();
  if (!sequence) return toast("Enter the sequence first.", "err");
  const data = await api(`/api/constructs/nt_rows?sequence=${encodeURIComponent(sequence)}`);
  renderNtGrid(data.nt_rows);
});

$("ntCopyBtn").addEventListener("click", async () => {
  const from = $("ntCopyFrom").value.trim();
  if (!from) return;
  try {
    const data = await api(`/api/constructs/nt_rows?disp_name=${encodeURIComponent(from)}`);
    renderNtGrid(data.nt_rows);
    toast(`Copied ${data.nt_rows.length} positions from ${from} (${data.source}).`, "ok");
  } catch (e) { toast(e.message, "err"); }
});

$("entityForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const record = {};
  $("entityFields").querySelectorAll("input,textarea").forEach((el) => { record[el.name] = el.value; });
  try {
    const data = await post("/api/entities", {
      entity_type: currentEntity.type,
      record,
      nt_rows: currentEntity.type === "construct" ? currentEntity.ntRows : null,
    });
    render(data.state);
    $("entityModal").close();
    toast("Created.", "ok");
    queueIndex += 1;
    const remaining = (S.validation.resolution_queue || []);
    if (remaining.length) {
      queue = remaining;
      setTimeout(() => openEntityWizard(remaining[0].entity_type, remaining[0].value), 220);
    }
  } catch (e) {
    $("entityError").textContent = e.message;
  }
});

document.querySelectorAll("[data-new-entity]").forEach((btn) =>
  btn.addEventListener("click", () => { queue = []; openEntityWizard(btn.dataset.newEntity, ""); })
);

/* ---------------------------------------------------------------- rows */

$("addRowBtn").addEventListener("click", async () => {
  const d = await post("/api/rows/add", { count: 1 });
  render(d.state);
});

$("delRowBtn").addEventListener("click", async () => {
  const uids = table.getSelectedData().map((r) => r.uid);
  if (!uids.length) return toast("Select some rows first.", "err");
  const d = await post("/api/rows/delete", { uids });
  render(d.state);
  toast(`${d.result.removed} row(s) removed.`, "ok");
});

/* ---------------------------------------------------------------- generate */

$("generateBtn").addEventListener("click", () => {
  const v = S.validation;
  const counts = Object.entries(S.staged || {}).map(([t, r]) => `${r.length} ${t.replace("_", " ")}${r.length === 1 ? "" : "s"}`);
  $("reviewBody").innerHTML = `
    <div class="review-block">
      <h4>About to create</h4>
      ${S.rows.length} sample${S.rows.length === 1 ? "" : "s"}${counts.length ? ", plus " + counts.join(", ") : ""}.
    </div>
    <div class="review-block">
      <h4>Into</h4>
      <code>${S.db_path}</code>
    </div>
    ${v.warning_count ? `<div class="review-block"><h4>Worth knowing</h4>${
      v.issues.filter((i) => i.severity === "warning").map((i) => i.message).join("<br>")
    }</div>` : ""}`;
  $("reviewModal").showModal();
});

$("confirmGenerateBtn").addEventListener("click", async () => {
  try {
    const data = await post("/api/generate", { mode: $("csvOnlyMode").checked ? "csv" : "hybrid" });
    render(data.state);
    const r = data.result;
    $("reviewBody").innerHTML = `
      <div class="review-block"><h4>Written</h4>${r.written.map((w) => `<code>${w}</code>`).join("")}</div>
      <div class="review-block"><h4>Run it</h4>${r.commands.map((c) => `<code>${c}</code>`).join("")}</div>`;
    toast(`Config files written for ${r.sample_count} samples.`, "ok");
  } catch (e) { toast(e.message, "err"); }
});

/* ------------------------------------------------------- reaction groups */

$("deriveGroupsBtn").addEventListener("click", () => deriveGroups(false));
$("relabelBtn").addEventListener("click", () => deriveGroups(true));

async function deriveGroups(relabel) {
  try {
    const data = await post("/api/groups/derive", {
      label_template: $("labelTemplate").value.trim() || null,
      relabel_all: relabel,
    });
    render(data.state);
    const r = data.result;
    let msg = `${r.groups} timecourse${r.groups === 1 ? "" : "s"} found`;
    if (r.times_written) msg += `, ${r.times_written} reaction time(s) filled`;
    const pending = (r.groups_without_ladder || []).length;
    if (pending) msg += `. ${pending} still need times.`;
    $("groupResult").textContent = msg;
    toast(msg, pending ? "info" : "ok");
  } catch (e) { toast(e.message, "err"); }
}

function ladderOptions(selectedId) {
  const opts = ['<option value="">— pick a ladder —</option>'];
  (S.ladders || []).forEach((l) => {
    const sel = l.id === selectedId ? " selected" : "";
    const suffix = /\(\s*\d+\s*tp\s*\)/i.test(l.name) ? "" : ` (${l.points.length} tp)`;
    opts.push(`<option value="${l.id}"${sel}>${l.name}${suffix}</option>`);
  });
  return opts.join("");
}

function renderGroups() {
  const body = $("groupRows");
  if (!body) return;
  const groups = S.groups || [];
  $("groupCount").textContent = groups.length || "";
  $("labelTemplate").value = S.label_template || "{temperature}_{n}";
  body.innerHTML = "";

  if (!groups.length) {
    body.innerHTML = `<tr><td colspan="6" class="hint">
      No timecourses yet. Fill in buffer, replicate, temperature and construct
      (from the name pattern or a bulk fill), then click “Find timecourses”.</td></tr>`;
    renderLadders();
    return;
  }

  groups.forEach((g) => {
    const tr = document.createElement("tr");
    if (g.needs_ladder || g.ladder_too_short) tr.className = "needs-ladder";

    const label = document.createElement("td");
    const input = document.createElement("input");
    input.className = "label-in";
    input.value = g.label || "";
    input.title = "reaction_group label written to the sheet";
    input.addEventListener("change", async () => {
      try {
        const d = await post("/api/groups/update", { key_id: g.key_id, label: input.value });
        render(d.state);
      } catch (e) { toast(e.message, "err"); }
    });
    label.appendChild(input);

    const cond = document.createElement("td");
    cond.className = "cond";
    cond.textContent = (S.group_key_columns || [])
      .map((c) => g.fields[c]).filter(Boolean).join(" · ");

    const count = document.createElement("td");
    count.textContent = g.row_count;

    const tps = document.createElement("td");
    tps.innerHTML = g.timepoints.length
      ? `<span class="tp-chips">${g.timepoints.map((t) => `<span class="tp-chip">tp${t}</span>`).join("")}</span>`
      : `<span class="hint">none in names</span>`;

    const times = document.createElement("td");
    const sel = document.createElement("select");
    sel.className = "input input-sm";
    sel.innerHTML = ladderOptions(g.ladder_id);
    sel.addEventListener("change", async () => {
      try {
        const d = await post("/api/groups/update", sel.value
          ? { key_id: g.key_id, ladder_id: sel.value }
          : { key_id: g.key_id, clear_ladder: true });
        render(d.state);
      } catch (e) { toast(e.message, "err"); }
    });
    times.appendChild(sel);
    const shown = document.createElement("div");
    shown.className = "times-cell" + (g.ladder_points.length ? "" : " empty");
    shown.textContent = g.ladder_points.length
      ? g.ladder_points.map(fmtSeconds).join(", ")
      : (g.timepoints.length ? "needs times" : "");
    if (g.ladder_too_short) {
      shown.classList.add("empty");
      shown.textContent += ` — only ${g.ladder_points.length} times for tp${g.max_timepoint}`;
    }
    times.appendChild(shown);

    const actions = document.createElement("td");
    const quick = document.createElement("button");
    quick.className = "btn btn-ghost btn-sm";
    quick.textContent = g.ladder_points.length ? "edit times" : "enter times";
    quick.onclick = () => promptLadder(g);
    actions.appendChild(quick);

    tr.append(label, cond, count, tps, times, actions);
    body.appendChild(tr);
  });
  renderLadders();
}

function fmtSeconds(value) {
  const n = Number(value);
  if (!isFinite(n)) return String(value);
  if (n >= 3600 && n % 3600 === 0) return `${n / 3600}h`;
  if (n >= 60 && n % 60 === 0) return `${n / 60}m`;
  return `${n}s`;
}

async function promptLadder(g) {
  const current = g.ladder_points.map((p) => p).join(", ");
  const text = window.prompt(
    `Reaction times for ${g.label} — ${g.timepoints.length ? "tp" + g.timepoints.join(", tp") : "no timepoints found"}\n\n` +
    `One per timepoint, in order. Bare numbers are seconds; you can also write 5min or 1h.`,
    current
  );
  if (text === null) return;
  try {
    const data = await post("/api/ladders", {
      ladder_id: g.ladder_id || null,
      name: g.ladder_id ? "" : g.label,
      text,
      assign_to: [g.key_id],
    });
    render(data.state);
    const r = data.result;
    toast(`${r.ladder.points.length} times saved${r.times_written ? `, ${r.times_written} row(s) updated` : ""}.`, "ok");
  } catch (e) { toast(e.message, "err"); }
}

function renderLadders() {
  const list = $("ladderList");
  if (!list) return;
  const ladders = S.ladders || [];
  list.innerHTML = "";
  if (!ladders.length) {
    list.innerHTML = `<span class="hint">None yet. Save one here and reuse it across groups — in a real run a handful of ladders usually cover everything.</span>`;
    return;
  }
  ladders.forEach((l) => {
    const used = (S.groups || []).filter((g) => g.ladder_id === l.id).length;
    const item = document.createElement("div");
    item.className = "ladder-item";
    item.innerHTML = `<b>${l.name}</b> <span class="hint">· ${used} group${used === 1 ? "" : "s"}</span>
      <span class="pts${l.monotonic ? "" : " bad"}">${l.points.map(fmtSeconds).join(", ")}${
        l.monotonic ? "" : "  ⚠ not increasing"}</span>`;
    const del = document.createElement("button");
    del.textContent = "×";
    del.title = "Delete ladder";
    del.onclick = async () => {
      try { render((await api(`/api/ladders/${l.id}`, { method: "DELETE" })).state); }
      catch (e) { toast(e.message, "err"); }
    };
    item.prepend(del);
    list.appendChild(item);
  });
}

$("newLadderBtn").addEventListener("click", async () => {
  try {
    const data = await post("/api/ladders", {
      name: $("newLadderName").value.trim() || "ladder",
      text: $("newLadderTimes").value,
    });
    render(data.state);
    $("newLadderName").value = ""; $("newLadderTimes").value = "";
    toast(`Ladder saved with ${data.result.ladder.points.length} timepoints.`, "ok");
  } catch (e) { toast(e.message, "err"); }
});

$("gridApplyBtn").addEventListener("click", async () => {
  try {
    const data = await post("/api/ladders/grid", {
      text: $("gridPaste").value,
      key_ids: (S.groups || []).map((g) => g.key_id),
    });
    render(data.state);
    const r = data.result;
    const msg = `${r.ladders_created} new ladder(s), ${r.ladders_reused} reused, ${r.times_written} row(s) filled.`;
    $("groupResult").textContent = msg;
    toast(msg, "ok");
    $("gridPaste").value = "";
  } catch (e) { toast(e.message, "err"); }
});

/* ---------------------------------------------------------------- init */

(async function init() {
  initTable();
  entitySchema = await api("/api/entities/schema");
  await loadTokens();
  try {
    const state = await api("/api/state");
    if (state.connected) {
      $("projectDir").value = state.project_dir || "";
      $("labelInput").value = state.label || "";
      $("connectStatus").textContent = "connected";
      $("connectStatus").className = "pill pill-ok";
      render(state);
    } else {
      openPanel("ingestPanel");
    }
  } catch (e) { /* not connected yet */ }
})();
