/* Ultra Fast Image Gen — canvas-first frontend. No frameworks, no build step. */

"use strict";

const $ = (id) => document.getElementById(id);

const els = {
  stage: $("stage"), empty: $("empty"), shot: $("shot"), shotImg: $("shot-img"),
  shotSeed: $("shot-seed"), copySeed: $("copy-seed"), reuseSeed: $("reuse-seed"), download: $("download-link"),
  busy: $("busy"), busyStage: $("busy-stage"), busyDetail: $("busy-detail"),
  rail: $("rail"), error: $("error"), progress: $("progress-bar"),
  chipModel: $("chip-model"), chipSize: $("chip-size"), chipTune: $("chip-tune"),
  chipBatch: $("chip-batch"), chipMore: $("chip-more"),
  popModel: $("pop-model"), popSize: $("pop-size"), popTune: $("pop-tune"),
  popBatch: $("pop-batch"), popMore: $("pop-more"), modelList: $("model-list"),
  presets: $("presets"), width: $("width"), height: $("height"), swap: $("swap-dims"),
  steps: $("steps"), stepsOut: $("steps-out"), guidance: $("guidance"), guidanceOut: $("guidance-out"),
  seed: $("seed"), count: $("count"), countOut: $("count-out"),
  animaBlock: $("anima-block"), animaPresets: $("anima-presets"),
  loraBlock: $("lora-block"), loraPath: $("lora-path"),
  loraStrength: $("lora-strength"), loraStrengthOut: $("lora-strength-out"),
  autoSave: $("auto-save"), outputDir: $("output-dir"), openFolder: $("open-folder"),
  addRefs: $("add-refs"), fileInput: $("file-input"), refTray: $("ref-tray"),
  prompt: $("prompt"), generate: $("generate"),
  deviceChip: $("device-chip"), storageToggle: $("storage-toggle"), storageDrawer: $("storage-drawer"),
  storageClose: $("storage-close"), storageList: $("storage-list"),
  storageTotal: $("storage-total"), storageMsg: $("storage-msg"),
};

let MODELS = [];
let modelId = null;
let refImages = [];
let lastSeed = null;
let pollTimer = null;
let pollFails = 0;
let activeJob = null;
let animaPreset = "Balanced";

const SETTINGS_KEY = "ufig-settings-v2";
const POPS = [
  ["chipModel", "popModel"], ["chipSize", "popSize"], ["chipTune", "popTune"],
  ["chipBatch", "popBatch"], ["chipMore", "popMore"],
];

/* ---------------- helpers ---------------- */

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* ignore */ }
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

const model = () => MODELS.find((m) => m.id === modelId);

function showError(msg) {
  els.error.textContent = msg;
  els.error.hidden = !msg;
}

/* ---------------- settings ---------------- */

function saveSettings() {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify({
    model: modelId, prompt: els.prompt.value,
    width: els.width.value, height: els.height.value,
    steps: els.steps.value, guidance: els.guidance.value, count: els.count.value,
    autoSave: els.autoSave.checked, outputDir: els.outputDir.value,
    loraPath: els.loraPath.value, loraStrength: els.loraStrength.value, animaPreset,
  }));
}

function loadSettings() {
  try { return JSON.parse(localStorage.getItem(SETTINGS_KEY)) || null; }
  catch { return null; }
}

/* ---------------- popovers ---------------- */

function closePops(except) {
  for (const [chip, pop] of POPS) {
    if (pop === except) continue;
    els[pop].hidden = true;
    els[chip].setAttribute("aria-expanded", "false");
  }
}

function bindPop(chipKey, popKey) {
  els[chipKey].addEventListener("click", () => {
    const open = !els[popKey].hidden;
    closePops();
    els[popKey].hidden = open;
    els[chipKey].setAttribute("aria-expanded", String(!open));
  });
}

/* ---------------- chips / state sync ---------------- */

function syncChips() {
  const m = model();
  if (!m) return;
  els.chipModel.textContent = "";
  if (m.setup_required) {
    const w = document.createElement("span");
    w.className = "warn";
    w.textContent = "⚠ ";
    els.chipModel.appendChild(w);
  }
  els.chipModel.appendChild(document.createTextNode(m.label));
  els.chipSize.textContent = `${els.width.value}×${els.height.value}`;
  const g = Number(els.guidance.value);
  els.chipTune.textContent = `${els.steps.value} steps${g ? ` · cfg ${g}` : ""}${els.seed.value ? ` · seed ${els.seed.value}` : ""}`;
  els.chipBatch.textContent = `×${els.count.value}`;

  els.stepsOut.textContent = els.steps.value;
  els.guidanceOut.textContent = els.guidance.value;
  els.countOut.textContent = els.count.value;
  els.loraStrengthOut.textContent = Number(els.loraStrength.value).toFixed(2);

  for (const b of els.presets.querySelectorAll(".preset")) {
    const on = b.dataset.size === els.width.value && b.dataset.size === els.height.value;
    b.setAttribute("aria-pressed", String(on));
  }
  els.addRefs.hidden = !m.img2img;
  if (!m.img2img && refImages.length) { refImages = []; renderRefs(); }
  els.animaBlock.hidden = m.id !== "anima";
  els.loraBlock.hidden = !m.lora;
}

function applyModelDefaults(m) {
  els.width.value = m.defaults.width;
  els.height.value = m.defaults.height;
  els.steps.value = m.defaults.steps;
  els.guidance.value = m.defaults.guidance;
}

function renderModelList() {
  els.modelList.textContent = "";
  for (const m of MODELS) {
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.type = "button";
    b.setAttribute("aria-pressed", String(m.id === modelId));

    const head = document.createElement("div");
    head.className = "m-head";
    const name = document.createElement("span");
    name.className = "m-name";
    name.textContent = m.label;
    head.appendChild(name);
    if (m.tag) {
      const tag = document.createElement("span");
      tag.className = "m-tag";
      tag.textContent = m.tag;
      head.appendChild(tag);
    }

    const note = document.createElement("span");
    note.className = "m-note";
    if (m.setup_required) {
      const warn = document.createElement("span");
      warn.className = "warn";
      warn.textContent = "Setup required — run scripts/setup_mflux_hs.sh. ";
      note.appendChild(warn);
    }
    note.appendChild(document.createTextNode(m.note || ""));
    note.title = m.note || "";

    b.append(head, note);
    b.addEventListener("click", () => {
      modelId = m.id;
      applyModelDefaults(m);
      renderModelList();
      syncChips();
      saveSettings();
      closePops();
    });
    li.appendChild(b);
    els.modelList.appendChild(li);
  }
}

function renderAnimaPresets() {
  const m = MODELS.find((x) => x.id === "anima");
  els.animaPresets.textContent = "";
  for (const name of m?.anima_presets || []) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "preset";
    b.textContent = name;
    b.setAttribute("aria-pressed", String(name === animaPreset));
    b.addEventListener("click", () => { animaPreset = name; renderAnimaPresets(); saveSettings(); });
    els.animaPresets.appendChild(b);
  }
}

/* ---------------- reference images ---------------- */

function renderRefs() {
  els.refTray.textContent = "";
  els.refTray.hidden = refImages.length === 0;
  refImages.forEach((src, i) => {
    const wrap = document.createElement("div");
    wrap.className = "thumb";
    const img = document.createElement("img");
    img.src = src;
    img.alt = `Reference ${i + 1}`;
    const rm = document.createElement("button");
    rm.className = "remove";
    rm.type = "button";
    rm.textContent = "×";
    rm.setAttribute("aria-label", `Remove reference ${i + 1}`);
    rm.addEventListener("click", () => { refImages.splice(i, 1); renderRefs(); });
    wrap.append(img, rm);
    els.refTray.appendChild(wrap);
  });
}

function addFiles(files) {
  if (!model()?.img2img) return;
  for (const f of files) {
    if (!f.type.startsWith("image/")) continue;
    if (refImages.length >= 6) break;
    const reader = new FileReader();
    reader.onload = () => { refImages.push(reader.result); renderRefs(); };
    reader.readAsDataURL(f);
  }
}

/* ---------------- generation ---------------- */

function setBusy(busy) {
  els.generate.disabled = busy;
  els.busy.hidden = !busy;
  if (!busy) {
    els.progress.classList.remove("indeterminate");
    els.progress.style.width = "0%";
  }
}

async function startGeneration() {
  // generate is disabled from setBusy(true) until the job reaches a terminal
  // state, so this also guards the Enter-during-POST race
  if (activeJob || els.generate.disabled) return;
  showError("");
  const m = model();
  if (!m) return;
  if (!els.prompt.value.trim()) { showError("Write a prompt first."); els.prompt.focus(); return; }

  const body = {
    model: m.id,
    prompt: els.prompt.value.trim(),
    width: Number(els.width.value),
    height: Number(els.height.value),
    steps: Number(els.steps.value),
    guidance: Number(els.guidance.value),
    seed: els.seed.value === "" ? null : Number(els.seed.value),
    count: Number(els.count.value),
    auto_save: els.autoSave.checked,
    output_dir: els.outputDir.value || null,
  };
  if (m.img2img && refImages.length) body.input_images = refImages;
  if (m.lora && els.loraPath.value) {
    body.lora_path = els.loraPath.value;
    body.lora_strength = Number(els.loraStrength.value);
  }
  if (m.id === "anima") body.anima_preset = animaPreset;

  closePops();
  setBusy(true);
  els.busyStage.textContent = "Queued";
  els.busyDetail.textContent = "";
  try {
    const { job_id } = await api("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    activeJob = job_id;
    pollFails = 0;
    clearInterval(pollTimer);
    pollTimer = setInterval(() => pollJob(job_id), 600);
  } catch (e) {
    setBusy(false);
    showError(e.message);
  }
}

function abandonJob(message) {
  clearInterval(pollTimer);
  activeJob = null;
  setBusy(false);
  showError(message);
}

async function pollJob(jobId) {
  let job;
  try {
    job = await api(`/api/jobs/${jobId}`);
    pollFails = 0;
  } catch (e) {
    if (jobId !== activeJob) return;
    // A 404 means the server restarted and forgot the job; transient network
    // errors get a grace window (~12s) before giving up.
    if (e.status === 404) abandonJob("The server restarted and lost this job — try again.");
    else if (++pollFails > 20) abandonJob("Lost contact with the server — is it still running?");
    return;
  }
  if (jobId !== activeJob) return;

  els.busyStage.textContent = job.queue_position > 0
    ? `Queued (#${job.queue_position + 1})`
    : job.stage;
  els.busyDetail.textContent = job.elapsed != null ? `${job.elapsed.toFixed(0)}s` : "";

  if (job.progress == null && job.status === "running") {
    els.progress.style.width = ""; // leftover inline width would hide the slide animation
    els.progress.classList.add("indeterminate");
  } else if (job.progress != null) {
    els.progress.classList.remove("indeterminate");
    els.progress.style.width = `${Math.round(job.progress * 100)}%`;
  }

  if (job.images.length) addToRail(job.images);

  if (["done", "error", "cancelled"].includes(job.status)) {
    clearInterval(pollTimer);
    activeJob = null;
    setBusy(false);
    if (job.status === "error") showError(job.error || "Generation failed.");
    if (job.images.length) {
      lastSeed = job.images[job.images.length - 1].seed;
      selectImage(job.images[job.images.length - 1]);
    }
  }
}

/* ---------------- rail / viewer ---------------- */

const seenUrls = new Set();

function addToRail(images) {
  for (const img of images) {
    if (seenUrls.has(img.url)) continue;
    seenUrls.add(img.url);
    const b = document.createElement("button");
    b.type = "button";
    b.setAttribute("aria-label", `View image with seed ${img.seed}`);
    const t = document.createElement("img");
    t.src = img.url;
    t.alt = "";
    b.appendChild(t);
    b.addEventListener("click", () => selectImage(img, b));
    els.rail.prepend(b);
    if (els.rail.children.length > 80) els.rail.lastChild.remove();
  }
}

function selectImage(img, btn) {
  els.empty.hidden = true;
  els.shot.hidden = false;
  els.shotImg.src = img.url;
  els.shotSeed.textContent = `seed ${img.seed}${img.saved ? " · saved" : ""}`;
  els.shotSeed.dataset.seed = img.seed;
  els.shotSeed.title = img.saved || "";
  els.download.href = img.url;
  els.download.setAttribute("download", `ufig-${img.seed}.png`);
  for (const child of els.rail.children) child.classList.remove("active");
  if (btn) btn.classList.add("active");
  else {
    const match = [...els.rail.children].find((c) => c.querySelector("img")?.src.endsWith(img.url));
    match?.classList.add("active");
  }
}

/* ---------------- storage drawer ---------------- */

async function refreshStorage() {
  const data = await api("/api/storage");
  els.storageTotal.textContent = data.models.length
    ? `Total: ${data.total_str}`
    : "Nothing downloaded yet — models download on first use.";
  els.storageList.textContent = "";
  for (const m of data.models) {
    const li = document.createElement("li");
    const name = document.createElement("span");
    name.className = "s-name";
    name.title = m.repo_id;
    name.textContent = m.name;
    const size = document.createElement("span");
    size.className = "s-size tab-nums";
    size.textContent = m.size_str;
    const del = document.createElement("button");
    del.className = "icon-link";
    del.type = "button";
    del.textContent = "Delete";
    let armed = false;
    del.addEventListener("click", async () => {
      if (!armed) {            // inline confirm: first click arms, second deletes
        armed = true;
        del.textContent = "Confirm?";
        del.classList.add("danger");
        setTimeout(() => { armed = false; del.textContent = "Delete"; del.classList.remove("danger"); }, 3000);
        return;
      }
      del.disabled = true;
      try {
        const r = await api("/api/storage/delete", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ key: m.key }),
        });
        els.storageMsg.textContent = r.message;
      } catch (e) {
        els.storageMsg.textContent = e.message;
      }
      refreshStorage();
    });
    li.append(name, size, del);
    els.storageList.appendChild(li);
  }
}

/* ---------------- init ---------------- */

function autoGrow() {
  els.prompt.style.height = "auto";
  els.prompt.style.height = `${Math.min(els.prompt.scrollHeight, 130)}px`;
}

async function init() {
  const [status, models] = await Promise.all([api("/api/status"), api("/api/models")]);
  MODELS = models;
  els.deviceChip.textContent = status.devices[0] || "cpu";
  els.outputDir.value = status.default_output_dir;

  const saved = loadSettings();
  modelId = saved && MODELS.some((m) => m.id === saved.model) ? saved.model : MODELS[0].id;
  applyModelDefaults(model());

  if (saved) {
    els.prompt.value = saved.prompt || "";
    if (saved.width) els.width.value = saved.width;
    if (saved.height) els.height.value = saved.height;
    if (saved.steps) els.steps.value = saved.steps;
    if (saved.guidance != null) els.guidance.value = saved.guidance;
    if (saved.count) els.count.value = saved.count;
    els.autoSave.checked = !!saved.autoSave;
    if (saved.outputDir) els.outputDir.value = saved.outputDir;
    if (saved.loraPath) els.loraPath.value = saved.loraPath;
    if (saved.loraStrength) els.loraStrength.value = saved.loraStrength;
    if (saved.animaPreset) animaPreset = saved.animaPreset;
  }

  renderModelList();
  renderAnimaPresets();
  syncChips();
  autoGrow();

  for (const [chip, pop] of POPS) bindPop(chip, pop);
  document.addEventListener("click", (e) => {
    if (!e.target.isConnected) return; // re-rendered controls (e.g. presets) detach mid-bubble
    if (!e.target.closest(".pop") && !e.target.closest(".chip")) closePops();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { closePops(); els.storageDrawer.hidden = true; }
  });

  for (const el of [els.steps, els.guidance, els.count, els.loraStrength, els.width, els.height, els.seed]) {
    el.addEventListener("input", () => { syncChips(); saveSettings(); });
  }
  for (const el of [els.outputDir, els.loraPath]) el.addEventListener("change", saveSettings);
  els.autoSave.addEventListener("change", saveSettings);
  els.prompt.addEventListener("input", () => { autoGrow(); });
  els.prompt.addEventListener("change", saveSettings);

  els.presets.addEventListener("click", (e) => {
    const b = e.target.closest(".preset");
    if (!b) return;
    els.width.value = b.dataset.size;
    els.height.value = b.dataset.size;
    syncChips();
    saveSettings();
  });

  els.swap.addEventListener("click", () => {
    [els.width.value, els.height.value] = [els.height.value, els.width.value];
    syncChips();
    saveSettings();
  });

  els.reuseSeed.addEventListener("click", () => {
    const s = els.shotSeed.dataset.seed;
    if (s) { els.seed.value = s; syncChips(); saveSettings(); }
  });

  els.copySeed.addEventListener("click", () => {
    navigator.clipboard?.writeText(els.shotSeed.dataset.seed || "");
  });

  els.generate.addEventListener("click", startGeneration);
  els.prompt.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); startGeneration(); }
  });

  for (const s of document.querySelectorAll(".sample")) {
    s.addEventListener("click", () => {
      els.prompt.value = s.textContent;
      autoGrow();
      els.prompt.focus();
      saveSettings();
    });
  }

  // reference images: button, file picker, drag-drop onto the page
  els.addRefs.addEventListener("click", () => els.fileInput.click());
  els.fileInput.addEventListener("change", () => { addFiles(els.fileInput.files); els.fileInput.value = ""; });
  document.addEventListener("dragover", (e) => {
    e.preventDefault();
    if (model()?.img2img) els.addRefs.classList.add("dragover");
  });
  document.addEventListener("dragleave", (e) => {
    if (e.relatedTarget === null) els.addRefs.classList.remove("dragover");
  });
  document.addEventListener("drop", (e) => {
    e.preventDefault();
    els.addRefs.classList.remove("dragover");
    if (e.dataTransfer?.files?.length) addFiles(e.dataTransfer.files);
  });

  els.openFolder.addEventListener("click", () =>
    api("/api/open_folder", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dir: els.outputDir.value || null }),
    }).catch((e) => showError(e.message))
  );

  els.storageToggle.addEventListener("click", () => {
    els.storageDrawer.hidden = !els.storageDrawer.hidden;
    if (!els.storageDrawer.hidden) refreshStorage().catch((e) => { els.storageMsg.textContent = e.message; });
  });
  els.storageClose.addEventListener("click", () => { els.storageDrawer.hidden = true; });
}

init().catch((e) => showError(`Failed to load: ${e.message}`));
