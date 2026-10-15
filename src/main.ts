import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { datasetName, escape as e, numberValue } from "./forms";
import "./style.css";

interface Settings { models_dir: string; storage_root: string; ollama_host: string; python: string }
interface Repo { id: string; path: string; commit_id: string; files: number; indexed: number }
interface Model { name: string; size: number }
interface Job { id: string; kind: string; status: string; latest: Record<string, unknown>; root: string }
interface Source { repo: string; path: string; start: string; end: string; content: string }
interface HFInfo { id: string; sha: string; files: { name: string; size?: number }[] }

const app = document.querySelector<HTMLDivElement>("#app")!;
app.innerHTML = `
  <aside><h1>Ollama<br><span>Workshop</span></h1><p class="muted">Local models · Your code</p>
  <nav>${["Models", "Repositories", "Datasets", "Training", "Discover", "Maintenance", "Settings"]
    .map(t => `<button data-tab="${t}">${t}</button>`).join("")}</nav>
  <p class="small">Repository contents stay local. Training uses a separate Python/CUDA environment.</p></aside>
  <main><header><div><p class="eyebrow">LOCAL LLM WORKBENCH</p><h2 id="title">Models</h2></div>
  <span id="connection">Connecting…</span></header>
  <div id="notice" role="status" aria-live="polite"></div><section id="page"></section>
  <section class="jobs"><div class="row"><h3>Background jobs</h3><button id="refresh-jobs">Refresh</button></div>
  <div id="jobs"></div><pre id="job-detail" tabindex="0"></pre></section></main>`;
const page = document.querySelector<HTMLElement>("#page")!;
let cfg: Settings;
let models: Model[] = [];
let repos: Repo[] = [];
let jobs: Job[] = [];
let currentTab = "Models";
let selectedJob = "";
let selectedRepo = "";

const value = (id: string) => (document.getElementById(id) as HTMLInputElement).value.trim();
const checked = (id: string) => (document.getElementById(id) as HTMLInputElement).checked;
const local = <T>(op: string, payload: Record<string, unknown> = {}) =>
  invoke<T>("local_service", { op, payload });
const api = <T>(action: string, payload: Record<string, unknown> = {}) =>
  invoke<T>("ollama", { action, payload });
const pretty = (data: unknown) => JSON.stringify(data, null, 2);
function notice(message: unknown, error = false) {
  const box = document.getElementById("notice")!;
  box.textContent = String(message);
  box.className = error ? "error" : "info";
}
async function guarded(work: () => Promise<void>, button?: HTMLButtonElement) {
  if (button) button.disabled = true;
  try { await work(); } catch (err) { notice(err, true); }
  finally { if (button) button.disabled = false; }
}
function bind(id: string, work: () => Promise<void>) {
  const button = document.getElementById(id) as HTMLButtonElement;
  button.addEventListener("click", () => { void guarded(work, button); });
}
function output(id: string, data: unknown) {
  document.getElementById(id)!.textContent = typeof data === "string" ? data : pretty(data);
}
function field(id: string, label: string, initial = "", type = "text") {
  return `<label>${e(label)}<input id="${id}" type="${type}" value="${e(initial)}"></label>`;
}
function modelOptions() {
  return models.map(m => `<option value="${e(m.name)}">${e(m.name)}</option>`).join("");
}
async function refreshModels() {
  try {
    const result = await api<{ models: Model[] }>("list");
    models = result.models || [];
    document.getElementById("connection")!.textContent = `Ollama · ${models.length} models`;
  } catch (err) {
    document.getElementById("connection")!.textContent = "Ollama unavailable";
    throw err;
  }
}
async function refreshJobs() {
  jobs = await invoke<Job[]>("list_jobs");
  renderJobs();
}
function renderJobs() {
  const container = document.getElementById("jobs")!;
  container.replaceChildren();
  if (!jobs.length) { container.textContent = "No jobs yet."; return; }
  for (const job of jobs) {
    const row = document.createElement("div");
    row.className = "job row";
    const title = document.createElement("button");
    title.textContent = `${job.kind} · ${job.status} · ${job.id.slice(0, 8)}`;
    title.onclick = () => { void guarded(async () => {
      selectedJob = job.id;
      output("job-detail", await invoke("job_details", { id: job.id }));
    }); };
    row.append(title);
    const progress = document.createElement("span");
    progress.className = "small";
    progress.textContent = pretty(job.latest);
    row.append(progress);
    if (["running", "cancelling"].includes(job.status)) {
      const cancel = document.createElement("button");
      cancel.textContent = "Cancel";
      cancel.disabled = job.status === "cancelling";
      cancel.onclick = () => { void guarded(async () => {
        await invoke("cancel_job", { id: job.id }); await refreshJobs();
      }, cancel); };
      row.append(cancel);
    }
    container.append(row);
  }
}
async function start(kind: string, payload: Record<string, unknown>) {
  const id = await invoke<string>("start_job", { kind, payload });
  notice(`Started ${kind} job ${id}. Progress appears below.`);
  await refreshJobs();
}

async function renderModels() {
  page.innerHTML = `<p>Models reported by the running Ollama daemon. The configured disk path is shown separately under Maintenance.</p>
    <div class="row">${field("pull-name", "Pull or update model", "qwen2.5-coder:1.5b")}
    <button id="pull">Pull model</button><button id="models-refresh">Refresh</button></div>
    <div id="model-list"></div><h3>Test inference</h3>
    <label>Model<select id="chat-model">${modelOptions()}</select></label>
    <label>Prompt<textarea id="prompt" rows="3">Explain what LoRA fine-tuning does.</textarea></label>
    <button id="chat">Generate locally</button><pre id="model-output"></pre>`;
  const list = document.getElementById("model-list")!;
  for (const model of models) {
    const row = document.createElement("div");
    row.className = "card row";
    const name = document.createElement("strong");
    name.textContent = `${model.name} · ${(model.size / 1024 ** 3).toFixed(2)} GiB`;
    row.append(name);
    for (const action of ["Inspect", "Update", "Delete"]) {
      const button = document.createElement("button");
      button.textContent = action;
      if (action === "Delete") button.className = "danger";
      button.onclick = () => { void guarded(async () => {
        if (action === "Inspect") output("model-output", await api("show", { model: model.name }));
        if (action === "Update") await start("pull", { model: model.name });
        if (action === "Delete" && window.confirm(`Delete Ollama model ${model.name}?`)) {
          await api("delete", { model: model.name }); await refreshModels(); await renderModels();
        }
      }, button); };
      row.append(button);
    }
    list.append(row);
  }
  bind("pull", async () => {
    if (!value("pull-name")) throw new Error("Enter a model name");
    if (window.confirm(`Download/update ${value("pull-name")}? This may download several GB.`))
      await start("pull", { model: value("pull-name") });
  });
  bind("models-refresh", async () => { await refreshModels(); await renderModels(); });
  bind("chat", async () => {
    const result = await api<{ message: { content: string } }>("chat", {
      model: value("chat-model"), messages: [{ role: "user", content: value("prompt") }],
    });
    output("model-output", result.message.content);
  });
}

async function renderRepositories() {
  repos = await local<Repo[]>("repositories");
  page.innerHTML = `<p>Index tracked text from selected Git roots. Ignored files, secrets, binaries, symlinks, and generated files are excluded.
    Filtering is heuristic: review source and licenses before creating a training dataset.</p>
    <div class="row">${field("repo-path", "Repository root", "/media/loufogle/Data/Repos/ollama-manager-gui")}
    <button id="ingest">Index / refresh</button></div>
    <label>Indexed repository<select id="repo-select"><option value="">Choose repository</option>
    ${repos.map(r => `<option value="${e(r.id)}">${e(r.path)} · ${r.files} files · ${e(r.commit_id.slice(0, 8))}</option>`).join("")}</select></label>
    <div class="row"><button id="files-load">Review indexed files</button>${field("repo-dataset", "New dataset name", "repo-source")}
    <button id="repo-export">Export reviewed files</button></div>
    <p class="small">Preview each file before checking it. Export creates raw-code language-modeling examples, not invented Q&amp;A.</p>
    <div class="split"><div id="repo-files" class="file-list"></div><pre id="repo-preview"></pre></div>
    <h3>Search or ask your code</h3>${field("repo-query", "Question / search terms", "")}
    <div class="row"><label>Ollama model<select id="repo-model">${modelOptions()}</select></label>
    <button id="repo-search">Search sources</button><button id="repo-chat">Ask locally with sources</button></div>
    <pre id="repo-result"></pre><pre id="ingest-report"></pre>`;
  (document.getElementById("repo-select") as HTMLSelectElement).value = selectedRepo;
  document.getElementById("repo-select")!.addEventListener("change", () => {
    selectedRepo = value("repo-select");
    document.getElementById("repo-files")!.replaceChildren();
    output("repo-preview", "");
  });
  bind("ingest", async () => {
    notice("Indexing tracked files. No repository code will be executed.");
    const result = await local<{ id: string }>("ingest", { path: value("repo-path") });
    selectedRepo = result.id;
    await renderRepositories();
    output("ingest-report", result);
  });
  bind("files-load", async () => {
    if (!value("repo-select")) throw new Error("Select an indexed repository");
    const files = await local<{ path: string }[]>("repo_files", { repo_id: value("repo-select") });
    const list = document.getElementById("repo-files")!;
    list.replaceChildren();
    for (const file of files) {
      const row = document.createElement("div");
      row.className = "row";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox"; checkbox.value = file.path; checkbox.disabled = true;
      const preview = document.createElement("button");
      preview.textContent = file.path;
      preview.onclick = () => { void guarded(async () => {
        output("repo-preview", await local("repo_preview", { repo_id: value("repo-select"), path: file.path }));
        checkbox.disabled = false;
      }, preview); };
      row.append(checkbox, preview); list.append(row);
    }
  });
  bind("repo-export", async () => {
    const approved_paths = Array.from(document.querySelectorAll<HTMLInputElement>("#repo-files input:checked")).map(c => c.value);
    if (!approved_paths.length) throw new Error("Preview and select source files first");
    if (!window.confirm(`Export ${approved_paths.length} reviewed files? Confirm you have permission to train on them.`)) return;
    const result = await local("repo_export", {
      repo_id: value("repo-select"), name: datasetName(value("repo-dataset")), approved_paths,
    });
    output("repo-result", result); notice("Dataset exported. Open Datasets to review it before training.");
  });
  bind("repo-search", async () => {
    output("repo-result", await local<Source[]>("search", { query: value("repo-query"), repo_id: value("repo-select") }));
  });
  bind("repo-chat", async () => {
    output("repo-result", await invoke("repository_chat", {
      model: value("repo-model"), question: value("repo-query"), repoId: value("repo-select"),
    }));
  });
}

async function renderDatasets() {
  const datasets = await local<string[]>("dataset_list");
  page.innerHTML = `<p>Import JSON/JSONL with text, instruction/input/output, or messages. Saves create a new immutable dataset.
    Repository exports retain file groups for leak-resistant validation splits.</p>
    <div class="row"><label>Saved dataset<select id="dataset-select"><option value="">Choose dataset</option>
    ${datasets.map(p => `<option>${e(p)}</option>`).join("")}</select></label><button id="dataset-load">Load</button>
    <label>Import file<input id="dataset-import" type="file" accept=".json,.jsonl"></label></div>
    ${field("dataset-name", "Save as new dataset", "training-data")}
    <label>Dataset editor<textarea id="dataset-text" rows="16" spellcheck="false"></textarea></label>
    <div class="row"><button id="dataset-validate">Validate</button><button id="dataset-save">Save new dataset</button></div>
    <pre id="dataset-output"></pre>`;
  bind("dataset-load", async () => {
    (document.getElementById("dataset-text") as HTMLTextAreaElement).value =
      await local<string>("dataset_read", { path: value("dataset-select") });
  });
  document.getElementById("dataset-import")!.addEventListener("change", () => { void guarded(async () => {
    const file = (document.getElementById("dataset-import") as HTMLInputElement).files?.[0];
    if (!file) return;
    if (file.size > 32 * 1024 * 1024) throw new Error("Editor import limit: 32 MiB");
    (document.getElementById("dataset-text") as HTMLTextAreaElement).value = await file.text();
  }); });
  bind("dataset-validate", async () => {
    output("dataset-output", await local("dataset_validate", { text: value("dataset-text") }));
  });
  bind("dataset-save", async () => {
    output("dataset-output", await local("dataset_save", {
      name: datasetName(value("dataset-name")), text: value("dataset-text"),
    }));
  });
}

async function renderTraining() {
  const datasets = await local<string[]>("dataset_list");
  page.innerHTML = `<p>LoRA/QLoRA trains adapters against original Transformers weights—not Ollama blobs.
    Start with a 0.5B–1.5B model and a short run on your 12 GB GPU. No model downloads occur during training.</p>
    <button id="preflight">Check Python / CUDA environment</button><pre id="preflight-output"></pre>
    <div class="grid">
    ${field("train-model", "Local Transformers snapshot (absolute path)")}
    <label>Dataset<select id="train-dataset">${datasets.map(p => `<option>${e(p)}</option>`).join("")}</select></label>
    <label>Method<select id="train-method"><option value="qlora">QLoRA · NF4 4-bit</option><option value="lora">LoRA · BF16/FP16</option></select></label>
    ${field("rank", "LoRA rank", "8", "number")}
    ${field("sequence_length", "Sequence length", "512", "number")}
    ${field("batch_size", "Micro-batch size", "1", "number")}
    ${field("gradient_accumulation", "Gradient accumulation", "8", "number")}
    ${field("learning_rate", "Learning rate", "0.0002", "number")}
    ${field("epochs", "Epochs (step limit takes precedence)", "1", "number")}
    ${field("max_steps", "Maximum optimizer steps (-1 uses epochs)", "20", "number")}
    ${field("save_steps", "Checkpoint / evaluation interval", "10", "number")}
    ${field("validation_fraction", "Validation fraction (grouped by source file)", "0.1", "number")}
    ${field("resume", "Resume checkpoint (workspace-relative, optional)")}
    </div>
    <label class="check"><input type="checkbox" id="reviewed">I reviewed the dataset, its sources/licenses, and the training settings.</label>
    <button id="train-start">Start local fine-tuning</button>
    <h3>Export a completed adapter</h3>
    <div class="row">${field("adapter-job", "Completed training job UUID")}
    ${field("adapter-dest", "New absolute export directory")}<button id="adapter-export">Export adapter</button></div>
    <pre id="training-output"></pre>`;
  bind("preflight", async () => { output("preflight-output", await local("preflight")); });
  bind("train-start", async () => {
    if (!checked("reviewed")) throw new Error("Review and approve the dataset first");
    const report = await local<{ ready: boolean }>("preflight");
    output("preflight-output", report);
    if (!report.ready) throw new Error("Training environment is not ready. See requirements in README and select its Python under Settings.");
    const payload: Record<string, unknown> = {
      model: value("train-model"), dataset: value("train-dataset"), method: value("train-method"),
      reviewed: true, resume: value("resume") || null,
    };
    for (const [id, low, high, integer] of [
      ["rank", 1, 256, true], ["sequence_length", 32, 8192, true], ["batch_size", 1, 16, true],
      ["gradient_accumulation", 1, 256, true], ["learning_rate", 1e-8, 0.1, false],
      ["epochs", 0.1, 100, false], ["max_steps", -1, 1000000, true],
      ["save_steps", 1, 100000, true], ["validation_fraction", 0, 0.49, false],
    ] as const) payload[id] = numberValue(value(id), low, high, integer);
    if (payload.max_steps === 0) throw new Error("Use -1 for epoch-based training or a positive step count");
    if (window.confirm("Start GPU training? This consumes GPU memory, disk space, and power."))
      await start("train", payload);
  });
  bind("adapter-export", async () => {
    output("training-output", await local("adapter_export", { job_id: value("adapter-job"), destination: value("adapter-dest") }));
  });
}

async function renderDiscover() {
  page.innerHTML = `<p>Discover Hugging Face repositories. Inspect files and sizes before explicitly starting a download.
    Training snapshots exclude executable model code and pickle weights.</p>
    <div class="row">${field("hf-query", "Search models", "Qwen2.5-Coder-0.5B")}<button id="hf-search">Search Hugging Face</button></div>
    <div id="hf-results"></div><div class="row">${field("hf-repo", "Repository ID", "Qwen/Qwen2.5-Coder-0.5B")}
    <button id="hf-info">Inspect files</button></div>
    <label>GGUF file<select id="hf-file"></select></label>
    <div class="row"><button id="hf-snapshot">Download training snapshot</button><button id="hf-gguf">Download selected GGUF</button></div>
    <pre id="hf-output"></pre>`;
  let info: HFInfo | undefined;
  bind("hf-search", async () => {
    const results = await local<{ id: string; downloads: number }[]>("hf_search", { query: value("hf-query") });
    const list = document.getElementById("hf-results")!; list.replaceChildren();
    for (const result of results) {
      const button = document.createElement("button");
      button.textContent = `${result.id} · ${result.downloads.toLocaleString()} downloads`;
      button.onclick = () => { (document.getElementById("hf-repo") as HTMLInputElement).value = result.id; };
      list.append(button);
    }
  });
  bind("hf-info", async () => {
    info = await local<HFInfo>("hf_info", { repo: value("hf-repo") });
    output("hf-output", info);
    document.getElementById("hf-file")!.innerHTML = info.files.filter(f => f.name.endsWith(".gguf"))
      .map(f => `<option value="${e(f.name)}">${e(f.name)} · ${((f.size || 0) / 1024 ** 3).toFixed(2)} GiB</option>`).join("");
  });
  for (const [id, kind] of [["hf-snapshot", "snapshot"], ["hf-gguf", "gguf"]]) bind(id, async () => {
    if (!info || info.id !== value("hf-repo")) throw new Error("Inspect this repository first");
    if (kind === "gguf" && !value("hf-file")) throw new Error("Select a GGUF file");
    if (window.confirm(`Download ${kind} from ${info.id}? Review the file sizes above before continuing.`))
      await start("download", { repo: info.id, revision: info.sha, kind, file: value("hf-file") });
  });
}

async function renderMaintenance() {
  page.innerHTML = `<p>Read-only inspection of <code>${e(cfg.models_dir)}</code>.</p>
    <p class="warning">This store has newer metadata directories. Destructive cleanup is disabled to avoid removing live or referenced weights.
    Selecting a folder does not change the running Ollama daemon's storage configuration.</p>
    <div class="row"><button id="scan">Scan references</button><button id="verify">Verify SHA256 (may take a while)</button>
    <button id="repair">Repair installed models</button></div>
    <pre id="maintenance-output"></pre>`;
  for (const [id, verify] of [["scan", false], ["verify", true]] as const) bind(id, async () => {
    notice(verify ? "Verifying model blobs; this reads all model data." : "Scanning manifests…");
    output("maintenance-output", await local("scan", { path: cfg.models_dir, verify }));
  });
  bind("repair", async () => {
    if (window.confirm("Re-pull every model reported by Ollama? This may download missing model data."))
      await start("repair", {});
  });
}

async function renderSettings() {
  page.innerHTML = `<p>The workspace stores the local search index, datasets, downloaded snapshots, adapters, and job history.
    It must be separate from your Ollama model store.</p>
    ${field("settings-models", "Ollama model store", cfg.models_dir)}
    ${field("settings-root", "Workspace root", cfg.storage_root)}
    ${field("settings-host", "Local Ollama origin", cfg.ollama_host)}
    ${field("settings-python", "Python executable (training virtual environment)", cfg.python)}
    <button id="settings-save">Save settings</button>
    <p class="small">Use environment variables HF_TOKEN, HF_API_TOKEN, or HUGGINGFACE_API_TOKEN for private/gated models.
    Tokens are never stored in this UI. Changing workspace switches the visible dataset/index storage.</p>`;
  bind("settings-save", async () => {
    const next: Settings = { models_dir: value("settings-models"), storage_root: value("settings-root"),
      ollama_host: value("settings-host"), python: value("settings-python") };
    await invoke("put_settings", { value: next }); cfg = next;
    notice("Settings saved."); await refreshModels();
  });
}

const tabs: Record<string, () => Promise<void>> = {
  Models: renderModels, Repositories: renderRepositories, Datasets: renderDatasets,
  Training: renderTraining, Discover: renderDiscover, Maintenance: renderMaintenance, Settings: renderSettings,
};
async function navigate(tab: string) {
  currentTab = tab;
  document.getElementById("title")!.textContent = tab;
  document.querySelectorAll<HTMLButtonElement>("[data-tab]").forEach(b => b.classList.toggle("active", b.dataset.tab === tab));
  page.textContent = "Loading…";
  await tabs[tab]();
}
document.querySelectorAll<HTMLButtonElement>("[data-tab]").forEach(button => {
  button.onclick = () => { void guarded(() => navigate(button.dataset.tab!)); };
});
bind("refresh-jobs", refreshJobs);

void guarded(async () => {
  cfg = await invoke<Settings>("get_settings");
  await listen<Job>("job-update", event => {
    const i = jobs.findIndex(j => j.id === event.payload.id);
    if (i >= 0) jobs[i] = event.payload; else jobs.push(event.payload);
    renderJobs();
    if (selectedJob === event.payload.id) void guarded(async () => {
      output("job-detail", await invoke("job_details", { id: selectedJob }));
    });
    if (event.payload.status === "failed") notice(event.payload.latest.error || "Job failed", true);
  });
  await refreshJobs();
  try { await refreshModels(); } catch (err) { notice(err, true); }
  await navigate(currentTab);
});
