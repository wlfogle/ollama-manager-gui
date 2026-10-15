(async () => {
  if (window.__WORKSHOP_SMOKE_RUNNING__) return;
  window.__WORKSHOP_SMOKE_RUNNING__ = true;
  const invoke = (command, args = {}) => window.__TAURI_INTERNALS__.invoke(command, args);
  const results = [];
  const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
  async function wait(predicate, message, timeout = 30000) {
    const start = Date.now();
    while (!predicate()) {
      if (Date.now() - start > timeout) throw new Error(message + ": " + document.body.innerText);
      await sleep(100);
    }
  }
  function fill(id, text) {
    const element = document.getElementById(id);
    if (!element) throw new Error("Missing input: " + id);
    element.value = text;
  }
  function click(id) {
    const element = document.getElementById(id);
    if (!element) throw new Error("Missing control: " + id);
    element.click();
  }
  async function tab(name, expected) {
    document.querySelector(`[data-tab="${name}"]`).click();
    await wait(() => document.getElementById(expected), "Could not open " + name);
  }
  try {
    window.confirm = () => true; // Only this opt-in test build auto-confirms fixture actions.
    await wait(() => document.getElementById("chat"), "Initial UI did not load");
    await wait(() => document.getElementById("connection").textContent.includes("models"), "Ollama IPC failed");
    results.push("native WebView loaded frontend and real Ollama model list");
    const cfg = await invoke("get_settings");
    const fixture = await invoke("local_service", { op: "smoke_fixture", payload: {} });
    await tab("Settings", "settings-save");
    fill("settings-python", fixture.python);
    click("settings-save");
    await wait(() => document.getElementById("notice").textContent === "Settings saved.", "Settings IPC failed");
    results.push("settings validation and persistence");
    await tab("Repositories", "ingest");
    fill("repo-path", fixture.repo);
    click("ingest");
    await wait(() => document.querySelector("#repo-select option[value]:not([value=''])"), "Repository ingestion failed");
    click("files-load");
    await wait(() => document.querySelector("#repo-files button"), "Source list failed");
    document.querySelector("#repo-files button").click();
    await wait(() => document.querySelector("#repo-files input:not(:disabled)"), "Source preview failed");
    document.querySelector("#repo-files input").checked = true;
    fill("repo-dataset", "desktop-repo");
    click("repo-export");
    await wait(() => document.getElementById("repo-result").textContent.includes("sha256"), "Dataset export failed");
    fill("repo-query", "workshop_fixture");
    click("repo-search");
    await wait(() => document.getElementById("repo-result").textContent.includes("workshop_fixture"), "Code retrieval failed");
    results.push("repository index, source preview, reviewed export, and local retrieval");
    fill("repo-query", "What does workshop_fixture return? Answer briefly and cite the source.");
    click("repo-chat");
    await wait(() => document.getElementById("repo-result").textContent.includes('"answer"'), "Grounded Ollama inference failed", 180000);
    results.push("real local Ollama grounded answer with retrieved source records");
    await tab("Datasets", "dataset-load");
    document.getElementById("dataset-select").value = "datasets/desktop-repo.jsonl";
    click("dataset-load");
    await wait(() => document.getElementById("dataset-text").value.includes("workshop_fixture"), "Dataset editor load failed");
    click("dataset-validate");
    await wait(() => document.getElementById("dataset-output").textContent.includes("rows"), "Dataset validation failed");
    results.push("dataset load, editing surface, and validation IPC");
    await tab("Training", "preflight");
    click("preflight");
    await wait(() => document.getElementById("preflight-output").textContent.includes('"ready": true'), "CUDA preflight failed", 60000);
    fill("train-model", fixture.model);
    document.getElementById("train-dataset").value = "datasets/tiny.jsonl";
    fill("max_steps", "2");
    fill("save_steps", "1");
    fill("sequence_length", "32");
    fill("gradient_accumulation", "1");
    document.getElementById("reviewed").checked = true;
    click("train-start");
    await wait(() => document.getElementById("jobs").textContent.includes("train · complete"), "Supervised QLoRA job did not complete", 120000);
    const jobs = await invoke("list_jobs");
    const job = jobs.find(j => j.kind === "train" && j.status === "complete");
    const detail = await invoke("job_details", { id: job.id });
    if (!detail.checkpoints.length) throw new Error("Native job has no checkpoints");
    fill("adapter-job", job.id);
    fill("adapter-dest", fixture.export);
    click("adapter-export");
    await wait(() => document.getElementById("training-output").textContent.includes(fixture.export), "Adapter export failed");
    results.push("real CUDA QLoRA through native IPC, streamed job completion, checkpoints, adapter export");
    await tab("Discover", "hf-search");
    results.push("discovery form mounted");
    await tab("Maintenance", "scan");
    click("scan");
    await wait(() => document.getElementById("maintenance-output").textContent.includes("cleanup_enabled"), "Model-store scan failed");
    results.push("read-only model-store scan");
    await invoke("desktop_smoke_report", { report: { ok: true, results, storage: cfg.storage_root } });
  } catch (error) {
    await invoke("desktop_smoke_report", { report: {
      ok: false, error: String(error), results,
      page: document.getElementById("page")?.innerText,
      notice: document.getElementById("notice")?.textContent,
    } });
  }
})();
