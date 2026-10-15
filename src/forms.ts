export function escape(value: unknown): string {
  return String(value ?? "").replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]!);
}

export function numberValue(value: string, min: number, max: number, integer = false): number {
  const n = Number(value);
  if (!value.trim() || !Number.isFinite(n) || n < min || n > max || (integer && !Number.isInteger(n))) {
    throw new Error(`Enter ${integer ? "an integer" : "a number"} between ${min} and ${max}`);
  }
  return n;
}

export function datasetName(value: string): string {
  if (!/^[A-Za-z0-9_-]{1,80}$/.test(value)) {
    throw new Error("Use 1–80 letters, digits, underscores, or hyphens for the dataset name");
  }
  return value;
}
