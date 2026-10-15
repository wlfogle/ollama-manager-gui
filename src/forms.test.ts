import { describe, expect, it } from "vitest";
import { datasetName, escape, numberValue } from "./forms";

describe("input boundaries", () => {
  it("escapes source and model names before HTML rendering", () => {
    expect(escape('<script>"&')).toBe("&lt;script&gt;&quot;&amp;");
  });
  it("rejects traversal dataset names", () => {
    expect(() => datasetName("../outside")).toThrow();
    expect(datasetName("repo-data_1")).toBe("repo-data_1");
  });
  it("rejects invalid hyperparameters rather than silently defaulting", () => {
    for (const value of ["", "NaN", "Infinity", "-1", "1.5"]) {
      expect(() => numberValue(value, 1, 256, true)).toThrow();
    }
    expect(numberValue("8", 1, 256, true)).toBe(8);
  });
});
