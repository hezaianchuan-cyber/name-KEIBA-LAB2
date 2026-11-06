const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
require("../supabase/functions/nar-odds-capture/_shared/trio_strategy.js");
const strategy = globalThis.KEIBALAB_TRIO_STRATEGY;

function rows(scores) {
  const raw = scores.map((score, i) => ({
    horseNo: String(i + 1),
    score,
    modelProbability: scores.length - i
  }));
  return raw;
}

test("50% blend fuses model and market implied win probabilities", () => {
  const result = strategy.fuse([
    { horseNo: "1", score: 90, modelProbability: 0.7 },
    { horseNo: "2", score: 80, modelProbability: 0.3 }
  ], [
    { type: "単勝", n1: "1", odds: 1 },
    { type: "単勝", n1: "2", odds: 3 }
  ], 0.5);
  assert.equal(result.marketBlendApplied, true);
  assert.ok(Math.abs(result.runners[0].probability - 0.725) < 1e-10);
  assert.ok(Math.abs(result.runners[1].probability - 0.275) < 1e-10);
  assert.ok(Math.abs(result.runners.reduce((sum, row) => sum + row.probability, 0) - 1) < 1e-10);
});

test("incomplete market coverage below 70% falls back to LAB probabilities", () => {
  const result = strategy.fuse([
    { horseNo: "1", score: 90, modelProbability: 0.5 },
    { horseNo: "2", score: 80, modelProbability: 0.3 },
    { horseNo: "3", score: 70, modelProbability: 0.2 }
  ], [{ type: "単勝", n1: "1", odds: 2 }], 0.5);
  assert.equal(result.marketBlendApplied, false);
  assert.equal(result.blendWeight, 0);
  assert.ok(Math.abs(result.runners[0].probability - 0.5) < 1e-10);
});

test("adaptive pool matches the four top-4/5/6/7 score-gap outcomes", () => {
  assert.equal(strategy.adaptivePoolSize(rows([100, 99, 98, 97, 70, 69, 68])), 4);
  assert.equal(strategy.adaptivePoolSize(rows([100, 99, 98, 97, 96, 60, 59])), 5);
  assert.equal(strategy.adaptivePoolSize(rows([100, 99, 98, 97, 96, 95, 60])), 6);
  assert.equal(strategy.adaptivePoolSize(rows([100, 99, 98, 97, 96, 95, 94])), 7);
});

test("adaptive pool generates C(n,3) trio candidates and matches official odds", () => {
  for (const [n, count] of [[4, 4], [5, 10], [6, 20], [7, 35]]) {
    const runners = rows([100, 99, 98, 97, 96, 95, 94]).slice(0, n);
    const oddsRows = [{ type: "3連複", n1: "1", n2: "2", n3: "3", odds: 12.5 }];
    const result = strategy.build(runners, oddsRows, 0.5, 100);
    assert.equal(result.poolSize, n);
    assert.equal(result.candidates.length, count);
    assert.ok(result.candidates.every(ticket => ticket.probability > 0));
    assert.equal(result.candidates.find(ticket => ticket.comb === "1-2-3").odds, 12.5);
  }
});

test("Web and Supabase both load the same shared strategy implementation", () => {
  const root = path.resolve(__dirname, "..");
  const web = fs.readFileSync(path.join(root, "index.html"), "utf8");
  const edge = fs.readFileSync(path.join(root, "supabase/functions/nar-odds-capture/index.ts"), "utf8");
  assert.ok(web.includes("./supabase/functions/nar-odds-capture/_shared/trio_strategy.js"));
  assert.ok(edge.includes('import "./_shared/trio_strategy.js"'));
  assert.ok(web.includes("adaptivePoolSize"));
  assert.ok(edge.includes("adaptivePoolSize"));
  assert.ok(web.includes("window.KEIBALAB_TRIO_STRATEGY.trioProbability"));
  assert.ok(edge.includes("strategy.build(modelRows,oddsRows,0.5,0.60)"));
  assert.ok(edge.includes("const shadowTickets=candidates.map("));
  assert.ok(web.includes("shadowTickets:ranked('3連複').map("));
});
