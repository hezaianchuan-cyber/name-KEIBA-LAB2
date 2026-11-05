import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import test from "node:test";

const html = readFileSync(new URL("../index.html", import.meta.url), "utf8");
const start = html.indexOf("function mergeLabTicketPlanRecord(");
const end = html.indexOf("\nfunction loadLabTicketPlans()", start);
assert.ok(start >= 0 && end > start, "merge helper must exist in index.html");

const context = {};
vm.runInNewContext(
  html.slice(start, end) + "\nglobalThis.mergeLabTicketPlanRecord = mergeLabTicketPlanRecord;",
  context,
);
const merge = context.mergeLabTicketPlanRecord;
const plain = (value) => JSON.parse(JSON.stringify(value));

test("newer automatic snapshots do not erase saved manual tickets", () => {
  const saved = {
    predictionId: "race-1",
    plan: {
      raceId: "race-1",
      createdAt: "2026-11-06T10:00:00.000Z",
      modifiedAt: "2026-11-06T10:05:00.000Z",
      userTicketsModifiedAt: "2026-11-06T10:05:00.000Z",
      userTickets: [{ type: "単勝", comb: "4", stake: 100 }],
    },
  };
  const newerAuto = {
    predictionId: "race-1",
    plan: { raceId: "race-1", createdAt: "2026-11-06T10:10:00.000Z", tickets: [] },
  };

  const result = merge(saved, newerAuto, newerAuto.plan);
  assert.equal(result.plan.createdAt, newerAuto.plan.createdAt);
  assert.deepEqual(plain(result.plan.userTickets), saved.plan.userTickets);
});

test("an explicit empty user-ticket list remains a deletion", () => {
  const deleted = {
    predictionId: "race-2",
    plan: {
      raceId: "race-2",
      createdAt: "2026-11-06T10:00:00.000Z",
      modifiedAt: "2026-11-06T10:15:00.000Z",
      userTicketsModifiedAt: "2026-11-06T10:15:00.000Z",
      userTickets: [],
    },
  };
  const newerAuto = {
    predictionId: "race-2",
    plan: { raceId: "race-2", createdAt: "2026-11-06T10:20:00.000Z" },
  };

  const result = merge(deleted, newerAuto, newerAuto.plan);
  assert.deepEqual(plain(result.plan.userTickets), []);
  assert.equal(result.plan.userTicketsModifiedAt, deleted.plan.userTicketsModifiedAt);
});

test("the newest user edit wins even when the automatic base plan is newer", () => {
  const previous = {
    predictionId: "race-3",
    plan: {
      raceId: "race-3",
      createdAt: "2026-11-06T10:30:00.000Z",
      userTicketsModifiedAt: "2026-11-06T10:25:00.000Z",
      userTickets: [{ type: "単勝", comb: "2", stake: 100 }],
    },
  };
  const incoming = {
    predictionId: "race-3",
    plan: {
      raceId: "race-3",
      createdAt: "2026-11-06T10:40:00.000Z",
      userTicketsModifiedAt: "2026-11-06T10:15:00.000Z",
      userTickets: [{ type: "単勝", comb: "8", stake: 100 }],
    },
  };

  const result = merge(previous, incoming, incoming.plan);
  assert.equal(result.plan.createdAt, incoming.plan.createdAt);
  assert.deepEqual(plain(result.plan.userTickets), previous.plan.userTickets);
});

test("a newer user edit is retained when the previous automatic snapshot wins", () => {
  const previous = {
    predictionId: "race-4",
    plan: { raceId: "race-4", createdAt: "2026-11-06T10:40:00.000Z" },
  };
  const incoming = {
    predictionId: "race-4",
    plan: {
      raceId: "race-4",
      createdAt: "2026-11-06T10:20:00.000Z",
      userTicketsModifiedAt: "2026-11-06T10:45:00.000Z",
      userTickets: [{ type: "単勝", comb: "7", stake: 200 }],
    },
  };

  const result = merge(previous, incoming, incoming.plan);
  assert.equal(result.plan.createdAt, previous.plan.createdAt);
  assert.deepEqual(plain(result.plan.userTickets), incoming.plan.userTickets);
});
