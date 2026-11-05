import { gzipSync, gunzipSync } from "npm:fflate@0.8.2";

const BUCKET = "keiba-odds-archive";
const PAGE_SIZE = 500;
const MAX_DAYS_PER_RUN = 5;
const MAX_OBJECT_BYTES = 50 * 1024 * 1024;
const RETENTION_DAYS = 14;

function hex(bytes: ArrayBuffer): string {
  return [...new Uint8Array(bytes)].map((v) => v.toString(16).padStart(2, "0")).join("");
}
async function sha256(bytes: Uint8Array): Promise<string> {
  return hex(await crypto.subtle.digest("SHA-256", bytes));
}
function authHeaders(key: string) {
  return { apikey: key, Authorization: "Bearer " + key };
}

Deno.serve(async (request) => {
  if (request.method !== "POST") return new Response("Method not allowed", { status: 405 });
  const serviceUrl = Deno.env.get("SUPABASE_URL")?.replace(/\/$/, "");
  const serviceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
  const expectedToken = Deno.env.get("NAR_CAPTURE_TOKEN");
  if (!serviceUrl || !serviceKey || !expectedToken) {
    return Response.json({ ok: false, error: "Required function secrets are missing" }, { status: 500 });
  }
  const supplied = (request.headers.get("authorization") || "").replace(/^Bearer\s+/i, "");
  if (!supplied || supplied !== expectedToken) return Response.json({ ok: false, error: "Unauthorized" }, { status: 401 });

  const started = Date.now();
  const cutoff = new Date(Date.now() - RETENTION_DAYS * 86400000).toISOString().slice(0, 10);
  const headers = authHeaders(serviceKey);
  const results: Array<Record<string, unknown>> = [];

  try {
    const daysResponse = await fetch(serviceUrl + "/rest/v1/rpc/keiba_pending_archive_dates", {
      method: "POST",
      headers: { ...headers, "Content-Type": "application/json" },
      body: JSON.stringify({ p_before: cutoff, p_limit: MAX_DAYS_PER_RUN }),
      signal: AbortSignal.timeout(30000),
    });
    if (!daysResponse.ok) throw new Error("Could not list archive dates: HTTP " + daysResponse.status + " " + (await daysResponse.text()).slice(0, 400));
    const days = await daysResponse.json();
    if (!Array.isArray(days)) throw new Error("Archive date RPC returned an invalid response");

    for (const entry of days) {
      const day = String(entry.race_date || "");
      if (!/^\d{4}-\d{2}-\d{2}$/.test(day)) continue;
      const rows: Record<string, unknown>[] = [];
      for (let offset = 0; ; offset += PAGE_SIZE) {
        const query = new URLSearchParams({
          select: "id,race_id,race_date,venue,race_number,captured_at,odds_data,source",
          race_date: "eq." + day, order: "id.asc", limit: String(PAGE_SIZE), offset: String(offset),
        });
        const pageResponse = await fetch(serviceUrl + "/rest/v1/odds_snapshots?" + query.toString(), {
          headers, signal: AbortSignal.timeout(30000),
        });
        if (!pageResponse.ok) throw new Error("Could not read " + day + ": HTTP " + pageResponse.status + " " + (await pageResponse.text()).slice(0, 400));
        const page = await pageResponse.json();
        if (!Array.isArray(page)) throw new Error("Invalid odds snapshot page for " + day);
        rows.push(...page);
        if (page.length < PAGE_SIZE) break;
      }
      if (!rows.length) continue;

      const ndjson = rows.map((row) => JSON.stringify(row)).join("\n") + "\n";
      const uncompressed = new TextEncoder().encode(ndjson);
      const compressed = gzipSync(uncompressed, { level: 9 });
      if (compressed.byteLength > MAX_OBJECT_BYTES) throw new Error(day + " archive exceeds the 50MB object limit");
      const checksum = await sha256(compressed);
      const objectPath = day + ".jsonl.gz";

      const upload = await fetch(serviceUrl + "/storage/v1/object/" + BUCKET + "/" + objectPath, {
        method: "POST",
        headers: { ...headers, "Content-Type": "application/gzip", "x-upsert": "true" },
        body: compressed, signal: AbortSignal.timeout(60000),
      });
      if (!upload.ok) throw new Error("Archive upload failed for " + day + ": HTTP " + upload.status + " " + (await upload.text()).slice(0, 400));

      const download = await fetch(serviceUrl + "/storage/v1/object/authenticated/" + BUCKET + "/" + objectPath, {
        headers, signal: AbortSignal.timeout(60000),
      });
      if (!download.ok) throw new Error("Archive read-back failed for " + day + ": HTTP " + download.status);
      const downloaded = new Uint8Array(await download.arrayBuffer());
      if (await sha256(downloaded) !== checksum) throw new Error("Archive checksum mismatch for " + day);
      const restored = new TextDecoder().decode(gunzipSync(downloaded));
      const verifiedRows = restored.split("\n").filter((line) => line.length > 0).length;
      if (verifiedRows !== rows.length) throw new Error("Archive row-count mismatch for " + day + ": " + verifiedRows + " != " + rows.length);

      const manifest = await fetch(serviceUrl + "/rest/v1/odds_archive_manifest", {
        method: "POST",
        headers: { ...headers, "Content-Type": "application/json", Prefer: "resolution=merge-duplicates,return=minimal" },
        body: JSON.stringify({
          race_date: day, object_path: objectPath, row_count: rows.length,
          uncompressed_bytes: uncompressed.byteLength, compressed_bytes: compressed.byteLength,
          sha256: checksum, verified_at: new Date().toISOString(),
        }),
        signal: AbortSignal.timeout(30000),
      });
      if (!manifest.ok) throw new Error("Manifest write failed for " + day + ": HTTP " + manifest.status + " " + (await manifest.text()).slice(0, 400));
      results.push({ race_date: day, rows: rows.length, compressed_bytes: compressed.byteLength, sha256: checksum, verified: true });
    }

    const healthResponse = await fetch(serviceUrl + "/rest/v1/rpc/keiba_storage_health", {
      method: "POST", headers: { ...headers, "Content-Type": "application/json" }, body: "{}",
      signal: AbortSignal.timeout(30000),
    });
    if (!healthResponse.ok) throw new Error("Storage health query failed: HTTP " + healthResponse.status);
    return Response.json({
      ok: true, cutoff_date_exclusive: cutoff, retention_days: RETENTION_DAYS,
      archived_days: results, storage_health: await healthResponse.json(),
      deleted_source_rows: 0,
      note: "Source rows are retained until archive readers are integrated and tested.",
      elapsed_ms: Date.now() - started,
    });
  } catch (error) {
    console.error("KEIBA odds archive failed", error);
    return Response.json({ ok: false, error: String(error).slice(0, 1000), archived_days: results, deleted_source_rows: 0, elapsed_ms: Date.now() - started }, { status: 502 });
  }
});
