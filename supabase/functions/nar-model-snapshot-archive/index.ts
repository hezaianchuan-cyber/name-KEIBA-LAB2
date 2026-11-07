import "jsr:@supabase/functions-js/edge-runtime.d.ts";

function validSnapshot(item: any): boolean {
  if (!item || typeof item !== "object" || Array.isArray(item)) return false;
  if (typeof item.race_id !== "string" || !item.race_id.trim()) return false;
  if (!/^\d{4}-\d{2}-\d{2}$/.test(String(item.race_date || ""))) return false;
  if (typeof item.model_version !== "string" || !item.model_version.trim()) return false;
  if (!Array.isArray(item.predictions) || item.predictions.length < 3) return false;
  if (!Array.isArray(item.candidate_combinations)) return false;
  const lock = item.prediction_lock;
  const payload = item.locked_payload;
  if (!lock || lock.version !== "nar-prediction-lock-v1" || lock.algorithm !== "sha256") return false;
  if (!/^[a-f0-9]{64}$/i.test(String(lock.digest || ""))) return false;
  if (lock.locked_at !== item.captured_at) return false;
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) return false;
  if (payload.race_key !== item.race_id ||
      payload.captured_at !== item.captured_at ||
      payload.scheduled_post_time !== item.scheduled_post_time ||
      payload.model_version !== item.model_version ||
      !Array.isArray(payload.predictions) ||
      payload.predictions.length !== item.predictions.length ||
      !Array.isArray(payload.odds)) return false;
  const captured = new Date(item.captured_at).getTime();
  const post = new Date(item.scheduled_post_time).getTime();
  const minutes = (post - captured) / 60000;
  return Number.isFinite(captured) && Number.isFinite(post) && minutes >= 2 && minutes <= 60;
}

Deno.serve(async (request: Request) => {
  if (request.method !== "POST") {
    return Response.json({ ok: false, error: "Method not allowed" }, { status: 405 });
  }
  const serviceUrl = (Deno.env.get("SUPABASE_URL") || "").replace(/\/+$/, "");
  const serviceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") || "";
  const expectedToken = Deno.env.get("NAR_CAPTURE_TOKEN") || "";
  if (!serviceUrl || !serviceKey || !expectedToken) {
    return Response.json({ ok: false, error: "Required function secrets are missing" }, { status: 500 });
  }
  const supplied = (request.headers.get("authorization") || "").replace(/^Bearer\s+/i, "");
  if (!supplied || supplied !== expectedToken) {
    return Response.json({ ok: false, error: "Unauthorized" }, { status: 401 });
  }
  try {
    const body = await request.json();
    if (!body || body.action !== "archive_model_snapshots" || !Array.isArray(body.snapshots)) {
      return Response.json({ ok: false, error: "Invalid archive request" }, { status: 400 });
    }
    const received = body.snapshots.length;
    const valid = body.snapshots.filter(validSnapshot);
    if (!valid.length) {
      return Response.json({ ok: true, received, upserted: 0, rejected: received });
    }
    const response = await fetch(
      serviceUrl + "/rest/v1/model_prediction_snapshots?on_conflict=race_id,captured_at",
      {
        method: "POST",
        headers: {
          apikey: serviceKey,
          Authorization: "Bearer " + serviceKey,
          "Content-Type": "application/json",
          Prefer: "resolution=ignore-duplicates,return=representation",
        },
        body: JSON.stringify(valid),
        signal: AbortSignal.timeout(45000),
      },
    );
    if (!response.ok) {
      throw new Error("Snapshot archive insert HTTP " + response.status + ": " +
        (await response.text()).slice(0, 500));
    }
    const saved = await response.json();
    return Response.json({
      ok: true,
      received,
      valid: valid.length,
      upserted: Array.isArray(saved) ? saved.length : 0,
      rejected: received - valid.length,
    });
  } catch (error) {
    console.error("model snapshot archive failed", error);
    return Response.json({ ok: false, error: String(error).slice(0, 500) }, { status: 502 });
  }
});
