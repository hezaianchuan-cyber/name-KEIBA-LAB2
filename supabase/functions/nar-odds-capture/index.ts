import { unzipSync } from "npm:fflate@0.8.2";

const RACE_URL = "https://www.keiba.go.jp/KeibaWeb/DataDownload/RaceDataDownload?type=daily";
const ODDS_URL = "https://www.keiba.go.jp/KeibaWeb/DataDownload/OddsDataDownload?type=daily";
const WINDOW_START_MIN = 30;
const WINDOW_END_MIN = 2;

function decodeCsv(bytes: Uint8Array): string {
  try { return new TextDecoder("shift_jis").decode(bytes); }
  catch { return new TextDecoder("utf-8").decode(bytes); }
}

function csvRows(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (ch === '"') {
      if (quoted && text[i + 1] === '"') { field += '"'; i++; }
      else quoted = !quoted;
    } else if (ch === "," && !quoted) {
      row.push(field); field = "";
    } else if ((ch === "\n" || ch === "\r") && !quoted) {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(field); field = "";
      if (row.some(v => v.length)) rows.push(row);
      row = [];
    } else field += ch;
  }
  row.push(field);
  if (row.some(v => v.length)) rows.push(row);
  return rows;
}

function classify(name: string): "races" | "odds" | null {
  const base = name.replaceAll("\\", "/").split("/").pop()!.toLowerCase();
  if (/racelist|race.?list|レース一覧/.test(base)) return "races";
  if (/odds|オッズ/.test(base)) return "odds";
  return null;
}

function digits(value: unknown): string {
  return String(value ?? "").replace(/\D/g, "");
}

function raceKey(venue: string, date: string, race: string): string | null {
  const d = digits(date), r = digits(race), v = venue.trim();
  if (!v || d.length !== 8 || !r) return null;
  return [v, d, String(Number(r))].join("|");
}

function postTime(value: string): { hour: number; minute: number } | null {
  const raw = digits(value);
  if (raw.length < 3 || raw.length > 4) return null;
  const padded = raw.padStart(4, "0");
  const hour = Number(padded.slice(0, 2)), minute = Number(padded.slice(2));
  if (hour > 23 || minute > 59) return null;
  return { hour, minute };
}

async function fetchZip(url: string): Promise<Record<string, Uint8Array>> {
  const response = await fetch(url, {
    headers: { "User-Agent": "KEIBA-LAB/1.0", "Referer": "https://www.keiba.go.jp/" },
    signal: AbortSignal.timeout(45000),
  });
  if (!response.ok) throw new Error(`NAR archive HTTP ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (bytes[0] !== 0x50 || bytes[1] !== 0x4b) throw new Error("NAR response was not a ZIP archive");
  return unzipSync(bytes);
}

Deno.serve(async (request) => {
  if (request.method !== "POST") return new Response("Method not allowed", { status: 405 });
  const serviceUrl = Deno.env.get("SUPABASE_URL");
  const serviceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
  const expectedToken = Deno.env.get("NAR_CAPTURE_TOKEN");
  if (!serviceUrl || !serviceKey || !expectedToken) {
    return Response.json({ ok: false, error: "Required function secrets are missing" }, { status: 500 });
  }
  const supplied = (request.headers.get("authorization") || "").replace(/^Bearer\s+/i, "");
  if (!supplied || supplied !== expectedToken) {
    return Response.json({ ok: false, error: "Unauthorized" }, { status: 401 });
  }

  try {
    const [raceZip, oddsZip] = await Promise.all([fetchZip(RACE_URL), fetchZip(ODDS_URL)]);
    const races = new Map<string, { venue: string; date: string; race: string; post: Date }>();
    const odds = new Map<string, Array<Record<string, unknown>>>();

    for (const [name, bytes] of Object.entries(raceZip)) {
      if (classify(name) !== "races") continue;
      for (const row of csvRows(decodeCsv(bytes))) {
        if (row.length < 4) continue;
        const key = raceKey(row[0], row[1], row[2]);
        const time = postTime(row[3]);
        if (!key || !time) continue;
        const d = digits(row[1]);
        const post = new Date(`${d.slice(0, 4)}-${d.slice(4, 6)}-${d.slice(6, 8)}T${String(time.hour).padStart(2, "0")}:${String(time.minute).padStart(2, "0")}:00+09:00`);
        if (!Number.isNaN(post.getTime())) races.set(key, { venue: row[0].trim(), date: d, race: String(Number(digits(row[2]))), post });
      }
    }

    let positiveOddsRows = 0;
    for (const [name, bytes] of Object.entries(oddsZip)) {
      if (classify(name) !== "odds") continue;
      for (const row of csvRows(decodeCsv(bytes))) {
        if (row.length < 10) continue;
        const key = raceKey(row[0], row[1], row[2]);
        const value = Number(String(row[7] || "").replace(/,/g, "").trim());
        if (!key || !Number.isFinite(value) || value <= 0) continue;
        positiveOddsRows++;
        const entry = {
          type: String(row[3] || "").trim(),
          n1: String(row[4] || "").trim(),
          n2: String(row[5] || "").trim(),
          n3: String(row[6] || "").trim(),
          odds: value,
          max_odds: String(row[8] || "").trim(),
          popularity: String(row[9] || "").trim(),
        };
        const list = odds.get(key) || [];
        list.push(entry);
        odds.set(key, list);
      }
    }

    const now = Date.now();
    const inserts: Array<Record<string, unknown>> = [];
    let racesInWindow = 0;
    for (const [key, race] of races) {
      const minutes = (race.post.getTime() - now) / 60000;
      if (minutes < WINDOW_END_MIN || minutes > WINDOW_START_MIN) continue;
      racesInWindow++;
      const rows = odds.get(key);
      if (!rows?.length) continue;
      inserts.push({
        race_id: key,
        race_date: `${race.date.slice(0, 4)}-${race.date.slice(4, 6)}-${race.date.slice(6, 8)}`,
        venue: race.venue,
        race_number: Number(race.race),
        odds_data: {
          captured_at: new Date(now).toISOString(),
          scheduled_post_time: race.post.toISOString(),
          minutes_to_post: Math.round(minutes * 100) / 100,
          odds: rows,
        },
        source: "NAR_OFFICIAL_DAILY_1MIN",
      });
    }

    let inserted = 0;
    if (inserts.length) {
      const response = await fetch(`${serviceUrl}/rest/v1/odds_snapshots`, {
        method: "POST",
        headers: {
          "apikey": serviceKey,
          "Authorization": `Bearer ${serviceKey}`,
          "Content-Type": "application/json",
          "Prefer": "return=minimal",
        },
        body: JSON.stringify(inserts),
      });
      if (!response.ok) throw new Error(`Supabase insert HTTP ${response.status}: ${(await response.text()).slice(0, 500)}`);
      inserted = inserts.length;
    }

    return Response.json({
      ok: true,
      checked_at: new Date(now).toISOString(),
      race_rows: races.size,
      positive_odds_rows: positiveOddsRows,
      races_in_preoff_window: racesInWindow,
      snapshots_inserted: inserted,
      window_minutes: [WINDOW_START_MIN, WINDOW_END_MIN],
    });
  } catch (error) {
    console.error(error);
    return Response.json({ ok: false, error: String(error) }, { status: 502 });
  }
});
