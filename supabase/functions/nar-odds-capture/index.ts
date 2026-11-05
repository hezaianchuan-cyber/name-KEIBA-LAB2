import { unzipSync } from "npm:fflate@0.8.2";

const RACE_URL = "https://www.keiba.go.jp/KeibaWeb/DataDownload/RaceDataDownload?type=daily";
const ODDS_URL = "https://www.keiba.go.jp/KeibaWeb/DataDownload/OddsDataDownload?type=daily";
const WINDOW_START_MIN = 30;
const WINDOW_END_MIN = 2;

function decodeCsv(bytes: Uint8Array): string {
  // Daily archives may be UTF-8 or Windows-31J/CP932. Strict UTF-8 alone
  // is not enough: some CP932 byte sequences can also decode as UTF-8.
  const hasUtf8Bom = bytes.length >= 3 &&
    bytes[0] === 0xef && bytes[1] === 0xbb && bytes[2] === 0xbf;
  const candidates: string[] = [];
  if (hasUtf8Bom) candidates.push(new TextDecoder("utf-8").decode(bytes));
  else {
    try { candidates.push(new TextDecoder("utf-8", { fatal: true }).decode(bytes)); } catch { /* try CP932 */ }
  }
  try { candidates.push(new TextDecoder("shift_jis").decode(bytes)); } catch { /* keep valid candidate */ }
  if (!candidates.length) throw new Error("NAR CSV encoding could not be decoded");
  const venues = ["帯広","門別","盛岡","水沢","浦和","船橋","大井","川崎","金沢","笠松","名古屋","園田","姫路","高知","佐賀"];
  const score = (value: string) => venues.reduce((total, venue) => {
    let count = 0, at = 0;
    while ((at = value.indexOf(venue, at)) !== -1) { count++; at += venue.length; }
    return total + count;
  }, 0);
  return candidates.reduce((best, value) => score(value) > score(best) ? value : best);
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


async function loadModelSnapshot(raceId:string,raceDate:string,capturedAt:string,serviceUrl:string,serviceKey:string):Promise<any|null>{
  const cutoffMs=new Date(capturedAt).getTime()-15*60*1000;
  if(!Number.isFinite(cutoffMs))return null;
  const params=new URLSearchParams();
  params.set("select","race_id,race_date,captured_at,scheduled_post_time,model_version,model_temperature,predictions,candidate_combinations,model_g_version,model_g_predictions,model_g_candidate_combinations");
  params.set("race_id","eq."+raceId);
  params.set("race_date","eq."+raceDate);
  params.append("captured_at","gte."+new Date(cutoffMs).toISOString());
  params.append("captured_at","lte."+capturedAt);
  params.set("order","captured_at.desc");
  params.set("limit","1");
  const response=await fetch(serviceUrl+"/rest/v1/model_prediction_snapshots?"+params.toString(),{
    headers:{"apikey":serviceKey,"Authorization":"Bearer "+serviceKey,"Accept":"application/json","Cache-Control":"no-cache"},
    signal:AbortSignal.timeout(12000),
  });
  if(!response.ok)throw new Error("Model snapshot lookup HTTP "+response.status+": "+(await response.text()).slice(0,300));
  const rows=await response.json();
  if(!Array.isArray(rows)||!rows.length)return null;
  const model=rows[0];
  const modelAt=new Date(model.captured_at).getTime();
  const postAt=new Date(model.scheduled_post_time).getTime();
  const oddsAt=new Date(capturedAt).getTime();
  if(!Number.isFinite(modelAt)||!Number.isFinite(postAt)||modelAt>oddsAt||postAt<=modelAt||oddsAt-modelAt>15*60*1000)return null;
  return model;
}

async function archiveModelSnapshots(snapshots:any[],serviceUrl:string,serviceKey:string):Promise<any>{
  if(!Array.isArray(snapshots))throw new Error("snapshots must be an array");
  const valid=snapshots.filter((item:any)=>{
    if(!item||typeof item.race_id!=="string"||typeof item.race_date!=="string"||
       typeof item.model_version!=="string"||!Array.isArray(item.predictions)||item.predictions.length<3||
       !item.prediction_lock||!/^[a-f0-9]{64}$/i.test(String(item.prediction_lock.digest||"")))return false;
    const captured=new Date(item.captured_at).getTime(),post=new Date(item.scheduled_post_time).getTime();
    const minutes=(post-captured)/60000;
    return Number.isFinite(captured)&&Number.isFinite(post)&&minutes>=2&&minutes<=30&&
      item.prediction_lock.locked_at===item.captured_at;
  });
  if(!valid.length)return {ok:true,received:snapshots.length,upserted:0,rejected:snapshots.length};
  const response=await fetch(serviceUrl+"/rest/v1/model_prediction_snapshots?on_conflict=race_id,captured_at",{
    method:"POST",
    headers:{"apikey":serviceKey,"Authorization":"Bearer "+serviceKey,"Content-Type":"application/json",
      "Prefer":"resolution=merge-duplicates,return=representation"},
    body:JSON.stringify(valid),
    signal:AbortSignal.timeout(20000),
  });
  if(!response.ok)throw new Error("Model snapshot archive HTTP "+response.status+": "+(await response.text()).slice(0,500));
  const saved=await response.json();
  return {ok:true,received:snapshots.length,upserted:Array.isArray(saved)?saved.length:valid.length,rejected:snapshots.length-valid.length};
}

async function loadValidatedThresholds():Promise<Record<string,number>>{
  try{
    const report=await fetchJson(REPO_RAW+"/data/backtest-summary.json");
    const types=report?.preoff_ticket_type_backtest?.bet_types||{},out:Record<string,number>={};
    for(const [kind,value] of Object.entries<any>(types)){
      if(kind!=="trio")continue;
      const threshold=Number(value?.selected_threshold);
      if(Number.isFinite(threshold)&&threshold>0&&value?.selected_holdout?.candidate_pass===true)out[kind]=threshold;
    }
    return out;
  }catch{return {};}
}
async function buildAutoTicketPlan(row:any,thresholds:Record<string,number>,serviceUrl:string,serviceKey:string):Promise<any>{
  const raceId=String(row.race_id||""),raceDate=String(row.race_date||""),capturedAt=String(row.captured_at||new Date().toISOString());
  const base={version:2,createdAt:capturedAt,date:raceDate.replace(/-/g,""),raceId,venue:String(row.venue||""),raceNo:String(row.race_number||""),autoGenerated:true,source:"supabase_one_minute_odds",oddsCapturedAt:capturedAt,modelCapturedAt:null};
  const model=await loadModelSnapshot(raceId,raceDate,capturedAt,serviceUrl,serviceKey).catch((error)=>{console.error("Model snapshot lookup failed:",error);return null;});
  const oddsRows=Array.isArray(row.odds_data?.odds)?row.odds_data.odds:[];
  if(!model)return {...base,decision:"SKIP",skipReason:"15分以内の予測スナップショットがありません",skipReasonCode:"MODEL_SNAPSHOT_NOT_AVAILABLE",totalStake:0,tickets:[],validatedBetTypes:Object.keys(thresholds).length};
  const predictions=[...(model.predictions||[])].sort((a:any,b:any)=>(Number(b.win_probability)||0)-(Number(a.win_probability)||0));
  const top=predictions.slice(0,5),topNumbers=new Set(top.map((x:any)=>horseNo(x.horse_no)));
  const firstNumbers=new Set(predictions.slice(0,2).map((x:any)=>horseNo(x.horse_no)));
  const oddsLookup=new Map<string,number>();
  for(const item of oddsRows){
    const kind=betKind(item.type);if(!kind)continue;
    const nums=[item.n1,item.n2,item.n3].filter((x:any)=>digitsOnly(x)).map(horseNo);
    if(nums.length!==REQUIRED[kind]||new Set(nums).size!==nums.length)continue;
    const price=Number(item.odds);if(Number.isFinite(price)&&price>0)oddsLookup.set(kind+"|"+combinationKey(kind,nums),price);
  }
  const candidates:any[]=[];
  for(const item of model.candidate_combinations||[]){
    const kind=String(item.bet_type||"");if(kind!=="trio"||!(kind in REQUIRED)||!(kind in thresholds))continue;
    const nums=(item.numbers||[]).map(horseNo);
    if(nums.length!==REQUIRED[kind]||new Set(nums).size!==nums.length||nums.some((n:string)=>!topNumbers.has(n)))continue;
    if((kind==="exacta"||kind==="trifecta")&&!firstNumbers.has(nums[0]))continue;
    const price=oddsLookup.get(kind+"|"+combinationKey(kind,nums)),prob=Number(item.probability);
    if(!(price&&price>0&&prob>0))continue;
    candidates.push({kind,type:BET_TYPES[kind],comb:nums.join("-"),odds:price,probability:prob,ev:price*prob});
  }
  const eligible:any[]=[];
  for(const [kind,threshold] of Object.entries(thresholds)){
    if(kind!=="trio")continue;
    const rows=candidates.filter(x=>x.kind===kind&&x.ev>=Math.max(1.05,threshold))
      .sort((a,b)=>(b.probability-a.probability)||(b.ev-a.ev)).slice(0,20);
    if(rows.length)eligible.push({kind,rows,avg:rows.slice(0,5).reduce((s,x)=>s+x.ev,0)/Math.min(5,rows.length),best:rows[0]});
  }
  eligible.sort((a,b)=>(b.avg-a.avg)||(b.best.probability-a.best.probability));
  const selected=eligible[0];
  if(!selected){
    const reason=!oddsRows.length?"1分オッズが未取得":!Object.keys(thresholds).length?"券種別の検証済みEV基準が未成立":"検証済みEV基準を満たす候補なし";
    const code=!oddsRows.length?"ODDS_NOT_CAPTURED":!Object.keys(thresholds).length?"NO_VALIDATED_TICKET_THRESHOLD":"EV_BELOW_VALIDATED_THRESHOLD";
    return {...base,decision:"SKIP",skipReason:reason,skipReasonCode:code,totalStake:0,tickets:[],modelCapturedAt:model.captured_at,modelVersion:model.model_version,oddsRows:oddsRows.length,validatedBetTypes:Object.keys(thresholds).length};
  }
  const rows=selected.rows,target=Math.min(2000,rows.length*100);
  const weights=rows.map(x=>Math.max(0.0001,(x.ev-1)*Math.sqrt(Math.max(x.probability,0.000001))));
  const total=weights.reduce((a,b)=>a+b,0);
  const tickets=rows.map((x,i)=>({...x,stake:total?Math.floor((target*weights[i]/total)/100)*100:0})).filter(x=>x.stake>=100);
  let allocated=tickets.reduce((s,x)=>s+x.stake,0),index=0;
  while(tickets.length&&allocated+100<=target){tickets[index].stake+=100;allocated+=100;index=(index+1)%tickets.length;}
  return {...base,decision:tickets.length?"BET":"SKIP",skipReason:"",skipReasonCode:"BET",selectedType:selected.kind,totalStake:allocated,
    tickets:tickets.map(x=>({type:x.type,comb:x.comb,stake:x.stake,odds:x.odds,probability:x.probability,ev:x.ev})),
    modelCapturedAt:model.captured_at,modelVersion:model.model_version,oddsRows:oddsRows.length,oddsMatchedCandidates:candidates.length,validatedBetTypes:Object.keys(thresholds).length};
}

Deno.serve(async (request) => {
  if (request.method !== "POST") return new Response("Method not allowed", { status: 405 });
  const serviceUrl = Deno.env.get("SUPABASE_URL");
  const serviceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
  const expectedToken = Deno.env.get("NAR_CAPTURE_TOKEN");
  if (!serviceUrl || !serviceKey || !expectedToken) {
    return Response.json({ ok: false, error: "Required function secrets are missing" }, { status: 500 });
  }
  const authorizationHeader = request.headers.get("authorization") || "";
  const supplied = authorizationHeader.replace(/^Bearer\s+/i, "");
  if (!supplied || supplied !== expectedToken) {
    return Response.json({
      ok: false,
      error: "Unauthorized",
      diagnostics: {
        authorization_header_present: authorizationHeader.length > 0,
        token_matches: supplied.length > 0 && supplied === expectedToken,
        supplied_token_length: supplied.length,
        expected_token_length: expectedToken.length,
      },
    }, { status: 401 });
  }

  let requestPayload:any={};
  try { requestPayload=await request.json(); } catch { requestPayload={}; }
  if(requestPayload?.action==="archive_model_snapshots"){
    try{
      const result=await archiveModelSnapshots(requestPayload.snapshots||[],serviceUrl,serviceKey);
      return Response.json(result);
    }catch(error){
      console.error("Model snapshot archive failed:",error);
      return Response.json({ok:false,error:String(error)}, {status:502});
    }
  }

  const runStartedAt = Date.now();
  const runSource = "pg_cron";
  async function writeRunLog(status: string, metrics: Record<string, unknown> = {}) {
    try {
      const response = await fetch(serviceUrl + "/rest/v1/odds_capture_runs", {
        method: "POST",
        headers: {
          "apikey": serviceKey,
          "Authorization": "Bearer " + serviceKey,
          "Content-Type": "application/json",
          "Prefer": "return=minimal",
        },
        body: JSON.stringify({
          started_at: new Date(runStartedAt).toISOString(),
          finished_at: new Date().toISOString(),
          status,
          trigger_source: runSource,
          elapsed_ms: Date.now() - runStartedAt,
          ...metrics,
        }),
      });
      if (!response.ok) console.error("Could not write odds capture run log:", response.status, (await response.text()).slice(0, 300));
    } catch (logError) {
      console.error("Could not write odds capture run log:", logError);
    }
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
    let ticketPlansSaved = 0;
    if (inserts.length) {
      const response = await fetch(serviceUrl + "/rest/v1/odds_snapshots", {
        method: "POST",
        headers: {
          "apikey": serviceKey,
          "Authorization": "Bearer " + serviceKey,
          "Content-Type": "application/json",
          "Prefer": "return=representation",
        },
        body: JSON.stringify(inserts),
      });
      if (!response.ok) throw new Error("Supabase insert HTTP " + response.status + ": " + (await response.text()).slice(0, 500));
      const insertedRows = await response.json();
      inserted = Array.isArray(insertedRows) ? insertedRows.length : inserts.length;
      const thresholds = await loadValidatedThresholds();
      for (const savedRow of Array.isArray(insertedRows) ? insertedRows : []) {
        const plan = await buildAutoTicketPlan(savedRow, thresholds, serviceUrl, serviceKey);
        const update = await fetch(serviceUrl + "/rest/v1/odds_snapshots?id=eq." + encodeURIComponent(String(savedRow.id)), {
          method: "PATCH",
          headers: {
            "apikey": serviceKey,
            "Authorization": "Bearer " + serviceKey,
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
          },
          body: JSON.stringify({ odds_data: { ...(savedRow.odds_data || {}), auto_ticket_plan: plan } }),
        });
        if (!update.ok) throw new Error("Could not attach auto ticket plan: HTTP " + update.status + " " + (await update.text()).slice(0, 300));
        ticketPlansSaved++;
      }
    }
    const metrics = {
      race_rows: races.size,
      positive_odds_rows: positiveOddsRows,
      races_in_preoff_window: racesInWindow,
      snapshots_inserted: inserted,
      auto_ticket_plans_saved: ticketPlansSaved,
    };
    await writeRunLog("success", metrics);
    return Response.json({
      ok: true,
      checked_at: new Date(now).toISOString(),
      ...metrics,
      elapsed_ms: Date.now() - runStartedAt,
      window_minutes: [WINDOW_START_MIN, WINDOW_END_MIN],
    });
  } catch (error) {
    console.error(error);
    await writeRunLog("error", { error_message: String(error).slice(0, 1000) });
    return Response.json({ ok: false, error: String(error) }, { status: 502 });
  }
});
