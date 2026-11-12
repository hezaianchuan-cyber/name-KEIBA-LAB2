import { unzipSync } from "npm:fflate@0.8.2";

const RACE_URL = "https://www.keiba.go.jp/KeibaWeb/DataDownload/RaceDataDownload?type=daily";
const START_MIN = 60;
const END_MIN = 2;

function digits(v: unknown): string { return String(v ?? "").replace(/\D/g, ""); }
function number(v: unknown): number | null {
  const raw = String(v ?? "").replace(/,/g, "").trim();
  if (!raw || raw === "-" || raw === "--") return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : null;
}
function postTime(v: unknown): {hour:number;minute:number}|null {
  const raw=digits(v); if(raw.length<3||raw.length>4)return null;
  const p=raw.padStart(4,"0"),hour=Number(p.slice(0,2)),minute=Number(p.slice(2));
  return hour<=23&&minute<=59?{hour,minute}:null;
}
function csvRows(text:string):string[][] {
  const rows:string[][]=[]; let row:string[]=[],field="",quoted=false;
  for(let i=0;i<text.length;i++){
    const ch=text[i];
    if(ch==='"'){if(quoted&&text[i+1]==='"'){field+='"';i++;}else quoted=!quoted;}
    else if(ch===","&&!quoted){row.push(field);field="";}
    else if((ch==="\n"||ch==="\r")&&!quoted){
      if(ch==="\r"&&text[i+1]==="\n")i++;
      row.push(field);field="";if(row.some(x=>x.length))rows.push(row);row=[];
    } else field+=ch;
  }
  row.push(field);if(row.some(x=>x.length))rows.push(row);return rows;
}
function decode(bytes:Uint8Array):string {
  const candidates:string[]=[];
  try{candidates.push(new TextDecoder("utf-8",{fatal:true}).decode(bytes));}catch{}
  try{candidates.push(new TextDecoder("shift_jis").decode(bytes));}catch{}
  if(!candidates.length)throw new Error("CSV encoding unsupported");
  const score=(s:string)=>["帯広","門別","船橋","大井","園田"].reduce((n,v)=>n+(s.split(v).length-1),0);
  return candidates.reduce((a,b)=>score(b)>score(a)?b:a);
}
function classify(name:string):"races"|"horses"|null {
  const base=name.replaceAll("\\","/").split("/").pop()!.toLowerCase();
  if(/racelist|race.?list|レース一覧/.test(base))return "races";
  if(/horselist|horse.?list|出馬表/.test(base))return "horses";
  return null;
}
function raceKey(v:string,d:string,r:string):string|null {
  const date=digits(d),race=digits(r),venue=v.normalize("NFKC").trim();
  if(venue!=="帯広"||date.length!==8||!race)return null;
  return venue+"|"+date+"|"+String(Number(race));
}
async function fetchArchive():Promise<Record<string,Uint8Array>> {
  const response=await fetch(RACE_URL,{headers:{"User-Agent":"KEIBA-LAB/1.0","Referer":"https://www.keiba.go.jp/"},
    signal:AbortSignal.timeout(25000)});
  if(!response.ok)throw new Error("NAR daily archive HTTP "+response.status);
  const bytes=new Uint8Array(await response.arrayBuffer());
  if(bytes[0]!==0x50||bytes[1]!==0x4b)throw new Error("NAR daily response was not ZIP");
  return unzipSync(bytes);
}
Deno.serve(async(req)=>{
  if(req.method!=="POST")return new Response("Method not allowed",{status:405});
  const url=Deno.env.get("SUPABASE_URL"),key=Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
  const token=Deno.env.get("NAR_BANEI_WEIGHT_CAPTURE_TOKEN");
  if(!url||!key||!token)return Response.json({ok:false,error:"Required secrets missing"},{status:500});
  if((req.headers.get("authorization")||"").replace(/^Bearer\s+/i,"")!==token)
    return Response.json({ok:false,error:"Unauthorized"},{status:401});
  try{
    const files=await fetchArchive(),races=new Map<string,{date:string;race:number;post:Date}>(),horses=new Map<string,any[]>();
    for(const [name,bytes] of Object.entries(files)){
      const kind=classify(name);if(!kind)continue;
      for(const row of csvRows(decode(bytes))){
        if(kind==="races"){
          if(row.length<4)continue;
          const raceId=raceKey(row[0],row[1],row[2]),time=postTime(row[3]);
          if(!raceId||!time)continue;
          const d=digits(row[1]);
          const post=new Date(d.slice(0,4)+"-"+d.slice(4,6)+"-"+d.slice(6,8)+"T"+
            String(time.hour).padStart(2,"0")+":"+String(time.minute).padStart(2,"0")+":00+09:00");
          if(!Number.isNaN(post.getTime()))races.set(raceId,{date:d,race:Number(digits(row[2])),post});
        } else {
          if(row.length<24)continue;
          const raceId=raceKey(row[0],row[1],row[2]);
          const horseNo=number(row[5]),name=String(row[6]||"").trim();
          if(!raceId||horseNo===null||!name)continue;
          const list=horses.get(raceId)||[];
          list.push({horseNo,name,carried:number(row[16]),body:number(row[22]),change:number(row[23])});
          horses.set(raceId,list);
        }
      }
    }
    const now=Date.now(),captured=new Date(now).toISOString(),inserts:any[]=[];
    const today=new Date(now).toISOString().slice(0,10);
    const existingUrl=url+"/rest/v1/banei_body_weight_snapshots?select=race_id,horse_number,carried_weight,body_weight,body_weight_change,captured_at&race_date=eq."+today+"&order=captured_at.desc&limit=20000";
    const existingResponse=await fetch(existingUrl,{headers:{apikey:key,Authorization:"Bearer "+key},signal:AbortSignal.timeout(20000)});
    if(!existingResponse.ok)throw new Error("Could not read existing Banei snapshots: HTTP "+existingResponse.status);
    const existingRows=await existingResponse.json();
    const latest=new Map<string,string>();
    for(const item of Array.isArray(existingRows)?existingRows:[]){
      const k=String(item.race_id)+"|"+String(item.horse_number);
      if(!latest.has(k))latest.set(k,[item.carried_weight??"",item.body_weight??"",item.body_weight_change??""].join("|"));
    }
    let racesInWindow=0,horseRows=0,rowsWithBodyWeight=0;
    for(const [raceId,race] of races){
      const minutes=(race.post.getTime()-now)/60000;
      if(minutes<END_MIN||minutes>START_MIN)continue;
      racesInWindow++;
      for(const horse of horses.get(raceId)||[]){
        horseRows++;if(horse.body!==null)rowsWithBodyWeight++;
        const snapshotHash=[horse.carried,horse.body,horse.change].map(x=>x===null?"":String(x)).join("|");
        const horseKey=raceId+"|"+String(horse.horseNo);
        if(latest.get(horseKey)===snapshotHash)continue;
        latest.set(horseKey,snapshotHash);
        inserts.push({race_id:raceId,race_date:race.date.slice(0,4)+"-"+race.date.slice(4,6)+"-"+race.date.slice(6,8),
          venue:"帯広",race_number:race.race,horse_number:horse.horseNo,horse_name:horse.name,
          carried_weight:horse.carried,body_weight:horse.body,body_weight_change:horse.change,
          captured_at:captured,scheduled_post_time:race.post.toISOString(),
          minutes_to_post:Math.round(minutes*100)/100,source:"NAR_OFFICIAL_DAILY_RACECARD",snapshot_hash:snapshotHash});
      }
    }
    let rowsSubmitted=0;
    for(let i=0;i<inserts.length;i+=500){
      const batch=inserts.slice(i,i+500);
      const response=await fetch(url+"/rest/v1/banei_body_weight_snapshots",{
        method:"POST",headers:{apikey:key,Authorization:"Bearer "+key,"Content-Type":"application/json",
          Prefer:"return=minimal"},body:JSON.stringify(batch),signal:AbortSignal.timeout(25000)});
      if(!response.ok)throw new Error("Supabase insert HTTP "+response.status+": "+(await response.text()).slice(0,400));
      rowsSubmitted+=batch.length;
    }
    return Response.json({ok:true,checked_at:captured,banei_races_in_window:racesInWindow,
      horse_rows_seen:horseRows,rows_with_body_weight:rowsWithBodyWeight,rows_submitted:inserts.length,
      rows_submitted:rowsSubmitted,window_minutes:[START_MIN,END_MIN],
      note:"Rows are timestamped observations, not proof of the exact public announcement time. Unchanged values are deduplicated."});
  }catch(error){console.error(error);return Response.json({ok:false,error:String(error)},{status:502});}
});
