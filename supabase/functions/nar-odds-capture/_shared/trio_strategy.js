/* One implementation shared by the Web UI and Supabase capture. */
(function (g) {
  "use strict";
  const no = v => { const d=String(v??"").replace(/\\D/g,""); return d?String(Number(d)):""; };
  const num = v => { const n=Number(String(v??"").replace(/,/g,"").trim()); return Number.isFinite(n)?n:null; };
  const label = v => String(v??"").normalize("NFKC").replace(/\\s/g,"");
  const comboKey = a => a.map(no).sort((x,y)=>Number(x)-Number(y)).join("-");
  function fuse(modelRows, oddsRows, weight=0.5) {
    const rows=(modelRows||[]).map(r=>({...r,horseNo:no(r.horseNo??r.horse_no??r.no),score:num(r.score)??0,modelProbability:Math.max(0,num(r.modelProbability??r.win_probability??r.probability??r.prob)??0)})).filter(r=>r.horseNo);
    const total=rows.reduce((s,r)=>s+r.modelProbability,0);
    if(!(total>0))return {runners:[],marketBlendApplied:false,marketCoverage:0};
    rows.forEach(r=>r.modelProbability/=total);
    const market=new Map();
    for(const o of oddsRows||[]){
      if(!label(o.type).includes("単勝"))continue;
      const n=no(o.n1??o.horseNo),price=num(o.odds??o.win_odds);
      if(!n||!(price>0))continue;
      const stamp=String(o.captured_at??o.capturedAt??""),old=market.get(n);
      if(!old||stamp>=old.stamp)market.set(n,{price,stamp});
    }
    const known=rows.filter(r=>market.has(r.horseNo)),coverage=known.length/rows.length;
    const sum=known.reduce((s,r)=>s+1/market.get(r.horseNo).price,0);
    const applied=coverage>=0.70&&sum>0,missingMass=rows.filter(r=>!market.has(r.horseNo)).reduce((s,r)=>s+r.modelProbability,0);
    const blend=Math.max(0,Math.min(1,num(weight)??0.5));
    const runners=rows.map(r=>{
      const mp=applied?(market.has(r.horseNo)?(1-missingMass)*(1/market.get(r.horseNo).price)/sum:r.modelProbability):null;
      return {...r,marketProbability:mp,probability:applied?(1-blend)*r.modelProbability+blend*mp:r.modelProbability,winOdds:market.get(r.horseNo)?.price??null};
    });
    const z=runners.reduce((s,r)=>s+r.probability,0)||1;
    runners.forEach(r=>r.probability/=z);
    runners.sort((a,b)=>b.probability-a.probability||b.score-a.score||Number(a.horseNo)-Number(b.horseNo));
    return {runners,marketBlendApplied:applied,marketCoverage:coverage,blendWeight:applied?blend:0};
  }
  function adaptivePoolSize(ranked, threshold=0.60) {
    const n=Math.min((ranked||[]).length,7); if(n<=3)return n;
    const s=ranked.slice(0,n).map(r=>num(r.score)); if(s.some(v=>v==null))return n;
    const mean=s.reduce((a,b)=>a+b,0)/n,sd=Math.sqrt(s.reduce((a,v)=>a+(v-mean)**2,0)/n);
    if(sd<=1e-12)return n;
    for(const boundary of [4,5,6])if(n>boundary&&(s[boundary-1]-s[boundary])/sd>=threshold)return boundary;
    return n;
  }
  function permutations(a){return a.length<=1?[a]:a.flatMap((v,i)=>permutations(a.filter((_,j)=>j!==i)).map(p=>[v,...p]));}
  function trioProbability(numbers,runners){
    const p=new Map(runners.map(r=>[r.horseNo,r.probability]));
    const total=runners.reduce((s,r)=>s+r.probability,0);
    return Math.min(1,permutations(numbers).reduce((sum,order)=>{let rem=total,v=1;for(const raw of order){const n=no(raw),x=p.get(n);if(x==null||rem<=0)return sum;v*=x/rem;rem-=x;}return sum+v;},0));
  }
  function build(modelRows,oddsRows,weight=0.5,threshold=0.60){
    const f=fuse(modelRows,oddsRows,weight),top7=f.runners.slice(0,7),poolSize=adaptivePoolSize(top7,threshold),pool=top7.slice(0,poolSize),prices=new Map();
    for(const o of oddsRows||[]){const l=label(o.type);if(!l.includes("3連複")&&!l.includes("三連複"))continue;const a=[o.n1,o.n2,o.n3].filter(v=>no(v));if(a.length!==3||new Set(a.map(no)).size!==3)continue;const price=num(o.odds);if(price>0)prices.set(comboKey(a),price);}
    const candidates=[];
    for(let i=0;i<pool.length;i++)for(let j=i+1;j<pool.length;j++)for(let k=j+1;k<pool.length;k++){
      const numbers=[pool[i].horseNo,pool[j].horseNo,pool[k].horseNo].sort((a,b)=>Number(a)-Number(b)),price=prices.get(comboKey(numbers))??null,p=trioProbability(numbers,f.runners);
      candidates.push({type:"3連複",kind:"trio",comb:numbers.join("-"),numbers,probability:p,odds:price,ev:price?p*price:null});
    }
    candidates.sort((a,b)=>b.probability-a.probability||(b.ev??-1)-(a.ev??-1));
    return {...f,top7,poolSize,pool,candidates,gapZThreshold:threshold};
  }
  g.KEIBALAB_TRIO_STRATEGY={fuse,adaptivePoolSize,trioProbability,build};
})(globalThis);
