#!/usr/bin/env python3
"""Local trial viewer for a sweep run — ALL trials, no sampling.

Because one self-contained HTML with all 19 models' full text is ~267 MB (won't open),
this writes ONE page per model (all 10,100 trials, ~10-14 MB each) plus an index.html
that links to them and shows the per-model judge-label summary. Open index.html in a
browser (file://) — no server, no network.

Baseline (clean, no-swap) text is deduplicated per prompt (100 unique / model) to keep
each page small. The trial list is paginated (the data is all embedded; only the nav is
paged) so even 10,100 trials render smoothly.

Usage: uv run python scripts/make_viewer.py [run_dir]
"""
from __future__ import annotations

import glob
import json
import os
import sys

RUN = sys.argv[1] if len(sys.argv) > 1 else "results/runs/sweep-v2"
EXPECTED = 10100
LABELS = ("ignored", "corrected", "flagged", "derailed")

CSS = """<style>
:root{--paper:#F7F6F3;--panel:#FFFFFF;--ink:#212528;--muted:#6C6F73;--blue:#2F5A8F;
 --blue-bg:#DEE7F2;--rust:#A64B22;--rust-bg:#F9E8DC;--line:#DAD7CF;--flag:#8A6D1D;
 --flag-bg:#F3E9C9;--aware:#8C2F39;--aware-bg:#F5DEE0;--green:#2E6B47;--green-bg:#DCEDE2;
 --violet:#5B4B8A;--violet-bg:#E6E1F2}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
a{color:var(--blue)}
header{display:flex;align-items:baseline;gap:14px;padding:14px 20px 10px;border-bottom:1px solid var(--line);flex-wrap:wrap}
header h1{font-size:17px;font-weight:650;margin:0;letter-spacing:-.01em}
header .sub{color:var(--muted);font-size:12.5px}
.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:10px 20px;border-bottom:1px solid var(--line);background:var(--panel)}
.chip{font:inherit;font-size:12.5px;padding:4px 10px;border:1px solid var(--line);border-radius:999px;background:var(--paper);cursor:pointer;color:var(--ink)}
.chip[aria-pressed="true"]{background:var(--blue);border-color:var(--blue);color:#fff}
.chip:focus-visible,.trial:focus-visible,.pg:focus-visible{outline:2px solid var(--blue);outline-offset:1px}
.mstats{font-variant-numeric:tabular-nums;color:var(--muted);font-size:12.5px;margin-left:auto}
.legend{padding:8px 20px;font-size:12px;line-height:1.7;color:var(--muted);border-bottom:1px solid var(--line);background:var(--paper)}
.legend b{color:var(--ink)}.legend em{font-style:normal;color:var(--ink);font-weight:500}
.lg{font-weight:600;padding:1px 7px;border-radius:999px;font-size:11px;white-space:nowrap}
.lg-lex{background:var(--flag-bg);color:var(--flag)}.lg-judge{background:var(--green-bg);color:var(--green)}
.lg-think{background:var(--violet-bg);color:var(--violet)}
main{display:grid;grid-template-columns:300px 1fr;min-height:calc(100vh - 150px)}
nav{border-right:1px solid var(--line);display:flex;flex-direction:column;max-height:calc(100vh - 150px);background:var(--panel)}
.pager{display:flex;gap:6px;align-items:center;padding:7px 10px;border-bottom:1px solid var(--line);font-size:12px;color:var(--muted)}
.pg{font:inherit;border:1px solid var(--line);background:var(--paper);border-radius:4px;padding:2px 8px;cursor:pointer;color:var(--ink)}
.pg[disabled]{opacity:.4;cursor:default}
.list{overflow-y:auto;flex:1}
.trial{display:block;width:100%;text-align:left;border:0;border-bottom:1px solid var(--line);background:none;padding:9px 12px;cursor:pointer;font:inherit;color:var(--ink)}
.trial:hover{background:var(--paper)}.trial[aria-current="true"]{background:var(--blue-bg)}
.trial .q{font-size:13px;line-height:1.35;display:block}
.trial .meta{display:flex;gap:5px;margin-top:4px;flex-wrap:wrap}
.badge{font-size:10.5px;padding:1px 7px;border-radius:999px;letter-spacing:.03em}
.b-ent{background:var(--blue-bg);color:var(--blue)}.b-rep{background:#ECEAE4;color:#5A5D61}
.b-flag{background:var(--flag-bg);color:var(--flag)}.b-aware{background:var(--aware-bg);color:var(--aware);font-weight:600}
.b-judge{background:var(--green-bg);color:var(--green)}.b-jaware{background:var(--green);color:#fff;font-weight:600}
.b-think{background:var(--violet-bg);color:var(--violet)}
.stage{padding:16px 20px;overflow-y:auto;max-height:calc(100vh - 150px)}
.trialhead{display:flex;flex-wrap:wrap;gap:8px 18px;align-items:baseline;margin-bottom:12px}
.trialhead .q{font-size:15.5px;font-weight:650}
.stat{font-size:12.5px;color:var(--muted);font-variant-numeric:tabular-nums}.stat b{color:var(--ink);font-weight:600}
.panes{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.pane{background:var(--panel);border:1px solid var(--line);border-radius:6px;min-width:0}
.pane h2{font-size:11px;text-transform:uppercase;letter-spacing:.09em;font-weight:650;margin:0;padding:8px 14px;border-bottom:1px solid var(--line);color:var(--muted)}
.pane.right h2{color:var(--rust)}.pane.left h2{color:var(--blue)}
.txt{padding:12px 14px;white-space:pre-wrap;overflow-wrap:break-word;font:12.5px/1.65 ui-monospace,"SF Mono",Menlo,Consolas,monospace;max-height:70vh;overflow-y:auto}
mark{border-radius:3px;padding:0 2px}mark.trig{background:var(--blue-bg);color:var(--blue)}
mark.swap{background:var(--rust-bg);color:var(--rust);font-weight:600}
.thought{background:#F1EFF7;border-left:3px solid #B4A9D6;border-radius:3px;padding:6px 10px;margin:0 0 10px;color:#4A4655}
.thought .tlabel{display:block;font:600 10px/1 system-ui,sans-serif;text-transform:uppercase;letter-spacing:.09em;color:var(--violet);margin-bottom:5px}
.empty{color:var(--muted);font-style:italic}
/* index */
.idx{max-width:1050px;margin:0 auto;padding:22px 20px}
table.idx-t{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
table.idx-t th,table.idx-t td{padding:6px 9px;border-bottom:1px solid var(--line);text-align:right;font-size:13px}
table.idx-t th{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);font-weight:600}
table.idx-t td.m,table.idx-t th.m{text-align:left}
table.idx-t td.m a{font-weight:600;text-decoration:none}
table.idx-t tr:hover td{background:var(--panel)}
@media (max-width:900px){main{grid-template-columns:1fr}nav{max-height:34vh}.panes{grid-template-columns:1fr}}
</style>"""

LEGEND = """<div class="legend">
<b>Two independent verdicts per trial.</b>
<span class="lg lg-lex">Lexical</span> = keyword detector: <em>Flagged</em> (said something about the
oddity) &rarr; <em>Aware</em> (named that a word was swapped).
<span class="lg lg-judge">Judge</span> = Gemma gives one label:
<em>ignored</em> / <em>corrected</em> / <em>flagged</em> (remarked the word was out of place) /
<em>derailed</em> (incoherence or fixation). <em>Switch-aware</em> = a strict step beyond flagging:
named that a word was <em>switched / substituted</em> (rare). Note <em>derailed</em> is disruption,
not necessarily recognition.
<span class="lg lg-think">Thinking</span> = the reply contains visible reasoning.</div>"""

PAGE_JS = r"""<script>
const D = __DATA__;                 // {model, clean:{pid:text}, trials:[...], stats:{}}
const PAGE = 250;
const esc = s => (s||'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const hiTrig = s => esc(s).replace(/\b(the)\b/gi,'<mark class="trig">$1</mark>');
const hiSwap = (s,rep)=>{const r=rep.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
  return esc(s).replace(new RegExp('\\b('+r+')\\b','gi'),'<mark class="swap">$1</mark>');};
function segment(s){const c=s.match(/<\/think\s*>/i);
  if(c){let t=s.slice(0,c.index).replace(/^\s*<think\b[^>]*>/i,'');return{thought:t,rest:s.slice(c.index+c[0].length)};}
  const o=s.match(/^\s*<think\b[^>]*>/i); if(o)return{thought:s.slice(o[0].length),rest:''}; return null;}
function renderTxt(s,hi){const g=segment(s); if(!g)return hi(s);
  let out='<div class="thought"><span class="tlabel">thinking</span>'+hi(g.thought)+'</div>';
  out+= g.rest.trim()?hi(g.rest.trim()):'<span class="empty">(no final answer — budget spent thinking)</span>';
  return out;}
const $=id=>document.getElementById(id);
let cur={f:'all',gi:0,page:0};
const FILT={all:()=>true, ent:t=>t.rep[0]==='M'&&t.rep.startsWith('Mandela'),
  jflag:t=>!!(t.jm&1), jder:t=>!!(t.jm&4), jcorr:t=>!!(t.jm&2),
  jaware:t=>!!(t.jm&8), lflag:t=>t.flag, laware:t=>t.aware, think:t=>t.th};
// jm bitmask: 1=flagged 2=corrected 4=derailed 8=switch-aware (labels are INDEPENDENT
// under the multi-label panel judge; a trial can carry several). 0 = ignored.
const jmStr = m => m ? ['flagged','corrected','derailed','switch-aware']
  .filter((_,i)=>m&(1<<i)).join('+') : 'ignored';
function filtered(){return D.trials.filter(FILT[cur.f]);}
document.querySelectorAll('.chip').forEach(c=>c.onclick=()=>{
  document.querySelectorAll('.chip').forEach(x=>x.setAttribute('aria-pressed','false'));
  c.setAttribute('aria-pressed','true'); cur.f=c.dataset.f; cur.gi=0; cur.page=0; render();});
document.addEventListener('keydown',e=>{
  if(e.key!=='ArrowDown'&&e.key!=='ArrowUp')return;
  if(/INPUT|SELECT/.test(document.activeElement.tagName))return; e.preventDefault();
  const n=filtered().length; cur.gi=Math.min(n-1,Math.max(0,cur.gi+(e.key==='ArrowDown'?1:-1)));
  cur.page=Math.floor(cur.gi/PAGE); render();});
function badge(c,t){return '<span class="badge '+c+'">'+t+'</span>';}
function render(){
  const F=filtered(), n=F.length, pages=Math.max(1,Math.ceil(n/PAGE));
  cur.page=Math.min(cur.page,pages-1);
  const start=cur.page*PAGE, slice=F.slice(start,start+PAGE);
  $('pgi').textContent = n? (start+1)+'–'+Math.min(start+PAGE,n)+' of '+n.toLocaleString():'0';
  $('prev').disabled=cur.page<=0; $('next').disabled=cur.page>=pages-1;
  $('list').innerHTML = slice.map((t,j)=>{const gi=start+j;
    return '<button class="trial" aria-current="'+(gi===cur.gi)+'" data-gi="'+gi+'">'+
      '<span class="q">'+esc(t.q)+'</span><span class="meta">'+
      (t.rep.startsWith('Mandela')?badge('b-ent',esc(t.rep)):badge('b-rep',esc(t.rep)))+
      (t.aware?badge('b-aware','sw-aware(lex)'):t.flag?badge('b-flag','flag(lex)'):'')+
      ((t.jm&8)?badge('b-jaware','judge: switch-aware'):t.jm?badge('b-judge','judge: '+jmStr(t.jm)):'')+
      (t.th?badge('b-think','thinking'):'')+'</span></button>';}).join('')
    || '<div class="txt empty">no trials match this filter</div>';
  $('list').querySelectorAll('.trial').forEach(b=>b.onclick=()=>{cur.gi=+b.dataset.gi;render();});
  const t=F[cur.gi]; if(!t){$('thead').innerHTML='';$('clean').innerHTML='';$('rx').innerHTML='';return;}
  const clean=D.clean[t.pid]||'';
  $('thead').innerHTML='<span class="q">'+esc(t.q)+'</span>'+
    '<span class="stat">the &rarr; <b>'+esc(t.rep)+'</b></span>'+
    '<span class="stat">swaps <b>'+t.subs+'</b></span>'+
    '<span class="stat">&Delta;surprisal <b>'+(t.ds==null?'—':(t.ds>0?'+':'')+t.ds.toFixed(2))+'</b></span>'+
    '<span class="stat">judge: <b>'+(t.jm==null?'—':jmStr(t.jm))+'</b></span>'+
    '<span class="stat">lexical: <b>'+(t.aware?'switch-aware':t.flag?'flagged':'quiet')+'</b></span>'+
    '<span class="stat">thinking: <b>'+(t.th?'yes':'—')+'</b></span>';
  $('clean').innerHTML=clean?renderTxt(clean,hiTrig):'<span class="empty">(no clean text)</span>';
  $('rx').innerHTML=t.rx?renderTxt(t.rx,s=>hiSwap(s,t.rep)):'<span class="empty">(empty reaction)</span>';
  const b=$('list').querySelector('.trial[aria-current="true"]'); if(b)b.scrollIntoView({block:'nearest'});
}
$('prev').onclick=()=>{cur.page--;render();}; $('next').onclick=()=>{cur.page++;render();};
render();
</script>"""


def model_page(model, data):
    chips = [("all", "All"), ("ent", "Mandela effect"), ("jflag", "Flagged (judge)"),
             ("jder", "Derailed (judge)"), ("jcorr", "Corrected (judge)"),
             ("jaware", "Switch-aware (judge)"), ("lflag", "Flagged (lex)"),
             ("laware", "Switch-aware (lex)"), ("think", "Thinking")]
    chip_html = "".join(
        f'<button class="chip" data-f="{f}" aria-pressed="{"true" if f=="all" else "false"}">{lbl}</button>'
        for f, lbl in chips)
    st = data["stats"]
    n = st["jn"] or 1
    lp = lambda k: f"{round(100*st['lab'].get(k,0)/n)}%"
    statline = (f"{st['n']:,} trials &middot; judge: flag {lp('flagged')} / ign {lp('ignored')} "
                f"/ derail {lp('derailed')} / corr {lp('corrected')} &middot; switch-aware "
                f"{100*st['aware']/n:.1f}% &middot; think {round(100*st['think']/st['n'])}% "
                f"&middot; {st['subs']:,} swaps &middot; untouched {100*st['untouched']/st['n']:.1f}%")
    head = (f'<title>{model} — Sleight of Word</title>{CSS}'
            f'<header><h1>{model}</h1><span class="sub"><a href="index.html">&larr; all models</a> '
            f'&middot; baseline vs. word-swapped &middot; all trials</span>'
            f'<span class="mstats">{statline}</span></header>'
            f'<div class="controls">{chip_html}</div>{LEGEND}'
            '<main><nav>'
            '<div class="pager"><button class="pg" id="prev">&lsaquo; prev</button>'
            '<button class="pg" id="next">next &rsaquo;</button><span id="pgi"></span></div>'
            '<div class="list" id="list"></div></nav>'
            '<section class="stage"><div class="trialhead" id="thead"></div>'
            '<div class="panes"><div class="pane left"><h2>Baseline &mdash; clean run</h2>'
            '<div class="txt" id="clean"></div></div>'
            '<div class="pane right"><h2>Sleight of word &mdash; swapped</h2>'
            '<div class="txt" id="rx"></div></div></div></section></main>')
    payload = {"model": model, "clean": data["clean"], "trials": data["trials"]}
    return head + PAGE_JS.replace("__DATA__", json.dumps(payload, separators=(",", ":")))


INDEX_LEGEND = [
    ("Model", "The judged chat model. Click to open all 10,100 of its trials."),
    ("Flagged", "Judge: the reply explicitly remarked that a word was strange, wrong, or out of place."),
    ("Ignored", "Judge: the reply continued as if nothing was wrong — no sign it registered the odd word."),
    ("Derailed", "Judge: the reply became incoherent, repetitive, off-topic, or fixated on the odd word."),
    ("Corrected", "Judge: the reply still delivered the correct answer despite the swap."),
    ("Switch aware", "Judge (strict): the reply explicitly stated a word had been switched / substituted."),
    ("Think", "Share of trials whose output contained visible reasoning / thinking text."),
    ("Swaps", "Total trigger occurrences (“the”) replaced across the model’s trials."),
    ("Untouched", "Share of trials with NO swap at all — the model never emitted “the”, so the output is identical to baseline and is not judged. Judge-label % are over the remaining (judged) trials."),
]


def index_page(summaries, run):
    data = []
    for m, s in summaries.items():
        n = s["jn"] or 1
        p = lambda k: round(100 * s["lab"].get(k, 0) / n)
        data.append({"m": m, "flagged": p("flagged"), "ignored": p("ignored"),
                     "derailed": p("derailed"), "corrected": p("corrected"),
                     "aware": round(100*s["aware"]/n, 1), "think": round(100*s["think"]/s["n"]),
                     "swaps": s["subs"], "untouched": round(100*s["untouched"]/s["n"], 1)})
    cols = [("m", "Model", "alpha"), ("flagged", "Flagged", "num"), ("ignored", "Ignored", "num"),
            ("derailed", "Derailed", "num"), ("corrected", "Corrected", "num"),
            ("aware", "Switch aware", "num"), ("think", "Think", "num"),
            ("swaps", "Swaps", "num"), ("untouched", "Untouched", "num")]
    legend = "".join(f'<div class="ldef"><b>{k}</b> — {v}</div>' for k, v in INDEX_LEGEND)
    return (f'<title>Sleight of Word — {run}</title>{CSS}'
            '<style>.idx-t th.sortable{cursor:pointer;user-select:none;white-space:nowrap}'
            '.idx-t th.sortable:hover{color:var(--ink)}.arrow{opacity:.35;font-size:10px;margin-left:3px}'
            'th.sorted .arrow{opacity:1;color:var(--blue)}'
            '.ldef{font-size:12px;line-height:1.5;color:var(--muted);margin:2px 0}'
            '.ldef b{color:var(--ink);font-weight:600;display:inline-block;min-width:96px}'
            '.legbox{margin-top:18px;padding:12px 14px;background:var(--panel);border:1px solid var(--line);border-radius:6px}</style>'
            f'<header><h1>Sleight of Word</h1><span class="sub">{run} &middot; '
            f'{len(summaries)} models &middot; click a model for all its trials &middot; click a column to sort</span></header>'
            '<div class="idx"><table class="idx-t"><thead><tr id="hrow"></tr></thead>'
            '<tbody id="tb"></tbody></table>'
            f'<div class="legbox"><div class="ldef" style="margin-bottom:6px"><b style="min-width:0">Columns</b></div>{legend}'
            '<div class="ldef" style="margin-top:8px;font-style:italic">Judge labels come from a jury of '
            'three LLM judges from distinct lineages; a label is assigned when at least two jurors agree. '
            'Percentages are of judged trials. Labels are INDEPENDENT '
            '(a reply can be flagged AND corrected AND derailed at once), so rows need not '
            'sum to 100%; Ignored = none of the labels apply.</div></div></div>'
            f'<script>\nconst ROWS={json.dumps(data)};\nconst COLS={json.dumps(cols)};\n'
            "let sk='flagged',sd=-1;\n"
            "function draw(){const hr=document.getElementById('hrow');\n"
            " hr.innerHTML=COLS.map(c=>{const on=c[0]===sk;const ar=on?(sd<0?'\\u25bc':'\\u25b2'):'\\u25c6';\n"
            "  return `<th class=\"sortable${c[0]==='m'?' m':''}${on?' sorted':''}\" data-k=\"${c[0]}\">${c[1]}<span class=\"arrow\">${ar}</span></th>`;}).join('');\n"
            " hr.querySelectorAll('th').forEach(th=>th.onclick=()=>{const k=th.dataset.k;if(k===sk)sd=-sd;else{sk=k;sd=(k==='m'?1:-1);}draw();});\n"
            " const rs=[...ROWS].sort((a,b)=>{let x=a[sk],y=b[sk];if(sk==='m')return sd*x.localeCompare(y);return sd*(x-y);});\n"
            " const fmt=(k,v)=>k==='m'?`<a href=\"viewer-${v}.html\">${v}</a>`:(k==='swaps'?v.toLocaleString():v+'%');\n"
            " document.getElementById('tb').innerHTML=rs.map(r=>'<tr>'+COLS.map(c=>`<td class=\"${c[0]==='m'?'m':''}\">${fmt(c[0],r[c[0]])}</td>`).join('')+'</tr>').join('');\n"
            "}\ndraw();\n</script>")


def main():
    files = sorted(glob.glob(os.path.join(RUN, "*.judged.jsonl")))
    summaries = {}
    for f in files:
        m = os.path.basename(f)[:-len(".judged.jsonl")]
        n = sum(1 for _ in open(f, encoding="utf-8"))
        if n < EXPECTED:
            print(f"  skip {m} ({n}/{EXPECTED})")
            continue
        clean, trials = {}, []
        stats = {"n": 0, "think": 0, "aware": 0, "subs": 0, "jn": 0, "untouched": 0, "lab": {}}
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            te = r.get("text_evidence") or {}
            mt = r.get("metrics") or {}
            th = r.get("thinking") or {}
            j = r.get("judge") or {}
            pid = r["prompt_id"]
            if pid not in clean:
                clean[pid] = r.get("clean_text") or ""
            ds = mt.get("delta_surprisal")
            # jm bitmask 1=flagged 2=corrected 4=derailed 8=switch-aware.
            # Multi-label (panel) verdicts use independent booleans; legacy single-label
            # verdicts fall back to `label`. None = trial not judged.
            def eff(k):
                v = j.get(k)
                return bool(v) if v is not None else (j.get("label") == k)
            judged = bool(j.get("label")) or any(
                j.get(k) is not None for k in ("flagged", "corrected", "derailed"))
            jm = None
            if judged:
                jm = ((1 if eff("flagged") else 0) | (2 if eff("corrected") else 0)
                      | (4 if eff("derailed") else 0) | (8 if j.get("aware") else 0))
            trials.append({"pid": pid, "q": r["query"], "rep": r["replacement"],
                           "rx": r.get("reaction_text") or "",
                           "subs": mt.get("num_substitutions") or 0,
                           "ds": round(ds, 2) if ds is not None else None,
                           "flag": bool(te.get("flagged")), "aware": bool(te.get("aware")),
                           "jm": jm,
                           "th": bool(th.get("detected"))})
            stats["n"] += 1
            stats["subs"] += mt.get("num_substitutions") or 0
            stats["untouched"] += 1 if (mt.get("num_substitutions") or 0) == 0 else 0
            stats["think"] += 1 if th.get("detected") else 0
            if jm is not None:
                stats["jn"] += 1
                for bit, name in ((1, "flagged"), (2, "corrected"), (4, "derailed")):
                    if jm & bit:
                        stats["lab"][name] = stats["lab"].get(name, 0) + 1
                if jm == 0:
                    stats["lab"]["ignored"] = stats["lab"].get("ignored", 0) + 1
                if jm & 8:
                    stats["aware"] += 1
        out = os.path.join(RUN, f"viewer-{m}.html")
        open(out, "w", encoding="utf-8").write(model_page(m, {"clean": clean, "trials": trials, "stats": stats}))
        summaries[m] = stats
        print(f"  {m}: {len(trials):,} trials -> {os.path.getsize(out)//1024//1024} MB")
    idx = os.path.join(RUN, "index.html")
    open(idx, "w", encoding="utf-8").write(index_page(summaries, os.path.basename(RUN.rstrip("/"))))
    print(f"\nwrote {idx} ({len(summaries)} models)")
    print(f"open it:  file://{os.path.abspath(idx)}")


if __name__ == "__main__":
    main()
