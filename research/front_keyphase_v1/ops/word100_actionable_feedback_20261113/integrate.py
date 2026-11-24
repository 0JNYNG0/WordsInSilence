"""Add only the verified message contract to 8821; keep model files unchanged."""
from pathlib import Path
import shutil,json,hashlib
U=Path(__file__).resolve().parent;WEB=U.parent/'web_tutorial_live_integration_20261113'
backup=U/'before_web';backup.mkdir(exist_ok=True)
paths=[WEB/'server.py',WEB/'site/live.js',WEB/'site/feedback.js',WEB/'site/index.html',WEB/'site/live.css']
for p in paths:
    dst=backup/p.name
    if not dst.exists():shutil.copy2(p,dst)

p=WEB/'server.py';s=p.read_text('utf-8')
assert 'from coaching import attach' not in s
s=s.replace('from contract import QuestProgress, SUPPORTED, TARGETS','from contract import QuestProgress, SUPPORTED, TARGETS\nsys.path.insert(0,str(O/"word100_actionable_feedback_20261113"))\nfrom coaching import attach')
s=s.replace("r=copy.deepcopy(SERVICE.result)","r=attach(SERVICE.result)")
s=s.replace("result.update(source='public_integration_replay'", "result=attach(result)\n                    result.update(source='public_integration_replay'")
s=s.replace("if p.is_relative_to(OLD.resolve()) and p.suffix in ['.mp4','.json']:self.file(p);return", "if p.is_relative_to(OLD.resolve()) and p.suffix in ['.mp4','.json']:\n                    if p.suffix=='.json' and p.name=='result.json':self.reply(attach(read(p)))\n                    else:self.file(p)\n                    return")
p.write_text(s,encoding='utf-8')

p=WEB/'site/live.js';s=p.read_text('utf-8')
s=s.replace("}else say(correction?", "}else if(data.coaching)say(data.coaching.summary);else say(correction?")
p.write_text(s,encoding='utf-8')

p=WEB/'site/index.html';s=p.read_text('utf-8')
s=s.replace('<div id="feedback-cards"></div>','<p id="coaching-summary"></p><div id="coaching-actions" aria-label="교정할 순간"></div><div id="feedback-cards"></div>')
s=s.replace('<button id="compare-stop">멈춤</button>', '<button id="compare-stop">멈춤</button><button id="replay-focus" hidden>이 부분만 다시 보기</button>')
p.write_text(s,encoding='utf-8')

p=WEB/'site/feedback.js';s=p.read_text('utf-8')
s=s.replace('referenceOnly=false;', 'referenceOnly=false,focusCue=null,clipEnd=null;',1)
s=s.replace('function selectCard(i){active=i;', 'function selectCard(i){focusCue=null;clipEnd=null;$(\'replay-focus\').hidden=true;active=i;')
# The existing card selector remains available. A confirmed component cue chooses
# its own measured interval instead of starting at an entire stage's beginning.
s=s.replace("export async function openFeedback", """function showCue(cue){
 const i=result.cards.findIndex(c=>cue.card_ids.includes(c.id));if(i<0)return;
 selectCard(i);focusCue=cue;setReferenceOnly(false);$('query-video').pause();
 $('feedback-message').textContent=cue.start_seconds.toFixed(1)+'초 부근 · '+cue.observation;
 $('feedback-instruction').textContent='다시 해 볼 방법: '+cue.action;
 $('query-video').currentTime=Math.max(0,cue.start_seconds-.15);$('replay-focus').hidden=false;sync();
}
$('replay-focus').onclick=()=>{if(!focusCue)return;setReferenceOnly(false);clipEnd=Math.min(result.timestamp_us.at(-1)/1e6,focusCue.end_seconds+.25);$('query-video').currentTime=Math.max(0,focusCue.start_seconds-.15);$('query-video').play().catch(()=>{});};
export async function openFeedback""")
s=s.replace("result=data;ref=reference;active=-1;desired=null;", "result=data;ref=reference;active=-1;desired=null;focusCue=null;clipEnd=null;$('replay-focus').hidden=true;$('coaching-summary').textContent=data.coaching?.summary||'';$('coaching-actions').replaceChildren();for(const cue of data.coaching?.cues||[]){const b=document.createElement('button');b.textContent=cue.start_seconds.toFixed(1)+'초 · '+(cue.kind==='direction'?'이동 방향':'손모양')+' 확인';b.onclick=()=>showCue(cue);$('coaching-actions').append(b);}")
s=s.replace("if(data.cards.length)selectCard(Math.max(0,data.cards.findIndex(c=>c.display_status==='difference')))","if(data.coaching?.cues?.length)showCue(data.coaching.cues[0]);else if(data.cards.length)selectCard(Math.max(0,data.cards.findIndex(c=>c.display_status==='difference')))")
s=s.replace("$('compare-play').onclick=()=>{active=-1;", "$('compare-play').onclick=()=>{clipEnd=null;active=-1;")
s=s.replace("$('compare-stop').onclick=()=>$('query-video').pause();", "$('compare-stop').onclick=()=>{clipEnd=null;$('query-video').pause();};")
s=s.replace("$('feedback-seek').oninput=()=>{setReferenceOnly(false);", "$('feedback-seek').oninput=()=>{clipEnd=null;setReferenceOnly(false);")
s=s.replace("function tick(){if(!$('query-video').paused)sync();", "function tick(){if(!$('query-video').paused){sync();if(clipEnd!==null&&$('query-video').currentTime>=clipEnd){$('query-video').pause();clipEnd=null;}}")
p.write_text(s,encoding='utf-8')

p=WEB/'site/live.css';s=p.read_text('utf-8');s+='''
#coaching-summary{font-size:14px;line-height:1.7;font-weight:600;padding:10px 12px;border-left:3px solid #29466d;background:#eaf0f7;margin:10px 0}
#coaching-summary:empty{display:none}#coaching-actions{display:flex;gap:6px;flex-wrap:wrap}#coaching-actions button{font:inherit;font-size:12px;border:1px solid #b98253;color:#523719;background:#fff7eb;padding:8px 12px;border-radius:5px;cursor:pointer}#replay-focus{color:#29466d;font-weight:700}
''';p.write_text(s,encoding='utf-8')
(U/'WEB_PATCH.json').write_text(json.dumps(dict(changed=[str(p) for p in paths],model_replaced=False,scope='coaching contract and precise replay controls; no inference/threshold change'),ensure_ascii=False,indent=2),encoding='utf-8')
print('Verified coaching contract patched into isolated8821; restart needed.')
