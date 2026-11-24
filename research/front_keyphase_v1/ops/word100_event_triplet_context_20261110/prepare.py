from triplet_core import *
from collections import Counter
import ast,shutil
assert not (U/'FROZEN.json').exists()
rr=old.rows();fit=[r for r in rr if r['role']=='fit'];dev=[r for r in rr if r['role']=='development'];ann=read(E/'REVIEW_LABELS.json');am={r['clip_id']:r for r in ann['rows']};checks=[]
def ck(n,v):checks.append(dict(name=n,passed=bool(v)));assert v,n
ck('100common',len(fit)==899 and len({r['class_index'] for r in fit})==100)
ck('id_split',not {r['person'] for r in fit}&{r['person'] for r in dev})
ck('sha_split',not {r['source_sha256'] for r in fit}&{r['source_sha256'] for r in dev})
labels=[];counts=Counter();states=Counter()
for r in fit:
 if r['clip_id'] not in am:continue
 a=am[r['clip_id']];ck('public_review:'+r['clip_id'],sha(a['sheet'])==a['sheet_sha256'] and a['source_sha256']==r['source_sha256'])
 ss=make_segments(a,len(r['frame_id']))
 for s in ss:
  ck('anchor_support',all(s['a']<=z['frame']<=s['b'] for z in s['anchors']))
  ck('not_exact_boundary_claim',s['source'] in ['reviewed_anchor_run','reviewed_progress_anchor'])
 counts.update(s['context'] for s in ss);states.update(s['state'] for s in ss)
 labels.append(dict(clip_id=r['clip_id'],person=r['person'],source_sha256=r['source_sha256'],segments=ss))
write(U/'SEGMENT_LABELS.json',dict(rows=labels,source=str(E/'REVIEW_LABELS.json'),source_sha256=sha(E/'REVIEW_LABELS.json'),review='Reuse existing Codex visual anchors; not new human/expert annotations. Run bounds are derived, not exact RGB movement boundaries.',core_meaning='Visible annotated core appearance within center, not whole-sign completion/correctness/contact.',reversal='Reverse all coordinates and labels together, context identities retained, no normal-word loss.',masked='Zero center relations -> unconfirmed core, never positive core.'))
write(U/'DATA_AUDIT.json',dict(common_videos=899,common_words=100,fine_videos=len(labels),fine_record_ids=sorted({r['person'] for r in labels}),fine_segments=sum(counts.values()),context_counts=dict(counts),core_state_counts=dict(states),development='78 historical public conditions, same 12 normal originals in record IDs11-13. No new independent test.',personal_data=0,labels_exact_boundaries=False))
cfg=read(L/'FROZEN.json');pins={str(p):sha(p) for p in [L/'GUARDED_COMPLETE.json',L/'FINAL_RESULT.json',L/'LINK_LABELS.json',L/'GUIDE_CONTRACTS.json',L/'runs/42/model.pt',E/'REVIEW_LABELS.json',U/'SEGMENT_LABELS.json']}
write(U/'FROZEN.json',dict(seeds=[42,43,44],primary_seed=42,epochs=12,steps=38,batch=24,extra=6,lr_encoder=.0003,lr_head=.001,loss=dict(word=.2,frame=.4,event=1.,core=.3),confidence=.65,observed_coverage=.8,pins=pins,protections=cfg['protections'],adoption='Preserve normal/speed/crossing and old shape/direction; no added false positives, no missing -> similar; improve beyond one expression before promotion. Fixed primary42; otherseeds stability not selection.',no_default_change=True))
# Copy a reviewed connector with an explicit span argument; old file remains untouched.
src=(L/'evaluate.py').read_text('utf-8');start=src.index('def warning(');end=src.index('\ndef compact(')
part=src[start:end].replace('def apply_links(baseline,prob,x,ts,fh,base,contracts):','def apply_links(baseline,prob,x,ts,fh,base,contracts,context_spans):').replace('ss=spans(prob,x,ts)','ss=context_spans')
(U/'connector.py').write_text('from triplet_core import *\nimport copy\norder_state=lc.order_state\n'+part,'utf-8')
write(U/'CONNECTOR_PROVENANCE.json',dict(source=str(L/'evaluate.py'),sha256=sha(L/'evaluate.py'),changes=['Explicit context_spans supplied by event triplet classifier; original connector otherwise unchanged.']))
for p in U.glob('*.py'):ast.parse(p.read_text('utf-8'))
write(U/'PRETRAIN_CHECK.json',dict(passed=True,count=len(checks),checks=checks))
write(U/'AUTHORIZATION.json',dict(user='Current 2026-11-10 before/middle/after action context, common100, direction/order/omission public verification',implementer='Codex',no_personal=True,no_download=True,no_camera=True,no_default_replacement=True))
for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:
 (U/'before').mkdir(exist_ok=True);shutil.copy2(p,U/'before'/p.name)
write(U/'CURRENT_EXECUTION.json',dict(status='PREPARED',pid=None))
summary='동작 앞·중간·뒤 관측을 묶는 사건 판정부와 중심 구간 핵심모습 관측 판정부 준비. 100단어899공통 유지, 기존36공개FIT의 실제 관측주석 재사용. C/T3F/웹 및 기존 손모양·방향·DTW 보존. 아직 결과 미확인.'
nxt='동결3seed학습→저장78조건 비교→오배정/순서/누락·정상속도교차검산. 과거평가 재실행·개인자료·단어별예외 금지.'
p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';txt=p.read_text('utf-8-sig')
for heading,text in [('## 8.',summary),('## 9.',nxt)]:
 i=txt.index('\n',txt.index(heading))+1;txt=txt[:i]+'\n### 2026-11-10 동작 앞·중간·뒤 맥락 학습 진행\n\n'+text+'\n근거 '+str(U)+'\n\n'+txt[i:]
p.write_text(txt,'utf-8');s=read(O/'STATE.json');s.update(current_stage=U.name,next_single_action=nxt);s[U.name]=dict(status='PREPARED',summary=summary,next=nxt,execution=str(U/'CURRENT_EXECUTION.json'));write(O/'STATE.json',s)
(ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+summary+'\n\n다음 한 작업: '+nxt+'\n현재단위 '+str(U)+'\n','utf-8')
event('prepared',videos=899,fine=36,segments=sum(counts.values()))
