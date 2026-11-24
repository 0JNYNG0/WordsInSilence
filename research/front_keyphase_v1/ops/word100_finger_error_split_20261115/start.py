from common_split import *
import re,shutil,collections
from datetime import datetime
from zoneinfo import ZoneInfo
assert not(U/'PLAN.json').exists();now=datetime.now(ZoneInfo('Asia/Seoul')).isoformat();used=set();manifests=list(read(H/'EXPOSURE_AUDIT.json')['manifests'])+[str(H/'FROZEN.json'),str(H/'LEFT_COVERAGE_SEARCH.json'),str(Q/'FROZEN.json'),str(Q/'FIT_LABELS_V2.json')]
for p in manifests:used|=set(re.findall(r'NIA_SL_WORD\d+_REAL\d+_F',Path(p).read_text('utf-8-sig')))
new=[]
for person in [10,16]:
 for word in ['WORD2700','WORD2718','WORD0071','WORD0949','WORD1290','WORD2748']:
  cid=f'NIA_SL_{word}_REAL{person:02d}_F';p=source(cid)
  new.append(dict(clip_id=cid,person=person,word=word,hand='L'if word=='WORD0071'else'R',path=str(p)if p else None,seen_fine=cid in used,prior_use='Past TCN evaluation/source collection; not globally unseen'))
write(U/'INVENTORY.json',dict(candidates=new,manifests={p:sha(p)for p in manifests}))
data=np.load(P/'INPUTS.npz');old=np.load(P/'SUPPLEMENT_PREDICTIONS.npz')['pred'];newp=np.load(Q/'HISTORIC_PRED_V2.npz')['quality_only'];rows=read(P/'INPUTS.json')['rows'];lost=[]
for i,r in enumerate(rows):
 if r['role']!='development':continue
 for j in range(4):
  if data['y'][i,j]>=0 and data['availability'][i,j]and old[i,j]==data['y'][i,j]and newp[i,j]<0:lost.append(dict(**r,finger=j,truth=int(data['y'][i,j]),old=int(old[i,j]),candidate=-1,public_path=str(source(r['clip_id']))))
write(U/'PRIOR_LOSS_AUDIT.json',dict(cells=len(lost),by_clip=dict(collections.Counter(r['clip_id']for r in lost)),by_finger=dict(collections.Counter(r['finger']for r in lost)),rows=lost))
write(U/'PLAN.json',dict(at=now,steps=['A audit prior40abstentions and persistent versus isolated errors; keep all archived outputs','B source-RGB quality and shape labels separately, training01..09; select diagnostic10/reserve16 BEFORE predictions, require positive wrong-track examples in reserved labels','C train quality-aware shared finger Conv+GRU from admitted shared-finger weights using original FIT plus verified new FIT; mask wrong/unknown tracks from shape targets; three fixed seeds, preserve original baseline','D compare standalone and conservative auxiliary on diagnostic + oldDEV, freeze before reserve. Replay122. Only passing feature goes web.'],gates=['No word/person/expected-shape inputs or exceptions; C/T3F/common100/baseDTW protected','OldDEV new confident errors <= baseline and correct loss <=2%; diagnostic and reserve both lower wrong counts with <=2% correct loss; report abstentions separately','Quality sensitivity needs confirmed positive observations from held-out source, not synthetic-only; source ambiguity is not negative truth','No new normal corrections or loss of previous coarse/direction/order/known finger detail20 on public122','No private footage/download/camera; no tuning on14/15/17/18 or reserve16'],limitations='Research RGB labels by Codex, not independent expert; source dataset is public KSL, local original video only'))
write(U/'PROTECTED.json',read(Q/'PROTECTED.json'));(U/'before').mkdir()
for p in [R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copy2(p,U/'before'/p.name)
s=read(O/'STATE.json');s['current_followup']=U.name;s['next_single_action']='관절 품질 오류와 정상 관절의 손모양 분류 오류를 분리한 정답 보강·학습·기존/예약 비교';s[U.name]=dict(status='SOURCE_AUDIT',plan=str(U/'PLAN.json'),at=now);write(O/'STATE.json',s)
doc=(R/'WordsInSilence_PROJECT_CONTEXT.md').read_text('utf-8-sig')
for h in ['## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','## 9. 다음 단일 작업']:doc=doc.replace(h+'\n\n',h+'\n\n### 2026-11-15 관절 오류·손모양 분류 오류 분리 보완 진행\n\n'+s['next_single_action']+'. `research/front_keyphase_v1/ops/'+U.name+'/PLAN.json`. 기존결과 before/ 보존, 이전14/15/17/18 재튜닝 금지, 사용자자료 미사용·C/T3F/현재웹 보존.\n\n',1)
(R/'WordsInSilence_PROJECT_CONTEXT.md').write_text(doc,'utf-8');(R/'NEW_CHAT_HANDOFF.md').write_text('# 관절 오류·손모양 분류 오류 분리 보완 진행\n\n'+s['next_single_action']+'\n\n현재계약 `research/front_keyphase_v1/ops/'+U.name+'/PLAN.json`; before/ 이전요약보존. 웹 G6-finger-20261115 유지. 사용자촬영/카메라 미사용.\n','utf-8')
write(U/'CURRENT_EXECUTION.json',dict(status='SOURCE_AUDIT',pid=None));print(json.dumps(dict(candidates=new,lost_by_clip=dict(collections.Counter(r['clip_id']for r in lost))),ensure_ascii=False))
