from holdout_common import *
import shutil,re
from datetime import datetime
from zoneinfo import ZoneInfo
assert not(U/'FROZEN.json').exists()
now=datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
exposure_files=[P/'INPUTS.json',D/'FINGER_INPUTS.json',D/'INTEGRATED_RESULTS.json',O/'word100_public_g4_20261114/EXTRACTED.json',O/'word100_handshape_transfer_20261113/INPUT_MANIFEST.json',O/'word100_feature_expansion_20261113/INPUTS_NEW.json',O/'demo_handshape_learning_20261113/INPUTS.json']
used=set();exposure={}
for p in exposure_files:
 ids=set(re.findall(r'NIA_SL_WORD\d+_REAL\d+_F',p.read_text('utf-8-sig')));used|=ids;exposure[str(p)]=dict(sha256=sha(p),clips=sorted(ids))
words=['WORD0003','WORD0071','WORD0949','WORD2551','WORD1290','WORD2700','WORD2748','WORD2718'];sources=[]
for person in [17,18]:
 for word in words:
  cid=f'NIA_SL_{word}_REAL{person}_F';assert cid not in used,cid
  metadata=O/'word100_t3f_heldout_20261019/data/records'/f'{cid}.json';r=read(metadata);path=O/'word100_t3f_heldout_20261019/data/clips'/cid/'source.mp4'
  assert sha(path)==r['source_sha256']
  sources.append(dict(clip_id=cid,word=word,label=r['canonical_label'],person=person,role='diagnostic'if person==17 else'confirmation_reserved',path=str(path),sha256=sha(path),metadata=str(metadata),metadata_sha256=sha(metadata),prior_use='Earlier TCN heldout evaluation. Not in listed fine-finger candidate inputs or G4. Not globally unseen.',selection='Fixed vocabulary representing both-hand index/open transitions, front-facing index, number combinations; select before current predictions'))
pins=read(P/'BEFORE_PINS.json')
for p in read(P/'CONNECTION.json')['files']:pins[p]=sha(p)
for p in list(P.glob('*.py'))+list(P.glob('*/*.pt'))+list(D.glob('finger_models_v2/*.pt')):pins[str(p)]=sha(p)
write(U/'PROTECTED.json',pins);write(U/'EXPOSURE_AUDIT.json',dict(manifests=exposure,selected_overlap=sorted({r['clip_id']for r in sources}&used),caveat='Prior TCN evaluation remains disclosed; freshness is relative to fine-finger candidate selection. Source review itself does not imply new labels were used for optimization.'))
write(U/'FROZEN.json',dict(at=now,sources=sources,model_pins={str(p):sha(p)for p in list(P.glob('*/*.pt'))+list(D.glob('finger_models_v2/*.pt'))},steps=['A Unused-source audit, direct RGB temporal labels before predictions','B Evaluate frozen existing observer on diagnostic17, separate observation errors from annotation/detector/context limitations','C If supported, repair generic evidence or train only earlier FIT identities; freeze before any reserved18 score','D Compare normal and errors, protected regressions; release only passed limited functions, no arbitrary handshape labels'],roles={'fit':'Existing people01..09 only; never17/18','diagnostic':'17 may guide repair, disclose exposure','confirmation':'18 held until repair/code frozen; if fails, no promotion by repeated tuning on18'},label_policy='1 visibly extended,0 visibly flexed,-1 ambiguous. Thumb excluded. Direction/front-facing recorded from RGB, not inferred label from word. Freeze continuous intervals and anchor labels with source images; do not impute hidden fingers from expected sign.',gates=['No new known-finger errors on confirmation','No old-correct-to-wrong on historic DEV','No new normal correction cues on existing122conditions','Unknown/overlap not error, original candidates/order/C/T3F preserved','No private training, webcam or target-word observer input'],public_cache_only=True,web_change_conditional=True,implementer='Codex',reviewer='Same Codex separate saved-output audit, not independent KSL expert'))
b=U/'before';b.mkdir()
for p in [R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copy2(p,b/p.name)
state=read(O/'STATE.json');state['current_followup']=U.name;state['next_single_action']='未사용 세부 손가락 후보용 공개16영상 정답 고정→17진단→필요한 공통 보완/학습→18보류검증 및 기존회귀→통과만 연결';state[U.name]=dict(status='A_SOURCE_AND_LABEL_REVIEW',at=now,plan=str(U/'FROZEN.json'));write(O/'STATE.json',state)
doc=(R/'WordsInSilence_PROJECT_CONTEXT.md').read_text('utf-8-sig')
for h in ['## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','## 9. 다음 단일 작업']:
 doc=doc.replace(h+'\n\n',h+'\n\n### 2026-11-15 미사용 공개 손가락 검증 및 조건부 보완 진행\n\n'+state['next_single_action']+'. `ops/'+U.name+'/FROZEN.json` 확인. 개인 촬영·C·T3-F 보호, 실제 웹캠 보류.\n\n',1)
(R/'WordsInSilence_PROJECT_CONTEXT.md').write_text(doc,'utf-8')
(R/'NEW_CHAT_HANDOFF.md').write_text('# 미사용 공개 손가락 검증 진행\n\n'+state['next_single_action']+'\n\n현재 `research/front_keyphase_v1/ops/'+U.name+'/FROZEN.json`, `CURRENT_EXECUTION.json`. 기존 웹 보조 G6-finger-20261115 보존. 17은 진단,18은 수정 고정 후 확인. 과거TCN평가 이력은 있으며 전역 미사용자료라고 부르지 않음. 사용자 촬영 금지/학습·튜닝 금지. W:미연결, C:캐시 사용.\n','utf-8')
write(U/'CURRENT_EXECUTION.json',dict(status='SOURCE_REVIEW',at=now,pid=None))
print('sources',len(sources),'overlap',0)
