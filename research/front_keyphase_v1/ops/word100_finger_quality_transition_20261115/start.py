from quality_common import *
import re,shutil
from datetime import datetime
from zoneinfo import ZoneInfo
assert not(U/'PLAN.json').exists();now=datetime.now(ZoneInfo('Asia/Seoul')).isoformat();used=set();manifests=list(read(H/'EXPOSURE_AUDIT.json')['manifests'])+[str(H/'FROZEN.json'),str(H/'LEFT_COVERAGE_SEARCH.json')]
for path in manifests:used |= set(re.findall(r'NIA_SL_WORD\d+_REAL\d+_F',Path(path).read_text('utf-8-sig')))
sources=[]
for person in [14,15]:
 for word in ['WORD2700','WORD2718','WORD0071','WORD0949']:
  cid=f'NIA_SL_{word}_REAL{person}_F';assert cid not in used;folder=O/'word100_autonomous_recognition_20261026/fresh199/data';meta=folder/'records'/f'{cid}.json';r=read(meta);path=folder/'clips'/cid/'source.mp4';expected=r.get('source_sha256',r.get('source',{}).get('sha256'));assert sha(path)==expected
  sources.append(dict(clip_id=cid,word=word,label=r['canonical_label'],person=person,role='diagnostic'if person==14 else'confirmation_reserved',path=str(path),sha256=sha(path),prior_use='Past TCN fresh evaluation; absent in examined fine-finger candidate manifests; not globally unseen'))
write(U/'FROZEN.json',dict(sources=sources))
write(U/'PLAN.json',dict(at=now,steps=['A Label source RGB shape changes separately from RTMW wrong-finger chains. Both can occur together; unknown stays unknown.','B Train a small shared-finger temporal Conv+GRU quality/change observer using onlyFIT01..09. No word/description/target shape input.','C Choose using diagnostic14 and historic DEV before reserve15 predictions. Compare existing finger decisions and simple constant-stable/constant-good baselines.','D Freeze, reserve15 once, public122normal/error regression and protected hashes. Only passing conservative observation guard may enter web.'],fit='PreviousFIT3,6,9 public sequences, source reviewed anew; never14/15/17/18',labels='quality:good/wrongfinger/unknown; phase:stable/changing/unknown. Nonexclusive heads; no fabricated corrected coordinates. 4longfingers only.',gates=['Cannot create any new confident finger claim or error card; only retain or abstain existing finger detail.','Reduce wrong finger claims in both diagnostic and reserve, with <=5% previous correct decisions lost on each plus historicalDEV.','No new normal correction warnings or loss of previously detected synthetic shape/direction differences in122cases.','No word-specific branching, no change toC/T3F/common100/baseDTW or user data.'],implementer='Codex',verification='Separate artifact recheck by same Codex, no independent KSL expert',web_conditional=True,no_download=True))
pins=read(H/'PROTECTED.json');write(U/'PROTECTED.json',pins);b=U/'before';b.mkdir()
for p in [R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copy2(p,b/p.name)
state=read(O/'STATE.json');state['current_followup']=U.name;state['next_single_action']='공개 원본으로 관절 혼선과 손모양 전환 정답을 분리 보강→작은 공통 관측 판정부 학습→미사용15확인→정상회귀→통과 기능만 연결';state[U.name]=dict(status='SOURCE_LABEL_REVIEW',plan=str(U/'PLAN.json'),at=now);write(O/'STATE.json',state)
doc=(R/'WordsInSilence_PROJECT_CONTEXT.md').read_text('utf-8-sig')
for h in ['## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','## 9. 다음 단일 작업']:doc=doc.replace(h+'\n\n',h+'\n\n### 2026-11-15 손가락 관측 혼선·변화 분리 작업 진행\n\n'+state['next_single_action']+'. `research/front_keyphase_v1/ops/'+U.name+'/PLAN.json`. 학습01..09/진단14/예약15, 이전17/18 재튜닝 금지. C·T3-F·사용자자료 보호.\n\n',1)
(R/'WordsInSilence_PROJECT_CONTEXT.md').write_text(doc,'utf-8');(R/'NEW_CHAT_HANDOFF.md').write_text('# 손가락 관측 혼선·변화 분리 작업 진행\n\n'+state['next_single_action']+'\n\n현재계약 `research/front_keyphase_v1/ops/'+U.name+'/PLAN.json`, 실행상태 CURRENT_EXECUTION.json. 이전결과는 before/에 보존. 웹 G6-finger-20261115 유지. 카메라/사용자자료 미사용.\n','utf-8')
write(U/'CURRENT_EXECUTION.json',dict(status='PREPARING_SOURCE_LABELS',pid=None));print('reserved',len(sources))
