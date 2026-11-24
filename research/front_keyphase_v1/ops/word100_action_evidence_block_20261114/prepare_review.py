from pathlib import Path
import json,hashlib,shutil,collections
from datetime import datetime
from zoneinfo import ZoneInfo
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2]
P=O/'word100_transition_observation_audit_20261114';T=O/'word100_terminal_relation_targets_20261114';G=O/'word100_web_completion_graph_20261111'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n','utf-8')
assert not(U/'REVIEW_INPUTS.json').exists()
now=datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
plan=dict(goal='Finish the current G2 action/description evidence block, not stop after each component comparison',
 user_authorization='Continue reinforcement and verification until the graph box is complete; report serious obstacles or decisions only',
 steps=['FIT same-shape/different-view vs actual finger-change source review',
 'Freeze a within-interval shape/orientation candidate after the observations support it',
 'Compare normal/speed/crossing/short/partial/missing and preserve old endpoint evidence',
 'Integrate qualified evidence into description action ownership and apply original G2 gates'],
 original_gates=read(G/'GATES.json')['stage2'],
 local_gate='No wrong maintained-shape change; retain all previously correct closing controls and original endpoints. No all-unknown success.',
 no_web_promotion_before_G3_G4=True,final8_unopened=True,private_data_used=False,
 old_runs_not_repeated=True,fit_records=list(range(2,8)),exposed_probe_records=[8,9],
 caveat='Same Codex source review is not independent human educational correctness. No guaranteed success or lowered gate.')
save(U/'PLAN.json',plan)
save(U/'CURRENT_EXECUTION.json',dict(status='SOURCE_REVIEW',at=now,training_running=False,block_complete=False))
backup=U/'before_start';backup.mkdir(exist_ok=True)
for p in [O/'STATE.json',R/'NEW_CHAT_HANDOFF.md',R/'WordsInSilence_PROJECT_CONTEXT.md',G/'TASK_GRAPH.json']:
 shutil.copy2(p,backup/p.name)
s=read(O/'STATE.json');s['current_followup']=U.name
s['next_single_action']='현재 G2 행동 근거 칸의 공개 FIT 원본 검토→변화/유지 비교→설명 구간 통합 검증을 연속 진행한다. 세부 실험 종료마다 전체 작업을 종료하지 않는다.'
s[U.name]=dict(status='IN_PROGRESS',unit=str(U),started=now,graph_box='G2',plan=str(U/'PLAN.json'),previous_unit=str(P),training_running=False,web_changed=False)
save(O/'STATE.json',s)
title='### 2026-11-14 사용자 요청: 세부 실험별 종료 대신 현재 G2 칸의 보완·검증을 연속 진행'
doc=(R/'WordsInSilence_PROJECT_CONTEXT.md').read_text('utf-8-sig')
for h in ['## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인','## 9. 다음 단일 작업']:
 marker=h+'\n\n';assert marker in doc
 doc=doc.replace(marker,marker+title+'\n\n'+s['next_single_action']+' 기존 통과 기준·개인자료 금지·C/T3F/공개100/웹 보존. 완료 조건을 낮추지 않는다.\n근거: `research/front_keyphase_v1/ops/'+U.name+'/PLAN.json`.\n\n',1)
(R/'WordsInSilence_PROJECT_CONTEXT.md').write_text(doc,'utf-8')
(R/'NEW_CHAT_HANDOFF.md').write_text(f'''# 현재 G2 칸 보완 진행 중 — {now}
사용자가 세부 실험 하나마다 멈추지 말고 그래프 칸 완료까지 이어가라고 요청했다.
현재단위 {U}; PLAN/CURRENT_EXECUTION 확인. 이전단위 {P}의 완료3학습/직전9학습 재실행금지.
공개 FIT의 자세 유지/방향 변화와 실제 손모양 변화 대조를 검토한 뒤 한 후보를 동결 비교하고 G2 원래 통과 기준까지 확인한다.
기존 C·T3-F/공개100/끝 모양 모델/웹 보존, 사용자·멘토 자료 접근/학습/튜닝금지, Final8 미개봉.
기존 노출probe08/09는 새로운 독립평가가 아니다. G2 실패가 남으면 원인단계로 돌아가며 통과를 꾸며내지 않는다.
이전 결과 상세는 이전단위RESULT_KO/RESULTS_REVIEW, 이번 단위before_start에 시작상태 백업.
현재 학습 실행 없음. 카메라·다운로드·Git 없음.
''','utf-8')
labels=read(T/'LABELS.json')['rows'];sources={r['clip_id']:r for r in labels}
pairs=read(P/'PAIR_LABELS.json')['rows'];font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',15);out=[]
for rec in range(2,8):
 selected=[r for r in pairs if r['record_id']==rec and not r['reverse']and not r['freeze']and r['origin']in ['RGB_open_to_closed','RGB_closed_shape_during_forearm_raise','RGB_maintained_shape_during_arm_motion']]
 sheet=Image.new('RGB',(1320,230*len(selected)),'white');d=ImageDraw.Draw(sheet)
 for row,r in enumerate(selected):
  cid=r['clip_id'];source=sources[cid];record=read(O/'word100_rtmw_joint_learning_20261108/records'/f'{cid}.json')
  raw=np.load(record['raw_path']);assert sha(Path(record['raw_path']))==record['raw_sha256'];assert sha(Path(source['source']))==source['source_sha256']
  cap=cv2.VideoCapture(source['source']);ids=np.rint(np.linspace(*r['source_frames'],6)).astype(int)
  for col,fid in enumerate(ids):
   cap.set(cv2.CAP_PROP_POS_FRAMES,int(fid));ok,im=cap.read();assert ok
   idx=int(np.flatnonzero(raw['frame_id']==fid)[0]);base=112 if r['hand']=='R'else 91
   pts=raw['xy_px'][idx,base:base+21];center=pts.mean(0);rad=max(75,float(np.ptp(pts,axis=0).max())*.8)
   h,w=im.shape[:2];a=np.maximum(0,(center-rad).astype(int));b=np.minimum([w,h],(center+rad).astype(int))
   crop=im[a[1]:b[1],a[0]:b[0]];rgb=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB));rgb.thumbnail((212,192));sheet.paste(rgb,(col*220+4,row*230+32))
   d.text((col*220+4,row*230+4),f'{cid[7:15]} {r["hand"]} f{fid}',fill='#17283f',font=font)
  cap.release()
  out.append(dict(pair_index=r['index'],clip_id=cid,record_id=rec,hand=r['hand'],frames=ids.tolist(),source_frames=r['source_frames'],
     expected_change=r['target'],origin=r['origin'],source=source['source'],source_sha256=source['source_sha256'],
     raw=record['raw_path'],raw_sha256=record['raw_sha256'],review_sheet=str(U/f'fit_review{rec:02}.jpg'),review_completed=False))
 sheet.save(U/f'fit_review{rec:02}.jpg',quality=94)
save(U/'REVIEW_INPUTS.json',dict(rows=out,scope='FIT only, all original change/maintenance intervals, six uniform native frames per interval; no predicted-error selection',
 previous_probe_failures_known=True,whole_education_truth=False))
print(json.dumps(dict(review_intervals=len(out),sheets=6,work_unit=str(U)),ensure_ascii=False))
