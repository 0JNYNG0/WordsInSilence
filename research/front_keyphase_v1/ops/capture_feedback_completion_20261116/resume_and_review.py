"""Resume approved graph and prepare source-only RGB review panels."""
from pathlib import Path
import json,sys
from datetime import datetime,timezone,timedelta
import cv2,numpy as np
U=Path(__file__).resolve().parent;O=U.parent;B=O/'basic_demo_feedback_20261115';ROOT=U.parents[3]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
now=datetime.now(timezone(timedelta(hours=9))).isoformat()
v=U/'completion_v1';v.mkdir(exist_ok=True);(v/'review').mkdir(exist_ok=True)
p=read(U/'plan.json');p['status']='IN_PROGRESS';p['resumed_at']=now;p['next']='B/C: source-reviewed shared handshape continuation and complete visible lesson evidence'
for n in p['nodes']:
 if n['id']in['B','C']:n['status']='IN_PROGRESS'
write(U/'plan.json',p)
write(v/'RESUME.json',dict(at=now,requested_by_user=True,training_private=False,scope='Capture-after-assessment; live coaching expansion remains deferred'))
state=read(O/'STATE.json');state['current_stage']='capture_feedback_completion_20261116';state['next_single_action']=p['next'];state['capture_feedback_completion_20261116']['status']='IN_PROGRESS';state['capture_feedback_completion_20261116']['resumed_at']=now;write(O/'STATE.json',state)
h=ROOT/'NEW_CHAT_HANDOFF.md';s=h.read_text(encoding='utf-8');s=s.replace('## 현재: 사용자 요청으로 일시 중단','## 직전 중단 기록: 사용자 요청으로 일시 중단',1)
s=s.replace('# 최신 인수인계 — 한국수어 피드백 게임','# 최신 인수인계 — 한국수어 피드백 게임\n\n## 현재: '+now+' 사용자 재개 승인\n중단 지시 해제. capture_feedback_completion_20261116 그래프 B/C부터 실행. 현재 completion_v1에서 공개 원본 손모양·손면 검수와 기존 모델 보강을 준비한다. 과거 중단 항목보다 이 재개 지시가 우선한다.\n',1);h.write_text(s,encoding='utf-8')
ctx=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';s=ctx.read_text(encoding='utf-8-sig');s=s.replace('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n','## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n### 2026-11-16 촬영후 평가 그래프 재개\n사용자 재개 승인으로 이전 중단 해제. 공개 원본 손모양·손면·도달 근거부터 B/C 보완, D/E/F 순서로 계속 진행. completion_v1/RESUME.json. 기존 C·T3-F·사용자자료 학습금지 유지.\n',1);s=s.replace('## 9. 다음 단일 작업\n','## 9. 다음 단일 작업\n\n### 2026-11-16 그래프 B/C 실행\n감사 공개 FIT 원본의 손모양·손면을 검수하고 공통 모양 모델을 보강·동일 기존 DEV 비교한다. 필요한 필수행동 근거와 과제완료를 연결한다. 이후 D 공개 정상/오류 검증, E 웹, F 실제 촬영 순서.\n',1);ctx.write_text(s,encoding='utf-8')
rows=read(B/'INPUTS.json')['rows'];gt=read(B/'source_corrected/SOURCE_TRUTH.json')['rows'];review=[]
for r in rows:
 if r['role']!='fit':continue
 anchors=[x['frame'] for x in gt if x['person']==r['person'] and x['state'] in[1,2]]
 ids=sorted(set([anchors[0],anchors[len(anchors)//2],anchors[-1]]))
 raw=dict(np.load(r['raw']));cap=cv2.VideoCapture(r['video']);tiles=[]
 for frame in ids:
  cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,img=cap.read();assert ok
  # One joint hand/forearm crop, no inferred labels or model predictions on RGB.
  pts=raw['xy_px'][frame,[5,6,7,8,9,10]+list(range(91,133))];pts=pts[np.isfinite(pts).all(1)]
  lo=np.maximum(np.floor(pts.min(0)-35),0).astype(int);hi=np.minimum(np.ceil(pts.max(0)+35),[img.shape[1],img.shape[0]]).astype(int)
  crop=img[lo[1]:hi[1],lo[0]:hi[0]];scale=min(500/crop.shape[1],330/crop.shape[0]);crop=cv2.resize(crop,None,fx=scale,fy=scale)
  tile=np.full((375,520,3),255,np.uint8);tile[38:38+crop.shape[0],10:10+crop.shape[1]]=crop
  cv2.putText(tile,f'P{r["person"]:02} F{frame}',(10,26),cv2.FONT_HERSHEY_SIMPLEX,.7,(20,30,45),2);tiles.append(tile)
  review.append(dict(clip_id=r['clip_id'],person=r['person'],frame=frame,role='fit',video=r['video'],channels=r['channels'],source_sha256=r['source_sha256']))
 cap.release();cv2.imwrite(str(v/'review'/f'p{r["person"]:02}.jpg'),np.hstack(tiles))
write(v/'REVIEW_INPUTS.json',dict(rows=review,labels_not_yet_written=True,review='RGB only; no model output used as target'))
print('PREPARED',len(review),'source anchors')
