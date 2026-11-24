"""Transcribe direct RGB review before training; no coordinate/model-generated boundaries."""
from pathlib import Path
import json,hashlib
U=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text('utf-8-sig'))
rows={r['clip_id']:r for r in read(U/'REVIEW_INPUTS.json')['rows']};labels=[]
def add(w,p,axis,events,holds,bg,note,side='R'):
 key=f'NIA_SL_WORD{w}_REAL{p:02d}_F';r=rows[key]
 labels.append(dict(clip_id=key,role=r['role'],person=p,axis=axis,moving=side,anchor='L' if side=='R' else 'R',events=[dict(onset=a,offset=b,direction=t) for a,b,t in events],holds=holds,background=bg,notes=note,sheet=r['sheet'],sheet_sha256=r['sheet_sha256'],source_sha256=r['source_sha256']))
for p,down,up,hold,bg in [
 (1,([59,61],[68,70],0),([72,74],[78,80],2),[80,90],[[32,38],[96,102]]),
 (3,([68,70],[78,80],0),([82,84],[88,90],2),[90,100],[[42,46],[108,112]]),
 (7,([50,52],[60,62],0),([64,66],[70,72],2),[72,78],[[24,28],[86,90]]),
 (11,([69,71],[77,79],0),([81,83],[87,89],2),[89,103],[[111,115]])]:
 add('0739',p,0,[down],[],bg,'입 앞에서 가슴 쪽으로 손이 내려오기 시작하고 옆으로 전환하는 범위. 접촉 정답 아님.')
 add('0739',p,1,[up],[hold],bg,'수평에 가깝던 전완이 세워진 뒤 유지. 내려가는 준비/복귀는 주요 상승과 분리.')
for p,a,b,h,bg in [(1,[52,54],[60,62],[64,80],[[32,40],[86,94]]),(3,[55,57],[65,67],[69,79],[[35,43],[87,95]]),(7,[51,53],[61,63],[65,75],[[27,35],[81,87]]),(11,[54,56],[64,66],[68,76],[[40,44],[84,92]])]:
 add('1133',p,2,[(a,b,2)],[h],bg,'오른손이 왼팔 위쪽에서 손목 쪽으로 이동한 뒤 유지. 피부 접촉/정확한 이동비율 정답은 아님.')
for p,es,h,bg in [
 (1,[([53,55],[59,61],0),([62,64],[67,69],2),([70,72],[77,79],0)],[81,95],[[37,45],[105,113]]),
 (3,[([42,44],[48,50],0),([51,53],[58,60],2),([61,63],[68,70],0)],[72,82],[[22,30],[90,96]]),
 (7,[([34,36],[40,44],0),([44,46],[50,54],2),([54,56],[62,66],0)],[66,76],[[18,26],[84,90]]),
 (11,[([52,54],[58,60],0),([60,62],[68,70],2),([70,72],[80,82],0)],[84,96],[[34,40]])]:
 add('1567',p,4,es,[h],bg,'양손의 위아래 관계가 세 차례 바뀜. 맞닿거나 교차하는 순간에도 이동을 삭제하지 않음. 방향전환은 앞 offset과 뒤 onset 범위로 표시.')
for p,a,b,h,bg in [(1,[37,39],[43,45],[47,55],[[15,23],[63,71]]),(3,[67,69],[75,77],[79,83],[[39,47],[93,99]]),(7,[44,46],[50,52],[54,60],[[20,28],[68,74]]),(11,[56,58],[62,64],[66,76],[[38,42],[84,88]])]:
 add('2523',p,3,[(a,b,2)],[h],bg,'얼굴 앞에서 양손이 벌어져 어깨 근처에 머무는 이동. 이 주석은 손끝 오므림/접촉 정답을 포함하지 않음.')
add('0003',1,4,[([35,37],[38,40],0),([40,42],[44,46],2),([46,48],[50,52],0)],[[53,59]],[[21,27],[67,71]],'양손을 앞뒤로 돌리면서 위아래 순서가 바뀌는 짧은 세 이동. 경계가 가까워 범위 겹침을 허용.')
add('0500',1,0,[([59,61],[65,67],0)],[[69,81]],[[41,43],[87,91]],'턱 부근 오른손이 반대손 쪽으로 내려와 유지.')
add('0678',1,0,[([60,62],[68,70],0)],[[72,86]],[[38,42],[90,94]],'턱 부근 손이 가슴 쪽으로 내려온 뒤 유지.')
add('1139',1,3,[],[[55,79]],[[85,91]],'검토 화면 시작 때 이미 접근 중이므로 시작 경계는 새로 확정하지 않음. 양손 근접 유지 구간만 채택.')
add('1290',1,4,[([46,48],[50,52],0),([52,54],[56,58],2),([58,60],[62,64],0)],[],[[37,41],[83,89]],'아래쪽 왼손을 기준으로 오른손이 내려갔다 올라갔다 다시 내려감. 65~79에도 작은 움직임이 있어 유지 정답으로 강요하지 않음.')
add('2551',1,0,[],[[55,65]],[[75,81]],'화면 시작 때 이미 올리는 중이며 손가락 펴는 변화와 옆 이동이 섞임. 수직 이동 시작/끝은 확정하지 않고 높이 유지에만 감독.')
for r in labels:
 n=rows[r['clip_id']]['original_frames']
 for e in r['events']:assert 0<=e['onset'][0]<=e['onset'][1]<=e['offset'][0]<=e['offset'][1]<n
 for a,b in r['holds']+r['background']:assert 0<=a<=b<n
 assert hashlib.sha256(Path(r['sheet']).read_bytes()).hexdigest()==r['sheet_sha256']
out=dict(rows=labels,reviewer='Codex direct RGB contact-sheet observation; not independent human/expert',purpose='Observable motion boundary ranges. Frame spacing2, intervals preserve temporal uncertainty. No phase labels from fixed percentage or model outputs.',limits=['Only10expressions18FIT4historicalDEV inspected;100wordcommon training is not100worddensegold.','No source person identity inferred from appearance; recorded IDs only.','Direction labels describe visible2Dbody-relative change, not educational correctness or hidden3Dcontact.','WORD1139/WORD2551 onset not visible in selected window, left unlabelled instead of invented.'],frozen_before_training=True)
(U/'OBSERVED_BOUNDARIES.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n','utf-8')
print(json.dumps({role:dict(videos=len({r['clip_id'] for r in labels if r['role']==role}),events=sum(len(r['events']) for r in labels if r['role']==role),holds=sum(len(r['holds']) for r in labels if r['role']==role)) for role in ['fit','development']},ensure_ascii=False))
