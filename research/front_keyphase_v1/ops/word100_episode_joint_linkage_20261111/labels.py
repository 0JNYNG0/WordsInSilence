from pathlib import Path
import json,hashlib
U=Path(__file__).resolve().parent;O=U.parent
rd=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
rows={r['clip_id']:r for r in rd(U/'REVIEW_INPUTS.json')['rows']};out=[]
def add(w,p,axis,events,holds,bg,note,dense=False):
 r=rows[f'NIA_SL_WORD{w}_REAL{p:02d}_F'];sheet=U/f'review/dense1290_{p}.jpg' if dense else Path(r['sheet'])
 out.append(dict(clip_id=r['clip_id'],role='fit' if p==3 else 'evaluation',person=p,word_id=r['word_id'],axis=axis,moving='R',anchor='L',events=[dict(onset=a,offset=b,direction=d) for a,b,d in events],holds=holds,background=bg,notes=note,source_sha256=r['source_sha256'],sheet=str(sheet),sheet_sha256=hashlib.sha256(sheet.read_bytes()).hexdigest()))
add('0003',3,4,[([57,60],[60,63],0),([63,66],[66,69],2),([69,72],[75,78],0)],[[84,90]],[[0,42],[105,120]],'세 번의 분명한 높낮이 전환. 78~84의 중간 전환은 경계 불확실로 비감독.')
add('0003',11,4,[([60,63],[66,69],0),([72,75],[78,81],2),([81,84],[87,90],0)],[[93,99]],[[0,42],[117,126]],'반복의 각 방향전환을 별도 사건으로 보존.')
add('0500',3,0,[([39,42],[45,48],0)],[[48,60]],[[0,9],[78,90]],'얼굴 앞에서 다른 손 근처까지 하강.')
add('0500',11,0,[([63,66],[69,72],0)],[[72,81]],[[0,33],[99,114]],'얼굴 앞에서 다른 손 근처까지 하강.')
add('0678',3,0,[([60,63],[72,75],0)],[[75,93]],[[0,36],[111,141]],'턱 부근에서 가슴 쪽으로 하강 후 유지.')
add('0678',11,0,[([60,63],[69,72],0)],[[75,93]],[[0,36],[111,129]],'턱 부근에서 가슴 쪽으로 하강 후 유지.')
add('1139',3,3,[([60,63],[66,69],0)],[[69,81]],[[0,45],[102,129]],'양손 접근 후 유지. 접촉 정답은 아님.')
add('1139',11,3,[([42,45],[48,51],0)],[[51,72]],[[0,27],[96,111]],'양손 접근 후 유지. 접촉 정답은 아님.')
add('1290',3,4,[([57,58],[60,61],0),([62,63],[65,66],2),([66,67],[69,70],0),([71,72],[74,75],2)],[],[[0,42],[102,117]],'1프레임 확대 검토. 작은 상하 왕복4개를 흔들림으로 제거하지 않음.',True)
add('1290',11,4,[([49,50],[52,53],0),([55,56],[58,59],2),([60,61],[63,64],0)],[],[[0,33],[105,117]],'1프레임 확대 검토. 확인한 왕복3개만 감독, 나머지 미확정.',True)
add('2551',3,0,[],[[72,81]],[[0,39],[102,117]],'손가락 변화와 옆 이동이 섞임. 높이 유지 구간만 감독.')
add('2551',11,0,[],[[54,69]],[[0,30],[90,99]],'손가락 펴기/접기와 손목 높이 유지는 구분. 세부 손가락 정답은 추가하지 않음.')
add('0739',15,0,[([51,54],[60,63],0)],[],[[0,24],[102,111]],'입 앞에서 가슴 쪽으로 하강.')
add('0739',15,1,[([63,66],[72,75],2)],[[75,84]],[[0,24],[102,111]],'전완 세움 뒤 유지.')
add('1133',15,2,[([51,54],[69,72],2)],[[72,81]],[[0,33],[108,117]],'팔을 따라 손목 근처로 이동.')
add('1567',15,4,[([48,51],[57,60],0),([60,63],[66,69],2),([69,72],[78,81],0)],[[84,93]],[[0,33],[111,120]],'양손 교차와 방향전환3개.')
add('2523',15,3,[([54,57],[60,63],2)],[[63,75]],[[0,33],[90,102]],'모인 손이 벌어진 후 유지. 손끝 변화 정답은 아님.')
for r in out:
 for e in r['events']:assert 0<=e['onset'][0]<=e['onset'][1]<=e['offset'][0]<=e['offset'][1]<rows[r['clip_id']]['original_frames']
def wr(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2),'utf-8')
wr(U/'NEW_OBSERVED_LABELS.json',dict(rows=out,review='Same Codex direct RGB, not independent human;3frame ranges except2dense sheets at1frame. No model scores used.',unobserved='Unreviewed movement omitted from supervised loss, not labelled normal/absent.'))
base=rd(O/'word100_observed_event_boundaries_20261110/OBSERVED_BOUNDARIES.json')['rows']
wr(U/'ALL_FIT_BOUNDARIES.json',dict(rows=[r for r in base if r['role']=='fit']+[r for r in out if r['role']=='fit']))
wr(U/'EVALUATION_LABELS.json',dict(rows=[r for r in out if r['role']=='evaluation'],no_tuning=True,scope='10public originals,6new-expression motion checks atID11 plus4focusID15. Historical TCN/development exposure retained; no globally unseen claim.'))
print('Direct RGB labels frozen:',len(out),'rows')
