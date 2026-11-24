from pathlib import Path
import json,hashlib
V=Path(__file__).resolve().parent/'proposal_adapter'
rows=json.loads((V/'RESERVED_PUBLIC.json').read_text('utf-8'))['rows'];by={r['word_id']:r for r in rows};out=[]
def add(word,axis,es,holds,bg,note):
 r=by[word];out.append(dict(clip_id=r['clip_id'],person=12,word_id=word,role='reserved_fine_evaluation',axis=axis,moving='R',anchor='L',events=[dict(onset=a,offset=b,direction=d) for a,b,d in es],holds=holds,background=bg,source_sha256=r['source_sha256'],sheet=r['sheet'],sheet_sha256=r['sheet_sha256'],notes=note))
add('WORD0003',4,[([51,54],[60,63],0),([60,63],[69,72],2),([72,75],[78,81],0)],[[84,90]],[[0,33],[108,120]],'높낮이 교차의 세 구간. 한 바퀴 횟수 정답이 아님.')
add('WORD0500',0,[([57,60],[63,66],0)],[[66,78]],[[0,33],[90,105]],'얼굴에서 다른 손 쪽으로 하강.')
add('WORD0678',0,[([54,57],[63,66],0)],[[66,78]],[[0,30],[96,108]],'턱 부근에서 가슴 쪽으로 하강 후 유지.')
add('WORD1139',3,[([51,54],[57,60],0)],[[60,72]],[[0,36],[90,105]],'두 손 접근 뒤 유지. 접촉 확정 아님.')
add('WORD1290',4,[([37,39],[41,43],0),([42,44],[47,49],2),([48,50],[51,53],0),([52,54],[57,59],2)],[[60,69]],[],'1프레임 RGB 순서 확인. 작은 왕복의 방향전환4구간, 이후 유지. 관절 예측값은 보지 않고 표시.')
add('WORD2551',0,[],[[63,72]],[[0,27],[90,99]],'손가락 변화/옆이동과 별도로 손목 높이 유지. 이 축에 주요 상하 동작 정답을 만들지 않음.')
assert sum(len(r['events']) for r in out)==10
for r in out:
 for e in r['events']:assert 0<=e['onset'][0]<=e['onset'][1]<=e['offset'][0]<=e['offset'][1]<by[r['word_id']]['original_frames']
(V/'EVALUATION_LABELS.json').write_text(json.dumps(dict(rows=out,review='Same Codex RGB review before adapter model scores; not independent expert annotation'),ensure_ascii=False,indent=2),'utf-8');(V/'REVIEW_INPUTS.json').write_text(json.dumps(dict(rows=rows),ensure_ascii=False,indent=2),'utf-8');print('Six reserved public originals: ten movement legs plus six holds')
