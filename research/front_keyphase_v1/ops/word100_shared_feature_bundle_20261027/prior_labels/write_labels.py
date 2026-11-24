"""Record this reviewer's image-only coarse feature draft; no model inference."""
import datetime, hashlib, json
from pathlib import Path
HERE=Path(__file__).resolve().parent
U=HERE.parent
VIS=U.parent/'word100_source_token_alignment_20261027/visual_inputs'
fields=['R_index_only','L_index_only','R_open_four','L_open_four','R_face_overlap','L_face_overlap','hands_overlap']
# Every vector below was entered after viewing the 16-image sheet. Specific enlarged
# saved frames listed in reviewed_files resolve borderline face/hand overlap.
labels={
 'V01':[[1,None,0,None,0,0,0],[1,None,0,None,0,0,0],[1,None,0,None,0,0,0]],
 'V02':[[1,None,0,None,1,0,0],[1,None,0,None,0,0,0],[1,None,0,None,0,0,0]],
 'V03':[[1,None,0,None,1,0,0],[1,None,0,None,1,0,0],[1,None,0,None,1,0,0]],
 'V04':[[1,0,0,1,0,0,1],[1,0,0,1,0,0,1],[1,0,0,1,0,0,1]],
 'V05':[[1,0,0,1,0,0,0],[1,0,0,1,0,0,1],[1,0,0,1,0,0,1]],
 'V06':[[1,0,0,1,0,0,0],[1,0,0,1,0,0,1],[1,0,0,1,0,0,1]],
 'V07':[[0,0,1,1,0,0,0],[0,0,1,1,0,0,0],[0,0,1,1,0,0,0]],
 'V08':[[0,0,1,1,0,0,0],[0,0,1,1,0,0,0],[0,0,1,1,0,0,0]],
 'V09':[[0,0,1,1,0,0,0],[0,0,1,1,0,0,0],[0,0,1,1,0,0,0]],
 'V10':[[0,None,0,None,1,0,0],[0,None,1,None,0,0,0],[0,None,1,None,0,0,0]],
 'V11':[[0,None,0,None,1,0,0],[0,None,0,None,0,0,0],[0,None,1,None,0,0,0]],
 'V12':[[0,None,1,None,1,0,0],[0,None,0,None,0,0,0],[0,None,1,None,0,0,0]],
}
notes={
 'V01':'오른 검지는 펴져 있고 다른 세 손가락은 접혀 보임. 내려둔 왼손의 세밀한 형태는 중간 굽힘/측면이라 null. 오른손은 이 고정 표본에서 얼굴 옆에 분리되어 있음.',
 'V02':'오른 검지 단독 폄. 5번은 검지 끝이 얼굴 면적에 겹치며 8·11번은 분리. 왼손 형태는 내려둔 측면 관찰의 불확실성을 유지.',
 'V03':'오른 검지 단독 폄. 5·8·11에서 오른손 일부가 얼굴 왼쪽 윤곽 안에 2D 투영됨. 물리 접촉 뜻은 아님. 왼손 형태는 null.',
 'V04':'오른 검지의 전방 단축투영과 접힌 다른 손가락, 왼손의 편 네 손가락이 보임. 양손 투영이 맞닿음. 얼굴과 분리.',
 'V05':'오른 검지 단독 폄과 왼손 넓은 폄. 5번은 두 손 사이 빈 공간, 8·11번은 손 투영 맞닿음.',
 'V06':'오른 검지 단독 폄과 왼손 넓은 폄. 5번은 두 손 분리, 8·11번은 투영 맞닿음.',
 'V07':'두 손의 네 손가락이 넓게 펴진 상태. 선택 표본에서 손 사이 간격이 보임. 얼굴과 분리.',
 'V08':'오른손은 세워 가로로, 왼손은 눕혀 펴짐. 5번 확대 원본에서도 손 윤곽 사이 간격이 보이며 물리 접촉을 추정하지 않음.',
 'V09':'두 손을 펴서 가슴 앞에 위치. 8번 확대 원본에서 작은 투영 간격이 보임. 팔 겹침을 손끼리 겹침으로 세지 않음.',
 'V10':'5번 오른손은 모은/굽힌 형태이며 얼굴을 가림. 8·11번은 네 손가락을 넓게 펴고 얼굴 밖으로 분리. 왼손 형태는 측면의 중간 굽힘으로 null.',
 'V11':'5·8번 오른손은 모은 형태,11번은 자연스러운 약한 굽힘을 가진 넓은 네 손가락 폄. 5번 손 일부만 이마와 겹침. 왼손은 형태 null.',
 'V12':'5번 오른 네 손가락이 넓게 펴지고 손 일부가 이마에 투영됨.8번은 모은 형태,11번은 넓은 폄으로 분리.왼손 형태 null.',
}
rows=[]
for alias,vectors in labels.items():
 for ordinal,vector in zip([5,8,11],vectors):
  rows.append(dict(alias=alias,ordinal=ordinal,**dict(zip(fields,vector)),note=notes[alias]))
seq=[]
for n in range(1,13):
 a=f'V{n:02d}'
 if n<=3:
  seq.append({'alias':a,'R':0,'L':None,'evidence_ordinals':[2,3,4,5,8,11,13,14],
   'note':'보이는 오른손 표본은 검지단독 또는 모인/느슨한 중간형이며 명확한 넓은 네손가락 폄과의 변화는 관찰되지 않음. 왼손은 내려둔 측면 형태의 연속 정보 부족. 표본 사이 변화 부재를 보장하지 않음.'})
 elif n<=6:
  seq.append({'alias':a,'R':0,'L':0,'evidence_ordinals':[3,4,5,6,8,10,11,12,13,14],
   'note':'관찰 가능한 오른손은 검지단독/그 외 접힌 손가락 상태, 왼손은 넓은 폄 유지. 이16표본에서 명확한 넓은 폄↔모임 변화가 관찰되지 않음; 반복횟수나 사이 프레임 변화 판단 아님.'})
 elif n<=9:
  seq.append({'alias':a,'R':0,'L':0,'evidence_ordinals':[2,3,4,5,6,8,10,11,12,13,14],
   'note':'관찰 가능한 두손은 넓게 편 형태를 유지하며 위치가 움직임. 넓은 폄↔모임 변화는 이 표본에서 관찰되지 않음.'})
 else:
  ev={'V10':[5,6,7,8,11],'V11':[5,6,8,9,10,11,13],'V12':[4,5,6,7,8,9,11]}[a]
  seq.append({'alias':a,'R':1,'L':None,'evidence_ordinals':ev,
   'note':'오른손에 명확한 모임/굽힘 표본과 넓은 네손가락 폄 표본이 모두 있고 앞뒤 변화가 보임. 정확한 반복횟수는 판정하지 않음. 내려둔 왼손 형태의 구체적 변화는 부족하여 null.'})
extra=['V01/frame_05_f0055.png','V02/frame_05_f0047.png','V03/frame_08_f0073.png',
       'V04/frame_05_f0058.png','V05/frame_05_f0053.png','V06/frame_08_f0073.png',
       'V08/frame_05_f0055.png','V09/frame_08_f0068.png','V11/frame_11_f0072.png','V12/frame_05_f0059.png',
       'V02/frame_08_f0060.png','V03/frame_11_f0084.png','V11/frame_05_f0056.png']
reviewed=[f'V{i:02d}/detail_contact.png' for i in range(1,13)]+extra
d={'rubric_sha256':hashlib.sha256((U/'RUBRIC.json').read_bytes()).hexdigest(),
 'reviewer':'Codex ksl_source_audit/prior_labels','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'scope':'12 previously extracted public image sheets,16 sampled images per clip; static fixed ordinals5,8,11 only; sequence from sampled images',
 'label_status':'동일 AI 계열의 이미지 관찰 주석 초안; 독립 사람/수어 전문가 정답이 아님',
 'prior_context':'Reviewer previously inspected source dictionaries; this draft is NOT fully source-blinded. Labels here were entered from actual view_image RGB examination, not source text, target labels, joint estimates, or root labels.',
 'source_text_used_to_fill_labels':False,'joint_estimates_used_as_truth':False,'user_material_accessed':False,'training_performed':False,
 'rows':rows,'sequences':seq,'reviewed_files':[{'path':str(VIS/x),'sha256':hashlib.sha256((VIS/x).read_bytes()).hexdigest()}for x in reviewed]}
assert len(rows)==36 and len(seq)==12
assert len({(r['alias'],r['ordinal'])for r in rows})==36
assert all(r[f] in (0,1,None) for r in rows for f in fields)
(HERE/'IMAGE_LABELS.json').write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
print('saved36 rows,12 sequences,25 actual viewed files; null static cells',sum(r[f] is None for r in rows for f in fields))
