"""Center-only visual draft by a second Codex agent, not human ground truth."""
from pathlib import Path
import json,hashlib
from datetime import datetime,timezone

OUT=Path(__file__).resolve().parent
U=OUT.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

# Assigned only after directly viewing these 18 center_only.png files.
# No neighboring/ordered frame, new INDEX, or other review labels were read.
OBS=[
('N01',1,0,'오른팔에서 이어지는 화면 왼쪽 손에서 검지가 수평으로 펴져 있고 나머지 긴 손가락은 접힌 윤곽이다. 약한 흐림은 있으나 네 손가락 펼침은 아니다.'),
('N02',1,0,'화면 왼쪽 목표 오른손에서 검지만 길게 펴져 있고 다른 긴 손가락들은 손바닥 쪽으로 굽어 있다.'),
('N03',1,0,'목표 오른손의 수평 검지와 접힌 나머지 손가락 윤곽이 구분된다. 반대 손은 별도 대상으로 보았다.'),
('N04',1,0,'목표 오른손에서 검지가 위로 펴져 있고 나머지 긴 손가락들은 접혀 있다. 엄지는 이 두 특징의 판정 대상에서 제외한다.'),
('N05',1,0,'화면 왼쪽의 목표 오른손에서 위로 펴진 검지와 접힌 나머지 손가락을 확인할 수 있다.'),
('N06',0,1,'목표 오른손은 화면 아래 왼쪽 팔에서 이어지는 손이다. 그 손의 네 긴 손가락이 펴져 벌어진 윤곽이 보인다. 얼굴 앞의 반대 손과 혼동하지 않았다.'),
('N07',1,0,'목표 오른손의 검지가 반대 손 쪽으로 수평으로 뻗고 나머지 긴 손가락이 접힌 형태가 보인다. 검지 끝 일부가 가까운 반대 손과 겹치지만 전체 배열은 식별된다.'),
('N08',1,0,'화면 왼쪽 목표 오른손의 검지가 비스듬히 펴지고 다른 긴 손가락이 접혀 있다.'),
('N09',1,0,'조금 흐리지만 목표 오른손에서 수평 검지와 접힌 다른 긴 손가락의 배열이 구별된다.'),
('N10',0,1,'반대 팔 위에 놓인 목표 오른손에서 여러 손가락의 등과 펴진 끝부분이 보인다. 네 긴 손가락을 펼친 배열로 판단한다.'),
('N11',0,None,'목표 오른손이 반대 팔 위에 납작하게 놓여 있고 검지만 단독으로 편 형태는 아니다. 다만 옆면·흐림 때문에 네 긴 손가락 각각의 폄은 충분히 확인되지 않는다.'),
('N12',0,None,'목표 오른손은 반대 팔 위에 납작하게 놓인 여러 손가락 윤곽으로 보이며 검지만 단독으로 편 형태는 아니다. 네 손가락은 옆면에서 서로 겹쳐 전부 펴졌는지는 보류한다.'),
('N13',None,None,'목표 오른손은 화면 왼쪽 팔에서 이어져 중앙의 반대 손 뒤로 겹친다. 앞쪽 손의 손가락을 목표 손 근거로 쓰면 안 되며, 가려진 목표 손의 두 특징은 모두 확정하기 어렵다.'),
('N14',0,0,'목표 오른손은 위쪽에 있는 손이며 엄지가 올라가고 긴 손가락은 접혀 있다. 검지만 펴거나 네 손가락을 편 상태가 아니다.'),
('N15',0,0,'목표 오른손의 엄지가 위로 펴져 있고 네 긴 손가락은 접힌 형태다. 두 대상 특징은 모두 없다.'),
('N16',0,None,'목표 오른손의 여러 손가락이 함께 수평으로 향해 검지만 단독으로 편 상태는 아니다. 옆면 겹침으로 네 손가락 각각이 얼마나 펴졌는지는 확인하기 어렵다.'),
('N17',0,1,'목표 오른손에서 네 긴 손가락의 분리된 끝부분과 펴진 배열이 보인다. 검지만 단독으로 펴진 모양은 아니다.'),
('N18',0,None,'목표 오른손은 화면 왼쪽 팔에서 얼굴 쪽으로 이어지는 손이다. 여러 손가락이 함께 뻗은 윤곽이므로 검지만 단독으로 편 형태는 아니다. 손이 옆면을 보여 네 손가락 각각의 폄은 보류한다.'),
]

OUT.mkdir(parents=True,exist_ok=True)
if (OUT/'CENTER_LABELS.json').exists():raise RuntimeError('Do not overwrite locked labels')
rows=[];pins={}
for cid,index_only,open_four,notes in OBS:
    p=U/'visual_inputs'/cid/'center_only.png';pins[str(p)]=sha(p)
    rows.append({'case_id':cid,'side':'R','index_only':index_only,'open_four':open_four,'features':{'index_only':index_only,'open_four':open_four},'viewed_path':str(p),'image_sha256':pins[str(p)],'notes_ko':notes,'basis':'single current RGB image only','temporal_inference_used':False})
assert len(rows)==18 and len({r['case_id'] for r in rows})==18
result={'created_utc':datetime.now(timezone.utc).isoformat(),'reviewer':'Codex /root/local_feature_inventory','status':'LOCKED_CURRENT_IMAGE_DRAFT','not_independent_human_ground_truth':True,'blinding':{'new_other_reviewer_labels_read':False,'new_ordered_shuffled_sheets_read':False,'new_INDEX_read':False,'prior_unit_frame_audit_performed':True,'new_center_only_images_viewed':18},'rubric':{'index_only':'1: index visibly extended while other three long fingers are folded; 0: visible contradictory configuration; null: insufficient visibility. Thumb is not used to force the label.','open_four':'1: four long fingers visibly extended; 0: visible contradictory configuration; null: fingers/angle/blur do not support deciding. Does not imply sign correctness.'},'rows':rows,'summary':{'feature_slots':36,'known':sum(v is not None for _,a,b,_ in OBS for v in (a,b)),'unknown':sum(v is None for _,a,b,_ in OBS for v in (a,b))},'no_training':True,'no_user_mentor_access':True,'no_source_edits':True}
(OUT/'CENTER_LABELS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
after={p:sha(p) for p in pins};assert pins==after
verify={'status':'PASS','center_images':18,'image_pins_unchanged':True,'pins':pins,'labels_sha256':sha(OUT/'CENTER_LABELS.json'),'labels_known':result['summary']['known'],'labels_unknown':result['summary']['unknown']}
(OUT/'CENTER_LABELS_VERIFY.json').write_text(json.dumps(verify,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':'LOCKED','summary':result['summary'],'labels_sha256':verify['labels_sha256']}))
