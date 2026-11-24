"""Transcribe root Codex RGB review. No model/geometry-derived target labels."""
from pathlib import Path
import json, hashlib, collections
U=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
# Each array corresponds to the three actually displayed source frames.
# U = wrist clearly above elbow; A = approximately across; D = below.
# '?' is intentionally unlabelled, NOT a negative/transition truth.
forearm={
 2:{2:['U','?','A'],3:['A','?','U']},
 3:{2:['U','?','A'],3:['A','?','U']},
 4:{2:['U','?','A'],3:['A','?','U']},
 5:{2:['U','A','A'],3:['A','?','U']},
 6:{2:['U','A','A'],3:['A','?','U']},
 7:{2:['U','?','A'],3:['A','?','U']},
 8:{2:['U','A','A'],3:['A','U','U']},
 9:{2:['U','A','A'],3:['A','?','U']}}
height={2:['R','?','L'],3:['R','R','L'],4:['R','R','L'],5:['L','?','L'],
        6:['R','?','L'],7:['R','R','L'],8:['L','R','L'],9:['R','L','L']}
labels={'forearm':{'U':'wrist_above_elbow','A':'forearm_across','D':'wrist_below_elbow'},
        'arm_region':{'P':'upper_arm_side','W':'wrist_side'},
        'separation':{'N':'hands_near','S':'hands_apart'},
        'height':{'R':'active_hand_higher','L':'opposite_hand_higher'}}
rows=[];keys={}
def add(r,domain,hand,code,note):
    key=(r['clip_id'],r['frame'],domain,hand)
    label=labels[domain].get(code)
    if key in keys:
        assert keys[key]['label']==label
        keys[key]['source_groups'].append(r['group_id']);return
    out={k:r[k]for k in ['clip_id','person','split','word_id','frame','source','source_sha256']}
    out.update(domain=domain,hand=hand,label=label,source_groups=[r['group_id']],note_ko=note,
        review_sheet=str(U/f"review_record{r['person']:02}.jpg"),
        reviewer='root Codex direct RGB review; not KSL expert or independent human',
        linguistic_correctness=None,semantic_completion=None)
    keys[key]=out;rows.append(out)
for r in read(U/'REVIEW_INPUTS.json')['rows']:
    p,f,s=r['person'],r['family'],r['review_slot']
    if f in [2,3]:
        add(r,'forearm','R',forearm[p][f][s],'팔꿈치에 대한 손목의 화면상 방향. 대각선 중간 자세는 보류. 손가락/3D/수어정오 라벨 아님.')
        add(r,'forearm','L','D','공개 원본에서 반대팔과 손목이 몸 아래에 보임. 화면 밖이면 관측 부족으로 처리해야 함.')
    if f==4:
        add(r,'forearm','R','A','오른팔 전완이 몸 앞에서 대체로 가로 방향으로 보임.')
        add(r,'arm_region','R',['P','?','W'][s],'반대팔에 대한 손바닥의 근위/손목 쪽 관계. 중간은 보류. 정확한 접촉이나 최소 쓸기 길이의 정답 아님.')
    if f==5:
        add(r,'height','R',height[p][s],'양손 중 어느 손이 높게 보이는가. 겹쳐 불명확한 순간은 보류. 반복 횟수/순환 전체완료 정답 아님.')
    if f==7:
        add(r,'separation','R',['N','S','S'][s],'처음 양손이 중앙에 가까이 있고 중간과 끝에서 이미 떨어져 보임. 손끝 접촉/오므림 정답 아님.')
out=dict(rows=rows,source_videos=32,fit_record_ids=list(range(2,8)),exposed_probe_record_ids=[8,9],
    classes=labels,review_inputs_sha256=sha(U/'REVIEW_INPUTS.json'),
    scope='Coarse visible body-relative component states; not required user duration or whole-sign correctness.',
    method='Visual review before training; model outputs/coordinates were not used to assign labels.',
    unknown_policy='No forced middle state, no negative target for ambiguity.',
    counts=dict(collections.Counter((r['split']+':'+r['domain']+':'+str(r['label']))for r in rows)))
with (U/'STATE_LABELS.json').open('x',encoding='utf-8')as f:json.dump(out,f,ensure_ascii=False,indent=2)
print('Review annotations:',len(rows),'known',sum(r['label']is not None for r in rows))
