"""Serialize this reviewer's RGB observations; no predictions, landmarks or search."""
from pathlib import Path
import json, hashlib

UNIT=Path(__file__).resolve().parents[1]
manifest=json.loads((UNIT/'evidence'/'DEV_CONTACT_MANIFEST.json').read_text(encoding='utf-8'))
details=json.loads((UNIT/'evidence'/'DEV_DETAIL_MANIFEST.json').read_text(encoding='utf-8'))
viewed=json.loads((UNIT/'review'/'DEV_IMAGES_VIEWED.json').read_text(encoding='utf-8'))
meta={c['alias']:c for c in manifest['clips']}

def hand(shape,direction,body,visibility='visible silhouette; fine joint angles unknown'):
    return dict(shape=shape,direction=direction,body_relation=body,visibility=visibility,palm_normal=None,exact_contact=None)

resthand=lambda side:hand('relaxed outline; exact finger configuration not adjudicated','approximately stationary',f'{side} hand beside lower torso')
leftdown=hand('relaxed/downward outline; detailed finger shape unknown','approximately stationary','left hand beside left lower torso')
def segment(alias,num,role,kind,interior,start,end,desc,r,l,subphase='movement',cycle=None,evidence=None,notes=None):
    pts=meta[alias]['source_pts_seconds']
    # Exclude certain-interior endpoints from uncertain boundary frame sets.
    # Clip first/last frames have a fixed recording edge instead of uncertainty.
    start=[start[0],min(start[1],interior[0]-1)] if interior[0]>0 else [0,0]
    end=[max(end[0],interior[1]+1),end[1]] if interior[1]<len(pts)-1 else [len(pts)-1,len(pts)-1]
    return dict(stage_id=f'{alias}_d{num:02d}',role=role,proposed_kind_key=kind,certain_interior=interior,
                start_bracket=start,end_bracket=end,interior_source_pts_seconds=[pts[i] for i in interior],
                fixed_recording_edges={'start':interior[0]==0,'end':interior[1]==len(pts)-1},
                boundary_source_pts_seconds={'start':[pts[i] for i in start],'end':[pts[i] for i in end]},
                description_ko=desc,hands={'R':r,'L':l},motion_subphase=subphase,
                cycle_or_unknown=cycle,correspondence_eligible=role=='core_candidate',
                evidence_frames=evidence or sorted(set([*interior,(interior[0]+interior[1])//2])),
                notes=notes or [],linguistic_core=None,sign_correctness=None)

clips=[]
def add(alias,segments,notes):
    c=meta[alias]; seen_frames=[]
    for s in c['sheets']:
        assert str(Path(s['path'])).replace('\\','/') in [str(Path(p)).replace('\\','/') for p in viewed['files']]
        seen_frames+=s['frame_ids']
    assert seen_frames==list(range(c['native_frame_count']))
    for s in segments:
        a,b=s['certain_interior']; sl,sh=s['start_bracket']; el,eh=s['end_bracket']
        assert 0<=sl<=sh<=a<=b<=el<=eh<c['native_frame_count']
        assert all(i in seen_frames for i in s['evidence_frames'])
    # Interiors are certain physical stage observations. All other frames stay
    # unadjudicated/boundary, rather than silently inheriting a nearby label.
    covered=set(i for s in segments for i in range(s['certain_interior'][0],s['certain_interior'][1]+1))
    unassigned=[i for i in seen_frames if i not in covered]
    ranges=[]
    for i in unassigned:
        if ranges and i==ranges[-1][1]+1:ranges[-1][1]=i
        else:ranges.append([i,i])
    clips.append(dict(alias=alias,native_frame_count=c['native_frame_count'],source_webm=c['source_webm'],
                      source_webm_sha256=c['source_webm_sha256'],timing_metadata=c['timing_metadata'],
                      reviewed_native_frame_ids=seen_frames,segments=segments,
                      non_interior_intervals=[{'frames':r,'status':'boundary_or_transition_unadjudicated','negative_label':False} for r in ranges],
                      reviewer_notes=notes))

a='P036'
add(a,[
segment(a,0,'rest','hands_down_rest',[0,53],[0,0],[54,56],'양손을 몸 옆으로 내린 채 대기한다.',resthand('right'),resthand('left'),'hold'),
segment(a,1,'preparation','right_open_hand_lifts_toward_face',[58,65],[54,58],[65,68],'오른손을 펴며 아래 몸 옆에서 입·코 앞까지 올린다. 왼손은 내려 있다.',hand('four fingers extended together; thumb separated','up','lower torso toward lower face'),leftdown),
segment(a,2,'core_candidate','right_open_hand_near_lower_face',[68,85],[66,68],[85,89],'오른손의 편 손가락 묶음이 입·코 앞에 놓인다. 작은 높이 변화 후 머문다.',hand('four fingers extended and close together; thumb separated','small up/down adjustment then near hold','in front of mouth/nose; exact face contact unknown'),leftdown,'arrival_and_hold',1,[68,70,75,80,85]),
segment(a,3,'core_candidate','right_closing_hand_descends_to_lateral_chest',[90,100],[86,90],[100,103],'오른손을 접으며 얼굴 앞에서 아래·자기 오른쪽 가슴 높이로 옮긴다. 전완이 옆으로 놓인다.',hand('closing transition into closed fist; early exact closure boundary unknown','down and toward own right; ends near horizontal forearm','from lower face to lateral upper chest'),leftdown,'closing_descent',1,[90,93,96,99,100],['88프레임은 아직 편 손가락이 보이므로 닫힌손 내부 구간에 넣지 않았다.']),
segment(a,4,'core_candidate','right_closed_hand_reorients_toward_shoulder',[104,113],[101,104],[113,116],'닫힌 오른손을 자기 오른쪽 어깨 쪽으로 당기며 전완을 더 세운다.',hand('closed fist','slight up/toward own right shoulder; forearm reorientation','beside upper chest toward own right shoulder'),leftdown,'reorientation',1,[104,107,110,113]),
segment(a,5,'core_candidate','right_closed_hand_reorients_toward_shoulder',[116,130],[113,116],[130,133],'오른쪽 어깨 옆에서 주먹과 세운 전완을 잠시 유지한다.',hand('closed fist','approximately stationary','beside right shoulder; forearm upright/oblique'),leftdown,'post_reorientation_hold',1,[116,118,124,130]),
segment(a,6,'return','right_hand_returns_down',[135,143],[131,135],[143,147],'오른손을 풀며 몸 옆으로 내린다.',hand('closed to relaxed/open transition','down','from right upper chest to right lower torso'),leftdown),
segment(a,7,'rest','hands_down_rest',[148,167],[144,148],[167,167],'양손을 내린 채 다시 대기한다.',resthand('right'),resthand('left'),'hold'),
],['초기 얼굴 앞 편손과 후반 어깨 옆 주먹을 모두 보존했다. 어느 하나만 언어적으로 필수라고 판단하지 않는다.','이 클립에서 동일 활동의 재시작은 보이지 않는다. cycle 1은 이 영상 내 발생 순서일 뿐이다.'])

a='P048'
add(a,[
segment(a,0,'rest','hands_down_rest',[0,57],[0,0],[57,60],'양손을 내린 채 대기한다.',resthand('right'),resthand('left'),'hold'),
segment(a,1,'preparation','left_forearm_extends_right_flat_hand_approaches_upper_arm',[63,72],[58,63],[72,76],'왼팔을 앞쪽으로 내밀며 왼손을 닫고, 편 오른손을 왼팔 위쪽으로 가져간다.',hand('four fingers extended together; flat silhouette','up and toward own left','toward left upper arm/elbow area'),hand('closed fingers; exact thumb detail partly hidden','up/forward','left arm extends in front of torso')),
segment(a,2,'core_candidate','right_flat_hand_moves_along_left_forearm',[77,95],[73,77],[95,98],'편 오른손이 왼팔 위쪽에서 왼쪽 손목 쪽으로 이동한다. 화면상 아래 성분과 팔을 따른 전방 성분이 함께 보인다.',hand('extended fingers together; flat hand','down and distally along opposite arm','over/near left upper arm-to-forearm-to-wrist course; contact pressure unknown'),hand('closed fist silhouette; finger details partly hidden','small settling then near hold','forward left arm serving as spatial reference'),'distal_stroke',1,[77,79,83,88,92,95],['실제 접촉·압력과 정확한 3차원 축은 RGB만으로 확정하지 않는다.']),
segment(a,3,'core_candidate','right_flat_hand_holds_over_left_distal_forearm',[98,102],[95,98],[102,106],'오른손이 왼손목 가까운 전완 위에서 잠시 머문다.',hand('extended fingers together','approximately stationary','over left distal forearm/wrist region'),hand('closed fist silhouette','approximately stationary','left arm held forward'),'hold',1,[98,100,102]),
segment(a,4,'return','both_hands_return_down',[108,118],[103,108],[118,122],'오른손이 왼팔에서 떨어지고 양팔이 아래로 돌아간다.',hand('flat/open to relaxed','down; away from opposite arm','toward right lower torso'),hand('closed to relaxed transition','down','toward left lower torso')),
segment(a,5,'rest','hands_down_rest',[123,143],[119,123],[143,143],'양손을 몸 옆에 내리고 대기한다.',resthand('right'),resthand('left'),'hold'),
],['한 번의 팔을 따른 이동과 끝의 유지가 보인다. 이를 얼굴 관련 동작이나 단순 R-down 비트만으로 치환하지 않는다.','왼팔과 오른손이 겹친 작은 영역의 손가락·접촉점은 관측 한계로 보존한다.'])

thumb=lambda direction,body:hand('thumb extended up; other fingers closed silhouette',direction,body)
a='P066'
add(a,[
segment(a,0,'rest','hands_down_rest',[0,46],[0,0],[46,50],'양손을 내린 채 대기한다.',resthand('right'),resthand('left'),'hold'),
segment(a,1,'preparation','both_closed_thumb_up_hands_lift_to_torso',[52,57],[47,52],[57,60],'왼손부터 몸 앞에 올리고 오른손도 뒤따르며 엄지를 세운 손모양을 만든다.',hand('relaxed to closed/thumb-up transition','up','lower torso to chest'),thumb('up then slows','in front of torso')),
segment(a,2,'core_candidate','right_thumb_up_hand_rises_above_left',[61,66],[58,61],[66,69],'엄지를 세운 오른손이 왼손보다 높아져 얼굴 앞까지 올라간다. 첫 오른손 상승이다.',thumb('up','right hand above left, near face'),thumb('near hold','below right hand in front of chest'),'rise_and_top',1,[61,63,65,66]),
segment(a,3,'core_candidate','left_thumb_up_hand_rises_above_right',[73,77],[68,73],[77,80],'두 손이 교차한 뒤 엄지를 세운 왼손이 위로 올라가고 오른손은 아래쪽에 놓인다. 첫 왼손 상승이다.',thumb('down then slows','below left hand in front of chest'),thumb('up then top','above right hand near face'),'rise_and_top',1,[73,74,76,77],['69–72 교차에서 손가락 일부가 겹친다. 교차 경계는 확실 내부로 확장하지 않았다.']),
segment(a,4,'core_candidate','right_thumb_up_hand_rises_above_left',[83,88],[78,83],[88,91],'오른손이 다시 왼손보다 높아져 얼굴 위쪽으로 올라간다. 두 번째 오른손 상승이다.',thumb('up then top','above left hand near face'),thumb('down then slows','below right hand in front of chest'),'rise_and_top',2,[83,85,87,88],['79–82의 교차·전환 구간은 별도 경계이다.']),
segment(a,5,'core_candidate','left_thumb_up_hand_rises_above_right',[96,100],[90,96],[100,104],'왼손이 다시 위로 올라가 이마 앞쪽에 놓이고 오른손은 턱·가슴 사이 아래쪽에 놓인다. 두 번째 왼손 상승이다.',thumb('down/settling','below left hand near chin/upper chest'),thumb('up then top','above right hand near forehead'),'rise_and_top',2,[96,97,99,100],['91–95의 교차는 가림과 전환을 포함한다. 미관측을 잘못된 손모양으로 처리하지 않는다.']),
segment(a,6,'core_candidate','left_thumb_up_hand_high_hold',[105,119],[100,105],[119,123],'왼손이 높고 오른손이 낮은 배치를 양 엄지를 세운 채 유지한다.',thumb('approximately stationary','lower hand near chin/upper chest'),thumb('approximately stationary','higher hand near forehead'),'hold',2,[105,110,115,119]),
segment(a,7,'return','both_hands_return_down',[125,136],[120,125],[136,140],'높던 왼손과 낮던 오른손을 몸 옆으로 내린다.',hand('thumb-up/closed to relaxed transition','down','toward right lower torso'),hand('thumb-up/closed to relaxed transition','down','toward left lower torso')),
segment(a,8,'rest','hands_down_rest',[141,167],[136,141],[167,167],'양손을 내린 뒤 대기한다.',resthand('right'),resthand('left'),'hold'),
],['동일 kind의 오른손 상승 2개, 왼손 상승 2개를 후보 집합으로 유지한다. 종류 일치와 회차 일치는 별도이다.','R1→L1→R2→L2/hold는 이 RGB에서 보인 발생 순서이며, 수어의 필수 반복 수나 언어적 정답이 아니다.','교차 자체는 보인다. 다만 손가락 세부와 종류 전환 경계가 불명확하므로 그 프레임을 wrong으로 취급하지 않는다.'])

openflat=lambda direction,body:hand('four fingers extended together; thumb separated; finger tips partly edge-on',direction,body)
a='P072'
add(a,[
segment(a,0,'rest','hands_down_rest',[0,61],[0,0],[61,65],'양손을 내린 채 대기한다.',resthand('right'),resthand('left'),'hold'),
segment(a,1,'preparation','bilateral_open_hands_lift_and_approach_center',[67,74],[62,67],[74,78],'편 양손을 올려 얼굴 앞 중앙으로 모은다.',openflat('up/inward','from lower torso toward lower face'),openflat('up/inward','from lower torso toward lower face')),
segment(a,2,'core_candidate','bilateral_open_hands_converge_near_face',[78,86],[74,78],[86,90],'편 양손의 손끝 쪽을 얼굴 앞에서 가까이 모아 삼각형에 가까운 빈 공간을 만든 뒤 머문다.',openflat('small inward/up adjustment then near hold','right side of central triangular opening in front of mouth/nose'),openflat('small inward/up adjustment then near hold','left side of central triangular opening in front of mouth/nose'),'converged_hold',1,[78,80,83,86],['손끝의 정확한 접촉 여부와 각 손가락 세부 각도는 확정하지 않는다.']),
segment(a,3,'core_candidate','bilateral_open_hands_separate_outward_down',[91,95],[87,91],[95,99],'양손을 얼굴 앞 중앙에서 좌우로 벌리며 조금 낮춘다.',openflat('outward toward own right and slightly down','center of lower face toward right shoulder width'),openflat('outward toward own left and slightly down','center of lower face toward left shoulder width'),'separation',1,[91,92,94,95]),
segment(a,4,'core_candidate','bilateral_open_hands_hold_apart_upper_chest',[100,107],[96,100],[107,112],'양손을 어깨 너비 정도로 벌린 채 가슴 위쪽에서 비슷한 높이로 유지한다.',openflat('approximately stationary','apart at right shoulder-width upper chest'),openflat('approximately stationary','apart at left shoulder-width upper chest'),'hold',1,[100,103,105,107]),
segment(a,5,'return','both_hands_return_down',[113,122],[108,113],[122,126],'양팔을 내려 몸 옆으로 돌아간다.',hand('extended to relaxed transition','down','toward right lower torso'),hand('extended to relaxed transition','down','toward left lower torso')),
segment(a,6,'rest','hands_down_rest',[127,143],[122,127],[143,143],'양손을 내린 채 대기한다.',resthand('right'),resthand('left'),'hold'),
],['중앙 모음, 벌림, 벌린 상태 유지를 구분한다. 얇게 보이는 손의 정확한 손바닥 법선은 unknown이다.','이 영상에서 재반복은 보이지 않지만 언어적으로 한 회만 필수라고 선언하지 않는다.'])

result=dict(schema_version=1,status='RGB_PHYSICAL_STAGE_DRAFT_FOR_PREFREEZE_REVIEW',reviewer='same Codex subagent A',
    scope='public DEV4 only; historical development exposure acknowledged; person-pair predictions/results/SUMMARY were not opened during this annotation',
    source_policy='read-only existing RGB display WebM and frame/PTS metadata; no model, landmarks, DTW or search; writes restricted to this new unit',
    frame_index_convention='zero-based native original frame index, inclusive interval endpoints; source PTS from existing media JSON, not 30 fps assumption',
    boundary_policy='Only certain_interior is eligible for same-physical-stage matching. Brackets are uncertain and are not widened into a same-stage label. All other frames remain boundary/transition/unadjudicated, not negatives.',
    hand_convention='anatomical performer R/L; directions describe visible screen movement plus explicitly named body relation',
    repetition_policy='Same proposed_kind_key may have multiple occurrence intervals. Physical type correspondence and occurrence-order correspondence must be reported separately. No one-to-one occurrence is forced.',
    limitations=['Same Codex provisional observation, not independent human/Deaf instructor truth.',
        'Not strictly blinded: reviewer has seen earlier conversations/development results; current individual-pair predictions were not read.',
        'All 624 native RGB frames were viewed in 26 continuous contact sheets; 22 selected frames also viewed in four native-resolution detail boards. This is not dense linguistic ground truth or precise per-frame finger/3D annotation.',
        'Existing display WebM is 960x540 and compressed; contact tiles are cropped/resized for display. Small fingers, occlusion, contact pressure, depth and palm normals may remain unknown.',
        'Stage boundaries are physically continuous; brackets express reviewer uncertainty rather than a proven instantaneous event boundary.',
        'Preparation/core_candidate/return/rest are visible-motion organization only. No sign meaning, required movement, correctness or teaching verdict is inferred.',
        'Kind vocabulary was discussed with the other Codex RGB reviewer after independent observations; common vocabulary is not independent agreement evidence.'],
    viewed_evidence={'view_log':str(UNIT/'review'/'DEV_IMAGES_VIEWED.json'),'contact_manifest':str(UNIT/'evidence'/'DEV_CONTACT_MANIFEST.json'),
        'detail_manifest':str(UNIT/'evidence'/'DEV_DETAIL_MANIFEST.json'),'actual_viewed_images':viewed['files'],
        'native_frames_viewed_count':sum(c['native_frame_count'] for c in clips),'contact_sheet_count':26,'detail_board_count':4,'detail_frame_count':sum(len(d['frame_ids']) for d in details)},clips=clips)
out=UNIT/'review'/'DEV_PHYSICAL_STAGES.json'
out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print('wrote',out,'clips',len(clips),'segments',sum(len(c['segments']) for c in clips),'core',sum(s['role']=='core_candidate' for c in clips for s in c['segments']))
