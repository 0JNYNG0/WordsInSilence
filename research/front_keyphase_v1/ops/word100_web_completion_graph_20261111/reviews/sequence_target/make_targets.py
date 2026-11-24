"""Serialize direct RGB observations and two narrowly scoped permutation targets.

No predictor, model, channel array, label tensor, or external source is read.
"""
from pathlib import Path
import json, hashlib, datetime, cv2, math
from PIL import Image, ImageDraw
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n','utf-8')
inp=json.loads((H/'INPUTS.json').read_text('utf-8'))
for r in inp['rows']:
    for p in r['panels']:p['reviewed']=True
write(H/'INPUTS.json',inp)

def observation(ids,role,description,uncertain=None):
    return dict(source_frame_ids=ids,role=role,observation_ko=description,unknown=uncertain or [],basis='direct_RGB_contact_sheet_review')

rows=[]
for src in inp['rows']:
    rec=dict(src)
    rec['reviewed_frame_ids']=sorted({f for p in src['panels'] for f in p['source_frame_ids']})
    rec['all_source_frames_reviewed']=len(rec['reviewed_frame_ids'])==src['frame_count']
    rec['frame_numbering']='zero-based source video frames; all endpoint ranges inclusive'
    rec['public_split_use']='historically exposed development material, diagnostic/regression only; never fresh holdout or FIT'
    rec['unknown_features']=['exact 3D palm orientation','physical contact','occluded individual digits','KSL expert semantic correctness','all acceptable linguistic variants']
    if src['person']!=11:
        if src['word_id']=='WORD0739':
            spans=([32,44,60,68,76,84,100,116] if src['person']==12 else [48,60,76,88,92,100,112,128])
            desc='4프레임 간격 개요에서 편 손을 얼굴 앞에 유지한 뒤 닫으며 가슴 높이로 내리고, 닫힌 손의 전완을 세운 뒤 복귀하는 흐름을 보았다. 정확한 전환 경계·빠른 잔여 동작은 이 간격만으로 확정하지 않는다.'
        else:
            spans=([32,44,52,56,60,76,92] if src['person']==12 else [44,52,56,60,64,68,80,96])
            desc='4프레임 간격 개요에서 양손을 올려 손끝을 가까이 둔 뒤 양손을 벌려 유지하고 내리는 흐름을 보았다. 엄지 모양의 세부 변화와 동작 경계는 확대·밀집 검토 전 미확정이다.'
        rec.update(status='OVERVIEW_ONLY_NO_PERMUTATION_GOLD',observations=[observation(spans,'coarse_overview',desc)],candidate=None)
        rec['next_review']='Review every frame around preparation, all transitions and return, inspect visible hand details, then define a new target before looking at diagnostic model output.'
        rows.append(rec);continue
    if src['word_id']=='WORD0739':
        rec['observations']=[
            observation(list(range(0,36)),'rest','양팔이 아래로 내려간 대기 상태다. 대상 주요 동작은 보이지 않는다.'),
            observation(list(range(36,53)),'preparation','오른손이 몸 아래에서 올라와 얼굴 앞에 도달한다. 36~39의 작은 시작 변화와 50~53의 도착 경계는 불확실하다.'),
            observation(list(range(53,69)),'main_A_hold','오른손의 펴진 손가락이 코·입 앞에 유지된다. 이 정적 모습은 B 시작 프레임에도 남는다. 별도 반복으로 세지 않는다.',['exact face contact','exact finger contact']),
            observation([67,68,69,70,71,72,73,74,75,76,77,78,79],'main_B_close_and_lower','67~68은 편 손 시작 상태, 69~71은 손을 내리며 접는 전환, 72~79는 닫힌 손이 가슴 높이 쪽으로 내려와 수평에 가까운 전완 자세에 도달한다.',['precise motion endpoint within 77..80']),
            observation(list(range(80,90)),'main_C_forearm_reorientation','80은 닫힌 손·가로에 가까운 전완의 출발 상태다. 81~88에 손을 올리며 전완을 더 세우고, 88~90은 세운 끝 자세다.',['exact onset within 80..82','exact angle']),
            observation(list(range(90,104)),'main_C_terminal_hold','닫힌 손과 세운 전완 자세를 유지한다. 유지 프레임은 별도 동작이 아니다.'),
            observation(list(range(104,120)),'return','104~106부터 손이 아래로 움직이고 손가락이 풀리며 몸 옆으로 돌아간다. 주먹 모양이 잠깐 남는다고 B(편 손에서 닫으며 얼굴→가슴 내리기)의 재실행으로 세지 않는다.'),
            observation(list(range(120,141)),'rest','양팔이 내려간 복귀 후 대기 상태다. 대상 주요 동작은 보이지 않는다.')]
        rec['boundary_uncertainty_frames']={'entry_onset':[35,39],'face_arrival':[50,53],'close_and_lower_onset':[68,70],'lowering_end':[77,80],'raising_onset':[80,82],'raising_arrival':[88,90],'return_onset':[103,106],'rest_after_return':[118,120]}
        parts=[('prefix',0,35),('A',36,66),('B',67,79),('C',80,119),('suffix',120,140)]
        order=['prefix','A','C','B','suffix']
        rec['block_content']={'A':'preparation plus complete deliberate open-front held presentation; shared initial posture at B does not mean A is absent there','B':'initial extended face-level hand, closing/lowering, closed chest-level arrival','C':'initial closed near-horizontal posture, complete raising/reorientation, terminal hold, entire return; return is not another B'}
        semantic=dict(target_pair=['close_chest_dynamic','raise_forearm_dynamic'],normal_relation='B_before_C',permuted_relation='C_before_B',normal_witness_source_frame_ids={'B':[67,70,72,77,79],'C':[80,82,86,88,90]},label_scope='Temporal order of these two complete visible dynamic descriptor occurrences, ignoring splice jumps. Does not label all guide pairs reversed.',excluded_pairs=[dict(pair=['open_front','close_chest'],reason='Static open-front posture necessarily survives inside the complete closing action initial state. Source A identity and this static descriptor are not interchangeable.'),dict(pair=['open_front','raise_forearm'],reason='Not the negative target of this permutation; A remains before C.')])
        cut_reason='79 and80 both show the closed chest-level near-horizontal endpoint/start posture. The B descriptor is complete by79; C retains its own start80 before clear reorientation81+. Cut79|80 is a mechanical edit coordinate, not an exact physiological boundary.67..68 preserve B initial open posture, so A-related order cannot be called universally wrong.'
    else:
        rec['observations']=[
            observation(list(range(0,33)),'rest','양팔을 아래로 내린 대기 상태다. 대상 주요 동작은 보이지 않는다.'),
            observation(list(range(33,43)),'preparation','양손을 몸 아래에서 앞쪽으로 올린다.33~36의 시작은 작고 손가락을 펴는 준비가 함께 나타난다.'),
            observation(list(range(42,53)),'main_A_approach','양손 손가락들이 비스듬히 펴진 상태에서 두 손끝이 가까워진다.44~48에는 떨어져 있던 손끝이49~52에 가까운 삼각형 윤곽을 만든다. 엄지는 각 손에서 옆으로 벌어진 모습이 보인다.',['exact fingertip contact']),
            observation([52,53,54,55],'main_A_terminal_hold','가까워진 양손끝의 삼각형 윤곽을 유지한다.54|55에서 A의 끝 자세와 B의 출발 자세를 각 블록에 보존할 수 있다.'),
            observation(list(range(55,65)),'main_B_separate','55은 가까운 손끝 시작 상태,56~61에 양손끝이 서로 멀어지며 손가락들이 가로에 가까운 방향으로 바뀐다.62~64에는 떨어진 끝 자세에 도달한다.',['thumb-to-index aperture after reorientation is not reliably visible; unknown, not closed-hand or fist']),
            observation(list(range(64,77)),'main_B_terminal_hold','양손이 떨어진 가로에 가까운 자세를 유지한다. 별도 새 벌림 동작이 아니다.'),
            observation(list(range(77,91)),'return','양손을 내리고 몸 옆으로 돌아간다. 이 동안 손 사이 간격만 작아지는 모습은 삼각형 끝점을 만드는 A의 완전 재실행으로 세지 않는다.'),
            observation(list(range(91,106)),'rest','양팔이 아래로 내려간 복귀 후 대기 상태다. 대상 주요 동작은 보이지 않는다.')]
        rec['boundary_uncertainty_frames']={'entry_onset':[32,36],'approach_onset':[40,44],'approach_arrival':[49,52],'separation_onset':[55,57],'separation_arrival':[61,64],'return_onset':[75,78],'rest_after_return':[88,91]}
        parts=[('prefix',0,32),('A',33,54),('B',55,90),('suffix',91,105)]
        order=['prefix','B','A','suffix']
        rec['block_content']={'A':'preparation plus complete bilateral fingertip-approach with terminal triangle-like held posture','B':'close-start posture, complete bilateral fingertip separation/reorientation, terminal hold, entire return'}
        semantic=dict(target_pair=['bilateral_fingertip_approach_to_triangle','bilateral_fingertip_separation'],normal_relation='A_before_B',permuted_relation='B_before_A',normal_witness_source_frame_ids={'A':[42,44,46,48,50,52,54],'B':[55,57,59,61,64]},label_scope='Order of the visible approach-to-triangle and separation descriptors with original internal direction preserved, ignoring splice jumps. Not gold for hidden thumb closure, all-finger contact or whole-sign educational correctness.',excluded_pairs=[])
        cut_reason='54 and55 retain the close triangle-like endpoint/start posture. A has completed the visible approach before54; B begins55 before clear separation56+. Entire source return77..90 belongs to B, leaving only reviewed rest outside the edit.'
    block={name:dict(source_span=[lo,hi],source_frame_ids=list(range(lo,hi+1))) for name,lo,hi in parts}
    q=[f for name in order for f in block[name]['source_frame_ids']]
    inverse={f:i for i,f in enumerate(q)}
    for name,b in block.items():b['query_frame_span']=[inverse[b['source_span'][0]],inverse[b['source_span'][1]]]
    joins=[]
    cursor=0
    for a,b in zip(order,order[1:]):
        cursor+=len(block[a]['source_frame_ids'])
        left=q[cursor-1];right=q[cursor]
        joins.append(dict(query_left=cursor-1,query_right=cursor,source_left=left,source_right=right,synthetic_discontinuity=right!=left+1))
    rec['candidate']=dict(id=src['clip_id']+'_complete_blocks',status='SOURCE_RGB_TARGET_AND_INDEX_MAP_READY_NOT_MODEL_EVALUATED',blocks=block,source_identity_normal_order=[n for n,_,_ in parts],source_identity_permuted_order=order,query_to_source_frame_ids=q,source_to_query_frame_ids=[inverse[f] for f in range(src['frame_count'])],semantic_direction_order=semantic,splice_joins=joins,mechanical_cut_justification=cut_reason,no_residual_target_outside_edit=dict(reviewed_source_spans=[block['prefix']['source_span'],block['suffix']['source_span']],observation='Both outside spans were reviewed frame by frame and show lowered-arm rest. All preparation, complete target actions, terminal holds and return are inside the moved blocks.',scope='At the reviewed visible descriptor resolution; this is not an expert guarantee of every hidden movement.'),join_policy='A jump between source-discontinuous frames is not a performed action or direction observation. No across-join trajectory is semantic gold. A model with unmasked temporal context may react to splice artifacts; report that limitation rather than selecting a seam radius from results.',preservation_checks=dict(all_source_frames_exactly_once=sorted(q)==list(range(src['frame_count'])),each_block_internally_ascending=all(all(y==x+1 for x,y in zip(b['source_frame_ids'],b['source_frame_ids'][1:])) for b in block.values()),all_source_frames_visually_reviewed=rec['all_source_frames_reviewed'],all_descriptor_witnesses_inside_single_block=all(set(v)<=set(block[k]['source_frame_ids']) for k,v in semantic['normal_witness_source_frame_ids'].items()),frames_interpolated=False,frames_reversed_within_block=False,frame_count_preserved=len(q)==src['frame_count']))
    rec['status']='DENSE_RGB_REVIEW_COMPLETE_TWO_DESCRIPTOR_PERMUTATION_TARGET'
    rows.append(rec)

targets=dict(task='public_development_complete_action_block_sequence_targets',created_at=datetime.datetime.now().astimezone().isoformat(),author='Codex sequence_target_review agent',reviewer_disclosure='Direct visual review and arithmetic checks by the same Codex agent; not independent human or KSL expert validation.',status='TWO_DENSE_TARGETS_FOUR_OVERVIEW_ONLY',input_manifest=str(H/'INPUTS.json'),input_manifest_sha256=sha(H/'INPUTS.json'),scope=dict(public_clips=6,dense_clips=2,source_frames_dense=247,new_model_inference=False,new_training=False,private_or_mentor_access=False,download=False,source_edits=False,web_changes=False,git_actions=False),exposure_disclosure=dict(historical_DEV_already_exposed=True,requested_historical_sequence_truth_audit_read=True,historical_audit_contains_old_model_state_fields=True,new_INTERVAL_or_other_current_prediction_outputs_read=False,annotation_basis='Only original RGB and prior descriptor/occurrence guidance; no current prediction used to select spans or labels.'),evaluation_contract=dict(source_identity_order='Exact algebraic order of specified source blocks. This alone does not establish semantic order.',semantic_direction_order='Only the named complete visible descriptor pair, with each action internal source order preserved; temporal joins excluded as performed actions.',whole_sign_semantic_completion='UNKNOWN; no expert semantic completion gold supplied.',static_posture_warning='The initial pose of a later dynamic action may also satisfy an earlier static descriptor. Do not count all reversed source identities as reversed semantic guide pairs.',natural_error_performance='NOT_ESTABLISHED. Splice-discontinuous synthetic diagnostics on repeated DEV cannot estimate real camera error performance.',not_for_training_or_tuning=True,no_accuracy_denominator_from_overview_only=True,normal_control_required='Use the same frames in original order; preserve the same extraction and measurement method.',no_cherry_pick='Freeze these source-derived maps and label scope before any new model output; retain both success and failure results.'),rows=rows)
write(H/'TARGETS.json',targets)

# Preview exactly the mapped frames. This is a diagnostic contact sheet, not inference.
for r in rows:
    if not r['candidate']:continue
    q=r['candidate']['query_to_source_frame_ids'];indices=set(range(0,len(q),4))|{len(q)-1}
    for s in r['candidate']['splice_joins']:
        if s['synthetic_discontinuity']:indices|={s['query_left'],s['query_right']}
    ids=sorted(indices);cap=cv2.VideoCapture(r['source']);canvas=Image.new('RGB',(5*240,math.ceil(len(ids)/5)*260),'white');draw=ImageDraw.Draw(canvas)
    for j,qi in enumerate(ids):
        sf=q[qi];cap.set(cv2.CAP_PROP_POS_FRAMES,sf);ok,bgr=cap.read();assert ok
        im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).crop((500,50,1420,1040));im.thumbnail((236,235));x=j%5*240;y=j//5*260
        canvas.paste(im,(x+(240-im.width)//2,y+23));draw.text((x+3,y+4),f'q{qi} <- source f{sf}',fill='black')
    cap.release();p=H/f'{r["clip_id"]}_permutation_preview.jpg';canvas.save(p,quality=95)
    r['candidate']['preview']=dict(path=str(p),sha256=sha(p),query_frame_ids=ids,source_frame_ids=[q[i] for i in ids],reviewed=False)
write(H/'TARGETS.json',targets)
checks=[]
for r in rows:
    assert sha(r['source'])==r['source_sha256']
    assert all(sha(p['path'])==p['sha256'] for p in r['panels'])
    if r['candidate']:assert all(r['candidate']['preservation_checks'][k] for k in ['all_source_frames_exactly_once','each_block_internally_ascending','all_source_frames_visually_reviewed','all_descriptor_witnesses_inside_single_block','frame_count_preserved'])
    checks.append(dict(clip_id=r['clip_id'],reviewed_frames=len(r['reviewed_frame_ids']),total_frames=r['frame_count'],dense=r['all_source_frames_reviewed'],hashes_match=True))
write(H/'CHECK.json',dict(status='PASS_SAME_AGENT_SOURCE_HASH_AND_MAP_ARITHMETIC',rows=checks,target_file_sha256=sha(H/'TARGETS.json'),no_model_execution=True,independent_expert_review=False))
print(json.dumps(checks))
