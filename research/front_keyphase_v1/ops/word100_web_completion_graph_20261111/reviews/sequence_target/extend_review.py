"""Finalize expanded RGB-only review after parent review; no model access."""
from pathlib import Path
import json,copy,hashlib,datetime,cv2,math
from PIL import Image,ImageDraw
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf-8')
inp=json.loads((H/'INPUTS.json').read_text('utf-8'));d=json.loads((H/'TARGETS.json').read_text('utf-8'))
for r in inp['rows']:
    for p in r['panels']:p['reviewed']=True
write(H/'INPUTS.json',inp)
sources={r['clip_id']:r for r in inp['rows']}
template=copy.deepcopy(d['rows'][0])
def obs(lo,hi,role,text):return dict(source_frame_ids=list(range(lo,hi+1)),role=role,observation_ko=text,unknown=[],basis='direct_RGB_contact_sheet_review')
configs={
11:dict(parts=[('prefix',0,35),('A',36,66),('B',67,79),('C',80,103),('suffix',104,140)],witness={'B':[67,70,72,77,79],'C':[80,82,86,88,90]}),
12:dict(parts=[('prefix',0,30),('A',31,58),('B',59,72),('C',73,101),('suffix',102,122)],witness={'B':[59,60,61,63,68,70,72],'C':[73,76,78,81,84,86]},bands={'entry_onset':[30,34],'face_arrival':[43,46],'close_and_lower_onset':[59,62],'lowering_end':[68,72],'raising_onset':[73,77],'raising_arrival':[82,85],'return_onset':[101,104],'rest_after_return':[113,116]},observations=[
obs(0,30,'rest','양팔을 내리고 대기한다. 대상 동작은 보이지 않는다.'),obs(31,45,'preparation','오른손이 올라가 입·코 앞에 도달한다.31~34의 시작은 작은 변화다.'),obs(46,60,'main_A_hold','펴진 손가락을 얼굴 앞에 유지한다.59~60은 B의 시작 자세에도 해당한다.'),obs(59,72,'main_B_close_and_lower','59~60의 편 손에서61~63 손가락을 접고,63~70에 닫힌 손이 얼굴 앞에서 가슴 높이로 아래·옆 방향으로 이동한다.69~72에는 전완이 가로에 가까운 끝 자세다. 가슴 중앙 접촉은 판정하지 않는다.'),obs(73,85,'main_C_forearm_reorientation','73의 닫힌 손/수평에 가까운 전완 상태에서75~83에 팔꿈치가 내려가며 전완이 더 세워진다. 손 자체의 위쪽 이동은 작지만83~85의 전완 방향 변화가 보인다.'),obs(85,101,'main_C_terminal_hold','세운 전완과 닫힌 손을 유지한다. 별도 반복이 아니다.'),obs(102,115,'return','손을 아래로 내리고 손가락을 풀어 몸 옆으로 돌아간다. 편 손·얼굴 앞에서 시작하는 닫기/내림 동작은 아니다. 전완을 다시 세우는 새 움직임도 보이지 않는다.'),obs(116,122,'rest','복귀한 대기 자세다.')]),
13:dict(parts=[('prefix',0,44),('A',45,76),('B',77,90),('C',91,114),('suffix',115,145)],witness={'B':[77,78,80,82,86,89,90],'C':[91,94,97,99,101]},bands={'entry_onset':[45,49],'face_arrival':[59,62],'close_and_lower_onset':[77,80],'lowering_end':[88,91],'raising_onset':[91,94],'raising_arrival':[99,102],'return_onset':[114,117],'rest_after_return':[125,128]},observations=[
obs(0,44,'rest','양팔을 내리고 대기한다. 대상 동작은 보이지 않는다.'),obs(45,61,'preparation','45~49 작은 준비 변화 뒤 오른손을 올려 얼굴 앞에 도달한다.'),obs(62,78,'main_A_hold','펴진 손가락이 얼굴 앞에 유지된다.77~78은 B의 시작 상태에도 포함된다.'),obs(77,90,'main_B_close_and_lower','77~78은 편 손 시작,79~82는 내려오며 접는 전환,83~89는 닫힌 손이 가슴 높이로 내려오는 구간이다.89~90에 전완이 가로에 가까운 끝 자세를 보인다.'),obs(91,102,'main_C_forearm_reorientation','91의 가로에 가까운 닫힌 손/전완 상태에서93~100에 전완을 더 세운다. 전체 손 위치 이동은 작지만 팔꿈치가 내려가 전완 방향이 변한다. 정확한90도 기준은 두지 않는다.'),obs(102,114,'main_C_terminal_hold','닫힌 손과 더 세운 전완을 유지한다. 별도 새 동작이 아니다.'),obs(115,127,'return','손을 내리며 풀고 몸 옆 대기자세로 돌아간다. 얼굴 앞의 편 손에서 시작하는 B 전체변화나 수평에서 세우는 C 전체변화의 반복은 보이지 않는다.'),obs(128,145,'rest','양팔을 내린 복귀 후 대기 자세다.')])}
for r in d['rows']:
    s=sources[r['clip_id']];r.update(panels=s['panels'],reviewed_frame_ids=sorted({f for p in s['panels'] for f in p['source_frame_ids']}));r['all_source_frames_reviewed']=len(r['reviewed_frame_ids'])==r['frame_count']
    if r['word_id']=='WORD2523':
        if r['candidate']:
            c=r['candidate'];g=c['semantic_direction_order'];g['semantic_order_negative_gold']=False;g['eligible_metric']='source_identity_order_only';g['semantic_relation_to_guide']='UNKNOWN'
            g['exclusion_reason']='The source guide may require only the close/roof-like static posture. B starts in that posture, so A-before-B semantic subsequence can remain even after B,A source permutation. Observed dynamic approach is not proven necessary for the guide. Do not put this clip in a semantic order error denominator.'
            g['label_scope']='Physical source-block provenance only; dynamic descriptor order can be described, but it is not guide semantic negative gold.'
            r['status']='DENSE_RGB_SOURCE_BLOCK_ORDER_ONLY_SEMANTIC_NEGATIVE_EXCLUDED'
        continue
    cfg=configs[r['person']]
    if r['person']!=11:
        r['observations']=cfg['observations'];r['boundary_uncertainty_frames']=cfg['bands'];r['unknown_features']=copy.deepcopy(template['unknown_features'])
        r['block_content']=copy.deepcopy(template['block_content']);r.pop('next_review',None)
    r['block_content']['C']='initial closed near-horizontal posture, full forearm reorientation, terminal hold only; global return remains in suffix'
    r['block_content']['suffix']='original global return plus rest; return starts closed/upright and relaxes downward, not another complete B or C'
    r['status']='DENSE_RGB_REVIEW_COMPLETE_LIMITED_DYNAMIC_PAIR_TARGET'
    block={name:dict(source_span=[lo,hi],source_frame_ids=list(range(lo,hi+1))) for name,lo,hi in cfg['parts']};order=['prefix','A','C','B','suffix']
    q=[f for n in order for f in block[n]['source_frame_ids']];inv={f:i for i,f in enumerate(q)}
    for b in block.values():b['query_frame_span']=[inv[b['source_span'][0]],inv[b['source_span'][1]]]
    joins=[];cursor=0
    for a,b in zip(order,order[1:]):
        cursor+=len(block[a]['source_frame_ids']);left=q[cursor-1];right=q[cursor];joins.append(dict(query_left=cursor-1,query_right=cursor,source_left=left,source_right=right,synthetic_discontinuity=right!=left+1))
    c=copy.deepcopy(template['candidate']);c.pop('preview',None);c.update(id=r['clip_id']+'_complete_dynamic_blocks',blocks=block,source_identity_normal_order=[n for n,_,_ in cfg['parts']],source_identity_permuted_order=order,query_to_source_frame_ids=q,source_to_query_frame_ids=[inv[f] for f in range(r['frame_count'])],splice_joins=joins)
    g=c['semantic_direction_order'];g.update(normal_witness_source_frame_ids=cfg['witness'],permuted_witness_query_frame_ids={k:[inv[f] for f in ids] for k,ids in cfg['witness'].items()},semantic_order_negative_gold=True,eligible_metric='limited_visible_dynamic_descriptor_pair_only',whole_sign_educational_error_gold=False)
    c['mechanical_cut_justification']=f'B ends at source{block["B"]["source_span"][1]} with closed chest-level near-horizontal posture; C starts{block["C"]["source_span"][0]} retaining that start posture before reorientation. C ends at last selected held posture{block["C"]["source_span"][1]}. Original global return remains suffix{block["suffix"]["source_span"]}. Exact physiological transitions remain bounded by the separately recorded uncertainty bands.'
    c['no_residual_target_outside_edit']=dict(reviewed_source_spans=[block['prefix']['source_span'],block['suffix']['source_span']],observation='All outside frames were directly viewed. Prefix is rest. Suffix contains original return plus rest, not a new open-to-closed face-to-chest B or near-horizontal-to-upright C. Return may retain closed/raised endpoint posture briefly; static posture is not a new full dynamic occurrence.',scope='Only the specified complete dynamic B/C descriptors. No claim that all outside frames lack motion or that all static guide predicates are absent.')
    c['preservation_checks']=dict(all_source_frames_exactly_once=sorted(q)==list(range(r['frame_count'])),each_block_internally_ascending=all(all(y==x+1 for x,y in zip(b['source_frame_ids'],b['source_frame_ids'][1:])) for b in block.values()),all_source_frames_visually_reviewed=r['all_source_frames_reviewed'],all_descriptor_witnesses_inside_single_block=all(set(v)<=set(block[k]['source_frame_ids']) for k,v in cfg['witness'].items()),frames_interpolated=False,frames_reversed_within_block=False,frame_count_preserved=len(q)==r['frame_count'],global_return_in_suffix=True)
    r['candidate']=c

for r in d['rows']:
    if r['word_id']!='WORD0739':continue
    c=r['candidate'];q=c['query_to_source_frame_ids'];ids=set(range(0,len(q),4))|{len(q)-1}
    for j in c['splice_joins']:
        if j['synthetic_discontinuity']:ids|={j['query_left'],j['query_right']}
    ids=sorted(ids);cap=cv2.VideoCapture(r['source']);canvas=Image.new('RGB',(1200,math.ceil(len(ids)/5)*260),'white');draw=ImageDraw.Draw(canvas)
    for j,qi in enumerate(ids):
        sf=q[qi];cap.set(cv2.CAP_PROP_POS_FRAMES,sf);ok,bgr=cap.read();assert ok
        im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).crop((500,50,1420,1040));im.thumbnail((236,235));x=j%5*240;y=j//5*260
        canvas.paste(im,(x+(240-im.width)//2,y+23));draw.text((x+3,y+4),f'q{qi} <- source f{sf}',fill='black')
    cap.release();p=H/f'{r["clip_id"]}_permutation_preview_v2.jpg';canvas.save(p,quality=95)
    c['preview']=dict(path=str(p),sha256=sha(p),query_frame_ids=ids,source_frame_ids=[q[i] for i in ids],reviewed=False)
d.update(status='THREE_LIMITED_DYNAMIC_PAIR_TARGETS_ONE_SOURCE_ONLY_TWO_OVERVIEW_ONLY',input_manifest_sha256=sha(H/'INPUTS.json'),finalized_at=None)
d['scope'].update(dense_clips=4,source_frames_dense=516,limited_semantic_descriptor_pairs=3,source_only_permutation_clips=1,overview_only_clips=2)
d['evaluation_contract']['semantic_direction_order']='Only WORD0739 REAL11/12/13 close_chest_dynamic versus raise_forearm_dynamic. Egypt is excluded: static roof posture may satisfy its guide and survives as initial B state.'
d['revision_record']=[dict(reason='Parent source review: Egyptian static precondition survives B,A so no semantic-negative label.',change='Egypt retains source-order map only; no model output consulted.'),dict(reason='Parent capture review: placing return inside C creates an intermediate rest before B.',change='All obedience transforms keep original global return in suffix. Added full RGB review of REAL12/13. No inference has run by reviewer.')]
write(H/'TARGETS.json',d)
print(json.dumps(dict(dense_frames=516,source_panels=sum(len(r['panels']) for r in d['rows']),limited_pairs=3)))
