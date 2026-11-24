"""Read saved guide provenance and freeze sparse, non-normative reference links."""
from pathlib import Path
import json,hashlib,collections
import numpy as np
U=Path(__file__).resolve().parent;ROOT=U.parent;O=ROOT.parent
def load(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf-8')
FINGERS=['thumb','index','middle','ring','little']
def main():
    support=O/'word100_learned_guide_feedback_20261102/GUIDE_SUPPORT.json'
    cards=O/'word100_guide_video_feedback_20261028/GUIDE_CARDS.json'
    clauses=O/'word100_predicate_supervision_20261102/CLAUSE_COMPONENT_LINKS.json'
    physical=O/'word100_guide_feature_supervision_20261102/PHYSICAL_GUIDE_BINDING_AUDIT_expanded.json'
    recent=O/'word100_guide_requirement_learning_20261111/GUIDE_CAPABILITY.json'
    rgb=ROOT/'annotation_review/ORDINARY_LABELS.json'
    files=[support,cards,clauses,physical,recent,rgb]
    guides=load(support)['guides']; labels=load(rgb)['rows']; provenance=[{'path':str(p),'sha256':sha(p)} for p in files]
    stage_rows=[];bindings=[];unbound=[];official={}
    action={
      'WORD0739:stage01':'오른손을 얼굴 앞에 세우는',
      'WORD0739:stage02':'손을 닫으면서 가슴 쪽으로 내리는',
      'WORD0739:stage03':'닫힌 오른손을 유지하며 팔뚝을 세우는',
      'WORD1133:stage01':'오른손으로 왼팔을 따라 쓸어내리는',
      'WORD1133:stage02':'오른손이 왼손목 부근에 이르는',
      'WORD1567:stage01':'양손을 번갈아 올리는',
      'WORD1567:stage02':'양손을 번갈아 올리는',
      'WORD1567:stage03':'양손을 번갈아 올리는',
      'WORD1567:stage04':'양손을 번갈아 올리는',
      'WORD1567:stage05':'양손의 높낮이를 유지하는',
      'WORD2523:stage01':'두 손을 가운데로 모으는',
      'WORD2523:stage02':'두 손을 양옆으로 옮기는',
      'WORD2523:stage03':'두 손을 떨어뜨린 자세를 유지하는'}
    for i,g in enumerate(guides):
        word=g['word_id'];cid=f'NIA_SL_{word}_REAL01_F'
        rec=load(O/'word100_rtmw_joint_learning_20261108/records'/f'{cid}.json');raw=np.load(rec['raw_path']);times={int(f):int(t) for f,t in zip(raw['frame_id'],raw['timestamp_us'])}
        a,b=g['frames'];stage=dict(stage_id=g['canonical_stage_id'],word_id=word,action_phrase=action[g['canonical_stage_id']],reference_clip_id=cid,reference_source_sha256=rec['source_sha256'],native_frame_interval_inclusive=[a,b],native_time_interval_us_inclusive=[times[a],times[b]],native_time_provenance={'path':rec['raw_path'],'sha256':rec['raw_sha256']},guide_observation=g['description'],source_pointer=f'/guides/{i}',source_file=str(support),source_sha256=sha(support),source_scope=g['description_scope'],source_cards=g['source_cards'],educational_tolerance_verified=False,full_interval_finger_states_verified=False)
        assert sha(rec['raw_path'])==rec['raw_sha256']
        stage_rows.append(stage)
        for c in g['source_cards']:
            # Preserve historical flags: a saved official phrase is not an official frame label.
            sp=Path(c['source']['path']);assert sha(sp)==c['source']['sha256']
            official[c['clause_id']]=dict(clause_id=c['clause_id'],text=c['text'],url=c['url'],source=c['source'],context_only=c['context_only'],source_action_verified=c['source_action_verified'],full_span_semantic_verified=c['full_span_semantic_verified'])
        for r in [r for r in labels if r['clip_id']==cid and a<=r['frame']<=b]:
            for finger,state in zip(FINGERS,r['states']):
                if state=='U':continue
                explicit=word=='WORD1567' and finger=='thumb' and state=='E'
                ids=[c['clause_id'] for c in g['source_cards']]
                bindings.append(dict(binding_id=f'{g["canonical_stage_id"]}:{r["frame"]}:{r["side"]}:{finger}',word_id=word,stage_id=g['canonical_stage_id'],hand=r['side'],finger=finger,reference_state=state,action_phrase=action[g['canonical_stage_id']],reference_anchors=[dict(clip_id=cid,frame=r['frame'],native_time_us=r['pts_us'],source_sha256=r['source_sha256'],image=r['image'],image_sha256=r['image_sha256'],annotation_file=str(rgb),annotation_sha256=sha(rgb))],clause_ids=['ancestor_thumbs'] if explicit else [],related_context_clause_ids=ids,source_directly_names_finger_state=explicit,explicit_clause_id='ancestor_thumbs' if explicit else None,source_requirement_level='explicit_thumb_extension_text_only' if explicit else 'reference_rgb_component_only',comparison_scope='sparse_reference_anchor_not_mandatory_word_rule',stage_interval_provenance=stage,allowed_output='observation_or_reference_difference_only_after_runtime_validation',educational_tolerance_verified=False))
    linked={(x['reference_anchors'][0]['clip_id'],x['reference_anchors'][0]['frame']) for x in bindings}
    for r in labels:
        if r['person']==1 and r['word'] in {'WORD0739','WORD1133','WORD1567','WORD2523'} and r['known'] and (r['clip_id'],r['frame']) not in linked:
            unbound.append(dict(clip_id=r['clip_id'],frame=r['frame'],side=r['side'],states=r['states'],reason='outside_saved_canonical_stage_interval; do not extrapolate to a clause'))
    output=dict(status='SOURCE_AND_SPARSE_ANCHOR_CONTRACT_ONLY_NOT_RUNTIME_RELEASE',sources=provenance,official_clauses=list(official.values()),stages=stage_rows,bindings=bindings,unbound_anchors=unbound,unsupported=['thumb_B_or_F','finger_abduction_or_spread','true_contact','pressure','whole_word_correctness','educational_tolerance','automatic_source_variant_resolution'],summary={'bindings':len(bindings),'explicit_finger_text_bindings':sum(b['source_directly_names_finger_state'] for b in bindings),'mandatory_educational_finger_rules':0,'unbound_anchor_hand_rows':len(unbound),'new_inference':0,'new_training':0},note='5지 means thumb in this saved source terminology, not all five fingers; sparse RGB observations are not frame-dense supervision. Every existing false semantic-verification flag is preserved.')
    write(U/'GUIDE_FINGER_BINDINGS.json',output);print(json.dumps(output['summary']))
if __name__=='__main__':main()
