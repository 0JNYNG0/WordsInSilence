from pathlib import Path
import hashlib,json
U=Path(__file__).resolve().parents[1];R=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=read(U/'before_motion_scope_review/TOKEN_OBSERVATIONS.json');new=read(U/'TOKEN_OBSERVATIONS.json')
assert len(old['annotations'])==len(new['annotations'])==38
changes=[];sequence=0;static=0
for a,b in zip(old['annotations'],new['annotations']):
    for k in set(a)-{'observed_features'}:assert a[k]==b[k],(a['annotation_id'],k)
    for f,g in zip(a['observed_features'],b['observed_features']):
        assert g['must_be_correct_for_sign'] is None
        if g['feature'] in ('movement','inter_hand_motion'):
            assert g['feature_scope']=='between_listed_samples_sequence_evidence' and g['instant_frame_motion_label'] is False;sequence+=1
            if a['kind']=='lowering':assert g['value']=='lowering_and_or_low_rest_samples'
        else:assert g['feature_scope']=='listed_samples_coarse_visual_state' and g['instant_frame_motion_label'] is None;static+=1
        for k in set(f)-{'value'}:assert f[k]==g[k],(a['annotation_id'],k)
        if f['value']!=g['value']:assert a['kind']=='lowering';changes.append(a['annotation_id'])
assert 'not instantaneous per-frame targets' in read(U/'LEARNING_CONTRACT.json')['labels']
assert read(U/'RESULT.json')['counts']['annotated_unique_frames']==141
out={'status':'PASS_MOTION_SCOPE_CORRECTION','annotation_groups':38,'unique_sampled_frames':141,'sequence_motion_claims':sequence,'static_visual_claims':static,'lowering_value_changes':changes,'evidence_times_frames_and_ids_unchanged':True,'no_dense_interpolation':True,'current_artifact_hashes':{n:sha(U/n) for n in ['TOKEN_OBSERVATIONS.json','SOURCE_FORM_LINKS.json','ANNOTATION_READINESS.json','LEARNING_CONTRACT.json','RESULT.json']},'source_decode_validation_reused':'TOKEN_ARITHMETIC_REVIEW.json; images and evidence unchanged'}
(R/'MOTION_SCOPE_REVIEW.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False))
