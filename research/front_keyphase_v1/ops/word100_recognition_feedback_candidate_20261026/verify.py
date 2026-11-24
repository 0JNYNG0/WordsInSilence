from pathlib import Path
import json,hashlib,datetime
from adapter import PublicResearchReplay
U=Path(__file__).resolve().parent;D=U.parent/'word100_dev_image_feedback_20261026';V=U.parent/'word100_recovery_dev300_20261026'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def verify():
    f=read(U/'MANIFEST.json');replay=PublicResearchReplay(U/'MANIFEST.json');raw=read(D/'FROZEN.json');e3={r['clip_id']:r for r in read(V/'E3_SCORES.json')['predictions']['native']};count=0;unsafe=0
    assert f['default_model']=='C' and f['new_classifier_inference']==f['new_training']==f['new_ranking']==0
    for row in f['rows']:
        query=read(row['query_path']);previous=None
        for model in ('C','T3F','E3'):
            result=replay.get(row['query_id'],model)
            expected=e3[row['query_id']] if model=='E3' else raw['scores'][model]['predictions'][row['query_id']]
            assert {k:result['prediction'][k] for k in ('status','prediction','top5','correct','top5_correct')}=={k:expected[k] for k in ('status','prediction','top5','correct','top5_correct')}
            assert result['practice_target']==dict(word_code=row['word_code'],label=row['label'])
            assert result['whole_gesture_correct'] is result['validated_correction'] is result['skill_score'] is None
            assert not result['inference_performed'] and not result['personal_data'] and not result['live_webcam_supported']
            assert len(result['references'])==2
            for saved,ref in zip(query['references'],result['references']):
                assert ref==read(saved['receipt']) and ref['query_id']==row['query_id'] and ref['reference_id']==saved['reference']
                assert ref['whole_gesture_correct'] is ref['skill_score'] is ref['validated_correction'] is None
                for hand in ref['source_separated_point_evidence']:
                    for angle in hand['angles']:
                        for event in angle['states']:
                            assert event['gesture_error'] is event['correction_required'] is None
                            unsafe+=1
            if previous is not None:assert result['references']==previous
            previous=result['references'];count+=1
        if count%150==0:print('VERIFIED_BINDINGS',count,900,flush=True)
    for qid,model in [('unknown_public_clip','C'),('personal_camera','E3'),(f['rows'][0]['query_id'],'new_model')]:
        try:replay.get(qid,model)
        except ValueError:pass
        else:raise AssertionError('Unsupported input accepted')
    assert count==900
    for path,digest in f['pins'].items():assert sha(path)==digest
    out=dict(at=datetime.datetime.now().astimezone().isoformat(),status='PASS900_SAVED_RECOGNITION_FEEDBACK_BINDINGS',queries=300,models=3,responses=count,paired_references=600,uncertainty_events_checked=unsafe,unsupported_requests_rejected=3,unchanged_feedback_across_models=True,independent_prediction_source_match=True,manifest_sha256=sha(U/'MANIFEST.json'),verifier_sha256=sha(__file__),method='SameCodex separate saved-source identity and nofalsecorrection contract checks; not new score, anatomical or usertest')
    with (U/'VERIFIED.json').open('x',encoding='utf8') as fp:json.dump(out,fp,ensure_ascii=False)
    print(json.dumps(out,ensure_ascii=False))

if __name__=='__main__':verify()
