from pathlib import Path
import json,hashlib,sys
AA=Path(__file__).resolve().parent;O=AA.parent;Z=O/'word100_partial_order_feedback_20261102';X=O/'word100_learned_component_extent_20261102'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    dest=AA/'EXECUTION_FROZEN.json';assert not dest.exists()
    cases=read(AA/'SAVED_CASES_MANIFEST.json')['cases'];assert sum(len(v) for v in cases.values())==16
    files={AA/p for p in ['SCOPE.json','run_demo.py','learner_adapter.py','order_worker.py','demo.html','demo.js','SAVED_CASES_MANIFEST.json']}
    files.update(X/p for p in ['learner_adapter.py','source_bridge.py','run_demo.py','demo.html','demo.js'])
    files.update(Z/p for p in ['order_feedback.py','order_runtime.py','RUNTIME_FROZEN.json','RUNTIME_CONTEXT.json','RUNTIME_RESULT_VERIFIED.json'])
    # Cold imports collect the actual full-capture Python closure. No warmup,
    # camera, model constructor, worker start, or model forward is called.
    sys.path.insert(0,str(AA));import learner_adapter
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file and Path(file).suffix=='.py' and Path(file).resolve().is_relative_to(O.resolve()):files.add(Path(file).resolve())
    old_demo=read(X/'DEMO_FROZEN.json')
    for p,h in old_demo['pins'].items():assert sha(p)==h,p
    files.update(Path(p) for p in old_demo['pins']);files.add(X/'DEMO_FROZEN.json')
    explicit={
        'word100_learned_guide_feedback_20261102':['NORMALIZATION.json','PRETRAINED_PROJECTION.npy','runs/projection_finetune_42/model.pt','runs/projection_finetune_42/COMPLETE.json','runs/frozen_projection_42/model.pt','runs/frozen_projection_42/COMPLETE.json'],
        'word100_learned_feature_retention_20261102':['calibration/42.json'],
        'word100_learned_body_relation_20261102':['NORMALIZATION.json','GUIDE_RELATION_SUPPORT.json','runs/rgb_geometry_42/model.pt'],
        'word100_learned_rgb_motion_20261102':['RGB_NORMALIZATION.json','runs/rgb_residual_42/model.pt'],
        'word100_learned_motion_order_20261102':['NORMALIZATION.json','MOTION_GUIDE_SUPPORT.json','runs/ordered_gru_42/model.pt'],
        'word100_semantic_feature_cycle_20261101':['comparison.py','NORMALIZATION.json','DATASET.json','runs/protected_shape_split_features_42/head.pt','runs/protected_shape_split_features_42/COMPLETE.json'],
        'word100_multipeople_live_feedback_20261101':['adapter.py'],
        'word100_guide_video_feedback_20261028':['motion_observer.py'],
        'word100_signrep_feasibility_20261029':['EXPORT.json','converted/signrep_features_fp32.xml','converted/signrep_features_fp32.bin']}
    for unit,names in explicit.items():files.update(O/unit/name for name in names)
    pins=read(Z/'RUNTIME_FROZEN.json')['pins']
    for p,h in pins.items():assert sha(p)==h,p
    files.update(Path(p) for p in pins)
    for word,kinds in cases.items():
        assert set(kinds)=={'normal','cut','order','gap'}
        for kind,r in kinds.items():
            assert sha(r['video'])==r['video_sha256']
            assert sha(r['result'])==r['result_sha256']
            out=read(r['result']);assert out['word_id']==word and out['source_sha256']==r['video_sha256']
            assert out['partial_order_feedback']['whole_sign_correctness'] is None
            files.update([Path(r['video']),Path(r['result'])])
    out=dict(status='FROZEN_BEFORE_LIVE_SERVER_AND_SINGLE_PUBLIC_INTEGRATION',
        scope=read(AA/'SCOPE.json'),pins={str(p.resolve()):sha(p) for p in sorted(files)},
        fresh_public_word='WORD0739',max_new_public_jobs=1,old_web_default_changed=False,whole_goal_complete=False)
    with dest.open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print('frozen live inputs/code',len(files))

if __name__=='__main__':main()
