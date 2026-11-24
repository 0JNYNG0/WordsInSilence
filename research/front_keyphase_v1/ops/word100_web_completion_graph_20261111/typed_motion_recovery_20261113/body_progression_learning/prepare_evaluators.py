from body_core import *

def main():
    changes=[]
    for name in ['evaluate_full.py','evaluate_local.py']:
        src=P/name; dst=B/name; assert not dst.exists()
        before=src.read_text('utf-8')
        after=before.replace('from presence_core import *','from body_core import *\nP=B',1).replace("P/'presence_core.py'","B/'body_core.py'")
        if name=='evaluate_full.py':
            needle="    for r in read(P/'TRAINING_COMPLETE.json')['rows']:"
            add="""    runtime=read(B/'RUNTIME_FEATURES_COMPLETE.json')
    runtime_map={(r['clip_id'],r['kind']):r for r in runtime['rows']}
    assert len(runtime_map)==82
    for pr in pm.values():
        br=runtime_map[(pr['clip_id'],pr['kind'])]
        assert pr['source']==br['source'] and pr['source_sha256']==br['source_sha256']
        assert pr['feature_path']==br['original_feature'] and pr['feature_sha256']==br['original_feature_sha256']
        pr['feature_path']=br['path'];pr['feature_sha256']=br['sha256']
"""
            assert after.count(needle)==1;after=after.replace(needle,add+needle)
            after=after.replace('cached frozen common100 features of exact same82scenarios; no new encoder inference','prepared body-only common100 features of exact same82 scenarios/spans; new-head inference only here')
            after=after.replace("B/'body_core.py',P/'TRAINING_COMPLETE.json'","B/'body_core.py',B/'RUNTIME_FEATURES_COMPLETE.json',P/'TRAINING_COMPLETE.json'")
        dst.write_text(after,'utf-8')
        changes.append(dict(source=str(src),source_sha256=sha(src),output=str(dst),output_sha256=sha(dst),same_scoring_and_denominators=True,body_only_inputs=True))
    src=S/'evaluate_static.py';dst=B/'evaluate_static.py';assert not dst.exists()
    after=src.read_text('utf-8').replace('from static_core import *','from body_core import *\nS=B',1)
    after=after.replace("meta['bags']","meta['static_bags']").replace("S/'static_core.py'","B/'body_core.py'")
    after=after.replace('training diagnostic on matched inputs in augmentation pool','training diagnostic on same source pose/span; input representation now body-only')
    after=after.replace('Match216 saved P audit rows exactly on seed/bag/candidate/span/start-middle-end anchor.','Match216 saved P audit rows on seed/bag/candidate/span/start-middle-end source pose. Representation changes to body-only, so not identical model input.')
    dst.write_text(after,'utf-8')
    changes.append(dict(source=str(src),source_sha256=sha(src),output=str(dst),output_sha256=sha(dst),same_scoring_and_denominators=True,body_only_inputs=True))
    write(B/'EVALUATOR_REUSE.json',dict(at=stamp(),rows=changes,old_sources_unchanged=True,new_eval_no_threshold_selection=True))

if __name__=='__main__':main()
