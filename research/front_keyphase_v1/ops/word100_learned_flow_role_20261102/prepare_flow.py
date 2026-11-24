import time,traceback
from flow_model import *

def main():
    assert not (FLOW_UNIT/'FLOW_INPUTS.json').exists();torch.set_num_threads(2)
    state=torch.load(PROTO,map_location='cpu',weights_only=True);lengths=state['prototype_lengths'].tolist()
    prototype=[state['prototype'][i,:n].numpy() for i,n in enumerate(lengths)]
    guides=read(P/'GUIDE_SUPPORT.json')['guides'];items={};fit=[];durations=[];pins={}
    for p in [PROTO,P/'GUIDE_SUPPORT.json',P/'alignment.py',FLOW_UNIT/'flow_model.py',FLOW_UNIT/'prepare_flow.py',V/'VERIFIED.json',V/'FROZEN.json',L/'PAIR_MANIFEST.json']:
        pins[str(p)]=sha(p)
    for split in ['fit','dev']:
        rr=get_rows(split);assert len(rr)==(252 if split=='fit' else 88)
        assert {r['person'] for r in rr}==set(range(1,10) if split=='fit' else [11,12,13])
        for r in rr:
            if split=='fit':pins[r['cache']]=r['cache_sha256']
            else:
                pins[r['input']]=r['input_sha256'];pins.update(dict(r['extra_pins']))
            start=time.perf_counter();z,path=flow_features(r,prototype,guides);seconds=time.perf_counter()-start;durations.append(seconds)
            assert np.array_equal(np.unique(path[:,0]),np.arange(len(z)))
            assert np.array_equal(np.unique(path[:,1]),np.arange(lengths[process.WORDS.index(r['word_id'])]))
            p=FLOW_OUT/'prepared'/f"{r['flow_key']}.npz";p.parent.mkdir(parents=True,exist_ok=True);assert not p.exists()
            np.savez_compressed(p,features=z,path=path,source_frame_id=r['mapping'],query_frame_id=np.arange(len(z)),image_available=r['available'])
            items[r['flow_key']]=dict(path=str(p),sha256=sha(p),clip_id=r['clip_id'],kind=r['kind'],word_id=r['word_id'],person=r['person'],frames=len(z),alignment_seconds=seconds)
            if split=='fit':fit.append(z[r['available'],:5])
        print(split,'flow prepared',len(rr),flush=True)
    f=np.concatenate(fit);mean=f.mean(0);std=f.std(0);std=np.where(std>1e-6,std,1.)
    for p,h in pins.items():assert sha(p)==h
    exclusive(FLOW_UNIT/'FLOW_INPUTS.json',dict(at=clock(),items=items,mean=mean.tolist(),std=std.tolist(),pins=pins,
        prototype_source=str(PROTO),supervision_changed=False,new_model_inference=0,old_scores_recomputed=0,
        feature_names=['learned_flow_position','alignment_distance','source_canonical_guide_membership_hint','local_reference_compression','local_query_reuse','image_available'],
        normalization_source='FIT01-09 only, available times; no thresholds selected',
        guide_scope='Canonical reviewed span membership is an INPUT HINT after DTW, never a dense target for the new person.',
        runtime_reference_videos_read=0,prototype_comparisons_per_clip=1,alignment_median_seconds=float(np.median(durations)),whole_goal_complete=False))
    print('flow contract: all native times and gaps retained, no new targets')

if __name__=='__main__':
    try:main()
    except BaseException:
        if not (FLOW_UNIT/'PREP_FIRST_FAILURE.txt').exists():(FLOW_UNIT/'PREP_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
