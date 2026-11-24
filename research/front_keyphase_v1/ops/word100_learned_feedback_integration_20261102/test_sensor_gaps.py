"""Keep the actual RGB evidence; interrupt only sensor features at reviewed centers."""
import numpy as np,torch,traceback,time
from common import G,O,R,V,U,P,read,sha,write,exclusive,frozen
from relation_runtime import RelationsEngine
from rgb_runtime import RGBMotionObserver

def main():
    torch.set_num_threads(4);plan,source,_=frozen();assert not (G/'SENSOR_GAP_RESULTS.json').exists()
    engine=RelationsEngine();motion=RGBMotionObserver('rgb_residual',42);rows=[];blind=[]
    for r,s in zip(source,plan['sources']):
        assert r['clip_id']==s['clip_id'];a=dict(np.load(s['input']));n=len(a['frame_id']);word=r['word_id']
        baseline=read(R/'feedback'/f'rgb_residual_42_{word}_reencoded_normal.json')
        old_shape=a['retained_probability']
        for label in r['labels']:
            center=label['center_frame'];y=np.asarray(label['y']);known=np.asarray(label['target_mask'],bool)
            assert a['frame_id'][center]==center and known.any()
            for kind,length in [('pose_only_3',3),('pose_only_7',7),('pose_and_image_7',7)]:
                key=f'{word}_f{center}_{kind}';dest=G/'gap_feedback'/f'{key}.json';pred=G/'gap_predictions'/f'{key}.npz'
                assert not dest.exists() and not pred.exists()
                b={k:v.copy() for k,v in a.items()};ix=np.arange(center-length//2,center+length//2+1);assert ix.min()>=0 and ix.max()<n
                b['x'][ix]=0;b['mask'][ix]=False
                if kind=='pose_and_image_7':b['rgb'][ix]=0;b['prelinear'][ix]=0;b['image_available'][ix]=False
                else:
                    assert np.array_equal(b['rgb'],a['rgb']) and np.array_equal(b['prelinear'],a['prelinear']) and np.array_equal(b['image_available'],a['image_available'])
                t=time.perf_counter();p=engine.infer(b,word);out=engine.compare(p,b,word)
                out=motion.attach(out,b,p['retained_probability'],p['relation_probability']);sec=time.perf_counter()-t
                assert out['whole_sign_correctness'] is None
                if kind=='pose_and_image_7':
                    assert set(ix)<=set(j for gap in out['evidence_gaps'] for j in gap['frames'])
                assert np.array_equal(b['frame_id'],a['frame_id']) and np.array_equal(b['timestamp_us'],a['timestamp_us'])
                exclusive(dest,out);pred.parent.mkdir(exist_ok=True)
                np.savez_compressed(pred,**{k:v for k,v in p.items() if isinstance(v,np.ndarray)})
                v=p['retained_probability'][center];old=old_shape[center];good=np.isfinite(v)&known
                rows.append(dict(case=key,word=word,center=center,kind=kind,frames=ix.tolist(),known=known.tolist(),y=y.tolist(),
                    baseline_probability=old.tolist(),probability=v.tolist(),baseline_correct=int(((old>=.5)==y)[known].sum()),
                    current_correct=int(((v>=.5)==y)[good].sum()),known_count=int(known.sum()),finite_known=int(good.sum()),
                    baseline_motions=baseline['motion_observations'],motions=out['motion_observations'],gap_frames=[j for gap in out['evidence_gaps'] for j in gap['frames']],
                    feedback_statuses=[z['status'] for z in out['stages']],head_pipeline_seconds=sec,original_time_preserved=True,
                    direct_center_observation_available=kind!='pose_and_image_7'))
        # Word selects the guide, never changes the observed shape/relations/latent.
        other=next(w for w in ['WORD0739','WORD1133','WORD1567','WORD2523'] if w!=word)
        one=engine.infer(a,word);two=engine.infer(a,other)
        for k in ['latent','probability','retained_probability','relation_probability']:
            np.testing.assert_allclose(one[k],two[k],rtol=0,atol=0,equal_nan=True)
        blind.append(dict(source=word,alternative_goal=other,observations_identical=True,guide_stage_scores_expected_to_change=True))
        print('gap diagnostics',word,flush=True)
    for p,h in plan['pins'].items(): assert sha(p)==h
    exclusive(G/'SENSOR_GAP_RESULTS.json',dict(rows=rows,goal_blind_checks=blind,no_new_backbone_extraction=True,no_training=True,
        scope='Artificial loss of sensor features at exact public labeled centers. Same actual RGB retained; no missing finger coordinates invented.'))
    print('Sensor diagnostics saved',len(rows),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        if not (G/'GAP_FIRST_FAILURE.txt').exists():(G/'GAP_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
        raise
