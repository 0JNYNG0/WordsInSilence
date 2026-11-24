"""One FIT diagnostic inference; saved DEV localization, no score revision."""
from sequence_model import *
from collections import Counter

def main():
    assert not (J/'BOUNDARY_AUDIT.json').exists();torch.set_num_threads(2)
    path=J/'runs/witness_ctc_42/model.pt';m=WitnessNet();m.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));m.eval()
    rows=[r for r in fit_rows() if r['kind']=='original'];assert len(rows)==36
    stats=read(D/'NORMALIZATION.json');items=[]
    for r in rows:
        x,im,l,ay,ry=batch_j([r],stats)
        with torch.inference_mode():a,role,s=m(x,im,l);ac=a.softmax(-1)[0].numpy();rp=role.softmax(-1)[0].numpy()
        _,conflict=hierarchy(rp,r['current'],r['available'])
        known=np.flatnonzero(r['role']>=0)
        items.append(dict(clip_id=r['clip_id'],person=r['person'],frames=len(r['x']),first_direct_role_frame=int(known[0]),
            first_direct_rest_frame=int(np.flatnonzero(r['role']==2)[0]),conflict_frames=np.flatnonzero(conflict).tolist(),
            first10_direct_labels=r['role'][:10].tolist(),first10_rest=r['current'][:10,9].tolist(),first10_main=rp[:10,1].tolist(),
            first10_action=ac[:10].argmax(1).tolist(),first10_available=r['available'][:10].tolist(),
            cache=r['cache'],cache_sha256=r['cache_sha256']))
    dev=read(J/'evaluations/witness_ctc_42.json');normal={'native','reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest'}
    dc=[dict(key=q['key'],kind=q['kind'],frames=q['rest_action_conflict_frames']) for q in dev['sequences'] if q['kind'] in normal and q['rest_action_conflict_frames']]
    df=[f for q in dc for f in q['frames']];ff=[f for r in items for f in r['conflict_frames']]
    out=dict(at=clock(),fit_rows=items,fit_native_inputs=36,fit_conflict_clips=sum(bool(r['conflict_frames']) for r in items),fit_conflict_frames=ff,
        fit_first10_annotated=sum(sum(v>=0 for v in r['first10_direct_labels']) for r in items),fit_first10_total=360,
        dev_saved_conflicts=dc,dev_normal_conflict_clips=len(dc),dev_conflict_frame_histogram=dict(Counter(df)),
        all_dev_conflicts_in_first4=all(f<4 for f in df),new_fit_diagnostic_inferences=36,new_dev_inference=0,new_training=0,old_scores_revised=False,
        scope='Temporal boundary localization, not permission to discard first4frames. A sign may start immediately. No automatic rest labels or semantic correctness inferred from model agreement.',
        pins={str(path):sha(path),str(J/'evaluations/witness_ctc_42.json'):sha(J/'evaluations/witness_ctc_42.json')},whole_goal_complete=False)
    exclusive(J/'BOUNDARY_AUDIT.json',out)
    print({k:v for k,v in out.items() if k not in ['fit_rows','dev_saved_conflicts','pins']})
    print([(r['clip_id'],r['first_direct_role_frame'],r['conflict_frames']) for r in items])
if __name__=='__main__':main()
