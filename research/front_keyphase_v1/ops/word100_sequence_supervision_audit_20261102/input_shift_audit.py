"""Descriptive feature diagnostics; no thresholds chosen, no inference."""
from sequence_model import *
from collections import Counter

def describe(v):
    if not len(v):return None
    v=np.asarray(v)
    return dict(points=len(v),finite=bool(np.isfinite(v).all()),abs_max=float(np.abs(v).max()),
        rms_median=float(np.median(np.sqrt((v*v).mean(1)))),rms_p95=float(np.quantile(np.sqrt((v*v).mean(1)),.95)))

def main():
    assert not (J/'INPUT_SHIFT_AUDIT.json').exists();stats=read(D/'NORMALIZATION.json');fit=fit_rows()
    targets={r['clip_id']:{p['frame']:p['action_family'] for p in r['points'] if p['mask']} for r in read(D/'DEV_ACTION_TARGETS.json')['rows']}
    dev=episode.attach_targets(process.add_rgb(process.old.inputs(False)),targets)
    normal={'native','reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest'}
    bins={};conflicts=[]
    for group,rows in [('FIT',fit),('DEV',dev)]:
        for r in rows:
            if group=='DEV' and r['kind'] not in normal:continue
            r['role']=r.get('role',np.where(r['episode']<0,-1,np.where(r['episode']<8,1,2)))
            x,rgb,l,_,_=batch_j([r],stats);x=x[0].numpy();rgb=rgb[0,:,:768].numpy()
            blocks=dict(latent=x[:,:64],rest_observer=x[:,64:74],geometry=x[:,74:88],displacement=x[:,102:116],rgb=rgb)
            selections={'direct_rest':r['episode']==8,'direct_action':(r['episode']>=0)&(r['episode']<8)}
            if group=='DEV':
                a=np.load(J/'predictions/witness_ctc_42'/f"{r['key']}.npz")
                conflict=(a['current_probability'][:,9]>=.8)&(a['activity_probability'][:,1]>=.8)&a['image_available']
                selections['conflict']=conflict
                if conflict.any():
                    positions=np.flatnonzero(conflict)
                    conflicts.append(dict(key=r['key'],kind=r['kind'],frames=positions.tolist(),
                        known_rest=int((conflict&(r['episode']==8)).sum()),known_action=int((conflict&(r['episode']>=0)&(r['episode']<8)).sum()),
                        otherwise_unlabelled=int((conflict&(r['episode']<0)).sum())))
            for label,mask in selections.items():
                for b,v in blocks.items():bins.setdefault((group,label,b),[]).append(v[mask])
    metrics={':'.join(k):describe(np.concatenate(v)) for k,v in bins.items()}
    out=dict(at=clock(),feature_summaries=metrics,conflict_cases=conflicts,conflict_kind_counts=dict(Counter(q['kind'] for q in conflicts)),
        clarification='Native conflict cases concentrate at first0-3frames, but reencoded/speed/trim controls also have longer start/end conflicts. The earlier commentary referring to all normal cases as just a short starting boundary was too broad; saved audit corrects this without changing any scores.',
        conclusion_scope='Descriptive ranges do not establish a causal normalization error or justify clipping. FIT succeeds on supervised starts; do not add automatic rest labels to all starts.',
        new_inference=0,new_training=0,threshold_changes=0,whole_goal_complete=False)
    exclusive(J/'INPUT_SHIFT_AUDIT.json',out)
    print('conflict kinds',out['conflict_kind_counts'])
    for k,v in metrics.items():
        if 'rest_observer' not in k:print(k,v)
if __name__=='__main__':main()
