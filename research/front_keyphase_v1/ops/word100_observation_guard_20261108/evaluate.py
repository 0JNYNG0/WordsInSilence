"""Frozen one-shot public regression. Reuse existing completed baselines."""
from prepare import *
from guard import quarantine
import torch
sys.path.insert(0,str(J));from runtime_candidate import Candidate,A
def rank(logits,classes,truth):
    v=np.asarray(logits);order=np.argsort(-v,kind='stable')
    return dict(top1=int(order[0]),correct=int(order[0])==truth,rank=int(np.flatnonzero(order==truth)[0]+1),logits=v.tolist())
def infer(model,xy,z):
    a=A.convert(xy,z['score'],z['timestamp_us'],z['frame_id'],int(z['source_width']),int(z['source_height']));x=a['tcn280']
    with torch.inference_mode():return model.tcn(torch.from_numpy(x[None]),torch.ones(1,len(x)))[0].numpy()
def main(mode):
    dest=U/('EVAL_'+mode+'.json');assert not dest.exists();torch.set_num_threads(4)
    frozen=read(U/'GUARD_FROZEN.json');assert all(sha(p)==h for p,h in frozen['pins'].items())
    model=Candidate();f=read(J/'FROZEN.json');saved={r['clip_id']:r for r in read(J/'tcn_runs/finetune_C_20261013/PREDICTIONS_full.json')};out=[]
    if mode=='full':
        rows=[dict(clip_id=cid,word_id=f['classes'][s['y']],raw=read(J/'records'/(cid+'.json'))['raw_path']) for cid,s in saved.items()]
    else:rows=[r for r in read(U/'INPUT_PLAN.json')['rows'] if r['role']!='calibration']
    for r in rows:
        cid=r['clip_id'];truth=f['classes'].index(r['word_id'])
        p=Path(r['raw']) if mode=='full' else Path(r['cache']) if r['cache'] else U/'crops'/(cid+'.npz')
        z=np.load(p);q=z['xy_px'];t=time.perf_counter();g=quarantine(q,z['score'],z['timestamp_us'],int(z['source_width']),int(z['source_height']));cost=time.perf_counter()-t
        changes=np.count_nonzero(np.isfinite(q).all(2)&~np.isfinite(g['xy_px']).all(2))
        base=None
        if mode=='full':base=rank(saved[cid]['logits'],f['classes'],truth)
        else:
            # The four already scored crops must use saved results, not rerun them.
            prev={x['clip_id']:x for x in read(P/'CROP_PROBE.json')['rows']}
            if cid in prev:base=dict(correct=prev[cid]['new_tcn']['top1_correct'],rank=prev[cid]['new_tcn']['truth_rank'],source='previous saved crop prediction')
            else:base=rank(infer(model,q,z),f['classes'],truth)
        after=rank(infer(model,g['xy_px'],z),f['classes'],truth) if changes else dict(base)
        row=dict(clip_id=cid,word=r['word_id'],frames=len(q),baseline=base,guarded=after,changed_points=int(changes),hand_masked=g['hand_quarantined'].sum(0).tolist(),wrist_masked=g['wrist_quarantined'].sum(0).tolist(),guard_seconds=cost)
        if mode=='crop':
            old=np.load(r['raw']);l,t,rr,b=r['crop'];stats=[]
            for side,start in enumerate([91,112]):
                a=old['xy_px'][:,start:start+21];outside=np.count_nonzero((a[:,:,0]<l)|(a[:,:,0]>=rr)|(a[:,:,1]<t)|(a[:,:,1]>=b),axis=1)>=18
                hh=q[:,start:start+21];inside=np.count_nonzero(np.isfinite(hh).all(2)&(hh[:,:,0]>=0)&(hh[:,:,0]<640)&(hh[:,:,1]>=0)&(hh[:,:,1]<480),axis=1)>=18
                stats.append(dict(outside_proxy=int(outside.sum()),false_inframe_before=int((outside&inside).sum()),false_inframe_after=int((outside&inside&~g['hand_quarantined'][:,side]).sum())))
            row['geometric_proxy']=stats
        out.append(row)
        with (U/('EVAL_'+mode+'_EVENTS.jsonl')).open('a',encoding='utf-8') as h:h.write(json.dumps(row)+'\n')
    summary=dict(n=len(out),before=sum(r['baseline']['correct'] for r in out),after=sum(r['guarded']['correct'] for r in out),regressions=[r['clip_id'] for r in out if r['baseline']['correct'] and not r['guarded']['correct']],recoveries=[r['clip_id'] for r in out if not r['baseline']['correct'] and r['guarded']['correct']],guard_ms_median=float(np.median([r['guard_seconds'] for r in out])*1000),changed_clips=sum(r['changed_points']>0 for r in out))
    write(dest,dict(rows=out,summary=summary,existing_development_not_fresh=True,training=False));print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main(sys.argv[1])
