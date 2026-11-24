"""FIT-only fixed-span hand-channel ablation; no training or error relabelling."""
from pathlib import Path
import sys,time,traceback
HERE=Path(__file__).resolve().parent;UNIT=HERE.parent
sys.path.insert(0,str(UNIT))
from common import old,read,write,sha,stamp,torch,np,OPS

def verdict(p,d,ok):
    if not ok:return 'observation_unknown'
    if d==1:return 'supported' if p[1]>=.65 and max(p[0],p[2])<.65 else 'uncertain'
    if p[d]>=.65 and p[2-d]<.65 and p[1]<.65:return 'supported'
    if p[2-d]>=.65 and p[d]<.35 and p[1]<.65:return 'opposite_direction'
    if p[1]>=.65:return 'movement_not_established'
    return 'uncertain'

def main():
    assert not (HERE/'RESULTS.json').exists()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    metadata=read(UNIT/'INPUTS_COMPLETE.json');training=read(UNIT/'TRAINING_COMPLETE.json')
    ready_path=OPS/'word100_shared_motion_learning_20261109/READY.json'
    ready={r['clip_id']:r for r in read(ready_path)['rows'] if r['role']=='fit'}
    selected={}
    for i,r in enumerate(metadata['rows']):
        if r['split']=='train' and r['view']=='full' and r['axis']==2 and not r['reverse']:
            selected.setdefault(r['clip_id'],dict(r,input_row=i))
    assert selected and all(r['person'] in range(1,8) for r in selected.values())
    runs=[r for r in training['rows'] if r['condition']=='reencoded_projected_geometry']
    assert {r['seed'] for r in runs}=={42,43,44}
    pins={str(p):sha(p) for p in [Path(__file__),UNIT/'common.py',UNIT/'INPUTS_COMPLETE.json',UNIT/'TRAINING_COMPLETE.json',ready_path]}
    for r in runs:assert sha(r['model'])==r['model_sha256'];pins[r['model']]=r['model_sha256']
    for cid in selected:
        r=ready[cid];assert sha(r['channel'])==r['channel_sha256'];pins[r['channel']]=r['channel_sha256']
    frozen=dict(at=stamp(),selection='First metadata row per FIT clip: train/full/axis2/reverseFalse, order preserved; no score selection',
                clips=list(selected),rows=list(selected.values()),seeds=[42,43,44],
                variant='Zero x[:,18:186] and x[:,195:279]; preserve body18, bodymask186:195, dt279 and source timestamps. Entire stored sequence uses the same removal.',
                meaning='Missing hand-information intervention, not a changed handshape or an incorrect sign label. No new target assigned.',
                comparison='Same source interval and exact same projected body geometry; full frozen encoder recalculated once per input variant; existing direction heads only.',
                constraints=dict(training=False,development_access=False,final8=False,raw_video_access=False,private_access=False,web_changed=False),pins=pins)
    assert not (HERE/'FROZEN.json').exists();write(HERE/'FROZEN.json',frozen)
    base=old.encoder();rc=old.load('_isolation_req',OPS/'word100_guide_requirement_learning_20261111/requirement_core.py')
    observer=rc.frozen_observer();models={}
    for r in runs:
        model=old.DirectionHead(observer);model.load_state_dict(torch.load(r['model'],map_location='cpu',weights_only=True));model.eval();models[r['seed']]=model
    saved=np.load(metadata['path']);assert sha(metadata['path'])==metadata['sha256']
    records=[];arrays={};start=time.perf_counter()
    for cid,r in selected.items():
        ch=np.load(ready[cid]['channel']);x=ch['full'];ts=ch['timestamp_us'];a,b=r['a'],r['b']
        altered=x.copy();altered[:,18:186]=0;altered[:,195:279]=0
        assert np.array_equal(x[:,:18],altered[:,:18]) and np.array_equal(x[:,186:195],altered[:,186:195]) and np.array_equal(x[:,279],altered[:,279])
        g,ok,info=old.geometry(x,ts,a,b,2,r['moving'],r['anchor'])
        gm,okm,infom=old.geometry(altered,ts,a,b,2,r['moving'],r['anchor'])
        assert np.array_equal(g,gm) and ok==okm and info==infom
        assert np.array_equal(g,saved['projected_geometry'][r['input_row']])
        h=old.hidden(base,old.normalized_pose(x));hm=old.hidden(base,old.normalized_pose(altered))
        ctx=np.r_[h[a],h[b],h[b]-h[a]];ctxm=np.r_[hm[a],hm[b],hm[b]-hm[a]]
        assert np.allclose(ctx,saved['context'][r['input_row']],atol=1e-6)
        arrays[cid+'_normal_context']=ctx;arrays[cid+'_hand_absent_context']=ctxm;arrays[cid+'_geometry']=g
        rr=dict(clip_id=cid,person=r['person'],span=[a,b],source_channel=ready[cid]['channel'],source_channel_sha256=ready[cid]['channel_sha256'],
                geometry_exact=True,normal_context_matches=True,eligible=bool(ok),geometry=info,context_max_abs_difference=float(np.max(np.abs(ctx-ctxm))),
                desired=r['target'],desired_source='Existing normal physical direction, NOT a new error label for missing hands',predictions=[])
        for seed,model in models.items():
            with torch.inference_mode():logits=model(torch.tensor(np.stack([ctx,ctxm])),torch.tensor(np.stack([g,g]))).numpy()
            exp=np.exp(logits-logits.max(-1,keepdims=True));prob=exp/exp.sum(-1,keepdims=True)
            states=[verdict(p,r['target'],ok) for p in prob]
            rr['predictions'].append(dict(seed=seed,normal_logits=logits[0].tolist(),hand_absent_logits=logits[1].tolist(),
                normal_probabilities=prob[0].tolist(),hand_absent_probabilities=prob[1].tolist(),normal_state=states[0],hand_absent_state=states[1],
                state_changed=states[0]!=states[1],class_changed=int(prob[0].argmax())!=int(prob[1].argmax()),probability_max_abs_difference=float(np.max(np.abs(prob[0]-prob[1])))))
        records.append(rr)
    out=HERE/'saved_features.npz';np.savez_compressed(out,**arrays)
    for p,h in pins.items():assert sha(p)==h,p
    preds=[p for r in records for p in r['predictions']]
    summary=dict(clips=len(records),seeds=len(models),paired_outputs=len(preds),geometry_exact_all=True,normal_context_matches_all=True,
                 context_changed_clips=sum(r['context_max_abs_difference']>1e-6 for r in records),
                 verdict_changed=sum(p['state_changed'] for p in preds),argmax_changed=sum(p['class_changed'] for p in preds),
                 max_probability_change=max(p['probability_max_abs_difference'] for p in preds))
    write(HERE/'RESULTS.json',dict(at=stamp(),status='COMPLETED_FIT_DIAGNOSTIC',rows=records,summary=summary,seconds=time.perf_counter()-start,
        saved_features=str(out),saved_features_sha256=sha(out),constraints=frozen['constraints'],reviewer='Delegated Codex motion_contract_review_1113; not human/KSL expert labels',
        interpretation='Any output change with exact fixed body geometry demonstrates hand-channel context influence at this fixed span. It does not establish whether each changed output is educationally wrong or prove accuracy of a new geometry-only model.'))
    print(summary,flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        HERE.mkdir(parents=True,exist_ok=True)
        (HERE/f'FIRST_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8')
        raise
