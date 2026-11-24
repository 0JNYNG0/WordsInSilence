from static_core import *

def main():
    D=S/'hand_information_contract';D.mkdir(exist_ok=True);assert not(D/'FROZEN.json').exists()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    rows=read(N/'TRAINING_INPUT_INDEX.json')['rows'];plans=[];features=[]
    ids=np.array([part*290+64+k*10+j for part in range(4)for k in range(10)for j in [0,1,2,3,8]])
    for i,r in enumerate(rows):
        if r['split']!='train'or r['origin']!='positive_source_extent'or r['view']!='full'or r['reverse']:continue
        assert sha(r['path'])==r['sha256'];z=np.load(r['path']);k=int(np.flatnonzero(z['y'])[0]);a=z['local_full'][k];b=z['local_no_hands'][k]
        assert np.array_equal(a[ids],b[ids]);axis=old.AX[r['family']]*2;assert a[64+axis*10+8]>=.8
        plans.append(dict(bag_index=i,clip_id=r['clip_id'],person=r['person'],family=r['family'],candidate_index=k,span=z['spans'][k].tolist(),source_sha256=r['source_sha256'],cache=r['path'],cache_sha256=r['sha256'],explicit_body_relation_equal=True))
        features.append(b)
    assert len(plans)==24 and {r['person']for r in plans}=={2,3,4,5,6,7}
    write(D/'FROZEN.json',dict(at=stamp(),plan=plans,purpose='FIT-only information-path audit: body progression should remain available when hand shape observations are unavailable.',
        transform='Reuse precomputed same-source same-span local_no_hands. Hand coordinates/masks/q-hand removed before frozen common100 encoding; body coordinates/masks/dt kept. Verify pooled explicit body relations bit-equal.',
        selection='all24 FIT full/forward source bags first existing positive candidate; no score-based selection',
        limits='Hand-information erasure is not a real wrong handshape and changes input distribution. Any change proves dependence, not sole cause of current wrong-handshape regression. Not educational correctness.',
        pins={str(p):sha(p)for p in[Path(__file__),S/'TRAINING_COMPLETE.json',PRIOR/'prepare_local_rank.py',PRIOR/'local_rank_core.py']},probe_access=False,private_access=False,final8=False))
    out=[]
    for run in read(S/'TRAINING_COMPLETE.json')['rows']:
        assert sha(run['model'])==run['model_sha256'] and sha(run['predictions'])==run['predictions_sha256'];st=torch.load(run['model'],weights_only=True,map_location='cpu');m=ProgressionHead(st['mean'].numpy(),st['std'].numpy());m.load_state_dict(st);m.eval();previous=np.load(run['predictions'])
        with torch.inference_mode():v=m(torch.tensor(np.asarray(features))).numpy()
        for i,p in enumerate(plans):
            j=old.F.index(p['family']);before=float(previous[f"logits_{p['bag_index']}"][p['candidate_index'],j]);after=float(v[i,j]);out.append(dict(p,seed=run['seed'],original_logit=before,handmasked_logit=after,original_progression=before>=0,handmasked_progression=after>=0,changed=(before>=0)!=(after>=0)))
    summary=[]
    for seed in [42,43,44]:
        r=[x for x in out if x['seed']==seed];summary.append(dict(seed=seed,pairs=len(r),original_positive=sum(x['original_progression']for x in r),handmasked_positive=sum(x['handmasked_progression']for x in r),changed=sum(x['changed']for x in r)))
    write(D/'RESULTS.json',dict(at=stamp(),rows=out,summary=summary,old_model_reinference=False,probe_access=False));print(summary,flush=True)

if __name__=='__main__':main()
