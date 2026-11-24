"""Separate saved-output/input accounting and NumPy GRU gate verification.
No candidate forward calls, trainer imports, optimizer or inference rescoring.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
U=Path(__file__).resolve().parent;O=U.parent
B=O/'word100_body_boundary_20261017';T=O/'word100_frozen_c_temporal_20261019'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

assert not (U/'INDEPENDENT.json').exists()
s=read(U/'SCOPE.json');r=read(U/'RESULT.json');idx=read(U/'COMPARISON_INDEX.json')
assert r['status']=='PASS_PREFLIGHT_ONLY_NOT_TRAINED' and r['scope_sha256']==sha(U/'SCOPE.json')
for p,h in s['pins'].items():assert sha(p)==h,p
for p,h in r['files'].items():assert sha(U/p)==h,p
assert len(idx)==899*2*2*3
expected={(x['clip_id'],v,g,seed) for x in s['rows'] for v in s['views'] for g in s['segments'] for seed in s['seeds']}
actual={(x['clip_id'],x['view'],x['segment'],x['seed']) for x in idx}
assert actual==expected and len(actual)==len(idx)
with np.load(U/'initial_outputs.npz',allow_pickle=False) as z:
    a=z['candidate'];b=z['reference'];old=z['stored'];si=z['stored_indices']
    assert a.shape==b.shape==(10788,100) and np.isfinite(a).all()
    assert np.array_equal(a,b)
    assert old.shape==(1798,100) and si.shape==(1798,) and len(set(si))==1798
    assert all(idx[i]['seed']==20261013 and idx[i]['segment']=='whole' for i in si)
    old_error=float(np.max(np.abs(a[si]-old)));assert old_error<=2e-5
    assert np.array_equal(a[si].argmax(1),old.argmax(1))
    logit_values=int(a.size)
# Recount all real input steps without using preflight's summaries.
frames=missing=with_internal=variants=0
lookup={(x['clip_id'],x['view'],x['segment'],x['seed']):x for x in idx}
for row in s['rows']:
    assert row['role']=='train_candidate' and row['signer_id'] in range(1,10)
    rec=read(B/'data/records'/f"{row['clip_id']}.json")
    for view in s['views']:
        with np.load(rec['views'][view]['channel_file'],allow_pickle=False) as z:
            for segment in s['segments']:
                x=z['x_'+segment];ts=z[segment+'_ts'];fid=z[segment+'_frame_id'];n=len(x)
                assert x.shape==(n,280) and np.isfinite(x).all() and len(ts)==len(fid)==n
                assert np.all(np.diff(fid)>0) and np.all(np.diff(ts)>0)
                assert x[0,279]==0 and np.array_equal(x[1:,279],np.diff(ts).astype(np.float32)/1e6)
                masks=x[:,195:237];assert np.isin(masks,[0,1]).all()
                gaps=~np.any(masks>0,axis=1)
                frames+=n;missing+=int(gaps.sum());with_internal+=int(np.any(gaps[1:-1]));variants+=1
                for seed in s['seeds']:
                    q=lookup[row['clip_id'],view,segment,seed]
                    assert q['frames']==n and q['both_hand_missing_frames']==int(gaps.sum())
assert (frames,missing,with_internal,variants)==(r['unique_input_variant_real_frames'],r['both_hand_missing_frames_retained'],r['variants_with_internal_missing'],3596)
# Independently implement PyTorch GRU gate equations in NumPy float64.
# This deliberately does not call the candidate, nn.GRU, or packed sequence.
with np.load(U/'gru_gate_fixture.npz',allow_pickle=False) as z:
    x=z['features'];lengths=z['lengths'];w=z['weight_ih_l0'];v=z['weight_hh_l0'];b=z['bias_ih_l0'];c=z['bias_hh_l0'];expected_h=z['expected']
    def sigmoid(t):return 1/(1+np.exp(-t))
    def step(xx,hh):
        ix=xx@w.T+b;ih=hh@v.T+c
        reset=sigmoid(ix[:,:16]+ih[:,:16]);update=sigmoid(ix[:,16:32]+ih[:,16:32])
        new=np.tanh(ix[:,32:]+reset*ih[:,32:])
        return (1-update)*new+update*hh
    h=np.zeros((len(x),16));steps=np.zeros(len(x),dtype=int)
    for t in range(x.shape[1]):
        active=t<lengths;h[active]=step(x[active,t],h[active]);steps+=active
    gate_error=float(np.max(np.abs(h-expected_h)));assert gate_error<1e-12
    assert np.array_equal(steps,lengths) and np.array_equal(x[0,4],np.zeros(32))
    skipped=np.zeros((1,16))
    for t in range(int(lengths[0])):
        if t!=4:skipped=step(x[0:1,t],skipped)
    zero_step_delta=float(np.max(np.abs(skipped-h[:1])));assert zero_step_delta>1e-6
# Verify all saved original tensors against the explicit untrained snapshot.
snap=torch.load(U/'UNTRAINED_seed20261013.pt',map_location='cpu',weights_only=True)
base=torch.load(T/'T3F_20261013_best.pt',map_location='cpu',weights_only=True)
assert snap['status']=='UNTRAINED_PREFLIGHT_ONLY'
assert snap['base_checkpoint_sha256']==sha(T/'T3F_20261013_best.pt')
assert all(torch.equal(value,snap['state_dict']['base.'+k]) for k,value in base['state_dict'].items())
assert torch.count_nonzero(snap['state_dict']['residual.weight'])==0
new_parameters=sum(v.numel() for k,v in snap['state_dict'].items() if not k.startswith('base.'))
assert new_parameters==4000
assert all(c['pass'] for c in r['checks'])
assert r['training_runs']==r['optimizer_steps']==r['user_recordings_read']==r['new_DEV_or_holdout_evaluation']==0
out={'status':'PASS','method':'Same Codex, separate saved-arithmetic/input recount and NumPy GRU recurrence; not independent agent/expert',
 'result_sha256':sha(U/'RESULT.json'),'initial_equal_vectors':10788,'logit_values_compared':logit_values,
 'stored_original_vectors':1798,'stored_max_abs_error':old_error,'unique_variants':variants,'real_frames':frames,
 'both_hand_missing_frames_retained':missing,'variants_with_internal_missing':with_internal,
 'manual_GRU_max_abs_error':gate_error,'real_zero_step_omission_changes_state_by':zero_step_delta,
 'frozen_snapshot_tensors':len(base['state_dict']),'new_parameters':new_parameters,'protected_pins':len(s['pins']),
 'no_new_performance_claim':True,'training_runs':0,'user_recordings_read':0}
(U/'INDEPENDENT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(out,ensure_ascii=False))
