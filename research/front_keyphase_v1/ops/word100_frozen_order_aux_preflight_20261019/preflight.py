"""Only initialization/contract verification. No optimizer or training steps."""
from pathlib import Path
import json, hashlib, sys, time, traceback
from datetime import datetime, timezone, timedelta
import numpy as np
import torch
from frozen_order_aux import FrozenOrderAux, FrozenCTemporal

U=Path(__file__).resolve().parent; O=U.parent; ROOT=U.parents[3]
B=O/'word100_body_boundary_20261017'; T=O/'word100_frozen_c_temporal_20261019'; A=O/'word100_sequence_input_audit_20261019'
SEEDS=(20261013,20261014,20261015)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);assert not p.exists(),p
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def statehash(m):
    h=hashlib.sha256()
    for k,v in sorted(m.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()
def now():return datetime.now(timezone(timedelta(hours=9))).isoformat()

def main():
    assert not (U/'SCOPE.json').exists(), 'Completed/failed evidence must not be overwritten'
    scope0=read(A/'SCOPE.json');done=read(A/'COMPLETE.json');rows=scope0['rows']
    assert len(rows)==899 and len({r['word_code'] for r in rows})==100
    assert {r['signer_id'] for r in rows}==set(range(1,10))
    assert all(r['role']=='train_candidate' for r in rows)
    pins={}
    files=[A/'SCOPE.json',A/'COMPLETE.json',A/'NEXT_PLAN.json',A/'INDEPENDENT.json',B/'training_CONTEXT.json',T/'TRAIN_CONTEXT.json',
           T/'temporal.py',T/'frozen_temporal.py',ROOT/'research/sign_model_pilot_v1/src/pilot/model_masked_v1.py',
           ROOT/'research/front_keyphase_v1/src/wis_lab/service/word100_candidate.py',ROOT/'research/front_keyphase_v1/src/wis_lab/service/word100_t3f.py',
           B/'training_C_20261013_best.pt',U/'frozen_order_aux.py',Path(__file__)]
    files += [T/f'T3F_{s}_best.pt' for s in SEEDS]
    for p in files:
        pins[str(p)]=sha(p)
        if str(p) in scope0['pins']:assert pins[str(p)]==scope0['pins'][str(p)]
    for r in rows:
        p=B/'data/records'/f"{r['clip_id']}.json";pins[str(p)]=sha(p)
        assert pins[str(p)]==scope0['pins'][str(p)]
        rec=read(p)
        for view in ('masked','sensor'):
            p=Path(rec['views'][view]['channel_file']);pins[str(p)]=sha(p)
            assert pins[str(p)]==r['views'][view]['sha256']
        p=A/'records'/f"{r['clip_id']}.json";pins[str(p)]=sha(p)
        assert pins[str(p)]==done['record_hashes'][r['clip_id']]
    write(U/'SCOPE.json',{'at':now(),'status':'FROZEN_IMPLEMENTATION_PREFLIGHT_ONLY','user_request':'작은 순서 기억 보조 후보 구현·비교 검증, 추가학습은 결과 함께 확인 후',
      'training_authorized':False,'training_runs':0,'optimizer_steps':0,'user_recordings_read':0,
      'rows':rows,'seeds':SEEDS,'views':['masked','sensor'],'segments':['whole','context'],
      'candidate':'All T3-F 47236 frozen + unidirectional GRU16 2400 + zero bias-free head1600',
      'gates':['10788 initial outputs equal reference T3F','1798 seed13 whole outputs match stored audit','frozen weights and BN','padding exclusion; all real missing steps and dt retained','gradient only auxiliary, no optimizer','independent GRU gate calculation'],
      'tolerance':{'direct_reference_max_abs':0.0,'stored_reference_max_abs':2e-5,'padding_batch_max_abs':2e-5},
      'forbidden':['training/optimizer steps','DEV/holdout inference','user/mentor data access','web or default replacement','input detector/normalization changes','Git','downloads/external transmission'],
      'implementation_and_verification':'Codex local, separate verification code; no independent agent or signing expert', 'pins':pins})
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    models={};refs={};hashes={};checks=[]
    for s in SEEDS:
        ck=torch.load(T/f'T3F_{s}_best.pt',map_location='cpu',weights_only=True)
        torch.manual_seed(s);m=FrozenOrderAux(ck['model_config'],ck['state_dict'])
        ref=FrozenCTemporal(**ck['model_config']);ref.load_state_dict(ck['state_dict']);ref.eval()
        assert sum(p.numel() for p in m.base.parameters())==47236
        assert sum(p.numel() for p in m.parameters() if p.requires_grad)==4000
        assert sum(p.numel() for p in m.parameters())==51236
        assert torch.count_nonzero(m.residual.weight)==0
        m.train();m.assert_frozen();models[s]=m;refs[s]=ref;hashes[s]=statehash(m)
    comparisons=[];newlog=[];reflog=[];storedlog=[];stored_indices=[];real_frames=0;nohand_frames=0;internal_missing_sequences=0
    started=time.monotonic();sample=None
    for j,row in enumerate(rows):
        cid=row['clip_id'];rec=read(B/'data/records'/f'{cid}.json');prior=read(A/'records'/f'{cid}.json')
        for view in ('masked','sensor'):
            with np.load(rec['views'][view]['channel_file'],allow_pickle=False) as z:
                for segment in ('whole','context'):
                    a=z['x_'+segment];fid=z[segment+'_frame_id'];ts=z[segment+'_ts']
                    assert np.isfinite(a).all() and len(a)==len(fid)==len(ts)
                    assert np.all(np.diff(fid)>0) and np.all(np.diff(ts)>0)
                    assert a[0,279]==0 and np.array_equal(a[1:,279],np.diff(ts).astype(np.float32)/1e6)
                    x=torch.from_numpy(a.copy()).unsqueeze(0);pad=torch.ones(1,len(a));before=x.clone()
                    missing=(a[:,195:237].sum(1)==0);real_frames+=len(a);nohand_frames+=int(missing.sum())
                    internal_missing_sequences+=int(missing[1:-1].any())
                    if sample is None:sample=x.clone()
                    for s in SEEDS:
                        m=models[s]
                        with torch.no_grad():
                            base,delta,mem,features,lengths=m.components(x,pad);actual=base+delta;expected=refs[s](x,pad)
                        assert int(lengths[0])==len(a) and features.shape[1]==len(a)
                        assert torch.count_nonzero(delta)==0 and torch.equal(actual,expected)
                        assert torch.equal(x,before)
                        idx=len(newlog);newlog.append(actual[0].numpy());reflog.append(expected[0].numpy())
                        comparisons.append({'clip_id':cid,'view':view,'segment':segment,'seed':s,'frames':len(a),'both_hand_missing_frames':int(missing.sum())})
                        if s==SEEDS[0] and segment=='whole':
                            p=next(p for p in prior['predictions'] if p['view']==view and p['condition']=='original')
                            old=np.asarray(p['T3F']['logits'],dtype=np.float32)
                            assert np.max(np.abs(actual[0].numpy()-old))<=2e-5
                            assert np.argmax(actual[0].numpy())==np.argmax(old)
                            storedlog.append(old);stored_indices.append(idx)
        if (j+1)%100==0:print(json.dumps({'videos':j+1,'total':899,'output_comparisons':len(newlog),'seconds':round(time.monotonic()-started,1)}),flush=True)
    np.savez_compressed(U/'initial_outputs.npz',candidate=np.stack(newlog),reference=np.stack(reflog),stored=np.stack(storedlog),stored_indices=np.array(stored_indices))
    write(U/'COMPARISON_INDEX.json',comparisons)
    # Contract tests use synthetic tensors or already-read TRAIN input; no learning steps.
    m=models[SEEDS[0]];checks.append({'name':'all_public_initial_equivalence','pass':True,'outputs':len(newlog)})
    torch.manual_seed(1729)
    lengths=torch.tensor([11,5,1]);features=torch.randn(3,19,32);features[1,5:]=30;features[2,1:]=-40
    packed=m.remember(features,lengths)
    single=torch.cat([m.remember(features[i:i+1,:n],torch.tensor([n])) for i,n in enumerate(lengths.tolist())])
    assert torch.allclose(packed,single,atol=1e-6,rtol=0)
    reordered=m.remember(features[[2,0,1]],lengths[[2,0,1]])
    assert torch.allclose(reordered,packed[[2,0,1]],atol=1e-6,rtol=0)
    reverse=m.remember(features[:1,:11].flip(1),torch.tensor([11]))
    order_delta=float((reverse-packed[:1]).abs().max().detach());assert order_delta>1e-5
    assert torch.equal(m.remember(features,lengths),packed)
    checks.append({'name':'real_last_state_padding_batch_order_reset','pass':True,'max_error':float((packed-single).abs().max().detach()),'synthetic_order_hidden_delta':order_delta,'semantic_accuracy_claim':False})
    # Save independent recurrence fixture, including an internal all-zero step. It is real.
    f=torch.randn(2,9,32,dtype=torch.float64);f[0,4]=0;ls=torch.tensor([9,6]);f[1,6:]=123
    import copy
    gm=copy.deepcopy(m.memory).double()
    from torch.nn.utils.rnn import pack_padded_sequence
    _,hh=gm(pack_padded_sequence(f,ls,batch_first=True,enforce_sorted=False))
    fixture={k:v.detach().numpy() for k,v in gm.state_dict().items()}
    fixture.update(features=f.numpy(),lengths=ls.numpy(),expected=hh[-1].detach().numpy())
    np.savez(U/'gru_gate_fixture.npz',**fixture)
    # Nonzero residual in a throwaway diagnostic copy exercises a trained-like path;
    # no parameter is fitted, and this copy is never saved/deployed.
    probe=copy.deepcopy(m);probe.residual.weight.data.copy_(torch.linspace(-.1,.1,1600).reshape(100,16))
    probe_hash=statehash(probe);n=sample.shape[1]
    xpad=torch.cat([sample,torch.full((1,17,280),23.)],1);pm=torch.cat([torch.ones(1,n),torch.zeros(1,17)],1)
    with torch.no_grad():a1=probe(sample,torch.ones(1,n));a2=probe(xpad,pm)
    err=float((a1-a2).abs().max());assert err<=2e-5
    checks.append({'name':'nonzero_head_full_model_padding','pass':True,'max_error':err})
    # Mark a real middle frame unobserved, without deleting its timestamp.
    gap=sample.clone();gap[:,n//2,:279]=0;original_dt=gap[:,:,279].clone()
    with torch.no_grad():_,_,_,fh,ll=probe.components(gap,torch.ones(1,n))
    assert fh.shape[1]==n and ll[0]==n and torch.equal(gap[:,:,279],original_dt)
    checks.append({'name':'real_missing_frame_and_clock_retained','pass':True,'frames':n})
    # Backward only: zero head initially gives zero GRU gradient, nonzero head opens it.
    m.zero_grad(set_to_none=True);m(sample,torch.ones(1,n)).square().mean().backward()
    assert all(p.grad is None for p in m.base.parameters())
    assert m.residual.weight.grad is not None and m.residual.weight.grad.abs().sum()>0
    assert all(p.grad is not None and p.grad.abs().sum()==0 for p in m.memory.parameters())
    probe.zero_grad(set_to_none=True);probe(sample,torch.ones(1,n)).square().mean().backward()
    assert all(p.grad is None for p in probe.base.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in probe.memory.parameters())
    assert statehash(probe)==probe_hash
    checks.append({'name':'gradients_aux_only_no_updates','pass':True,'zero_head_first_gru_gradient_zero_expected':True,'nonzero_probe_gru_gradient':True,'optimizer_steps':0})
    for kind in ('empty','nonprefix','fractional','nonfinite','clock'):
        xx=sample.clone();pp=torch.ones(1,n)
        if kind=='empty':pp.zero_()
        elif kind=='nonprefix':pp[:,n//2]=0
        elif kind=='fractional':pp[:,0]=.5
        elif kind=='nonfinite':xx[:,0,0]=float('nan')
        elif kind=='clock':xx[:,1,279]=0
        try:m(xx,pp)
        except ValueError:pass
        else:raise AssertionError('Malformed input accepted: '+kind)
    checks.append({'name':'malformed_inputs_fail_closed','pass':True,'cases':5})
    for s,model in models.items():
        model.assert_frozen();assert statehash(model)==hashes[s]
        model.zero_grad(set_to_none=True)
    # Explicit untrained snapshot only; not placed in any serving/model registry.
    p=U/'UNTRAINED_seed20261013.pt';torch.save({'status':'UNTRAINED_PREFLIGHT_ONLY','model_config':ck['model_config'],'seed':SEEDS[0],'state_dict':m.state_dict(),'base_checkpoint_sha256':sha(T/f'T3F_{SEEDS[0]}_best.pt')},p)
    snap=torch.load(p,map_location='cpu',weights_only=True)
    restored=FrozenOrderAux(snap['model_config'],refs[SEEDS[0]].state_dict());restored.load_state_dict(snap['state_dict'])
    with torch.no_grad():assert torch.equal(restored(sample,torch.ones(1,n)),m(sample,torch.ones(1,n)))
    checks.append({'name':'untrained_save_reload_exact','pass':True})
    for p,h in pins.items():assert sha(p)==h,p
    write(U/'RESULT.json',{'status':'PASS_PREFLIGHT_ONLY_NOT_TRAINED','at':now(),'scope_sha256':sha(U/'SCOPE.json'),
      'public_videos':899,'words':100,'signers':9,'input_variants':3596,'initial_output_comparisons':len(newlog),'stored_whole_comparisons':len(storedlog),
      'unique_input_variant_real_frames':real_frames,'both_hand_missing_frames_retained':nohand_frames,'variants_with_internal_missing':internal_missing_sequences,
      'parameters':{'frozen':47236,'GRU':2400,'residual_head':1600,'new_trainable':4000,'total':51236},
      'checks':checks,'all_weights_and_buffers_unchanged':True,'initial_state_hashes':hashes,
      'protected_pins':len(pins),'training_runs':0,'optimizer_steps':0,'user_recordings_read':0,'new_DEV_or_holdout_evaluation':0,
      'seconds':round(time.monotonic()-started,2),'files':{p.name:sha(p) for p in [U/'initial_outputs.npz',U/'COMPARISON_INDEX.json',U/'gru_gate_fixture.npz',U/'UNTRAINED_seed20261013.pt']}})
    print('PASS: initialization and contracts; no training or optimizer steps',flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        path=U/'FIRST_ERROR.txt'
        if not path.exists():path.write_text(traceback.format_exc(),encoding='utf8')
        raise
