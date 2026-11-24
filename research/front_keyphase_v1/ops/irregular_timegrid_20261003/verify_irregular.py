"""Codex independent verification only. No camera, training or user inputs."""
from pathlib import Path
import json
import hashlib
import sys
import numpy as np

H=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
array_sha=lambda x:hashlib.sha256(f'{x.shape}|{x.dtype}|'.encode()+np.ascontiguousarray(x).tobytes()).hexdigest()
cfg=read(H/'CONFIG_LOCAL_ONLY.json')
sys.path[:0]=[str(H),cfg['helper_dir'],cfg['lab_root'],cfg['pilot_src']]
import corruptions
import timegrid
from verify_reference import independent_restore
from pilot.web_input import to_native_like
from pilot.data import build_channels
import tools_prep_union93_candidate as u
import torch

CASES=['isolated_drop','burst_drop','clock_noise']
ARMS=['original','corrupted','previous','linear']

def reference_corrupt(xy,pr,ts,sid,case):
    digests=[hashlib.sha256('|'.join(['WIS_TIMEGRID_IRREG_V1',sid,case,str(i)]).encode()).digest() for i in range(len(ts))]
    if case=='clock_noise':
        ids=np.arange(len(ts))
        times=ts.copy()
        for i in range(1,len(ts)-1):
            times[i]+=int.from_bytes(digests[i][:2],'big')%2001-1000
    else:
        ids=[0]
        while ids[-1]!=len(ts)-1:
            cur=ids[-1]
            step=(1+(digests[cur][0]&1)) if case=='isolated_drop' else (3 if digests[cur][0]%4==0 else 1)
            ids.append(min(cur+step,len(ts)-1))
        ids=np.asarray(ids,dtype=np.int64)
        times=ts[ids].copy()
    assert np.all(np.diff(times)>0) and times[0]==ts[0] and times[-1]==ts[-1]
    return xy[ids].copy(),pr[ids].copy(),times,ids

def load_raw(m,group,source_hashes):
    if group=='native':
        assert m['role']=='validation' and m['signer_id'] in [11,12,13]
        assert sha(m['path'])==m['sha256'];source_hashes[m['path']]=m['sha256']
        with np.load(m['path'],allow_pickle=False) as z:
            return z['xy_native'],z['coordinate_present'],z['timestamp_nominal_us']
    assert m['role']=='PREPARED_TRAINING_CANDIDATE_NOT_REGISTERED' and m['signer_id'] in [1,2]
    for k in ['input','observation']:
        assert sha(m[k+'_path'])==m[k+'_sha256'];source_hashes[m[k+'_path']]=m[k+'_sha256']
    with np.load(m['input_path'],allow_pickle=False) as z,np.load(m['observation_path'],allow_pickle=False) as o:
        pos={int(fid):i for i,fid in enumerate(o['frame_id'])}
        assert len(pos)==len(o['frame_id'])
        idx=[pos[int(fid)] for fid in z['frame_id']]
        assert np.array_equal(o['timestamp_us'][idx],z['timestamp_us'])
        xy,pr=to_native_like(o['xy_px'][idx],o['usable'][idx].astype(bool));ts=z['timestamp_us']
        assert np.allclose(build_channels(xy,pr,ts)['x'],z['x'],atol=1e-5,rtol=0)
        return xy,pr,ts

def main():
    for p,digest in cfg['pins'].items():assert sha(p)==digest,p
    torch.set_num_threads(2)
    model,classes=u.rebuild_from_payload(torch.load(cfg['checkpoint'],map_location='cpu',weights_only=True));model.eval()
    groups={'native':read(cfg['native_metadata'])['annotation_dev'],'mediapipe':read(cfg['mp_manifest'])['rows']}
    out={'status':'PASS','reviewer':'Codex','groups':{},'counts':{'corruptions':0,'restorations':0,'forwards':0},'max_logit_diff':0.0}
    source_hashes={}
    report=read(H/'result/RESULT.json')
    for group,rows in groups.items():
        ids=[m['id'] if group=='native' else m['sample_id'] for m in rows]
        truth=np.asarray([classes.index(m['label']) for m in rows])
        with np.load(H/'result'/f'SCORES_{group}.npz',allow_pickle=False) as z:
            assert z['ids'].tolist()==ids and z['classes'].tolist()==classes
            assert z['cases'].tolist()==CASES and z['arms'].tolist()==ARMS
            logits=z['logits'];assert logits.shape==(len(rows),3,4,93) and np.isfinite(logits).all()
            assert np.array_equal(z['true_indices'],truth)
        with np.load(cfg['saved_native_logits'] if group=='native' else cfg['saved_mp_logits'],allow_pickle=False) as old:
            original=old['original_logits'] if group=='native' else old['logits'][:,old['arms'].tolist().index('mediapipe')]
            for ci in range(3):assert np.array_equal(logits[:,ci,0],original)
        pred=logits.argmax(-1);correct=pred==truth[:,None,None]
        cases={}
        for ci,case in enumerate(CASES):
            totals=dict(zip(ARMS,correct[:,ci].sum(0).tolist()))
            per_signer={}
            for sid in sorted({m['signer_id'] for m in rows}):
                ix=[i for i,m in enumerate(rows) if m['signer_id']==sid]
                per_signer[str(sid)]=dict(n=len(ix),**dict(zip(ARMS,correct[ix,ci].sum(0).tolist())))
            comparisons={}
            for a,b in [(1,3),(1,2),(0,3)]:
                gain=~correct[:,ci,a]&correct[:,ci,b];loss=correct[:,ci,a]&~correct[:,ci,b]
                comparisons[ARMS[a]+'->'+ARMS[b]]={'gain':int(gain.sum()),'loss':int(loss.sum()),'changed':int((pred[:,ci,a]!=pred[:,ci,b]).sum()),'gain_ids':[ids[i] for i in np.flatnonzero(gain)],'loss_ids':[ids[i] for i in np.flatnonzero(loss)]}
            cases[case]={'totals':totals,'per_signer':per_signer,'comparisons':comparisons,'estimated_slots':0,'observed_slots':0,'long_bracket_grid_points':0,'unsupported_slots':0}
            rep=report['groups'][group]['cases'][case]
            assert rep['per_arm_correct']==totals and rep['per_signer']==per_signer
            for pair,v in comparisons.items():
                rc=rep['comparisons'][pair]
                assert (rc['gains_n'],rc['losses_n'],rc['top1_changed'])==(v['gain'],v['loss'],v['changed'])
                assert [d['id'] for d in rc['gains']]==v['gain_ids'] and [d['id'] for d in rc['losses']]==v['loss_ids']
            for label in sorted({m['label'] for m in rows}):
                ix=[i for i,m in enumerate(rows) if m['label']==label]
                assert rep['per_class'][label]==dict(n=len(ix),**dict(zip(ARMS,correct[ix,ci].sum(0).tolist())))
        for i,m in enumerate(rows):
            xy,pr,ts=load_raw(m,group,source_hashes)
            for ci,case in enumerate(CASES):
                cx,cp,ct,ix=reference_corrupt(xy,pr,ts,ids[i],case)
                actual=corruptions.corrupt(xy,pr,ts,ids[i],case)
                for key,value in [('xy',cx),('present',cp),('ts',ct),('indices',ix)]:assert np.array_equal(actual[key],value,equal_nan=True),(ids[i],case,key)
                out['counts']['corruptions']+=1
                inputs={'corrupted':build_channels(cx,cp,ct)['x']}
                saved_inputs=report['groups'][group]['cases'][case]['rows'][i]['case']['input_sha256']
                assert array_sha(inputs['corrupted'])==saved_inputs['corrupted']
                for method in ['previous','linear']:
                    ref=independent_restore(cx,cp,ct,method);act=timegrid.restore(cx,cp,ct,method)
                    for key in ['present','ts','observed','estimated','exact_source_index']:assert np.array_equal(ref[key],act[key]),(ids[i],case,method,key)
                    assert np.allclose(ref['xy'],act['xy'],atol=2e-5,rtol=0,equal_nan=True)
                    assert not (act['observed']&act['estimated']).any()
                    assert np.array_equal(act['present'].astype(bool),act['observed']|act['estimated'])
                    for j,t in enumerate(ref['ts']):
                        if ref['exact_source_index'][j]>=0:continue
                        right=int(np.searchsorted(ct,t));left=right-1
                        if left>=0 and right<len(ct) and ct[right]-ct[left]>2e6/30+2:
                            assert not act['present'][j].any() and not act['estimated'][j].any()
                            if method=='linear':cases[case]['long_bracket_grid_points']+=1
                    inputs[method]=build_channels(ref['xy'],ref['present'],ref['ts'])['x']
                    actual_x=build_channels(act['xy'],act['present'],act['ts'])['x']
                    assert np.allclose(actual_x,inputs[method],atol=1e-5,rtol=0)
                    assert array_sha(actual_x)==saved_inputs[method]
                    out['counts']['restorations']+=1
                    if method=='linear':
                        cases[case]['estimated_slots']+=int(ref['estimated'].sum());cases[case]['observed_slots']+=int(ref['observed'].sum())
                        cases[case]['unsupported_slots']+=int((~ref['present']).sum())
                if i in cfg['repro_indices'][group]:
                    for arm,x in inputs.items():
                        with torch.inference_mode():p=model(torch.from_numpy(x[None]),torch.ones((1,len(x)),dtype=torch.float32))[0].numpy()
                        delta=float(np.max(np.abs(p-logits[i,ci,ARMS.index(arm)])))
                        assert delta<=1e-5,(group,ids[i],case,arm,delta)
                        out['max_logit_diff']=max(out['max_logit_diff'],delta);out['counts']['forwards']+=1
        out['groups'][group]=cases
    checks={}
    for group,cases in out['groups'].items():
        for case,d in cases.items():
            c=d['comparisons']['corrupted->linear']
            checks[group+'/'+case]={'net':c['gain']-c['loss'],'newly_wrong':c['loss'],'per_signer_nondecrease':all(v['linear']>=v['corrupted'] for v in d['per_signer'].values())}
    out['admission_checks']=checks
    out['web_admission_pass']=checks['native/isolated_drop']['net']>=7 and all(v['net']>=0 and v['newly_wrong']==0 and v['per_signer_nondecrease'] for v in checks.values())
    assert out['web_admission_pass']==report['admission']['all_conditions_hold']
    for p,digest in source_hashes.items():assert sha(p)==digest
    for p,digest in cfg['pins'].items():assert sha(p)==digest
    out['source_files_hashed']=len(source_hashes)
    with (H/'INDEPENDENT_VERIFICATION.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps({'status':out['status'],'counts':out['counts'],'max_logit_diff':out['max_logit_diff'],'web_admission_pass':out['web_admission_pass'],'checks':checks},ensure_ascii=False))

if __name__=='__main__':main()
