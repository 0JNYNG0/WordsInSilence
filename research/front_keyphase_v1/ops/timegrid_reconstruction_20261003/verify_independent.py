"""Codex verification, independent reference/input rebuild and score recount. Local public data only."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from verify_reference import independent_restore

H = Path(__file__).resolve().parent
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg = read(H/'CONFIG_LOCAL_ONLY.json')
sys.path[:0] = [cfg['lab_root'], cfg['pilot_src'], str(H)]
from pilot.data import build_channels
from pilot.web_input import to_native_like
import timegrid
import tools_prep_union93_candidate as u
import torch

for p,digest in cfg['pins'].items():
    assert sha(p) == digest, p
model, classes = u.rebuild_from_payload(torch.load(cfg['checkpoint'],map_location='cpu',weights_only=True))
model.eval()
torch.set_num_threads(2)
meta_groups = {'native':read(cfg['native_metadata'])['annotation_dev'], 'mediapipe':read(cfg['mp_manifest'])['rows']}
audit = {'status':'PASS','reviewer':'Codex','groups':{},'new_forward_count':0,'max_probe_logit_diff':0.0,'camera':0,'training':0,'user_data':0}
source_hashes={}
reported=read(H/'result'/'RESULT.json')
for group, meta in meta_groups.items():
    z=np.load(H/'result'/f'SCORES_{group}.npz',allow_pickle=False)
    assert z['classes'].tolist()==classes
    arms=z['arms'].tolist()
    assert arms==['original','sparse','endpoint_reference','previous','linear']
    ids=[m['id'] if group=='native' else m['sample_id'] for m in meta]
    assert z['ids'].tolist()==ids
    score=z['logits']; truth=np.array([classes.index(m['label']) for m in meta])
    assert score.shape==(len(meta),5,93) and np.isfinite(score).all()
    assert np.array_equal(z['true_indices'],truth)
    pred=score.argmax(2); correct=pred==truth[:,None]
    totals=dict(zip(arms,correct.sum(0).tolist()))
    comparisons={}
    for a,b in [(1,4),(1,3),(3,4),(0,2),(2,4)]:
        gain=~correct[:,a]&correct[:,b];loss=correct[:,a]&~correct[:,b]
        comparisons[arms[a]+'->'+arms[b]]={'gains':int(gain.sum()),'losses':int(loss.sum()),'changed':int((pred[:,a]!=pred[:,b]).sum()),'gain_ids':[ids[i] for i in np.where(gain)[0]],'loss_ids':[ids[i] for i in np.where(loss)[0]]}
    per_signer={}
    for signer in sorted({m['signer_id'] for m in meta}):
        ix=[i for i,m in enumerate(meta) if m['signer_id']==signer]
        per_signer[str(signer)]=dict(n=len(ix),**dict(zip(arms,correct[ix].sum(0).tolist())))
    rep=reported['groups'][group]
    assert rep['per_arm_correct']==totals and rep['per_signer']==per_signer
    for name,v in comparisons.items():
        c=rep['comparisons'][name]
        assert (c['gains_n'],c['losses_n'],c['top1_changed'])==(v['gains'],v['losses'],v['changed'])
        assert [r['id'] for r in c['gains']]==v['gain_ids'] and [r['id'] for r in c['losses']]==v['loss_ids']
    for label in sorted({m['label'] for m in meta}):
        ix=[i for i,m in enumerate(meta) if m['label']==label]
        assert rep['per_class'][label]==dict(n=len(ix),**dict(zip(arms,correct[ix].sum(0).tolist())))
    if group=='native':
        old=np.load(cfg['saved_native_logits'],allow_pickle=False)
        assert np.array_equal(score[:,0],old['original_logits']) and np.array_equal(score[:,1],old['decimated_logits'])
    else:
        old=np.load(cfg['saved_mp_logits'],allow_pickle=False)
        assert np.array_equal(score[:,0],old['logits'][:,old['arms'].tolist().index('mediapipe')])
    counts={'clean_exact':0,'restores_verified':0,'estimated_joint_slots':0,'observed_joint_slots':0,'source_end_lost_rows':0}
    probes=[]
    for i,m in enumerate(meta):
        if group=='native':
            assert m['role']=='validation' and m['signer_id'] in (11,12,13)
            assert sha(m['path'])==m['sha256'];source_hashes[m['path']]=m['sha256']
            with np.load(m['path'],allow_pickle=False) as raw:
                xy,pr,ts=raw['xy_native'],raw['coordinate_present'],raw['timestamp_nominal_us']
        else:
            assert m['role']=='PREPARED_TRAINING_CANDIDATE_NOT_REGISTERED' and m['signer_id'] in (1,2)
            for pk,hk in [('input_path','input_sha256'),('observation_path','observation_sha256')]:
                assert sha(m[pk])==m[hk];source_hashes[m[pk]]=m[hk]
            with np.load(m['input_path'],allow_pickle=False) as prep,np.load(m['observation_path'],allow_pickle=False) as obs:
                table={int(fid):j for j,fid in enumerate(obs['frame_id'])};assert len(table)==len(obs['frame_id'])
                ix=[table[int(fid)] for fid in prep['frame_id']]
                assert np.array_equal(obs['timestamp_us'][ix],prep['timestamp_us'])
                xy,pr=to_native_like(obs['xy_px'][ix],obs['usable'][ix].astype(bool));ts=prep['timestamp_us']
                assert np.allclose(build_channels(xy,pr,ts)['x'],prep['x'],atol=1e-5,rtol=0)
        base=build_channels(xy,pr,ts)['x']
        last=2*((len(ts)-1)//2)
        crop=build_channels(xy[:last+1],pr[:last+1],ts[:last+1])['x']
        counts['source_end_lost_rows']+=last!=len(ts)-1
        for method in ['previous','linear']:
            clean=timegrid.restore(xy,pr,ts,method=method)
            for k,expected in [('xy',xy),('present',pr),('ts',ts)]:
                assert np.array_equal(clean[k],expected,equal_nan=True),(group,ids[i],method,k,'clean')
            assert clean['identity'] and not clean['estimated'].any()
            assert np.array_equal(build_channels(clean['xy'],clean['present'],clean['ts'])['x'],base)
            actual=timegrid.restore(xy[::2],pr[::2],ts[::2],method=method)
            ref=independent_restore(xy[::2],pr[::2],ts[::2],method)
            for k in ['present','ts','observed','estimated','exact_source_index']:
                assert np.array_equal(actual[k],ref[k]),(group,ids[i],method,k)
            assert np.allclose(actual['xy'],ref['xy'],atol=2e-5,rtol=0,equal_nan=True),(group,ids[i],method,'coords')
            assert not (actual['observed']&actual['estimated']).any()
            assert np.array_equal(actual['present'],actual['observed']|actual['estimated'])
            assert actual['ts'][0]==ts[0] and actual['ts'][-1]==ts[last]
            counts['restores_verified']+=1
            if method=='linear':
                counts['estimated_joint_slots']+=int(actual['estimated'].sum())
                counts['observed_joint_slots']+=int(actual['observed'].sum())
            if i in cfg['repro_indices'][group]:
                inp=build_channels(ref['xy'],ref['present'],ref['ts'])['x']
                with torch.inference_mode():
                    lg=model(torch.from_numpy(inp[None]),torch.ones((1,len(inp)),dtype=torch.float32))[0].numpy()
                delta=float(np.max(np.abs(lg-score[i,arms.index(method)])))
                assert delta<=1e-5,(group,ids[i],method,delta)
                probes.append({'index':i,'arm':method,'max_abs_logit_diff':delta})
                audit['new_forward_count']+=1;audit['max_probe_logit_diff']=max(audit['max_probe_logit_diff'],delta)
        counts['clean_exact']+=1
        if i in cfg['repro_indices'][group]:
            with torch.inference_mode():
                lg=model(torch.from_numpy(crop[None]),torch.ones((1,len(crop)),dtype=torch.float32))[0].numpy()
            delta=float(np.max(np.abs(lg-score[i,2])))
            assert delta<=1e-5
            probes.append({'index':i,'arm':'endpoint_reference','max_abs_logit_diff':delta})
            audit['new_forward_count']+=1;audit['max_probe_logit_diff']=max(audit['max_probe_logit_diff'],delta)
    subset=correct[:,0]&~correct[:,1]
    assert rep['outcome_checks']['clean_identity_rows']==counts['clean_exact']
    assert rep['original_correct_sparse_wrong_recoveries']['n']==int(subset.sum())
    assert rep['original_correct_sparse_wrong_recoveries']['correct_in_linear']==int((subset&correct[:,4]).sum())
    assert rep['durations']['rows_final_frame_dropped_by_slice']==counts['source_end_lost_rows']
    net=totals['linear']-totals['sparse']
    signer_ok=all(v['linear']>=v['sparse'] for v in per_signer.values())
    expected_pass=net>=(7 if group=='native' else 0) and signer_ok and counts['clean_exact']==len(meta)
    key='primary_pass' if group=='native' else 'corroboration_conditions_met'
    assert rep['outcome_checks'][key]==expected_pass
    audit['groups'][group]={'rows':len(meta),'totals':totals,'per_signer':per_signer,'comparisons':comparisons,'inputs':counts,'probes':probes,'original_correct_sparse_wrong':int(subset.sum()),'linear_recovers_those':int((subset&correct[:,4]).sum()),'previous_recovers_those':int((subset&correct[:,3]).sum()),'new_errors_vs_full_original':int((correct[:,0]&~correct[:,4]).sum())}
for p,digest in source_hashes.items():assert sha(p)==digest
for p,digest in cfg['pins'].items():assert sha(p)==digest
audit['source_hashes_checked']=len(source_hashes)
with (H/'INDEPENDENT_VERIFICATION.json').open('x',encoding='utf-8') as f:json.dump(audit,f,ensure_ascii=False,indent=2)
print(json.dumps({'status':audit['status'],'groups':{k:{'totals':v['totals'],'inputs':v['inputs'],'recovers':v['linear_recovers_those']} for k,v in audit['groups'].items()},'max_probe_logit_diff':audit['max_probe_logit_diff'],'new_forward_count':audit['new_forward_count']},ensure_ascii=False))
