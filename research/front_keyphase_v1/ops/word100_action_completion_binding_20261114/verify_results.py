"""Independent code path for saved arithmetic/protection checks, no inference."""
from pathlib import Path
import json,hashlib,collections
import numpy as np
import torch
U=Path(__file__).resolve().parent;O=U.parent;G=O/'word100_web_completion_graph_20261111';N=G/'typed_motion_recovery_20261113/null_candidate_learning'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cfg=read(U/'TRAINING_FROZEN.json');train=read(U/'TRAINING_COMPLETE.json');meta=read(U/'ABSENCE_BAGS.json');local=read(U/'LOCAL_COMPARISON.json');full=read(U/'FULLPATH_COMPARISON.json');channel=read(U/'COMPLETION_CHANNEL_VERIFIED.json')
assert channel['passed'] and channel['source_sha256']==sha(U/'FULLPATH_COMPARISON.json')
assert train['training_frozen_sha256']==sha(U/'TRAINING_FROZEN.json');assert full['training_complete_sha256']==sha(U/'TRAINING_COMPLETE.json')
checks=0
for p,h in {**cfg['pins'],**cfg['protected']}.items():assert sha(Path(p))==h;checks+=1
for r in train['rows']:
    assert r['steps']==1200 and r['log'][-1]['step']==1200
    for p,h in [(r['model'],r['model_sha256']),(r['predictions'],r['predictions_sha256']),(r['initial_model'],r['initial_sha256'])]:assert sha(Path(p))==h;checks+=1
    old=torch.load(r['initial_model'],map_location='cpu',weights_only=True);new=torch.load(r['model'],map_location='cpu',weights_only=True)
    assert torch.equal(old['rank.mean'],new['rank.mean']) and torch.equal(old['rank.std'],new['rank.std'])
    assert all(meta['rows'][int(i)]['person']in range(2,8) for i in r['bag_draw_counts'])
    assert {meta['rows'][int(i)]['person']for i in r['bag_draw_counts']}==set(range(2,8));checks+=3
arrays=np.load(meta['arrays']);locmap={(r['seed'],r['index']):r for r in local['rows']}
for r in train['rows']:
    p=np.load(r['predictions']);parent=next(a for a in cfg['parents']if a['seed']==r['seed']);bp=np.load(parent['predictions'])
    for b in meta['rows']:
        j=[2,3,4,5,7].index(b['family']);ix=b['candidate_indices'];y=arrays[f"y_{b['index']}"];record=locmap[r['seed'],b['index']]
        for key,values in [('parent',bp),('new',p)]:
            scores=values[f"logits_{b['original_bag_index']}"][ix,j];idx=int(np.argmax(scores));accepted=float(scores[idx])>float(values['null_logits'][j]);q=record[key]
            assert q['null_selected']==(not accepted)
            assert q['chosen_original_candidate']==(ix[idx]if accepted else None)
            assert q['full_source_correct']==(bool(accepted and y[idx])if y.any()else None)
            checks+=3
for r in local['summary']:
    rr=[a for a in local['rows']if(a['seed'],a['split'],a['kind'])==(r['seed'],r['split'],r['kind'])];metric='full_source_correct'if r['kind']=='positive_source_extent'else'no_full_extent_correct'
    assert r['bags']==len(rr);assert r['parent_correct']==sum(a['parent'][metric]is True for a in rr);assert r['new_correct']==sum(a['new'][metric]is True for a in rr);checks+=3
# Full path reuses the original body admission gate and candidate arrays.
parent=read(N/'evaluation/COMPARISON.json');pm={(r['seed'],r['clip_id'],r['kind']):r for r in parent['predictions']}
fm={(r['clip_id'],r['kind']):r for r in full['predictions']};losses=[];error_disappear=[]
for r in full['rows']:
    m=fm[r['clip_id'],r['kind']];assert sha(Path(m['path']))==m['sha256'];f=np.load(m['feature_path']);pred=np.load(m['path']);lu={tuple(s):i for i,s in enumerate(f['spans'])}
    pr=pm[r['seed'],r['clip_id'],r['kind']];pa=np.load(pr['path']);score=pred[f"logits_{r['seed']}"];null=pred[f"null_{r['seed']}"]
    assert score.shape==(len(f['spans']),5) and np.isfinite(score).all()
    for i,c in enumerate(r['conditions']['typed_extent']['cards']):
        orig=next(v for v in r['conditions']['fixed_rank']['cards']if v['card']==c['card'])
        if c['family']==2:assert c==orig;continue
        j=[2,3,4,5,7].index(c['family']);sp=pa[f'spans_{i}'];ids=[lu[tuple(s)]for s in sp];vv=np.where(pa[f'presence_{i}']>=0,score[ids,j],-np.inf)
        accept=len(vv)>0 and np.isfinite(vv).any() and vv.max()>null[j]
        assert (c['chosen']is not None)==accept
        if accept:assert c['chosen']['span']==sp[int(vv.argmax())].tolist()
        if orig['chosen']is not None and c['chosen']is None:
            item=dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],card=c['card'])
            if r['kind']in ['full_normal','speed075','speed125']:losses.append(item)
            if r['kind']in ['right_closed_hand_main','main_reversed','closed_hand_and_reversed']:error_disappear.append(item)
        checks+=2
assert len(full['rows'])==246 and len(fm)==82
# Exact source-omission diagnostics: compare the same target action, not all
# cards/no-truth cards lumped together.
omission=read(U/'OMISSION_DIAGNOSTIC.json');om=[]
for r in omission['rows']:
    for c in r['cards']:
        if not c['original_witnesses'] or c['retained_source_witnesses']:continue
        q=next(a for a in full['rows']if(a['seed'],a['clip_id'],a['kind'])==(r['seed'],r['clip_id'],r['kind']))
        newer=next(a for a in q['conditions']['typed_extent']['cards']if a['card']==c['card'])
        om.append(dict(seed=r['seed'],clip_id=r['clip_id'],card=c['card'],old_selected=c['span']is not None,new_selected=newer['chosen']is not None))
assert len(om)==9
out=dict(passed=True,checks=checks,training_runs=3,normal_speed_selected_action_losses=losses,
    wrong_shape_or_direction_actions_removed=error_disappear,omitted_target_cases=om,
    model_adoption=False,reason='Local full-source protected losses plus normal/speed selection losses and error-action exclusions.',
    G2='REWORK',web_changed=False,common100_changed=False,private_used=False,final8_opened=False,
    actor='root Codex separate saved arithmetic and source verification; not independent KSL expert',
    pins={str(p):sha(p)for p in [U/'TRAINING_COMPLETE.json',U/'LOCAL_COMPARISON.json',U/'FULLPATH_COMPARISON.json',U/'COMPLETION_CHANNEL_VERIFIED.json']})
with (U/'RESULTS_VERIFIED.json').open('x',encoding='utf-8')as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps({k:out[k]for k in ['passed','checks','training_runs','model_adoption']},ensure_ascii=False))
print('normal/speed losses',len(losses),'error exclusions',len(error_disappear),'omitted target selected',sum(x['old_selected']for x in om),'->',sum(x['new_selected']for x in om))
