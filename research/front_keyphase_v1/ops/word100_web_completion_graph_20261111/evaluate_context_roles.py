"""New checkpoints only; reuse existing observations and order evaluations."""
from graph import *
from prepare_linker import load
from interval_pipeline import select_cards,order_observations
from collections import Counter
import sys,numpy as np,torch
O=U.parent;R=O/'word100_candidate_role_learning_20261111';G=O/'word100_guide_requirement_learning_20261111';D=O/'word100_action_feedback_demo_20261108';W=U/'context_role_training'


def main():
    assert (W/'TRAINING_COMPLETE.json').exists() and not (W/'EVALUATION_COMPLETE.json').exists()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    rc=load('_ctx_eval_requirements',G/'requirement_core.py');sys.modules['requirement_core']=rc
    role=load('_ctx_eval_role',R/'role_core.py');om=load('_ctx_eval_order',G/'order_binding.py')
    cfg=read(W/'FROZEN.json');phasecfg=read(D/'FROZEN.json')['phase']
    normal=read(R/'EVALUATION_INPUTS.json')['rows'];dynamic=read(U/'dynamic_blocks/INPUTS_COMPLETE.json')['rows']
    sources=[dict(s,kind='dynamic_complete_blocks') for s in dynamic]+normal
    assert len(sources)==82 and len(set((s['clip_id'],s['kind']) for s in sources))==82
    labels={r['clip_id']:r for r in read(O/'word100_event_guide_linkage_20261110/LINK_LABELS.json')['rows']}
    write(W/'EVALUATION_FROZEN.json',dict(conditions=cfg['conditions'],seeds=cfg['seeds'],primary_seed=42,
      source_pins={str(p):role.sha(p) for p in [R/'EVALUATION_INPUTS.json',U/'dynamic_blocks/INPUTS_COMPLETE.json',W/'TRAINING_COMPLETE.json',Path(__file__),U/'interval_pipeline.py',U/'occurrence_binding.py']},
      scope='Previously exposed DEV78 and source-audited synthetic4; no fresh final or whole-sign educational accuracy',
      metric_taxonomy='Fold predicted entry/return/transition 8..10 to nonmain0 for legacy8 anchors',new_model_only=True,web_changed=False))
    cache={};out=[];frames=[];memo={};reused=0;computed=0
    for file,dyn in [(U/'occurrence_supervision/RESULTS.json',False),(U/'dynamic_blocks/RESULTS.json',True)]:
        for r in read(file):
            word=r['clip_id'].split('_')[2]
            for p in r['pairs']:
                prob=p.get('order_probability',p.get('probability')); spans=p['spans']
                if prob is None or any(s is None for s in spans):continue
                lo,hi=sorted([int(round(sum(s)/2)) for s in spans]);ga,gb=[next(c['groups'][0][0] for c in rc.CARDS[word] if c['id']==i) for i in p['cards']]
                key=(r['clip_id'],'dynamic_complete_blocks' if dyn else r['kind'],r['seed'],lo,hi,ga,gb)
                if key in memo:assert memo[key]==prob,(key,memo[key],prob)
                memo[key]=prob
    for s in sources:
        assert role.sha(s['path'])==s['sha256']
        if s.get('baseline_sha256'):assert role.sha(s['baseline'])==s['baseline_sha256']
        cache[s['clip_id'],s['kind']]=({k:v for k,v in np.load(s['path']).items()},read(s['baseline']))
    for seed in cfg['seeds']:
        order=om.OrderedBinding();order.load_state_dict(torch.load(G/'order_runs'/str(seed)/'model.pt',weights_only=True,map_location='cpu'));order.eval()
        for condition in cfg['conditions']:
            net=role.RoleAdapter(seed);net.load_state_dict(torch.load(W/'runs'/condition/str(seed)/'model.pt',weights_only=True,map_location='cpu'));net.eval()
            for s in sources:
                cid,kind=s['clip_id'],s['kind'];word=cid.split('_')[2];z,b=cache[cid,kind]
                with torch.inference_mode():logits=net(torch.tensor(z['z']));logits[:,10]=-1e4;rp=logits.softmax(-1).numpy()
                p=W/'predictions'/condition/str(seed)/f'{cid}_{kind}.npz';p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,role=rp)
                cards=select_cards(rc.CARDS[word],rc.FAMILIES[word],np.asarray(b['phase_probability']),rp,np.asarray(b['observed_hand_mask']),z['ts'],phasecfg);pairs=[]
                for a,bb in order_observations(cards):
                    aa,bs=a['span'],bb['span'];prob=None;state='association_unknown'
                    if aa is not None and bs is not None:
                        lo,hi=sorted([int(round(sum(aa)/2)),int(round(sum(bs)/2))]);ga,gb=[next(c['groups'][0][0] for c in rc.CARDS[word] if c['id']==i) for i in [a['id'],bb['id']]];key=(cid,kind,seed,lo,hi,ga,gb)
                        if key not in memo:
                            with torch.inference_mode():memo[key]=float(order(torch.tensor(rc.observation(z['fh'],lo,hi)[None]),torch.tensor(rc.requirement('order',guides=[ga,gb])[None])).sigmoid()[0])
                            computed+=1
                        else:reused+=1
                        prob=memo[key];state='supported' if aa[1]<bs[0] and prob>=.65 else 'reversed' if bs[1]<aa[0] and prob<=.35 else 'ambiguous'
                    pair=dict(cards=[a['id'],bb['id']],spans=[aa,bs],state=state,order_probability=prob,semantic_completion=None)
                    if kind=='dynamic_complete_blocks':
                        seams=[j['query_right'] for j in s['target']['splice_joins'] if j['synthetic_discontinuity']]
                        pair['selected_span_crosses_splice']=any(l<j<=h for sp in [aa,bs] if sp for l,h in [sp] for j in seams)
                        limited=word=='WORD0739' and pair['cards']==['close_chest','raise_forearm'];pair['limited_dynamic_target']=limited;pair['spans_within_target_occurrences']=None
                        if limited and aa and bs:
                            ba,ca=[s['target']['blocks'][k]['query_frame_span'] for k in ['B','C']]
                            pair['spans_within_target_occurrences']=bool(ba[0]<=aa[0]<=aa[1]<=ba[1] and ca[0]<=bs[0]<=bs[1]<=ca[1])
                    pairs.append(pair)
                out.append(dict(condition=condition,seed=seed,clip_id=cid,kind=kind,cards=cards,pairs=pairs,role_prediction=str(p),role_sha256=role.sha(p)))
                if kind=='full_normal' and cid in labels:
                    for a in labels[cid]['anchors']:
                        target=role.jc.ep.MAP.get(a['label'],-1)
                        if target>=0:
                            raw=int(rp[a['frame']].argmax());frames.append(dict(condition=condition,seed=seed,clip_id=cid,frame=a['frame'],target=target,prediction_raw=raw,prediction_legacy8=raw if raw<8 else 0))
    write(W/'RESULTS.json',out);write(W/'FRAME_SCORES.json',frames)
    summaries=[]
    for condition in cfg['conditions']:
        for seed in cfg['seeds']:
            selected=[r for r in out if r['condition']==condition and r['seed']==seed]
            ff=[v for v in frames if v['condition']==condition and v['seed']==seed]
            pairs=[p for r in selected if r['kind']=='dynamic_complete_blocks' for p in r['pairs'] if p.get('limited_dynamic_target')]
            summaries.append(dict(condition=condition,seed=seed,legacy8=dict(correct=sum(v['target']==v['prediction_legacy8'] for v in ff),total=len(ff)),
              states={k:dict(Counter(p['state'] for r in selected if r['kind']==k for p in r['pairs'])) for k in sorted(set(r['kind'] for r in selected))},
              limited_dynamic=dict(correct_reversed_mapped=sum(p['state']=='reversed' and p['spans_within_target_occurrences'] is True and not p['selected_span_crosses_splice'] for p in pairs),total=len(pairs))))
    write(W/'EVALUATION_COMPLETE.json',dict(summary=summaries,order_scores_reused=reused,new_order_observations=computed,web_changed=False,final_test_unopened=True));print(summaries)


if __name__=='__main__':main()
