"""Independent saved-result audit; no model inference or evaluation rerun."""
from pathlib import Path
import json, hashlib, datetime
from collections import Counter
import numpy as np

HERE=Path(__file__).resolve().parent
GRAPH=HERE.parent
PRIOR=GRAPH/'candidate_description_supervision'
OPS=GRAPH.parent

def read(path): return json.loads(Path(path).read_text('utf-8-sig'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stamp(): return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()
checks=[]
def ck(name,ok,detail=None):
    checks.append(dict(name=name,ok=bool(ok),detail=detail))

def pins(obj,label):
    for path,expected in obj.items():
        p=Path(path)
        ck(label+':'+str(p),p.is_file() and sha(p)==expected)

def verdict(p,desired,eligible):
    if not eligible:return 'observation_unknown'
    if desired==1:return 'supported' if p[1]>=.65 and max(p[0],p[2])<.65 else 'uncertain'
    if p[desired]>=.65 and p[2-desired]<.65 and p[1]<.65:return 'supported'
    if p[2-desired]>=.65 and p[desired]<.35 and p[1]<.65:return 'opposite_direction'
    if p[1]>=.65:return 'movement_not_established'
    return 'uncertain'

def status(card):
    states=[e['state'] for e in card['evidence']]
    if card['span'] is None:return 'association_unknown'
    if 'difference' in states or 'opposite_direction' in states:return 'difference_evidence'
    return 'supported_features_only' if states and all(s=='supported' for s in states) else 'insufficient_evidence'

def key(row):return row['seed'],row['clip_id'],row['kind']
def main():
    destination=HERE/'VERIFICATION.json'
    assert not destination.exists(),'Preserve existing verification; no overwrite'
    cfg=read(HERE/'FROZEN.json'); meta=read(HERE/'INPUTS_COMPLETE.json')
    audit=read(HERE/'INPUT_CONTRACT_AUDIT.json')
    train=read(HERE/'TRAINING_COMPLETE.json'); tc=read(HERE/'TRAINING_CONTRACT.json')
    ef=read(HERE/'evaluation/FROZEN.json'); result=read(HERE/'evaluation/RESULTS.json')
    for label,values in [('freeze',cfg['pins']),('inputs',meta['pins']),('training',tc['pins']),('evaluation',ef['pins'])]:pins(values,label)
    pins({meta['path']:meta['sha256'],meta['source_meta']:meta['source_sha256']},'input_files')
    oldp=read(PRIOR/'projected_direction_learning/INPUTS_COMPLETE.json')
    oldo=read(PRIOR/'boundary_direction_learning/SCOPED_INPUTS.json')
    pins({oldp['path']:oldp['sha256'],oldo['path']:oldo['sha256']},'saved_input_files')
    x=np.load(meta['path']); p=np.load(oldp['path']); o=np.load(oldo['path']);rows=meta['rows'];n=len(rows)
    ck('row_metadata_unchanged',rows==oldp['rows'])
    ck('typed_axes_only',all(r['axis'] in [1,2,3,4] for r in rows))
    ck('labels_unchanged',np.array_equal(x['y'],p['y']) and np.array_equal(x['y'],o['y']))
    ck('original_geometry_unchanged',np.array_equal(x['original_geometry'],o['g']))
    ck('projected_geometry_unchanged',np.array_equal(x['projected_geometry'],p['g']))
    ck('recorded_old_context_exact',np.array_equal(x['original_context'],p['context']))
    for k in x.files:ck('finite_'+k,np.isfinite(x[k]).all() and len(x[k])==n)
    normal=np.array([not r['reverse'] for r in rows]);reversed_mask=~normal
    fit=np.array([r['split']=='train' for r in rows]);probe=~fit
    delta=np.max(np.abs(x['context']-p['context']),axis=1)
    ck('normal_context_matches',np.allclose(x['context'][normal],p['context'][normal],atol=1e-6),float(delta[normal].max()))
    changed=int((delta[fit&reversed_mask]>1e-6).sum());reverse_count=int((fit&reversed_mask).sum())
    ck('fit_reverse_count',reverse_count==audit['fit_reverse_count'])
    ck('fit_reverse_changes',changed==audit['fit_reverse_context_changed'])
    ck('fit_reverse_max_delta',np.isclose(delta[fit&reversed_mask].max(),audit['max_fit_context_difference']))
    ck('source_sha_disjoint',not {r['source_sha256'] for i,r in enumerate(rows) if fit[i]} & {r['source_sha256'] for i,r in enumerate(rows) if probe[i]})
    ck('person_split',all(r['person'] in (cfg['fit_people'] if fit[i] else cfg['probe_people']) for i,r in enumerate(rows)))
    ck('only_declared_splits',all(r['split'] in ['train','probe'] for r in rows))
    for name,g in [('original',o['g']),('projected',p['g'])]:
        ck(name+'_fit_norm_mean',np.allclose(g[fit].mean(0),meta['norm'][name]['mean']))
        ck(name+'_fit_norm_std',np.allclose(np.maximum(g[fit].std(0),.01),meta['norm'][name]['std']))
    ck('projected_norm_saved_baseline',np.allclose(meta['norm']['projected']['mean'],oldp['norm']['mean']) and np.allclose(meta['norm']['projected']['std'],oldp['norm']['std']))
    ck('fixed_training_contract',tc['seeds']==cfg['seeds']==[42,43,44] and tc['steps']==cfg['steps']==800 and tc['selected']=='final_step_only')
    expected_runs={(c,s) for c in cfg['conditions'] for s in cfg['seeds']}
    ck('six_unique_runs',{(r['condition'],r['seed']) for r in train['rows']}==expected_runs and len(train['rows'])==len(expected_runs))
    training_summary=[]
    for r in train['rows']:
        pins({r[k]:r[k+'_sha256'] for k in ['model','predictions']},'run')
        logits=np.load(r['predictions'])['logits']
        ck('run_prediction_shape:'+str((r['condition'],r['seed'])),logits.shape==(n,3) and np.isfinite(logits).all())
        ck('final_step:'+str((r['condition'],r['seed'])),r['log'][-1]['step']==cfg['steps'])
        for split,mask in [('train',fit),('probe',probe)]:
            training_summary.append(dict(condition=r['condition'],seed=r['seed'],split=split,n=int(mask.sum()),correct=int((logits[mask].argmax(-1)==x['y'][mask]).sum())))
    shadow=read(PRIOR/'local_rank_learning/evaluation/shadow_components/RESULTS.json')
    originals={key(r):r for r in shadow['rows']}; out={key(r):r for r in result['rows']}
    ck('all_original_row_keys',set(out)==set(originals) and len(out)==len(result['rows']))
    ck('full_82_scenarios_each_seed',all(len([k for k in out if k[0]==s])==82 for s in cfg['seeds']))
    saved={c:{key(r):r for r in read(PRIOR/folder/'evaluation/RESULTS.json')['rows']} for c,folder in [('saved_old_context_original','boundary_direction_learning'),('saved_old_context_projected','projected_direction_learning')]}
    expected_conditions=set(ef['conditions']);preserved=0;changes=[]
    byprediction={(r['condition'],*key(r),r['card']):r for r in result['direction_predictions']}
    ck('unique_direction_predictions',len(byprediction)==len(result['direction_predictions']))
    used_predictions=set()
    for kk,row in out.items():
        prior=originals[kk]['conditions'];conditions=row['conditions']
        ck('condition_set:'+str(kk),set(conditions)==expected_conditions)
        ck('baseline_exact:'+str(kk),conditions['baseline']==prior['baseline'])
        ck('local_old_exact:'+str(kk),conditions['local_old']==prior['local_full'])
        for c in expected_conditions-{'baseline','local_old'}:
            cards=conditions[c];oldcards=prior['local_full'];ck('card_count:'+str((c,kk)),len(cards)==len(oldcards))
            for card,oldcard in zip(cards,oldcards):
                ck('card_metadata:'+str((c,kk,card['card'])),{k:v for k,v in card.items() if k not in ['status','evidence']}=={k:v for k,v in oldcard.items() if k not in ['status','evidence']})
                ck('event_count:'+str((c,kk,card['card'])),len(card['evidence'])==len(oldcard['evidence']))
                ck('card_status:'+str((c,kk,card['card'])),card['status']==status(card))
                for event,previous in zip(card['evidence'],oldcard['evidence']):
                    eid=(c,*kk,card['card'])
                    if event['task']!='direction' or event.get('axis')!=2:
                        ck('nonaxis2_exact:'+str(eid),event==previous);preserved+=1;continue
                    ck('axis2_identity:'+str(eid),{k:v for k,v in event.items() if k not in ['probabilities','state','model_scope']}=={k:v for k,v in previous.items() if k not in ['probabilities','state','model_scope']})
                    if c in saved:
                        source=next(q for q in saved[c][kk]['conditions']['local_tuned'] if q['card']==card['card'])
                        source_event=next(q for q in source['evidence'] if q['task']=='direction' and q.get('axis')==2)
                        ck('saved_axis2_exact:'+str(eid),event==source_event)
                    else:
                        ck('prediction_exists:'+str(eid),eid in byprediction)
                        if eid not in byprediction:continue
                        r=byprediction[eid];used_predictions.add(eid);v=np.asarray(r['logits'],float);prob=np.exp(v-v.max());prob/=prob.sum()
                        ck('softmax:'+str(eid),np.allclose(prob,r['probabilities'],atol=1e-6))
                        ck('verdict:'+str(eid),verdict(prob,r['desired'],r['eligible'])==r['state']==event['state'])
                        ck('event_probability:'+str(eid),np.array_equal(event['probabilities'],r['probabilities']))
                        ck('span_and_query:'+str(eid),event['span']==r['span'] and event['desired']==r['desired'] and r['axis']==2)
                        if previous['state']!=event['state']:changes.append(dict(condition=c,seed=kk[0],clip_id=kk[1],kind=kk[2],card=card['card'],before=previous['state'],after=event['state']))
    ck('all_predictions_used',used_predictions==set(byprediction))
    pins({r['source']:r['source_sha256'] for r in result['direction_predictions']},'eval_sources')
    summary_expected=[]
    for seed in cfg['seeds']:
        for condition in ef['conditions']:
            for kind in sorted({r['kind'] for r in result['rows']}):
                events=[e for row in result['rows'] if row['seed']==seed and row['kind']==kind for card in row['conditions'][condition] for e in card['evidence'] if e['task']=='direction' and e.get('axis')==2]
                summary_expected.append(dict(seed=seed,condition=condition,kind=kind,n=len(events),states=dict(Counter(e['state'] for e in events))))
    sk=lambda r:(r['seed'],r['condition'],r['kind'])
    ck('all_condition_denominators_and_counts',sorted(summary_expected,key=sk)==sorted(result['summary'],key=sk))
    protected=read(GRAPH/'PROTECTIONS.json');pins(protected,'protected')
    ck('frozen_protection_count',len(protected)==477)
    for label,record in [('freeze',cfg),('training',train),('evaluation_freeze',ef)]:
        for name in ['private_access','web_changed','final8']:ck(label+'_'+name,record.get(name) is False)
    report=dict(at=stamp(),status='PASS' if all(c['ok'] for c in checks) else 'FAIL',checks=len(checks),failed=[c for c in checks if not c['ok']],
                rows=len(result['rows']),direction_predictions=len(result['direction_predictions']),preserved_nonaxis2_events=preserved,protected_files=len(protected),
                fit_rows=int(fit.sum()),probe_rows=int(probe.sum()),fit_reverse_context_changed=changed,fit_reverse_count=reverse_count,
                training_agreement=training_summary,summary=summary_expected,changes=changes,
                no_model_inference=True,adopted=False,full_G2_pass=False,
                reviewer='Delegated Codex motion_contract_review_1113 separate script and process; not independent human/KSL ground truth',
                limitations=cfg['caveats'],script_sha256=sha(__file__))
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
    print(json.dumps({k:report[k] for k in ['status','checks','failed','rows','direction_predictions','preserved_nonaxis2_events','protected_files','fit_reverse_context_changed']},ensure_ascii=False),flush=True)
    if report['failed']:raise SystemExit(1)

if __name__=='__main__':main()
