"""Saved-result arithmetic and provenance audit; no model forward or rescoring."""
from pathlib import Path
import json, hashlib, collections, ast
import numpy as np
import torch

N=Path(__file__).resolve().parent
O=N.parent
U=O/'word100_attempt_event_supervision_20261109'
P=O/'word100_separate_phase_learning_20261109'
checks=[]
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(test,label):
    if not bool(test): raise AssertionError(label)
    checks.append(label)
def write(p,d): Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def summarize(decisions):
    return dict(clips=len(decisions),
      difference=sum(any(c['display_status']=='difference' for c in d['cards']) for d in decisions),
      uncertain=sum(any(c['display_status']=='uncertain' for c in d['cards']) for d in decisions),
      missing=sum(any(c['status']=='missing_candidate' for c in d['cards']) for d in decisions),
      similar=sum(all(c['display_status']=='similar' for c in d['cards'] if c.get('required',True)) for d in decisions),
      required_present=sum(all(all(e['span'] is not None for e in c['phase_evidence']) for c in d['cards'] if c.get('required',True)) for d in decisions))

f=read(N/'FROZEN.json');inp=read(N/'INPUTS.json');old=read(U/'INPUTS.json')
check(sha(N/'INPUTS.json')==f['input_sha256'],'frozen input inventory')
for path,h in {**f['protected'],**f['code_sha256'],**read(N/'EVAL_CONTRACT.json')['code_pins'],**read(N/'EVAL_CODE_PINS.json')}.items():
    check(sha(path)==h,'unchanged '+path)
tf=read(N/'TRAIN_FROZEN.json')
check(tf['code_sha256']==sha(N/'train.py') and tf['core_sha256']==sha(N/'core.py'),'unchanged training program')
cache={}
for r in inp['rows']:
    check(sha(r['path'])==r['sha256'],'input sha '+r['clip_id']+'/'+r['kind'])
    z=np.load(r['path']);cache[(r['clip_id'],r['kind'])]={k:z[k] for k in z.files}
    check(r['clip_id'].startswith('NIA_SL_WORD') and r['person'] in list(range(1,10))+[11,12,13],'public record '+r['clip_id'])
check(len(inp['rows'])==564,'564 conditions from 48 originals')
fit=[r for r in inp['rows'] if r['role']=='fit'];dev=[r for r in inp['rows'] if r['role']=='development']
check(len({r['clip_id'] for r in fit})==36 and len({r['clip_id'] for r in dev})==12,'36FIT12DEV originals')
check(not {r['source_sha256'] for r in fit}&{r['source_sha256'] for r in dev},'original hash separation')
for run in sorted((N/'runs').iterdir()):
    d=read(run/'COMPLETE.json');checkpoint=torch.load(run/'model.pt',map_location='cpu',weights_only=True)
    check(d['sha256']==sha(run/'model.pt'),'checkpoint '+run.name)
    check(d['epochs']==100 and len(d['history'])==100,'100 epochs '+run.name)
    check(all(np.isfinite(x['loss']) for x in d['history']),'finite loss '+run.name)
    held=set(map(int,run.name.split('_')[1:])) if run.name.startswith('heldout_') else set()
    people=set(range(1,10))-held
    check(set(d['fit_people'])==people,'people split '+run.name)
    selected=[r for r in fit if r['person'] in people]
    check(set(d['fit_clips'])=={r['clip_id'] for r in selected},'clip split '+run.name)
    check(not set(d['views_seen'])-set(f['training_views']),'no unapproved views '+run.name)
    gg=np.concatenate([cache[(r['clip_id'],r['kind'])]['g'] for r in selected if r['kind'] in ['full_normal','actual_crop']])
    mean=gg.mean(0);std=gg.std(0).clip(.05);mean[22:33]=mean[55:]=0;std[22:33]=std[55:]=1
    check(np.array_equal(mean,checkpoint['mean'].numpy()) and np.array_equal(std,checkpoint['std'].numpy()),'train-only normalization '+run.name)
check(len(list((N/'runs').iterdir()))==6,'six completed training runs')

web=read(N/'WEB_BASELINE.json');base={}
for r in web['rows']:
    check(sha(r['decision'])==r['sha256'],'web saved decision '+r['key'])
    base[r['key']]=read(r['decision'])
check(len(base)==237 and not web['TCN_rerun'],'237 feedback baselines; no TCN rerun')
summary=read(N/'COMPARISON.json');diagnostics={};saved={}
for name,result in summary['models'].items():
    rows=read(N/'scores'/f'{name}_cases.json');decisions={};preds={};normal_changes=[];rescued=[];unresolved=[]
    check(len(rows)==237 and len({r['key'] for r in rows})==237,'237 unique conditions '+name)
    for r in rows:
        check(sha(r['decision'])==r['sha256'],'candidate saved decision '+name+'/'+r['key'])
        d=read(r['decision']);decisions[r['key']]=d;p=np.load(N/'predictions'/name/f"{r['key']}.npz")['prob'];preds[r['key']]=p
        check(np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-6),'valid probabilities '+name+'/'+r['key'])
        check(d['all_candidates_preserved'] and d['unassigned_is_not_error'],'retained evidence '+name+'/'+r['key'])
        for c in d['cards']:
            bank={(tuple(x['group']),tuple(x['span'])) for x in c['candidate_bank']}
            retained={(tuple(e['group']),tuple(s)) for e in c['phase_evidence'] for s in e['available_runs']}
            check(bank<=retained,'candidate bank not dropped '+name+'/'+r['key']+'/'+c['id'])
            check(not c['educational_wrong'],'not expert error truth '+name+'/'+r['key']+'/'+c['id'])
            if c['selected_route']=='geometry' and c.get('required',True):
                rescued.append(dict(key=r['key'],card=c['id'],status=c['display_status'],span=c['query_indices']))
        if r['kind']=='full_normal':
            for a,b in zip(base[r['key']]['cards'],d['cards']):
                if a['display_status']!=b['display_status']:
                    normal_changes.append(dict(key=r['key'],card=b['id'],before=a['display_status'],after=b['display_status'],selected=b.get('query_indices'),conflict_spans=[x['span'] for x in b.get('alternative_run_conflicts',[])],reason=b.get('uncertainty_reason')))
        if r['kind'] in ['omit_final_event','mask_final_event'] and all(c['display_status']=='similar' for c in d['cards'] if c.get('required',True)):
            unresolved.append(dict(key=r['key'],cards=[dict(id=c['id'],span=c.get('query_indices')) for c in d['cards']]))
        if r['inventory']=='main144' and r['kind'] in ['close_R','close_L','close_both']:
            key=f"main144_{r['clip_id']}_full_normal"
            other=np.load(N/'predictions'/name/f'{key}.npz')['prob']
            check(np.array_equal(p,other),'shape-independent output '+name+'/'+r['key'])
        if r['inventory']=='legacy93' and r['kind']=='right_closed_hand_main':
            other=np.load(N/'predictions'/name/f"legacy93_{r['clip_id']}_crop_normal.npz")['prob']
            check(np.array_equal(p,other),'matched crop shape output invariant '+name+'/'+r['key'])
    for kind,expected in result['by_kind'].items():
        keys=[r['key'] for r in rows if r['inventory']+'/'+r['kind']==kind]
        check(summarize([base[k] for k in keys])==expected['before'],'baseline counts '+name+'/'+kind)
        check(summarize([decisions[k] for k in keys])==expected['after'],'candidate counts '+name+'/'+kind)
    diagnostics[name]=dict(required_geometry_rescues=rescued,normal_regressions=normal_changes,remaining_false_completion=unresolved)
    saved[name]=preds

records=read(N/'STAGE_RECORDS.json');seen=set();point_summary={}
for r in records:
    key=(r['model'],r['clip_id'],r['kind'],r['frame'])
    check(key not in seen,'unique source frame diagnostic');seen.add(key)
    p=saved[r['model']][f"main144_{r['clip_id']}_{r['kind']}"]
    check(int(p[r['index']].argmax())==r['geometry_pred'],'saved point argmax')
for name in summary['models']:
    rr=[r for r in records if r['model']==name and r['kind']=='full_normal']
    point_summary[name]=dict(correct=sum(r['target']==r['geometry_pred'] for r in rr),total=len(rr))
held=read(N/'HELDOUT_CHECK.json');aggregated={}
for r in held['rows']:
    check(sha(r['predictions'])==r['sha256'],'heldout saved prediction hash')
    check(r['person'] not in read(N/'runs'/r['model']/'COMPLETE.json')['fit_people'],'heldout model did not fit recordID')
    p=np.load(r['predictions'])['phase']
    for point in r['records']:check(int(p[point['index']].argmax())==point['pred'],'heldout argmax')
for kind in held['by_kind']:
    rr=[r for r in held['rows'] if r['kind']==kind];pp=[p for r in rr for p in r['records']]
    d=dict(clips=len(rr),clips_with_direct_gold=sum(bool(r['records']) for r in rr),correct=sum(p['pred']==p['target'] for p in pp),total=len(pp))
    check(d==held['by_kind'][kind],'heldout arithmetic '+kind);aggregated[kind]=d

# Static audit: algorithm is not conditional on the four word IDs or names.
for filename in ['core.py','train.py','compare.py']:
    tree=ast.parse((N/filename).read_text(encoding='utf-8'))
    literals=[x.value for x in ast.walk(tree) if isinstance(x,ast.Constant) and isinstance(x.value,str)]
    check(not any(any(w in s for w in ['WORD0739','WORD1133','WORD1567','WORD2523','순종','잘하다','조상','이집트']) for s in literals),'no word-specific rule '+filename)
check(summary['primary']=='final_42' and not summary['new_web_model_deployed'],'no seed selection or deployment')
gate=dict(normal_preservation_pass=False,all_missing_event_success_removed=False,all_missing_sensor_success_removed=False,
    exact_shape_independence_pass=True,protected_files_pass=True,all_fixed_conditions_pass=True,
    deploy=False,reason='Normal full-view uncertainty rises3to4; one omitted event and one sensor-masked event still all-similar. Correct counts do not imply overall acceptance.')
write(N/'DIAGNOSTIC.json',dict(models=diagnostics,stage_points=point_summary,heldout=aggregated,scope='Saved-output diagnosis only; no inference, threshold selection or rescoring. Repeated inventories are not independent people.'))
write(N/'VERIFIED.json',dict(status='PASS',checks=len(checks),gates=gate,neural_runs=0,training_runs=0,
    verifier='Codex separate saved arithmetic/code audit; not a separate expert or human annotator',
    source_pins={str(N/p):sha(N/p) for p in ['COMPARISON.json','HELDOUT_CHECK.json','FROZEN.json','TRAIN_FROZEN.json','EVAL_CONTRACT.json','INPUTS.json','DIAGNOSTIC.json']},
    audit_code_sha256=sha(__file__)))
print(json.dumps(dict(checks=len(checks),status='PASS',stage_points=point_summary,gates=gate),ensure_ascii=False))
