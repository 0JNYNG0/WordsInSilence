# Separate saved-artifact arithmetic verification; stdlib only, no model imports.
from pathlib import Path
import json,hashlib
U=Path(__file__).resolve().parent;O=U.parent
def rd(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def objsha(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
checks=[]
def ck(name,v):checks.append(dict(name=name,passed=bool(v)));assert v,name
cfg=rd(U/'FROZEN.json');result=rd(U/'RESULT.json');fr=rd(U/'FULLPATH_COMPLETE.json')['rows']
for p,h in {**cfg['pins'],**cfg['protections'],**rd(U/'CODE_PINS.json'),**rd(U/'EVAL_PINS.json')}.items():ck('pin:'+p,sha(p)==h)
for seed in cfg['seeds']:
 c=rd(U/'runs'/str(seed)/'COMPLETE.json');ck('modelhash',sha(U/'runs'/str(seed)/'model.pt')==c['sha256']);ck('899_every_epoch',len(c['history'])==cfg['epochs'] and all(r['common_seen']==899 for r in c['history']));ck('encoder_really_updated',c['encoder_delta']>0)
ck('unique_fullpath',len(fr)==702 and len({(r['seed'],r['mode'],r['clip_id'],r['kind']) for r in fr})==702)
for r in fr:
 ck('result_hash',sha(r['result'])==r['sha256']);d=rd(r['result']);b=rd(d['baseline']);ck('baselinehash',sha(d['baseline'])==d['baseline_sha256'])
 for k,h in d['protected'].items():ck('protected_'+k,objsha(b[k])==h)
 for x,y in zip(d['cards'],b['cards']):ck('shape_unchanged',x.get('shape_evidence')==y.get('shape_evidence'));ck('motion_unchanged',x.get('motion_evidence')==y.get('motion_evidence'))
 statuses=[c['display_status'] for c in d['cards']];ck('display_exact',statuses==r['after']['display']);ck('warning_exact',('difference' in statuses)==r['after']['warning'])
for g in result['groups']:
 rs=[r for r in fr if (r['seed'],r['mode'],r['kind'])==(g['seed'],g['mode'],g['kind'])];ck('group_denominator',len(rs)==g['n']);ck('group_warnings',sum(r['after']['warning'] for r in rs)==g['after_warning']);ck('group_similar',sum(r['after']['similar_no_warning'] for r in rs)==g['after_similar'])
for source,file in [('historical','HISTORICAL_DIAGNOSTIC.json'),('additional','ADDITIONAL_DIAGNOSTIC.json')]:
 ds=rd(U/file)
 for g in [g for g in result['diagnostics'] if g['source']==source]:
  rs=[r for r in ds if r['seed']==g['seed'] and r['kind']==g['kind'] and (g['scope']=='all' or (r['truth'].get('family') is not None)==(g['scope']=='four_focus'))];ck('diag_n',len(rs)==g['n']);matched=direction=both=exact=0
  for r in rs:
   a=r['automatic'][g['mode']];t=r['truth'];d=a['direction'];f=a['family'];ok=bool(d and d['eligible'] and d['prediction']==t['direction']);matched+=a['matched'];direction+=ok;both+=ok and (t.get('family') is None or bool(f and f['pred']==t['family']));exact+=a['both_boundaries_in_range']
  ck('diag_recomputed',(matched,direction,both,exact)==(g['matched'],g['direction_correct'],g['both_correct'],g['boundaries_in_range']))
 # Candidate cannot satisfy two separate motion occurrences.
 for seed in cfg['seeds']:
  for clip,kind in {(r['clip_id'],r['kind']) for r in ds}:
   rows=[r for r in ds if r['seed']==seed and r['clip_id']==clip and r['kind']==kind]
   for mode in ['baseline','interval_only','role_only','combined','selected_only']:
    spans=[(e['axis'],e['moving'],e['a'],e['b']) for r in rows if (e:=r['automatic'][mode]['event'])];ck('one_to_one',len(spans)==len(set(spans)))
fit=rd(U/'ALL_FIT_BOUNDARIES.json')['rows'];ev=rd(U/'EVALUATION_LABELS.json')['rows'];ck('fine_person_split',not {r['person'] for r in fit}&{r['person'] for r in ev});ck('source_split',not {r['source_sha256'] for r in fit}&{r['source_sha256'] for r in ev});ck('new_fit_events',sum(len(r['events']) for r in fit)==39);ck('new_eval_events',sum(len(r['events']) for r in ev)==16)
for r in rd(U/'OBSERVATIONS.json')['rows']:ck('observation_hash',sha(r['file'])==r['sha256'])
out=dict(passed=True,checks=len(checks),reviewer='Same Codex, separately implemented stdlib arithmetic; not independent expert or human ground truth.',result_sha256=sha(U/'RESULT.json'),items=checks);(U/'VERIFICATION.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),'utf-8');print('PASS',len(checks))
