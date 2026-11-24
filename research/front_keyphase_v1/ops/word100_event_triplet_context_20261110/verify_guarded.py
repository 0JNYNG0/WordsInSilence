from pathlib import Path
import json,hashlib
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n','utf-8')
def cmp(d):
 w=any(c['display_status']=='difference' for c in d['cards']);s=not w and all(c['display_status']=='similar' for c in d['cards'] if c.get('required',True))
 return dict(warning=w,similar=s,uncertain=not w and not s)
checks=[]
def ck(n,v):checks.append(dict(name=n,passed=bool(v)));assert v,n
rr=read(U/'GUARDED_COMPLETE.json')['rows'];ck('234',len(rr)==234);stats={};changes=[];reg=[]
for r in rr:
 ck('hashes',all(sha(r[k])==r[k+'_sha256'] for k in ['candidate','baseline']) and sha(r['result'])==r['sha256']);a=read(r['result']);b=read(r['baseline']);c=read(r['candidate'])
 ck('evidence_kept',all(a[k]==c[k] for k in ['triplet_segments','guide_link_bindings','path','tcn_top3','phase_events','phase_partials']))
 for aa,cc in zip(a['cards'],c['cards']):
  ck('warning_preserved',cc['display_status']!='difference' or aa['display_status']=='difference');ck('no_new_positive',aa['display_status']!='similar' or cc['display_status']=='similar')
  au=aa['triplet_completion_audit'];bb=next(x for x in c['guide_link_bindings'] if x['card']==aa['id']);es=[e for e in bb['movement_links'] if e['eligible'] and e['confidence']>=.65 and e['prediction']!=1];pp={(p['a'],p['b']):p for p in c['triplet_segments']};v=[]
  for e in es:
   p=pp.get(tuple(e['span']));v.append(bool(p and p['observed'] and p['context'] in bb['contexts'] and p['confidence']>=.65 and p['core_state']=='progress' and p['core_probabilities'][1]>=.65 and not e['complete']))
  expected=bool(cc['display_status']=='similar' and v and all(v) and not any(e['complete'] for e in bb['movement_links']))
  ck('veto_arithmetic',expected==au['withdraw_positive'])
 aa,bb=cmp(a),cmp(b)
 if aa!=bb:changes.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],before=bb,after=aa))
 if r['kind'] in ['full_normal','speed075','speed125'] and aa!=bb:reg.append(changes[-1])
 cats=[r['kind']]+(['crossing'] if r['kind']=='full_normal' and 'WORD1567' in r['clip_id'] else [])
 for k in cats:
  stats.setdefault(str(r['seed']),{});s=stats[str(r['seed'])].setdefault(k,{z:dict(n=0,warning=0,similar=0,uncertain=0) for z in ['before','after']})
  for side,data in [('before',bb),('after',aa)]:
   s[side]['n']+=1
   for key in ['warning','similar','uncertain']:s[side][key]+=int(data[key])
cfg=read(U/'FROZEN.json')
for p,h in cfg['protections'].items():ck('protected',sha(p)==h)
ck('first_result_preserved',sha(U/'RESULT.json')==read(U/'VERIFICATION.json')['result_sha256'])
write(U/'GUARD_RESULT.json',dict(automatic=stats,changes=changes,normal_regressions=reg,adopt=False,reason='Normal/speed preservation fails. Primary seed removes one false similar but also one normal and both speed variants. Other seeds unstable. No threshold tuning; candidate preserved but not adopted.'))
write(U/'GUARD_VERIFICATION.json',dict(passed=True,count=len(checks),checks=checks,result_sha256=sha(U/'GUARD_RESULT.json')))
print(json.dumps(dict(passed=True,checks=len(checks),changes=changes,normal_regressions=len(reg)),ensure_ascii=False))
