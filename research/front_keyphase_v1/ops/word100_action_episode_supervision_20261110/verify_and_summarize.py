"""Separate stdlib-only arithmetic audit of saved predictions. No model imports."""
from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib,statistics,math,ast
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2),'utf-8')
checks=[]
def check(name,ok):
 checks.append(dict(name=name,passed=bool(ok)))
 if not ok:raise AssertionError(name)
def compact(o):
 states=[c['display_status'] for c in o['cards']]
 return dict(display=states,warning='difference' in states,similar_no_warning=bool(states) and all(s=='similar' for s in states))
def iou(a,b):
 inter=max(0,min(a[1],b[1])-max(a[0],b[0])+1)
 return inter/(a[1]-a[0]+b[1]-b[0]+2-inter)
def match(events,truth):
 edges=[(iou((t['onset'][0],t['offset'][1]),(e['a'],e['b'])),i,j) for i,t in enumerate(truth) for j,e in enumerate(events) if (t['axis'],t['moving'])==(e['axis'],e['moving'])]
 out={};taken=set()
 for overlap,i,j in sorted(edges,reverse=True):
  if overlap>=.5 and i not in out and j not in taken:out[i]=(j,overlap);taken.add(j)
 return out
cfg=read(U/'FROZEN.json');ep=read(U/'EVALUATION_FROZEN.json')
for p,h in {**cfg['protections'],**cfg['pins'],**read(U/'CODE_PINS.json'),**ep['pins']}.items():check('pinned '+p,sha(p)==h)
for p in U.glob('*.py'):ast.parse(p.read_text('utf-8'));check('syntax '+p.name,True)
check('preflight',read(U/'PREFLIGHT.json')['passed'])
fits=[];train=[]
for seed in cfg['seeds']:
 c=read(U/f'runs/{seed}/COMPLETE.json');seen=c['seen'];fits.append(set(seen))
 check(f'training {seed}',len(seen)==len(set(seen))==899 and len({v.split('_')[2] for v in seen})==100 and len(c['history'])==12 and all(v['common_seen']==899 for v in c['history']))
 check(f'train identities {seed}',{int(v.split('REAL')[1].split('_')[0]) for v in seen}==set(range(1,10)))
 check(f'checkpoint {seed}',sha(U/f'runs/{seed}/model.pt')==c['sha256'] and c['encoder_delta']>0)
 train.append({k:v for k,v in c.items() if k not in ['seen','history']})
check('identical common sets',fits[0]==fits[1]==fits[2])
ex=read(U/'EXTRACTION_COMPLETE.json')['rows'];check('additional4 frames483',len(ex)==4 and sum(r['frames'] for r in ex)==483)
for r in ex:
 for k,h in [('source','source_sha256'),('raw','raw_sha256'),('sheet','sheet_sha256')]:check(r['clip_id']+k,sha(r[k])==r[h])
 check(r['clip_id']+' split/exposure',r['person']==14 and r['clip_id'] not in fits[0] and not r['fully_unseen_project_data'] and bool(r['prior_tcn_evaluation']))
labels=read(U/'EVALUATION_LABELS.json');check('label counts',len(labels['historical'])==7 and len(labels['additional'])==7)
for t in labels['historical']+labels['additional']:
 check('label source '+t['clip_id']+str(t['onset']),sha(t['sheet'])==t['sheet_sha256'] and t['clip_id'] not in fits[0])
rows=read(U/'FULLPATH_COMPLETE.json')['rows'];check('234 fixed outputs',len(rows)==234 and len({(r['seed'],r['clip_id'],r['kind']) for r in rows})==234)
group=defaultdict(lambda:dict(n=0,before_warnings=0,after_warnings=0,before_similar=0,after_similar=0,changed=0))
changed=[];loaded={};warnings=[]
OLD_MAP={0:0,1:1,2:2,3:3,4:4,5:4,6:5,7:5,8:5,9:5,10:5,11:6,12:7,13:7}
for r in rows:
 key=(r['seed'],r['clip_id'],r['kind']);old=read(r['baseline']);new=read(r['result']);loaded[key]=(old,new)
 check('output hashes '+str(key),sha(r['baseline'])==r['baseline_sha256'] and sha(r['result'])==r['sha256'])
 a,b=compact(old),compact(new);check('independent counts '+str(key),a==r['before'] and b==r['after'])
 check('preserved input + old evidence '+str(key),all(old[k]==new[k] for k in ['tcn_top3','path','frame_id','timestamp_us','phase_events','phase_partials','phase_probability','observed_hand_mask']) and all(c.get('shape_evidence')==d.get('shape_evidence') and c.get('motion_evidence')==d.get('motion_evidence') for c,d in zip(old['cards'],new['cards'])))
 check('old warning retained '+str(key),not a['warning'] or b['warning'])
 check('no positive promotion '+str(key),all(d['display_status']!='similar' or c['display_status']=='similar' for c,d in zip(old['cards'],new['cards'])))
 g=group[(r['seed'],r['kind'])];g['n']+=1;g['before_warnings']+=a['warning'];g['after_warnings']+=b['warning'];g['before_similar']+=a['similar_no_warning'];g['after_similar']+=b['similar_no_warning'];g['changed']+=a!=b
 if a!=b:changed.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],before=a,after=b,result=r['result']))
 for prior,c in zip(old['cards'],new['cards']):
  z=c['episode_binding'];eligible=[]
  for m in z['movements']:
   sign=m['prediction']-1;delta=m['geometry'].get('signed_change',0);same=(sign==0 or (delta>0)-(delta<0)==sign)
   check('direction eligibility '+str(key)+c['id']+str(m['span']),m['eligible']==(m['geometry_eligible'] and same))
   if m['eligible'] and m['confidence']>=.65 and sign:eligible.append(sign)
  for q in z['conflict_checks']:
   vals=old['phase_probability'][q['span'][0]:q['span'][1]+1];votes=sum(max(p)>=.65 and OLD_MAP[max(range(len(p)),key=p.__getitem__)]!=q['family'] for p in vals)
   check('conflict votes '+str(key)+c['id']+str(q['span']),votes==q['votes'] and q['conflict']==(votes>len(vals)/2))
  if c['display_status']!=prior['display_status']:
   check('only guarded negative additions '+str(key)+c['id'],c['display_status']=='difference' and not z['conflict'] and not c.get('observation_quarantine_overlap') and not c['educational_wrong'])
   if 'episode_direction' in c.get('difference_components',[]):check('negative sign supported '+str(key)+c['id'],set(eligible)=={-z['expected_sign']} and z['state']=='contradicted')
   warnings.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],card=c['id'],components=c['difference_components'],movements=z['movements']))
check('normal speed unchanged',all(r['before']==r['after'] for r in rows if r['kind'] in ['full_normal','speed075','speed125']))
check('missing all unchanged',all(r['before']==r['after'] and not r['after']['similar_no_warning'] for r in rows if r['kind'] in ['all_hand_missing','mask_final_event','omit_final_event']))
cross=[r for r in rows if 'WORD1567' in r['clip_id'] and r['kind'] in ['full_normal','speed075','speed125']];check('crossing normal/speed preserved',len(cross)==27 and all(r['before']==r['after'] for r in cross))
diagnostic=[];misses=[]
for filename,scope in [('DIAGNOSTIC_SCORES.json','historical'),('ADDITIONAL_DIAGNOSTIC_SCORES.json','additional')]:
 ds=read(U/filename);byclip=defaultdict(list)
 for d in ds:byclip[(d['seed'],d['clip_id'],d['kind'])].append(d)
 for key,dd in byclip.items():
  seed,clip,kind=key
  if scope=='historical':old,new=loaded[key];candidate={'old':old['phase_events']+old['phase_partials'],'new':new['episode_events']+new['episode_partials']}
  else:
   old=read(U/'additional_baseline'/f'{clip}_{kind}.json');new=read(U/'additional_results'/str(seed)/f'{clip}_{kind}.json');candidate={'old':old['phase_events']+old['phase_partials'],'new':new['events']+new['partials']}
  for method,es in candidate.items():
   es=[e for e in es if e['b']-e['a']>=3];mm=match(es,[d['truth'] for d in dd])
   for i,d in enumerate(dd):
    a=d['automatic'][method];m=mm.get(i);check('one-to-one '+scope+str(key)+method+str(i),a['matched']==(m is not None) and (not m or a['event']==es[m[0]] and abs(a['overlap']-m[1])<1e-9))
 for seed in cfg['seeds']:
  for kind in ['all']+sorted({d['kind'] for d in ds}):
   sub=[d for d in ds if d['seed']==seed and (kind=='all' or d['kind']==kind)]
   met=dict(scope=scope,seed=seed,kind=kind,n=len(sub),manual_old_family=sum(d['manual_old_family']['prediction']==d['truth']['family'] for d in sub),manual_new_family=sum(d['manual_new_family']['prediction']==d['truth']['family'] for d in sub),manual_direction=sum(d['manual_direction']['eligible'] and d['manual_direction']['prediction']==d['truth']['direction'] for d in sub))
   for method in ['old','new']:
    aa=[(d['automatic'][method],d['truth']) for d in sub]
    met[method]=dict(matched=sum(a['matched'] for a,t in aa),direction=sum(a['matched'] and a['direction']['eligible'] and a['direction']['prediction']==t['direction'] for a,t in aa),joint=sum(a['matched'] and a['direction']['eligible'] and a['direction']['prediction']==t['direction'] and a['family']['prediction']==t['family'] for a,t in aa),both_boundaries_in_range=sum(a['both_boundaries_in_range'] for a,t in aa))
   diagnostic.append(met)
 for d in ds:
  a=d['automatic']['new'];t=d['truth']
  if not a['matched'] or not a['direction']['eligible'] or a['direction']['prediction']!=t['direction']:
   misses.append(dict(scope=scope,seed=d['seed'],clip_id=d['clip_id'],kind=d['kind'],truth=t,new=a,old=d['automatic']['old']))
times=[r['seconds'] for r in rows if r['seed']==42];frames=[r['frames'] for r in rows if r['seed']==42]
summary=dict(status='LIMITED_DIAGNOSTIC_GAIN_NO_DEFAULT_ADOPTION',primary_seed=42,training=train,data=read(U/'DATA_AUDIT.json'),groups=[dict(seed=k[0],kind=k[1],**v) for k,v in group.items()],changes=changed,diagnostic=diagnostic,automatic_failures=misses,crossing_checks=len(cross),latency=dict(scope='New episode head, proposal decoding and supplement only; excludes frozen latent preparation, RTMW, existing TCN/DTW/web/camera.',clips=len(times),frames_min=min(frames),frames_max=max(frames),median_seconds=statistics.median(times),p95_seconds=sorted(times)[math.ceil(.95*len(times))-1],max_seconds=max(times)),additional_extraction=dict(videos=len(ex),frames=sum(r['frames'] for r in ex),seconds=[r['seconds'] for r in ex],scope='Offline GPU RTMW + sequential video read; not post-recording latency.'),decision='Keep existing L baseline; store new model as isolated supplementary research candidate. New boundary model alone regressed on additional diagnostic events; no default/web adoption.',limitations=['Repeated historical development fixtures, not unseen independent correctness test.','Additional4 reserved for this feedback comparison but used by old TCN evaluation.','Whole-reverse examples are paired synthetic conditions, not14 independent videos.','Same Codex RGB review and separate arithmetic, not independent expert truth.','Detailed guide links4 expressions, boundary labels10 expressions; common100 words not dense feedback labels for100.','Direction gain does not resolve order or omission.'])
write(U/'RESULT.json',summary)
write(U/'VERIFICATION.json',dict(passed=True,count=len(checks),checks=checks,method='Separate stdlib-only saved-output hash/arithmetic/one-to-one audit by same Codex; no re-inference, no independent-human claim.',result_sha256=sha(U/'RESULT.json')))
write(U/'NEW_WARNING_EVIDENCE.json',warnings)
print(json.dumps(dict(passed=True,checks=len(checks),primary_changes=[x for x in changed if x['seed']==42],diagnostic=[x for x in diagnostic if x['seed']==42 and x['kind']=='all'],latency=summary['latency']),ensure_ascii=False,indent=2))
