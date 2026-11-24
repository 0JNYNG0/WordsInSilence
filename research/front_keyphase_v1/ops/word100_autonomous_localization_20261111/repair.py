from localization_core import *
import inspect
assert not (U/'boundary_integrity_repair'/'EVALUATION_COMPLETE.json').exists()
R=U/'boundary_integrity_repair';R.mkdir(exist_ok=True)
write(R/'REASON.json',dict(cause='Local boundary refinement could extend across a predicted direction transition, overlapping opposite-direction event intervals. Contract violation found after first scoring.',fix='Clamp boundary refinement to its own predicted directional run; retain all original first-run outputs, no weights/thresholds/labels changed.',evaluation_status='Post-score implementation repair; results are development replays, not a second fresh test.',no_score_based_threshold_tuning=True))
src=inspect.getsource(infer_local)
src=src.replace("la=max(a,lo-rad);hb=min(b,hi+rad)","la=lo;hb=hi")
# Changed decoding only. Saved neural probabilities are reused, with no new
# localizer/TCN/DTW inference and no completed baseline re-evaluation.
prefix=src[:src.index(' labels=sp.argmax')]
src='def decode_saved(sp,bp,valid,cfg):\n'+src[src.index(' labels=sp.argmax'):]
ns=dict(bc=bc,np=np);exec(src,ns);decode_saved=ns['decode_saved']
(R/'decoder.py').write_text('import numpy as np\n'+src,'utf-8')
sys.modules['motion_core']=mc;me=load('_repair_old_helpers',M/'evaluate.py')
from guide_binding import attach
cfg=read(U/'FROZEN.json');torch.set_num_threads(4);torch.set_num_interop_threads(1);base=prior.Network('coupled',read(old.P/'NORMALIZATION.json'));base.load_state_dict(torch.load(old.P/'runs/coupled_42/model.pt',weights_only=True,map_location='cpu'));base.eval()
orig=read(U/'DECISION_CACHE.json');pins={r['key']:r for r in orig};runtime={r['key']:r for r in lc.be.ev.prepare_runtime()};oldrows={r['clip_id']:r for r in old.rows()};raws={r['clip_id']:r for folder in [J,V,U] for r in read(folder/'EXTRACTION_COMPLETE.json')['rows']};meta={}
for r in read(U/'EVENT_RESULTS.json'):
 if r['seed']==42:meta[r['clip_id']+'_'+r['kind']]=r
fr0=read(U/'FULLPATH_COMPLETE.json')['rows'];cases=[]
for p in orig:
 key=p['key'];clip=key[:key.index('_F')+2];kind=key[len(clip)+1:];bp=read(p['baseline_file']);before=bp['baseline'] if 'baseline' in bp and isinstance(bp['baseline'],dict) else bp
 if key in runtime:c=runtime[key];group='historical_replay';fh,_=lc.be.ev.frozen_forward(base,c['x'],before.get('word',-1) if isinstance(before.get('word'),int) else oldrows[clip]['word']);c.update(group=group)
 else:
  m=meta[key];group=m['group']
  if group=='prior13_replay':rr=oldrows[clip];x=prior.physical(rr['x_full']);x=x[::-1].copy() if kind=='whole_reverse' else x;ts=rr['timestamp_us']
  else:x,ts=me.convert(raws[clip]['raw'],kind=='whole_reverse')
  ts=(np.asarray(ts)/({'speed075':.75,'speed125':1.25}.get(kind,1))).astype(np.int64);fh,_=lc.be.ev.frozen_forward(base,x,oldrows.get(clip,{}).get('word',-1));c=dict(x=x,ts=ts,group=group)
 c.update(key=key,clip_id=clip,kind=kind,fh=fh,baseline=before,baseline_file=p['baseline_file'],old_cache={tuple(d['key']):d['value'] for d in p['old_direction']});cases.append(c)
profiles=read(U/'GUIDE_ENDPOINT_PROFILES.json')['profiles'];results=[];events=[];frs=[];changed=0;observations=[];previous_dirs=read(U/'DIRECTION_RESULTS.json');previous_events=read(U/'EVENT_RESULTS.json');previous_frs=read(M/'FULLPATH_COMPLETE.json')['rows'];motionfiles={(r['seed'],r['clip_id'],r['kind']):r for r in previous_frs if r['mode']=='motion_only'}
def unique(es):return list({(e['a'],e['b'],e['axis'],e['moving']):e for e in es if e['b']-e['a']>=3}.values())
def direction(c,e,motion,anchor='L'):
 key=(e['a'],e['b'],e['axis'],e['moving'],anchor)
 if key not in c['old_cache']:c['old_cache'][key]=jc.integration.direction(base,c['x'],c['ts'],c['fh'],*key[:3],key[3],key[4])
 od=c['old_cache'][key];nd=mc.direction(motion,c['x'],c['ts'],*key[:3],key[3],key[4]);return od,mc.fused(od,nd)
for seed in cfg['seeds']:
 motion=mc.MotionNet();motion.load_state_dict(torch.load(M/'runs'/str(seed)/'model.pt',weights_only=True,map_location='cpu'),strict=False);motion.eval()
 for c in cases:
  z=np.load(U/'observations'/str(seed)/f"{c['key']}.npz");oldobs=read(U/'observations'/str(seed)/f"{c['key']}.json");ob=decode_saved(z['state'],z['boundary'],z['valid'],cfg);es=unique(ob['events']+ob['partials']);different=es!=oldobs['events'];changed+=different;mt=dict(seed=seed,group=c['group'],clip_id=c['clip_id'],kind=c['kind']);p=R/'observations'/str(seed)/f"{c['key']}.json";write(p,dict(events=es,source_probabilities=str(U/'observations'/str(seed)/f"{c['key']}.npz"),changed=different));observations.append(dict(mt,file=str(p),changed=different))
  assert not any(a['axis']==b['axis'] and a['moving']==b['moving'] and max(a['a'],b['a'])<min(a['b'],b['b']) for i,a in enumerate(es) for b in es[i+1:])
  if c['group']=='historical_replay':
   for mode in ['localization_only','combined']:
    oldresult=next(r for r in fr0 if r['seed']==seed and r['clip_id']==c['clip_id'] and r['kind']==c['kind'] and r['mode']==mode)
    if not different:frs.append(dict(oldresult,reused_first_result=True));continue
    mr=motionfiles[seed,c['clip_id'],c['kind']];before=copy.deepcopy(c['baseline'])
    if mode=='combined':before['cards']=read(mr['result'])['cards']
    t=time.perf_counter();out=attach(before,es,c['x'],c['ts'],c['fh'],base,motion,profiles,mode=='combined',c['old_cache']);seconds=time.perf_counter()-t;p=R/'feedback'/str(seed)/mode/f"{c['key']}.json";protected=read(oldresult['result'])['protected'];write(p,dict(cards=out['cards'],bindings=out['localization_bindings'],ledger=out['localization_ledger'],protected=protected,baseline=c['baseline_file'],baseline_sha256=sha(c['baseline_file'])));frs.append(dict(oldresult,result=str(p),sha256=sha(p),after=le.compact(out),binding_seconds=seconds,reused_first_result=False))
  else:
   rs=[r for r in previous_dirs if r['seed']==seed and r['clip_id']==c['clip_id'] and r['kind']==c['kind']];ee=next(r for r in previous_events if r['seed']==seed and r['clip_id']==c['clip_id'] and r['kind']==c['kind'])
   if not different:results.extend(rs);events.append(ee);continue
   if c['group']=='prior13_replay':selected=read(M/'fresh_baseline'/f"{c['key']}.json")['selected']
   elif c['group']=='reserved14':selected=read(U/'reserved_baseline'/f"{c['key']}.json")['selected']
   else:
    obs=next(r for r in read(V/'OBSERVATIONS.json')['rows'] if r['seed']==42 and r['clip_id']==c['clip_id'] and r['kind']==c['kind']);selected=read(obs['file'])['selected']
   be=unique(c['baseline']['phase_events']+c['baseline']['phase_partials']+selected);union=unique(be+es);matches={name:lc.be.match_events(v,[r['truth'] for r in rs]) for name,v in [('localized',es),('union',union)]}
   for i,r in enumerate(rs):
    r=copy.deepcopy(r)
    for name,collection in [('localized',es),('union',union)]:
     m=matches[name].get(i);v=dict(matched=m is not None,event=None,overlap=0.)
     if m:
      e=collection[m[0]];oldv=r['automatic'][name]
      if oldv['matched'] and all(oldv['event'][k]==e[k] for k in ['a','b','axis','moving']):v=oldv
      else:
       od,nd=direction(c,e,motion,r['truth']['anchor']);v.update(event=e,overlap=float(m[1]),before=od,after=nd,onset_error=min(abs(e['a']-z) for z in r['truth']['onset']),offset_error=min(abs(e['b']-z) for z in r['truth']['offset']))
     r['automatic'][name]=v
    results.append(r)
   # Review interval coverage separately; unused/unknown regions are not false positives.
   labelpath=M/'EVALUATION_LABELS.json' if c['group']=='prior13_replay' else U/'EVALUATION_LABELS.json' if c['group']=='reserved14' else (J if any(a['clip_id']==c['clip_id'] for a in read(J/'EVALUATION_LABELS.json')['rows']) else V)/'EVALUATION_LABELS.json'
   non=[]
   for a in read(labelpath)['rows']:
    if a['clip_id']==c['clip_id']:
     for lo,hi in a['holds']+a.get('background',[]):non.append(dict(axis=a['axis'],moving=a['moving'],span=[len(c['x'])-1-hi,len(c['x'])-1-lo] if c['kind']=='whole_reverse' else [lo,hi]))
   false=sum(any(e['axis']==a['axis'] and e['moving']==a['moving'] and a['span'][0]<=e['a']<=e['b']<=a['span'][1] for a in non) for e in es)
   events.append(dict(ee,new_events=len(es),union_events=len(union),new_matched=len(matches['localized']),union_matched=len(matches['union']),known_nonmotion_new=false,events=es))
 event('boundary_integrity_repair_evaluated',seed=seed,changed_cases=changed)
write(R/'DIRECTION_RESULTS.json',results);write(R/'EVENT_RESULTS.json',events);write(R/'FULLPATH_COMPLETE.json',dict(rows=frs));write(R/'HOLD_RESULTS.json',read(U/'HOLD_RESULTS.json'));write(R/'OBSERVATIONS.json',dict(rows=observations));write(R/'EVALUATION_COMPLETE.json',dict(conditions=len(cases),seeds=cfg['seeds'],changed_case_seed_pairs=changed,neural_localizer_reruns=0,old_TCN_DTW_runs=0,weights_changed=False,labels_changed=False,overlapping_direction_events_after_repair=0));write(U/'CURRENT_EXECUTION.json',dict(status='REPAIR_EVALUATED_PENDING_VERIFICATION',pid=None))
