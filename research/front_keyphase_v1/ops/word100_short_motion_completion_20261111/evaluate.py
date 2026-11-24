from motion_core import *
import traceback,os
from collections import defaultdict
L,B=jc.L,jc.B
def jhash(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def dk(a,b,k,m='R',anchor='L'):return (int(a),int(b),int(k),m,anchor)
def old_lookup(c,a,b,k,m='R',anchor='L'):
 key=dk(a,b,k,m,anchor)
 if key not in c['old_direction']:
  assert c['group']=='reserved_id13',('missing frozen old decision',c['key'],key)
  c['old_direction'][key]=jc.integration.direction(BASE,c['x'],c['ts'],c['fh'],a,b,k,m,anchor)
 return c['old_direction'][key]
def save_old(c,d,anchor='L'):
 c['old_direction'][dk(*d['span'],d['axis'],d['moving'],anchor)]=d
def convert(raw,reverse=False):
 z=np.load(raw);idx=np.arange(len(z['frame_id']));idx=idx[::-1] if reverse else idx;ts=z['timestamp_us'];rt=lc.be.ev.rt
 ob=rt.consumer.quarantine(z['xy_px'][idx],z['score'][idx],ts,int(z['source_width']),int(z['source_height']))
 cv=rt.consumer.A.convert(ob['xy_px'],z['score'][idx],ts,z['frame_id'],int(z['source_width']),int(z['source_height']))
 return cv['tcn280'],ts
def truths(anns,clip,n,reverse):
 tt=[];hh=[]
 for r in anns:
  if r['clip_id']!=clip:continue
  for e in r['events']:
   a,b=e['onset'],e['offset'];d=e['direction']
   if reverse:a,b,d=[n-1-b[1],n-1-b[0]],[n-1-a[1],n-1-a[0]],2-d
   tt.append(dict(axis=r['axis'],moving=r['moving'],anchor=r['anchor'],onset=a,offset=b,span=[a[0],b[1]],direction=d))
  for a,b in r['holds']:hh.append(dict(axis=r['axis'],moving=r['moving'],anchor=r['anchor'],span=[n-1-b,n-1-a] if reverse else [a,b],direction=1))
 return tt,hh
def prepare():
 global BASE
 BASE=prior.Network('coupled',read(old.P/'NORMALIZATION.json'));BASE.load_state_dict(torch.load(old.P/'runs/coupled_42/model.pt',map_location='cpu',weights_only=True));BASE.eval()
 rows=old.rows();wm={r['word_id']:r['word'] for r in rows if r['word']>=0};byclip={r['clip_id']:r for r in rows};out=[]
 observations={(r['clip_id'],r['kind']):r for r in read(V/'OBSERVATIONS.json')['rows'] if r['seed']==42}
 histrows={(r['clip_id'],r['kind']):r for r in read(V/'FULLPATH_COMPLETE.json')['rows'] if r['seed']==42 and r['mode']=='interval_only'}
 frozen={(r['clip_id'],r['kind']):r for r in read(L/'GUARDED_COMPLETE.json')['rows'] if r['seed']==42}
 for c in lc.be.ev.prepare_runtime():
  clip,kind=c['row']['clip_id'],c['row']['kind'];r=frozen[clip,kind];assert sha(r['result'])==r['sha256'];before=read(r['result']);v=histrows[clip,kind];assert sha(v['result'])==v['sha256'];iv=read(v['result']);o=observations[clip,kind];assert sha(o['file'])==o['sha256'];ob=read(o['file'])
  c.update(clip_id=clip,kind=kind,group='historical_replay',baseline=before,baseline_file=r['result'],interval_result=iv,interval_summary=v['after'],selected=ob['selected'],old_direction={},truth=[],holds=[])
  for bind in iv['bindings']:
   for d in bind['movements']:save_old(c,d)
  out.append(c)
 diagnostics=read(V/'DIAGNOSTIC.json')
 for group,folder in [('additional_replay',J),('reserved_id12',V)]:
  anns=read(folder/'EVALUATION_LABELS.json')['rows']
  for r in read(folder/'EXTRACTION_COMPLETE.json')['rows']:
   assert sha(r['raw'])==r['raw_sha256']
   for kind in ['normal','whole_reverse']:
    x,ts=convert(r['raw'],kind=='whole_reverse');tt,hh=truths(anns,r['clip_id'],len(x),kind=='whole_reverse');p=folder/'additional_baseline'/f"{r['clip_id']}_{kind}.json";o=observations[r['clip_id'],kind];assert sha(o['file'])==o['sha256'];ob=read(o['file'])
    c=dict(key=r['clip_id']+'_'+kind,clip_id=r['clip_id'],kind=kind,group=group,x=x,ts=ts,truth=tt,holds=hh,baseline=read(p),baseline_file=str(p),selected=ob['selected'],old_direction={})
    for d in diagnostics:
     if d['seed']!=42 or d['clip_id']!=r['clip_id'] or d['kind']!=kind:continue
     save_old(c,d['manual_direction'],d['truth']['anchor'])
     for z in d['automatic'].values():
      if z['direction'] is not None:save_old(c,z['direction'],d['truth']['anchor'])
    # Holds were not scored by the old direction head in the prior experiment.
    # New baseline hold scoring is allowed; previously completed motion scores remain frozen.
    fh,_=lc.be.ev.frozen_forward(BASE,x,wm.get(r['word_id'],-1))
    for h in hh:
     a,b=h['span'];key=dk(a,b,h['axis'],h['moving'],h['anchor'])
     if key not in c['old_direction']:c['old_direction'][key]=jc.integration.direction(BASE,x,ts,fh,a,b,h['axis'],h['moving'],h['anchor'])
    out.append(c)
 BN=bc.BoundaryNet();BN.load_state_dict(torch.load(B/'runs/42/model.pt',map_location='cpu',weights_only=True));BN.eval();LN=lc.LinkNet();LN.load_state_dict(torch.load(L/'runs/42/model.pt',map_location='cpu',weights_only=True));LN.eval()
 net=jc.JointNet();net.load_state_dict(torch.load(J/'runs/42/model.pt',map_location='cpu',weights_only=True));ck=torch.load(V/'runs/42/model.pt',map_location='cpu',weights_only=True);net.quality.load_state_dict(ck['quality']);net.pair.load_state_dict(ck['pair']);net.eval();anns=read(U/'EVALUATION_LABELS.json')['rows']
 for r in read(U/'RESERVED_PUBLIC.json')['rows']:
  rr=byclip[r['clip_id']];assert r['source_sha256']==rr['source_sha256'] and sha(r['source'])==r['source_sha256'];x0=prior.physical(rr['x_full']);ts0=np.asarray(rr['timestamp_us']);assert len(x0)==r['original_frames'];n=len(x0)
  for kind in ['normal','whole_reverse','speed075','speed125']:
   rev=kind=='whole_reverse';x=x0[::-1].copy() if rev else x0;ts=ts0.copy();ts=(ts/({'speed075':.75,'speed125':1.25}.get(kind,1))).astype(np.int64);fh,stage=lc.be.ev.frozen_forward(BASE,x,wm.get(r['word_id'],-1));bp,q=lc.be.infer(BN,x,ts,fh);ee,pp=lc.be.propose(bp,q,stage);lp=le.infer(LN,x,ts);bb=dict(phase_events=ee,phase_partials=pp,guide_link_probabilities=lp.tolist());ob=jc.infer(net,x,ts,fh,ee+pp);tt,hh=truths(anns,r['clip_id'],n,rev)
   c=dict(key=r['clip_id']+'_'+kind,clip_id=r['clip_id'],kind=kind,group='reserved_id13',x=x,ts=ts,fh=fh,baseline=bb,selected=ob['selected'],truth=tt,holds=hh,old_direction={});p=U/'fresh_baseline'/f"{c['key']}.json";write(p,dict(baseline=bb,selected=ob['selected'],source_sha256=r['source_sha256']));c['baseline_file']=str(p);out.append(c)
 event('evaluation_inputs_ready',conditions=len(out));return out
def unique(es):return list({(e['a'],e['b'],e['axis'],e['moving']):e for e in es if e['b']-e['a']>=3}.values())
def supplement(c,net,combined):
 before=c['baseline'];out=copy.deepcopy(before);contracts=jc.ep.family_contracts()[before['word']];binds=[];oldkeys={(e['a'],e['b'],e['axis'],e['moving']) for e in before['phase_events']+before['phase_partials']};cache={}
 for card,spec,ib in zip(out['cards'],contracts,c['interval_result']['bindings']):
  b=copy.deepcopy(ib);mov=[]
  for d in ib['movements']:
   a,z=d['span'];key=(a,z,d['axis'],d['moving'])
   if not combined and key not in oldkeys:continue
   if key not in cache:cache[key]=direction(net,c['x'],c['ts'],a,z,d['axis'],d['moving'])
   nd=fused(d,cache[key]);nd['interval_complete_candidate']=nd.pop('complete',False);nd['semantic_complete']=False;mov.append(nd)
  signs={d['prediction']-1 for d in mov if d['eligible'] and d['confidence']>=.65 and d['prediction']!=1};expected=spec['sign'];state='not_specified' if expected is None else 'unknown'
  if expected is not None and len(signs)==1:state='supported' if signs=={expected} else 'contradicted'
  elif expected is not None and len(signs)>1:state='mixed'
  b.update(movements=mov,state=state);card['short_motion_binding']=b;binds.append(b)
  if state=='contradicted' and not b['conflict'] and not card.get('observation_quarantine_overlap'):jc.integration.add_warning(card,'short_motion_direction','표시된 움직임이 참고와 반대 방향으로 보여요. 이동 방향을 확인해 보세요.')
 for order in c['interval_result']['order']:
  if order['state']=='contradicted' and not order['conflict']:
   ids=order['pair'];cards=[v for v in out['cards'] if v['id'] in ids]
   if not any(v.get('observation_quarantine_overlap') for v in cards):jc.integration.add_warning(cards[-1],'joint_order','표시된 행동의 순서를 확인해 보세요.')
 for a,b in zip(out['cards'],before['cards']):assert a.get('shape_evidence')==b.get('shape_evidence') and a.get('motion_evidence')==b.get('motion_evidence')
 out['short_motion_bindings']=binds;return out
def main():
 assert not (U/'EVALUATION_COMPLETE.json').exists()
 for p,h in {**read(U/'FROZEN.json')['protections'],**read(U/'FROZEN.json')['pins'],**read(U/'CODE_PINS.json'),**read(U/'EVAL_PINS.json')}.items():assert sha(p)==h,p
 torch.set_num_threads(4);torch.set_num_interop_threads(1);write(U/'CURRENT_EXECUTION.json',dict(status='EVALUATING',pid=os.getpid()));cases=prepare();drs=[];hrs=[];frs=[];crs=[];cacheold=[]
 for seed in [42,43,44]:
  net=MotionNet();net.load_state_dict(torch.load(U/'runs'/str(seed)/'model.pt',map_location='cpu',weights_only=True),strict=False);net.eval()
  for c in cases:
   meta={k:c[k] for k in ['group','clip_id','kind']};meta['seed']=seed
   if c['group']=='historical_replay':
    for mode in ['motion_only','combined']:
     start=time.perf_counter();after=supplement(c,net,mode=='combined');elapsed=time.perf_counter()-start;p=U/'feedback'/str(seed)/mode/f"{c['key']}.json";protected={k:jhash(c['baseline'][k]) for k in ['tcn_top3','path','frame_id','phase_events','phase_partials']};write(p,dict(baseline=c['baseline_file'],baseline_sha256=sha(c['baseline_file']),cards=after['cards'],bindings=after['short_motion_bindings'],protected=protected));frs.append(dict(meta,mode=mode,before=le.compact(c['baseline']),interval_only=c['interval_summary'],after=le.compact(after),result=str(p),sha256=sha(p),seconds=elapsed))
   else:
    choices={'baseline':unique(c['baseline']['phase_events']+c['baseline']['phase_partials']),'combined':unique(c['baseline']['phase_events']+c['baseline']['phase_partials']+c['selected'])};matches={k:lc.be.match_events(es,c['truth']) for k,es in choices.items()}
    for i,t in enumerate(c['truth']):
     a,b=t['span'];od=old_lookup(c,a,b,t['axis'],t['moving'],t['anchor']);nd=direction(net,c['x'],c['ts'],a,b,t['axis'],t['moving'],t['anchor']);d=dict(meta,truth=t,manual=dict(before=od,raw_new=nd,after=fused(od,nd)),automatic={})
     for mode,es in choices.items():
      m=matches[mode].get(i);e=es[m[0]] if m else None;v=dict(matched=bool(e),overlap=float(m[1]) if m else 0.,event=e)
      if e:
       od=old_lookup(c,e['a'],e['b'],t['axis'],t['moving'],t['anchor']);nd=direction(net,c['x'],c['ts'],e['a'],e['b'],t['axis'],t['moving'],t['anchor']);v.update(before=od,raw_new=nd,after=fused(od,nd))
      d['automatic'][mode]=v
     drs.append(d)
    for h in c['holds']:
     a,b=h['span'];od=old_lookup(c,a,b,h['axis'],h['moving'],h['anchor']);nd=direction(net,c['x'],c['ts'],a,b,h['axis'],h['moving'],h['anchor']);hrs.append(dict(meta,truth=h,before=od,raw_new=nd,after=fused(od,nd)))
    for e in c['selected']:
     # Completion is evaluated on every selected interval, including wrong/unmatched ones.
     guard=completion_guard(net,e,c['x'],c['ts']);same=[t for t in c['truth'] if t['axis']==e['axis'] and t['moving']==e['moving'] and lc.be.ev.iou(t['span'],[e['a'],e['b']])>=.5];timing='unmatched'
     if same:
      t=max(same,key=lambda t:lc.be.ev.iou(t['span'],[e['a'],e['b']]));timing='early' if e['b']<t['offset'][0] else 'end_in_range' if e['b']<=t['offset'][1] else 'late'
     crs.append(dict(meta,event=e,offset_relation=timing,guard=guard))
  event('new_motion_evaluated',seed=seed,conditions=len(cases))
 for c in cases:cacheold.append(dict(key=c['key'],group=c['group'],baseline_file=c['baseline_file'],baseline_sha256=sha(c['baseline_file']),old_direction=[dict(key=list(k),value=v) for k,v in c['old_direction'].items()]))
 write(U/'DIRECTION_RESULTS.json',drs);write(U/'HOLD_RESULTS.json',hrs);write(U/'FULLPATH_COMPLETE.json',dict(rows=frs));write(U/'COMPLETION_RESULTS.json',crs);write(U/'BASELINE_PINS.json',cacheold);write(U/'EVALUATION_COMPLETE.json',dict(conditions=len(cases),seeds=[42,43,44],historical_conditions=78,prior_public_originals=16,fresh_fine_originals=6,new_fine_labels_not_global_unseen=True,old_motion_results_reused=True,old_hold_and_fresh_baselines_first_scoring=True));write(U/'CURRENT_EXECUTION.json',dict(status='EVALUATED_PENDING_VERIFICATION',pid=None))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'EVAL_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
