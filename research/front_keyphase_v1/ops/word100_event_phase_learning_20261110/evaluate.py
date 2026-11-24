from phase_core import *
import copy,traceback,statistics
F=O/'word100_completion_fullpath_20261110';D=O/'word100_action_feedback_demo_20261108';POS=O/'word100_positive_direction_20261109'
sys.modules['common']=load('_ep_fullcommon',F/'common.py');rt=load('_ep_raw_helpers',F/'runtime.py')
groups=read(O/'word100_gru_semantic_finetune_20261109/FROZEN.json')['semantic_stage_groups'];gid={g:i for i,gg in enumerate(groups) for g in gg}
sys.path.insert(0,str(A));transforms=load('_ep_transforms',A/'evaluate.py')
def iou(a,b):return max(0,min(a[1],b[1])-max(a[0],b[0])+1)/(max(a[1],b[1])-min(a[0],b[0])+1)
def infer(net,x,ts,fh):
 g=old.geometry(x,ts);q=relation_features(x)
 with torch.inference_mode():ph,_,h=net(torch.tensor(g[None]),torch.tensor(q[None]),torch.tensor(fh[None]),torch.tensor([len(x)]))
 return ph.softmax(-1)[0].numpy(),q
def all_proposals(ph,q,linked,stage):
 ev=[];parts=[]
 for k in range(10):
  valid=q[:,k,-2:].max(-1)>.5;ee,pp=proposals(ph[:,k],valid,linked)
  for e in ee:
   a,b=e['a'],e['b'];pr=np.array([stage[a:b+1,g].sum(-1).mean() for g in groups]);e.update(axis=k//2,moving='L' if k%2 else 'R',semantic_group=int(pr.argmax()),semantic_confidence=float(pr.max()),observed=float(valid[a:b+1].mean()));ev.append(e)
  parts.extend(dict(p,axis=k//2,moving='L' if k%2 else 'R') for p in pp)
 return ev,parts
def frozen_forward(base,x,word):
 z=shared.normalize(x)
 with torch.inference_mode():o=base.shared(torch.tensor(z[None]),torch.tensor([len(z)]),torch.tensor([word]))
 return o['latent'][0].numpy(),o['stage'][0].softmax(-1).numpy()
def component_eval(net,base,focus):
 result=[]
 for variant in ['normal','upper','slow','fast','reversed_main','missing_2','missing_long']:
  for r in focus:
   x,ids,ts=transforms.transform(r,variant);fh,stage=frozen_forward(base,x,r['word']);ph,q=infer(net,x,ts,fh)
   for linked in [False,True]:
    events,parts=all_proposals(ph,q,linked,stage);used=set()
    # One-to-one overlap matching for evaluation only. Runtime never knows these anchors.
    candidates=[]
    for pi,p in enumerate(r['pairs']):
     if p['target']==1:continue
     aa=int(np.argmin(abs(ids-p['a'])));bb=int(np.argmin(abs(ids-p['b'])));a,b=sorted([aa,bb])
     for ei,e in enumerate(events):
      if e['axis']==p['axis'] and e['moving']==p.get('moving','R'):candidates.append((iou([a,b],[e['a'],e['b']]),pi,ei))
    matches={}
    for overlap,pi,ei in sorted(candidates,reverse=True):
     if overlap<.5 or pi in matches or ei in used:continue
     matches[pi]=(ei,overlap);used.add(ei)
    for pi,p in enumerate(r['pairs']):
     aa=int(np.argmin(abs(ids-p['a'])));bb=int(np.argmin(abs(ids-p['b'])));a,b=sorted([aa,bb]);k=p['axis']*2+(p.get('moving','R')=='L');known=q[a:b+1,k,-2:].max(-1)>.5;ok=bool(known.mean()>=.8 and known.sum()>=4)
     if p['target']==1:
      pred=int(ph[a:b+1,k].mean(0).argmax());result.append(dict(variant=variant,linked=linked,task='hold',clip_id=r['clip_id'],pair=pi,target=4,pred=pred if ok else -1,eligible=ok));continue
     match=matches.get(pi);e=events[match[0]] if match else None;prediction=-1;confidence=0.;geook=False
     if e:
      geo,geook,_=prior.geometry(x,ts,e['a'],e['b'],p['axis'],p.get('moving','R'),p.get('anchor','L'))
      with torch.inference_mode():pr=base.movement(torch.tensor(fh[None]),torch.tensor(geo[None]),torch.tensor([[0,e['a'],e['b']]])).softmax(-1)[0].numpy()
      prediction=int(pr.argmax()) if geook else -1;confidence=float(pr.max())
     result.append(dict(variant=variant,linked=linked,task='movement_interval',clip_id=r['clip_id'],pair=pi,axis=p['axis'],target=2-p['target'] if aa>bb else p['target'],pred=prediction,eligible=ok,matched=bool(match),overlap=float(match[1]) if match else 0.,boundary_error_frames=float((abs(e['a']-a)+abs(e['b']-b))/2) if e else None,confidence=confidence,confirmed_anchors=[a,b],predicted_span=[e['a'],e['b']] if e else None,partial_count=len(parts),all_proposals=len(events)))
 return result
def prepare_runtime():
 records=read(A/'ORDER_ONLY_COMPLETE.json')['rows'];raws={r['clip_id']+'_'+r['kind']:r for r in read(F/'REPLAY_INPUTS.json')['rows']};full={r['clip_id']:r for r in raws.values() if r['kind']=='full_normal'};edits={r['clip_id']+'_'+r['kind']:r for r in read(POS/'COMPARISON.json')['rows']};reverse={r['clip_id']:r for r in read(A/'FULLPATH_COMPLETE.json')['rows'] if r['model']=='learned_core_42' and r['kind']=='reversed_core_auto_order'};out=[]
 for r in records:
  key=r['clip_id']+'_'+r['kind'];baseline=read(r['result'])
  if key in edits:z=np.load(edits[key]['prediction']);x=z['x'];ts=z['timestamp_us'];fid=z['frame_id'];route='stored_edited_coordinates'
  else:
   src=full[r['clip_id']] if r['kind']=='reversed_core_auto_order' else raws[key];assert sha(src['path'])==src['sha256'];z=np.load(src['path']);raw={k:z[k].copy() for k in ['xy_px','score','timestamp_us','frame_id','source_width','source_height']}
   if r['kind']=='reversed_core_auto_order':
    lo,hi=reverse[r['clip_id']]['reversed_source_frame_span'];ii=np.flatnonzero((raw['frame_id']>=lo)&(raw['frame_id']<=hi))
    for k in ['xy_px','score']:raw[k][ii]=raw[k][ii[::-1]]
   ob=rt.consumer.quarantine(raw['xy_px'],raw['score'],raw['timestamp_us'],int(raw['source_width']),int(raw['source_height']));conv=rt.consumer.A.convert(ob['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']));x=conv['tcn280'];ts=conv['timestamp_us'];fid=conv['frame_id'];route='cached_raw_to_channels_only_no_TCN_or_DTW_rerun'
  assert fid.tolist()==baseline['frame_id'];out.append(dict(row=r,key=key,x=x,ts=ts,fid=fid,baseline=baseline,route=route))
 return out
def supplement(baseline,events,parts,x):
 out=copy.deepcopy(baseline);checks=[];defs=read(D/'CARDS.json')['words'][baseline['word']];av=shared.available(shared.normalize(x))
 for i in range(len(out['cards'])-1):
  ca,cb=out['cards'][i:i+2];sa=ca.get('query_indices');sb=cb.get('query_indices');check=dict(pair=[ca['id'],cb['id']],state='unknown')
  if not sa or not sb:check['reason']='no_original_window';checks.append(check);continue
  a,b=min(sa[0],sb[0]),max(sa[1],sb[1]);ga=gid[defs[i]['groups'][-1][-1]];gb=gid[defs[i+1]['groups'][0][0]];aa=[];bb=[]
  for e in events:
   if e['semantic_confidence']<.65 or e['b']<a or e['a']>b:continue
   if e['semantic_group']==ga:aa.append([e['a'],e['b']])
   if e['semantic_group']==gb:bb.append([e['a'],e['b']])
  hands=set(ca['hands']+cb['hands']);cols=[1 if h=='R' else 4 for h in hands];ok=bool(av[a:b+1,cols].all(1).mean()>=.8 and not ca.get('observation_quarantine_overlap') and not cb.get('observation_quarantine_overlap'))
  if aa and bb and ok:
   if max(v[1] for v in bb)<min(v[0] for v in aa):check['state']='contradicted'
   elif max(v[1] for v in aa)<min(v[0] for v in bb):check['state']='supported'
  check.update(span=[a,b],eligible=ok,first_event_spans=aa,second_event_spans=bb,all_candidates_preserved=True);checks.append(check)
  if check['state']=='contradicted' and cb['display_status'] in ['similar','difference']:
   cb.update(status='difference_candidate',display_status='difference',message='관측된 주요 동작의 순서가 참고와 다르게 보여요. 표시된 구간을 확인해 보세요.',educational_wrong=False);cb['candidate_message']=cb['message'];cb.setdefault('difference_components',[])
   if 'phase_order' not in cb['difference_components']:cb['difference_components'].append('phase_order')
 out['phase_events']=events;out['phase_partials']=parts;out['phase_order_checks']=checks;out['phase_scope']='Observed component phases; not exact human boundary or semantic completion truth.';return out
def compact(d):
 ds=[c['display_status'] for c in d['cards']];return dict(display=ds,warning='difference' in ds,similar_no_warning=all(c['display_status']=='similar' for c in d['cards'] if c.get('required',True)) and 'difference' not in ds,motion=[c.get('motion_evidence') for c in d['cards']],shape=[c.get('shape_evidence') for c in d['cards']])
def main():
 assert (U/'TRAINING_COMPLETE.json').exists() and not (U/'EVALUATION_COMPLETE.json').exists();cfg=read(U/'FROZEN.json');protocol=read(U/'EVAL_FROZEN.json')
 for p,h in protocol['pins'].items():assert sha(p)==h,p
 torch.set_num_threads(4);torch.set_num_interop_threads(1);dev=[r for r in rows() if r['role']=='development'];focus=[r for r in dev if r['word']>=0];base=prior.Network('coupled',read(P/'NORMALIZATION.json'));base.load_state_dict(torch.load(P/'runs/coupled_42/model.pt',weights_only=True,map_location='cpu'));base.eval();runtime=prepare_runtime()
 word_indices={r['word_id']:r['word'] for r in focus}
 for c in runtime:c['fh'],c['stage']=frozen_forward(base,c['x'],word_indices[c['baseline']['word']])
 allrun=[];summaries={}
 for seed in cfg['seeds']:
  net=PhaseNet();net.load_state_dict(torch.load(U/'runs'/str(seed)/'model.pt',weights_only=True,map_location='cpu'));net.eval();comp=component_eval(net,base,focus);write(U/'scores'/f'{seed}.json',comp);wordrows=[]
  with torch.inference_mode():
   for lo in range(0,len(dev),24):
    ss=dev[lo:lo+24];b=shared.batch(ss);fo=base.shared(b['x'],b['lengths'],b['word'])
    for j,r in enumerate(ss):
     x=prior.physical(r['x_full']);r['g_full']=old.geometry(x,r['timestamp_us']);r['q_full']=relation_features(x);r['frozen_h_full']=fo['latent'][j,:len(x)].numpy();r['phase_y']=np.zeros((len(x),10,5),np.float32);r['phase_w']=np.zeros((len(x),10),np.float32)
    g,q,fh,l,_,_=batch(ss);_,pr,_=net(g,q,fh,l);wordrows.extend(dict(clip_id=r['clip_id'],target=r['class_index'],pred=int(p)) for r,p in zip(ss,pr.argmax(-1)))
  write(U/'scores'/f'{seed}_words.json',wordrows)
  for c in runtime:
   t=time.perf_counter();ph,q=infer(net,c['x'],c['ts'],c['fh']);events,parts=all_proposals(ph,q,True,c['stage']);out=supplement(c['baseline'],events,parts,c['x']);elapsed=time.perf_counter()-t;path=U/'fullpath'/str(seed)/c['key']/'after.json';write(path,out)
   assert out['tcn_top3']==c['baseline']['tcn_top3'] and out['path']==c['baseline']['path'];assert compact(out)['motion']==compact(c['baseline'])['motion'];assert compact(out)['shape']==compact(c['baseline'])['shape']
   allrun.append(dict(seed=seed,clip_id=c['row']['clip_id'],kind=c['row']['kind'],before={k:v for k,v in compact(c['baseline']).items() if k not in ['motion','shape']},after={k:v for k,v in compact(out).items() if k not in ['motion','shape']},result=str(path),sha256=sha(path),phase_seconds=elapsed,frames=len(c['x']),model_frames=len(ph),old_shape_direction_identical=True,tcn_dtw_preserved=True,events=len(events),partials=len(parts)))
  su={}
  for var in sorted({r['variant'] for r in comp}):
   su[var]={}
   for linked in [False,True]:
    mm=[r for r in comp if r['variant']==var and r['linked']==linked and r['task']=='movement_interval'];hh=[r for r in comp if r['variant']==var and r['linked']==linked and r['task']=='hold'];ma=[r for r in mm if r['matched']];su[var][str(linked)]=dict(moving_intervals=len(mm),matched=len(ma),direction_correct=sum(r['pred']==r['target'] for r in mm),eligible=sum(r['eligible'] for r in mm),mean_overlap=float(np.mean([r['overlap'] for r in mm])),mean_boundary_error_matched=float(np.mean([r['boundary_error_frames'] for r in ma])) if ma else None,hold_intervals=len(hh),hold_correct=sum(r['pred']==r['target'] for r in hh))
  summaries[str(seed)]=dict(component=su,aux_word_correct=sum(r['target']==r['pred'] for r in wordrows),aux_word_total=len(wordrows));event('evaluated',seed=seed,normal=su['normal'])
 write(U/'FULLPATH_COMPLETE.json',dict(rows=allrun,no_private=True,no_default_change=True));write(U/'EVALUATION_COMPLETE.json',dict(models=summaries,scope='HistoricalDEV:21moving+11hold intervals; boundary anchors not dense semantic phase gold.78automatic savedbaselines×3seeds. No oldTCN/DTW benchmark rerun.'))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'EVAL_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
