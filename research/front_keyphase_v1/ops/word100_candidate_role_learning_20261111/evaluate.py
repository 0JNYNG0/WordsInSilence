from role_core import *
from integration import bind,unique
import os,traceback

def main():
 assert (U/'TRAINING_COMPLETE.json').exists() and not (U/'EVALUATION_COMPLETE.json').exists()
 torch.set_num_threads(4);torch.set_num_interop_threads(1);cfg=read(U/'FROZEN.json')
 for p,h in cfg['pins'].items():assert sha(p)==h,p
 if (U/'EVAL_PINS.json').exists() and not (U/'FIRST_EVAL_PINS.json').exists():write(U/'FIRST_EVAL_PINS.json',read(U/'EVAL_PINS.json'))
 pins={str(p):sha(p) for p in [U/'role_core.py',U/'integration.py',U/'evaluate.py',U/'FROZEN.json',U/'ROLE_LABELS.json']};write(U/'EVAL_PINS.json',dict(pins=pins,conditions=78,modes=['role_only','quality_only','combined'],baseline='Stored previous final-safe cards; no old TCN/DTW re-execution',post_score_tuning=False,omission_guard=dict(coverage=.9,max_gap_us=100000,max_target_prob=.35)))
 frozen=frozen_model();oldadapter=RoleAdapter();base=prior.Network('coupled',read(old.P/'NORMALIZATION.json'));base.load_state_dict(torch.load(old.P/'runs/coupled_42/model.pt',weights_only=True,map_location='cpu'));base.eval();base.requires_grad_(False)
 rows=old.rows();byid={r['clip_id']:r for r in rows};baselines={(r['seed'],r['clip_id'],r['kind']):r for r in read(A/'guide_warning_gate/VERIFIED_FULLPATH.json')['rows']};original={r['key']:r for r in read(A/'DECISION_CACHE.json')};repair={(r['seed'],r['clip_id'],r['kind']):r for r in read(A/'boundary_integrity_repair/FULLPATH_COMPLETE.json')['rows'] if r['mode']=='combined'};cases=lc.be.ev.prepare_runtime();assert len(cases)==78
 inputs=[];cached={r['key']:r for r in read(U/'EVALUATION_INPUTS.json')['rows']} if (U/'EVALUATION_INPUTS.json').exists() else {}
 for c in cases:
  key=c['key'];clip=c['row']['clip_id'];p=original[key];assert sha(p['baseline_file'])==p['baseline_sha256'];before=read(p['baseline_file']);c['original']=before
  if key in cached:
   r=cached[key];assert sha(r['path'])==r['sha256'];z=np.load(r['path']);assert np.array_equal(z['x'],c['x']);c['fh']=z['fh'];c['ob']=dict(z=z['z'],h=torch.tensor(z['h']),q=torch.tensor(z['q']),fh=torch.tensor(z['fh'][None]));inputs.append(r);continue
  c['fh'],_=lc.be.ev.frozen_forward(base,c['x'],byid[clip]['word']);c['ob']=features(frozen,c['x'],c['ts'],c['fh']);cp=U/'evaluation_inputs'/f'{key}.npz';cp.parent.mkdir(exist_ok=True)
  np.savez_compressed(cp,x=c['x'],ts=c['ts'],fh=c['fh'],z=c['ob']['z'],h=c['ob']['h'].numpy(),q=c['ob']['q'].numpy());inputs.append(dict(key=key,clip_id=clip,kind=c['row']['kind'],source_sha256=byid[clip]['source_sha256'],path=str(cp),sha256=sha(cp),baseline=p['baseline_file'],baseline_sha256=p['baseline_sha256']))
 write(U/'EVALUATION_INPUTS.json',dict(rows=inputs,training_source_sha_disjoint=not {r['source_sha256'] for r in inputs}&{r['source_sha256'] for r in read(U/'TRAIN_INPUTS_COMPLETE.json')['rows']},evaluation_status='Previously exposed development replay; not independent unseen test'))
 gold={r['clip_id']:r for r in read(L/'LINK_LABELS.json')['rows']};records=[];frames=[];candidate_scores=[];write(U/'CURRENT_EXECUTION.json',dict(status='EVALUATING',pid=os.getpid()))
 for seed in cfg['seeds']:
  net=RoleAdapter(seed);net.load_state_dict(torch.load(U/'runs'/str(seed)/'model.pt',weights_only=True,map_location='cpu'));net.eval();motion=mc.MotionNet();motion.load_state_dict(torch.load(M/'runs'/str(seed)/'model.pt',weights_only=True,map_location='cpu'),strict=False);motion.eval()
  for c in cases:
   clip,kind=c['row']['clip_id'],c['row']['kind'];ref=baselines[seed,clip,kind];assert sha(ref['result'])==ref['sha256'];prior_safe=read(ref['result']);before=copy.deepcopy(c['original']);before['cards']=prior_safe['cards'];rp0=jc.legacy_role(c['original']);rp0=np.c_[rp0,np.zeros((len(rp0),3))];t0=time.perf_counter()
   with torch.inference_mode():logits=net(torch.tensor(c['ob']['z']));logits[:,10]=-1e4;rp=logits.softmax(-1).numpy()
   role_seconds=time.perf_counter()-t0;new_events=read(A/'boundary_integrity_repair/observations'/str(seed)/f"{c['key']}.json")['events'];legacy=before['phase_events']+before['phase_partials'];all_events=unique(legacy+new_events);cache={tuple(d['key']):d['value'] for d in original[c['key']]['old_direction']};rd=read(repair[seed,clip,kind]['result'])
   # Cached fused direction results are reused for unchanged candidate spans.
   for b in rd['bindings']:
    for d in b['observed']:
     e=d['localizer_event'];cache[(e['a'],e['b'],e['axis'],e['moving'],'L')]=d
   for mode,prob,model,es in [('role_only',rp,oldadapter,all_events),('quality_only',rp0,net,all_events),('combined',rp,net,all_events)]:
    t=time.perf_counter();out=bind(before,prob,c['ob'],es,model,frozen,c['x'],c['ts'],c['fh'],base,motion,cache,mode);elapsed=time.perf_counter()-t
    for a,b in zip(out['cards'],before['cards']):
     assert a.get('shape_evidence')==b.get('shape_evidence') and a.get('motion_evidence')==b.get('motion_evidence')
     if b['display_status']=='difference':assert a['display_status']=='difference'
    protected={k:hashlib.sha256(json.dumps(out[k],sort_keys=True).encode()).hexdigest() for k in ['path','tcn_top3','frame_id','phase_events','phase_partials']};assert all(out[k]==before[k] for k in protected)
    p=U/'feedback'/str(seed)/mode/f"{c['key']}.json";write(p,dict(cards=out['cards'],ledger=out['role_candidate_ledger'],selected=out['role_selected'],pair_checks=out['role_pair_checks'],order_checks=out['role_order_checks'],completion=out['role_completion_checks'],role_probabilities=out['role_probabilities'],protected=protected,baseline=ref['result'],baseline_sha256=ref['sha256']));records.append(dict(seed=seed,clip_id=clip,kind=kind,mode=mode,before=compact(before['cards']),after=compact(out['cards']),result=str(p),sha256=sha(p),role_seconds=role_seconds,binding_seconds=elapsed,omission_candidates=sum(v['state']=='omission_candidate' for v in out['role_completion_checks'])))
   if kind=='full_normal' and clip in gold:
    for a in gold[clip]['anchors']:
     f=a['frame'];target=jc.ep.MAP.get(a['label'],-1)
     if target<0:continue
     pred=int(rp[f].argmax());frames.append(dict(seed=seed,clip_id=clip,frame=f,target=target,before=int(rp0[f].argmax()),after=pred if pred<8 else 0,raw_after=pred,confidence=float(rp[f].max())))
    for e in out['role_candidate_ledger']:
     ys=[jc.ep.MAP.get(a['label'],-1) for a in gold[clip]['anchors'] if e['a']<=a['frame']<=e['b'] and a['label']>=0]
     if len(ys)<2 or len(set(ys))!=1:continue
     a,b=e['a'],e['b'];oldfam=int(rp0[a:b+1].mean(0).argmax());newfam=e['family'] if e['family']<8 else 0;candidate_scores.append(dict(seed=seed,clip_id=clip,span=[a,b],axis=e['axis'],moving=e['moving'],target=ys[0],before=oldfam,after=newfam,accepted=e['accepted'],scope='Sparse same-family anchors inside candidate, not exact boundary correctness'))
  event('evaluated',seed=seed,conditions=len(cases))
 write(U/'FULLPATH_COMPLETE.json',dict(rows=records));write(U/'FRAME_ROLE_SCORES.json',frames);write(U/'CANDIDATE_ROLE_SCORES.json',candidate_scores);write(U/'EVALUATION_COMPLETE.json',dict(conditions=78,seeds=cfg['seeds'],modes=3,outputs=len(records),old_TCN_DTW_reruns=0,old_final_results_reused=True));write(U/'CURRENT_EXECUTION.json',dict(status='EVALUATED_PENDING_VERIFICATION',pid=None))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'EVAL_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
