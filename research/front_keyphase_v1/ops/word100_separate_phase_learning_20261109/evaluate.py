from lib import *
from collections import Counter
import statistics,traceback

def phase_records(phase,row,kind,name):
 records=[]
 for i in np.flatnonzero(row['stage_gold']>=0):
  p=phase[i];j=int(p.argmax());pred=GROUPS.get(j-1,-1);target=GROUPS[int(row['stage_gold'][i])]
  records.append(dict(task='stage',model=name,kind=kind,clip_id=row['clip_id'],frame=int(row['frame_id'][i]),target=target,pred=pred,confident=bool(p[j]>=.65)))
 return records

def main():
 assert (P/'TRAINING_COMPLETE.json').exists();assert not (P/'EVALUATION_COMPLETE.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1)
 cfg=read(P/'FROZEN.json');rows={r['clip_id']:r for r in load_rows()};inputs=read(P/'EVAL_INPUTS.json');write(P/'EVAL_FROZEN.json',dict(primary=cfg['primary'],models=[f'{c}_{s}' for c in cfg['conditions'] for s in cfg['seeds']],controls_sha256=sha(P/'EVAL_INPUTS.json'),stage_groups=GROUPS,thresholds='Reuse unchanged runtime phase/shape/direction thresholds',no_model_selection=True,code_sha256=sha(P/'evaluate.py')))
 baseline=[]
 for r in inputs['rows']:
  if r['kind']=='full_normal':
   z=np.load(r['path']);baseline+=phase_records(z['phase_old'],rows[r['clip_id']],r['kind'],'current_web_head')
 write(P/'BASELINE_STAGE_RECORDS.json',baseline)
 mc=torch.load(M/'runs/motion_only_42/model.pt',map_location='cpu',weights_only=True);mm=motion.MotionNet();mm.load_state_dict(mc['model']);mm.eval();results={}
 for cond in cfg['conditions']:
  for seed in cfg['seeds']:
   name=f'{cond}_{seed}';net=Heads(cond=='matched_relations');net.load_state_dict(torch.load(P/'runs'/name/'model.pt',map_location='cpu',weights_only=True));net.eval();records=[];cases=[];motion_rows=[];times=[]
   with torch.inference_mode():
    for r in inputs['rows']:
     assert sha(r['path'])==r['sha256'];z=np.load(r['path']);h=torch.from_numpy(z['h'][None]);a=torch.from_numpy(z['p'][None]);g=torch.from_numpy(z['g'][None]);valid=torch.ones(1,h.shape[1],dtype=torch.bool);word=torch.tensor([r['word']]);t=time.perf_counter();ph,d0=net(h,a,g,word,valid,torch.zeros_like(a));_,d1=net(h,a,g,word,valid,torch.ones_like(a));elapsed=time.perf_counter()-t
     phase=ph[0].softmax(-1).numpy();d0=d0[0].sigmoid().numpy();d1=d1[0].sigmoid().numpy();out=decide(r['word_id'],phase,d0,d1,z['p'],z['raw_x'],z['timestamp_us'],z['frame_id'])
     if r['kind']=='full_normal':records+=phase_records(phase,rows[r['clip_id']],r['kind'],name);times.append(elapsed)
     outpath=P/'decisions'/name/f"{r['clip_id']}_{r['kind']}.json";write(outpath,out)
     cases.append(dict(clip_id=r['clip_id'],kind=r['kind'],word=r['word_id'],statuses=[c['status'] for c in out['cards']],display=[c.get('display_status') for c in out['cards']],components=[c.get('difference_components',[]) for c in out['cards']],required=[c.get('required',True) for c in out['cards']],phase_presence=[all(e['span'] is not None for e in c['phase_evidence']) for c in out['cards']],decision=str(outpath),sha256=sha(outpath)))
     if seed==42:
      dst=P/'predictions'/name/f"{r['clip_id']}_{r['kind']}.npz";dst.parent.mkdir(exist_ok=True,parents=True);np.savez_compressed(dst,phase=phase,different_if0=d0,different_if1=d1)
      # Existing small direction learner, evaluated on NEW automatic intervals.
      done=set()
      for c in out['cards']:
       mo=c.get('motion_evidence');span=mo.get('span') if mo else None
       if not span or tuple(span) in done:continue
       done.add(tuple(span));f,eligible,info=motion.features(z['raw_x'],z['timestamp_us'],*span,'along_other_forearm',mo['moving_hand'],mo['anchor_arm']);p=mm((torch.tensor(f)-mc['mean'])/mc['std']).softmax(-1).numpy();pr=int(p.argmax());conf=float(p.max());state='unknown' if not eligible or conf<.65 or pr==1 else 'supported' if pr-1==mo['expected_sign'] else 'contradicted'
       motion_rows.append(dict(clip_id=r['clip_id'],kind=r['kind'],span=span,geometry=mo['state'],learned=state,confidence=conf,eligible=bool(eligible),info=info))
   summary={}
   for kind in sorted({r['kind'] for r in cases}):
    cs=[r for r in cases if r['kind']==kind];summary[kind]=dict(clips=len(cs),difference_clips=sum('difference' in r['display'] for r in cs),uncertain_clips=sum('uncertain' in r['display'] for r in cs),missing_candidate_clips=sum('missing_candidate' in r['statuses'] for r in cs),all_required_similar=sum(all(s=='similar' for s,req in zip(r['display'],r['required']) if req) for r in cs),all_required_phase_present=sum(all(s for s,req in zip(r['phase_presence'],r['required']) if req) for r in cs))
   result=dict(stage=dict(correct=sum(r['pred']==r['target'] for r in records),total=len(records),confident_correct=sum(r['confident'] and r['pred']==r['target'] for r in records),withheld=sum(not r['confident'] for r in records)),auto=summary,head_seconds=dict(median=statistics.median(times),p95=float(np.percentile(times,95)),scope='New temporal/difference heads only, includes duplicate phase forward; excludes frozen base, detector, DTW, I/O; one pass per public clip, not a whole-app benchmark'))
   write(P/'scores'/f'{name}_records.json',records);write(P/'scores'/f'{name}_cases.json',cases);write(P/'scores'/f'{name}_motion.json',motion_rows);write(P/'scores'/f'{name}.json',result);results[name]=result;event('evaluated',model=name,stage=result['stage'],normal=summary['full_normal'],head_seconds=result['head_seconds'])
 write(P/'EVALUATION_COMPLETE.json',dict(models=results,baseline_stage=dict(correct=sum(r['pred']==r['target'] for r in baseline),total=len(baseline)),common_backbone_unchanged=sha(CHECKPOINT)==cfg['base_sha256'],web_model_changed=False));write(P/'CURRENT_EXECUTION.json',dict(status='EVALUATED_PENDING_VERIFY',pid=None))
if __name__=='__main__':
 try:main()
 except BaseException:
  (P/f'EVAL_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
