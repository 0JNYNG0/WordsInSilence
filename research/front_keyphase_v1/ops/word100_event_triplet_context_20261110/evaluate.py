from triplet_core import *
from connector import apply_links
import copy,traceback
def main():
 assert (U/'TRAINING_COMPLETE.json').exists() and not (U/'EVALUATION_COMPLETE.json').exists()
 cfg=read(U/'FROZEN.json')
 for p,h in {**cfg['pins'],**cfg['protections'],**read(U/'EVAL_PINS.json')}.items():assert sha(p)==h,p
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 runtime=lc.be.ev.prepare_runtime();saved={r['clip_id']+'_'+r['kind']:r for r in read(L/'GUARDED_COMPLETE.json')['rows'] if r['seed']==42};contracts=read(L/'GUIDE_CONTRACTS.json')['words'];gold={r['clip_id']:r for r in read(U/'EVALUATION_LABELS.json')['rows']}
 wordmap={r['word_id']:r['word'] for r in old.rows() if r['word']>=0}
 base=prior.Network('coupled',read(old.P/'NORMALIZATION.json'));base.load_state_dict(torch.load(old.P/'runs/coupled_42/model.pt',weights_only=True,map_location='cpu'));base.eval()
 for c in runtime:
  r=saved[c['key']];assert sha(r['result'])==r['sha256'];c['baseline']=read(r['result']);c['baseline_file']=r['result'];c['fh'],_=lc.be.ev.frozen_forward(base,c['x'],wordmap[c['baseline']['word']])
 rows=[];scores=[];frames=[]
 for seed in cfg['seeds']:
  net=TripletNet();net.load_state_dict(torch.load(U/'runs'/str(seed)/'model.pt',weights_only=True,map_location='cpu'));net.eval()
  for c in runtime:
   before=c['baseline'];t=time.perf_counter();pred,prob=predict_segments(net,c['x'],c['ts'],candidate_spans(before));ss=accepted_spans(pred)
   out=apply_links(before,prob,c['x'],c['ts'],c['fh'],base,contracts[before['word']],ss);out=guard.guard(out,before)
   out['triplet_segments']=pred;out['triplet_core_is_not_completion']=True;out['previous_guide_link_bindings']=before['guide_link_bindings'];out['triplet_version']='v1_event_identity_plus_center_only_core'
   seconds=time.perf_counter()-t
   for key in ['path','tcn_top3','frame_id','phase_events','phase_partials']:assert out[key]==before[key]
   for a,b in zip(out['cards'],before['cards']):assert a.get('shape_evidence')==b.get('shape_evidence') and a.get('motion_evidence')==b.get('motion_evidence')
   dest=U/'fullpath'/str(seed)/c['key']/'after.json';write(dest,out)
   rows.append(dict(seed=seed,clip_id=c['row']['clip_id'],kind=c['row']['kind'],baseline=c['baseline_file'],baseline_sha256=sha(c['baseline_file']),result=str(dest),sha256=sha(dest),before=le.compact(before),after=le.compact(out),seconds=seconds,frames=len(c['x']),segments=len(pred)))
   if c['row']['kind']=='full_normal' and c['row']['clip_id'] in gold:
    gg=gold[c['row']['clip_id']]['segments'];ps,_=predict_segments(net,c['x'],c['ts'],[(g['a'],g['b']) for g in gg]);oldprob=np.asarray(before['guide_link_probabilities'])
    for g,p in zip(gg,ps):
     op=oldprob[g['a']:g['b']+1].mean(0);scores.append(dict(seed=seed,clip_id=c['row']['clip_id'],gold=g,old_context=int(op.argmax()),new=p))
   event('case',seed=seed,key=c['key'],changed=rows[-1]['before']!=rows[-1]['after'])
 write(U/'FULLPATH_COMPLETE.json',dict(rows=rows));write(U/'SEGMENT_SCORES.json',scores)
 write(U/'EVALUATION_COMPLETE.json',dict(seeds=cfg['seeds'],conditions=78,outputs=len(rows),gold_windows_per_seed=24,historical_development=True,new_independent_test=False,core_status_diagnostic_only=True))
 write(U/'CURRENT_EXECUTION.json',dict(status='EVALUATED_PENDING_VERIFICATION',pid=None))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'EVAL_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
