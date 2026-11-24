"""New fixed candidate runs on historical public inputs, archived baseline reused."""
from common_action import *
import copy,traceback
# Only import the archived runtime, never execute its experiment entrypoints.
sys.modules['common']=load('_ca_fullpath_common',F/'common.py')
rt=load('_ca_fullpath_runtime',F/'runtime.py')
def compact(d):
 return dict(display=[c['display_status'] for c in d['cards']],status=[c['status'] for c in d['cards']],required=[c.get('required',True) for c in d['cards']],top1=d.get('tcn_top3',[{}])[0].get('word'),all_required_similar=all(c['display_status']=='similar' for c in d['cards'] if c.get('required',True)))
def main():
 assert not (U/'FULLPATH_COMPLETE.json').exists()
 cfg=read(U/'FROZEN.json');norm=read(U/'NORMALIZATION.json')
 records=read(F/'REPLAY_INPUTS.json')['rows'];positive=O/'word100_positive_direction_20261109'
 edits=[r for r in read(positive/'COMPARISON.json')['rows'] if r['kind'] in ['right_closed_hand_main','main_reversed','closed_hand_and_reversed']]
 assert len(edits)==9
 write(U/'FULLPATH_FROZEN.json',dict(inputs=str(F/'REPLAY_INPUTS.json'),inputs_sha256=sha(F/'REPLAY_INPUTS.json'),additional_edits=[dict(clip_id=r['clip_id'],kind=r['kind'],path=r['prediction'],sha256=sha(r['prediction'])) for r in edits],models=[f'{c}_{s}' for c in cfg['conditions'] for s in cfg['seeds']],old_baseline='Saved F after.json, includes opt-in completion guard; current default web unchanged. Edited channels use saved positive-direction decisions; this baseline distinction is reported.',replace='Only isolated feedback encoder and shared learned direction; all old missing/completion/coverage guards still run. New common observation output is never itself promoted to educational correctness.',time_scope='cached raw RTMW -> existing consumer -> TCN, new feedback, DTW, completion and final quarantine; no detector/camera/UI'))
 c,a=rt.instantiate();pe=sys.modules['positive_evidence'];original_along=pe.along_forearm;results=[]
 for cond in cfg['conditions']:
  for seed in cfg['seeds']:
   name=f'{cond}_{seed}';net=CommonActionNet(norm['mean'],norm['std']);net.load_state_dict(torch.load(U/'runs'/name/'model.pt',weights_only=True,map_location='cpu'));net.eval();c.base.feedback=net.base;cache={}
   def capture(module,inputs,output):cache['h']=output['latent']
   hook=net.base.register_forward_hook(capture)
   def learned_along(x,ts,phase,contract):
    old=original_along(x,ts,phase,contract);span=old.get('span');out=copy.deepcopy(old);out['old_geometry_state']=old['state'];out['learned_model']=name
    if not span:return out
    g,ok,info=motion.features(x,ts,*span,'along_other_forearm',contract['moving_hand'],contract['anchor_arm'])
    with torch.inference_mode():pr=net.movement(cache['h'],torch.from_numpy(g[None]),torch.tensor([[0,*span]])).softmax(-1)[0].numpy()
    pred=int(pr.argmax());conf=float(pr.max());state='unknown' if not ok or conf<cfg['confidence'] or pred==1 else 'supported' if pred-1==contract['expected_sign'] else 'contradicted'
    out.update(state=state,learned_class=pred,learned_confidence=conf,learned_eligible=bool(ok),learned_evidence=info);return out
   pe.along_forearm=learned_along
   for r in records:
    key=r['clip_id']+'_'+r['kind'];folder=U/'fullpath'/name/key
    assert sha(r['path'])==r['sha256'];z=np.load(r['path']);raw={k:z[k] for k in ['xy_px','score','timestamp_us','frame_id','source_width','source_height']};baseline=read(F/'replay_results'/key/'after.json')
    try:
     started=time.perf_counter();out=c.raw(raw,r['word_id']);seconds=time.perf_counter()-started
     assert len(out['frame_id'])==r['frames'];assert out['tcn_top3']==baseline['tcn_top3'];assert out['reference_video_reads']==0 and out['rgb_model_calls']==0
     # Expose relation observations for each candidate span, without pretending these
     # values prove exact contact, a whole sign, or a not-yet-labelled fingertip shape.
     for card in out['cards']:
      span=card.get('query_indices');observations=[]
      if span:
       for kind in motion.KINDS:
        g,ok,info=motion.features(a.converted,raw['timestamp_us'],*span,kind)
        with torch.inference_mode():pr=net.movement(cache['h'],torch.from_numpy(g[None]),torch.tensor([[0,*span]])).softmax(-1)[0].numpy()
        observations.append(dict(kind=kind,class_index=int(pr.argmax()) if ok else None,confidence=float(pr.max()),available=bool(ok),state='observed' if ok and float(pr.max())>=cfg['confidence'] else 'unknown',educational_correctness=False))
      card['common_observations']=observations
     seconds=time.perf_counter()-started;write(folder/'after.json',out)
     results.append(dict(model=name,clip_id=r['clip_id'],kind=r['kind'],status='OK',baseline=compact(baseline),candidate=compact(out),seconds=seconds,result=str(folder/'after.json'),sha256=sha(folder/'after.json'),frames_preserved=True,tcn_identical=True))
    except Exception:
     err=traceback.format_exc();write(folder/'ERROR.json',dict(error=err));results.append(dict(model=name,clip_id=r['clip_id'],kind=r['kind'],status='ERROR',error=err));event('fullpath_failure',model=name,clip=r['clip_id'],kind=r['kind'])
   for r in edits:
    z=np.load(r['prediction']);old=read(r['decision'])['after'];key=r['clip_id']+'_'+r['kind'];t=time.perf_counter();out=c.channels(z['x'],z['timestamp_us'],z['frame_id'],r['word']);dt=time.perf_counter()-t
    folder=U/'edited_channels'/name/key;write(folder/'after.json',out);results.append(dict(model=name,clip_id=r['clip_id'],kind=r['kind'],status='OK',baseline=compact(old),candidate=compact(out),seconds=dt,result=str(folder/'after.json'),sha256=sha(folder/'after.json'),route='saved edited channels; no raw detector rerun'))
   hook.remove();event('fullpath_model_complete',model=name,rows=len([r for r in results if r['model']==name]))
 pe.along_forearm=original_along
 write(U/'FULLPATH_COMPLETE.json',dict(rows=results,old_model_rerun=False,default_web_changed=False,private_used=False,errors_in_denominator=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  (U/f'FULLPATH_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
