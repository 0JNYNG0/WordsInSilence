from core import *
import copy,traceback
sys.modules['common']=load('_bf_full_common',F/'common.py');rt=load('_bf_full_runtime',F/'runtime.py')
def compact(d):
 return dict(display=[c['display_status'] for c in d['cards']],status=[c['status'] for c in d['cards']],required=[c.get('required',True) for c in d['cards']],all_required_similar=all(c['display_status']=='similar' for c in d['cards'] if c.get('required',True)),shape_states=[s['state'] for c in d['cards'] for s in c.get('shape_evidence',[])],motion_states=[c['motion_evidence']['state'] for c in d['cards'] if c.get('motion_evidence')])
def main():
 assert (U/'TRAINING_COMPLETE.json').exists() and not (U/'FULLPATH_COMPLETE.json').exists();cfg=read(U/'FROZEN.json');norm=read(U/'NORMALIZATION.json')
 records=read(F/'REPLAY_INPUTS.json')['rows'];P=O/'word100_positive_direction_20261109';edits=[r for r in read(P/'COMPARISON.json')['rows'] if r['kind'] in ['right_closed_hand_main','main_reversed','closed_hand_and_reversed']];assert len(records)==60 and len(edits)==9
 write(U/'FULLPATH_FROZEN.json',dict(raw_inputs_sha256=sha(F/'REPLAY_INPUTS.json'),edited_inputs=[dict(path=r['prediction'],sha256=sha(r['prediction'])) for r in edits],route='Stable old encoder/phase/difference/DTW and completion guard; newly trained shape and motion evidence only, in isolated process',condition=cfg['conditions'],seeds=cfg['seeds'],time_scope='Cached RTMW coordinates to TCN/feedback/evidence/DTW/completion; no detector/camera/IO/UI'))
 c,a=rt.instantiate();pe=sys.modules['positive_evidence'];old_refine=rt.consumer.refine;old_along=pe.along_forearm;outrows=[]
 for cond in cfg['conditions']:
  for seed in cfg['seeds']:
   name=f'{cond}_{seed}';net=Network(cond,norm);net.load_state_dict(torch.load(U/'runs'/name/'model.pt',weights_only=True,map_location='cpu'));net.eval();cache={}
   def new_refine(base,attrs,x,ts,fid,phase):
    z=shared.normalize(x);word=c.base.f['feedback']['words'].index(cache['word'])
    with torch.inference_mode():o=net.shared(torch.from_numpy(z[None]),torch.tensor([len(z)]),torch.tensor([word]))
    cache['h']=o['latent'];cache['phase']=phase;cache['new_attributes']=o['attributes'][0].sigmoid().numpy()
    result=old_refine(base,cache['new_attributes'],x,ts,fid,phase);result['factorized_observation_model']=name;result['phase_and_dtw_from_unchanged_base']=True;return result
   def new_along(x,ts,phase,contract):
    old=old_along(x,ts,phase,contract);out=copy.deepcopy(old);span=old.get('span')
    if not span:return out
    g,ok,info=geometry(x,ts,*span,2,contract['moving_hand'],contract['anchor_arm'])
    with torch.inference_mode():pr=net.movement(cache['h'],torch.from_numpy(g[None]),torch.tensor([[0,*span]])).softmax(-1)[0].numpy()
    pred=int(pr.argmax());conf=float(pr.max());out.update(old_geometry_state=old['state'],state='unknown' if not ok or conf<.65 or pred==1 else 'supported' if pred-1==contract['expected_sign'] else 'contradicted',learned_class=pred,learned_confidence=conf,learned_eligible=bool(ok),learned_evidence=info,learned_model=name);return out
   rt.consumer.refine=new_refine;pe.along_forearm=new_along
   for r in records:
    key=r['clip_id']+'_'+r['kind'];folder=U/'fullpath'/name/key;assert sha(r['path'])==r['sha256'];z=np.load(r['path']);raw={k:z[k] for k in ['xy_px','score','timestamp_us','frame_id','source_width','source_height']};old=read(F/'replay_results'/key/'after.json');cache['word']=r['word_id']
    try:
     t=time.perf_counter();out=c.raw(raw,r['word_id']);elapsed=time.perf_counter()-t
     assert out['tcn_top3']==old['tcn_top3'];assert out['path']==old['path'];assert len(out['frame_id'])==r['frames'];assert out['reference_video_reads']==0 and out['rgb_model_calls']==0
     write(folder/'after.json',out);outrows.append(dict(model=name,clip_id=r['clip_id'],kind=r['kind'],status='OK',seconds=elapsed,baseline=compact(old),candidate=compact(out),result=str(folder/'after.json'),sha256=sha(folder/'after.json'),tcn_identical=True,dtw_identical=True,frames_preserved=True))
    except Exception:
     err=traceback.format_exc();write(folder/'ERROR.json',dict(error=err));outrows.append(dict(model=name,clip_id=r['clip_id'],kind=r['kind'],status='ERROR',error=err));event('fullpath_error',model=name,key=key)
   for r in edits:
    cache['word']=r['word'];key=r['clip_id']+'_'+r['kind'];z=np.load(r['prediction']);old=read(r['decision'])['after'];t=time.perf_counter();out=c.channels(z['x'],z['timestamp_us'],z['frame_id'],r['word']);elapsed=time.perf_counter()-t;folder=U/'edited_channels'/name/key;write(folder/'after.json',out)
    outrows.append(dict(model=name,clip_id=r['clip_id'],kind=r['kind'],status='OK',seconds=elapsed,baseline=compact(old),candidate=compact(out),result=str(folder/'after.json'),sha256=sha(folder/'after.json'),route='Stored edited channels, not natural learner errors'))
   event('fullpath_complete_model',model=name,conditions=69)
 rt.consumer.refine=old_refine;pe.along_forearm=old_along
 write(U/'FULLPATH_COMPLETE.json',dict(rows=outrows,old_model_reexecution='Stable runtime branch executed as component of new candidate; archived baseline results reused, no old benchmark rescoring',default_web_changed=False,private_used=False,errors_in_denominator=True))
if __name__=='__main__':
 try:main()
 except BaseException:
  (U/f'FULL_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
