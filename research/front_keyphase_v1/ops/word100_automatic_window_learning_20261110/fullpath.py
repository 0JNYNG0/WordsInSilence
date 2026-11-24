from window_core import *
import copy,traceback
F=O/'word100_completion_fullpath_20261110';D=O/'word100_action_feedback_demo_20261108';POS=O/'word100_positive_direction_20261109'
sys.modules['common']=load('_aw_full_common',F/'common.py');rt=load('_aw_full_runtime',F/'runtime.py');guard=load('_aw_existing_conflict',P/'evidence_conflict_guard.py')
def compact(d):return dict(display=[c['display_status'] for c in d['cards']],status=[c['status'] for c in d['cards']],required=[c.get('required',True) for c in d['cards']],all_required_similar=all(c['display_status']=='similar' for c in d['cards'] if c.get('required',True)),shape_states=[s['state'] for c in d['cards'] for s in c.get('shape_evidence',[])],motion_states=[c['motion_evidence']['state'] for c in d['cards'] if c.get('motion_evidence')])
def main():
 assert (U/'TRAINING_COMPLETE.json').exists() and not (U/'FULLPATH_COMPLETE.json').exists();cfg=read(U/'FROZEN.json');protocol=read(U/'EVAL_FROZEN.json')
 for p,h in protocol['code_pins'].items():assert sha(p)==h,p
 records=read(F/'REPLAY_INPUTS.json')['rows'];edits=[r for r in read(POS/'COMPARISON.json')['rows'] if r['kind'] in ['right_closed_hand_main','main_reversed','closed_hand_and_reversed']];assert len(records)==60 and len(edits)==9
 labelled={r['clip_id']:r for r in prior.rows() if r['role']=='development' and r['orders']}
 c,a=rt.instantiate();pe=sys.modules['positive_evidence'];old_refine=rt.consumer.refine;old_along=pe.along_forearm;definitions=read(D/'CARDS.json')['words'];outrows=[]
 frozen=prior.Network('coupled',read(P/'NORMALIZATION.json'));frozen.load_state_dict(torch.load(P/'runs/coupled_42/model.pt',map_location='cpu',weights_only=True));frozen.eval()
 for mode in cfg['conditions']:
  for seed in cfg['seeds']:
   name=f'{mode}_{seed}';net=WindowNet(mode,read(U/'NORMALIZATION.json'));net.load_state_dict(torch.load(U/'runs'/name/'model.pt',map_location='cpu',weights_only=True));net.eval();cache={}
   def new_refine(base,attrs,x,ts,fid,phase):
    z=shared.normalize(x);word=c.base.f['feedback']['words'].index(cache['word']);g=geometry(x,ts)
    with torch.inference_mode():
     obs=frozen.shared(torch.tensor(z[None]),torch.tensor([len(z)]),torch.tensor([word]));h,_=net.encode(torch.tensor(g[None]),torch.tensor([len(g)]))
    cache.update(h=h[0],x=x);result=old_refine(base,obs['attributes'][0].sigmoid().numpy(),x,ts,fid,phase);result['automatic_window_model']=name;return result
   def new_along(x,ts,phase,contract):
    old=old_along(x,ts,phase,contract);out=copy.deepcopy(old);span=old.get('span')
    if not span:return out
    q=torch.tensor(relations(x,2,contract['moving_hand'],contract['anchor_arm']))
    with torch.inference_mode():pr,cp,att,valid=net.movement(cache['h'],q,*span,2);pr=pr.softmax(-1).numpy()
    pred=int(pr.argmax());conf=float(pr.max());ok=bool(valid.float().mean()>=.8 and valid.sum()>=4)
    out.update(old_geometry_state=old['state'],state='unknown' if not ok or conf<.65 or pred in [1,3] else 'supported' if pred-1==contract['expected_sign'] else 'contradicted',learned_class=pred,learned_confidence=conf,learned_eligible=ok,learned_model=name,core_attention=att.numpy().tolist(),core_probability=cp.sigmoid().numpy().tolist(),observed_fraction=float(valid.float().mean()),multi_direction=pred==3)
    return out
   def finish(out):
    # Pre-existing conflict safety applies identically to all candidates.
    raw=copy.deepcopy(out);out=guard.apply(out);before_order=compact(out);checks=[];defs=definitions[cache['word']];av=shared.available(shared.normalize(cache['x']))
    for i in range(len(out['cards'])-1):
     ca,cb=out['cards'][i:i+2];da,db=defs[i:i+2];sa=ca.get('query_indices');sb=cb.get('query_indices')
     if not sa or not sb:checks.append(dict(pair=[ca['id'],cb['id']],state='unknown',reason='no_window'));continue
     ga=da['groups'][-1][-1];gb=db['groups'][0][0];lo=min(sa[0],sb[0]);hi=max(sa[1],sb[1])
     with torch.inference_mode():logit,att=net.ordering(cache['h'],lo,hi,ga,gb);pr=float(logit.sigmoid())
     ok=bool(av[lo:hi+1].any(1).mean()>=.8 and not ca.get('observation_quarantine_overlap') and not cb.get('observation_quarantine_overlap'));state='unknown' if not ok or .35<pr<.65 else 'supported' if pr>=.65 else 'contradicted'
     checks.append(dict(pair=[ca['id'],cb['id']],guide_pair=[ga,gb],span=[lo,hi],probability=pr,eligible=ok,state=state,attention_centres=(att.numpy()*np.arange(lo,hi+1)[:,None]).sum(0).tolist()))
     # Can only flag an observed order difference. Never certify completion,
     # fill an absent event or override an existing missing/quarantine decision.
     if state=='contradicted' and cb['display_status'] in ['similar','difference']:
      cb.update(status='difference_candidate',display_status='difference',message='동작 순서가 참고와 다르게 보여요. 표시된 두 구간을 확인해 보세요.',educational_wrong=False);cb.setdefault('difference_components',[]).append('order');cb['candidate_message']=cb['message']
    out['learned_order_checks']=checks;return raw,out,before_order
   rt.consumer.refine=new_refine;pe.along_forearm=new_along
   for r in records:
    key=r['clip_id']+'_'+r['kind'];folder=U/'fullpath'/name/key;assert sha(r['path'])==r['sha256'];z=np.load(r['path']);rawin={k:z[k] for k in ['xy_px','score','timestamp_us','frame_id','source_width','source_height']};old=read(F/'replay_results'/key/'after.json');last=read(P/'fullpath/coupled_42'/key/'after.json');cache['word']=r['word_id']
    try:
     t=time.perf_counter();output=c.raw(rawin,r['word_id']);raw,out,before_order=finish(output);elapsed=time.perf_counter()-t
     assert out['tcn_top3']==old['tcn_top3'] and out['path']==old['path'] and len(out['frame_id'])==r['frames'];assert out['reference_video_reads']==0 and out['rgb_model_calls']==0
     write(folder/'raw.json',raw);write(folder/'after.json',out);outrows.append(dict(model=name,clip_id=r['clip_id'],kind=r['kind'],status='OK',seconds=elapsed,baseline=compact(old),previous=compact(guard.apply(last)),candidate_raw=compact(raw),before_order=before_order,candidate=compact(out),result=str(folder/'after.json'),sha256=sha(folder/'after.json'),tcn_identical=True,dtw_identical=True,frames_preserved=True))
    except Exception:
     err=traceback.format_exc();write(folder/'ERROR.json',dict(error=err));outrows.append(dict(model=name,clip_id=r['clip_id'],kind=r['kind'],status='ERROR',error=err));event('fullpath_error',model=name,key=key)
   for r in edits:
    cache['word']=r['word'];key=r['clip_id']+'_'+r['kind'];z=np.load(r['prediction']);old=read(r['decision'])['after'];last=read(P/'edited_channels/coupled_42'/key/'after.json');folder=U/'edited_channels'/name/key
    t=time.perf_counter();output=c.channels(z['x'],z['timestamp_us'],z['frame_id'],r['word']);raw,out,before_order=finish(output);elapsed=time.perf_counter()-t;write(folder/'raw.json',raw);write(folder/'after.json',out)
    outrows.append(dict(model=name,clip_id=r['clip_id'],kind=r['kind'],status='OK',seconds=elapsed,baseline=compact(old),previous=compact(guard.apply(last)),candidate_raw=compact(raw),before_order=before_order,candidate=compact(out),result=str(folder/'after.json'),sha256=sha(folder/'after.json'),route='Stored controlled edited coordinates, not natural learner errors'))
   for r in records:
    if r['kind']!='full_normal' or r['clip_id'] not in labelled:continue
    lab=labelled[r['clip_id']];anchors=[int(lab['frame_id'][p[k]]) for p in lab['pairs']+lab['orders'] for k in ['a','b']];z=np.load(r['path']);rawin={k:z[k].copy() for k in ['xy_px','score','timestamp_us','frame_id','source_width','source_height']};ii=np.flatnonzero((rawin['frame_id']>=min(anchors))&(rawin['frame_id']<=max(anchors)))
    assert len(ii)>=4 and np.all(np.diff(ii)==1)
    for k in ['xy_px','score']:rawin[k][ii]=rawin[k][ii[::-1]]
    cache['word']=r['word_id'];t=time.perf_counter();output=c.raw(rawin,r['word_id']);raw,out,before_order=finish(output);elapsed=time.perf_counter()-t;folder=U/'reversed_order'/name/r['clip_id'];write(folder/'raw.json',raw);write(folder/'after.json',out)
    outrows.append(dict(model=name,clip_id=r['clip_id'],kind='reversed_core_auto_order',status='OK',seconds=elapsed,candidate_raw=compact(raw),before_order=before_order,candidate=compact(out),result=str(folder/'after.json'),sha256=sha(folder/'after.json'),reversed_source_frame_span=[min(anchors),max(anchors)],frames_preserved=len(out['frame_id'])==r['frames'],route='New control from cached public coordinates, reverse all reviewed main actions; no archived baseline for this new condition'))
   event('fullpath_complete_model',model=name,conditions=78)
 rt.consumer.refine=old_refine;pe.along_forearm=old_along;write(U/'FULLPATH_COMPLETE.json',dict(rows=outrows,default_web_changed=False,private_used=False,errors_in_denominator=True,time_scope='Cached RTMW coordinates through unchangedTCN/oldphase/DTW/frozenPshape/newmotion/neworder/completionguards. Excludes detector/camera/IO/UI.'))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'FULL_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
