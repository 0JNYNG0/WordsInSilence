"""Posthoc bounded composition: preserve P shape/direction, use primary42 order only to flag conflicts.
No training or threshold search. Archived69old conditions reused;9new reversals get first old-baseline inference.
"""
from window_core import *
import copy,traceback
F=O/'word100_completion_fullpath_20261110';POS=O/'word100_positive_direction_20261109';V=U/'order_only_followup_v2'
sys.modules['common']=load('_oo_common',F/'common.py');rt=load('_oo_runtime',F/'runtime.py');guard=load('_oo_guard',P/'evidence_conflict_guard.py')
def compact(d):return dict(display=[c['display_status'] for c in d['cards']],all_required_similar=all(c['display_status']=='similar' for c in d['cards'] if c.get('required',True)),motion_states=[c['motion_evidence']['state'] for c in d['cards'] if c.get('motion_evidence')])
def main():
 assert not (U/'ORDER_ONLY_COMPLETE.json').exists();write(U/'ORDER_ONLY_PLAN_v2.json',dict(posthoc=True,reason='New motion head regressed. Retain archivedPcoupled42 shape/direction+existing conflict guard; use primary42 learned ordering solely as a contradiction veto. Never promotes uncertain or marks completion.',selection='Primary42 fixed before evaluation; no best-seed selection. No new weights/thresholds.',inputs='69archived conditions reused.9new reversed-core controls were not previously evaluated byP, so baseline is computed first time only for those.',repair='First assertion preserved: motion contradiction can supply a previously missing optional query window. Never transfer ordering across unequal windows. Recompute only new order queries on old windows, or mark absent old window unknown; no frame/condition deletion.',gate='No normal/speed/crossing/shape-only regression; no missing->normal; actual automatic-order defect improved; all failures preserved. HistoricalDEV posthoc diagnostic, not fresh unbiased test.',protections=read(U/'FROZEN.json')['protections'],code_sha256=sha(__file__)))
 c,a=rt.instantiate();pe=sys.modules['positive_evidence'];old_refine=rt.consumer.refine;old_along=pe.along_forearm;net=prior.Network('coupled',read(P/'NORMALIZATION.json'));net.load_state_dict(torch.load(P/'runs/coupled_42/model.pt',weights_only=True,map_location='cpu'));net.eval();cache={}
 def ref(base,attrs,x,ts,fid,phase):
  z=shared.normalize(x);word=c.base.f['feedback']['words'].index(cache['word'])
  with torch.inference_mode():o=net.shared(torch.tensor(z[None]),torch.tensor([len(z)]),torch.tensor([word]))
  cache['h']=o['latent'];return old_refine(base,o['attributes'][0].sigmoid().numpy(),x,ts,fid,phase)
 def along(x,ts,phase,contract):
  old=old_along(x,ts,phase,contract);out=copy.deepcopy(old);span=old.get('span')
  if not span:return out
  g,ok,info=prior.geometry(x,ts,*span,2,contract['moving_hand'],contract['anchor_arm'])
  with torch.inference_mode():pr=net.movement(cache['h'],torch.tensor(g[None]),torch.tensor([[0,*span]])).softmax(-1)[0].numpy()
  pred=int(pr.argmax());conf=float(pr.max());out.update(old_geometry_state=old['state'],state='unknown' if not ok or conf<.65 or pred==1 else 'supported' if pred-1==contract['expected_sign'] else 'contradicted',learned_class=pred,learned_confidence=conf,learned_eligible=bool(ok),learned_evidence=info,learned_model='coupled_42');return out
 rt.consumer.refine=ref;pe.along_forearm=along;manifest=read(F/'REPLAY_INPUTS.json')['rows'];main={r['clip_id']:r for r in manifest if r['kind']=='full_normal'};records=[r for r in read(U/'FULLPATH_COMPLETE.json')['rows'] if r['model']=='learned_core_42'];outrows=[]
 edits={r['clip_id']+'_'+r['kind']:r for r in read(POS/'COMPARISON.json')['rows']};defs=read(O/'word100_action_feedback_demo_20261108/CARDS.json')['words'];ordernet=WindowNet('learned_core',read(U/'NORMALIZATION.json'));ordernet.load_state_dict(torch.load(U/'runs/learned_core_42/model.pt',weights_only=True,map_location='cpu'));ordernet.eval()
 for r in records:
  key=r['clip_id']+'_'+r['kind'];source=read(r['result']);new_baseline=False
  if r['kind']=='reversed_core_auto_order':
   entry=main[r['clip_id']];z=np.load(entry['path']);raw={k:z[k].copy() for k in ['xy_px','score','timestamp_us','frame_id','source_width','source_height']};lo,hi=r['reversed_source_frame_span'];ii=np.flatnonzero((raw['frame_id']>=lo)&(raw['frame_id']<=hi))
   for k in ['xy_px','score']:raw[k][ii]=raw[k][ii[::-1]]
   cache['word']=entry['word_id'];baseline=guard.apply(c.raw(raw,entry['word_id']));new_baseline=True;write(V/key/'first_baseline.json',baseline)
  else:
   parent='edited_channels' if r['kind'] in ['right_closed_hand_main','main_reversed','closed_hand_and_reversed'] else 'fullpath';baseline=guard.apply(read(P/parent/'coupled_42'/key/'after.json'))
  same_windows=[x.get('query_indices') for x in source['cards']]==[x.get('query_indices') for x in baseline['cards']];order_checks=copy.deepcopy(source['learned_order_checks'])
  if not same_windows:
   if key in edits:z=np.load(edits[key]['prediction']);x=z['x'];ts=z['timestamp_us'];word=edits[key]['word']
   else:
    observed=rt.consumer.quarantine(raw['xy_px'],raw['score'],raw['timestamp_us'],int(raw['source_width']),int(raw['source_height']));conv=rt.consumer.A.convert(observed['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']));x=conv['tcn280'];ts=conv['timestamp_us'];word=entry['word_id']
   with torch.inference_mode():h,_=ordernet.encode(torch.tensor(geometry(x,ts)[None]),torch.tensor([len(x)]))
   av=shared.available(shared.normalize(x));order_checks=[]
   for i in range(len(baseline['cards'])-1):
    ca,cb=baseline['cards'][i:i+2];sa=ca.get('query_indices');sb=cb.get('query_indices')
    if not sa or not sb:order_checks.append(dict(pair=[ca['id'],cb['id']],state='unknown',reason='preserved_baseline_has_no_window'));continue
    ga=defs[word][i]['groups'][-1][-1];gb=defs[word][i+1]['groups'][0][0];lo=min(sa[0],sb[0]);hi=max(sa[1],sb[1])
    with torch.inference_mode():pr=float(ordernet.ordering(h[0],lo,hi,ga,gb)[0].sigmoid())
    ok=bool(av[lo:hi+1].any(1).mean()>=.8 and not ca.get('observation_quarantine_overlap') and not cb.get('observation_quarantine_overlap'));state='unknown' if not ok or .35<pr<.65 else 'supported' if pr>=.65 else 'contradicted';order_checks.append(dict(pair=[ca['id'],cb['id']],guide_pair=[ga,gb],span=[lo,hi],probability=pr,eligible=ok,state=state,recomputed_for_different_preserved_window=True))
  out=copy.deepcopy(baseline);changed=[];byid={x['id']:x for x in out['cards']}
  for check in order_checks:
   if check['state']!='contradicted' or not check.get('eligible'):continue
   card=byid[check['pair'][1]]
   if card['display_status'] not in ['similar','difference'] or card.get('observation_quarantine_overlap'):continue
   card.update(status='difference_candidate',display_status='difference',message='동작 순서가 참고와 다르게 보여요. 표시된 두 구간을 확인해 보세요.',educational_wrong=False);card['candidate_message']=card['message'];card.setdefault('difference_components',[]).append('order');changed.append(card['id'])
  out['learned_order_checks']=order_checks;out['order_only_followup']=dict(posthoc=True,changed=changed,motion_and_shape_from_preserved_P=True);write(V/key/'after.json',out)
  outrows.append(dict(clip_id=r['clip_id'],kind=r['kind'],before=compact(baseline),after=compact(out),changed=changed,new_baseline_inference=new_baseline,result=str(V/key/'after.json'),sha256=sha(V/key/'after.json'),query_windows_identical_to_prior_candidate=same_windows,query_windows_preserved=[x.get('query_indices') for x in out['cards']]==[x.get('query_indices') for x in baseline['cards']],path_identical=out['path']==baseline['path'],tcn_identical=out['tcn_top3']==baseline['tcn_top3']))
 rt.consumer.refine=old_refine;pe.along_forearm=old_along;write(U/'ORDER_ONLY_COMPLETE.json',dict(rows=outrows,default_web_changed=False,posthoc=True,only_new_baseline_inferences=9,old69_rerun=False))
 event('order_only_followup_complete',conditions=len(outrows),changed=sum(bool(r['changed']) for r in outrows))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'ORDER_ONLY_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
