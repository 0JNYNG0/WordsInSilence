"""Posthoc use of observed partial motion, separate from whole-event completion."""
from phase_core import *
import copy,traceback
sys.path.insert(0,str(U));ev=load('_ep_evaluation_helpers',U/'evaluate.py')
def main():
 assert not (U/'PARTIAL_COMPLETE.json').exists();write(U/'PARTIAL_PLAN.json',dict(posthoc=True,reason='Primary phase locator finds14/21moving intervals but only4linked; do not erase valid observed partial movement. Reuse savedphaseoutputs and frozenPdirection on new localized spans. No new phase inference/training.',primary_seed=42,rule='All observed non-hold directions on relevant partial/complete spans must agree. New negative cue only if it corroborates the archived geometric contrary-direction evidence. No support/complete promotion, no coordinate repair. Conflicts remain unknown.',threshold=.65,min_observed=.8,min_frames=4,no_new_hyperparameter_search=True,old_model_protected=True,source_outputs=str(U/'FULLPATH_COMPLETE.json')))
 torch.set_num_threads(4);torch.set_num_interop_threads(1);base=prior.Network('coupled',read(P/'NORMALIZATION.json'));base.load_state_dict(torch.load(P/'runs/coupled_42/model.pt',weights_only=True,map_location='cpu'));base.eval();runtime={r['key']:r for r in ev.prepare_runtime()}
 # Existing shared relation contracts. They are data for actions, not new word-ID rules.
 spec=ev.rt.consumer.refine.__globals__;contracts=spec['CONTRACTS'];records=[r for r in read(U/'FULLPATH_COMPLETE.json')['rows'] if r['seed']==42];outrows=[]
 for r in records:
  key=r['clip_id']+'_'+r['kind'];c=runtime[key];saved=read(r['result']);before=c['baseline'];out=copy.deepcopy(saved);spans=saved['phase_events']+saved['phase_partials']
  # Manifest mapping is shared by all expressions; no target answer enters phase localization.
  wi=shared.WORDS.index(before['word']);fh,_=ev.frozen_forward(base,c['x'],wi);cache={};added=[]
  for card in out['cards']:
   m=card.get('motion_evidence');contract=contracts.get(card['id'])
   if not m or not m.get('span') or not contract:continue
   lo,hi=m['span'];evidence=[]
   for q in spans:
    if q['axis']!=2 or q['moving']!=contract['moving_hand']:continue
    a,b=max(lo,q['a']),min(hi,q['b'])
    if b-a+1<4:continue
    kk=(a,b,contract['moving_hand'],contract['anchor_arm'])
    if kk not in cache:
     g,ok,info=prior.geometry(c['x'],c['ts'],a,b,2,contract['moving_hand'],contract['anchor_arm'])
     with torch.inference_mode():pr=base.movement(torch.tensor(fh[None]),torch.tensor(g[None]),torch.tensor([[0,a,b]])).softmax(-1)[0].numpy()
     cache[kk]=dict(span=[a,b],class_index=int(pr.argmax()),confidence=float(pr.max()),eligible=bool(ok),complete_phase_chain=q.get('complete',False),observation=info)
    evidence.append(cache[kk])
   signs={q['class_index']-1 for q in evidence if q['eligible'] and q['confidence']>=.65 and q['class_index']!=1};expected=contract['expected_sign'];state='unknown' if len(signs)!=1 else 'supported' if signs=={expected} else 'contradicted';card['localized_direction_evidence']=dict(state=state,observations=evidence,whole_event_complete=False,partial_evidence_not_discarded=True)
   if state=='contradicted' and m.get('old_geometry_state')=='contradicted' and not card.get('observation_quarantine_overlap'):
    bad=[q['span'] for q in evidence if q['eligible'] and q['confidence']>=.65 and q['class_index']-1==-expected];card.update(status='difference_candidate',display_status='difference',message='관측된 이동 구간에서 참고와 반대 방향이 보여요. 표시된 부분의 이동 방향을 확인해 보세요.',educational_wrong=False);card['candidate_message']=card['message'];card.setdefault('difference_components',[])
    if 'localized_direction' not in card['difference_components']:card['difference_components'].append('localized_direction')
    card['localized_difference_spans']=bad;added.append(card['id'])
  assert ev.compact(out)['motion']==ev.compact(before)['motion'] and ev.compact(out)['shape']==ev.compact(before)['shape'];assert out['path']==before['path'] and out['tcn_top3']==before['tcn_top3'];dst=U/'partial_followup'/key/'after.json';write(dst,out)
  outrows.append(dict(clip_id=r['clip_id'],kind=r['kind'],added=added,before={k:v for k,v in ev.compact(before).items() if k not in ['motion','shape']},after={k:v for k,v in ev.compact(out).items() if k not in ['motion','shape']},result=str(dst),sha256=sha(dst),old_shape_direction_identical=True,tcn_dtw_preserved=True))
 write(U/'PARTIAL_COMPLETE.json',dict(rows=outrows,posthoc=True,default_web_changed=False));event('partial_evidence_verified',conditions=len(outrows),new_cues=sum(bool(r['added']) for r in outrows))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'PARTIAL_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
