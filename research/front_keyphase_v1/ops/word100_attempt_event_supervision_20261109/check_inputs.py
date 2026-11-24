from unit import *
def main():
 rows=load_rows();fit=[r for r in rows if r['role']=='fit'];dev=[r for r in rows if r['role']=='development'];checks=[]
 def ck(ok,msg):checks.append(msg);assert ok,msg
 ck(not ({r['person'] for r in fit}&{r['person'] for r in dev}),'record ID split')
 ck(not ({r['source_sha256'] for r in fit}&{r['source_sha256'] for r in dev}),'source SHA split')
 normal={r['clip_id']:r for r in rows if r['kind']=='full_normal'};omissions=[]
 reviewed={r['clip_id']:r for r in read(S/'REVIEW_INPUTS.json')['rows']};ann={r['clip_id']:r for r in read(S/'ANNOTATIONS.json')['rows']}
 for r in rows:
  n=normal[r['clip_id']];idx=r['source_indices'];ck(np.isfinite(r['raw']).all(),'finite '+r['kind']);ck(np.all(np.diff(r['timestamp_us'])>0),'time '+r['kind'])
  if r['kind'].startswith('close'):
   # Palms and wrist/arm trajectory unchanged; reversal is accounted by mapping.
   anchors=[1,2,3,4,5,6,9,14,18,22,26,30,35,39,43,47]
   ck(np.allclose(r['raw'][:,:102].reshape(-1,51,2)[:,anchors],n['raw'][idx,:102].reshape(-1,51,2)[:,anchors]),'shape-only palm preservation')
  if r['kind'] in ['close_R','close_L','close_both','close_reverse','reverse_main']:
   ck(np.array_equal(r['sy'],n['sy'][idx]),'attempt identity mapped, not normal correctness')
  if r['kind'] in ['all_hand_missing','mask_final_event']:ck(not r['pmask'].any() and not r['sw'].any(),'missing excluded from correctness training')
  if r['kind']=='omit_final_event':
   removed=[k for k in range(13) if GROUPS[k] in r['removed_groups']];gold=ann[r['clip_id']]['stages'];anchors=[i for k in removed for i in gold.get(str(k),[])]
   ck(not set(anchors)&set(r['frame_id']),'all reviewed final-event anchors removed')
   ck(np.all(r['present'][r['removed_groups']]==0),'separate absent-event target')
   omissions.append(dict(clip_id=r['clip_id'],role=r['role'],kept_pre_event_frame=r['cut'],removed_reviewed_anchors=anchors,tail_source_frames=r['frame_id'][-6:].tolist(),sheet=reviewed[r['clip_id']]['sheet']))
 write(U/'EVENT_REVIEW.json',dict(rows=omissions,reviewer='Same Codex inspected original RGB sheets for all12 omissions before new model evaluation',scope='Whole later visual action removed after last reviewed predecessor frame; not independent educational error truth; sparse RGB anchors do not label every event boundary',excluded=['optional wrist hold','unfixed repetition count','exact fingertip contact']))
 write(U/'INPUT_CHECK.json',dict(status='PASS',checks=len(checks),items=checks,fit_cases=len(fit),dev_cases=len(dev),omissions=len(omissions),original_record_ids={'fit':list(range(1,10)),'dev':[11,12,13]}));event('inputs_checked',checks=len(checks),omissions=len(omissions))
if __name__=='__main__':main()
