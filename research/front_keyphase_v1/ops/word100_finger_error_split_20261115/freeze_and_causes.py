from common_split import *
from datetime import datetime
diag=read(U/'diagnostic_RESULTS.json');hist=read(U/'historic_RESULTS.json')
rows=[]
for role,d in [('diagnostic',diag),('historic',hist)]:
 for r in d['rows']:
  for j in range(4):
   y=r['shape'][j];pr=r['predictions']
   if y>=0 and pr['available'][j]and pr['base'][j]>=0 and pr['base'][j]!=y:
    rows.append(dict(role=role,clip_id=r['clip_id'],frame=r['frame'],hand=r['hand'],finger=j,truth=y,quality_truth=r['quality'][j],base=pr['base'][j],shape=pr['shape_only'][j],combined=pr['combined'][j],quality=pr['quality'][j],cause='visibly_wrong_chain'if r['quality'][j]==0 else'own_chain_shape_decision'if r['quality'][j]==1 else'chain_quality_not_adjudicated'))
write(U/'ERROR_CAUSES.json',dict(rows=rows,notes=['Single-frame shape truth in rotation/change should be checked with source context, not changed to fit model output','Quality guard suppresses unreliable evidence; does not reconstruct a finger or prove sign correct','Quality reserved positive support is zero; cannot claim quality sensitivity generalized']))
pins={str(p):sha(p)for p in list((U/'models').rglob('*.pt'))+[U/'LABELS.json',U/'split_runtime.py',U/'split_model.py',U/'TRAIN_CONFIG.json',U/'evaluate_split.py']}
write(U/'CANDIDATE_FROZEN.json',dict(at=datetime.now().isoformat(),selected_research_mode='combined',comparison_modes=['base','shape_only','compatible_shape','quality_only','combined'],selection_reason='Combined reduces diagnostic wrong5to2 with11/910 historic correct abstentions, within2% loss gate; research-only. Shape-only produces13 historic errors and is not eligible. Compatible gains coverage but does not meet predefined wrong-reduction gate.',quality_release_block='No confirmed reserved16 corrupt-chain positives; quality cannot be promoted even if normal checks pass',no_updates_after_reserve=True,pins=pins))
