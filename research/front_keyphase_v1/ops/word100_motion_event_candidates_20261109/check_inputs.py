from core import *
def main():
 rows=load_rows();checks=[]
 def ck(b,s):assert bool(b),s;checks.append(s)
 fit=[r for r in rows if r['role']=='fit'];dev=[r for r in rows if r['role']=='development'];ck(not {r['source_sha256'] for r in fit}&{r['source_sha256'] for r in dev},'source split');ck({r['person'] for r in fit}==set(range(1,10)) and {r['person'] for r in dev}=={11,12,13},'record split')
 normal={r['clip_id']:r for r in rows if r['kind']=='full_normal'}
 for r in rows:
  ck(np.isfinite(r['g']).all() and np.all(np.diff(r['timestamp_us'])>0),'finite input and monotone time');ck(r['g'].shape[1]==66,'feature dimensionality')
  if r['kind'] in ['close_R','close_L','close_both']:ck(np.array_equal(r['g'],normal[r['clip_id']]['g']),'finger-change geometry invariance')
  if r['kind']=='actual_crop':ck(r['role']=='fit' and r['person'] in range(1,10),'actual crop is FIT only');ck(np.array_equal(r['frame_id'],normal[r['clip_id']]['frame_id']),'same frame labels no DTW transfer')
  if r['kind'] in ['mask_final_event','all_hand_missing']:ck(not r['weight'].any(),'missing sensor not normal supervision')
 old=read(P/'EVAL_INPUTS.json')['rows'];by={(r['clip_id'],r['kind']):r for r in old}
 for r in old:
  if r['kind']!='right_closed_hand_main':continue
  x=np.load(r['path']);n=np.load(by[(r['clip_id'],'crop_normal')]['path']);ck(np.array_equal(geometry(x['raw_x'],x['timestamp_us']),geometry(n['raw_x'],n['timestamp_us'])),'old whole-hand geometry exact invariance')
 for p,h in read(N/'FROZEN.json')['protected'].items():
  if isinstance(h,str) and len(h)==64:ck(sha(p)==h,'protected '+p)
 write(N/'INPUT_CHECK.json',dict(status='PASS',checks=len(checks),items=checks,actual_cropped_fit=sum(r['kind']=='actual_crop' for r in fit),no_private=True))
 event('inputs_checked',checks=len(checks))
if __name__=='__main__':main()
