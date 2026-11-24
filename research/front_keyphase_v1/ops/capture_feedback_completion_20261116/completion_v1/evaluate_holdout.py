from visible_lesson import *
def main():
 torch.set_num_threads(2);out=V/'holdout';assert not(out/'RESULT.json').exists();r=read(out/'EXTRACTED.json')['rows'][0]
 write(out/'SOURCE_TRUTH.json',dict(clip_id=r['clip_id'],source_sha256=r['sha256'],reviewer='Codex native RGB review, not sign-language expert',
  evidence='review/dense.jpg: frames40..76 every2; right visible upright open, left open flatter hand. Two approaches:44..46,56..58; later quiet hold.',
  approximate_arrival_spans=[[44,46],[56,58]],right_shape=0,left_shape=0,right_projected_plane=0,left_projected_plane=1,
  physical_contact=None,whole_word_correct=None,role='Excluded from candidate FIT and selection; labels recorded before candidate output.'))
 raw=dict(np.load(r['raw_path']));net=VisibleLesson();rows=[]
 for kind in['normal','first_only','constant','missing','fast','slow','scale']:
  z={k:v.copy()for k,v in raw.items()}
  if kind=='first_only':
   for k in['xy_px','score','timestamp_us','frame_id']:z[k]=z[k][:48]
  if kind=='constant':z['xy_px']=np.repeat(z['xy_px'][46:47],len(z['frame_id']),axis=0)
  if kind=='missing':z['xy_px'][:,91:]=np.nan
  if kind=='fast':z['timestamp_us']=(z['timestamp_us']*.65).astype(np.int64)
  if kind=='slow':z['timestamp_us']=(z['timestamp_us']*1.5).astype(np.int64)
  if kind=='scale':z['xy_px']=z['xy_px']*.75+np.array([90,40],np.float32)
  q=net.analyze(z);supported=all(c['status']=='supported'for c in q['criteria']);expected=kind in['normal','fast','slow','scale']
  rows.append(dict(kind=kind,check=supported==expected,supported=supported,output=q))
 write(out/'RESULT.json',dict(rows=rows,passed=sum(r['check']for r in rows),total=len(rows),web_admitted=False))
 for r in rows:print(r['kind'],r['check'],{c['id']:c['status']for c in r['output']['criteria']},r['output']['events'])
if __name__=='__main__':main()
