"""Frozen rubric, literal RGB drafts and unchanged full-interval ROI guard."""
from quality import read,save,sha,U,OLD,ALL,SIDES,DIRS
import numpy as np
def main():
 root=U/'fresh';rows=read(root/'INPUTS.json')['rows'];vis={r['alias']:r for r in read(root/'VISUAL_INDEX.json')['rows']}
 primary=read(root/'RGB_LABELS_A.json')['rows']+read(root/'RGB_LABELS_B.json')['rows'];secondary={r['alias']:r for r in read(root/'RGB_LABELS_ROOT.json')['rows']}
 assert len(primary)==32 and len({r['alias'] for r in primary})==32;prim={r['alias']:r for r in primary};features=read(OLD/'RGB_RUBRIC.json')['features'];audit=[];cross=[];pins={}
 for row in rows:
  alias=row['alias'];p=prim[alias];q=secondary.get(alias);v=vis[alias];a=dict(np.load(row['input'],allow_pickle=False));ob=dict(np.load(row['observation'],allow_pickle=False));n=len(a['x']);y=np.zeros((n,len(ALL)),np.float32);mask=np.zeros_like(y,bool);box=row['box'];look={int(fid):i for i,fid in enumerate(a['frame_id'])};entries=[]
  def inside(i,s):
   off=54 if s=='R' else 33;xy=ob['xy_px'][i,off:off+21];valid=ob['observed'][i,off:off+21]&np.isfinite(xy).all(1);xy=xy[valid]
   return bool(len(xy) and ((xy[:,0]>=box[0])&(xy[:,0]<box[2])&(xy[:,1]>=box[1])&(xy[:,1]<box[3])).all())
  def add(feature,val,second,i,inds,kind):
   assert val in [0,1,None] and second in [0,1,None,'not_reviewed']
   initial=val;status='single_Codex_RGB_draft'
   if second!='not_reviewed':
    cross.append(dict(alias=alias,feature=feature,frame=int(a['frame_id'][i]),primary=val,secondary=second))
    if val is not None and val==second:status='same_Codex_cross_agreement'
    else:val=None;status='cross_unknown_or_disagreement'
   sides=SIDES if feature=='hands_overlap' else [feature[0]];roi=all(inside(ii,s) for ii in inds for s in sides)
   if val is not None and not roi:val=None;status='outside_or_no_direct_hand_ROI_evidence'
   if val is not None:y[i,ALL.index(feature)]=val;mask[i,ALL.index(feature)]=True
   entries.append(dict(feature=feature,frame=int(a['frame_id'][i]),kind=kind,initial=initial,secondary=second,value=val,status=status,interval_frame_ids=[int(a['frame_id'][inds[0]]),int(a['frame_id'][inds[-1]])]))
  assert {r['ordinal'] for r in p['static']}=={3,6,9}
  for st in p['static']:
   ordinal=st['ordinal'];i=look[v['frames'][ordinal]['frame_id']];qs=next((z for z in q['static'] if z['ordinal']==ordinal),None) if q else None;assert len(st['values'])==9
   for j,f in enumerate(features):add(f,st['values'][j],qs['values'][j] if qs else 'not_reviewed',i,[i],'static')
  assert {(z['start_ordinal'],z['end_ordinal']) for z in p['motion']}=={(0,3),(3,6),(6,9),(9,12)}
  for mot in p['motion']:
   lo,hi=mot['start_ordinal'],mot['end_ordinal'];sec=(v['frames'][lo]['seconds']+v['frames'][hi]['seconds'])/2;i=int(np.abs(a['timestamp_us']/1e6-sec).argmin());inds=list(range(look[v['frames'][lo]['frame_id']],look[v['frames'][hi]['frame_id']]+1));qs=next((z for z in q['motion'] if z['start_ordinal']==lo),None) if q else None
   for side in SIDES:
    direct=mot[side];assert direct in read(OLD/'RGB_RUBRIC.json')['motion_values']
    for di in DIRS:
     value=None if direct in [None,'mixed'] else int(di in direct);second='not_reviewed' if qs is None else None if qs[side] in [None,'mixed'] else int(di in qs[side]);add(side+'_motion_'+di,value,second,i,inds,'interval')
  for side in SIDES:
   ia,ib=ALL.index(side+'_index_only'),ALL.index(side+'_open_four');assert not ((y[:,ia]==1)&(y[:,ib]==1)&mask[:,ia]&mask[:,ib]).any(),alias
  dest=root/'targets'/f'{alias}.npz';dest.parent.mkdir(exist_ok=True);assert not dest.exists();np.savez_compressed(dest,y=y,target_mask=mask);pins[str(dest)]=sha(dest);audit.append(dict(alias=alias,known=int(mask.sum()),rows=entries))
 joint=[z for z in cross if z['primary'] is not None and z['secondary'] is not None];bad=[z for z in joint if z['primary']!=z['secondary']]
 for p in [root/'RGB_LABELS_A.json',root/'RGB_LABELS_B.json',root/'RGB_LABELS_ROOT.json',root/'INPUTS.json',root/'SELECTION.json',OLD/'RGB_RUBRIC.json',__file__]:pins[str(p)]=sha(p)
 save('fresh/LABEL_AUDIT.json',dict(status='RECONCILED_PROVISIONAL_PHYSICAL_LABELS',human_truth=False,roi_policy='Same previous guard. Every original frame of interval must have direct observed hand points inside fixed ROI; no body-wrist fallback. All clips/frames retained.',cross_reviewed_videos=8,joint_known=len(joint),known_disagreements=len(bad),known_disagreement_rate=len(bad)/max(1,len(joint)),disagreements=bad,cross=cross,rows=audit,known=sum(x['known'] for x in audit),pins=pins))
 print('labels',sum(x['known'] for x in audit),'cross',len(bad),'/',len(joint),flush=True)
if __name__=='__main__':main()
