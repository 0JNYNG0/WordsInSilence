"""Sparse RGB observations entered before candidate predictions, not DTW gold."""
from common import *

def main():
 assert not (U/'FROZEN.json').exists()
 reviews={r['clip_id']:r for r in read(U/'REVIEW_INPUTS.json')['rows']}
 ready=read(V/'FEEDBACK_READY.json');annotations=[];conflicts=[];rows=[]
 # person: (guide-index -> exact reviewed RGB frames, direction endpoint tuples)
 stages={
 '0739':{
 1:({0:[48,52,57,61],1:[66,70],2:[79,84,88]},[(61,70,0,-1),(70,79,1,1),(48,57,0,0),(79,88,1,0)]),
 4:({0:[73,79,86],1:[100],2:[113,120,127]},[(86,100,0,-1),(100,113,1,1),(113,127,1,0)]),
 7:({0:[40,44,49],1:[63],2:[72,77]},[(49,63,0,-1),(63,72,1,1),(40,49,0,0)]),
 11:({0:[54,60,65],1:[77],2:[89,95,100]},[(65,77,0,-1),(77,89,1,1),(89,100,1,0)]),
 12:({0:[46,52,57],1:[69,74],2:[85,91,97]},[(57,69,0,-1),(74,85,1,1),(69,74,1,0)]),
 13:({0:[60,67,73],1:[86,93],2:[106,112]},[(73,86,0,-1),(93,106,1,1),(86,93,1,0)])},
 '1133':{
 1:({3:[50,53,57,61],4:[65,68,72,76,80]},[(50,61,2,1),(65,76,2,0)]),
 4:({3:[76,79,83,86,90],4:[93,97,100,104]},[(79,93,2,1),(93,104,2,0)]),
 7:({3:[46,50,54,58],4:[65,69,73]},[(50,65,2,1),(65,73,2,0)]),
 11:({3:[53,56,59,63],4:[66,70,73,76]},[(56,66,2,1),(66,76,2,0)]),
 12:({3:[65,69,72,76],4:[79,83,86]},[(69,79,2,1),(79,86,2,0)]),
 13:({3:[55,59,63,67],4:[72,76,80,84]},[(59,72,2,1),(72,84,2,0)])},
 '1567':{
 1:({5:[52],6:[61],7:[69],9:[82,86,90,95,99]},[(56,61,4,-1),(61,69,4,1),(69,82,4,-1),(82,95,4,0)]),
 4:({5:[65],6:[76],7:[88],9:[105,111,116]},[(65,76,4,-1),(76,88,4,1),(88,105,4,-1),(105,116,4,0)]),
 7:({5:[36],6:[41],7:[51,56],9:[66,71,76]},[(36,41,4,-1),(41,51,4,1),(56,66,4,-1),(66,71,4,0)]),
 11:({5:[50],6:[58],7:[70],8:[81]},[(50,58,4,-1),(58,70,4,1),(70,81,4,-1)]),
 12:({5:[48],6:[59],7:[70],9:[86,91]},[(48,59,4,-1),(59,70,4,1),(70,86,4,-1),(86,91,4,0)]),
 13:({5:[36,42],6:[48],7:[60,65],9:[77,83,89]},[(42,48,4,-1),(48,60,4,1),(65,77,4,-1),(77,89,4,0)])},
 '2523':{
 1:({10:[35,39],12:[44,48,53,57]},[(39,44,3,1),(48,57,3,0)]),
 4:({10:[76,80,85,89],11:[93],12:[97,101,105]},[(89,97,3,1),(97,105,3,0)]),
 7:({10:[40,44,47],11:[50],12:[53,56,59]},[(47,53,3,1),(53,59,3,0)]),
 11:({10:[51,54],11:[58],12:[61,64,67,70,73]},[(54,61,3,1),(64,73,3,0)]),
 12:({10:[49,53,56],11:[60],12:[63,67,70,74,77]},[(56,63,3,1),(67,77,3,0)]),
 13:({10:[59],11:[64],12:[69,74,79]},[(59,69,3,1),(69,79,3,0)])}}
 # Avoid assigning unseen fingers at the top edge of the review crop.
 ancestor_shape={1:[56,65,73],4:[70,82,93],7:[36,41,46,51,56,61],11:[50,54,58,62,66,73,77],12:[48,54,59,64,70,75,81],13:[36,42,48,54,65,71,77,83,89]}
 for r in ready['rows']:
  assert sha(r['path'])==r['sha256'];z=np.load(r['path'],allow_pickle=False);data={k:z[k].copy() for k in z.files}
  oldy=data['y'].copy();oldm=data['ym'].copy();oldsy=data['sy'].copy();oldsw=data['sw'].copy()
  point=[];pairs=[];orders=[];cid=r['clip_id'];w=cid[11:15] # checked below via word token
  w=cid.split('_')[2][4:];p=int(r['person']);direct=np.zeros_like(data['ym']);stage_gold=np.full(len(data['x']),-1,np.int64)
  if cid in reviews:
   review=reviews[cid];assert sha(review['source'])==review['source_sha256'];assert sha(review['raw'])==review['raw_sha256']
   sf,pa=stages[w][p];visible=set(review['frames']);index={int(f):i for i,f in enumerate(data['frame_id'])}
   def shape(f,values):
    assert f in visible and f in index
    i=index[f]
    for j,y in values.items():
     if oldm[i,j] and oldy[i,j]!=y:conflicts.append(dict(clip_id=cid,frame=f,kind='attribute',target=j,old=float(oldy[i,j]),new=y))
     data['y'][i,j]=y;data['ym'][i,j]=1;direct[i,j]=1
    point.append(dict(frame=f,attributes={str(j):y for j,y in values.items()}))
   for s,frames in sf.items():
    for f in frames:
     assert f in visible and f in index,(cid,f,sorted(visible));i=index[f];stage_gold[i]=s
     if oldsw[i]>0 and oldsy[i]!=s:conflicts.append(dict(clip_id=cid,frame=f,kind='stage',old=int(oldsy[i]),new=s,old_weight=float(oldsw[i])))
     data['sy'][i]=s;data['sw'][i]=1
     if w=='0739':shape(f,{1:int(s==0),3:int(s!=0)})
     elif w=='1133':shape(f,{1:1,3:0,2:0,4:1})
   if w=='1567':
    for f in ancestor_shape[p]:shape(f,{1:0,2:0,3:1,4:1})
   for a,b,axis,sign in pa:
    assert a in visible and b in visible and a<b
    pairs.append(dict(a=index[a],b=index[b],frame_a=a,frame_b=b,axis=axis,target=sign+1))
   if w!='1567':
    distinct=[s for s in sorted(sf) if s!=11] # transition not a required separate held pose
    for s,t in zip(distinct,distinct[1:]):
     a=sf[s][len(sf[s])//2];b=sf[t][len(sf[t])//2];assert a<b
     orders.append(dict(a=index[a],b=index[b],frame_a=a,frame_b=b,guide_a=s,guide_b=t))
   annotations.append(dict(clip_id=cid,role=r['role'],person=p,word=w,stages=sf,shape=point,pairs=pairs,orders=orders,review_sheet=review['sheet']))
  data.update(old_y=oldy,old_ym=oldm,direct_ym=direct,stage_gold=stage_gold)
  path=U/'inputs'/f'{cid}.npz';path.parent.mkdir(exist_ok=True);np.savez_compressed(path,**data)
  rows.append(dict(r,path=str(path),sha256=sha(path),pairs=pairs,orders=orders,source_input=r['path'],source_sha256=r['sha256']))
 anno=dict(reviewer='Codex RGB visual review; no model predictions shown; not independent human/expert gold',
  scopes=['four visible fingers open/closed, not thumb correctness','five body-relative direction axes','description-linked sparse stages, not continuous frame gold'],
  axis_names=['right_wrist_up','right_forearm_upright','R_palm_toward_L_wrist','hands_separate','R_up_relative_to_L'],
  classes=['decrease','no_distinct_change','increase'],rows=annotations,conflicts=conflicts,
  repetition_policy='Repeated ancestor phases collapsed for semantic score; exact repeat count and reversed alternation not educational error gold.',
  uncertainty='No labels for Egypt edge-on four-finger detail, crop-clipped high-hand shape, unseen between-anchor frames or physical contact.')
 write(U/'ANNOTATIONS.json',anno);write(U/'READY.json',dict(rows=rows))
 protected=[V/'feedback_runs/pose_transfer_42/model.pt',D/'runs/42/model.pt',V/'FEEDBACK_NORMALIZATION.json',J/'feedback_model.py']
 # Preserve actual deployed files and both original TCN checkpoint families via their recorded pins.
 previous=read(O/'word100_positive_direction_20261109/FROZEN.json')
 for path in previous.get('pins',{}):
  if Path(path).exists():protected.append(Path(path))
 cfg=dict(seeds=[42,43,44],primary_seed=42,conditions=['frozen_heads','finetune'],epochs=80,batch_size=6,
  lr_base=.0001,lr_heads=.001,weight_decay=.01,gradient_clip=2.,losses=dict(attributes=1.,stage=.35,prototype=.1,direction=.5,order=.25),
  initialization='deployed pose_transfer_42 for every condition and seed; final epoch only',
  reverse_training='Alternate normal and main-interval temporal reversal for reviewed FIT clips; label indices follow content; sign reverses. No prototype loss on reversed inputs.',
  evaluation='DEV11-13 fixed; both normal and synthetic controls; reviewed endpoints are supplied to pair tests, not autonomous window detection.',
  uncertainty='minimum 80% required-axis observations through span; unsupported predictions are withheld; direction confidence .65 and order .65/.35 fixed',
  semantic_stage_groups=[[0],[1],[2],[3],[4],[5,7],[6,8,9],[10],[11],[12]],
  deployment='offline only; old demo phase head depends on old latent and cannot be silently reused with finetuned base',
  source_pins={str(p):sha(p) for p in protected},data_pins={r['path']:r['sha256'] for r in rows},
  annotations_sha256=sha(U/'ANNOTATIONS.json'),normalization='unchanged FIT-only stats; no DEV tuning',private_used=False)
 write(U/'FROZEN.json',cfg)
 event('labels_prepared',reviewed_clips=len(annotations),shape_cells=int(sum(np.load(r['path'])['direct_ym'].sum() for r in rows)),pairs=sum(len(r['pairs']) for r in rows),orders=sum(len(r['orders']) for r in rows),conflicts=len(conflicts))
if __name__=='__main__':main()
