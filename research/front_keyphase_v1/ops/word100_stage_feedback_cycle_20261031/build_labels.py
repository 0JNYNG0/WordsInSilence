"""Assemble provisional RGB labels without word-conditioned inference or score selection."""
from prepare_bundle import *
STATIC=read(U/'RGB_RUBRIC.json')['features']
MOTION=[f'{s}_motion_{d}' for s in ['R','L'] for d in ['up','down','left','right']]
TARGETS=STATIC+MOTION
def label_rows(name):
 d=read(U/'review'/name)
 return d['rows']
def main():
 rows=read(U/'INPUTS.json')['rows'];vis={r['alias']:r for r in read(U/'VISUAL_INDEX.json')['rows']}
 pri=label_rows('RGB_LABELS_A.json')+label_rows('RGB_LABELS_B.json');sec=label_rows('RGB_LABELS_SECONDARY.json')
 assert len(pri)==80 and len({r['alias'] for r in pri})==80
 pp={r['alias']:r for r in pri};ss={r['alias']:r for r in sec};assert len(ss)==24
 old=read(OPS/'word100_signrep_temporal_head_20261029/LABELS.json')
 old_index=read(OPS/'word100_signrep_temporal_head_20261029/INPUTS.json')['rows'];oldmap={r['clip_id']:r['alias'] for r in old_index}
 latest=read(OPS/'word100_signrep_temporal_head_20261029/LABEL_REVIEW.json')
 disagreements=[];reviewcells=[];out=[];audit=[];all_support={}
 for r in rows:
  a=r['alias'];v=vis[a];frame_by={f['ordinal']:f for f in v['frames']};data=dict(np.load(r['input'],allow_pickle=False));n=len(data['x'])
  values={};forced_unknown=set();evidence=[];window_requirements={}
  # Prior validated values remain provisional, with explicit known downgrade guards.
  oa=oldmap.get(r['clip_id'])
  if oa:
   for k in latest['uncertain_downgrades']:
    if k['alias']==oa:forced_unknown.add((k['frame_id'],k['feature']))
   for z in old['static']:
    if z['alias']==oa:
     for f,val in z['labels'].items():
      if val is not None:values.setdefault((z['frame_id'],f),[]).append((val,'prior_RGB_rechecked'))
   for z in old['sequences']:
    if z['source_alias']==oa:
     for zf in z['frames']:
      for f,val in zf['direct'].items():
       target=z['side']+'_'+f
       if val is not None and target in TARGETS:values.setdefault((zf['frame_id'],target),[]).append((val,'prior_RGB_sequence'))
  p=pp[a];second=ss.get(a)
  assert {z['ordinal'] for z in p['static']}=={3,6,9}
  for z in p['static']:
   ordinal=z['ordinal'];assert len(z['values'])==len(STATIC)
   q=next((q for q in second['static'] if q['ordinal']==ordinal),None) if second else None
   for j,(f,pval) in enumerate(zip(STATIC,z['values'])):
    assert pval in (0,1,None)
    val=pval;state='single_RGB_draft'
    if q:
     sval=q['values'][j];assert sval in (0,1,None)
     reviewcells.append(dict(alias=a,ordinal=ordinal,feature=f,primary=pval,secondary=sval))
     if pval==sval and pval is not None:state='two_Codex_drafts_agree'
     else:
      val=None;state='unknown_or_disagreement';forced_unknown.add((frame_by[ordinal]['frame_id'],f))
      if pval is not None or sval is not None:disagreements.append(dict(alias=a,ordinal=ordinal,feature=f,primary=pval,secondary=sval))
    fid=frame_by[ordinal]['frame_id']
    if val is not None:values.setdefault((fid,f),[]).append((val,state))
    evidence.append(dict(frame_id=fid,seconds=frame_by[ordinal]['seconds'],feature=f,value=val,status=state,kind='static',image=frame_by[ordinal]['file']))
  assert {(z['start_ordinal'],z['end_ordinal']) for z in p['motion']}=={(0,3),(3,6),(6,9),(9,12)}
  for z in p['motion']:
   start,end=z['start_ordinal'],z['end_ordinal'];t=(frame_by[start]['seconds']+frame_by[end]['seconds'])/2;idx=int(np.abs(data['timestamp_us']/1e6-t).argmin());fid=int(data['frame_id'][idx])
   q=next((q for q in second['motion'] if q['start_ordinal']==start and q['end_ordinal']==end),None) if second else None
   for side in ['R','L']:
    direct=z[side];assert direct in read(U/'RGB_RUBRIC.json')['motion_values']
    status='single_RGB_draft'
    for d in ['up','down','left','right']:
     f=side+'_motion_'+d;val=None if direct in [None,'mixed'] else int(d in direct)
     window_requirements[(fid,f)]=(frame_by[start]['frame_id'],frame_by[end]['frame_id'])
     if q:
      sv=None if q[side] in [None,'mixed'] else int(d in q[side])
      reviewcells.append(dict(alias=a,start=start,end=end,feature=f,primary=val,secondary=sv))
      if val is not None and val==sv:status='two_Codex_drafts_agree'
      else:
       if val is not None or sv is not None:disagreements.append(dict(alias=a,start=start,end=end,feature=f,primary=val,secondary=sv))
       val=None;status='unknown_or_disagreement'
     if val is not None:values.setdefault((fid,f),[]).append((val,status))
     else:forced_unknown.add((fid,f))
     evidence.append(dict(frame_id=fid,seconds=float(data['timestamp_us'][idx]/1e6),feature=f,value=val,status=status,kind='interval_direction',direction=direct,interval_frames=[frame_by[start]['frame_id'],frame_by[end]['frame_id']],interval_seconds=[frame_by[start]['seconds'],frame_by[end]['seconds']]))
  y=np.zeros((n,len(TARGETS)),np.float32);mask=np.zeros_like(y,bool);roi_drops=[];conflicts=[]
  raw=read_observation(r['observation']);lookup={int(f):i for i,f in enumerate(data['frame_id'])};b=r['box']
  def in_roi(i,f):
   sides=['R','L'] if f=='hands_overlap' else [f[0]]
   for side in sides:
    off=54 if side=='R' else 33;good=raw['observed'][i,off:off+21]&np.isfinite(raw['xy_px'][i,off:off+21]).all(-1)
    points=raw['xy_px'][i,off:off+21][good]
    if len(points):
     within=(points[:,0]>=b[0])&(points[:,0]<b[2])&(points[:,1]>=b[1])&(points[:,1]<b[3])
     if not within.all():return False
    else:
     # A pose wrist does not establish that a hand is inside the RGB ROI.
     # Keep the frame/input but withhold this target until image bounds exist.
     return False
   return True
  for (fid,f),vs in values.items():
   if f not in TARGETS:continue
   if (fid,f) in forced_unknown:continue
   i=lookup[fid];distinct={v for v,_ in vs}
   if len(distinct)>1:conflicts.append(dict(frame_id=fid,feature=f,values=vs));continue
   a0,b0=window_requirements.get((fid,f),(fid,fid))
   check_indices=[ii for ff,ii in lookup.items() if a0<=ff<=b0]
   if not all(in_roi(ii,f) for ii in check_indices):roi_drops.append(dict(frame_id=fid,feature=f,window_frames=[a0,b0]));continue
   y[i,TARGETS.index(f)]=next(iter(distinct));mask[i,TARGETS.index(f)]=True
  for s in ['R','L']:
   ia,ib=TARGETS.index(s+'_index_only'),TARGETS.index(s+'_open_four')
   assert not ((y[:,ia]==1)&(y[:,ib]==1)&mask[:,ia]&mask[:,ib]).any(),a
  targetfile=U/'targets_v2'/f'{a}.npz';targetfile.parent.mkdir(exist_ok=True);assert not targetfile.exists();np.savez_compressed(targetfile,y=y,target_mask=mask)
  out.append(dict(**r,targets_file=str(targetfile),targets_sha256=sha(targetfile),known_cells=int(mask.sum())))
  audit.append(dict(alias=a,observations=evidence,conflicts=conflicts,roi_exclusions=roi_drops,secondary_review=bool(second)))
 for j,f in enumerate(TARGETS):
  support={}
  for val in [0,1]:
   match=[]
   for r in out:
    if r['role']!='fit':continue
    z=np.load(r['targets_file'],allow_pickle=False);n=int(((z['y'][:,j]==val)&z['target_mask'][:,j]).sum())
    if n:match.append(dict(alias=r['alias'],word=r['word_id'],person=r['person'],cells=n))
   support[str(val)]=dict(words=sorted({x['word'] for x in match}),people=sorted({x['person'] for x in match}),cells=sum(x['cells'] for x in match),clips=len(match))
  passed=all(len(support[str(v)]['words'])>=4 and len(support[str(v)]['people'])>=3 for v in [0,1])
  all_support[f]=dict(passed=passed,by_value=support)
 selected=[f for f in TARGETS if all_support[f]['passed']]
 # Cross-review is a quality signal, not an independent accuracy certificate.
 jointly=[x for x in reviewcells if x['primary'] not in [None,'mixed'] and x['secondary'] not in [None,'mixed']]
 bad=[x for x in jointly if x['primary']!=x['secondary']]
 save('LABEL_AUDIT.json',dict(status='PROVISIONAL_LABELS_RECONCILED',targets=TARGETS,selected_targets=selected,support=all_support,rows=audit,disagreements=disagreements,reviewed_cells=len(reviewcells),joint_known=len(jointly),known_disagreements=len(bad),known_disagreement_rate=len(bad)/max(len(jointly),1),bad=bad,truth_status='Same Codex RGB draft; no human ground truth. Conflicting or uncertain audit cells withheld. ROI gate is geometric safety, not semantic visibility.'))
 if len(bad)/max(len(jointly),1)>.05:
  save('LABEL_REVIEW_REQUIRED.json',dict(reason='more_than_5pct_known_cross_review_disagreement',bad=bad));raise RuntimeError('Resolve RGB review disagreement before training')
 assert len(selected)>=5,selected
 save('LEARNING_INPUTS.json',dict(rows=out,all_targets=TARGETS,targets=selected,target_indices=[TARGETS.index(f) for f in selected],labels_status='PROVISIONAL_NOT_HUMAN_TRUTH',source_pins={str(p):sha(p) for p in [U/'SELECTION.json',U/'RGB_RUBRIC.json',U/'review/RGB_LABELS_A.json',U/'review/RGB_LABELS_B.json',U/'review/RGB_LABELS_SECONDARY.json',U/'review/guide_stage_map.json']}))
 print(json.dumps(dict(selected=selected,not_supported=[f for f in TARGETS if f not in selected],known=sum(r['known_cells'] for r in out),cross_review=len(jointly),conflicts=len(bad)),ensure_ascii=False))
if __name__=='__main__':main()
