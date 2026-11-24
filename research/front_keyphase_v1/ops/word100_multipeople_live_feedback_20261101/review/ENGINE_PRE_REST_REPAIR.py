"""Local, target-conditioned multi-person reference comparison.

The bank learns empirical normal ranges on public FIT only. This is not a
linguistic-correctness oracle. Every reference keeps its complete order.
"""
from pathlib import Path
import json,time,hashlib
from functools import lru_cache
import numpy as np
U=Path(__file__).resolve().parent; O=U.parent
FAMILIES=['R_shape','L_shape','R_angles','L_angles','hand_relation','arm_path','hand_axis']
LABELS={'R_shape':'오른손 모양(학습한 항목)','L_shape':'왼손 모양(학습한 항목)','R_angles':'오른 손가락 굽힘 관측','L_angles':'왼 손가락 굽힘 관측','hand_relation':'양손의 상대 위치','arm_path':'손·팔의 이동과 위치','hand_axis':'손목에서 손가락 뿌리로 향하는 방향'}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def load_arrays(row):
 with np.load(row['input'],allow_pickle=False) as z:a={k:z[k].copy() for k in ['x','mask','rgb','frame_id','timestamp_us']}
 with np.load(row['probability_file'],allow_pickle=False) as z:a['probabilities']=z['probabilities'].copy()
 a['feature_names']=read(U/'PUBLIC_INPUTS.json')['feature_names'];return a

def bridged(mask,ts,limit=.2):
 """Evidence support for an RGB/time estimate, never interpolate coordinates."""
 result=mask.copy(); good=np.flatnonzero(mask)
 for a,b in zip(good[:-1],good[1:]):
  if ts[b]-ts[a]<=limit:result[a:b+1]=True
 return result

def shape_evidence(a,side):
 names=a['feature_names'];m=np.asarray(a['mask'],bool)&np.isfinite(a['x'])
 wi=[names.index(side+'_hand_wrist_'+d) for d in ['x','y']]
 ai=[i for i,n in enumerate(names) if n.startswith(side+'_') and '_angle' in n and 'elbow' not in n]
 native=m[:,wi].all(1)&(m[:,ai].sum(1)>=10)
 context=bridged(native,a['timestamp_us']/1e6)
 return native,context&~native

def channels(a):
 x=np.asarray(a['x'],float);m=np.asarray(a['mask'],bool)&np.isfinite(x);x=np.where(m,x,0)
 p=np.asarray(a['probabilities'],float);names=a['feature_names']; ts=a['timestamp_us']/1e6
 if x.shape!=(len(ts),69) or p.shape!=(len(ts),7) or len(ts)<16:raise ValueError('Analysis shape mismatch')
 def cols(ns):return [names.index(n) for n in ns]
 result={}
 for side,pi in [('R',[0,1,3]),('L',[2,4])]:
  ai=[i for i,n in enumerate(names) if n.startswith(side+'_') and '_angle' in n and 'elbow' not in n]
  wi=cols([side+'_hand_wrist_x',side+'_hand_wrist_y'])
  native,bridge=shape_evidence(a,side);evidence=native|bridge
  pm=np.isfinite(p[:,pi])&evidence[:,None]
  conflict=(np.sum(p[:,pi]>=.5,axis=1)>1)
  pm[conflict]=False
  # Predicted attributes and directly measured projected angles remain separate.
  result[side+'_shape']=(np.nan_to_num(p[:,pi]),pm)
  result[side+'_angles']=(x[:,ai],m[:,ai])
 ri=cols(['hands_wrist_delta_x','hands_wrist_delta_y','hands_wrist_distance'])
 result['hand_relation']=(x[:,ri],m[:,ri])
 ri=cols([s+'_'+point+'_'+d for s in ['L','R'] for point in ['pose_wrist','elbow'] for d in ['x','y']])
 result['arm_path']=(x[:,ri],m[:,ri])
 ri=cols([s+'_palm_axis_'+d for s in ['L','R'] for d in ['x','y']])
 result['hand_axis']=(x[:,ri],m[:,ri])
 return result

def matrix(q,r,main,active=None):
 costs=[]; coverage=[]
 for name in FAMILIES:
  qx,qm=q[name];rx,rm=r[name]
  # Equal per-coordinate weights, shared available coordinates only.
  valid=qm[:,None,:]&rm[None,:,:]
  count=valid.sum(2)
  distance=np.sqrt(((qx[:,None,:]-rx[None,:,:])**2*valid).sum(2)/np.maximum(count,1))
  distance[count==0]=1.0
  # In preparation/ending only body/arm context aligns; resting finger state is irrelevant.
  w=np.ones(len(main)) if name=='arm_path' else main.astype(float)
  if active is not None and name in active:w=w*active[name]
  costs.append(np.minimum(distance,3)*w[None,:]);coverage.append(w)
 return np.sum(costs,axis=0)/np.maximum(np.sum(coverage,axis=0)[None,:],1)

def path_dtw(cost):
 n,m=cost.shape
 dp=np.full((n+1,m+1),np.inf); dp[0,0]=0.; moves=np.zeros((n,m),np.uint8)
 # Start/end are preserved. Additional wait/unequal frame count is allowed.
 for i in range(n):
  for j in range(m):
   v=(dp[i,j],dp[i,j+1]+.03,dp[i+1,j]+.03); k=0 if v[0]<=v[1] and v[0]<=v[2] else 1 if v[1]<=v[2] else 2
   dp[i+1,j+1]=cost[i,j]+v[k];moves[i,j]=k
 path=[];i,j=n-1,m-1
 while i>=0 and j>=0:
  path.append((i,j));k=moves[i,j]
  if k==0:i-=1;j-=1
  elif k==1:i-=1
  else:j-=1
 if i!=-1 or j!=-1:raise AssertionError('Incomplete full path')
 return np.asarray(path[::-1],np.int32)

def core_mask(row,ts):
 main=np.zeros(len(ts),bool)
 for lo,hi in row['intervals']:main|=(ts>=lo)&(ts<=hi)
 if not main.any():raise ValueError('Reference main interval absent')
 return main

def compare_one(a,r,row):
 qch=channels(a);rch=channels(r); rt=r['timestamp_us']/1e6;qt=a['timestamp_us']/1e6
 main=core_mask(row,rt);active={name:np.ones(len(rt)) for name in FAMILIES}
 for stage in row['stages']:
  lo,hi=stage['frames'];rest=stage.get('confirmed_rest_sides',[])
  for side in rest:
   for suffix in ['shape','angles']:active[side+'_'+suffix][lo:hi+1]=0
  if rest:active['hand_relation'][lo:hi+1]=0
  if len(rest)==2:active['hand_axis'][lo:hi+1]=0
 costs=matrix(qch,rch,main,active);path=path_dtw(costs);qi,ri=path.T
 pairs=[]
 for stage in row['stages']:
  sel=(ri>=stage['frames'][0])&(ri<=stage['frames'][1])
  if not sel.any():raise AssertionError('Reference stage silently dropped')
  qs,rs=qi[sel],ri[sel]; families={}; resting=stage.get('confirmed_rest_sides',[])
  for name in FAMILIES:
   qx,qm=qch[name];rx,rm=rch[name]
   required=not ((name in ['R_shape','R_angles'] and 'R' in resting) or (name in ['L_shape','L_angles'] and 'L' in resting) or (name=='hand_relation' and resting) or (name=='hand_axis' and len(resting)==2))
   if name=='hand_axis' and resting:
    rm=rm.copy()
    for side in resting:rm[:,slice(0,2) if side=='L' else slice(2,4)]=False
   # Each expected reference frame-coordinate counted once; query cannot shrink denominator.
   all_ref=np.arange(stage['frames'][0],stage['frames'][1]+1)
   expected=rm[all_ref].sum(); good_count=0; errors=[]; missing=[];peak=0.;peak_at=None
   dim_expected=rm[all_ref].sum(0);dim_observed=np.zeros(rm.shape[1],int)
   for j in all_ref:
    qq=qs[rs==j]; need=rm[j]
    if not need.any():continue
    usable=qm[qq]&need[None,:]
    observed=usable.any(0);good_count+=int(observed.sum());dim_observed+=observed
    if not observed.any():missing.append(float(rt[j]));continue
    d=np.abs(qx[qq]-rx[j]);d[~usable]=np.nan
    if np.isfinite(d).any() and float(np.nanmax(d))>=peak:
     peak=float(np.nanmax(d));ii,dd=np.unravel_index(np.nanargmax(d),d.shape)
     peak_at=dict(query_frame=int(qq[ii]),reference_frame=int(j),query_seconds=float(qt[qq[ii]]),reference_seconds=float(rt[j]),coordinate=int(dd),representative_tie='last reference, first linked query/coordinate')
    # Worst linked difference prevents a long query interval being erased by its best frame.
    per=np.nanmax(d[:,observed],axis=0)
    errors.extend(per.tolist())
   query_expected=query_observed=0;query_missing=[];qdim_expected=np.zeros(rm.shape[1],int);qdim_observed=np.zeros(rm.shape[1],int)
   for i in np.unique(qs):
    need=rm[rs[qs==i]].any(0);query_expected+=int(need.sum());query_observed+=int((qm[i]&need).sum());qdim_expected+=need;qdim_observed+=qm[i]&need
    if need.any() and not (qm[i]&need).any():query_missing.append(float(qt[i]))
   ref_coverage=float(good_count/expected) if expected else None
   query_coverage=float(query_observed/query_expected) if query_expected else None
   coverage=min(ref_coverage,query_coverage) if ref_coverage is not None and query_coverage is not None else None
   dimension_coverage=np.minimum(dim_observed/np.maximum(dim_expected,1),qdim_observed/np.maximum(qdim_expected,1))
   if name.endswith('_shape'):coverage=float(dimension_coverage.min())
   families[name]=dict(distance=float(np.quantile(errors,.9)) if errors else None,peak_distance=peak if errors else None,peak_at=peak_at,coverage=coverage,reference_coverage=ref_coverage,query_coverage=query_coverage,expected=int(expected),observed=good_count,query_expected=query_expected,query_observed=query_observed,dimension_reference_expected=dim_expected.tolist(),dimension_reference_observed=dim_observed.tolist(),dimension_query_expected=qdim_expected.tolist(),dimension_query_observed=qdim_observed.tolist(),dimension_coverage=dimension_coverage.tolist(),missing_reference_seconds=missing,missing_query_seconds=query_missing,comparison_required=required,reference_rest_sides=resting)
   if name.endswith('_shape'):
    qnative,qbridge=shape_evidence(a,name[0]);rnative,rbridge=shape_evidence(r,name[0]); uq=np.unique(qs)
    families[name]['support_provenance']=dict(query_native_geometry_frames=int(qnative[uq].sum()),query_context_bridge_frames=int(qbridge[uq].sum()),query_frames=len(uq),reference_native_geometry_frames=int(rnative[all_ref].sum()),reference_context_bridge_frames=int(rbridge[all_ref].sum()),reference_frames=len(all_ref),meaning='native geometry support proxy versus short RGB/time context; neither is human verified truth',coordinates_interpolated=False)
  uniqueq=np.unique(qs); uniquer=np.unique(rs)
  compression=float(len(uniqueq)/max(len(uniquer),1))
  pairs.append(dict(id=stage['id'],title=stage['title'],source_description=stage.get('description',''),
    source_url=stage.get('source_url'),query_range_seconds=[float(qt[qs.min()]),float(qt[qs.max()])],
    reference_range_seconds=[float(rt[rs.min()]),float(rt[rs.max()])],families=families,compression=compression,
    canonical_stage_id=stage.get('canonical_stage_id',stage['id']),boundary_provenance=stage.get('boundary_provenance','unknown'),
    stage_alignment='estimated monotonic correspondence; not independently labeled same-phase truth'))
 return dict(id=row['clip_id'],label=f"공개 수행자 {row['person']:02}",person=row['person'],
   distance=float(np.mean(costs[qi,ri][main[ri]])),stages=pairs,path=path.tolist(),query_frame_times=qt.tolist(),reference_frame_times=rt.tolist())

@lru_cache(maxsize=1)
def bank():
 d=read(U/'REFERENCE_BANK.json');return d,{r['clip_id']:load_arrays(r) for r in d['rows']}

def catalog():
 d,_=bank();out=[]
 for word in sorted({r['word_id'] for r in d['rows']}):
  rows=[r for r in d['rows'] if r['word_id']==word]
  out.append(dict(id=word,label=rows[0]['label'],references=[dict(id=r['clip_id'],label=f"공개 수행자 {r['person']:02}",media_path=r['source'],description='대표 영상은 하나씩 볼 수 있지만 분석은 모든 공개 참고와 비교합니다.') for r in rows]))
 return dict(words=out,limits=dict(max_duration_seconds=20),model=d['observer'],scope='새 촬영과 여러 공개 참고의 관측 특징 비교. 교육적 정오 전체를 보증하지 않습니다.')

def aggregate(comparisons,ranges):
 selected=sorted(comparisons,key=lambda x:x['distance'])[:2]
 result=[]
 # The SAME two whole-reference sequences support all stages/families, no patchwork minima.
 for si,stage in enumerate(selected[0]['stages']):
  if any(len(c['stages'])!=len(selected[0]['stages']) or c['stages'][si].get('canonical_stage_id',c['stages'][si]['id'])!=stage.get('canonical_stage_id',stage['id']) for c in selected):raise ValueError('Incompatible stage identities/order')
  features=[]
  for family in FAMILIES:
   vals=[c['stages'][si]['families'][family] for c in selected]
   expected=any(v['expected'] for v in vals)
   usable=[v['coverage'] is not None and v['coverage']>=.65 and v['distance'] is not None for v in vals]
   cutoff=ranges[family]; peakcut=cutoff.get('peak') if isinstance(cutoff,dict) else None
   cutoff=cutoff.get('p90') if isinstance(cutoff,dict) else cutoff
   fits=[v['distance'] is not None and cutoff is not None and v['distance']<=cutoff and (peakcut is None or v.get('peak_distance') is not None and v['peak_distance']<=peakcut) for v in vals]
   if all(not v.get('comparison_required',True) for v in vals):status='NOT_REQUIRED_REFERENCE_REST'
   elif any(not v.get('comparison_required',True) for v in vals):status='INSUFFICIENT_EVIDENCE'
   elif not expected or not all(usable) or cutoff is None:status='INSUFFICIENT_EVIDENCE'
   elif all(fits):status='SIMILAR_REFERENCE_RANGE'
   elif not any(fits):status='DIFFERENCE_CANDIDATE'
   else:status='INSUFFICIENT_EVIDENCE'
   features.append(dict(name=LABELS[family],status=status,distances=[v['distance'] for v in vals],coverage=[v['coverage'] for v in vals],peak_distances=[v.get('peak_distance') for v in vals],peak_at=[v.get('peak_at') for v in vals],support_provenance=[v.get('support_provenance') for v in vals],normal_range_limit=cutoff,normal_peak_limit=peakcut,reference_ids=[c['id'] for c in selected]))
  states=[v['status'] for v in features]
  status='DIFFERENCE_CANDIDATE' if 'DIFFERENCE_CANDIDATE' in states else 'INSUFFICIENT_EVIDENCE' if not states or 'INSUFFICIENT_EVIDENCE' in states else 'SIMILAR_REFERENCE_RANGE'
  texts={'DIFFERENCE_CANDIDATE':'여러 정상 참고에서 배운 범위를 벗어난 특징이 있습니다. 표시한 시간대의 손모양·움직임을 참고와 나란히 확인해 보세요.',
   'INSUFFICIENT_EVIDENCE':'가림·관측 부족 또는 참고 간 의견 차이로 이 구간을 확정하기 어렵습니다. 이 결과를 동작 오류로 판단하지 않습니다.',
   'SIMILAR_REFERENCE_RANGE':'아래에 표시한 비교 항목이 공개 참고에서 관측한 차이 범위 안에 있습니다. 그 외 특징까지 맞았다는 뜻은 아닙니다.'}
  result.append(dict(id=str(si+1),title=stage['title'],status=status,message=texts[status],features=features,
    query_range_seconds=stage['query_range_seconds'],reference_range_seconds=stage['reference_range_seconds'],
    source_description=stage['source_description'],source_url=stage.get('source_url'),reference_id=selected[0]['id'],canonical_stage_id=stage.get('canonical_stage_id'),boundary_provenance=[c['stages'][si].get('boundary_provenance') for c in selected]))
 return result,[c['id'] for c in selected]

def evaluate_arrays(a,word):
 d,arrays=bank();rows=[r for r in d['rows'] if r['word_id']==word]
 if len(rows)<2:raise ValueError('Insufficient independent references')
 comparisons=[compare_one(a,arrays[r['clip_id']],r) for r in rows]
 ranges=read(U/'NORMAL_RANGES.json')['ranges'];stages,selected=aggregate(comparisons,ranges)
 for c in comparisons:
  for stage in c['stages']:
   fs=[]
   for family,v in stage['families'].items():
    cutoff=ranges[family];limit=cutoff.get('p90') if isinstance(cutoff,dict) else cutoff;peakcut=cutoff.get('peak') if isinstance(cutoff,dict) else None
    if not v.get('comparison_required',True):status='NOT_REQUIRED_REFERENCE_REST'
    elif v['coverage'] is None or v['coverage']<.65 or v['distance'] is None or limit is None:status='INSUFFICIENT_EVIDENCE'
    elif v['distance']<=limit and (peakcut is None or v['peak_distance']<=peakcut):status='SIMILAR_REFERENCE_RANGE'
    else:status='DIFFERENCE_CANDIDATE'
    fs.append(dict(name=LABELS[family],status=status,coverage=[v['coverage']],distances=[v['distance']],peak_at=[v.get('peak_at')],support_provenance=[v.get('support_provenance')]))
   st=[v['status'] for v in fs];stage['features']=fs
   stage['status']='DIFFERENCE_CANDIDATE' if 'DIFFERENCE_CANDIDATE' in st else 'INSUFFICIENT_EVIDENCE' if 'INSUFFICIENT_EVIDENCE' in st else 'SIMILAR_REFERENCE_RANGE'
   stage['message']='이 참고 한 편과의 개별 비교입니다. 통합 안내는 같은 두 참고의 전체 흐름을 함께 확인합니다.'
 return dict(word=word,label=rows[0]['label'],stages=stages,references=comparisons,selected_whole_references=selected,
   total_reference_people=len(rows),normal_range_source='public FIT leave-one-person-out; user input never changes ranges',
   limitations=['손가락 네 개의 펴짐/접힘 등 제한된 특징을 비교하며 엄지·정확한 접촉·모든 손모양을 판정하지 않습니다.',
     '단계 시간 대응은 DTW의 추정이며 반복 회차의 정답을 보장하지 않습니다.',
     '비슷함은 비교 가능한 관측 항목의 공개 참고 범위를 뜻합니다. 수어 전체의 교육적 정오나 의미 전달을 확정하지 않습니다.'])

def analyze_submission(path,jobdir,progress=None):
 from adapter import analyze_video
 jobdir=Path(jobdir);request=read(jobdir/'request.json');word=request['word'];start=time.perf_counter()
 import av
 with av.open(str(path)) as c:
  stream=c.streams.video[0];duration=float(stream.duration*stream.time_base) if stream.duration else (float(c.duration)/1e6 if c.duration else None)
  if duration is not None and (duration<.6 or duration>20):raise ValueError('0.6초~20초 길이의 한 사람 수어 영상을 넣어 주세요.')
 if progress:progress('관절과 영상 특징 분석',10,'새 촬영을 이 컴퓨터에서 분석합니다.')
 a=analyze_video(path,jobdir,progress)
 if a['status'].startswith('UNSUPPORTED'):
  return dict(word=word,label=word,stages=[dict(id='input',title='촬영 확인',status='INSUFFICIENT_EVIDENCE',message='상반신 기준 또는 영상 길이를 확인할 수 없습니다. 상반신이 충분히 보이게 다시 촬영해 주세요.')],references=[],latency=a['latency'])
 if bank()[0]['observer']=='expanded_nine_people':
  from train_expansion import infer_candidate
  a['probabilities']=infer_candidate(a)
 if progress:progress('여러 사람의 참고와 비교',85,'개별 동작 순서를 보존해 시간대를 맞춥니다.')
 result=evaluate_arrays(a,word)
 result['submission']=dict(duration_seconds=float(a['timestamp_us'][-1]-a['timestamp_us'][0])/1e6,frames=len(a['x']))
 result['latency']={**a['latency'],'total_analysis_seconds':time.perf_counter()-start}
 result['training_used_submission']=False
 save(jobdir/'FEEDBACK.json',result)
 if progress:progress('분석 완료',100,'사용자 촬영은 학습·튜닝·참고 자료에 추가하지 않았습니다.')
 return result
