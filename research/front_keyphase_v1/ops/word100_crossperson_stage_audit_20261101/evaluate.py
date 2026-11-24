"""New RGB-stage audit of saved outputs. Does not import/run any matcher."""
from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib,sys
U=Path(__file__).resolve().parent;O=U.parent
PREV=O/'word100_core_context_split_20261101'
CTX=O/'word100_context_sequence_match_20261031'
REL=O/'word100_relational_sequence_compare_20261031'
PHASE=O/'word100_core_stage_evidence_20261031'
METHODS=['global_dtw','mean_posture','combined_context','core_first']
STATES=['same_kind_core','boundary','different_kind_core','unreviewed_or_uncertain','reference_uncertain','no_candidate']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 with Path(p).open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def inside(i,r):return r[0]<=i<=r[1]
def at_boundary(clip,index):return [r for r in clip['regions'] if r.get('outer') and inside(index,r['outer']) and not (r.get('core') and inside(index,r['core']))]
def target_at(clip,index):
 assert type(index) is int and index>=0
 if 'native_frame_count' in clip:assert index<clip['native_frame_count']
 rs=[r for r in clip['regions'] if r.get('core') and inside(index,r['core'])]
 return rs[0] if not at_boundary(clip,index) and len(rs)==1 and rs[0].get('kind') else dict(kind=None,id=None,occurrence=None)
def point(query,target,index):
 if index is not None:
  assert type(index) is int and index>=0
  if 'native_frame_count' in query:assert index<query['native_frame_count']
 if target.get('kind') is None:return dict(state='reference_uncertain',regions=[],occurrence='unassessed')
 if index is None:return dict(state='no_candidate',regions=[],occurrence='unassessed')
 boundary=at_boundary(query,index)
 if boundary:return dict(state='boundary',regions=[r['id'] for r in boundary],occurrence='unassessed')
 definite=[r for r in query['regions'] if r.get('core') and inside(index,r['core'])]
 # Overlapping or uncertain physical annotations are not converted to wrong.
 if len(definite)>1:return dict(state='unreviewed_or_uncertain',regions=[r['id'] for r in definite],occurrence='unassessed')
 if definite:
  r=definite[0]
  if r.get('kind') is None:return dict(state='unreviewed_or_uncertain',regions=[r['id']],occurrence='unassessed')
  same=r['kind']==target['kind'];occ='unassessed'
  if same and r.get('occurrence') is not None and target.get('occurrence') is not None:
   occ='same_documented_ordinal' if r['occurrence']==target['occurrence'] else 'different_documented_ordinal'
  return dict(state='same_kind_core' if same else 'different_kind_core',regions=[r['id']],occurrence=occ)
 return dict(state='unreviewed_or_uncertain',regions=[],occurrence='unassessed')
def trace(query,reference,path,core):
 # Each expected reference frame has total weight 1, irrespective of stretching.
 by=defaultdict(list)
 assert len(path)==len({tuple(p) for p in path}),'Unexpected duplicate saved link; preserved for review'
 for i,j in path:
  assert type(i) is int and type(j) is int and i>=0 and j>=0
  if 'native_frame_count' in query:assert i<query['native_frame_count']
  if 'native_frame_count' in reference:assert j<reference['native_frame_count']
  if inside(j,core):by[j].append(i)
 out=Counter();missing=0
 for j in range(core[0],core[1]+1):
  ids=by.get(j,[])
  if not ids:missing+=1;out['unmapped_reference_frame']+=1;continue
  for i in ids:out[point(query,target_at(reference,j),i)['state']]+=1/len(ids)
 n=core[1]-core[0]+1
 return dict(expected_reference_frames=n,unmapped_reference_frames=missing,reference_equal_weight_counts=dict(out),fractions={k:v/n for k,v in out.items()},saved_link_count=sum(len(v) for v in by.values()))
def freeze():
 labels=read(U/'PHYSICAL_CORRESPONDENCE.json');rows=read(U/'AUTHORIZATION.json')['rows'];assert len(labels['clips'])==16 and {c['alias'] for c in labels['clips']}=={r['alias'] for r in rows}
 for c in labels['clips']:
  covered=set()
  for r in c['regions']:
   assert r['kind'] is None or r['kind'] in labels['kinds']
   assert r['region_type'] in ['observed_physical_core','boundary','unknown']
   a,b=r['outer'];assert type(a) is int and type(b) is int and 0<=a<=b<c['native_frame_count']
   ids=set(range(a,b+1));assert not ids&covered;covered|=ids
   if r['core']:assert r['core']==r['outer'] and r['kind'] is not None
   else:assert r['region_type']=='boundary' and r['kind'] is None
 assert all(sha(PREV/p)==h for p,h in read(PREV/'COMPLETION.json')['pins'].items())
 files=[U/n for n in ['AUTHORIZATION.json','PLAN_KO.txt','PHYSICAL_CORRESPONDENCE.json','merge_labels.py','evaluate.py','review/PLAN_REVIEW.json','review/PLAN_REVIEW_DTW_ADDENDUM.json','review/CENTER_PROTOCOL_REVIEW.json','review/EVALUATOR_REVIEW.json','review/LABEL_RECONCILIATION.json','review/FIT_IMAGES_VIEWED.json','review/DEV_IMAGES_VIEWED.json']]
 files += [PREV/'COMPLETION.json',PREV/'RESULT_MANIFEST.json',CTX/'RESULT_MANIFEST.json',REL/'PAIR_PROVENANCE.json',PHASE/'PHASE_DATA.json']
 files += [U/'review'/n for n in ['FIT_PHYSICAL_STAGES.json','DEV_PHYSICAL_STAGES.json']]
 files += [PHASE/'media'/(r['alias']+'.webm') for r in read(U/'AUTHORIZATION.json')['rows']]
 for x in read(PREV/'RESULT_MANIFEST.json')['results']:
  if x['kind']=='person_pair':files += [Path(x['file']),CTX/'results'/(x['name']+'.json'),REL/'pairs'/(x['name']+'.json')]
 write(U/'FROZEN.json',dict(status='FROZEN_NEW_PHYSICAL_STAGE_AUDIT_BEFORE_PAIR_PREDICTIONS_READ',new_searches=0,methods=METHODS,denominators=dict(pairs=36,total=117,fit=78,dev=39),annotation='same Codex RGB physical draft, not independent human KSL truth; posthoc development audit',not_new_unseen_evaluation=True,pins={str(p):sha(p) for p in files}))
def evaluate():
 frozen=read(U/'FROZEN.json');assert all(sha(p)==h for p,h in frozen['pins'].items())
 labels=read(U/'PHYSICAL_CORRESPONDENCE.json');clips={c['alias']:c for c in labels['clips']};ph={c['alias']:c for c in read(PHASE/'PHASE_DATA.json')['clips']};rows={r['alias']:r for r in read(U/'AUTHORIZATION.json')['rows']};events=[];cases=[];seen=set()
 expected={(a,b,s['segment_id']) for a,ar in rows.items() for b,br in rows.items() if a!=b and br['role']=='fit' and ar['word_id']==br['word_id'] for s in ph[b]['segments'] if s['role']=='core_candidate'}
 assert len(expected)==117 and set(clips)==set(rows)
 for row in read(PREV/'RESULT_MANIFEST.json')['results']:
  if row['kind']!='person_pair':continue
  d=read(row['file']);old=read(CTX/'results'/(row['name']+'.json'));base=read(REL/'pairs'/(row['name']+'.json'));q,r=d['query'],d['reference'];assert q!=r and rows[r]['role']=='fit' and rows[q]['word_id']==rows[r]['word_id']
  assert old['query']==base['query']==q and old['reference']==base['reference']==r
  olda={a['segment_id']:a for a in old['anchors']};meana={a['segment_id']:a for a in base['free_order']['anchors']};caseevents=[]
  required={s['segment_id']:s['core'] for s in ph[r]['segments'] if s['role']=='core_candidate'}
  for anchors in [d['anchors'],old['anchors'],base['stages'],base['free_order']['anchors']]:
   assert len(anchors)==len(required) and {x['segment_id'] for x in anchors}==set(required)
   for x in anchors:assert x['reference_core']==required[x['segment_id']]
  for a in d['anchors']:
   sid=a['segment_id'];key=(q,r,sid);assert key in expected and key not in seen;seen.add(key);core=a['reference_core'];refcenter=sum(core)//2;target=target_at(clips[r],refcenter);gp=[p for p in base['path'] if inside(p[1],core)];gids=sorted({p[0] for p in gp});gb=next(s for s in base['stages'] if s['segment_id']==sid);ob=olda[sid].get('best');nb=a.get('selected');ma=meana[sid]
   methods={
    'global_dtw':dict(center=gids[(len(gids)-1)//2] if gids else None,span=[min(gids),max(gids)] if gids else None,path=gp,groups=gb['groups'],saved_state=None),
    'mean_posture':dict(center=ma.get('query_center'),span=None,path=None,groups=None,saved_state=ma.get('state')),
    'combined_context':dict(center=sum(ob['query_core'])//2 if ob else None,span=ob['query_core'] if ob else None,path=[p for p in ob['path'] if inside(p[1],core)] if ob else [],groups=ob['phases']['core'] if ob else None,saved_state=olda[sid]['state']),
    'core_first':dict(center=sum(nb['query_core'])//2 if nb else None,span=nb['query_core'] if nb else None,path=nb['core_path'] if nb else [],groups=nb['groups'] if nb else None,saved_state=a['state'])}
   entry=dict(case=row['name'],query=q,reference=r,label=rows[q]['label'],role=rows[q]['role'],segment_id=sid,reference_core=core,reference_center=refcenter,target=target,methods={},query_regions=clips[q]['regions'])
   for name,m in methods.items():
    m['saved_window_center']=m['center']
    if m['path'] is not None:
     linked=sorted({p[0] for p in m['path'] if p[1]==refcenter});m['center']=linked[(len(linked)-1)//2] if linked else None
    m.update(point(clips[q],target,m['center']));m['trace']=trace(clips[q],clips[r],m['path'],core) if m['path'] is not None else None
    # Location/phase agreement does not erase any stored feature observation lack.
    m['insufficient_groups']=[k for k,v in (m['groups'] or {}).items() if v['expected']>0 and not v['sufficient']];m['reference_unobserved_groups']=[k for k,v in (m['groups'] or {}).items() if v['expected']==0];m['observation_unassessed']=m['groups'] is None;m['sign_correct']=None;m['sign_error']=None
    entry['methods'][name]=m
   events.append(entry);caseevents.append(entry)
  cases.append(dict(id=row['name'],query=q,reference=r,role=rows[q]['role'],label=rows[q]['label'],query_pts=ph[q]['seconds'],reference_pts=ph[r]['seconds'],query_regions=clips[q]['regions'],reference_regions=clips[r]['regions'],events=caseevents))
 assert seen==expected and len(cases)==36 and len(events)==117 and sum(e['role']=='fit' for e in events)==78
 summary={}
 for role in ['fit','dev','all']:
  es=[e for e in events if role=='all' or ('fit' if e['role']=='fit' else 'dev')==role];ms={}
  for name in METHODS:
   vs=[e['methods'][name] for e in es];counts=Counter(v['state'] for v in vs)
   paths=[v['trace'] for v in vs if v['trace'] is not None];n=sum(v['expected_reference_frames'] for v in paths);totals=Counter()
   for v in paths:totals.update(v['reference_equal_weight_counts'])
   ms[name]=dict(total=len(es),states={k:counts[k] for k in STATES},same_kind_percent=100*counts['same_kind_core']/len(es),occurrences=dict(Counter(v['occurrence'] for v in vs)),same_kind_but_insufficient=sum(v['state']=='same_kind_core' and bool(v['insufficient_groups']) for v in vs),path=dict(expected_reference_frames=n,reference_equal_weight_counts=dict(totals),fractions={k:v/n for k,v in totals.items()} if n else None) if paths else None)
  summary[role]=ms
 changes={}
 for name in METHODS[:-1]:
  changes[name]=dict(gained=[dict(query=e['query'],reference=e['reference'],segment=e['segment_id'],role=e['role']) for e in events if e['methods']['core_first']['state']=='same_kind_core' and e['methods'][name]['state']!='same_kind_core'],lost=[dict(query=e['query'],reference=e['reference'],segment=e['segment_id'],role=e['role']) for e in events if e['methods'][name]['state']=='same_kind_core' and e['methods']['core_first']['state']!='same_kind_core'])
 write(U/'ASSESSMENT.json',dict(status='COMPLETE_PENDING_INDEPENDENT_VERIFICATION',frozen_sha256=sha(U/'FROZEN.json'),events=events,summary=summary,changes=changes,new_model_inference=0,educational_correctness_tested=False))
 write(U/'REPORT_DATA.json',dict(summary=summary,changes=changes,cases=cases,annotation_scope=labels['scope'],kinds=labels['kinds']))
 print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':{'freeze':freeze,'evaluate':evaluate}[sys.argv[1]]()
