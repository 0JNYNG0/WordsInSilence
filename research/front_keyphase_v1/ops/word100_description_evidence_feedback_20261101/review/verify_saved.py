"""Independent arithmetic over saved inputs/results; no root runner or matcher.
Group geometry is reconstructed here; trace reduction uses scalar loops.
"""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,math,ast
import numpy as np
U=Path(__file__).resolve().parents[1];O=U.parent;REL=O/'word100_relational_sequence_compare_20261031';OLD=O/'word100_ordered_motion_alignment_20261101';PH=O/'word100_core_stage_evidence_20261031'
checks=0;failures=[];stats=Counter()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ck(v,n):
 global checks
 checks+=1
 if not v:failures.append(n)
def eq(a,b,n):
 if isinstance(b,dict):
  ck(isinstance(a,dict) and a.keys()==b.keys(),n+' keys')
  if isinstance(a,dict):
   for k,v in b.items():
    if k in a:eq(a[k],v,n+'/'+str(k))
 elif isinstance(b,list):
  ck(isinstance(a,list) and len(a)==len(b),n+' length')
  if isinstance(a,list):
   for i,(x,y) in enumerate(zip(a,b)):eq(x,y,n+'/'+str(i))
 elif isinstance(b,float):ck(isinstance(a,(int,float)) and math.isfinite(a) and math.isclose(a,b,abs_tol=1e-10,rel_tol=1e-10),n+' arithmetic')
 else:ck(type(a)==type(b) and a==b,n+' value')
def runs(seq):
 ordered=sorted(set(seq));out=[]
 for i in ordered:
  if out and out[-1][-1]+1==i:out[-1][-1]=i
  else:out.append([i,i])
 return out
def features_independent(z):
 x=np.array(z['x'],dtype=float);m=np.array(z['mask'],dtype=bool)&np.isfinite(x);n=len(x);out={}
 for side,o in [('L',0),('R',28)]:
  for suffix,ids,div in [('shape',list(range(o+11,o+25)),1),('location',[o,o+1,o+8,o+9],1),('axis',[o+25,o+26],2),('direction',[63,64] if side=='L' else [67,68],2)]:
   vals=[[float(x[i,j]/div) if m[i,j] else 0.0 for j in ids] for i in range(n)]
   out[side+'_'+suffix]=(vals,[[bool(m[i,j]) for j in ids] for i in range(n)])
  vals=[];valid=[];rel=[];rv=[];other=28 if o==0 else 0
  for i in range(n):
   vx=float(x[i,o+2]-x[i,o+4]);vy=float(x[i,o+3]-x[i,o+5]);length=math.hypot(vx,vy)
   ok=bool(m[i,o+2:o+6].all() and length>1e-8)
   ids=[o+4,o+5,o+10]
   vals.append([float(x[i,j]) if m[i,j] else 0.0 for j in ids]+([vx/length/2,vy/length/2,length] if ok else [0.0]*3))
   valid.append([bool(m[i,j]) for j in ids]+[ok]*3)
   dx=float(x[i,other+2]-x[i,other+4]);dy=float(x[i,other+3]-x[i,other+5]);norm2=dx*dx+dy*dy
   good=bool(m[i,o:o+2].all() and m[i,other+2:other+6].all() and norm2>1e-8)
   ax=float(x[i,o]-x[i,other+4]);ay=float(x[i,o+1]-x[i,other+5])
   rel.append([(ax*dx+ay*dy)/norm2,(dx*ay-dy*ax)/norm2] if good else [0.0,0.0]);rv.append([good,good])
  out[side+'_arm']=(vals,valid);out[side+'_to_'+('R' if side=='L' else 'L')+'_forearm']=(rel,rv)
 ids=[56,57,58];out['hands_relation']=([[float(x[i,j]) if m[i,j] else 0.0 for j in ids] for i in range(n)],[[bool(m[i,j]) for j in ids] for i in range(n)])
 return out
GM={f'{h}_{a}':f'{h}_{b}' for h in 'LR' for a,b in [('finger_angles14','shape'),('palm_axis2d','axis'),('hand_position2d','location'),('arm2d','arm')]}
GM.update(two_hand_relation2d='hands_relation',L_to_R_forearm2d='L_to_R_forearm',R_to_L_forearm2d='R_to_L_forearm')

f=read(U/'FROZEN.json');before={p:sha(p) for p in f['pins']}
for p,h in f['pins'].items():ck(before[p]==h,'frozen pin '+p)
rows={r['alias']:r for r in read(U/'AUTHORIZATION.json')['rows']};arrays={};feat={}
for alias,row in rows.items():
 ck(sha(row['input'])==row['input_sha256'],'input16 sha '+alias)
 with np.load(row['input'],allow_pickle=False) as z:arrays[alias]={k:z[k] for k in ['x','mask','frame_id','timestamp_us']}
 ck(np.array_equal(arrays[alias]['frame_id'],np.arange(len(arrays[alias]['x']))),'original frame row identity '+alias)
 feat[alias]=features_independent(arrays[alias])
phase={c['alias']:c for c in read(PH/'PHASE_DATA.json')['clips']}
contract=read(REL/'SOURCE_FEATURE_CONTRACT.json');clauses={c['id']:dict(c,word_id=e['word_id']) for e in contract['expressions'] for c in e['clauses']}
dev={(d['query'],d['reference'],x['segment_id'],x['group']):x for d in read(REL/'DEVELOPMENT.json')['results'] if d['condition']=='B_same_path' for x in d['decisions']}
ranges={(x['reference'],x['segment_id'],x['group']):x for x in read(REL/'REFERENCE_RANGES.json')['ranges']['B_same_path']}
names=None
for node in ast.parse((REL/'relational.py').read_text(encoding='utf-8')).body:
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='NAMES' for t in node.targets):names=ast.literal_eval(node.value)
oldrows=[r for r in read(OLD/'RESULT_MANIFEST.json')['results'] if r['kind']=='person_pair'];old={r['name']:r for r in oldrows}
manifest=read(U/'RESULT_MANIFEST.json');report=read(U/'REPORT_DATA.json');summary=read(U/'SUMMARY.json')
results={}
for row in manifest['results']:
 ck(sha(row['file'])==row['sha256'],'new result manifest hash')
 d=read(row['file']);ck(d['id'] not in results,'unique result');results[d['id']]=d
ck(len(results)==len(manifest['results'])==36 and set(results)==set(old),'36 fixed pairs')
rc={d['id']:d for d in report['cases']};ck(len(rc)==len(report['cases'])==36,'36 report cases')
source_cards_by_reference=Counter();expected_summary={};unavailable_records=[]
for name,result in results.items():
 q,r=result['query'],result['reference'];oldrow=old[name];guard=read(oldrow['file']);basepath=REL/'pairs'/(name+'.json');base=read(basepath)
 ck(sha(oldrow['file'])==oldrow['sha256'],'prior policy hash '+name)
 ck(q==base['query']==guard['query'] and r==base['reference']==guard['reference'],'clip identities '+name)
 ck(rows[r]['role']=='fit' and q!=r and rows[q]['word_id']==rows[r]['word_id'],'fit direct reference cohort')
 eq(result['path'],base['path'],'unchanged global path '+name);eq(guard['path'],base['path'],'prior path lineage '+name)
 eq(result,rc[name],'report exact result '+name)
 ck(result==rc[name],'report parsed values exactly unchanged '+name)
 for key,value in [('role',rows[q]['role']),('label',rows[q]['label']),('query_person',rows[q]['person']),('reference_person',rows[r]['person']),('source_pair',str(basepath)),('source_pair_sha256',sha(basepath)),('learned_description_understanding',False),('educational_correctness',None)]:eq(result[key],value,name+'/'+key)
 for alias,key in [(q,'query_pts'),(r,'reference_pts')]:eq(result[key],[int(t)/1e6 for t in arrays[alias]['timestamp_us']],name+'/'+key)
 segs={s['segment_id']:s for s in phase[r]['segments']};original={s['segment_id']:s for s in base['stages']};got={s['id']:s for s in result['stages']};guards={s['segment_id']:s for s in guard['anchors']}
 ck(set(got)==set(original) and len(got)==len(result['stages']),'all old stages kept '+name)
 for sid,saved in original.items():
  gotstage=got[sid];seg=segs[sid];core=saved['reference_core'];pp=[p for p in base['path'] if core[0]<=p[1]<=core[1]];stats['stages']+=1
  for key,value in [('reference_core',core),('query_frames',saved['query_frames']),('path',pp),('reference_rgb_description',seg['description_ko']),('source_uncovered_anchors',seg.get('source_uncovered_anchors',[]))]:eq(gotstage[key],value,name+'/'+sid+'/'+key)
  expected_alignment={k:guards[sid][k] for k in ['status','flags','range_relation']};expected_alignment['not_proof_of_correct_stage']=True;eq(gotstage['alignment'],expected_alignment,'alignment uncertainty separate')
  ck(gotstage['source_scope']=='listed source anchors only; numeric metrics cover full physical core','source context vs measurement span')
  ck(set(gotstage['groups'])==set(saved['groups'])==set(feat[q]),'all13 groups retained')
  for group,metric in saved['groups'].items():
   g=gotstage['groups'][group];label=name+'/'+sid+'/'+group;stats['group_opportunities']+=1
   ck(g['saved_metric']==metric and g['saved_decision']==dev.get((q,r,sid,group)) and g['saved_range']==ranges.get((r,sid,group)),label+'/saved objects exactly unchanged')
   eq(g['saved_metric'],metric,label+'/original metric');eq(g['saved_decision'],dev.get((q,r,sid,group)),label+'/old DEV decision');eq(g['saved_range'],ranges.get((r,sid,group)),label+'/old reference range');eq(g['name'],names[group],label+'/group name')
   xv,xm=feat[q][group];yv,ym=feat[r][group];width=len(yv[0]);nodes=[]
   total=0.0;compared=0;expected=0;pathmissing=0;peak=None
   for ri in range(core[0],core[1]+1):
    qs=[qi for qi,rj in pp if rj==ri];ck(bool(qs),'every reference frame linked '+label);stats['reference_trace_rows']+=1;stats['query_links_across_groups']+=len(qs)
    components=[];node_missing=0
    for k in range(width):
     ds=[]
     for qi in qs:
      if ym[ri][k] and xm[qi][k]:ds.append(min(abs(xv[qi][k]-yv[ri][k]),1.0));stats['shared_link_cells']+=1
      elif ym[ri][k]:node_missing+=1
     components.append(max(ds) if ds else None)
    observed=sum(v is not None for v in components);exp=sum(ym[ri]);num=sum(v for v in components if v is not None)
    node=dict(reference_frame=ri,query_frames=qs,reference_valid=ym[ri],query_valid=[xm[qi] for qi in qs],reference_values=[v if ok else None for v,ok in zip(yv[ri],ym[ri])],query_values=[[v if ok else None for v,ok in zip(xv[qi],xm[qi])] for qi in qs],difference_by_component=components,observed_components=observed,expected_components=exp,possible_components=width,path_missing_cells=node_missing,difference=num/observed if observed else None)
    nodes.append(node);total+=num;compared+=observed;expected+=exp;pathmissing+=node_missing
    vals=[v for v in components if v is not None]
    if vals:peak=max(vals) if peak is None else max(peak,max(vals))
   eq(g['trace'],nodes,label+'/independent scalar trace')
   eq(compared,metric['compared'],label+'/unique observed reference cells');eq(expected,metric['expected'],label+'/required reference cells');eq(pathmissing,metric['path_missing_cells'],label+'/missing query link cells')
   eq(metric['mean_difference'],total/compared if compared else None,label+'/saved mean from trace');eq(metric['peak_difference'],peak,label+'/saved peak from trace')
   eq(g['reference_missing_ranges'],runs(x['reference_frame'] for x in nodes if x['expected_components']<width),label+'/reference missing ranges')
   eq(g['query_missing_reference_ranges'],runs(x['reference_frame'] for x in nodes if x['path_missing_cells']),label+'/query missing ranges')
   obs='unobserved' if metric['compared']==0 else 'partial_observation' if metric['reference_unobserved']!=0 or metric['path_missing_cells']!=0 or not metric['sufficient'] else 'complete_observation'
   dec=dev.get((q,r,sid,group));range_state=dec['state'] if dec else None
   fbstate='comparison_evidence_insufficient' if obs!='complete_observation' else {'larger_than_fit_reference_range':'observed_difference_to_review','within_small_reference_range':'within_small_reference_sample'}.get(range_state,'measurement_only_no_range')
   expectedfb=dict(status=fbstate,observation=obs,original_range_state=range_state,educational_correct=None,educational_error=None,correction_instruction=None,important_difference_proven=False,allowable_difference_proven=False)
   eq(g['feedback'],expectedfb,label+'/no semantic verdict')
  cards=[]
  for link in seg.get('source_links',[]):
   clause=clauses[link['clause_id']];fitanchors=[x for x in clause['anchors'] if x['alias']==r and x['role']=='fit']
   ck(len(fitanchors)==1 and clause['word_id']==rows[r]['word_id'],'direct FIT source evidence only')
   ck(clause['runtime_scope']!='inactive_different_form','alternative form inactive')
   ck(link['text']==clause['source_span'] and link['url']==clause['source_url'] and link['status']==clause['previous_visual_match_status'],'source clause identity')
   anchorids={x['frame_id'] for x in fitanchors[0]['anchors']};frames=link['anchor_frames'];ck(len(frames)==len(set(frames)) and bool(frames) and set(frames)<=anchorids,'only original direct source anchors')
   ck(all(core[0]<=j<=core[1] for j in frames),'source anchors in physical context core')
   groups=[]
   for k in clause['groups']:
    if k in GM and GM[k] not in groups:groups.append(GM[k])
   card=dict(clause_id=clause['id'],text=link['text'],url=link['url'],source_entry_id=clause['source_entry_id'],source=clause['source'],reference_segment_id=sid,numeric_context_core=core,full_span_semantic_verified=False,source_status=link['status'],reference_anchor_frames=frames,applicable_numeric_groups=groups,context_only_groups=[k for k in clause['groups'] if k not in GM],limitations=clause['unsupported_components'],restrictions=clause['mapping_stage_restrictions'],scope='reference anchor context only; no framewise semantic target',named_shape_verified=False,source_action_verified=False,training_target=False,
    query_anchor_links=[dict(reference_frame=j,query_frames=[i for i,k in pp if k==j]) for j in frames])
   cards.append(card);stats['cards']+=1;source_cards_by_reference[r]+=1
  eq(gotstage['source_cards'],cards,name+'/'+sid+'/source cards no propagation')
  if not cards:unavailable_records.append(dict(case=name,reference=r,segment=sid))
ck(stats['stages']==117 and stats['group_opportunities']==1521,'117 stages1521 group opportunities')
for role in ['fit','development']:
 rr=[d for d in results.values() if d['role']==role];ss=[s for d in rr for s in d['stages']];gg=[g for s in ss for g in s['groups'].values()];linked=[g for s in ss for k,g in s['groups'].items() if any(k in c['applicable_numeric_groups'] for c in s['source_cards'])]
 expected_summary[role]=dict(pairs=len(rr),stages=len(ss),source_context_stages=sum(bool(s['source_cards']) for s in ss),source_cards=sum(len(s['source_cards']) for s in ss),group_opportunities=len(gg),all_status=dict(Counter(g['feedback']['status'] for g in gg)),linked_group_opportunities=len(linked),linked_status=dict(Counter(g['feedback']['status'] for g in linked)),alignment_uncertain_stages=sum(bool(s['alignment']['flags']) for s in ss))
 eq(summary[role],expected_summary[role],'summary '+role)
ck(expected_summary['fit']['stages']==78 and expected_summary['development']['stages']==39,'fit78 DEV39 preserved')
for k,v in [('new_inference',False),('new_training',False),('source_clauses',len(clauses)),('inactive_clauses',[c['id'] for c in clauses.values() if c['runtime_scope']=='inactive_different_form']),('automatic_correct_or_wrong_verdicts',0),('accuracy_improvement_measured',False)]:eq(summary[k],v,'summary limit '+k)
ck(math.isfinite(summary['total_seconds']) and summary['total_seconds']>=0,'finite run duration')
eq(report['summary'],summary,'report summary');eq(report['group_names'],names,'report group names');eq(report['source_clause_catalog'],list(clauses.values()),'source catalog unchanged')
ck(len(f['protected'])==32,'32 protected files')
for p,h in f['protected'].items():ck(sha(p)==h,'protected sha '+p)
for p,h in f['pins'].items():ck(sha(p)==h==before[p],'old input/output pin unchanged '+p)
ck(not (U/'RUNNING.json').exists(),'execution completed')
out=dict(status='PASS_SAVED_VERIFICATION' if not failures else 'FAIL_SAVED_VERIFICATION',reviewer='same Codex A independent arithmetic; not independent human source/linguistic truth',checks=checks,failures=failures,scope='36 saved person pairs117 cores; original16 public inputs only',stats=dict(stats),summary=expected_summary,source_cards_by_reference=dict(source_cards_by_reference),source_unlinked_stage_count=len(unavailable_records),protected_files=32,input_files=16,frozen_pins=len(f['pins']),
 interpretation=['Original full paths, stage cores, metrics, DEV decisions and fit-reference ranges are unchanged. New trace localizes saved differences without fitting or selecting a new candidate.',
 'Each original reference component reduces all shared query-link differences with abs, clip1, then max; absent evidence remains None. Every linked query value/mask is preserved, even if its reference component is unobserved.',
 'Source clauses attach only to existing directly reviewed FIT reference anchors; numerical metrics span the whole original physical core. Source context is not a semantic label for all frames.',
 'Partial reference groups cannot be promoted to complete observation merely because legacy sufficient is true. Alignment uncertainty, measurement support and source semantic uncertainty stay separate.',
 'within_small_reference_sample does not prove allowable variation; observed_difference_to_review does not prove an important/sign error. No teaching correction or named handshape is inferred.',
 'Previously used four-expression development data; no fresh generalization or trained description understanding claim.'],
 activity=dict(new_inference=0,new_alignment=0,new_DP=0,new_training=0,root_run_reinvocations=0,old_writes=0),hashes={n:sha(U/n) for n in ['FROZEN.json','RESULT_MANIFEST.json','SUMMARY.json','REPORT_DATA.json','evidence.py','run.py']})
(U/'review/SAVED_VERIFICATION.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(out['status'],checks,'checks',len(failures),'failures');print(json.dumps(expected_summary,ensure_ascii=False,indent=2));print('STATS',dict(stats),'SOURCE_REFS',dict(source_cards_by_reference))
if failures:print('\n'.join(failures[:30]));raise SystemExit(1)
