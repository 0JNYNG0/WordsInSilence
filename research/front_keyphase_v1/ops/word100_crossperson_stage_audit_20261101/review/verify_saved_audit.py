"""Independently join frozen saved results; no matcher/model/DP imports or calls."""
from pathlib import Path
from collections import Counter,defaultdict
from fractions import Fraction
import json,hashlib,math
U=Path(__file__).resolve().parents[1];O=U.parent
P=O/'word100_core_context_split_20261101';C=O/'word100_context_sequence_match_20261031';R=O/'word100_relational_sequence_compare_20261031'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=0;failures=[]
def ck(v,message):
 global checks
 checks+=1
 if not v:failures.append(message)
def eq(actual,expected,label):
 if isinstance(expected,dict):
  ck(isinstance(actual,dict) and set(actual)==set(expected),label+' keys')
  if isinstance(actual,dict):
   for k,v in expected.items():
    if k in actual:eq(actual[k],v,label+'/'+str(k))
 elif isinstance(expected,list):
  ck(isinstance(actual,list) and len(actual)==len(expected),label+' length')
  if isinstance(actual,list):
   for i,(a,e) in enumerate(zip(actual,expected)):eq(a,e,label+'/'+str(i))
 elif isinstance(expected,(float,Fraction)):
  ck(isinstance(actual,(int,float)) and math.isfinite(actual) and math.isclose(actual,float(expected),rel_tol=1e-10,abs_tol=1e-10),label+' number')
 else:ck(actual==expected,label+' value')

frozen=read(U/'FROZEN.json');audit=read(U/'ASSESSMENT.json');report=read(U/'REPORT_DATA.json')
ck(audit['frozen_sha256']==sha(U/'FROZEN.json'),'assessment frozen digest')
pin_before={p:sha(p) for p in frozen['pins']}
for p,h in frozen['pins'].items():ck(pin_before[p]==h,'frozen pin '+p)
rows={r['alias']:r for r in read(U/'AUTHORIZATION.json')['rows']}
labels=read(U/'PHYSICAL_CORRESPONDENCE.json');clips={c['alias']:c for c in labels['clips']}
ph={c['alias']:c for c in read(O/'word100_core_stage_evidence_20261031/PHASE_DATA.json')['clips']}
METHODS=('global_dtw','mean_posture','combined_context','core_first')
STATES=('same_kind_core','boundary','different_kind_core','unreviewed_or_uncertain','reference_uncertain','no_candidate')

# Build a finite lookup table independently of root point/target_at helpers.
tables={}
for alias,c in clips.items():
 table=[None]*c['native_frame_count']
 for reg in c['regions']:
  for frame in range(reg['outer'][0],reg['outer'][1]+1):
   ck(table[frame] is None,'disjoint label '+alias)
   table[frame]=reg
 tables[alias]=table
def target(alias,i):
 reg=tables[alias][i]
 return reg if reg is not None and reg['core'] is not None and reg['kind'] is not None else dict(kind=None,id=None,occurrence=None)
def classify(alias,t,i):
 if t['kind'] is None:return dict(state='reference_uncertain',regions=[],occurrence='unassessed')
 if i is None:return dict(state='no_candidate',regions=[],occurrence='unassessed')
 ck(type(i) is int and 0<=i<len(tables[alias]),'query index bounds')
 r=tables[alias][i]
 if r is None:return dict(state='unreviewed_or_uncertain',regions=[],occurrence='unassessed')
 if r['core'] is None:return dict(state='boundary',regions=[r['id']],occurrence='unassessed')
 if r['kind'] is None:return dict(state='unreviewed_or_uncertain',regions=[r['id']],occurrence='unassessed')
 same=r['kind']==t['kind'];occ='unassessed'
 if same and r['occurrence'] is not None and t['occurrence'] is not None:
  occ='same_documented_ordinal' if r['occurrence']==t['occurrence'] else 'different_documented_ordinal'
 return dict(state='same_kind_core' if same else 'different_kind_core',regions=[r['id']],occurrence=occ)
def pathtrace(q,r,path,core):
 linked=defaultdict(list)
 ck(len(path)==len(set(tuple(p) for p in path)),'no duplicate saved path links')
 for qi,ri in path:
  ck(type(qi) is int and type(ri) is int and 0<=qi<len(tables[q]) and 0<=ri<len(tables[r]),'original path row bounds')
  if core[0]<=ri<=core[1]:linked[ri].append(qi)
 counts=defaultdict(Fraction);missing=0
 for ri in range(core[0],core[1]+1):
  links=linked[ri]
  if not links:missing+=1;counts['unmapped_reference_frame']+=1
  else:
   for qi in links:counts[classify(q,target(r,ri),qi)['state']]+=Fraction(1,len(links))
 n=core[1]-core[0]+1
 ck(sum(counts.values())==n,'all reference frame weight retained')
 return dict(expected_reference_frames=n,unmapped_reference_frames=missing,reference_equal_weight_counts=dict(counts),fractions={s:v/n for s,v in counts.items()},saved_link_count=sum(len(x) for x in linked.values()))

expectedpairs={(q,r) for q in rows for r in rows if q!=r and rows[r]['role']=='fit' and rows[q]['word_id']==rows[r]['word_id']}
expected={(q,r,s['segment_id']) for q,r in expectedpairs for s in ph[r]['segments'] if s['role']=='core_candidate'}
ck(len(expectedpairs)==36 and len(expected)==117,'fixed36 pairs117 opportunities')
events={(e['query'],e['reference'],e['segment_id']):e for e in audit['events']}
ck(len(audit['events'])==len(events)==117 and set(events)==expected,'no missing or duplicate event, including None')
reportcases={(c['query'],c['reference']):c for c in report['cases']}
ck(len(report['cases'])==len(reportcases)==36 and set(reportcases)==expectedpairs,'report cases complete')
seen=set();methodstats={k:Counter() for k in METHODS};source_files=[];references=Counter();center_differences=[]

for entry in read(P/'RESULT_MANIFEST.json')['results']:
 if entry['kind']!='person_pair':continue
 paths=[Path(entry['file']),C/'results'/(entry['name']+'.json'),R/'pairs'/(entry['name']+'.json')]
 source_files.extend(paths);new,old,base=[read(p) for p in paths]
 q,r=new['query'],new['reference'];seen.add((q,r));ck((q,r) in expectedpairs,'saved allowed pair')
 ck(old['query']==base['query']==q and old['reference']==base['reference']==r,'source clip identities')
 rc=reportcases[(q,r)]
 for field,value in [('id',entry['name']),('role',rows[q]['role']),('label',rows[q]['label']),('query_pts',ph[q]['seconds']),('reference_pts',ph[r]['seconds']),('query_regions',clips[q]['regions']),('reference_regions',clips[r]['regions'])]:eq(rc[field],value,'case '+entry['name']+'/'+field)
 original={s['segment_id']:s['core'] for s in ph[r]['segments'] if s['role']=='core_candidate'}
 na={a['segment_id']:a for a in new['anchors']};oa={a['segment_id']:a for a in old['anchors']};ba={a['segment_id']:a for a in base['stages']};ma={a['segment_id']:a for a in base['free_order']['anchors']}
 for x in [na,oa,ba,ma]:
  ck(set(x)==set(original),'original reference opportunities kept')
  for sid in original:eq(x[sid]['reference_core'],original[sid],'source core '+sid)
 ce={(e['query'],e['reference'],e['segment_id']):e for e in rc['events']}
 ck(len(rc['events'])==len(ce)==len(original),'case event count')
 for sid,core in original.items():
  k=(q,r,sid);e=events[k];ref=(core[0]+core[1])//2;t=target(r,ref)
  references[rows[q]['role']]+=core[1]-core[0]+1
  eq(ce[k],e,'report event same '+str(k))
  for field,value in [('case',entry['name']),('role',rows[q]['role']),('label',rows[q]['label']),('reference_core',core),('reference_center',ref),('target',t),('query_regions',clips[q]['regions'])]:eq(e[field],value,str(k)+'/'+field)
  ck(set(e['methods'])==set(METHODS),'four conditions complete')
  gp=[p for p in base['path'] if core[0]<=p[1]<=core[1]];gids=sorted({p[0] for p in gp});ob=oa[sid].get('best');nb=na[sid].get('selected')
  source={
   'global_dtw':dict(path=gp,groups=ba[sid]['groups'],saved_state=None,span=[gids[0],gids[-1]] if gids else None,saved_center=gids[(len(gids)-1)//2] if gids else None),
   'mean_posture':dict(path=None,groups=None,saved_state=ma[sid].get('state'),span=None,saved_center=ma[sid].get('query_center')),
   'combined_context':dict(path=[p for p in ob['path'] if core[0]<=p[1]<=core[1]] if ob else [],groups=ob['phases']['core'] if ob else None,saved_state=oa[sid]['state'],span=ob['query_core'] if ob else None,saved_center=sum(ob['query_core'])//2 if ob else None),
   'core_first':dict(path=nb['core_path'] if nb else [],groups=nb['groups'] if nb else None,saved_state=na[sid]['state'],span=nb['query_core'] if nb else None,saved_center=sum(nb['query_core'])//2 if nb else None)}
  for name,s in source.items():
   v=e['methods'][name];path=s['path'];linked=sorted({p[0] for p in path if p[1]==ref}) if path is not None else []
   center=linked[(len(linked)-1)//2] if linked else None
   if path is None:center=s['saved_center']
   for field in ['path','groups','saved_state','span']:eq(v[field],s[field],str(k)+'/'+name+'/'+field)
   eq(v['saved_window_center'],s['saved_center'],str(k)+'/'+name+'/original center')
   eq(v['center'],center,str(k)+'/'+name+'/new linked reference center')
   if center!=s['saved_center']:center_differences.append(dict(query=q,reference=r,segment=sid,method=name,saved_window_center=s['saved_center'],audit_reference_link_center=center))
   classification=classify(q,t,center)
   for field,value in classification.items():eq(v[field],value,str(k)+'/'+name+'/'+field)
   trace=pathtrace(q,r,path,core) if path is not None else None;eq(v['trace'],trace,str(k)+'/'+name+'/whole reference trace')
   groups=s['groups'] or {};insufficient=[g for g in groups if groups[g]['expected']>0 and not groups[g]['sufficient']];unobserved=[g for g in groups if groups[g]['expected']==0]
   eq(v['insufficient_groups'],insufficient,str(k)+'/'+name+'/missing evidence');eq(v['reference_unobserved_groups'],unobserved,str(k)+'/'+name+'/reference unobserved')
   ck(v['observation_unassessed']==(s['groups'] is None),'mean evidence unassessed')
   ck(v['sign_correct'] is None and v['sign_error'] is None,'no educational verdict')
   methodstats[name]['none_centers']+=center is None;methodstats[name]['missing_ref_link']+=path is not None and not linked
   methodstats[name]['same_kind_but_insufficient']+=classification['state']=='same_kind_core' and bool(insufficient)
   methodstats[name]['observations']+=1

ck(seen==expectedpairs,'all36 saved pairs read once')
eq(report['summary'],audit['summary'],'report summary same');eq(report['changes'],audit['changes'],'report changes same')
eq(report['kinds'],labels['kinds'],'report kinds unchanged');eq(report['annotation_scope'],labels['scope'],'report limitations unchanged')
recalc={}
for role in ['fit','dev','all']:
 es=[e for e in events.values() if role=='all' or ('fit' if e['role']=='fit' else 'dev')==role]
 ck(len(es)=={'fit':78,'dev':39,'all':117}[role],'fixed group denominator '+role);recalc[role]={}
 for name in METHODS:
  values=[e['methods'][name] for e in es];counts=Counter(v['state'] for v in values);traces=[v['trace'] for v in values if v['trace'] is not None]
  ck(sum(counts.values())==len(es),'states retain unknown '+role+'/'+name)
  n=sum(t['expected_reference_frames'] for t in traces);sums=Counter()
  for tr in traces:sums.update(tr['reference_equal_weight_counts'])
  if traces:ck(math.isclose(sum(sums.values()),n,rel_tol=1e-12),'reference equal weight mass')
  recalculated=dict(total=len(es),states={s:counts[s] for s in STATES},same_kind_percent=100*counts['same_kind_core']/len(es),occurrences=dict(Counter(v['occurrence'] for v in values)),same_kind_but_insufficient=sum(v['state']=='same_kind_core' and bool(v['insufficient_groups']) for v in values),path=dict(expected_reference_frames=n,reference_equal_weight_counts=dict(sums),fractions={s:v/n for s,v in sums.items()}) if traces else None)
  eq(audit['summary'][role][name],recalculated,'summary '+role+'/'+name);recalc[role][name]=recalculated
for method in METHODS[:-1]:
 for change,direction in [('gained',True),('lost',False)]:
  expected_changes={(e['query'],e['reference'],e['segment_id'],e['role']) for e in events.values() if (e['methods']['core_first']['state']=='same_kind_core')==direction and (e['methods'][method]['state']=='same_kind_core')!=direction}
  actual=audit['changes'][method][change];got={(x['query'],x['reference'],x['segment'],x['role']) for x in actual}
  ck(len(actual)==len(got) and got==expected_changes,'gained/lost exact '+method+'/'+change)
protected=read(O/'word100_semantic_episode_pilot_20261031/FROZEN.json')['protected']
ck(len(protected)==32,'protected32 count')
for p,h in protected.items():ck(sha(p)==h,'protected file unchanged '+p)
for row in rows.values():ck(sha(row['input'])==row['input_sha256'],'input values/masks hash unchanged '+row['alias'])
for p,h in frozen['pins'].items():ck(sha(p)==h==pin_before[p],'frozen artifact unchanged after audit '+p)
ck(audit['new_model_inference']==0 and audit['educational_correctness_tested'] is False,'no inference/educational score')
status='PASS_SAVED_AUDIT_VERIFICATION' if not failures else 'FAIL_SAVED_AUDIT_VERIFICATION'
out=dict(status=status,reviewer='same Codex A, independently checks root stored arithmetic; authored DEV draft, not independent human linguistic truth',checks=checks,failures=failures,
 scope='Read frozen saved outputs only, no new matcher/DP/alignment/model/training',pairs=len(seen),opportunities=len(events),fixed_denominators=dict(all=117,fit=78,dev=39),conditions=list(METHODS),
 original_reference_frame_denominators=dict(references),reference_frame_denominator_all=sum(references.values()),
 original_source_files_verified=len(source_files),frozen_pins_verified=len(frozen['pins']),protected_files_verified=len(protected),public_input_files_verified=len(rows),
 representative_note='New primary center is lower median of unique ORIGINAL saved query indices linked to floor((reference_core_start+reference_core_end)/2). It differs from the saved query window/core center; the latter is separately preserved. Mean posture retains the old center because no path exists.',
 changed_representative_count=len(center_differences),changed_representatives=center_differences,method_support=dict(methodstats),summary=recalc,
 limitations=['Physical draft correspondence, not real sign correctness or semantic accuracy. Historical development exposure remains.',
 'All opportunities including missing centers, uncertain references, boundaries and unknowns remain in denominators. Unmapped whole-path reference frames retain unit mass.',
 'Each reference frame has unit mass shared equally across its saved linked query frames. This is not path-link micro-accuracy.',
 'Mean-posture temporal-path statistic is not applicable, explicitly None; its center still has all117 opportunities.',
 'Same physical kind does not erase stored missing feature support. Repeat ordinal agreement is separate from type agreement.',
 'Old source cores, paths, masks/inputs and stored scores remain unmodified; fixed-input hashes and frozen source hashes checked.'],
 source_hashes={n:sha(U/n) for n in ['FROZEN.json','PHYSICAL_CORRESPONDENCE.json','evaluate.py','ASSESSMENT.json','REPORT_DATA.json']})
dest=U/'review/SAVED_AUDIT_VERIFICATION.json'
dest.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(status,checks,'checks',len(failures),'failures','frame denominator',sum(references.values()))
print(json.dumps({r:{m:{'same':v['states']['same_kind_core'],'total':v['total'],'referenceframes':None if v['path'] is None else v['path']['expected_reference_frames']} for m,v in ms.items()} for r,ms in recalc.items()},indent=2))
if failures:print('\n'.join(failures[:25]));raise SystemExit(1)
