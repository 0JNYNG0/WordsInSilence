"""Synthetic candidate web-wrapper tests; no server/model/video/DTW starts."""
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from copy import deepcopy
from unittest.mock import patch
import hashlib,importlib.util,json,sys
import numpy as np

U=Path(__file__).resolve().parents[1];OLD=U.parent/'word100_multipeople_live_feedback_20261101'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run():
 spec=importlib.util.spec_from_file_location('synthetic_research_wrapper',U/'run_server.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 checks=[]
 def check(name,value):
  if not value:raise AssertionError(name)
  checks.append(name)
 before={str(OLD/n):sha(OLD/n) for n in ['server.py','adapter.py','engine.py','live_engine.py','evidence_guard.py','app.js','index.html']}
 paths=['run_server.py','app.js','index.html','detail_enrichment.py','candidate_adapter.py']
 initial={str(U/n):sha(U/n) for n in paths}
 with TemporaryDirectory(prefix='research_wrapper_test_',dir=U/'review') as folder:
  root=Path(folder).resolve()
  if not root.is_relative_to(U.resolve()):raise ValueError('Unsafe synthetic cleanup root')
  names=['f'+str(i) for i in range(69)];a=dict(x=np.zeros((16,69)),mask=np.ones((16,69),bool),probabilities=np.zeros((16,7)),frame_id=np.arange(16),timestamp_us=np.arange(16)*33333)
  feature=dict(name='손목에서 손가락 뿌리로 향하는 방향',status='DIFFERENCE_CANDIDATE',peak_at=[dict(query_frame=3,reference_frame=3,query_seconds=.1,reference_seconds=.1)],distances=[.2],coverage=[.8])
  stage=dict(id='1',canonical_stage_id='WORD0001:stage01',status='DIFFERENCE_CANDIDATE',message='원래 경고',features=[feature],query_range_seconds=[0,.5])
  reference=dict(id='R1',distance=.1,path=[[i,i] for i in range(16)],query_frame_times=(a['timestamp_us']/1e6).tolist(),reference_frame_times=(a['timestamp_us']/1e6).tolist(),stages=[deepcopy(stage)])
  original=dict(word='WORD0001',stages=[deepcopy(stage)],references=[reference],selected_whole_references=['R1','R2'],latency=dict(total_adapter_seconds=1,total_analysis_seconds=2),training_used_submission=False)
  job=root/'job';(job/'adapter').mkdir(parents=True);np.savez(job/'adapter/analysis.npz',**a)
  m.save_new(job/'adapter/COMPLETE.json',dict(feature_names=names));m.save_new(root/'NORMAL_RANGES.json',dict(ranges={'hand_axis':{'p90':.1,'peak':.3}}))
  events=[];progress=[]
  base=SimpleNamespace(U=root,bank=lambda:(dict(observer='protected_baseline',rows=[dict(clip_id='R1',word_id='WORD0001'),dict(clip_id='R2',word_id='WORD0001'),dict(clip_id='OTHER',word_id='WORD0002')]),{'R1':{},'R2':{},'OTHER':{}}))
  def analyze(path,directory,callback):
   events.append('analysis');callback(dict(stage='complete'));callback('완료',100,'guard done');return deepcopy(original)
  live=SimpleNamespace(catalog=lambda:dict(words=[],model='protected_baseline'),analyze_submission=analyze)
  witness=dict(query=dict(index=3,frame_id=3,timestamp_us=99999),reference=dict(index=3,frame_id=3,timestamp_us=99999),coordinate=0,abs_residual=.5)
  episode=dict(start=witness['query'],end=witness['query'],query_indices=[3],sample_count=1,source_pts_span_us=0,singleton=True,dimensions=[0],max_abs_residual=.5,max_witnesses=[witness,deepcopy(witness)],context_before={},context_after={})
  def enrich(values,rows,bank_arrays,result,ranges):
   events.append('enrich')
   check('exact saved baseline values reach enrichment',all(np.array_equal(values[k],v) for k,v in a.items()) and values['feature_names']==names)
   check('word-filtered rows and same FIT cutoffs',[r['clip_id'] for r in rows]==['R1','R2'] and ranges=={'hand_axis':{'p90':.1,'peak':.3}})
   out=deepcopy(result);out['detail_enrichment']=dict(status='ADDED',status_changes=0)
   family=dict(interpretation=dict(kind='PROJECTED_2D_HAND_AXIS_DIFFERENCE',text='2D 투영 축이며 실제 회전 확인값이 아닙니다.'),trace=[dict(q=3,raw=.5)],exceedances={'peak_number_exceedance':dict(threshold=.3,above_query_samples=1,episodes=[episode])})
   out['references'][0]['stages'][0]['detail_enrichment']=dict(families={'hand_axis':family})
   out['stages'][0]['features'][0]['detail_enrichment']=dict(interpretation=family['interpretation'],references=[dict(reference_id='R1',json_pointer='/references/0/stages/0/detail_enrichment/families/hand_axis')])
   return out
  wrapper=m.ResearchEngine(base,live,SimpleNamespace(enrich=enrich))
  result=wrapper.analyze_submission('synthetic-not-opened',job,lambda *a,**k:progress.append((a,k)))
  check('one base analysis then one enrichment',events==['analysis','enrich'])
  m.preserve_original(original,result);check('all original values and selections preserved',True)
  full=read(job/'DETAIL_FEEDBACK.json');compact=read(job/'RESEARCH_FEEDBACK.json')
  check('full result saved without trace loss',len(full['references'][0]['stages'][0]['detail_enrichment']['families']['hand_axis']['trace'])==1)
  f=compact['references'][0]['stages'][0]['detail_enrichment']['families']['hand_axis'];ep=f['exceedances']['peak_number_exceedance']['episodes'][0]
  check('compact response removes only additive trace','trace' not in f and f['full_trace_saved_locally'])
  check('same pointer resolves to compact evidence',f['full_detail_json_pointer']==result['stages'][0]['features'][0]['detail_enrichment']['references'][0]['json_pointer'])
  check('singleton retained as one sample with zero span',ep['sample_count']==1 and ep['source_pts_span_us']==0 and ep['singleton'])
  check('representative is explicit and full tie witnesses preserved',ep['representative_max_witness']==witness and ep['max_witness_count']==2 and ep['representative_only_all_witnesses_saved_locally'] and len(full['references'][0]['stages'][0]['detail_enrichment']['families']['hand_axis']['exceedances']['peak_number_exceedance']['episodes'][0]['max_witnesses'])==2)
  check('returned result equals saved compact JSON',result==compact)
  check('separate full and compact timing recorded',(job/'DETAIL_TIMING.json').exists() and (job/'RESPONSE_SAVE_TIMING.json').exists() and read(job/'RESPONSE_SAVE_TIMING.json')['compact_response_bytes']==(job/'RESEARCH_FEEDBACK.json').stat().st_size)
  check('original latency untouched',result['latency']==original['latency'])
  check('final100 only after details',sum(len(args)>1 and args[1]==100 for args,kw in progress)==1 and progress[-1][0][1]==100)
  check('progress98 Korean',any(args[:2]==('차이의 근거 정리',98) for args,kw in progress))
  check('catalog identifies research route only',wrapper.catalog()['research_demo']['candidate_adapter'] and wrapper.catalog()['model']=='protected_baseline')
  for label,mutation in [('number',lambda r:r['references'][0].update(distance=.2)),('status',lambda r:r['stages'][0].update(status='SIMILAR_REFERENCE_RANGE')),('list',lambda r:r['references'].append({}))]:
   altered=deepcopy(result);mutation(altered)
   try:m.preserve_original(original,altered)
   except AssertionError:check('preservation rejects altered '+label,True)
   else:check('preservation rejects altered '+label,False)
  unsupported=dict(references=[],stages=[dict(status='INSUFFICIENT_EVIDENCE')],latency={})
  def no_reference(a,rows,arrays,result,ranges):
   check('unsupported requires no input reconstruction',a=={} and rows==[] and arrays=={} and ranges=={})
   return dict(result,detail_enrichment=dict(status='NO_REFERENCE_COMPARISON'))
  unsupported_job=root/'unsupported';unsupported_job.mkdir()
  other=m.ResearchEngine(SimpleNamespace(),SimpleNamespace(analyze_submission=lambda *a:deepcopy(unsupported)),SimpleNamespace(enrich=no_reference))
  check('unsupported result stays unsupported',other.analyze_submission('not-opened',unsupported_job)['stages']==unsupported['stages'])
  imports=[];mods={name:SimpleNamespace(__file__=str(path)) for name,path in [('adapter',U/'candidate_adapter.py'),('engine',OLD/'engine.py'),('evidence_guard',OLD/'evidence_guard.py'),('_candidate_preserved_live_engine',OLD/'live_engine.py'),('_candidate_detail_enrichment',U/'detail_enrichment.py'),('_candidate_preserved_server',OLD/'server.py')]}
  def fake_load(name,path):imports.append((name,Path(path)));return mods[name]
  prior=sys.dont_write_bytecode
  try:
   with patch.object(m,'load_file',side_effect=fake_load):s,w,c=m.components()
  finally:sys.dont_write_bytecode=prior
  check('candidate alias installed before engine imports',imports[0]==('adapter',U/'candidate_adapter.py'))
  check('server assets/session ROOT redirects only in memory',s.ROOT==U and w.base is mods['engine'] and c is mods['adapter'])
  with patch.object(sys,'argv',['run_server.py','--port','8802']),patch.object(m,'components',side_effect=AssertionError('must refuse before load')):
   try:m.main()
   except ValueError:check('existing8802 port refused before models',True)
   else:check('existing8802 port refused before models',False)
 oldjs=(OLD/'app.js').read_text(encoding='utf8');newjs=(U/'app.js').read_text(encoding='utf8')
 for begin,end in [('// SYNC_MATH_BEGIN','// SYNC_MATH_END'),('async function showPeak','function featureText')]:
  check('old source playback/peak behavior retained '+begin,oldjs[oldjs.index(begin):oldjs.index(end)]==newjs[newjs.index(begin):newjs.index(end)])
 check('UI retains statuses and peak buttons','짧은 변화의 원시점 보기' in newjs and "appendDetailView(box,f,stage,refId);" in newjs)
 for p,h in {**before,**initial}.items():check('source unchanged during tests '+p,sha(p)==h)
 out=dict(status='PASS_SYNTHETIC_RESEARCH_WRAPPER',checks=len(checks),results=checks,source_pins={**before,**initial,str(Path(__file__)):sha(__file__)},
  actual_server_starts=0,actual_model_inferences=0,actual_DTWs=0,actual_video_decodes=0,camera_access=0,
  limits=['Mock engine/enrichment proves routing and preservation; real upload/browser is root-owned and not performed by this test.','Original latency fields remain unchanged; additive timing and server REQUEST_TIMING account for enrichment and serialization.'])
 output=U/'review/RESEARCH_SERVER_SELFTEST.json'
 with output.open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
 print(json.dumps(dict(status=out['status'],checks=len(checks),sha256=sha(output))))

if __name__=='__main__':run()
