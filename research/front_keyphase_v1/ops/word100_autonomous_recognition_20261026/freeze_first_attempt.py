from common import *
from collections import Counter
import re,shutil,ast
def main():
 assert read(U/'SAVED_VERIFICATION.json')['status']=='PASS' and read(U/'SAVED_COMPARISON.json')['development_pass'] and read(U/'TIMING.json')['status']=='PASS_TIMING_ONLY_TRAIN_PROBES'
 d=U/'fresh299';d.mkdir();old=read(B/'training_CONTEXT.json');oldtest=read(E/'EVAL_CONTEXT.json');codes=old['word_codes'];labels=old['labels'];source=O/'word_dataset_plan_20261013/FRONT_VIDEO_JOIN.jsonl'
 allrows=[json.loads(x) for x in source.read_text('utf-8').splitlines()];selected=[r for r in allrows if r['word_code'] in codes and 14<=int(re.search(r'_REAL(\d+)',r['clip_id'])[1])<=16]
 assert len(selected)==300;invalid=[r for r in selected if not r['metadata_video_join_eligible']];assert [x['clip_id'] for x in invalid]==['NIA_SL_WORD0500_REAL14_U_F']
 rows=[];archive={}
 for r in sorted(selected,key=lambda q:q['clip_id']):
  if not r['metadata_video_join_eligible']:continue
  sid=int(re.search(r'_REAL(\d+)',r['clip_id'])[1]);label=codes.index(r['word_code']);assert r['role']=='reserved_metadata_only' and r['labels']==[labels[label]] and not r['problems'];assert len(r['video_entries'])==1
  ent=r['video_entries'][0];assert ent['filename_signer_id']==sid==ent['folder_id'] and ent['extra'] is None
  st=Path(ent['archive']).stat();archive[ent['archive']]=[st.st_size,st.st_mtime_ns]
  rows.append({**r,'original_role':r['role'],'role':'heldout_evaluation_only','signer_id':sid,'label_index':label,'canonical_label':labels[label]})
 assert Counter(x['signer_id'] for x in rows)=={14:99,15:100,16:100};assert len(rows)==299
 priorids={r['clip_id'] for r in old['ordered_rows']}|{r['clip_id'] for r in oldtest['rows']};assert priorids.isdisjoint(r['clip_id'] for r in rows)
 hashes=set(oldtest['prior_source_sha256'])
 for p in (E/'data/records').glob('*.json'):hashes.add(read(p)['source_sha256'])
 oldentries={(r['video_entries'][0]['bytes'],r['video_entries'][0]['crc32']) for r in oldtest['rows']}
 oldentries|={(r['video_entries'][0]['bytes'],r['video_entries'][0]['crc32']) for r in read(O/'word100_training_20261013/MANIFEST.json')}
 assert not any((r['video_entries'][0]['bytes'],r['video_entries'][0]['crc32']) in oldentries for r in rows)
 # Adapt already-verified extraction and numerical input verification, never old files.
 s=(E/'common.py').read_text('utf-8').replace('O=U.parent;ROOT=O.parents[2]','O=U.parents[1];ROOT=O.parents[2]');(d/'common.py').write_text(s,encoding='utf-8')
 s=(E/'prepare_inputs.py').read_text('utf-8').replace('assert done==200','assert done==299').replace('total=200','total=299');(d/'prepare_inputs.py').write_text(s,encoding='utf-8')
 s=(E/'verify.py').read_text('utf-8').split('\ndef recount(scores):')[0];s=re.sub(r'\b200\b','299',s);s+='\nif __name__=="__main__":inputs("--early" in sys.argv)\n';(d/'verify_inputs.py').write_text(s,encoding='utf-8')
 s=(E/'evaluate.py').read_text('utf-8').replace('assert len(pred[v])==200','assert len(pred[v])==299')
 s=s.replace("for seed in SEEDS:\n   out=", "for seed in ((20261013,) if arm=='C' else SEEDS):\n   out=")
 s=s.replace('models=6,predictions=2400','models=4,predictions=2392');(d/'evaluate.py').write_text(s,encoding='utf-8')
 for p in d.glob('*.py'):ast.parse(p.read_text('utf-8'),str(p))
 models=oldtest['models'];pins={k:v for k,v in oldtest['pins'].items()}
 for p in [U/'FROZEN.json',U/'SAVED_VERIFICATION.json',U/'SAVED_COMPARISON.json',source,*d.glob('*.py')]:pins[str(p)]=sha(p)
 for p,h in pins.items():assert sha(p)==h,p
 context=dict(status='FROZEN_BEFORE_FRESH_EXTRACTION_AND_SCORES',at=now(),rows=rows,labels=labels,word_codes=codes,models=models,prior_source_sha256=sorted(hashes),archive_stat=archive,pins=pins,requested=299,unresolved_metadata_exclusion=[x['clip_id'] for x in invalid],raw_metadata_split='AIHub Training folder; project reserved person IDs14-16, never train on this cohort',candidate_frozen=str(U/'FROZEN.json'))
 write(d/'EVAL_CONTEXT.json',context)
 write(d/'FROZEN_PROTOCOL.json',dict(at=now(),cohort='All eligible public reserved100word REAL14-16=299; 1 metadata identity unresolved recorded before inference',candidate='Uniform softmax average of fixed T3F three seeds; no refit',models_evaluated=['C13','T3F13','T3F14','T3F15'],primary='sensor top1 E3 vs fixed T3F13',secondary='masked top1/top5/per-person, gains and regressions, exact paired uncertainty',success='sensor top1 higher; masked/top5 nondecreasing; per-person results all reported',exposure='One-time final check for this frozen candidate. Do not tune on outcomes or reuse as untouched after evaluation.',failures='Keep all299 including extraction/calibration failures in denominator as incorrect; metadata exclusion fixed from previous record, not new result filtering.',prohibited=['user/mentor footage','training or choosing parameters on fresh cohort','production model replacement','data download or external transfer']))
 write(d/'EXPOSURE_AUDIT.json',dict(at=now(),status='NO_LOCAL_PUBLIC_EXPERIMENT_EXPOSURE_FOUND_FOR_299_ELIGIBLE_IDS',search='rg public ops/runs/cache/pilot JSON/JSONL/PY/TXT, excluded inventory/backups/synthetic/private/user/web/session trees',hits=['word100_temporal_public_bundle_20261025/EVALUATION_PROTOCOL.json','word100_temporal_public_bundle_20261025/EVALUATION_KO.txt','word100_t3f_heldout_20261019/FROZEN_PROTOCOL.json','word100_t3f_heldout_20261019/freeze.py'],hit_reason='Only reserved-cohort description and already-known ineligible WORD0500_REAL14_U_F; no eligible per-video inference found.',trained_ids_disjoint=True,known200_disjoint=True,source_crc_size_disjoint=True,eligible_count=299,independent_people_ids=[14,15,16],limit='Local recorded public-work audit, not guarantee of external-world non-exposure; metadata Training folder is not our training set.'))
 assert shutil.disk_usage(d).free>12_000_000_000
 write(U/'CURRENT_EXECUTION.json',dict(at=now(),status='FRESH299_FROZEN_PENDING_PREFLIGHT',unit=str(d),records=299,source_bytes=sum(r['video_entries'][0]['bytes'] for r in rows),next='preflight then prepare, verify, one-pass evaluate, independent score audit',estimated_input_minutes=35),new=True)
 print('Frozen299',Counter(x['signer_id'] for x in rows),'bytes',sum(r['video_entries'][0]['bytes'] for r in rows))
if __name__=='__main__':main()
