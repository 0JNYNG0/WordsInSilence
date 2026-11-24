from common import *
import zipfile,collections,re
assert not (F/'EXPOSURE_AUDIT.json').exists()
inventory=read(F/'PUBLIC_ARCHIVE_INVENTORY.json')['rows'];rows=[];groups={}
for r in inventory:groups.setdefault(r['archive'],[]).append(r)
for p,rr in groups.items():
 with zipfile.ZipFile(p) as z:
  for r in rr:
   data=z.read(r['member']);rows.append(dict(r,source_sha256=hashlib.sha256(data).hexdigest()))
known={r['clip_id']:r for r in read(J/'SOURCES.json')}
old_eval={}
for folder in ['word100_t3f_heldout_20261019/data/records','word100_autonomous_recognition_20261026/fresh199/data/records']:
 for r in rows:
  p=O/folder/(r['clip_id']+'.json')
  if p.exists():
   d=read(p);old_eval[r['clip_id']]=dict(path=str(p),source_sha256=d['source_sha256'])
hashes=collections.defaultdict(list)
for r in rows:hashes[r['source_sha256']].append(r['clip_id'])
for r in rows:
 r['exact_duplicates']=[c for c in hashes[r['source_sha256']] if c!=r['clip_id']];r['current_joint_pipeline_role']=known.get(r['clip_id'],{}).get('role');r['earlier_tcn_evaluation']=old_eval.get(r['clip_id']);r['folder_identity_consistent']=int(re.search(r'(\d+)(?:-1)?/',r['member']).group(1))==r['person']
 r['eligible_untouched_candidate']=r['person'] not in list(range(1,10))+[11,12,13,16] and not r['exact_duplicates'] and r['folder_identity_consistent'] and not r['earlier_tcn_evaluation']
write(F/'EXPOSURE_AUDIT.json',dict(rows=rows,archives=33,untouched_candidates=[r['clip_id'] for r in rows if r['eligible_untouched_candidate']],scope='Exact byte duplicates and recorded IDs, no face identity inference. ID16 quarantined by prior audit. Prior TCN-evaluated sources not called project-untouched.',prior_tcn_record_paths=[x['path'] for x in old_eval.values()]))
print('EXPOSURE_AUDIT',dict(untouched=[r['clip_id'] for r in rows if r['eligible_untouched_candidate']],duplicates={r['clip_id']:r['exact_duplicates'] for r in rows if r['person'] in [10,16,17,18] and r['exact_duplicates']}))
