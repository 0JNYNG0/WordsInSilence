from common import *
import re,zipfile,collections,shutil
def main():
 backup=U/'freeze_first_attempt.py';assert not backup.exists();shutil.copyfile(U/'freeze_fresh.py',backup)
 ctx=read(B/'training_CONTEXT.json');codes=ctx['word_codes'];allrows=[json.loads(x) for x in (O/'word_dataset_plan_20261013/FRONT_VIDEO_JOIN.jsonl').read_text('utf-8').splitlines()]
 selected=[x for x in allrows if x['word_code'] in codes and 14<=int(re.search(r'_REAL(\d+)',x['clip_id'])[1])<=16 and x['metadata_video_join_eligible']]
 prior=read(O/'word100_training_20261013/MANIFEST.json')+read(E/'EVAL_CONTEXT.json')['rows'];keymap=collections.defaultdict(list)
 for r in prior:
  e=r['video_entries'][0];keymap[e['bytes'],e['crc32']].append(r)
 found=[]
 for r in selected:
  e=r['video_entries'][0];matches=keymap[e['bytes'],e['crc32']]
  if not matches:continue
  h=hashlib.sha256()
  with zipfile.ZipFile(e['archive']) as z:
   with z.open(e['member']) as stream:
    for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
  refs=[]
  for p in matches:
   record=read((B if p['clip_id'] in {x['clip_id'] for x in ctx['ordered_rows']} else E)/'data/records'/f"{p['clip_id']}.json")
   refs.append(dict(clip_id=p['clip_id'],role=p['role'],sha256=record['source']['sha256'] if 'source' in record and 'sha256' in record['source'] else record['source_sha256']))
  found.append(dict(reserved_clip_id=r['clip_id'],reserved_source=e,reserved_sha256=h.hexdigest(),matches=refs,exact_duplicate=any(p['sha256']==h.hexdigest() for p in refs)))
 assert len(found)==4 and all(x['exact_duplicate'] for x in found)
 write(U/'FIRST_FREEZE_ERROR.json',dict(at=now(),kind='Actual public dataset duplication detected before new score evaluation',stage='CRC_SIZE_PREFLIGHT',failing_assertion='reserved sources must be disjoint from prior1199/known200',error='4 REAL16 files share CRC/size with REAL11 DEV',raw_count=299,first_attempt_preserved=str(backup)))
 write(U/'DUPLICATE_AUDIT.json',dict(at=now(),status='EXACT_SHA256_DUPLICATES_CONFIRMED_BEFORE_EVAL',rows=found,decision='Do not count duplicated DEV clips as unseen people. Remove4from independent cohort before score access; keep excluded list and original300 accounting. Remaining295 plus1unresolved4duplicate=300. No threshold/score-based filtering.',remaining_by_signer={'14':99,'15':100,'16':96},warning='Whole signer16 may still share identity in other videos. Filename ID alone is not verified person identity; visually audit independent representative clips before calling3new people.'))
 print('4 exact SHA256 matches; 295 nonduplicate eligible clips remain.')
if __name__=='__main__':main()
