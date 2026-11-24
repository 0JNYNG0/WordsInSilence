from pathlib import Path
import json,sys,importlib.util,zipfile,hashlib
U=Path(__file__).resolve().parent;O=U.parent
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
write=lambda p,v:Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n','utf-8')
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
project=load('acceptance_projection',O/'word100_visual_feedback_g5_20261114/adapter.py').project
extension=load('acceptance_fingers',O/'word100_finger_repair_20261115/projection_extension.py')
guides=read(O/'word100_action_episode_supervision_20261110/GUIDE_FAMILIES.json')['words'];checks=[]
for r in read(U/'STATIC_LINK_RESULTS.json')['rows']:
 row=read(r['output'])['after'];g=[s for s in guides[r['clip_id'].split('_')[2]]if s['required']];p=extension.project_fingers(project,row,g,row['runtime']['frames']);old=read(r['output'])['before'];b=extension.project_fingers(project,old,g,old['runtime']['frames'])
 assert len(p)==len(b);assert all(x['whole_sign_correct']is None for x in p)
 if r['kind']=='normal':assert sum(x['status']=='difference'for x in p)<=sum(x['status']=='difference'for x in b)
 if r['kind']=='prefix_before_goal':assert any(x['status']!='similar'for x in p)
 checks.append(dict(clip_id=r['clip_id'],kind=r['kind'],before=[x['status']for x in b],after=[x['status']for x in p],passed=True))
write(U/'PROJECTION_VERIFIED.json',dict(passed=True,cases=len(checks),rows=checks,scope='Actual unchanged web projection applied to paired runtime results, no camera/browser interaction.'))

inventory=read(O/'demo_scope_portability_review_20261113/INVENTORY.json');found=[]
archives=sorted(Path('W:/WordsInSilence/data/AIHub_103/videos').rglob('*.zip'))
for ap in archives:
 with zipfile.ZipFile(ap)as z:
  for item in z.infolist():
   if 'SEN0354' in item.filename and Path(item.filename).suffix.lower()=='.mp4':found.append(dict(archive=str(ap),member=item.filename))
target_rows=[]
for code in ['WORD0099','WORD1290']:
 rr=inventory['targets'][code];verified=[]
 for r in rr:
  ap=Path(r['archive'])
  with zipfile.ZipFile(ap)as z:info=z.getinfo(r['member'])
  assert info.file_size==r['bytes']and f'{info.CRC:08x}'==r['crc32']
  verified.append(r['record_id'])
 target_rows.append(dict(word=code,frontal_records=verified,metadata_entry_crc_verified=True,full_file_crc_reread=False,recognition_current100=(code=='WORD1290'),detailed_feedback_ready=False))
greeting=[p for p in inventory['previews']if 'SEN0354'in str(p)]
for p in greeting:
 if 'video'in p and 'sha256'in p:assert hashlib.sha256(Path(p['video']).read_bytes()).hexdigest()==p['sha256']
write(U/'DEMO_SCOPE_REVIEW.json',dict(user_revision='Old4words excluded from actual game demo, retained only as generic regression evidence. Actual target basic game vocabulary to be selected.',candidate_words=target_rows,greeting=dict(target='AIHub 안녕하세요 SEN0354, not NIKL 인사',current_word_archive_count=len(archives),found_in_word_archives=found,existing_local_preview=greeting,status='One previously verified frontal reference; multi-person source set not established'),prior_apple_review='WORD0099 person05 source variant unresolved; exclude from same-expression FIT until meaning confirmed, not label as wrong.',suggested_order=['감사: current100 recognition and18source records; structured action truth/completion/error feedback still required','사과:18sources, extra recognition output candidate exists but not admitted; resolve source variant and train/verify feedback','안녕하세요: locate adequate multi-person source first; do not substitute another greeting sign'],not_done=['No new basic-word training this turn','No game word-list replacement or quest success change','No new downloads','No private footage'],presentation_priority='Working gameplay first, actual demo evidence second. Do not defer core success/error decisions to S-CURT.'))
print('projection26 verified; basic-word source availability checked',len(archives),'archives; greetings',len(found))
