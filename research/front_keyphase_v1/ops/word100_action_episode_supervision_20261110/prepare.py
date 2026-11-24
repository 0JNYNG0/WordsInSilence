from episode_core import *
from collections import Counter
import shutil,zipfile
assert not (U/'FROZEN.json').exists()
rr=old.rows();fit=[r for r in rr if r['role']=='fit'];dev=[r for r in rr if r['role']=='development'];ann=read(E/'REVIEW_LABELS.json');am={r['clip_id']:r for r in ann['rows']};boundary=read(B/'OBSERVED_BOUNDARIES.json')['rows'];contracts=family_contracts();checks=[]
def ck(n,v):checks.append(dict(name=n,passed=bool(v)));assert v,n
ck('100common',len(fit)==899 and len({r['class_index'] for r in fit})==100)
ck('disjoint_people',not {r['person'] for r in fit}&{r['person'] for r in dev})
ck('disjoint_sha',not {r['source_sha256'] for r in fit}&{r['source_sha256'] for r in dev})
episodes=[];counts=Counter()
for r in fit:
 y,w=lc.targets(r,am.get(r['clip_id']));y=np.array([MAP.get(int(v),-1) for v in y],np.int64)
 if r['clip_id'] in am:
  a=am[r['clip_id']];ck('review_provenance',sha(a['sheet'])==a['sheet_sha256'] and a['source_sha256']==r['source_sha256'])
 for b in boundary:
  if b['role']!='fit' or b['clip_id']!=r['clip_id']:continue
  ck('boundary_source',sha(b['sheet'])==b['sheet_sha256'] and b['source_sha256']==r['source_sha256'])
  spec=[c for c in contracts.get(r['word_id'],[]) if c['axis']==b['axis'] and not c.get('parent')]
  for ev in b['events']:
   # Directly reviewed onset/offset; family is a guide-clause link, not a new measurement.
   ep=dict(clip_id=r['clip_id'],role='fit',person=r['person'],axis=b['axis'],moving=b['moving'],anchor=b['anchor'],onset=ev['onset'],offset=ev['offset'],direction=ev['direction'],family=spec[0]['families'][0] if len(spec)==1 else None,guide=spec[0]['id'] if len(spec)==1 else None,source_sha256=r['source_sha256'],sheet=b['sheet'],sheet_sha256=b['sheet_sha256'],bounds='RGB reviewed intervals',middle='frames between reviewed onset and offset; not a percent-of-duration guess')
   episodes.append(ep)
   if ep['family'] is not None:
    a,hi=ev['onset'][0],ev['offset'][1];y[a:hi+1]=ep['family'];w[a:hi+1]=1
 counts.update(y[w==1].tolist())
 if w.sum():
  p=U/'labels'/f"{r['clip_id']}.npz";p.parent.mkdir(exist_ok=True);np.savez_compressed(p,y=y,w=w)
write(U/'ACTION_EPISODES.json',dict(rows=episodes,fit_role_videos=36,fit_boundary_videos=18,roles=FAMILY_NAMES,source_anchors=str(E/'REVIEW_LABELS.json'),source_boundaries=str(B/'OBSERVED_BOUNDARIES.json'),review='Existing Codex RGB annotations linked to guide clauses; not independent human expert annotations.',wrong_shape='Attempt identity persists; existing separate shape head retained.',unknown='Absent observations are not completed events.'))
write(U/'GUIDE_FAMILIES.json',dict(words=contracts,scope='Parent and arrival child share action family, position states are separate. No word-ID exception in inference.'))
# New feedback evaluation records are reserved before scores; these already had older TCN evaluation.
ex=read(O/'word100_completion_fullpath_20261110/EXPOSURE_AUDIT.json');extra=[r for r in ex['rows'] if r['person']==14];ck('four_extra_public',len(extra)==4);reserved=[]
for r in extra:
 ck('extra_sha_not_fit',r['source_sha256'] not in {x['source_sha256'] for x in fit+dev})
 p=U/'additional_public'/f"{r['clip_id']}.mp4";p.parent.mkdir(exist_ok=True)
 with zipfile.ZipFile(r['archive'],'r') as z:data=z.read(r['member'])
 ck('extra_source_sha',hashlib.sha256(data).hexdigest()==r['source_sha256']);p.write_bytes(data)
 reserved.append(dict(clip_id=r['clip_id'],person=14,word_id=r['word'],source=str(p),source_sha256=r['source_sha256'],role='additional_feedback_evaluation_only',prior_tcn_evaluation=r['earlier_tcn_evaluation'],fully_unseen_project_data=False))
write(U/'EXPOSURE_AUDIT.json',dict(rows=reserved,source_manifest=str(O/'word100_completion_fullpath_20261110/EXPOSURE_AUDIT.json'),source_manifest_sha256=sha(O/'word100_completion_fullpath_20261110/EXPOSURE_AUDIT.json'),scope='Reserved recordID14, not used for this feedback training/tuning. Older TCN evaluation exposure explicitly retained; no globally unseen claim.',quarantined_id16=True))
cfg=read(L/'FROZEN.json');bcfg=read(B/'FROZEN.json');pins={str(p):sha(p) for p in [U/'ACTION_EPISODES.json',U/'GUIDE_FAMILIES.json',U/'EXPOSURE_AUDIT.json',B/'OBSERVED_BOUNDARIES.json',E/'REVIEW_LABELS.json',L/'GUARDED_COMPLETE.json',T/'FINAL_RESULT.json']}
write(U/'FROZEN.json',dict(seeds=[42,43,44],primary_seed=42,epochs=12,steps=38,batch=24,extra_roles=4,extra_boundaries=4,lr_encoder=.0003,lr_head=.001,word_loss=.2,boundary_loss=1.,role_loss=1.,pos_weight=bcfg['pos_weight'],appearance_dropout_every=4,confidence=.65,protections=cfg['protections'],pins=pins,evaluation='78 historical conditions +4 reserved recordID14 originals. Manual reviewed and automatic boundaries compared separately; no parameter/seed selection from these scores.',adoption='Normal/speed/crossing and old shape/direction unchanged, reliable gain needed for candidate adoption; no web/model default replacement.'))
write(U/'DATA_AUDIT.json',dict(common_videos=899,common_words=100,fit_role_videos=36,fit_boundary_videos=18,fit_motion_events=len(episodes),guide_linked_events=sum(e['family'] is not None for e in episodes),role_direct_or_boundary_frame_counts=dict(counts),additional_public=4,additional_record_ids=[14],personal_data=0,new_independent_test=False))
write(U/'PRETRAIN_CHECK.json',dict(passed=True,count=len(checks),checks=checks));write(U/'AUTHORIZATION.json',dict(user='Current request all4 steps, episode annotations, manual-vs-auto diagnosis, supervised training and public comparison; next grounded plan',implementer='Codex',no_private=True,no_download=True,no_camera=True,no_default_change=True))
for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:
 (U/'before').mkdir(exist_ok=True);shutil.copy2(p,U/'before'/p.name)
summary='開始·進行·끝을 한 행동으로 연결하는 공통 사건학습 준비.100단어899유지,기존공개RGB경계와순간주석연결,직접구간/자동구간분리검증.추가공개ID14네영상예약(과거TCN노출명시),개인자료0.아직결과미확인.'
nxt='정답·코드검사→3seed고정학습→78개발/추가4공개/직접대자동비교→저장검산과다음근거계획. 기존완료작업재실행금지.'
p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';txt=p.read_text('utf-8-sig')
for heading,value in [('## 8.',summary),('## 9.',nxt)]:
 i=txt.index('\n',txt.index(heading))+1;txt=txt[:i]+'\n### 2026-11-10 한 행동의 시작·진행·끝 연결 진행\n\n'+value+'\n근거 '+str(U)+'\n\n'+txt[i:]
p.write_text(txt,'utf-8');s=read(O/'STATE.json');s.update(current_stage=U.name,next_single_action=nxt);s[U.name]=dict(status='PREPARED',summary=summary,next=nxt,execution=str(U/'CURRENT_EXECUTION.json'));write(O/'STATE.json',s)
(ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+summary+'\n\n다음 한 작업: '+nxt+'\n현재단위 '+str(U)+'\n','utf-8');write(U/'CURRENT_EXECUTION.json',dict(status='PREPARED',pid=None));event('prepared',events=len(episodes),linked=sum(e['family'] is not None for e in episodes))
