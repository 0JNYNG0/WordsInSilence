from role_core import *
import shutil,cv2
from PIL import Image,ImageDraw
assert not (U/'SETUP_COMPLETE.json').exists()
for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json',ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt']:
 (U/'before').mkdir(exist_ok=True);shutil.copy2(p,U/'before'/p.name)
write(U/'AUTHORIZATION.json',dict(user='검토한 1~4단계 계획대로 모두 진행',steps=['Reuse and verify public action-role annotations; augment candidate labels','Fine-tune existing quality/pair and role heads','Connect observed chronology, direction, missing versus unobserved, visual feedback evidence','Compare final feedback and verify saved outputs'],implementer='Codex',review='Separate saved arithmetic/code audit by same Codex; not independent KSL expert',private_data=False,web_default_change=False,downloads=False,camera=False))
write(U/'CURRENT_EXECUTION.json',dict(status='PREPARING',pid=None))
rows=old.rows();fit=[r for r in rows if r['role']=='fit'];dev=[r for r in rows if r['role']=='development'];ann=read(jc.ep.E/'REVIEW_LABELS.json')['rows'];chosen=[a for a in ann if a['person'] in [2,4,6,8,9]]
assert len(fit)==899 and len({r['class_index'] for r in fit})==100
assert not {r['source_sha256'] for r in fit}&{r['source_sha256'] for r in dev}
fm={r['clip_id']:r for r in fit};review=[]
for a in chosen:
 assert a['clip_id'] in fm and sha(a['sheet'])==a['sheet_sha256'] and a['source_sha256']==fm[a['clip_id']]['source_sha256']
 assert (P/'labels'/f"{a['clip_id']}.npz").exists()
 # Original RGB strips, including both ends. No model prediction is rendered.
 cap=cv2.VideoCapture(a['source']);assert cap.isOpened();n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));frames=sorted(set([0,n-1]+a['frames']));canvas=Image.new('RGB',(6*240,3*185),'white');draw=ImageDraw.Draw(canvas)
 for j,f in enumerate(frames):
  cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,img=cap.read();assert ok,(a['clip_id'],f)
  h,w=img.shape[:2];img=img[:int(h*.78),int(w*.16):int(w*.84)];im=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB));im.thumbnail((238,160));xx=(j%6)*240;yy=(j//6)*185;canvas.paste(im,(xx+(240-im.width)//2,yy+22));draw.text((xx+3,yy+3),f'{a["clip_id"][7:]} frame {f}',fill='black')
 cap.release();p=U/'review'/f"{a['clip_id']}.jpg";p.parent.mkdir(exist_ok=True);canvas.save(p,quality=92);review.append(dict(a,expanded_sheet=str(p),expanded_sheet_sha256=sha(p),source_frames=n,already_role_trained=True))
write(U/'REVIEW_INPUTS.json',dict(rows=review,not_new_twenty_training_videos=True,source='Existing public RGB; no predictions shown'))
write(U/'EXPOSURE_AUDIT.json',dict(fit_videos=899,fit_words=100,fit_record_ids=sorted({r['person'] for r in fit}),evaluation_record_ids=sorted({r['person'] for r in dev}),old_role_videos=len([a for a in ann if a['clip_id'] in fm]),planned_twenty_already_role_trained=True,evaluation='Historical 78 conditions and existing direct annotations are development replays, not a new independent test. Record15 four focus expressions already exposed. No claim of fresh data without per-clip exposure audit.',fit_dev_sha_disjoint=True,private_data=0))
write(U/'PROTECTIONS.json',read(A/'FROZEN.json')['protections'])
summary='승인1~4단계 시작. 기존공개 역할주석 재사용·후보별 행동역할/설명연결 보강.100단어공통부와기존손모양방향/C·T3F/웹보존.20대상영상은기존학습이력있음,새자료주장금지.개인자료0.'
s=read(O/'STATE.json');s.update(current_stage=U.name,next_single_action='공개RGB 역할정답 검토→고정3학습→후보역할/최종교정 비교→별도검산 및보고');s[U.name]=dict(status='PREPARING',summary=summary,execution=str(U/'CURRENT_EXECUTION.json'));write(O/'STATE.json',s)
(ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+summary+'\n\n현재단위 '+str(U)+'\n다음: '+s['next_single_action']+'\n기존완료실험재실행금지. 과거모델평가출력은저장결과를재사용한다.\n','utf-8')
write(U/'SETUP_COMPLETE.json',dict(review_videos=len(review),common_videos=len(fit)));event('setup_complete',review_videos=len(review))
