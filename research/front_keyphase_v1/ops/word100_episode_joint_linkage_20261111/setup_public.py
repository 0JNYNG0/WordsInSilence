from pathlib import Path
import json,hashlib,zipfile,datetime,shutil
import cv2
from PIL import Image,ImageDraw
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,o):Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2),'utf-8')
assert not (U/'REVIEW_INPUTS.json').exists()
extra=['WORD0003','WORD0500','WORD0678','WORD1139','WORD1290','WORD2551'];focus=['WORD0739','WORD1133','WORD1567','WORD2523']
sources={r['clip_id']:r for r in read(O/'word100_rtmw_joint_learning_20261108/SOURCES.json')}
ex=read(O/'word100_completion_fullpath_20261110/EXPOSURE_AUDIT.json')['rows'];exp={r['clip_id']:r for r in ex}
ids=[(f'NIA_SL_{w}_REAL{p:02d}_F','fit_additional' if p==3 else 'public_evaluation') for w in extra for p in [3,11]]+[(f'NIA_SL_{w}_REAL15_F','reserved_feedback_evaluation') for w in focus]
for d in ['public_rgb','review','before']:(U/d).mkdir(exist_ok=True)
write(U/'AUTHORIZATION.json',dict(date='2026-11-11',request='User approved all4 steps, overlapping linkage/action causes, generic public comparison,100word common learning,protect C/T3F and no personal training/tuning.',scope=['existing public RGB review','isolated supervised learning','saved baseline comparison','separate arithmetic verification','documentation'],forbidden=['private_or_mentor_inputs','download','external_transfer','git','camera','default_model_replacement'],reviewer='Codex, no independent human/expert claim'))
out=[];archives={};fitsha={r['source_sha256'] for r in sources.values() if r['role']=='fit'}
for clip,role in ids:
 s=sources.get(clip) or exp[clip];person=int(clip.split('REAL')[1].split('_')[0]);word=clip.split('_')[2];p=U/'public_rgb'/f'{clip}.mp4'
 z=archives.setdefault(s['archive'],zipfile.ZipFile(s['archive']));data=z.read(s['member']);assert hashlib.sha256(data).hexdigest()==s['source_sha256'];p.write_bytes(data)
 if person>=11:assert s['source_sha256'] not in fitsha
 cap=cv2.VideoCapture(str(p));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));frames=list(range(0,n,3));canvas=Image.new('RGB',(1500,((len(frames)+5)//6)*190),'white');d=ImageDraw.Draw(canvas)
 for j,f in enumerate(frames):
  cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,bgr=cap.read();assert ok;h,w=bgr.shape[:2];im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).crop((int(.26*w),int(.03*h),int(.74*w),int(.88*h)));im.thumbnail((246,166));x=j%6*250;y=j//6*190;canvas.paste(im,(x+(250-im.width)//2,y+22));d.text((x+3,y+3),f'{word} ID{person} f{f}',fill='black')
 cap.release();dst=U/'review'/f'{clip}.jpg';canvas.save(dst,quality=91)
 out.append(dict(clip_id=clip,role=role,person=person,word_id=word,source=str(p),source_sha256=s['source_sha256'],sheet=str(dst),sheet_sha256=sha(dst),frames=frames,original_frames=n,existing_rtmw=s.get('rtmw_cache'),earlier_tcn_evaluation=s.get('earlier_tcn_evaluation'),exposure='ID03 common FIT already used; newly reviewed fine labels' if person==3 else 'ID11 historical common development, new motion annotation for6expressions' if person==11 else 'ID15 old TCN evaluation, reserved for this feedback comparison; not globally unseen'))
for z in archives.values():z.close()
write(U/'REVIEW_INPUTS.json',dict(rows=out,selection='6 existing extra expressions x IDs03/11 plus4focus xID15, fixed before new model scores; no appearance-based identity judgments.'))
write(U/'CURRENT_EXECUTION.json',dict(status='PUBLIC_RGB_REVIEW',pid=None))
for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copy2(p,U/'before'/p.name)
summary='승인된4단계 실행 시작. 공통 구간연결과 행동역할의 단독/결합 비교를 준비한다. 기존4외6표현의 공개03/11 영상과4표현15를 예측 없이 원본 검토. C·T3F/웹/100단어공통 보존,개인자료0. 성능결과 아직 없음.'
nxt='공개RGB 정답 보강·출처분리→후보연결과 역할 판정부 동결→필요한 학습·4조건 비교→저장검산·보고. 과거완료실험 재실행금지.'
p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';t=p.read_text('utf-8')
for sec,body in [('## 8. 현재 상태',summary),('## 9. 다음 단일 작업',nxt)]:
 i=t.index('\n',t.index(sec))+1;t=t[:i]+'\n### 2026-11-11 구간·행동 결합 보완 진행\n\n'+body+'\n근거 '+str(U)+'\n'+t[i:]
p.write_text(t,'utf-8');s=read(O/'STATE.json');s.update(current_stage=U.name,next_single_action=nxt);s[U.name]=dict(status='PUBLIC_RGB_REVIEW',summary=summary,next=nxt,execution=str(U/'CURRENT_EXECUTION.json'));write(O/'STATE.json',s)
(ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+summary+'\n\n다음 한 작업: '+nxt+'\n현재 단위 '+str(U)+'\n','utf-8')
print(json.dumps(dict(review_videos=len(out),fit=6,evaluation=10),ensure_ascii=False))
