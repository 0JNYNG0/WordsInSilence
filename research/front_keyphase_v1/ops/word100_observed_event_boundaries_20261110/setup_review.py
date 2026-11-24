from pathlib import Path
import json,hashlib,zipfile,cv2,math,datetime
from PIL import Image,ImageDraw
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n','utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (U/'REVIEW_INPUTS.json').exists()
for name in ['review','public_rgb']:(U/name).mkdir(exist_ok=True)
focus=['WORD0739','WORD1133','WORD1567','WORD2523'];extra=['WORD0003','WORD0500','WORD0678','WORD1139','WORD1290','WORD2551']
ids=[f'NIA_SL_{w}_REAL{p:02d}_F' for w in focus for p in [1,3,7,11]]+[f'NIA_SL_{w}_REAL01_F' for w in extra]
sources={r['clip_id']:r for r in read(O/'word100_rtmw_joint_learning_20261108/SOURCES.json')}
anchors={r['clip_id']:r for r in read(O/'word100_event_completion_20261109/REVIEW_LABELS.json')['rows']}
base_pairs={r['clip_id']:r['pairs'] for r in read(O/'word100_shared_motion_learning_20261109/READY.json')['rows']};extra_pairs=read(O/'word100_broad_factorized_learning_20261110/ADDITIONAL_LABELS.json')['motion'];out=[]
write(U/'AUTHORIZATION.json',dict(user_request='공개 학습 영상의 실제 시작·방향 전환·끝 정답 보강 후 학습·기존 방식 비교.100단어 공통/기존 모양방향/C·T3F보존,개인촬영학습튜닝금지.',scope='Local public RGB review, observed boundary labels, frozen3seed continuation, saved baseline comparison; no web default change.',forbidden=['private_or_mentor_inputs','downloads','external_transfer','git','original_edits','default_model_replacement'],selection='Before new review:4focus expressions×IDs1,3,7fit+11historicalDEV;6additional expressions×ID1fit.10expressions18FIT4DEV. No model predictions on review sheets.',label_policy='Intervals for visually observed onset/turn/end, not exact hidden contact or expert semantic correctness. No20/60/20 groundtruth. Ambiguity explicitly bounded/unlabelled.',author='Codex; not independent human/expert'))
archives={}
for clip in ids:
 s=sources[clip];p=U/'public_rgb'/f'{clip}.mp4'
 if not p.exists():
  z=archives.setdefault(s['archive'],zipfile.ZipFile(s['archive']));p.write_bytes(z.read(s['member']))
 assert sha(p)==s['source_sha256']
 if clip in anchors:lo,hi=anchors[clip]['prior_main'];lo-=14;hi+=14
 else:pp=extra_pairs.get(clip) or base_pairs[clip];lo=min(q['a'] for q in pp)-16;hi=max(q['b'] for q in pp)+16
 cap=cv2.VideoCapture(str(p));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));lo=max(0,lo);hi=min(n-1,hi);frames=list(range(lo,hi+1,2));canvas=Image.new('RGB',(6*250,math.ceil(len(frames)/6)*200),'white');draw=ImageDraw.Draw(canvas)
 for j,f in enumerate(frames):
  cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,bgr=cap.read();assert ok
  rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);h,w=rgb.shape[:2];im=Image.fromarray(rgb).crop((int(.28*w),int(.05*h),int(.72*w),int(.85*h)));im.thumbnail((246,176));x=(j%6)*250;y=(j//6)*200;canvas.paste(im,(x+(250-im.width)//2,y+23));draw.text((x+3,y+3),f'{s["word_id"]} P{s["person"]} f{f}',fill='black')
 cap.release();dst=U/'review'/f'{clip}.jpg';canvas.save(dst,quality=90)
 out.append(dict(clip_id=clip,word_id=s['word_id'],person=s['person'],role=s['role'],source=str(p),source_sha256=sha(p),sheet=str(dst),sheet_sha256=sha(dst),frames=frames,original_frames=n,review_scope='RGB only, no learned prediction overlays'))
for z in archives.values():z.close()
write(U/'REVIEW_INPUTS.json',dict(rows=out));write(U/'CURRENT_EXECUTION.json',dict(status='RGB_REVIEW',pid=None))
summary='실제 RGB에서 동작 시작·방향전환·끝 경계를 확인하는 후속 진행.10표현18공개FIT/4역사DEV를 예측값 없이 검토.100단어899공통과 기존모양/방향·C·T3F/웹보존. 새 학습/성과 아직 미확인.'
nxt='현재단위 REVIEW_INPUTS와 경계주석 확인 후 사전검증·3seed학습·저장기준비교. 이전완료실험재실행금지. 불확실경계는 범위로 두며 실제관측과 약한추정분리. 사용자자료접근0.'
s=(ROOT/'WordsInSilence_PROJECT_CONTEXT.md').read_text('utf-8-sig')
for head,body in [('## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인',summary),('## 9. 다음 단일 작업',nxt)]:s=s.replace(head,head+'\n\n### 2026-11-10 실제 동작 경계 정답 보강 진행\n'+body+'\n근거 '+str(U)+'\n',1)
(ROOT/'WordsInSilence_PROJECT_CONTEXT.md').write_text(s,'utf-8');d=read(O/'STATE.json');d.update(current_stage=U.name,next_single_action=nxt,last_updated=datetime.datetime.now().astimezone().isoformat());d[U.name]=dict(status='RGB_REVIEW',summary=summary,next=nxt,execution=str(U/'CURRENT_EXECUTION.json'));write(O/'STATE.json',d)
(ROOT/'NEW_CHAT_HANDOFF.md').write_text('# WordsInSilence 최신 인수인계\n\n'+summary+'\n\n'+nxt+'\n현재단위 '+str(U)+'\n','utf-8')
print(json.dumps(dict(sheets=len(out),fit=sum(r['role']=='fit' for r in out),dev=sum(r['role']=='development' for r in out)),ensure_ascii=False))
