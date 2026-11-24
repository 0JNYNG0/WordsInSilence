from link_core import *
from collections import Counter
import shutil,ast
assert not (U/'FROZEN.json').exists()
cfg=read(B/'FROZEN.json');ann=read(E/'REVIEW_LABELS.json');rr=old.rows();fit=[r for r in rr if r['role']=='fit'];dev=[r for r in rr if r['role']=='development'];am={r['clip_id']:r for r in ann['rows']};checks=[]
def ck(n,v):checks.append(dict(name=n,passed=bool(v)));assert v,n
ck('100common',len(fit)==899 and len({r['class_index'] for r in fit})==100)
ck('idsplit',not {r['person'] for r in fit}&{r['person'] for r in dev})
ck('sha_disjoint',not {r['source_sha256'] for r in fit}&{r['source_sha256'] for r in dev})
ck('all_anchors_fit',set(am)<={r['clip_id'] for r in fit})
counts=Counter();nvalid=0;fine=0
for r in fit:
 y,w=targets(r,am.get(r['clip_id']));counts.update(y[w==1].tolist());nvalid+=int((w==1).sum());fine+=int(bool((w>0).sum()))
 if r['clip_id'] in am:
  a=am[r['clip_id']];ck('review_sha:'+r['clip_id'],sha(a['sheet'])==a['sheet_sha256'] and a['source_sha256']==r['source_sha256'])
 ck('labelvalid:'+r['clip_id'],np.all((y>=0)&(y<=9)|(w==0)))
 p=U/'labels'/f"{r['clip_id']}.npz"
 if w.sum():p.parent.mkdir(exist_ok=True);np.savez_compressed(p,y=y,w=w)
contracts=compile_cards();ck('optional_arrival_not_obligatory',all(not c['required'] for cc in contracts.values() for c in cc if c.get('parent')))
write(U/'GUIDE_CONTRACTS.json',dict(words=contracts,source=str(be.ev.D/'CARDS.json'),source_sha256=sha(be.ev.D/'CARDS.json'),scope='Structured visible body contexts and movement relations; not LLM comprehension or expert correctness labels. Finger shape is checked separately, never required for event identity.'))
# Visual review of already-public original sheets, before candidate outputs.
selected={
 'WORD0739':{1:[53,57,61,65,67],2:[75,77,79],3:[89,93,97,101,103],0:[115]},
 'WORD1133':{0:[40,42,88,90,92],4:[50,52,54,56,58,60,62],5:[66,68,70,72,74,76,78]},
 'WORD1567':{0:[34,36,38],6:[48,50,52,66,68,70,72],7:[58,60,62,78,80,82,84,86,88]},
 'WORD2523':{0:[38,86,88],8:[48,50,52,54,56],9:[64,66,68,70,72,74,76]}}
inputs={r['clip_id']:r for r in read(B/'REVIEW_INPUTS.json')['rows']};gold=[]
for word,dd in selected.items():
 cid=f'NIA_SL_{word}_REAL11_F';inp=inputs[cid];ck('dev_review_sha',sha(inp['sheet'])==inp['sheet_sha256']);gold.append(dict(clip_id=cid,role='development',anchors=[dict(frame=f,label=k) for k,ff in dd.items() for f in ff],sheet=inp['sheet'],sha256=inp['sheet_sha256']))
write(U/'LINK_LABELS.json',dict(fit_source=str(E/'REVIEW_LABELS.json'),fit_sha256=sha(E/'REVIEW_LABELS.json'),fit_mapping='label0->background; positive(label+1)//2 collapses progress/core to visible body context',rows=gold,review='Same Codex image review, not independent expert; only listed frames, no transition interpolation in evaluation.',contexts=['background','hand near face','forearm across chest','forearm upright','along opposite forearm','near opposite wrist','right hand higher','left hand higher','hands together near face','hands apart']))
ex=read(O/'word100_completion_fullpath_20261110/EXPOSURE_AUDIT.json');fresh=read(O/'word100_completion_fullpath_20261110/FRESH_COMPLETE.json')
write(U/'DATA_AUDIT.json',dict(common_fit=899,common_words=100,fit_anchor_videos=fine,fit_anchors=nvalid,fit_class_counts=dict(counts),fit_record_ids=sorted({r['person'] for r in fit}),development_record_ids=sorted({r['person'] for r in dev}),new_review_frames=sum(len(r['anchors']) for r in gold),new_review_videos=4,new_review_record_ids=[11],independent_new_test=False,reason='ID10 candidates in previous exposure audit have already been evaluated in FRESH_COMPLETE; ID11-13 historicalDEV. ID14-15 previously TCN-evaluated, ID16 prior duplicate quarantine; no unseen set claimed.',prior_exposure_audit=str(O/'word100_completion_fullpath_20261110/EXPOSURE_AUDIT.json'),prior_fresh_completed=bool(fresh),scope='Historical development comparison; no personal sources accessed.'))
for p in U.glob('*.py'):ast.parse(p.read_text('utf-8'))
prot=cfg['protections'];pins={str(p):sha(p) for p in [B/'FULLPATH_COMPLETE.json',B/'RESULT.json',B/'runs/42/model.pt',E/'REVIEW_LABELS.json',U/'LINK_LABELS.json',U/'GUIDE_CONTRACTS.json',U/'link_core.py']}
write(U/'FROZEN.json',dict(seeds=[42,43,44],primary_seed=42,epochs=12,steps=38,batch=24,extra_supervised=6,lr_encoder=.0003,lr_head=.001,word_loss=.2,threshold=.65,minimum_span_us=60000,protections=prot,pins=pins,classes=10,all100_per_epoch=True,direction_threshold=.65,adoption='Normal/speed/crossing and old shape/direction preserved; downstream warning gain required, no false normal promotion for missing. No production replacement.',evaluation='78 stored public conditions x3 seeds, 4 historicalDEV original RGB reviewed frame contexts; no baseline TCN/DTW rerun.',reversal='Time reversal teaches context identity, not a normal-word label. Whole-core reversal is not automatically a validated sign error; report fixture detection separately.'))
write(U/'PRETRAIN_CHECK.json',dict(passed=True,count=len(checks),checks=checks));write(U/'CURRENT_EXECUTION.json',dict(status='PREPARED',pid=None));write(U/'AUTHORIZATION.json',dict(user='2026-11-10 found movements to guide actions; direction/order/omission; preserve100common,C,T3F; no usertraining/tuning',implementation='Codex local',scope='Isolated research learning and saved baseline comparison',no_download=True,no_camera=True,no_private=True,no_default_change=True))
for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:
 (U/'before').mkdir(exist_ok=True);shutil.copy2(p,U/'before'/p.name)
summary='관측 사건을 설명 행동과 연결하는 공통 신경망 보강 준비. 100단어899공통 유지, 기존 공개36FIT의 관측 주석을 신체관계9종+배경으로 분리, 손모양·방향·기존DTW동결. 4역사DEV 추가시점주석 확인, 아직새성능미확인.'
nxt='LINK_LABELS/GUIDE_CONTRACTS 검증 후 3seed학습과 저장78자동조건 비교·별도검산. 특정단어예외/DEV튜닝/개인자료/기본웹교체금지.'
p=ROOT/'WordsInSilence_PROJECT_CONTEXT.md';txt=p.read_text('utf-8-sig')
for heading,text in [('## 8.',summary),('## 9.',nxt)]:
 i=txt.index('\n',txt.index(heading))+1;txt=txt[:i]+'\n### 2026-11-10 관측 사건과 설명 행동 연결 진행\n\n'+text+'\n근거 '+str(U)+'\n\n'+txt[i:]
p.write_text(txt,'utf-8');s=read(O/'STATE.json');s.update(current_stage=U.name,next_single_action=nxt);s[U.name]=dict(status='PREPARED',summary=summary,next=nxt,execution=str(U/'CURRENT_EXECUTION.json'));write(O/'STATE.json',s)
(ROOT/'NEW_CHAT_HANDOFF.md').write_text('# 최신 인수인계\n\n'+summary+'\n\n'+nxt+'\n현재단위 '+str(U)+'\n','utf-8');event('prepared',fit=899,anchor_videos=fine,anchors=nvalid,counts=dict(counts))
