"""Public-data diagnosis only. No fitting, no user observation reads, no inference."""
import json, hashlib, zipfile
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent
B=U.parents[1]
P=B/'ops/word100_body_boundary_20261017'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
words=['WORD0500','WORD0003','WORD1642','WORD1884']
manifest=read(B/'ops/word100_training_20261013/MANIFEST.json')
rows=[r for r in manifest if r['word_code'] in words and r['role']=='train_candidate']
assert len(rows)==36 and all(1<=r['signer_id']<=9 for r in rows)
scope={'date':datetime.now().isoformat(),'mode':'PUBLIC_TRAIN_DIAGNOSTIC_ONLY',
'selection_reason':'Previously reported confusion labels, not a user-data fitting objective.',
'words':words,'train_ids':list(range(1,10)),'clips':36,
'measurements_fixed_before_execution':['original annotation consistency','core interval coverage by whole/context spans',
'per-hand palm5 observation fraction and longest missing run within annotation','sampling timestamps',
'stored C development predictions reused across 3 seeds, 2 views','10 evenly spaced RGB frames inside annotation per source'],
'prohibited':['training','user observation loading','new model inference','threshold selection','model/web changes'],
'interpretation':'Descriptive selected-class audit, not independent model-performance evaluation.'}
(U/'SCOPE.json').write_text(json.dumps(scope,ensure_ascii=False,indent=2),encoding='utf-8')
(U/'public_videos').mkdir(exist_ok=True)
annotation_zip=Path('W:/WordsInSilence/data/AIHub_103/annotations/1.Training/[라벨]01_real_word_morpheme.zip')
az=zipfile.ZipFile(annotation_zip); anames=az.namelist();archives={};out=[];pins={}
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',19)
small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',16)
def maxrun(x):
    cur=longest=0
    for v in x:
        cur=cur+1 if v else 0;longest=max(longest,cur)
    return longest
for word in words:
    sheet=Image.new('RGB',(1800,9*142),'#f2f6fb');draw=ImageDraw.Draw(sheet)
    for j,m in enumerate(sorted([r for r in rows if r['word_code']==word],key=lambda x:x['signer_id'])):
        rid=m['clip_id'];rp=P/'data/records'/f'{rid}.json';rec=read(rp);pins[str(rp)]=sha(rp)
        src=rec['source'];old=B/'ops/word100_role_dtw_20261017/public_videos'/f'{rid}.mp4'
        path=old if old.exists() else U/'public_videos'/f'{rid}.mp4'
        if not path.exists():
            if src['archive'] not in archives:archives[src['archive']]=zipfile.ZipFile(src['archive'])
            raw=archives[src['archive']].read(src['member'])
            assert hashlib.sha256(raw).hexdigest()==src['sha256']
            path.write_bytes(raw)
        assert sha(path)==src['sha256']
        amatches=[n for n in anames if n.endswith(f'{rid}_morpheme.json')]
        assert len(amatches)==1
        araw=az.read(amatches[0]);ann=json.loads(araw.decode('utf-8-sig'))
        labels=[attr['name'] for d in ann['data'] for attr in d['attributes']]
        intervals=[[d['start'],d['end']] for d in ann['data']]
        assert labels==m['labels'] and intervals==m['intervals']
        a,b=intervals[0]
        row={'clip_id':rid,'word_code':word,'label':m['canonical_label'],'signer_id':m['signer_id'],
             'video':str(path),'source':src,'annotation_member':amatches[0],
             'annotation_sha256':hashlib.sha256(araw).hexdigest(),'annotation':ann,
             'core_seconds':b-a,'views':{}}
        cap=cv2.VideoCapture(str(path));fps=cap.get(cv2.CAP_PROP_FPS)
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        row['video_fps']=fps;row['video_frames']=count
        draw.text((4,j*142),f"{m['canonical_label']} | TRAIN {m['signer_id']:02} | annotation {a:.2f}-{b:.2f}s",font=font,fill='#172a41')
        for k,t in enumerate(np.linspace(a,b,10)):
            f=min(count-1,round(t*fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(f));ok,bgr=cap.read();assert ok
            rgb=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).resize((180,101))
            sheet.paste(rgb,(k*180,j*142+23))
            draw.text((k*180+4,j*142+124),f'{f/fps:.2f}s',font=small,fill='#172a41')
        cap.release()
        for view in ['native','masked','sensor']:
            v=rec['views'][view];op=Path(v['observation']);cp=Path(v['channel_file'])
            for f in [op,cp]:
                actual=sha(f);assert rec['outputs'][str(f)]==actual; pins[str(f)]=actual
            obs=np.load(op);ch=np.load(cp)
            ts=obs['timestamp_us']/1e6;core=(ts>=a)&(ts<=b);assert core.any()
            usable=obs['usable'].astype(bool);observed=obs['observed'].astype(bool)
            rawdt=np.diff(ts);positive=rawdt[rawdt>0]
            info={'fps_median':float(1/np.median(positive)),'core_frames':int(core.sum()),'hands':{}}
            for kind in ['whole','context']:
                lo,hi=v['feature_info'][kind+'_span']
                inside=core & (np.arange(len(ts))>=lo)&(np.arange(len(ts))<=hi)
                info[kind]={'span':[lo,hi],'seconds':float(ts[hi]-ts[lo]),
                            'frames':int(hi-lo+1),'core_frames_retained':int(inside.sum()),
                            'core_coverage_fraction':float(inside.sum()/core.sum()),
                            'channel_frames':int(ch['x_'+kind].shape[0])}
                assert info[kind]['frames']==info[kind]['channel_frames']
            for side,start in [('L',33),('R',54)]:
                palm=np.array([0,5,9,13,17])+start
                good=usable[:,palm].all(1)
                obs_good=(usable[:,palm]&observed[:,palm]).all(1)
                info['hands'][side]={'usable_palm5_fraction':float(good[core].mean()),
                     'observed_palm5_fraction':float(obs_good[core].mean()),
                     'longest_missing_frames':maxrun(~good[core]),
                     'longest_missing_seconds_nominal':maxrun(~good[core])*float(np.median(positive))}
            row['views'][view]=info
        out.append(row)
    sheet.save(U/f'{word}_CORE_TRAIN9.jpg',quality=95)
    print(word,'9 audited',flush=True)
for z in archives.values():z.close()
az.close()
classes={w:next(r for r in manifest if r['word_code']==w)['label_index'] for w in words}
class_names={r['label_index']:r['canonical_label'] for r in manifest}
dev=[]
for seed in [20261013,20261014,20261015]:
    p=P/f'training_C_{seed}_RESULT.json';r=read(p);pins[str(p)]=sha(p)
    for view in ['masked','sensor']:
        for pred in r['predictions'][view]:
            if pred['label_index'] not in classes.values():continue
            order=np.argsort(-np.array(pred['logits']),kind='stable')
            dev.append({'seed':seed,'view':view,'clip_id':pred['clip_id'],'signer_id':pred['signer_id'],
                        'label':class_names[pred['label_index']],'correct':bool(order[0]==pred['label_index']),
                        'target_rank':int(np.where(order==pred['label_index'])[0][0]+1),
                        'top3':[class_names[int(i)] for i in order[:3]],
                        'logit_margin':float(pred['logits'][pred['label_index']]-max(v for i,v in enumerate(pred['logits']) if i!=pred['label_index']))})
def stats(v):
    return {'min':float(min(v)),'median':float(np.median(v)),'max':float(max(v))}
summaries={}
for w in words:
    wr=[r for r in out if r['word_code']==w]
    ss={'label':wr[0]['label'],'train_videos':9,'core_seconds':stats([r['core_seconds'] for r in wr]),'views':{}}
    for view in ['native','masked','sensor']:
        ss['views'][view]={'whole_seconds':stats([r['views'][view]['whole']['seconds'] for r in wr]),
          'core_coverage_min':min(r['views'][view]['whole']['core_coverage_fraction'] for r in wr),
          'context_coverage_min':min(r['views'][view]['context']['core_coverage_fraction'] for r in wr),
          'palm5_observed':{s:stats([r['views'][view]['hands'][s]['observed_palm5_fraction'] for r in wr]) for s in ['L','R']}}
    dd=[r for r in dev if r['label']==ss['label']]
    ss['stored_dev']={'unique_clips':len(set(r['clip_id'] for r in dd)),'predictions':len(dd),'correct':sum(r['correct'] for r in dd),'errors':[r for r in dd if not r['correct']]}
    summaries[w]=ss
result={'scope':scope,'annotation_archive':str(annotation_zip),'records':out,'summary':summaries,
        'stored_development_predictions':dev,'pins':pins,'inference_runs':0,'training_runs':0,'user_observation_reads':0}
(U/'AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summaries,ensure_ascii=False,indent=2))

