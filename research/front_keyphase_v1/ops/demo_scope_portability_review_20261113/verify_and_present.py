"""Verify cached inventory against original FIT annotations and show RGB examples."""
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
from PIL import Image, ImageDraw, ImageFont

U=Path(__file__).resolve().parent
ROOT=U.parents[3]
SRC=Path('W:/WordsInSilence/data/AIHub_103')
r=json.loads((U/'INVENTORY.json').read_text(encoding='utf-8'))
checks=[]

def check(name, ok):
    checks.append({'check':name, 'pass':bool(ok)})

with zipfile.ZipFile(next((SRC/'annotations/1.Training').glob('*.zip'))) as z:
    for code, rows in r['targets'].items():
        fit=[x for x in rows if 1<=x['record_id']<=9]
        check(code+' FIT recorded IDs 01..09', sorted(x['record_id'] for x in fit)==list(range(1,10)))
        check(code+' unique front records', len(rows)==len({x['clip_id'] for x in rows})==18)
        for row in fit:
            meta=row['metadata']
            raw=z.read(meta['member'])
            check(row['clip_id']+' original annotation hash', hashlib.sha256(raw).hexdigest()==meta['source_payload_sha256'])

decodes=[]
for preview in r['previews']:
    cap=cv2.VideoCapture(preview['video'])
    count=0
    while True:
        ok, im=cap.read()
        if not ok:break
        count+=1
    cap.release()
    check(preview['clip_id']+' decoded all frames', count==preview['total_frames'])
    decodes.append({'clip_id':preview['clip_id'], 'frames':count})

# This is the already acquired SEN source, not a new equivalence to NIKL 인사.
gpath=ROOT/'research/front_keyphase_v1/runs/greeting_source_review_v1/raw/NIA_SL_SEN0354_REAL01_F.mp4'
acq=json.loads((ROOT/'research/front_keyphase_v1/evidence/greeting_candidate_source_20260918/acquisition.json').read_text(encoding='utf-8-sig'))
g=next(x for x in acq['videos'] if x['raw_label']=='안녕하세요')
check('greeting previously acquired public source hash',hashlib.sha256(gpath.read_bytes()).hexdigest()==g['sha256'])
cap=cv2.VideoCapture(str(gpath));n=0
while True:
    ok, im=cap.read()
    if not ok:break
    n+=1
cap.release()
check('greeting RGB decodes', n>0)

font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',30)
small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',20)
board=Image.new('RGB',(1080,540),'white');d=ImageDraw.Draw(board)
d.text((22,14),'HDD 원본에서 확인한 숫자 손모양',font=font,fill='#172b49')
for i,(code,f) in enumerate([('WORD2633',41),('WORD2635',50),('WORD2637',62)]):
    p=next(x for x in r['previews'] if x['code']==code)
    cap=cv2.VideoCapture(p['video']);cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read();cap.release();assert ok
    h,w=im.shape[:2];crop=im[int(h*.12):int(h*.68),int(w*.28):int(w*.68)]
    image=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB));image.thumbnail((344,350))
    x=12+i*360
    board.paste(image,(x+(344-image.width)//2,66))
    d.text((x+12,385),f'숫자 { [2,4,6][i] }',font=font,fill='#172b49')
    d.text((x+12,432),f'{f/30:.2f}초 · {code}',font=small,fill='#344862')
d.text((22,491),'공개 영상의 실제 장면 · 모양 유지 예시 · 새 AI 판정 결과가 아님',font=small,fill='#344862')
board.save(U/'numbers_2_4_6.png')

# Compact greeting preview, preserving original chronological frames.
board=Image.new('RGB',(1280,510),'white');d=ImageDraw.Draw(board)
d.text((18,10),'안녕하세요 · 별도로 확보해 둔 공개 SEN0354 영상',font=font,fill='#172b49')
cap=cv2.VideoCapture(str(gpath))
for i,f in enumerate([38,50,63,77]):
    cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read();assert ok
    h,w=im.shape[:2];crop=im[int(h*.05):int(h*.85),int(w*.28):int(w*.72)]
    image=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB));image.thumbnail((310,390))
    board.paste(image,(5+320*i+(310-image.width)//2,62))
    d.text((16+320*i,466),f'{f/30:.2f}초',font=small,fill='#172b49')
cap.release();board.save(U/'greeting.png')

report={'status':'PASS' if all(x['pass'] for x in checks) else 'FAIL',
        'checks':checks,'decoded_public_record01':decodes,
        'greeting':{'source':str(gpath),'frames':n,'raw_label':'안녕하세요','collection':'SEN, separately cached on C:',
                    'not_word3000':True,'not_current100':True,'multisigner_rgb_ready':False,
                    'past_decision':'Do not merge with NIKL 인사; 2026-09-22 SEMANTIC_DECISION.json'},
        'scope':'FIT annotation hashes and representative RGB only. No model performance or educational correctness evaluation.',
        'operator':'Codex', 'reviewer':'same Codex separate arithmetic/source checks; no independent human KSL review'}
(U/'VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':report['status'],'checks':len(checks),'failed':[x for x in checks if not x['pass']], 'greeting_frames':n},ensure_ascii=False))
