"""Saved-output visual audit only; no detection."""
from render import *

rows={r['clip_id']:r for r in read(U/'FROZEN.json')['rows']}
targets=read(U/'REVIEW_TARGETS.json')
ids=read(U/'SUPPLEMENTAL_COUNTS.json')['overlap_review_ids']+[141,142,159]
cards=[]
for tid in ids:
    a=targets[tid];r=rows[a['clip_id']];i=a['index']
    e=read(U/'records'/f"{a['clip_id']}.json")['frames'][i]
    with np.load(r['views']['sensor']['raw']) as z:
        base=z['hand_xy_px'][i,:int(z['raw_count'][i])].copy()
    cap=cv2.VideoCapture(r['video']);cap.set(cv2.CAP_PROP_POS_FRAMES,e['frame_id']);ok,bgr=cap.read();cap.release();assert ok
    bgr=bgr[:r['box'][3],:r['box'][2]]
    allpoints=list(base)
    for ad in a['additions']:allpoints.append(np.array(e[ad['mode']][ad['candidate_index']]['xy']))
    pp=np.array(allpoints).reshape(-1,2)
    mn=np.maximum(pp.min(0)-75,0).astype(int)
    mx=np.minimum(pp.max(0)+75,[bgr.shape[1],bgr.shape[0]]).astype(int)
    raw=bgr[mn[1]:mx[1],mn[0]:mx[0]].copy()
    combined=bgr.copy()
    for p in base:skeleton(combined,p,(255,150,0))
    for ad in a['additions']:
        skeleton(combined,np.array(e[ad['mode']][ad['candidate_index']]['xy']),
                 (0,210,0) if ad['mode']=='image' else (240,40,240))
    card=Image.new('RGB',(1100,380),'white');d=ImageDraw.Draw(card)
    d.text((8,3),f"{tid} {r['label']} {r['signer_id']:02d} f{i} | 파랑: 기존 / 초록: 전체 재검출 / 자홍: 국소 재검출",font=font,fill='black')
    card.paste(fit(bgr,(350,340)),(0,35))
    card.paste(fit(raw,(350,340)),(365,35))
    card.paste(fit(combined[mn[1]:mx[1],mn[0]:mx[0]],(350,340)),(735,35))
    card.save(U/'evidence'/f'detail_{tid:03d}.jpg',quality=96)
    cards.append(card)
for page in range(math.ceil(len(cards)/3)):
    sheet=Image.new('RGB',(1100,380*3),'#eee')
    for j in range(3):
        if page*3+j<len(cards):sheet.paste(cards[page*3+j],(0,j*380))
    sheet.save(U/'evidence'/f'overlap_{page+1}.jpg',quality=96)
print('Rendered',len(ids),'overlap/shape checks')
