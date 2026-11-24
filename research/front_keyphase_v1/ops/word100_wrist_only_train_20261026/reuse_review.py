"""Reuse exact already-viewed public frames; render only outstanding final risk panels."""
from pathlib import Path
import json,sys,hashlib,datetime,math
import numpy as np
sys.stdout.reconfigure(encoding='utf8')
J=Path('C:/MyUnityProjects/WordsInSilence/research/front_keyphase_v1/ops/word100_wrist_only_train_20261026')
I=J.parent/'word100_wrist_only_inputs_20261026'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key(r):return (r['uid'],r['index'],r['side'])
def main(unit):
    import cv2
    u=Path(unit).resolve();risk=read(u/'RISK_INDEX.json');verified=read(u/'ALL_VERIFIED.json')
    assert risk['verified_sha256']==sha(u/'ALL_VERIFIED.json') and verified['status']=='PASS'
    dest=u/'pending_visual';assert not dest.exists();dest.mkdir()
    reuse={};pins={str(u/'RISK_INDEX.json'):sha(u/'RISK_INDEX.json'),str(u/'ALL_VERIFIED.json'):sha(u/'ALL_VERIFIED.json')}
    if u==I:
        src=J/'interim_visual_v2';ri=read(src/'INDEX.json');rr=read(src/'REVIEW.json');assert rr['reviewed_frames']==len(ri['rows'])
        for r in ri['rows']:
            if sha(r['output'])==r['output_sha256'] and sha(r['image'])==ri['pins'][r['image']]:reuse[key(r)]=str(src/'REVIEW.json')
        pins[str(src/'REVIEW.json')]=sha(src/'REVIEW.json');pins[str(src/'INDEX.json')]=sha(src/'INDEX.json')
        src=I/'incremental_yolo_v1';ri=read(src/'INDEX.json');rr=read(src/'REVIEW.json');assert rr['index_sha256']==sha(src/'INDEX.json')
        assert rr['reviewed_indices']==list(range(len(ri['rows'])))
        for r in ri['rows']:
            assert sha(r['output'])==r['output_sha256'] and sha(r['record'])==r['record_sha256']
            assert sha(r['image'])==ri['pins'][r['image']]
            reuse[key(r)]=str(src/'REVIEW.json')
        pins[str(src/'REVIEW.json')]=sha(src/'REVIEW.json');pins[str(src/'INDEX.json')]=sha(src/'INDEX.json')
    chosen=[r for r in risk['rows'] if 'visual_page' in r];pending=[];reused=[];cache={}
    for k,r in enumerate(chosen):
        assert r['visual_page']==k//8
        if key(r) in reuse:reused.append(dict(uid=r['uid'],index=r['index'],side=r['side'],review= reuse[key(r)],final_review_index=r['review_index']));continue
        page=u/'full_visual'/f'risk_{k//8:03d}.jpg'
        if str(page) not in cache:cache={str(page):cv2.imread(str(page))};assert cache[str(page)] is not None;pins[str(page)]=sha(page)
        q=k%8;panel=cache[str(page)][q//2*320:(q//2+1)*320,q%2*640:(q%2+1)*640]
        assert panel.shape==(320,640,3)
        j=len(pending);a=j%8
        if a==0:sheet=np.full((1280,1280,3),255,np.uint8)
        sheet[a//2*320:(a//2+1)*320,a%2*640:(a%2+1)*640]=panel
        pending.append(dict(r,pending_index=j,pending_page=j//8))
        if a==7:assert cv2.imwrite(str(dest/f'pending_{j//8:03d}.jpg'),sheet,[cv2.IMWRITE_JPEG_QUALITY,97])
    if pending and len(pending)%8:assert cv2.imwrite(str(dest/f'pending_{(len(pending)-1)//8:03d}.jpg'),sheet,[cv2.IMWRITE_JPEG_QUALITY,97])
    result=dict(at=datetime.datetime.now().astimezone().isoformat(),status='FINAL_RISK_REVIEW_REUSED_EXACT_FRAMES_PENDING_REMAINDER',unit=str(u),total_selected=len(chosen),reused_count=len(reused),pending_count=len(pending),pages=math.ceil(len(pending)/8),reused=reused,pending=pending,pins=pins,ready_for_training=False,policy='Exact same uid/frame-index/side and pinned outputs/images only. Prior anatomical uncertainty remains; reuse is not independent validation. Pending panels require visual inspection. No input, model, thresholds or inference changes.')
    for p,h in pins.items():assert sha(p)==h
    with (u/'REVIEW_REUSE.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('status','total_selected','reused_count','pending_count','pages')}))
if __name__=='__main__':main(sys.argv[1])
