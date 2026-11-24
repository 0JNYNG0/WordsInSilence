"""Render saved public evidence for explanation only; no model/data scoring."""
import base64,hashlib,io,json
from pathlib import Path
from PIL import Image

U=Path(__file__).resolve().parent;O=U.parent
R=O/'word100_observation_lineage_repair_20261101'
V=Path('C:/Users/local-user/.codex/visualizations/2026/09/18/01a0b4a2-0a79-7b80-b823-d6469869041f')
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=read(R/'review/RECOVERY_VISUAL_REVIEW.json')
rgb=read(O/'word100_action_feedback_casebook_20261101/review/RGB_MANIFEST.json')
index={(x['alias'],x['frame_id']):x for x in rgb['full_frames']}
reviews={(x['alias'],x['frame_id']):x for x in source['frames']}
pins={str(R/'review/RECOVERY_VISUAL_REVIEW.json'):sha(R/'review/RECOVERY_VISUAL_REVIEW.json'),
      str(R/'review/RESIDUAL_PLAN_REVIEW.json'):sha(R/'review/RESIDUAL_PLAN_REVIEW.json')}
crop=[620,100,1320,820]
assets={};audit=[]
def asset(alias,fid):
    key=f'{alias}_{fid}'
    if key in assets:return key
    rec=reviews.get((alias,fid),{}).get('original_rgb') or index[(alias,fid)]
    assert sha(rec['image'])==rec['image_sha256']
    im=Image.open(rec['image']).convert('RGB');assert im.size==(1920,1080)
    # Display-only native-pixel crop. No resampling, inference or coordinate change.
    b=io.BytesIO();im.crop(crop).save(b,format='JPEG',quality=88)
    assets[key]='data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode('ascii')
    pins[rec['image']]=rec['image_sha256']
    audit.append({'alias':alias,'frame_id':fid,'timestamp_us':rec['timestamp_us'],
                  'source_image':rec['image'],'source_sha256':rec['image_sha256'],
                  'display_crop_only':crop,'encoded_bytes':len(b.getvalue())})
    return key
def ledger(alias,fid):
    p=R/'recovery_records'/f'{alias}.json';pins[str(p)]=sha(p)
    return next(x for x in read(p)['frames'] if x['frame_id']==fid)
def points(fr):
    return {group:[{'xy':[[round(x-crop[0],3),round(y-crop[1],3)] for x,y in hand['xy']]} for hand in fr[group]] for group in ('full','crop')}
def box(fr):
    boxes=list(fr['boxes'].values());assert boxes
    x0=max(crop[0],min(x[0] for x in boxes)-20);y0=max(crop[1],min(x[1] for x in boxes)-20)
    x1=min(crop[2],max(x[2] for x in boxes)+20);y1=min(crop[3],max(x[3] for x in boxes)+20)
    return [x0-crop[0],y0-crop[1],x1-x0,y1-y0]
cases=[]
specs=[('P048',74,'잘하다 · 2.467초','확대했을 때만 왼손 후보','전체 화면: 오른손만 · 손 주변 확대: 양손 후보','왼손 추가 보류'),
       ('P063',72,'조상 · 2.400초','확대했을 때만 왼손 후보','전체 화면: 오른손만 · 손 주변 확대: 양손 후보','왼손 추가 보류'),
       ('P063',57,'조상 · 1.900초','겹친 손의 관절 후보가 다름','전체 화면과 확대 결과의 손가락 위치가 서로 불일치','왼손 추가 보류'),
       ('P066',71,'조상 · 2.367초','어느 팔의 손인지 불확실','새 손 후보를 현재 팔에 하나로 연결하지 못함','오른손 추가 보류')]
for a,f,title,cause,detail,result in specs:
    fr=ledger(a,f);assert fr['additions']==[]
    cases.append({'id':f'{a}_{f}','title':title,'cause':cause,'detail':detail,'result':result,
                  'asset':asset(a,f),'zoom':box(fr),'candidates':points(fr),'frame':f})
fr=ledger('P072',70);part=points(fr);supported={}
for add in fr['additions']:
    decision=next(d for d in fr['decisions'] if d['index']==add['full_index'])
    supported[str(add['full_index'])]=decision['joint_supported']
assert sum(sum(v) for v in supported.values())==36
precision={'asset':asset('P072',70),'title':'이집트 · 2.333초 · 70프레임','zoom':box(fr),
           'candidates':part,'supported':supported}
alignment=[{'asset':asset('P066',68),'title':'공개 영상 A · 2.267초 · 68프레임','relation':'위: 오른손 / 아래: 왼손'},
           {'asset':asset('P063',58),'title':'공개 참고 B · 1.933초 · 58프레임','relation':'위: 왼손 / 아래: 오른손'}]
data={'assets':assets,'cases':cases,'alignment':alignment,'precision':precision}
template=(U/'view.template.html').read_text('utf-8')
assert template.count('/*SAVED_EVIDENCE_JSON*/null')==1
fragment=template.replace('/*SAVED_EVIDENCE_JSON*/null',json.dumps(data,ensure_ascii=False,separators=(',',':')))
out=V/'remaining-sign-evidence.html';out.write_text(fragment,encoding='utf-8')
assert out.stat().st_size<1000000
for p,h in pins.items():assert sha(p)==h,p
(U/'VIEW_MANIFEST.json').write_text(json.dumps({'status':'BUILT_PENDING_BROWSER_CHECK','fragment':str(out),'sha256':sha(out),
 'bytes':out.stat().st_size,'pins':pins,'images':audit,'frames_displayed':len(audit),'residual_hand_cases':4,
 'source_results_modified':False,'new_inference':0,'new_DTW':0,'new_training':0,'user_mentor_data_used':False,
 'display_only':'Native-pixel crop of saved public JPEG; matching coordinates shifted only for display. No replacement of analysis ROI.',
 'limitations':'All keypoints are model estimates; accepted candidates are not independent landmark truth. Paired public frames show an observed relation conflict, not learner error.'},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'path':str(out),'bytes':out.stat().st_size,'source_images':len(audit)},ensure_ascii=False))
