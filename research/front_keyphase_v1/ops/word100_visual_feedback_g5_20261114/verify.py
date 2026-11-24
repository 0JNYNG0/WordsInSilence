"""Separate evidence/media/HTTP audit. Does not call models or a camera."""
from pathlib import Path
import json,hashlib,urllib.request,urllib.error,ast,re
import cv2,numpy as np
U=Path(__file__).resolve().parent;O=U.parent;checks=[]
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(name,v):
    checks.append(dict(name=name,pass_=bool(v)))
    if not v:raise AssertionError(name)
def request(path,headers=None):
    try:
        with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8821'+path,headers=headers or {}),timeout=20) as r:return r.status,r.headers,r.read()
    except urllib.error.HTTPError as e:return e.code,e.headers,e.read()

raw=read(O/'word100_public_g4_20261114/CONFLICT_REGRESSION.json');rows={(r['clip_id'],r['kind']):r for group in ('public_normal','public_continuous','public_splice')for r in raw['rows'][group]if r['condition']=='current_g3'}
guides=read(O/'word100_action_episode_supervision_20261110/GUIDE_FAMILIES.json')['words'];catalog=read(U/'site/catalog.json');routes=read(U/'ROUTES.json')
state_counts={};diffcases=0;conflict=0;cuecount=0
for entry in catalog['cases']:
    d=read(routes[entry['url']]);rid=d['id'];original=rows[d['clip_id'],d['kind']];n=d['frame_count'];source={c['card']:c for c in original['joint_cards']}
    check(rid+' lengths',len(d['xy'])==len(d['timestamp_us'])==len(d['source_mapping'])==n)
    check(rid+' timestamps',all(b>a for a,b in zip(d['timestamp_us'],d['timestamp_us'][1:])))
    check(rid+' public scope',d['source']=='public_saved_G4' and d['training_use'] is False and d['whole_sign_correct'] is None)
    check(rid+' alignment labelled honestly',d['alignment']=='side_by_side_native_time_not_DTW')
    if d['kind']=='continuous_full_reverse':check(rid+' reverse map',d['source_mapping']==list(range(n-1,-1,-1)))
    if d['kind']=='all_hands_missing':check(rid+' missing joints not drawn',all(all(p is None for p in xy[91:])for xy in d['xy']))
    for c in d['cards']:
        s=source[c['id']];tag=rid+':'+c['id'];state_counts[c['status']]=state_counts.get(c['status'],0)+1
        check(tag+' source unmodified',c['source_card']==s)
        check(tag+' no full pass',c['whole_sign_correct'] is None and c['completion'] is None)
        check(tag+' guide exact',c['instruction']==next(g['instruction']for g in guides[d['word']]if g['id']==c['id']))
        check(tag+' uncertain cannot produce correction',c['status']!='uncertain' or not c['cues'])
        if s['state']=='evidence_conflict':
            conflict+=1;check(tag+' conflict not orange or good',c['status']=='uncertain' and not c['cues'] and '엇갈' in c['message'])
        for cue in c['cues']:
            cuecount+=1;part,index=cue['evidence'].split('/');e=s[part][int(index)]
            check(tag+' cue provenance '+cue['evidence'],c['status']=='difference' and s['state']=='attributable_feature_difference')
            if cue['kind'] in ('direction','shape'):check(tag+' exact difference interval',cue['spans']==[e['span']])
            else:
                actual=sorted(m['span']for group in e['witness_spans']for m in group);check(tag+' exact order witnesses',cue['spans']==actual and e['state']=='reverse');check(tag+' both reference actions',cue['card_ids']==e['cards'])
            check(tag+' bounds',all(0<=a<=b<n for a,b in cue['spans']))
            check(tag+' hands',set(cue['hands'])<=set(('L','R')) and bool(cue['hands']))
        if c['status']=='similar':check(tag+' positive qualified',s['state'] in ('measured_features_agree','measured_features_and_goal_agree'))
    if entry['difference']:diffcases+=1
    status,headers,body=request(entry['url']);check(rid+' HTTP data',status==200 and json.loads(body)==d)

media_checks=[]
for url,path in routes.items():
    if not url.endswith('_reverse.mp4'):continue
    cid=Path(path).name.removesuffix('_reverse.mp4');src=routes[f'/feedback-g5/media/{cid}.mp4']
    a=cv2.VideoCapture(src);b=cv2.VideoCapture(path);n=int(a.get(cv2.CAP_PROP_FRAME_COUNT));check(cid+' media frames',n==int(b.get(cv2.CAP_PROP_FRAME_COUNT)))
    errors=[]
    for i in (0,n//2,n-1):
        a.set(cv2.CAP_PROP_POS_FRAMES,n-1-i);ok1,x=a.read();b.set(cv2.CAP_PROP_POS_FRAMES,i);ok2,y=b.read();check(cid+' frame decodes '+str(i),ok1 and ok2)
        x=cv2.resize(x,(y.shape[1],y.shape[0]));err=float(np.abs(x.astype(float)-y.astype(float)).mean());errors.append(err);check(cid+' reverse pixel correspondence '+str(i),err<6)
    a.release();b.release();media_checks.append(dict(clip=cid,frames=n,pixel_mae=errors))

for p,h in read(U/'FROZEN_INPUTS.json').items():check('input pinned '+Path(p).name,sha(p)==h)
protected=read(O/'word100_web_completion_graph_20261111/PROTECTIONS.json')
for p,h in protected.items():check('protected '+p,sha(p)==h)
for r in read(U/'CONNECTION.json')['files']:
    check('rollback backup '+r['path'],sha(r['backup'])==r['before_sha256']);check('after hash '+r['path'],sha(r['path'])==r['after_sha256'])
for p in U.glob('*.py'):ast.parse(p.read_text('utf-8-sig'))
status,_,body=request('/');check('existing homepage integration',status==200 and b'g5-review-link' in body)
status,_,body=request('/feedback-g5/');check('viewer served',status==200 and '어느 부분'.encode() in body)
for path in ('/feedback-g5/../../NEW_CHAT_HANDOFF.md','/feedback-g5/data/unknown.json','/feedback-g5/media/../../STATE.json'):
    check('no arbitrary file '+path,request(path)[0]==404)
video=next(k for k in routes if k.endswith('_reverse.mp4'));status,h,b=request(video,{'Range':'bytes=0-127'});check('video range seek',status==206 and len(b)==128 and h.get('Accept-Ranges')=='bytes')
status,_,body=request('/api/meta');meta=json.loads(body);status,_,body=request('/api/status',{'X-Session-Token':meta['token']});state=json.loads(body)
check('camera remains off',not state['camera_thread_active']);check('capture not run',state['state']=='ready')
result=dict(status='PASS',checks=len(checks),cases=len(catalog['cases']),cards=state_counts,cues=cuecount,difference_cases=diffcases,conflict_cards=conflict,protected_files=len(protected),media_correspondence=media_checks,camera_opened=False,training=False,model_accuracy_claim=False,verifier='Codex separate source/media/API audit; same author, not independent expert',details=checks)
(U/'VERIFIED.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items()if k not in ('details','media_correspondence')},ensure_ascii=False))
