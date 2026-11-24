from common import *
import shutil,datetime
f=read(OLD/'FROZEN.json');m=read(O/'word100_temporal_public_bundle_20261025/MANIFEST.json')
new=[r for r in m if r['signer_id']==7]
assert len(new)==7
pins={str(OLD/'FROZEN.json'):sha(OLD/'FROZEN.json'),str(OLD/'INDEX.json'):sha(OLD/'INDEX.json')}
for r in f['rows']+new:
    for key,hkey in [('arrays','arrays_sha256'),('rgb_path','rgb_sha256')]:
        assert sha(r[key])==r[hkey];pins[r[key]]=r[hkey]
    p=r['source_assets']['detections'];pins[p]=sha(p)
for r in read(OLD/'INDEX.json'):
    for k in ('arrays','records'):assert sha(r[k])==r[k+'_sha256'];pins[r[k]]=r[k+'_sha256']
for p,h in f['protected'].items():assert sha(p)==h
for p in [U/'candidate.py',U/'common.py',O/'word100_fullframe_hand_merge_20261020/candidate.py',OLD/'yolo_hand_best.pt']:pins[str(p)]=sha(p)
(U/'before').mkdir(exist_ok=False)
for p in [ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copyfile(p,U/'before'/p.name)
write(U/'FROZEN.json',{'authorization':'2026-10-25 user approved five-scene uncertainty diagnosis and selective YOLO after MP redetection; public only; preserve C/T3-F and user recordings.',
 'at':datetime.datetime.now().astimezone().isoformat(),'old_rows':f['rows'],'new_rows':new,'pins':pins,'protected':f['protected'],
 'parameters':rule.PARAMS,'conditions':['M1 saved MP redetection','S strict supplemental full-hand agreement only after M1 miss','P same selective calls, wrist-only evidence separated from finger shape'],
 'trigger':'Full native M1 entire-side absence; current pose wrist visible inside fixed ROI and current elbow usable. Not ROI-only absence, not target label, not correctness label. Whole-frame YOLO once, crops for triggered sides only.',
 'new_wrist_rule':'unique full candidate for side; unique crop with same current arm; both confident wrist coordinates within unchanged0.08shoulder; original duplicate/identity guards. Wrist-only never authorizes finger angles.',
 'controls':'No original M1 points changed. Preserve 35 historic crossing frames and all normal present hands. Partial hands untouched; outside/uncertain wrists deferred.',
 'scoring':'Reuse frozen coarse AI wrist ranges; no new precision21GT. Same-old39 diagnostic replay not independent generalization. Seven signer07 videos unseen by selected YOLO benchmark provide separate extension, not new-person performance claim.',
 'latency':'Fresh selective YOLO calls plus prior paired M1 timings for old39: component estimate, not measured whole-systemFPS. New07 actual YOLO timing only; no invented M1 timing.',
 'stop':'No tuning after scores; arithmetic/source-preservation failure stops. Negative effectiveness reported.',
 'forbidden':['user or mentor data access','training','TCN inference','production edits','08/09 test reuse','Git','downloads','external transfer']})
print('FROZEN',len(f['rows']),len(new),sum(r['frames'] for r in new))
