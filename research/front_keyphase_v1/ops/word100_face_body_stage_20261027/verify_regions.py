"""Controlled geometric checks, not human sign correctness tests."""
import sys,json,copy
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4];LAB=ROOT/'research/front_keyphase_v1';U=Path(__file__).resolve().parent
sys.path.insert(0,str(LAB/'src'))
from wis_lab.core.observation import Observation
from wis_lab.capture.face_landmarks import join_face_observation
from wis_lab.core.face_body_relations import relations,region_geometry,REGIONS
F=json.loads((U/'FROZEN.json').read_text(encoding='utf-8'));results=[]
for r in F['rows']:
 p=U/'data'/r['clip_id'];obs=Observation.load(str(p/'masked.npz'));face=join_face_observation(p/'face_roi.npz',obs)
 # First frame, all hands removed to avoid treating an artificial test point as a hand hull.
 centers,axes,radii,valid=region_geometry(face['xy_px'][0],obs.xy_px[0],obs.usable[0]);assert valid.all()
 for region in (0,1,2,3,4,7):
  for offset in (0,.1):
   q=copy.deepcopy(obs);q.xy_px[:,33:]=np.nan;q.observed[:,33:]=False;q.usable[:,33:]=False
   q.xy_px[0,62]=centers[region]+axes[0]*radii[region,0]*offset
   q.observed[0,62]=q.usable[0,62]=True
   rel=relations(q,face);label=int(rel['labels'][0,1,1])
   assert label==region,(r['clip_id'],REGIONS[region],offset,label)
   results.append(dict(clip_id=r['clip_id'],region=REGIONS[region],offset_radius=offset,label=int(label)))
report=dict(status='PASS',controlled_points=len(results),videos=len(F['rows']),
 meaning='Region centres and small within-region shift remain distinct under current geometry; not linguistic tolerance validation',
 tests=results)
(U/'REGION_CHECKS.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('REGION_CHECKS',len(results))
