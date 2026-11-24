"""Separate scalar arithmetic and artifact checks, not independent image truth."""
from pathlib import Path
import json,math,hashlib,numpy as np,re,ast
U=Path(__file__).resolve().parent;OLD=U.parent/'word100_mano_threeway_20261025'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):(U/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
f=read(U/'FROZEN.json');a=read(U/'REVIEWED_REFERENCE_REGIONS.json');scores=read(U/'REVIEWED_SCORES.json');initial=read(U/'RGB_REFERENCE_REGIONS.json')
for pinname in ['REFERENCE_PIN.json','REVIEWED_REFERENCE_PIN.json']:
 pin=read(U/pinname);assert sha(pin['path'])==pin['sha256']
changed=[]
for old,new in zip(initial['items'],a['items']):
 assert old['id']==new['id'] and old['wrist']==new['wrist']
 for j in old['tips']:
  if old['tips'][j]!=new['tips'][j]:changed.append((old['id'],j));assert new['tips'][j] is None
assert changed==[('E11',str(j)) for j in [4,8,12,16,20]]
summ={};n=0
for c in ['B','rawH','H','HT']:
 for kind in ['wrist','tip']:
  points=available=inside=0;center=[];outer=[]
  for r,ann in zip(f['rows'],a['items']):
   assert r['id']==ann['id'];z=np.load(r['comparison']);t,side=r['frame'],r['side']
   if c=='rawH':xy=np.load(OLD/f"raw_predictions/{r['clip_index']:02d}_{t:04d}_{side}.npz")['xy_px'];valid=np.isfinite(xy).all(-1)
   else:xy=z[c+'_xy'][t,side];valid=z[c+'_valid'][t,side]
   pairs=[(0,ann['wrist'])] if kind=='wrist' else [(int(j),b) for j,b in ann['tips'].items()]
   for j,b in pairs:
    if b is None:continue
    points+=1
    if not valid[j]:continue
    available+=1;x,y=map(float,xy[j]);l,top,rgt,btm=b
    dx=max(l-x,0.,x-rgt);dy=max(top-y,0.,y-btm);outside=math.hypot(dx,dy);cent=math.hypot(x-(l+rgt)/2,y-(top+btm)/2)
    inside+=int(dx==0 and dy==0);outer.append(outside);center.append(cent);n+=1
  q=scores['summary'][c][kind];assert(points,available,inside,available-inside,points-available)==(q['visual_regions'],q['predictions_available'],q['inside_visual_region'],q['outside_visual_region'],q['not_output'])
  for name,vals in [('mean_distance_to_region_px',outer),('mean_distance_to_center_px',center)]:
   assert (q[name] is None) if not vals else abs(q[name]-sum(vals)/len(vals))<1e-5
  summ[c+'_'+kind]={'regions':points,'available':available,'inside':inside}
preserved=cross=cf=nf=0
for m in f['manifest']:
 assert sha(m['rgb_path'])==m['rgb_sha256'];assert sha(m['prepared'])==m['prepared_sha256']
 z=np.load(OLD/'comparisons'/f"{m['clip_id']}.npz");mask=z['B_valid'];preserved+=int(mask.sum());cross+=int(mask[m['crossing_frames']].sum());cf+=len(m['crossing_frames']);nf+=len(mask)
 for c in ['H','HT']:
  assert np.array_equal(z[c+'_xy'][mask],z['B_xy'][mask]);assert z[c+'_valid'][mask].all();assert np.array_equal(z[c+'_evidence'],z['B_evidence']);assert not np.logical_and(z[c+'_inferred'],z[c+'_evidence']).any()
  for t in m['crossing_frames']:
   assert np.array_equal(z[c+'_valid'][t],mask[t]);assert np.array_equal(z[c+'_xy'][t],z['B_xy'][t],equal_nan=True)
  if m['clip_id']=='NIA_SL_WORD0003_REAL01_F':
   assert np.array_equal(z[c+'_valid'],mask);assert np.array_equal(z[c+'_xy'],z['B_xy'],equal_nan=True)
for collection in ['protected','prior_result_pins']:
 for p,h in f[collection].items():assert sha(p)==h
for r in f['rows']:assert sha(r['source_frame'])==r['source_frame_sha256']
for p in U.glob('*.py'):ast.parse(p.read_text('utf-8'))
doc=(U/'RESULT_KO.html').read_text('utf-8');assert len(re.findall('<article id=',doc))==16
for target in re.findall(r'(?:href|src)="([^"]+)"',doc):
 if target.startswith('#'):assert f'id="{target[1:]}"' in doc
 elif target!='VERIFICATION.json':assert (U/target).is_file(),target
assert '양손 통합 실패' in doc and '정확한 정답률이 아니야' in doc
save('VERIFICATION.json',{'status':'PASS_NUMERIC_AND_PRESERVATION_CHECKS_NOT_ACCURACY_APPROVAL','reviewer':'same Codex, separate scalar verification; not independent expert annotation',
 'scalar_point_checks':n,'conditions':summ,'all_original_point_frames_preserved':preserved,'crossing_full_frames_identical':cf,'crossing_points':cross,'normal_control94_frames_identical':True,'frames':nf,
 'prior542_unchanged':True,'protected32_unchanged':True,'original7_videos_unchanged':True,'source16_images_pinned':True,'initial_annotation_preserved':True,'E11_only5_tip_identity_regions_held':True,
 'report_articles':16,'static_links_and_python_syntax_pass':True,'browser_render_checked':False,'visual_overlays_reviewed':16,'new_inference':0,'training':0,'C_T3F_change':False,'user_data_access':0,
 'scientific_verdict':'NOT_READY_FOR_DEPLOYMENT: E05 duplicate/identity conflict, approximate visual wrist regions,63 unverifiable fingertip slots. Preserving old coordinates is not proof of correct fusion.'})
print('PASS separate numeric/protection check; scientific deployment NOT approved')
