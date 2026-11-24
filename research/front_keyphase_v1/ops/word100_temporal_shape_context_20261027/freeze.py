"""Fixed limited temporal-shape evidence experiment, before dense RGB viewing."""
import json,hashlib,shutil
from pathlib import Path
from datetime import datetime,timezone
U=Path(__file__).resolve().parent;ROOT=U.parents[3];P=U.parent/'word100_shared_feature_bundle_20261027'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not (U/'FROZEN.json').exists()
 idx={r['alias']:r for r in read(P/'visual_inputs/INDEX.json')['rows']}
 groups=[('ambiguous_index','R',8,['V04','V05','V06']),('side_view','R',5,['X10','X11','X12']),
         ('low_rest','L',8,['V01','V02','V03']),('normal_open_crossing','R',8,['V07','V08','V09']),
         ('sampled_change_star','R',6,['V10','V11','V12']),('sampled_change_obey','R',6,['X13','X14','X15'])]
 rows=[]
 for group,side,ordinal,aliases in groups:
  for a in aliases:
   r=idx[a];f=next(f for f in r['frames'] if f['ordinal']==ordinal)
   rows.append({'case_id':f'T{len(rows)+1:02d}','source_alias':a,'group':group,'side':side,'source':r['source'],
                'source_sha256':r['source_sha256'],'clip_id':r['clip_id'],'word_id':r['word_id'],'label':r['label'],
                'recorded_person_id':r['recorded_person_id'],'inherited_role':r['role'],'center_frame_id':f['frame_id'],
                'center_pts_us':f['pts_us'],'source_sample_ordinal':ordinal,'display_only_crop_xyxy':r['display_only_crop_xyxy']})
 frozen={'at':datetime.now(timezone.utc).isoformat(),'user_scope':'Approved forward/backward temporal-shape comparison on public videos; conditional later feature training after evidence validation.',
         'selection':'18 windows selected from previous frozen observations: 6 active-hand ambiguities,3 low-rest controls,3 normal open/crossing controls,6 change candidates. Not random population sample.',
         'rows':rows,'dense_offsets_frames':list(range(-6,7)),'max_gap_us':200000,'min_support_frames_each_direction':2,
         'features':['index_only','open_four'],'rubric_source':str(P/'RUBRIC.json'),
         'conditions':{'A':'Current direct RGB image evidence draft only','B':'A + causal past support, bounded 200ms; uncertain current remains an estimate',
                       'C':'A + agreeing past and future support, both within200ms, contradictory evidence blocks imputation'},
         'algorithm_freeze':'If target direct label exists retain it. Otherwise past must have >=2 known states and all known states agree inside window; B uses that state. C additionally requires >=2 future known states, all agree and equal past. Identity/gap/visibility concerns remain separately flagged. No word-specific logic. Unknown is never0.',
         'evaluation':['Real unknown centers: report inferred/remaining unknown without correctness claim.',
                       'Mask every known interior state separately; keep its pre-mask draft label for agreement only, not human accuracy. Evaluate causal and bilateral fill/errors/rejections.',
                       'Keep each independently visible state and observed change, normal crossing, and low-rest evidence separate.'],
         'readiness':'This is evidence-layer comparison from image-reviewed states, not an autonomous RGB recognizer. Same-AI drafts insufficient for verified supervised targets; do not train a model only to imitate these guessed labels.',
         'preserve':['C','T3-F','operating web/DTW','previous frozen outputs','user/mentor material'],
         'forbidden':['user/mentor access','new detector','downloads','Git','target word input','long-gap forced filling','claiming hidden true shape'],
         'pins':{str(P/'FROZEN.json'):sha(P/'FROZEN.json'),str(P/'DRAFT_LABELS.json'):sha(P/'DRAFT_LABELS.json'),str(P/'RUBRIC.json'):sha(P/'RUBRIC.json'),str(P/'visual_inputs/INDEX.json'):sha(P/'visual_inputs/INDEX.json')}}
 (U/'FROZEN.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 (U/'before').mkdir(exist_ok=True)
 for name,p in [('WordsInSilence_PROJECT_CONTEXT.md',ROOT/'WordsInSilence_PROJECT_CONTEXT.md'),('NEW_CHAT_HANDOFF.md',ROOT/'NEW_CHAT_HANDOFF.md'),('STATE.json',ROOT/'research/front_keyphase_v1/ops/STATE.json')]:shutil.copy2(p,U/'before'/name)
 print('Frozen18 windows,13 consecutive frames each,2 hand-shape fields; A/B/C fixed before new dense images.')
if __name__=='__main__':main()
