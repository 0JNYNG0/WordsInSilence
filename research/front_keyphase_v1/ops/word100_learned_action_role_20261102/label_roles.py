"""Sparse visual process labels, not inferred from classifier confidence."""
from pathlib import Path
import json,hashlib
from datetime import datetime
A=Path(__file__).resolve().parent;O=A.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
# Each row lists explicitly reviewed travelling / presented-action / lowered-rest
# moments. Unlisted review moments remain unknown. These are physical process
# annotations by Codex, not an independent teacher's linguistic judgment.
TABLE={
'WORD0739':{
1:([43,64,65,99],[49,52,55,73,87,90],[]),
2:([64,67,103],[45,51,54,57,97,100],[109]),
3:([50,72,106],[56,59,62,84,100,103],[112]),
4:([108,134,140],[72,78,81,84,93,128,131],[]),
5:([38,73,98],[44,47,50,58,86,89],[]),
6:([74,111,117],[51,57,60,63,86,105,108],[]),
7:([35,56,57,83,86],[41,44,47,66,80],[92]),
8:([63,76,105],[45,51,54,57,93,96,99],[]),
9:([55,90],[39,45,48,51,69,84,87],[96]),
11:([71,85,109,115],[52,58,61,64,74,103],[]),
12:([43,61,81,105],[49,52,55,99,102],[111]),
13:([80,117,120],[59,65,68,71,78,96,114],[126])},
'WORD1133':{
1:([43,46,83,89],[49,60,77,80],[37]),
2:([39,42,79,82],[45,57,76],[33,88]),
3:([44,47,90],[50,60,78,81,84],[38]),
4:([68,71,105,108,114],[74,84,102],[62]),
5:([38,41,80],[44,55,68,71,74],[32]),
6:([36,39,66,72],[42,48,60,63],[30]),
7:([35,41,75,78],[44,47,57,72],[84]),
8:([65],[68,71,83,93,96,99,105],[59]),
9:([38,71,74],[44,47,50,56,68],[80,81]),
11:([46,78,81,87],[49,52,61,75],[40]),
12:([53,59,87,88,90,96],[62,65,75,84],[]),
13:([44,47,88,94],[50,64,82,85],[38])},
'WORD1567':{
1:([50,106],[56,59,62,64,65,73,94,97],[]),
2:([],[44,50,53,56,60,71,92,95,98],[]),
3:([36,93],[39,42,45,57,66,81,84],[30]),
4:([56,125],[62,65,68,71,81,84,95,113,116,119],[]),
5:([40,96,102],[46,49,52,57,58,68,77,90,93],[]),
6:([44,99],[47,50,53,55,63,72,87,90],[38]),
7:([32,79,85],[38,41,44,47,58,73,76],[]),
8:([55,118],[61,64,67,73,75,84,94,106,109,112],[]),
9:([111],[54,60,63,66,69,76,77,99,102],[]),
11:([43],[46,49,54,64,72,75,78,84],[37]),
12:([39,100],[45,48,51,55,63,77,88,91],[]),
13:([97],[34,37,40,44,54,71,85,88],[28])},
'WORD2523':{
1:([28,31,59,60],[34,41,54,57],[22,66]),
2:([43,84,90],[49,52,55,58,59,78,81],[]),
3:([49,55,87,93],[58,61,70,72,81,84],[]),
4:([71,114,120],[77,80,83,91,92,108,111],[]),
5:([51,83,86,92],[57,60,63,67,80],[]),
6:([53,87,90],[59,62,65,70,84],[96]),
7:([30,36,64,67],[39,42,47,48,61],[73]),
8:([40,46,78,84],[49,52,57,58,72,75],[]),
9:([55,87],[58,61,66,68,70,81,84],[49,93]),
11:([44,79,82],[50,53,56,60,76],[88]),
12:([43,82],[49,52,55,59,76,79],[88]),
13:([51,54,85,86],[57,65,80,83],[45,92])}}

def main():
    manifest=read(A/'REVIEW_MANIFEST.json');bg=read(O/'word100_learned_state_context_balance_20261102/BACKGROUND_TARGETS.json')
    rows=[];counts={};reviewed=0
    for r in manifest['rows']:
        assert sha(r['source'])==r['source_sha256']
        for p in r['samples']:assert sha(p['path'])==p['sha256'];reviewed+=1
        sets=TABLE[r['word_id']][r['person']];d={}
        for role,frames in enumerate(sets):
            for f in frames:assert f in r['frames'] and f not in d,(r['clip_id'],f);d[f]=role
        points=[dict(frame=f,label=d.get(f,-1),mask=f in d,basis='chronological_RGB_sheet') for f in r['frames']]
        b=next(x for x in bg['rows'] if x['clip_id']==r['clip_id'])
        for p in b['points']:
            if p['mask'] and p['frame'] not in r['frames']:points.append(dict(frame=p['frame'],label=2,mask=True,basis='T_reviewed_lowered_rest'))
        for p in points:counts[(r['role'],p['label'])]=counts.get((r['role'],p['label']),0)+1
        rows.append({k:r[k] for k in ['clip_id','word_id','person','role','source','source_sha256']}|dict(points=sorted(points,key=lambda x:x['frame'])))
    assert len(rows)==48 and reviewed==485
    for s in manifest['sheets']:assert sha(s['path'])==s['sha256']
    output=dict(at=datetime.now().astimezone().isoformat(),reviewer='Codex visual review; NOT independent human/KSL expert truth',
      classes=['travelling_between_reference_configurations_or_preparing_returning','presented_reference_action_or_configuration','visibly_lowered_rest'],
      semantics='Class0 is still movement evidence, not meaningless or incorrect. Class1 includes continuous repeated action/crossings and does not imply stillness, contact, full-word correctness or precise linguistic boundary. Only viewed points supervised; no dense interpolation and no labels from model predictions. Unknowns masked. Old physical-state anchors do NOT automatically become role labels.',
      limitations=['Sparse chronology cannot establish every boundary.','Source cards are partial context associations; no full-span semantic verification.','Same Codex label and arithmetic checks are not independent annotation.','RecordedID split, not biometric identity validation.'],
      review_manifest_sha256=sha(A/'REVIEW_MANIFEST.json'),old_background_sha256=sha(O/'word100_learned_state_context_balance_20261102/BACKGROUND_TARGETS.json'),
      counts={str(k):v for k,v in counts.items()},rows=rows)
    save(A/'ROLE_TARGETS.json',output)
    save(A/'REVIEW_VERIFICATION.json',dict(status='PASS_ARTIFACT_AND_VISUAL_REVIEW',public_clips=48,moments=485,sheets=16,source_frame_hashes_match=True,
      prior_session='52325 no longer available; terminal exit unavailable after context restoration. No re-extraction; complete log/manifest/artifacts checked separately.',targets_sha256=sha(A/'ROLE_TARGETS.json')))
    print(output['counts'])
if __name__=='__main__':main()
