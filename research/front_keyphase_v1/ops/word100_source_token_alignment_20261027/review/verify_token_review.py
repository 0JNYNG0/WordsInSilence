"""Separate-agent read-only frame, annotation and preservation verification."""
from pathlib import Path
from collections import Counter
import hashlib,json
import av
import numpy as np
from PIL import Image
U=Path(__file__).resolve().parents[1];R=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def require(c,s):
    if not c:raise AssertionError(s)
def main():
    index=read(U/'visual_inputs/INDEX.json');ann=read(U/'TOKEN_OBSERVATIONS.json');result=read(U/'RESULT.json');ready=read(U/'ANNOTATION_READINESS.json');links=read(U/'SOURCE_FORM_LINKS.json');contract=read(U/'LEARNING_CONTRACT.json');rows={r['alias']:r for r in index['rows']}
    expected={f'V{3*g+s:02d}':(w,s) for g,w in enumerate(['WORD0949','WORD1139','WORD1290','WORD2551']) for s in (1,2,3)}
    require(set(rows)==set(expected),'aliases');frame_lookup={}; nframes=0;maxpts=0;unique_source=set()
    old=U.parent/'word100_face_body_stage_20261027'
    for alias,row in rows.items():
        require((row['word_id'],row['recorded_person_id'])==expected[alias],'alias mapping')
        require(sha(row['source'])==row['source_sha256'],'video content hash');unique_source.add(row['source_sha256'])
        targets={f['frame_id']:f for f in row['frames']};require(len(targets)==16,'sixteen distinct samples')
        with np.load(old/'data'/row['clip_id']/'masked.npz',allow_pickle=False) as z:obs_ids=z['frame_id'].copy();obs_ts=z['timestamp_us'].copy()
        seen=set();first=None
        with av.open(row['source']) as video:
            for fid,frame in enumerate(video.decode(video.streams.video[0])):
                if first is None:first=frame.pts
                if fid not in targets:continue
                rec=targets[fid];pts=round(float((frame.pts-first)*frame.time_base)*1e6);require(obs_ids[fid]==fid,'saved frame id')
                d=max(abs(pts-rec['pts_us']),abs(pts-int(obs_ts[fid])));maxpts=max(maxpts,d);require(d<=1,'source PTS')
                rgb=frame.to_ndarray(format='rgb24');require(hashlib.sha256(rgb.tobytes()).hexdigest()==rec['rgb_sha256'],'source RGB hash')
                with Image.open(rec['file']) as im:require(np.array_equal(np.asarray(im.convert('RGB')),rgb),'PNG RGB equals source decode')
                frame_lookup[(alias,rec['ordinal'])]=rec;seen.add(fid);nframes+=1
        require(seen==set(targets),'all requested frames found')
    require(len(unique_source)==12,'unique source content')
    used=set();kind_counts=Counter();feature_counts=Counter();movement_scopes=[]
    for a in ann['annotations']:
        row=rows[a['alias']];require(a['clip_id']==row['clip_id'] and a['word_id']==row['word_id'] and a['recorded_person_id']==row['recorded_person_id'],'annotation identity')
        require(a['dense_interval_truth'] is False and 'not linguistic phase order' in a['chronology_note'],'sparse groups not phases')
        require(a['training_eligible'] is False and all(x is None for x in a['unknown'].values()),'unknowns retained')
        times=[]
        for e in a['evidence']:
            rec=frame_lookup[(a['alias'],e['ordinal'])]
            require(all(e[k]==rec[k] for k in ['frame_id','pts_us','rgb_sha256']),'evidence matching')
            require((U/e['file']).resolve()==Path(rec['file']).resolve(),'file matching');used.add((a['alias'],e['frame_id']));times.append(e['pts_us']/1e6)
        require(a['sampled_time_extent_seconds']==[min(times),max(times)],'extent only first last listed')
        kind_counts[a['kind']]+=1
        for f in a['observed_features']:
            require(f['must_be_correct_for_sign'] is None and f['truth_status']=='IMAGE_SUPPORTED_DRAFT','no requiredness promotion');feature_counts[f['feature']]+=1
            if f['feature'] in ['movement','inter_hand_motion']:movement_scopes.append({'id':a['annotation_id'],'value':f['value'],'temporal_scope':f.get('temporal_scope') or f.get('feature_scope') or a.get('feature_scope'),'fields':list(f)})
    actual={'clips':len(rows),'expressions':len({r['word_id'] for r in rows.values()}),'recorded_people':len({r['recorded_person_id'] for r in rows.values()}),'rgb_samples':nframes,'annotation_groups':len(ann['annotations']),'annotated_unique_frames':len(used),'coarse_links':sum(x['state']!='FORM_LINK_UNRESOLVED' for x in links['links'])}
    for k,v in actual.items():require(result['counts'][k]==v,('count',k))
    require(result['observation_feature_counts']==dict(feature_counts),'feature totals')
    require(result['counts']['training_runs']==ready['training_runs']==result['counts']['full_variant_verified']==result['counts']['correctness_labels']==0,'no training/meaning truth')
    require(not contract['current_training_eligible'],'learning contract')
    require(ann['source_index_sha256']==sha(U/'visual_inputs/INDEX.json') and ann['review_observation_sha256']==sha(R/'OBSERVATIONS_STAGE1.json'),'lineage')
    require(links['source_profiles_sha256']==sha(U/'SOURCE_PROFILES.json'),'source lineage')
    source_pins=read(U/'visual_inputs/SOURCE_PINS_BEFORE.json');require(source_pins==read(U/'visual_inputs/SOURCE_PINS_AFTER.json'),'source pin capture equality')
    for p,h in source_pins.items():require(sha(p)==h,('source pin',p))
    output_pins=read(U/'visual_inputs/OUTPUT_PINS.json')
    for p,h in output_pins.items():require(sha(p)==h,('output pin',p))
    protected=read(U.parent/'word100_face_relation_alignment_20261027/FROZEN.json')['protected']
    for p,h in protected.items():require(sha(p)==h,('C/T3F/operating pin',p))
    out={'status':'PASS_SEPARATE_FRAME_AND_ANNOTATION_ARITHMETIC_REVIEW','reviewer':'Codex /root/linguistic_eval_review','counts':actual,'kind_counts':dict(kind_counts),'feature_counts':dict(feature_counts),'independent_source_decode_rgb_matches':nframes,'maximum_pts_difference_us':int(maxpts),'source_pins_unchanged':len(source_pins),'output_pins_unchanged':len(output_pins),'C_T3F_operating_pins_unchanged':len(protected),'movement_scope_snapshot':movement_scopes,'artifact_hashes':{n:sha(U/n) for n in ['TOKEN_OBSERVATIONS.json','SOURCE_FORM_LINKS.json','ANNOTATION_READINESS.json','LEARNING_CONTRACT.json','RESULT.json']},'limits':['Arithmetic is not linguistic truth','Sparse frame observations are same-model coarse drafts','Movement between samples must not become per-frame truth','No model training or operational changes']}
    (R/'TOKEN_ARITHMETIC_REVIEW.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:out[k] for k in ['status','counts','independent_source_decode_rgb_matches','maximum_pts_difference_us','source_pins_unchanged','output_pins_unchanged','C_T3F_operating_pins_unchanged']},ensure_ascii=False))
if __name__=='__main__':main()
