"""Audit direct RGB occurrence drafts against inherited role labels; no training."""
from pathlib import Path
import json,hashlib,datetime
from collections import Counter,defaultdict
import numpy as np
U=Path(__file__).resolve().parent;G=U.parent;O=G.parent;ROOT=O.parents[2]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def absolute(p):
    p=Path(p);return p if p.is_absolute() else ROOT/p
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

CLAUSES={'open_front':1,'close_chest':2,'raise_forearm':3,'gather_hands':6,'spread_hands':7}
def role_family(role,clause,phase=''):
    if phase.startswith('rest_'):return 0
    if role=='preparation':return 8
    if role in ['return','finishing']:return 9
    return CLAUSES.get(clause)
def phase_family(phase):
    if phase.startswith('rest_'):return 'rest'
    if phase in ['presented_hold','terminal_posture','terminal_hold']:return 'hold'
    if phase=='closing_and_lowering':return 'start'
    if phase in ['entry','closed_hand_lowering','forearm_reorientation','lowering_return','relaxing_return','lowering_and_relaxing_return']:return 'progress'
    return 'unknown'

def main():
    target=U/'DIRECT_ROLE_AUDIT.json';assert not target.exists()
    bp=G/'reviews/obedience_occurrence_labels.json';ep=G/'reviews/egypt_occurrence_labels.json';rp=O/'word100_candidate_role_learning_20261111/ROLE_LABELS.json'
    b=read(bp);e=read(ep);roles=read(rp);roleby={r['clip_id']:r for r in roles['rows']}
    sourcepins=[dict(path=str(p),sha256=sha(p)) for p in [bp,ep,rp]]
    data={};imagepins={}
    def imagepin(path,expected):
        p=absolute(path);actual=sha(p);assert actual==expected,(str(p),actual,expected)
        imagepins[str(p)]=actual
        return dict(path=str(p),sha256=actual)
    for c in b['clips']:
        cid=c['clip_id'];assert c['split']=='FIT' and c['person'] in [1,4,7]
        frameimgs=defaultdict(list)
        for panel in c['review_images']:
            pin=imagepin(panel['path'],panel['sha256'])
            for f in panel['frames']:frameimgs[f].append(dict(pin,source_frame_id=f,frame_in_contact_sheet=True))
        boundaries=[]
        for oc in c['occurrences']:
            for k,v in oc.items():
                if k.endswith('_boundary_uncertainty_frames'):
                    boundaries.append(dict(occurrence_id=oc['occurrence_id'],field=k,range=v,kind='internal_phase' if k.startswith('movement_end') else 'occurrence'))
        candidates=defaultdict(list)
        for oc in c['occurrences']:
            for obs in oc['direct_observations']:
                for f in obs['frames']:
                    assert f in c['reviewed_frame_ids'] and f in frameimgs
                    candidates[f].append(dict(occurrence_id=oc['occurrence_id'],clause_id=oc['guide_clause'],role=oc['role'],hands=[oc['hand']],
                        family=role_family(oc['role'],oc['guide_clause'],obs['phase']),intra_action_phase=phase_family(obs['phase']),source_phase=obs['phase'],
                        shape_observation_separate=obs.get('shape_evidence'),description=obs['observation_ko'],
                        explicitly_eligible_in_source=f in obs['role_supervision_frames_outside_boundary_bands']))
        data[cid]=dict(person=c['person'],source=c['source'],source_sha256=c['source_sha256'],reviewed=set(c['reviewed_frame_ids']),
            image_by_frame=frameimgs,boundaries=boundaries,candidates=candidates,source_type='obedience',source_label_file=str(bp))
        boundaryset={f for z in boundaries for f in range(z['range'][0],z['range'][1]+1)}
        assert boundaryset==set(c['role_boundary_exclusion_frames'])
    egypt_frames=[]
    for manifest in e['visual_sources'][:2]:
        assert sha(manifest['path'])==manifest['sha256'];sourcepins.append(manifest)
        egypt_frames.extend(read(manifest['path'])['rows'])
    for c in e['rows']:
        cid=c['clip_id'];assert c['role']=='fit' and c['record_id'] in [1,4,7]
        frameimgs=defaultdict(list)
        for f in egypt_frames:
            if f['clip_id']!=cid:continue
            assert f['source_sha256']==c['source_sha256']
            frameimgs[f['frame_id']].append(dict(imagepin(f['image'],f['image_sha256']),source_frame_id=f['frame_id'],frame_in_contact_sheet=False))
        boundaries=[];candidates=defaultdict(list)
        for oc in c['episodes']:
            for k in ['onset_frame_range','offset_frame_range','shape_change_end_frame_range','separation_settling_end_frame_range']:
                if k in oc and oc[k] is not None:
                    boundaries.append(dict(occurrence_id=oc['occurrence_id'],field=k,range=oc[k],kind='occurrence' if k in ['onset_frame_range','offset_frame_range'] else 'internal_phase'))
            refs={f['frame_id']:f for f in oc['evidence_frames']}
            for f in oc['confirmed_core_frames']:
                assert f in frameimgs and f in refs
                phase='progress'
                if oc['clause_id']=='gather_hands':phase='hold'
                elif oc['clause_id']=='spread_hands':
                    settle=oc.get('separation_settling_end_frame_range')
                    if settle and f>settle[1]:phase='hold'
                    elif settle and settle[0]<=f<=settle[1]:phase='unknown_progress_or_hold'
                    else:phase='progress'
                candidates[f].append(dict(occurrence_id=oc['occurrence_id'],clause_id=oc['clause_id'],role=oc['role'],hands=oc['hands'],
                    family=role_family(oc['role'],oc['clause_id']),intra_action_phase=phase,source_phase='RGB_direct_core_with_phase_brackets',
                    shape_observation_separate=oc['visible_features'],description=oc['requirement_status'],
                    explicitly_eligible_in_source=None))
            for f in oc.get('rest_confirmation_frames',[]):
                assert f['frame_id'] in frameimgs
                candidates[f['frame_id']].append(dict(occurrence_id=oc['occurrence_id'],clause_id=None,role='background',hands=oc['hands'],family=0,
                    intra_action_phase='rest',source_phase='rest_confirmation',shape_observation_separate=None,description='Rest after arm lowering directly reviewed; exact return end remains bracketed.',explicitly_eligible_in_source=None))
        data[cid]=dict(person=c['record_id'],source=c['source'],source_sha256=c['source_sha256'],reviewed=set(frameimgs),
            image_by_frame=frameimgs,boundaries=boundaries,candidates=candidates,source_type='egypt',source_label_file=str(ep))
    assert len(data)==6
    ready={r['clip_id']:r for r in read(O/'word100_shared_motion_learning_20261109/READY.json')['rows'] if r['clip_id'] in data}
    assert len(ready)==6 and all(r['role']=='fit' and r['person'] in [1,4,7] for r in ready.values())
    resultrows=[];clip_summaries=[]
    for cid,c in data.items():
        old=roleby[cid];meta=ready[cid]
        assert old['source_sha256']==c['source_sha256']==meta['source_sha256']
        assert sha(c['source'])==c['source_sha256']
        assert sha(meta['path'])==meta['sha256']
        # Only frame IDs are read, not features, masks, model outputs or DEV arrays.
        with np.load(meta['path'],allow_pickle=False) as z:ids=z['frame_id'].copy()
        assert len(ids)==len(old['y'])==len(old['w']) and len(set(ids.tolist()))==len(ids)
        imap={int(f):i for i,f in enumerate(ids)}
        start=len(resultrows)
        for f in sorted(c['reviewed']):
            assert f in imap;i=imap[f];oy=int(old['y'][i]);ow=float(old['w'][i]);opts=c['candidates'].get(f,[])
            hit=[q for q in c['boundaries'] if q['range'][0]<=f<=q['range'][1]]
            proposed={q['family'] for q in opts if q['family'] is not None}
            new=next(iter(proposed)) if len(proposed)==1 else None
            unsure=[]
            if not opts:unsure.append('viewed_RGB_without_explicit_occurrence_or_rest_label')
            if hit:unsure.append('inside_inclusive_boundary_uncertainty_band')
            if len(proposed)>1:unsure.append('multiple_occurrence_role_interpretations')
            if opts and new is None:unsure.append('family_mapping_unknown')
            if any(q['explicitly_eligible_in_source'] is False for q in opts) and not hit:unsure.append('source_annotation_defers_role')
            if unsure:
                category='uncertain';subtype='boundary' if hit else 'unassigned_or_conflicting_RGB'
            elif oy<0 or ow<=0:
                category='new_addition';subtype='previously_unlabeled_frame'
            elif oy==new:
                category='duplicate';subtype='existing_direct_w1' if ow>=.999 else 'existing_weak_confirmed_by_direct_draft'
            else:
                category='conflict';subtype='existing_direct_w1_conflict' if ow>=.999 else 'existing_weak_conflict'
            cause=None
            if category=='conflict':
                if new in [8,9] and oy==0:cause='Existing background grouped a visible preparation/return movement; new direct temporal role separates it.'
                elif new==8 and oy in range(1,8):cause='Visible preparation resembles a main action shape before that action is established.'
                elif new==9 and oy in range(1,8):cause='Main-action label extends into visible return; action end and lowered-rest movement differ.'
                elif new in [1,3,6,7] and oy==0:cause='A directly observed held main-action state was previously background; a terminal hold is not a new occurrence.'
                else:cause='Existing temporal boundary/family assignment differs from direct occurrence evidence; review the pinned RGB before any change.'
            resultrows.append(dict(clip_id=cid,record_id=c['person'],source_frame_id=f,sequence_index=i,source_sha256=c['source_sha256'],
                source_images=c['image_by_frame'][f],source_label_file=c['source_label_file'],
                old_family=oy,old_family_name=roles['roles'][oy] if oy>=0 else 'unlabeled',old_weight=ow,
                proposed_family=new,proposed_family_name=roles['roles'][new] if new is not None else None,
                category=category,subtype=subtype,direct_occurrence_evidence=opts,boundary_bands=hit,uncertainty_reasons=unsure,
                conflict_cause=cause,automatic_overwrite_allowed=False,training_applied=False,
                stage_label_requires_exact_boundary=False,semantic_complete=None))
        rows=resultrows[start:]
        clip_summaries.append(dict(clip_id=cid,reviewed_frames=len(rows),categories=dict(Counter(q['category'] for q in rows)),
            eligible_direct_roles=dict(Counter(str(q['proposed_family']) for q in rows if q['category']!='uncertain')),
            phase_counts=dict(Counter(o['intra_action_phase'] for q in rows if q['category']!='uncertain' for o in q['direct_occurrence_evidence'])),
            source_sha256=c['source_sha256'],frame_id_array_path=meta['path'],frame_id_array_sha256=meta['sha256'],frame_id_only_read=True,
            source_frame_count=len(ids),unviewed_frames_not_added=len(ids)-len(rows),boundaries=c['boundaries']))
    counts=dict(Counter(q['category'] for q in resultrows));subtypes=dict(Counter(q['subtype'] for q in resultrows))
    groups={k:[dict(clip_id=q['clip_id'],frame_id=q['source_frame_id'],old=q['old_family'],old_weight=q['old_weight'],proposed=q['proposed_family'],subtype=q['subtype']) for q in resultrows if q['category']==k] for k in ['new_addition','duplicate','conflict','uncertain']}
    conflictcauses=[dict(clip_id=q['clip_id'],frame_id=q['source_frame_id'],old=q['old_family_name'],new=q['proposed_family_name'],old_weight=q['old_weight'],reason=q['conflict_cause'],source_images=q['source_images']) for q in resultrows if q['category']=='conflict']
    assert all(sha(p['path'])==p['sha256'] for p in sourcepins)
    out=dict(status='DIRECT_ROLE_AUDIT_ONLY_REQUIRES_ROOT_REVIEW_BEFORE_TRAINING',at=datetime.datetime.now().astimezone().isoformat(),
        scope=dict(public_fit_clips=6,recorded_ids=[1,4,7],expressions=['WORD0739','WORD2523'],reviewed_frames=len(resultrows),unviewed_interpolation_frames=0),
        roles=roles['roles'],source_label_pins=sourcepins,image_pins=[dict(path=p,sha256=s) for p,s in imagepins.items()],
        policy=['Role and handshape are separate labels. Handshape is copied only as explanatory evidence, never used to overwrite old shape targets.',
          'Terminal hold stays in the same main occurrence: Egypt family7, upright forearm family3. Preparation8 and return9 are separate.',
          'All inclusive onset/offset and internal motion/shape-boundary bands are excluded from strong new labels. Internal-band roles that may be clear remain in uncertainty list for root review.',
          'New start means an explicitly described early change (closing_and_lowering), not an exact onset time. No exact endpoint or semantic completion is asserted.',
          'Every old w1 conflict is preserved and blocked from automatic overwrite. A previous w1 includes inherited reviewed intervals and is not independent expert truth.',
          'Only actually viewed source frames with pinned RGB evidence are enumerated. Unviewed frames are never interpolated.'],
        counts=counts,subtype_counts=subtypes,clip_summaries=clip_summaries,frame_lists=groups,conflict_cause_table=conflictcauses,rows=resultrows,
        disclosure='This audits existing direct Codex RGB drafts and performs source/hash/frame arithmetic. It is not new independent human/KSL expert ground truth.',
        verification=dict(all_source_video_sha_match=True,all_source_image_sha_match=True,all_old_role_source_sha_match=True,frame_id_mapping_exact=True,
            only_fit_selected=True,source_label_files_unchanged=True),
        training_ran=False,models_modified=False,old_labels_modified=False,dev_or_private_access=False,feature_arrays_loaded=False,
        code_sha256=sha(__file__))
    write(target,out)
    print(json.dumps(dict(counts=counts,subtypes=subtypes,clips=clip_summaries,conflicts=groups['conflict']),ensure_ascii=False))

if __name__=='__main__':main()
