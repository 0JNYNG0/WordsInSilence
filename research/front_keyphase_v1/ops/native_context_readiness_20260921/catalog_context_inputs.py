"""Read-only metadata audit for surrounding context of existing training/development clips."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict, Counter
import hashlib
import json
import math
import zipfile

HERE=Path(__file__).resolve().parent
LAB=HERE.parents[1]; PILOT=LAB.parent/'sign_model_pilot_v1'
PREP=Path('C:/Users/local-user/.codex/.chatgpt-projects/g-p-6a12a39d590c8191960741b826fa75ff/WordsInSilence_학습준비_20260912')
SEN=LAB/'evidence/sentence_target_catalog_20260919'
allowed=set(range(1,10))|{11,12,13}
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def new_json(path, obj):
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def main():
    assert not (HERE/'eligibility.json').exists()
    source=read(PILOT/'expand/manifest.json')
    source=[r for r in source if r['signer_id'] in allowed and r['split_role'] in ('train','validation')]
    assert Counter(r['split_role'] for r in source)=={'train':2095,'validation':696}
    grouped=defaultdict(list)
    for r in source:
        grouped[r['clip_id']].append({**r,'existing_npz':str(PILOT/'expand/npz'/Path(r['npz']).name),
                                      'reference_label':r['word_label'],'raw_label':r['word_label']})
    additional=read(SEN/'apology_native_prep.json')['rows']
    assert len(additional)==12 and {r['signer_id'] for r in additional}==allowed
    for r in additional:
        assert r['sentence_id']=='0356'
        grouped[r['clip_id']].append({
            'sample_id':r['clip_id']+'_apology', 'clip_id':r['clip_id'], 'signer_id':r['signer_id'],
            'split_role':'train' if r['signer_id']<=9 else 'validation',
            'annotation_index':r['annotation_index'], 'start_s':r['annotation_start_s'], 'end_s':r['annotation_end_s'],
            'existing_npz':r['npz'], 'npz_sha256':r['npz_sha256'], 'first_frame':r['first_frame'],
            'last_frame':r['last_frame'], 'reference_label':'죄송하다','raw_label':'미안합니다',
            'semantic_mapping':'Reuse user-confirmed existing treatment mapping; no new lexical merge',
        })
    archive=Path(read(SEN/'catalog.json')['label_archive'])
    stat=archive.stat()
    known_archives={r['signer_id']:r for r in additional}
    pins={str(p):sha(p) for p in (PILOT/'expand/manifest.json',SEN/'apology_native_prep.json',SEN/'catalog.json',
          LAB/'ops/public_anchor_strength_20260921/SELECTED_CANDIDATE.json',
          LAB/'ops/visibility_native_pair_20260919/comparison_policy.json')}
    new_json(HERE/'plan.json',{
        'schema':'native_context_metadata_readiness_plan_v1', 'at':datetime.now(timezone(timedelta(hours=9))).isoformat(),
        'previous_goal_turn_class':'progress: completed26 public input profiles and12 new frozen-candidate forwards with7-pair contrast',
        'rule':'All existing train2095/development696 native samples plus already-approved apology9/3; no score-based or word-based selection. Whole-clip eligibility requires exactly one complete source annotation and one matching source label.',
        'allowed_signers':sorted(allowed),'source_pins':pins,'label_archive':str(archive),
        'label_archive_stat':{'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns},
        'work':'Read exact source annotation members for these clips; reuse old full-frame indexes where available; no new central-directory scan in this metadata phase.',
        'forbidden':['No reserved/participant arrays','No new training or inference','No ZIP rediscovery/download','No lexical approval inferred from metadata','No new runtime cropping or model promotion'],
        'limits':['Single supplied annotation is not proof of complete true action coverage','Nominal native frame clock is not verified video PTS'],
    })
    index_by_signer={}; index_meta={}
    for signer in sorted(allowed):
        index_path=PREP/f'keypoint_index_{signer:02}.json'
        arc=Path(known_archives[signer]['source_archive']); st=arc.stat()
        old_stat=(known_archives[signer]['source_archive_bytes'],known_archives[signer]['source_archive_mtime_ns'])
        assert (st.st_size,st.st_mtime_ns)==old_stat, str(arc)
        old=read(index_path)
        assert Path(old['archive']).resolve()==arc.resolve()
        byclip=defaultdict(list)
        for m in old['members']:
            byclip[m['clip_id']].append(m)
        index_by_signer[signer]=byclip
        index_meta[str(signer)]={'path':str(index_path),'sha256':sha(index_path),'indexed_clips':len(byclip),
                                 'archive':str(arc),'archive_bytes':st.st_size,'archive_mtime_ns':st.st_mtime_ns,
                                 'current_stat_matches_prior_prep':True}
    rows=[]; source_annotations={}
    with zipfile.ZipFile(archive) as z:
        for clip_id, samples in sorted(grouped.items()):
            signer=samples[0]['signer_id']; assert signer in allowed
            member=f'morpheme/{signer:02}/{clip_id}_morpheme.json'
            row={'clip_id':clip_id,'signer_id':signer,'split_role':samples[0]['split_role'],
                 'source_samples':samples,'annotation_member':member,'status':'ERROR'}
            try:
                raw=z.read(member); doc=json.loads(raw); annotations=doc['data']
                source_annotations[clip_id]={'member':member,'sha256':hashlib.sha256(raw).hexdigest(),'document':doc}
                row['annotation_count']=len(annotations)
                row['all_labels']=[[v.get('name') for v in (a.get('attributes') or [])] for a in annotations]
                row['duration_s']=doc.get('metaData',{}).get('duration')
                if len(annotations)!=1:
                    row.update(status='EXCLUDED',reason='MULTIPLE_OR_ZERO_ANNOTATIONS')
                elif len(row['all_labels'][0])!=1:
                    row.update(status='EXCLUDED',reason='AMBIGUOUS_ATTRIBUTES')
                elif len(samples)!=1 or samples[0]['annotation_index']!=0 or row['all_labels'][0][0]!=samples[0]['raw_label']:
                    row.update(status='EXCLUDED',reason='REFERENCE_LABEL_OR_ANNOTATION_MISMATCH')
                else:
                    a=annotations[0]; sample=samples[0]; duration=float(row['duration_s'])
                    assert all(math.isfinite(float(v)) for v in (a['start'],a['end'],duration))
                    assert 0<=a['start']<a['end']<=duration
                    assert a['start']==sample['start_s'] and a['end']==sample['end_s']
                    row.update(status='ELIGIBLE_BY_SOURCE_ANNOTATION',reference_label=sample['reference_label'],
                               focus_window_s=[a['start'],a['end']],existing_full_frame_index_members=len(index_by_signer[signer].get(clip_id,[])))
            except Exception as exc:
                row['error']=type(exc).__name__+': '+str(exc)
            rows.append(row)
    assert (archive.stat().st_size,archive.stat().st_mtime_ns)==(stat.st_size,stat.st_mtime_ns)
    eligible=[r for r in rows if r['status']=='ELIGIBLE_BY_SOURCE_ANNOTATION']
    summary={
        'source_samples':len(source)+len(additional),'source_clips':len(rows),
        'statuses':dict(Counter(r['status'] for r in rows)),
        'exclusion_reasons':dict(Counter(r.get('reason','') for r in rows if r['status']=='EXCLUDED')),
        'eligible_by_role':dict(Counter(r['split_role'] for r in eligible)),
        'eligible_labels_by_role':{role:len({r['reference_label'] for r in eligible if r['split_role']==role}) for role in ('train','validation')},
        'eligible_by_signer':dict(Counter(r['signer_id'] for r in eligible)),
        'eligible_with_existing_full_index':sum(r['existing_full_frame_index_members']>0 for r in eligible),
        'eligible_without_saved_full_index':sum(r['existing_full_frame_index_members']==0 for r in eligible),
    }
    new_json(HERE/'source_annotations.json',source_annotations)
    new_json(HERE/'eligibility.json',{'status':'METADATA_COMPLETE_NOT_FRAME_OR_TRAINING_APPROVAL','summary':summary,
          'index_provenance':index_meta,'rows':rows,'source_annotations_sha256':sha(HERE/'source_annotations.json'),
          'plan_sha256':sha(HERE/'plan.json'),'script_sha256':sha(__file__),
          'reserved_arrays_read':0,'user_arrays_read':0,'keypoint_payloads_read':0,'central_directory_scans':0,
          'read_incident':'Initial line-based preview of one-line expand/manifest.json emitted excessive metadata including reserved split metadata; no reserved coordinates, videos or predictions opened. Subsequent reads parse/filter before bounded output. No selection uses reserved metadata.'})
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return 0 if all(r['status']!='ERROR' for r in rows) else 1


if __name__=='__main__':
    raise SystemExit(main())
