"""Bounded local metadata audit; no raw video, user data, inference or training."""
from pathlib import Path
from collections import Counter, defaultdict
import ast
import hashlib
import json
import zipfile

U = Path(__file__).resolve().parent
OPS = U.parent.parent
PINS = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    p = Path(p)
    PINS[str(p)] = sha(p)
    return json.loads(p.read_text(encoding='utf-8-sig'))


def save(n, value):
    (U/n).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def array_headers(p):
    """Read NPZ headers only, never load or infer coordinates."""
    out = {}
    with zipfile.ZipFile(p) as z:
        for n in z.namelist():
            with z.open(n) as f:
                magic = f.read(8)
                assert magic[:6] == b'\x93NUMPY'
                size = int.from_bytes(f.read(2 if magic[6] == 1 else 4), 'little')
                out[n.removesuffix('.npy')] = ast.literal_eval(f.read(size).decode('latin1').strip())
    return out


def main():
    gloss_path = OPS/'word100_temporal_public_bundle_20261025/GLOSS_REFERENCE_INDEX.json'
    gloss = read(gloss_path)['glossary']
    rows = read(OPS/'word100_training_20261013/MANIFEST.json')
    prep = read(OPS/'word100_body_boundary_20261017/PREP_CONTEXT.json')
    assert len(gloss) == 100 and len(rows) == 1199
    assert sorted(x['classifier_index'] for x in gloss) == list(range(100))
    assert {x['clip_id'] for x in rows} == {x['clip_id'] for x in prep['ordered_rows']}
    assert set(x['role'] for x in rows) == {'train_candidate', 'development_only'}
    groups = defaultdict(list)
    for r in rows:
        groups[r['word_code']].append(r)
        assert r['signer_id'] in (range(1,10) if r['role']=='train_candidate' else (11,12,13))
    labels = []
    for g in sorted(gloss, key=lambda v:v['classifier_index']):
        rr = groups[g['source_word_id']]
        assert {r['canonical_label'] for r in rr} == {g['display_label']}
        assert all(r['label_index']==g['classifier_index'] for r in rr)
        assert g['phonological_attributes'] is None
        labels.append({
            'word_id':g['source_word_id'], 'label':g['display_label'],
            'classifier_index':g['classifier_index'], 'gloss_key':g['gloss_key'],
            'train_clips':sum(r['role']=='train_candidate' for r in rr),
            'dev_clips':sum(r['role']=='development_only' for r in rr),
            'dictionary_variant_match':'UNVERIFIED',
            'phonological_attributes':None,
            'linguistic_acceptability_labels':None,
        })
    save('CURRENT100_LABELS.json', {
        'schema':'wis_current100_label_inventory_v1', 'source':str(gloss_path),
        'source_sha256':PINS[str(gloss_path)], 'count':len(labels), 'labels':labels,
        'warning':'Korean spelling is a lookup candidate only. Source word ID does not certify dictionary variant identity, required features, or acceptable error limits.'})

    # Existing public records identify observations; only TRAIN/DEV rows are opened.
    record_dir=OPS/'word100_body_boundary_20261017/data/records'
    observations=[]; missing=[]; roles=Counter(); people=defaultdict(set)
    for r in rows:
        p=record_dir/(r['clip_id']+'.json'); record=read(p)
        assert record['role']==r['role'] and record['word_code']==r['word_code']
        roles[r['role']]+=1; people[r['role']].add(r['signer_id'])
        d={'clip_id':r['clip_id'],'word_id':r['word_code'],'label':r['canonical_label'],
           'recorded_person_id':r['signer_id'],'role':r['role'],'record':str(p),
           'frames':record['frames'],'word_interval_seconds':r['intervals'],
           'views':{k:v['observation'] for k,v in record['views'].items() if k in ('native','masked','sensor')}}
        for k,v in d['views'].items():
            if not Path(v).exists():missing.append({'clip':r['clip_id'],'view':k,'path':v})
        observations.append(d)
    assert not missing
    save('PUBLIC_TRAIN_DEV_INPUT_INDEX.json', {'count':len(observations),'rows':observations,
          'scope':'Metadata paths/existence only; no array replay, inference or score reuse',
          'training':0,'user_or_mentor_data_access':0,'reserved_payload_access':0})

    # Inspect the actual selected 1199 label payloads, not just the derived index.
    acquisition=read(OPS/'metadata_acquisition_20261011/ACQUISITION_RESULT.json')
    archive=next(x['destination'] for x in acquisition['labels'] if x['official_catalog_entry']['fileStreCours'].split('/')[1]=='1.Training')
    metadata_keys=Counter(); event_keys=Counter(); attribute_keys=Counter(); top_keys=Counter(); payload_pins={}
    samples=[]
    with zipfile.ZipFile(archive) as z:
        for r in rows:
            member=f"morpheme/{r['signer_id']:02d}/{r['clip_id']}_morpheme.json"
            raw=z.read(member); doc=json.loads(raw.decode('utf-8-sig'))
            top_keys.update(doc.keys()); metadata_keys.update(doc['metaData'].keys())
            for event in doc['data']:
                event_keys.update(event.keys())
                for attribute in event.get('attributes',[]):attribute_keys.update(attribute.keys())
            names=[a.get('name') for e in doc['data'] for a in e.get('attributes',[])]
            assert names == r['labels']
            assert [[e['start'],e['end']] for e in doc['data']] == r['intervals']
            payload_pins[member]=hashlib.sha256(raw).hexdigest()
            if len(samples)<1:samples.append({'archive':archive,'member':member,'payload':doc})
    save('AIHUB_MORPHEME_AUDIT.json', {'selected_payloads':len(payload_pins),'archive':archive,
          'top_keys':dict(top_keys),'metadata_keys':dict(metadata_keys),'event_keys':dict(event_keys),
          'attribute_keys':dict(attribute_keys),'samples':samples,'payload_sha256':payload_pins,
          'finding':'Word name and temporal span are present. The inspected payloads provide no required handshape/contact/location feature labels or acceptable/error examples.'})

    face=read(OPS/'word100_face_body_stage_20261027/FROZEN.json')
    face_rows=face['rows']; assert len(face_rows)==18
    assert all(r['role']=='train_candidate' and r['signer'] in (1,2,3) for r in face_rows)
    sample=observations[0]
    face_sample=OPS/'word100_face_body_stage_20261027/data'/face_rows[0]['clip_id']/'face_roi.npz'
    nikl=read(OPS/'corpus_annotations_20261014/FULL_AUDIT.json')
    nikl_schema=read(OPS/'corpus_annotations_20261014/SCHEMA_SAMPLES.json')
    nikl_discovery=read(OPS/'corpus_annotations_20261014/CURRENT100_LABEL_DISCOVERY.json')
    coord=read(OPS/'word100_aihub_coordinate_quality_20261026/FROZEN.json')
    coord_index=read(OPS/'word100_aihub_coordinate_quality_20261026/INDEX.json')
    assert all(r['role']=='train_candidate' and r['signer_id']==2 for r in coord['rows'])
    save('LOCAL_INPUT_AUDIT.json', {
        'schema':'wis_linguistic_local_audit_v1','status':'COMPLETE_BOUNDED_METADATA_AUDIT',
        'method':'Existing metadata and selected public TRAIN/DEV word annotations; NPZ headers only. No videos, reserved data, user or mentor footage opened.',
        'current100':{'glossary_rows':100,'phonological_attributes_nonnull':0,
                      'train_dev_role_counts':dict(roles),'recorded_person_ids':{k:sorted(v) for k,v in people.items()},
                      'missing_observation_paths':missing,'train_clips_per_word':dict(Counter(x['train_clips'] for x in labels)),
                      'dev_clips_per_word':dict(Counter(x['dev_clips'] for x in labels))},
        'observations':{'index':str(U/'PUBLIC_TRAIN_DEV_INPUT_INDEX.json'),
                        'record_schema':list(read(record_dir/(sample['clip_id']+'.json'))),
                        'sample_clip':sample['clip_id'],'sample_npz':sample['views']['masked'],
                        'array_headers':array_headers(sample['views']['masked']),
                        'interpretation':'33 body + 21 anatomical left-hand + 21 anatomical right-hand points. observed/usable/conf and original time distinguish observations from missing coordinates. Coordinates are measured evidence, not correctness labels.'},
        'face_subset':{'manifest':str(OPS/'word100_face_body_stage_20261027/FROZEN.json'),
                       'videos':18,'words':len(set(r['word'] for r in face_rows)),
                       'recorded_person_ids':[1,2,3],'sample_npz':str(face_sample),'array_headers':array_headers(face_sample),
                       'interpretation':'Face landmark observations can supply face-part relationship features; neither contact truth nor expression analysis.'},
        'aihub_keypoints':{'manifest':str(OPS/'word100_aihub_coordinate_quality_20261026/FROZEN.json'),
                           'archive':coord['archive'],'sample_videos':len(coord['rows']),'json_members':len(coord_index['members']),
                           'scope':'100 public TRAIN REAL02 front videos only',
                           'schema_from_reader':{'people.pose_keypoints_2d':'25*(x,y,confidence)',
                                'people.hand_left_keypoints_2d':'21*(x,y,confidence)',
                                'people.hand_right_keypoints_2d':'21*(x,y,confidence)',
                                'people.hand_left_keypoints_3d':'optional 21*4; units/camera basis require verification',
                                'people.hand_right_keypoints_3d':'optional 21*4; units/camera basis require verification'},
                           'quality_report':str(OPS/'word100_aihub_coordinate_quality_20261026/RESULT_KO.txt'),
                           'interpretation':'Provider numerical coordinates may complement observations after synchronization and quality checks. Confidence 1 is not 100% accuracy; no semantic correctness ground truth.'},
        'nikl_notes_only':{'prior_audit':str(OPS/'corpus_annotations_20261014/FULL_AUDIT.json'),
                           'schema_note':str(OPS/'corpus_annotations_20261014/SCHEMA_SAMPLES.json'),
                           'documents_previously_audited':nikl['documents_parsed'],
                           'manual_labels':'sign_script strong/weak contains gloss_id, express, start/end; directions mean agreement/person relation, not hand trajectory vectors.',
                           'nms_excluded':'Nonmanual annotations exist but expressions excluded from current scope.',
                           'landmarks':'Current received archive contains no landmarks per prior full audit. Original archive not reopened.',
                           'current100_spelling_candidates':sum(bool(v) for v in nikl_discovery['matches'].values()),
                           'not_variant_matches':True},
        'label_gap':{'available':['Word ID/name','word temporal span','multiple recorded performers','joint observations','face region observations on 18 TRAIN clips'],
                     'not_found_in_inspected_material':['Per-variant obligatory vs optional handshape/location/motion features','Expert-verified permissible variation boundaries','Expert-labeled incorrect handshape/contact examples','Verified same-variant mapping between AIHub and dictionary entries'],
                     'synthetic_controls':'Masked joints, time warps, changed positions or angles test measurement sensitivity, not whether Korean Sign Language is linguistically incorrect.',
                     'next_learning_gate':'Do not train a semantic acceptability judge from word names or provider coordinates alone. Feature extraction and cautious reference comparison can be validated first; supervised labels need provenance and unresolved-variant status.'},
        'no_training':True,'no_inference':True,'no_runtime_changes':True,'no_source_writes':True,
        'implementation_actor':'Codex local_feature_inventory child','independent_expert_validation':False,
        'source_pins':PINS,
    })
    after={p:sha(p) for p in PINS}
    assert after==PINS
    save('VERIFY.json', {'status':'PASS','checks':['100 unique classifier indices and label agreement','1199 TRAIN/DEV rows only','1199 selected source annotation payload names and intervals agree','all indexed native/masked/sensor observation paths exist','18 face clips all TRAIN IDs01/02/03','all inspected metadata source hashes unchanged'],
                        'source_files_hashed':len(PINS),'same_agent_self_check':True})
    print(json.dumps({'labels':100,'rows':len(rows),'roles':dict(roles),'morpheme_keys':dict(attribute_keys),'sources_unchanged':True},ensure_ascii=False))


if __name__=='__main__':main()
