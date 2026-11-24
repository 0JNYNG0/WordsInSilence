"""Read-only public metadata/cache inventory; no images, inference or label creation."""
from pathlib import Path
from collections import Counter, defaultdict
import json, hashlib, re
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
OPS = ROOT / 'research/front_keyphase_v1/ops'
OUT = Path(__file__).parent
WORDS = {'WORD0739':'순종','WORD1133':'잘하다','WORD1567':'조상','WORD2523':'이집트'}
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def pin(p): return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def person(cid): return int(re.search(r'_REAL(\d+)_',cid)[1])
def path_info(p, expected=None):
    p=Path(p); d={'path':str(p),'exists':p.is_file()}
    if d['exists']:
        d.update(bytes=p.stat().st_size,sha256=sha(p))
        if expected is not None: d['matches_recorded_sha256']=d['sha256']==expected
    return d

sources={k:OPS/v for k,v in {
 'front_join':'word_dataset_plan_20261013/FRONT_VIDEO_JOIN.jsonl',
 'global_roles':'word_dataset_plan_20261013/PLAN.json',
 'tcn_manifest':'word100_training_20261013/MANIFEST.json',
 'tcn_frozen':'word100_training_20261013/FROZEN_EXPERIMENT.json',
 't3f_training':'word100_frozen_c_temporal_20261019/TRAIN_CONTEXT.json',
 't3f_ready':'word100_frozen_c_temporal_20261019/READY.json',
 'stage_selection':'word100_stage_feedback_cycle_20261031/SELECTION.json',
 'stage_inputs':'word100_stage_feedback_cycle_20261031/LEARNING_INPUTS.json',
 'semantic_dataset':'word100_semantic_feature_cycle_20261101/DATASET.json',
 'semantic_preparation':'word100_semantic_feature_cycle_20261101/PREPARATION.json',
 'semantic_contract':'word100_semantic_feature_cycle_20261101/FEATURE_CONTRACT.json',
 'bank_frozen':'word100_train_reference_bank_20261026/FROZEN.json',
}.items()}
selected=[]
with sources['front_join'].open(encoding='utf-8') as f:
    for line in f:
        r=json.loads(line)
        if r.get('word_code') in WORDS:selected.append(r)
stage=read(sources['stage_inputs']); stage_idx={r['clip_id']:r for r in stage['rows']}
sem=read(sources['semantic_dataset']); sem_idx=defaultdict(list)
for r in sem['rows']: sem_idx[r['clip_id']].append(r)
bank=read(sources['bank_frozen']); bank_video={r['clip_id']:r['source'].get('video') for r in bank['rows'] if any(w in r['clip_id'] for w in WORDS)}
labels=[]; inventory=[]
for r in sorted(selected,key=lambda x:x['clip_id']):
    cid=r['clip_id']; pid=person(cid); w=r['word_code']
    recpath=OPS/'word100_body_boundary_20261017/data/records'/f'{cid}.json'
    rec=read(recpath) if recpath.is_file() else None
    sr=stage_idx.get(cid); semantic=sem_idx[cid]
    item={'clip_id':cid,'word_id':w,'label':WORDS[w],'recorded_person_id':pid,'historical_source_role':r['role'],
          'word_interval_seconds':r['intervals'],'metadata_clean':r['prior_metadata_clean'],
          'metadata_video_join_eligible':r['metadata_video_join_eligible'],'archive_members':r['video_entries'],
          'native_record':pin(recpath) if rec else None,'native_observation':None,'source_video_caches':[],
          'stage_head_role':sr['role'] if sr else None,'stage_rgb69_input':path_info(sr['input'],sr['input_sha256']) if sr else None,
          'semantic7_exact_centers':[{'frame_id':x['center_frame'],'timestamp_us':x['center_timestamp_us'],'known_targets':[t for t,m in zip(sem['targets'],x['target_mask']) if m],'role':x['role']} for x in semantic]}
    for e in item['archive_members']:
        a=Path(e['archive'])
        if not a.exists():
            alternate=Path(str(a).replace('videos\\1.Training','videos\\수어 영상\\1.Training'))
            if alternate.exists():a=alternate
        e['resolved_archive']=str(a);e['archive_exists']=a.is_file()
    if rec:
        op=rec['views']['native']['observation']
        item['native_observation']=path_info(op,rec['outputs'].get(op));item['native_frames']=rec['frames']
    candidates=[]
    if sr:candidates.append(sr['source'])
    if bank_video.get(cid):candidates.append(bank_video[cid])
    # Exact public cache locations already identified by the saved producer manifests/code.
    candidates += [str(OPS/'word100_wrist_only_inputs_20261026/sources'/f'{cid}.mp4'),
                   str(Path('W:/WordsInSilence/data/AIHub_103/word100_web_references_20261013')/f'{cid}.mp4')]
    for p in dict.fromkeys(candidates):
        if Path(p).is_file():item['source_video_caches'].append(path_info(p,rec['source']['sha256'] if rec else None))
    if sr:
        # One small label NPZ at a time; no RGB/native/full input tensors loaded.
        with np.load(sr['targets_file'],allow_pickle=False) as z:
            mask=z['target_mask']; yy=z['y']
            # The exact original three target indices in the native 17-target schema.
            old3={t:{'known':int(mask[:,j].sum()),'positive':int(((yy[:,j]==1)&mask[:,j]).sum()),'negative':int(((yy[:,j]==0)&mask[:,j]).sum())} for t,j in [('R_index_only',0),('R_open_four',2),('L_open_four',3)]}
        item['native_stage_labels']={'file':path_info(sr['targets_file'],sr['targets_sha256']),'old3_support':old3,'semantic7_not_automatically_filled':True}
    else:item['native_stage_labels']=None
    inventory.append(item)

by_person=[]
for pid in range(1,19):
    rr=[r for r in inventory if r['recorded_person_id']==pid]
    by_person.append({'person_id':pid,'clips':len(rr),'clean_joined':sum(r['metadata_video_join_eligible'] and r['metadata_clean'] for r in rr),
      'native_observation_files':sum(bool(r['native_observation'] and r['native_observation']['exists']) for r in rr),
      'cached_video_clips':sum(bool(r['source_video_caches']) for r in rr),
      'stage_rgb69_clips':sum(bool(r['stage_rgb69_input'] and r['stage_rgb69_input']['exists']) for r in rr),
      'stage_labeled_clips':sum(bool(r['native_stage_labels']) for r in rr),
      'semantic7_centers':sum(len(r['semantic7_exact_centers']) for r in rr)})
sem_support=[]
for pid in sorted({r['person'] for r in sem['rows']}):
    rr=[r for r in sem['rows'] if r['person']==pid]
    current=[r for r in rr if r['word_id'] in WORDS]
    sem_support.append({'person_id':pid,'roles':sorted({r['role'] for r in rr}),'all_words_clips':len({r['clip_id'] for r in rr}),'all_words_centers':len(rr),'four_expression_clips':len({r['clip_id'] for r in current}),'four_expression_centers':len(current),
      'four_expression_support':{t:{'known':sum(bool(r['target_mask'][j]) for r in current),'positive':sum(bool(r['target_mask'][j]) and r['y'][j]==1 for r in current),'negative':sum(bool(r['target_mask'][j]) and r['y'][j]==0 for r in current)} for j,t in enumerate(sem['targets'])}})
t3f=read(sources['t3f_training']); tids=t3f['ordered_train_ids']; tc=read(sources['tcn_frozen'])
report={'status':'READ_ONLY_PUBLIC_INVENTORY_COMPLETE','scope':'Four specified expressions only for per-clip inventory; public manifests for model lineage and roles. No private/user/mentor data, pixels, new inference/training/extraction or downloads.',
 'recorded_id_not_biometric_identity':'REAL ID is the dataset recorded performer identity; independent physical-person deduplication was not repeated here.',
 'tcn_C_T3F':{'task':'100-word clip classifier, not seven time-resolved physical attributes','train_clips':len(tids),'train_by_id':dict(sorted(Counter(person(x) for x in tids).items())),'train_ids':tc['train_signers'],'dev_clips':tc['dev_clips'],'dev_ids':tc['dev_signers'],'T3F':'Frozen original C backbone plus trained temporal contrast branch; same 899 training IDs. Do not replace these models.',
  'four_expression_train_clips':sum(any(w in x for w in WORDS) for x in tids)},
 'semantic7':{'targets':sem['targets'],'rows':len(sem['rows']),'clips':len({r['clip_id'] for r in sem['rows']}),'role_support':sem_support,'time_scope':'One exact center label per 13-frame input window; do not propagate center labels to the 13 frames or word labels to frame features.','label_quality':'Existing direct RGB physical labels from same-Codex review; not independent human linguistic/educational truth.'},
 'four_expression_inventory':inventory,'by_person':by_person,'source_pins':{k:pin(v) for k,v in sources.items()},
 'role_constraints':{'current_fit':[1,2,3],'current_development':[4],'original_head_locked_test':[5,6],'semantic_new_feature_expression_evaluation':[7,8],'additional_fit_candidate_from_existing_1_to_9':[9],
  'unassigned_source_id':[10],'historical_TCN_development':[11,12,13],'historical_source_reserved':[14,15,16],'official_validation_reserved':[17,18],
  'rule':'Preserve the union of existing development/evaluation restrictions. Earlier TCN training exposure is disclosed even when a clip is head-held-out. Do not move 4,5,6,7,8 or 11..18 into new FIT.'},
 'nine_person_answer':{'nine_ids_public_sources':'IDs01..09 exist for every expression (36 clips); this is a possible nine-recorded-ID study population, not nine new training people.',
  'nine_fit_without_reassigning_roles':False,'immediate_safe_reference_or_training_candidate_ids':[1,2,3,9],
  'new_fit_9_has_time_feature_truth':False,'why':'ID09 has clip identity/word core interval and native data, but no exact semantic7 target labels in the audited current training source. IDs05/06 remain held out; IDs07/08 remain evaluation.'},
 'realization':[
  'Use existing source clip labels and main-action intervals for a separate normal target-action/reference-variation objective on FIT01/02/03 plus proposed09. This can learn or summarize normal performance variation; it cannot supervise required handshape/phase correctness or learner-error rejection.',
  'For the seven-feature head, retain existing FIT and labels. If adding ID09, fix a small center-window selection from source time metadata before predictions, review only observable physical attributes, retain unknowns, and audit per-target positive/negative support. A model prediction is not a new label.',
  'Test supported targets on existing locked05/06 labels without converting them to FIT; preserve07/08 expression evaluation. For target-known four-expression tests on07/08, new bounded direct physical labels are needed before scoring those targets.',
  'ID10 is a possible newly assigned FIT source only after a scoped exposure audit and new frozen role; no native/current-head cache was assumed. Keep historical reserved14..18 for separately authorized evaluation, not the route to nine FIT.',
  'If new frame truth is insufficient, stop that head-training branch and still deliver multi-reference live evidence with explicit observed/model-estimated/unknown states. Do not fabricate word-to-frame labels or stall the demo for full annotation.'
 ],
 'remaining_unknown':['Independent semantic/educational error truth and accepted variant/required-repeat labels are absent.','Archive membership is not a fresh complete decode/CRC audit; source cache hashes were checked only where local files exist.','Availability does not demonstrate improvement or make the repeated DEV cohort fresh.'],
 'actions':{'new_model_inference':0,'new_training':0,'new_extraction':0,'new_RGB_review':0,'source_modifications':0,'private_search':0}}
OUT.mkdir(parents=True,exist_ok=True)
dest=OUT/'PUBLIC_PEOPLE_INVENTORY.json'
assert not dest.exists(), 'Preserve prior audit result'
dest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'result':pin(dest),'by_person':by_person,'semantic7_support':sem_support,'tcn':report['tcn_C_T3F']},ensure_ascii=False))
