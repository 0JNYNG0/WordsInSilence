from pathlib import Path
import json,hashlib,datetime,shutil
from collections import Counter
from PIL import Image
from sequence_bundle import build_feature,summarize,FEATURES
U=Path(__file__).resolve().parent;O=U.parent;R=U.parents[3]
P=O/'word100_temporal_transition_guard_20261027';T=O/'word100_temporal_shape_context_20261027';S=O/'word100_shared_feature_bundle_20261027'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
assert not (U/'BUNDLE.json').exists(),'Saved result is immutable'
if not (U/'FROZEN.json').exists():
    (U/'before').mkdir(exist_ok=True)
    for p in [R/'WordsInSilence_PROJECT_CONTEXT.md',R/'NEW_CHAT_HANDOFF.md',O/'STATE.json']:shutil.copy2(p,U/'before'/p.name)
    paths=[P/'COMPARISON.json',P/'visual_inputs/INDEX.json',T/'visual_inputs/INDEX.json',S/'DRAFT_LABELS.json',S/'LEARNING_MANIFEST.json',S/'RUBRIC.json',S/'FROZEN.json',S/'visual_inputs/INDEX.json',Path(__file__),U/'sequence_bundle.py']
    contract={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'authorization':'User requested current-direction review, concrete next plan and continuation; public observation sequence preparation only.',
      'scope':'Reuse all36 fixed public dense windows (12 expressions,3 recorded IDs), no additional outcomes-based sampling.',
      'rules':['Preserve every original feature 0/1/null and PTS; never equate not-open-four with fist.',
               'Runs only group consecutive equal observed samples; unknown splits runs; timestamps and all raw points remain.',
               'Different known anchors bound a feature change; same endpoints around null do not close the gap.',
               'Keep full window denominator if hand/feature disappears; no unknown converted to match.',
               'Join prior shared drafts on exact clip/frame/PTS/side/feature, no nearest-time join. Audit disagreement, never overwrite source or prefer the new reviewer.',
               'Preserve existing expression/recorded-ID roles; both sets previously inspected, not fresh accuracy evaluation.',
               'No prediction/feature training/DTW/web/model changes; external human truth absent until demonstrated.'],
      'acceptance':['936 feature values and468 recorded frames round-trip exactly','All earlier12 bounded changes and8 same-endpoint gaps retained','All leading/trailing/all-unknown runs explicit','Exact shared-frame image andPTS audit; conflicts kept separate','Metamorphic speed/missingness/short pulse/order checks use synthetic contract cases, not sign correctness'],
      'pins':{str(p):sha(p) for p in paths},'protected':read(O/'word100_face_relation_alignment_20261027/FROZEN.json')['protected']}
    dump(U/'FROZEN.json',contract)
frozen=read(U/'FROZEN.json');assert all(sha(p)==h for p,h in frozen['pins'].items())
assert all(sha(p)==h for p,h in frozen['protected'].items())
d=read(P/'COMPARISON.json');dense={r['case_id']:r for root in [P,T] for r in read(root/'visual_inputs/INDEX.json')['rows']}
shared=read(S/'DRAFT_LABELS.json');manifest=read(S/'LEARNING_MANIFEST.json')
roles={r['clip_id']:r for r in manifest['sequence']}
static={(r['clip_id'],r['frame_id'],r['pts_us']):r for r in shared['rows']}
clips={r['clip_id']:r for r in read(S/'visual_inputs/INDEX.json')['rows']}
joined=[];audits=[];image_pairs={};sources={}
for seq in d['sequences']:
    row=dense[seq['case_id']];clip=row['clip_id'];side=row['side'];role=roles[clip]['role']
    assert row['recorded_person_id']==roles[clip]['recorded_person_id']
    frames=sorted(row['frames'],key=lambda x:x['actual_pts_us']);assert len(frames)==len(seq['frames'])==13
    sf=[]
    for a,b in zip(seq['frames'],frames):
        assert (a['frame_id'],a['pts_us'])==(b['frame_id'],b['actual_pts_us'])
        sf.append({'frame_id':a['frame_id'],'pts_us':a['pts_us'],'direct':a['direct'],
                   'image_key':a['image_key'],'full_image':b['full_file'],'focus_image':b['focus_file']})
        key=(clip,a['frame_id'],a['pts_us'])
        if key in static:
            sr=static[key]
            pair=(sr['image'],b['full_file'])
            if pair not in image_pairs:
                with Image.open(pair[0]) as x,Image.open(pair[1]) as y:
                    same=x.size==y.size and x.convert('RGB').tobytes()==y.convert('RGB').tobytes()
                assert same,('Pixel mismatch',pair)
                image_pairs[pair]=same
                for p in pair:sources[p]=sha(p)
            for feature in FEATURES:
                sk=side+'_'+feature;v=a['direct'][feature];old=sr['labels'][sk]
                status='both_unknown' if v is None and old is None else 'one_unknown' if v is None or old is None else 'known_agree' if v==old else 'known_conflict'
                audits.append({'case_id':seq['case_id'],'clip_id':clip,'frame_id':a['frame_id'],'pts_us':a['pts_us'],'side':side,'feature':feature,
                               'shared_value':old,'dense_value':v,'status':status,'source_status':sr['status'][sk],
                               'shared_primary':sr['primary'][sk],'shared_secondary':sr['secondary'][sk],
                               'source_image':sr['image'],'dense_image':b['full_file'],'role':role,
                               'resolution':'preserve_both_no_truth_arbitration','training_target':None})
    joined.append({'case_id':seq['case_id'],'clip_id':clip,'word_id':row['word_id'],'label':row['label'],
                   'recorded_person_id':row['recorded_person_id'],'side':side,'role':role,
                   'source_alias':roles[clip]['alias'],'window_only':True,'frames':sf,
                   'features':{feature:build_feature(sf,feature) for feature in FEATURES},
                   'inherited_sparse_sequence_draft':roles[clip]['labels'][side],
                   'sparse_sequence_scope':roles[clip]['scope'],
                   'source_label_status':'AI image observation drafts, not independent human truth'})
joined.sort(key=lambda x:x['case_id'])
counts=summarize(joined)
assert len(joined)==36 and counts['feature_samples']==936
assert counts['bounded_feature_change']==12 and counts['same_endpoints_unresolved_gap']==8
out={'status':'PREPARED_VERIFIED_SEQUENCE_REPRESENTATION_NOT_LEARNED','counts':counts,
     'windows':len(joined),'frames':sum(len(s['frames']) for s in joined),'roles':dict(Counter(s['role'] for s in joined)),
     'sequences':joined,'exact_static_join':{'counts':dict(Counter(r['status'] for r in audits)),'feature_pairs':len(audits),'image_pairs':len(image_pairs),'rows':audits},
     'label_readiness':{'human_verified_labels':False,'training_ready':False,'model_gain_established':False,
                      'why':'Saved labels are AI drafts; agreeing reviews and exact pixels do not establish true hidden shape or educational correctness.'},
     'source_pins':{**frozen['pins'],**sources},'production_changed':False,'training':False}
dump(U/'BUNDLE.json',out)
review=[x for x in audits if x['status']!='known_agree']
dump(U/'REVIEW_QUEUE.json',{'scope':'All exact-shared-time discrepancies, no labels overwritten. Review current image then context; do not force agreement.', 'rows':review,'counts':dict(Counter(r['status'] for r in review))})
print(json.dumps({k:v for k,v in out.items() if k in ['status','counts','windows','frames','roles','label_readiness']},ensure_ascii=False,indent=2))
print('exact_shared_join',out['exact_static_join']['counts'],'imagepairs',len(image_pairs))
