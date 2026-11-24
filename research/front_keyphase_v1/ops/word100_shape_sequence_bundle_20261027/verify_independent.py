"""Verify saved sequence contracts independently; never imports producer code."""
from pathlib import Path
from collections import Counter, defaultdict
from itertools import groupby
import hashlib, json, sys, traceback
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')
U=Path(__file__).resolve().parent; OPS=U.parent
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def pixel(p):
    with Image.open(p) as im:
        im=im.convert('RGB')
        return im.size,hashlib.sha256(im.tobytes()).hexdigest()

checks=[]; failures=[]; counts=Counter(); logic_conflicts=[]
def check(name,condition,detail=None):
    ok=bool(condition); checks.append({'check':name,'pass':ok,'detail':detail})
    if not ok: failures.append(name)

result={'verification_scope':'Separate code verification of saved contracts, not new human ground truth, detector accuracy or project performance.',
        'producer_imported':False,'source_modified':False,'training':False,'new_detection':False,
        'new_dtw':False,'user_mentor_data_access':False,'protected_unchanged':False}
try:
    frozen=read(U/'FROZEN.json'); bundle=read(U/'BUNDLE.json')
    before={str(U/'FROZEN.json'):sha(U/'FROZEN.json'),str(U/'BUNDLE.json'):sha(U/'BUNDLE.json')}
    guard=OPS/'word100_temporal_transition_guard_20261027'
    old=OPS/'word100_temporal_shape_context_20261027'
    shared=OPS/'word100_shared_feature_bundle_20261027'
    source=read(guard/'COMPARISON.json'); sparse=read(shared/'DRAFT_LABELS.json')
    sharedf=read(shared/'FROZEN.json')
    sharedmeta={r['clip_id']:r for r in sharedf['rows']}
    denseidx={r['case_id']:r for p in (guard/'visual_inputs/INDEX.json',old/'visual_inputs/INDEX.json') for r in read(p)['rows']}
    srcseq={s['case_id']:s for s in source['sequences']}
    seqs={s['case_id']:s for s in bundle['sequences']}
    check('36 unique original windows',len(seqs)==len(bundle['sequences'])==36 and set(seqs)==set(srcseq)==set(denseidx))
    sourcepins=frozen['pins']|bundle['source_pins']
    check('Frozen and bundle source pins consistent',all(bundle['source_pins'].get(p)==h for p,h in frozen['pins'].items()))
    check('Source pins unchanged',all(sha(p)==h for p,h in sourcepins.items()),len(sourcepins))
    protected=frozen['protected']
    check('Protected C/T3F/web/DTW pins unchanged',all(sha(p)==h for p,h in protected.items()),len(protected))
    role_by_sourceframe=defaultdict(set); pixel_cache={}; expected_links=[]; aggregate=Counter(); roles=Counter()
    logicalstates=Counter()
    for cid,s in seqs.items():
        orig=srcseq[cid]; idx=denseidx[cid]; sf=sharedmeta[idx['clip_id']]
        assert s['clip_id']==idx['clip_id']==sf['clip_id']
        assert idx['source_sha256']==sf['source']['sha256']
        assert s['word_id']==idx['word_id']==sf['word_id']
        assert s['recorded_person_id']==idx['recorded_person_id']==sf['recorded_person_id']==orig['person_id']
        assert s['side']==idx['side']==orig['side']
        assert s['source_alias']==idx['source_alias']
        assert s['role']==idx['inherited_role']==sf['role']
        assert s['window_only'] is True
        roles[s['role']]+=1
        frames=s['frames']; original=orig['frames']; imap={x['frame_id']:x for x in idx['frames']}
        assert len(frames)==len(original)==len(imap)==13
        assert [x['frame_id'] for x in frames]==sorted(imap)
        for n,(a,b) in enumerate(zip(frames,original)):
            z=imap[a['frame_id']]
            assert a['frame_id']==b['frame_id'] and a['pts_us']==b['pts_us']==z['actual_pts_us']
            assert a['image_key']==b['image_key']==z['image_key']
            assert a['direct']==b['direct'] and set(a['direct'])=={'index_only','open_four'}
            assert a['full_image']==z['full_file'] and a['focus_image']==z['focus_file']
            if n: assert a['frame_id']==frames[n-1]['frame_id']+1 and a['pts_us']>frames[n-1]['pts_us']
            for v in a['direct'].values(): assert v is None or type(v) is int and v in (0,1)
            pix=pixel(a['full_image']); pixel_cache[a['full_image']]=pix
            assert pix[1]==z['rgb_sha256']; counts['decoded_dense_RGB']+=1
            counts['original_frames']+=1; counts['original_feature_cells']+=2
            key=(idx['source_sha256'],a['frame_id']); role_by_sourceframe[key].add(s['role'])
            logicalstates[str((a['direct']['index_only'],a['direct']['open_four']))]+=1
            if a['direct']['index_only']==a['direct']['open_four']==1: logic_conflicts.append({'case_id':cid,'frame_id':a['frame_id']})
        for feature,rep in s['features'].items():
            vals=[x['direct'][feature] for x in original]
            assert rep['values']==vals
            spans=[]; offset=0
            for value,g in groupby(vals):
                group=list(g); spans.append((offset,offset+len(group)-1,value)); offset+=len(group)
            assert len(rep['runs'])==len(spans)
            roundtrip=[]
            for r,(a,b,v) in zip(rep['runs'],spans):
                assert r['value']==v and r['start_index']==a and r['end_index']==b
                assert r['indices']==list(range(a,b+1)) and r['count']==b-a+1
                assert r['start_pts_us']==frames[a]['pts_us'] and r['end_pts_us']==frames[b]['pts_us']
                assert r['start_frame_id']==frames[a]['frame_id'] and r['end_frame_id']==frames[b]['frame_id']
                assert r['kind']==('unobserved_run' if v is None else 'observed_equal_feature_run')
                assert 'not proof' in r['time_scope']
                roundtrip.extend([v]*(b-a+1)); aggregate[r['kind']]+=1
            assert roundtrip==vals
            aggregate['feature_runs']+=len(spans)
            expected=[]; prev=None
            for i,v in enumerate(vals):
                if v is None: continue
                if prev is not None and (v!=vals[prev] or i!=prev+1):
                    kind='bounded_feature_change' if v!=vals[prev] else 'same_endpoints_unresolved_gap'
                    expected.append((prev,i,kind))
                prev=i
            assert len(rep['links'])==len(expected)
            for link,(a,b,kind) in zip(rep['links'],expected):
                assert link['kind']==kind and link['from_index']==a and link['to_index']==b
                assert link['from_value']==vals[a] and link['to_value']==vals[b]
                assert link['from_pts_us']==frames[a]['pts_us'] and link['to_pts_us']==frames[b]['pts_us']
                assert link['unobserved_indices']==list(range(a+1,b))
                assert all(vals[i] is None for i in range(a+1,b))
                assert link['exact_transition_time'] is None and link['intermediate_shape'] is None and link['linguistic_importance'] is None
                aggregate[kind]+=1; expected_links.append((cid,feature,kind,a,b))
            unknownspans=[x for x in spans if x[2] is None]
            assert len(rep['gaps'])==len(unknownspans)
            for gap,(a,b,v) in zip(rep['gaps'],unknownspans):
                assert gap['from_index']==a and gap['to_index']==b and gap['indices']==list(range(a,b+1))
                assert gap['left_anchor']==(a-1 if a else None)
                assert gap['right_anchor']==(b+1 if b<12 else None)
                kind=('entire_window_unobserved' if a==0 and b==12 else 'leading_unobserved' if a==0 else 'trailing_unobserved' if b==12 else 'interior_unobserved')
                assert gap['kind']==kind and gap['value'] is None and gap['gap_not_imputed'] is True
                aggregate[kind]+=1
            known=sum(v is not None for v in vals); cov=rep['coverage']
            assert cov['expected_samples']==13 and cov['known_samples']==known and cov['unobserved_samples']==13-known
            assert cov['observed_fraction']==known/13
            assert cov['expected_from_pts_us']==frames[0]['pts_us'] and cov['expected_to_pts_us']==frames[-1]['pts_us']
            for k in ('semantic_complete','no_change_certified','whole_sign_scope'): assert rep[k] is False
            assert rep['feedback_verdict'] is None and rep['handshape_name'] is None
            aggregate['feature_samples']+=13; aggregate['known_samples']+=known; aggregate['unobserved_samples']+=13-known
    check('936 original values and 468 frame metadata retained',counts['original_feature_cells']==936 and counts['original_frames']==468)
    check('468 dense images match saved decoded RGB hashes',counts['decoded_dense_RGB']==468)
    check('Independent runs, gaps and coverage match',dict(aggregate)==bundle['counts'],dict(aggregate))
    check('Unknown splits runs and cannot become false zero',aggregate['unobserved_samples']==191 and aggregate['known_samples']==745)
    check('All 12 changes and 8 unresolved same-endpoint gaps preserved',Counter(x[2] for x in expected_links)=={'bounded_feature_change':12,'same_endpoints_unresolved_gap':8})
    event_kind_map={'observed_state_change_interval':'bounded_feature_change','same_endpoints_unresolved_middle':'same_endpoints_unresolved_gap'}
    original_events={(e['case_id'],e['feature'],event_kind_map[e['kind']],e['from_index'],e['to_index']) for e in source['events']}
    check('All previous events retained exactly',original_events==set(expected_links) and len(source['events'])==len(expected_links))
    check('Leading trailing whole-window unknown explicit',(aggregate['leading_unobserved'],aggregate['trailing_unobserved'],aggregate['entire_window_unobserved'])==(5,1,10))
    check('Inherited 16/8/8/4 roles preserved',dict(roles)==bundle['roles']=={'fit_candidate':16,'held_person':8,'held_expression_seen_person':8,'held_expression_new_person':4})
    check('No observed index-only/open-four positive-positive contradictions',not logic_conflicts,logic_conflicts)
    # Reconstruct the exact sparse/dense join without nearest-time matching.
    smap={(r['clip_id'],r['frame_id'],r['pts_us']):r for r in sparse['rows']}
    assert len(smap)==len(sparse['rows'])
    expected_join={}; image_pairs=set(); statuses=Counter()
    for r in sparse['rows']:
        sf=sharedmeta[r['clip_id']]
        role_by_sourceframe[(sf['source']['sha256'],r['frame_id'])].add(r['role'])
    for cid,s in seqs.items():
        for fr in s['frames']:
            row=smap.get((s['clip_id'],fr['frame_id'],fr['pts_us']))
            if row is None: continue
            for feature in ('index_only','open_four'):
                k=s['side']+'_'+feature; sv=row['labels'][k]; dv=fr['direct'][feature]
                status='both_unknown' if sv is None and dv is None else 'one_unknown' if sv is None or dv is None else 'known_agree' if sv==dv else 'known_conflict'
                key=(cid,fr['frame_id'],s['side'],feature); expected_join[key]=(row,fr,sv,dv,status)
                statuses[status]+=1; image_pairs.add((row['image'],fr['full_image']))
    saved=bundle['exact_static_join']; actual={ (r['case_id'],r['frame_id'],r['side'],r['feature']):r for r in saved['rows'] }
    assert len(actual)==len(saved['rows']) and set(actual)==set(expected_join)
    for key,(row,fr,sv,dv,status) in expected_join.items():
        j=actual[key]; k=key[2]+'_'+key[3]
        assert j['clip_id']==row['clip_id'] and j['pts_us']==row['pts_us']==fr['pts_us']
        assert j['shared_value']==sv and j['dense_value']==dv and j['status']==status
        assert j['source_status']==row['status'][k] and j['shared_primary']==row['primary'][k] and j['shared_secondary']==row['secondary'][k]
        assert j['source_image']==row['image'] and j['dense_image']==fr['full_image']
        assert j['role']==row['role']==seqs[key[0]]['role']
        assert j['resolution']=='preserve_both_no_truth_arbitration' and j['training_target'] is None
    for a,b in image_pairs:
        pa=pixel(a); pb=pixel_cache[b]
        assert pa==pb
    check('76 exact static feature pairs preserve both sources',len(actual)==saved['feature_pairs']==76 and dict(statuses)==saved['counts'],dict(statuses))
    check('38 shared/dense image pairs have identical decoded RGB',len(image_pairs)==saved['image_pairs']==38)
    role_conflicts=[{'source_sha256':k[0],'frame_id':k[1],'roles':sorted(v)} for k,v in role_by_sourceframe.items() if len(v)>1]
    check('Original source SHA and frame do not cross roles',not role_conflicts,{'original_frame_keys':len(role_by_sourceframe),'conflicts':role_conflicts})
    check('No training production change or ground-truth upgrade',bundle['training'] is False and bundle['production_changed'] is False and all(bundle['label_readiness'][k] is False for k in ('human_verified_labels','training_ready','model_gain_established')))
    check('Read-only verification preserved frozen/bundle inputs',all(sha(p)==h for p,h in before.items()))
    result.update(counts=dict(counts),aggregate=dict(aggregate),logical_feature_combinations=dict(logicalstates),logic_conflicts=logic_conflicts,
                  source_pins_checked=len(sourcepins),protected_pins_checked=len(protected),protected_unchanged=not any(x['check']=='Protected C/T3F/web/DTW pins unchanged' and not x['pass'] for x in checks),
                  reviewed_sha256=before,static_join_counts=dict(statuses),duplicate_role_conflicts=role_conflicts)
except Exception:
    failures.append('unexpected_assertion_or_execution_error');result['error']=traceback.format_exc()
result.update(status='PASS_SAVED_CONTRACT_VERIFICATION' if not failures else 'FAIL',checks=checks,failures=failures,
              event_kind_translation={'observed_state_change_interval':'bounded_feature_change','same_endpoints_unresolved_middle':'same_endpoints_unresolved_gap'},
              initial_checker_repair='Preserved first attempt failed due to checker schema assumptions: old/new event kind names require explicit equivalent mapping; shared source SHA lives at source.sha256. Source data and bundle were unchanged.',
              limits=['All labels remain same-AI visual drafts, not independent human truth.','Two binary features are not a complete handshape; 0/0 does not mean a fist or unchanged shape.','13 sampled consecutive frames do not establish a whole-sign phase or exact movement timing.','No hidden hand shape, educational correctness, TCN/DTW improvement, or automatic recovery is verified.'])
out=U/'INDEPENDENT_VERIFY.json'
if out.exists():
    previous=read(out);result['prior_attempts']=previous.get('prior_attempts',[])+[{k:v for k,v in previous.items() if k!='prior_attempts'}]
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(result,ensure_ascii=False,indent=2))
raise SystemExit(0 if not failures else 1)
