"""Research evidence adapter; no detector, learner, DTW optimizer, or verdict.

API:
  project_class_probabilities(p45, class_names, hand_present=..., identity_resolved=...)
  observed_hand({...}, hand_present=..., identity_resolved=...)
  compare_references(query_frames, references)

Each frame is {time_s: float, hands: {L: hand, R: hand}}. Each reference is
{reference_id, frames, path:[[query_index,reference_index],...],
 required:{L:{feature:[bool,...]},R:{feature:[bool,...]}}}.
Missing feature requirements must not be inferred from query observations.
The caller owns the already-frozen path and reference-required masks.
"""
from pathlib import Path
from functools import lru_cache
import copy
import hashlib
import json
import math

U = Path(__file__).resolve().parent
FEATURES = ('index_only', 'open_four', 'four_closed')
HANDS = ('L', 'R')
THRESHOLD = .8

def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

@lru_cache(maxsize=1)
def _mapping():
    protocol = json.loads((U/'BRIDGE_PROTOCOL.json').read_text(encoding='utf-8'))
    p = U/'research/DEEPHAND_PHYSICAL_PARTIAL_MAPPING.json'
    if _sha(p) != protocol['mapping_sha256']:
        raise ValueError('The source mapping differs from the frozen bridge protocol')
    if protocol['mass_rule']['estimate_threshold'] != THRESHOLD:
        raise ValueError('Threshold contract mismatch')
    raw = json.loads(p.read_text(encoding='utf-8'))['rows']
    result = {r['class_name']: tuple(r[f] for f in FEATURES) for r in raw}
    if len(raw) != 45 or len(result) != 45 or any(v not in (0,1,None) for row in result.values() for v in row):
        raise ValueError('Expected 45 unique, partially mapped author classes')
    return result, protocol['mapping_sha256']

def _boolean(value, name):
    if type(value) is not bool:
        raise ValueError(name + ' must be an explicit boolean')
    return value

def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (float,int)) or not math.isfinite(value):
        raise ValueError(name + ' must be a finite number')
    return float(value)

def _state_value(value):
    if value is not None and (type(value) is not int or value not in (0,1)):
        raise ValueError('Feature state must be integer 0, 1, or null')
    return value

def project_class_probabilities(probabilities, class_names, *, hand_present, identity_resolved):
    """Preserve raw softmax and pool source-diagram mass without recalibration.

These are model estimates even when their largest mass is high. No anatomical
side is inferred. Unknown mapping mass is not redistributed to known classes.
"""
    hand_present = _boolean(hand_present,'hand_present')
    identity_resolved = _boolean(identity_resolved,'identity_resolved')
    mapping, mapping_hash = _mapping()
    names = list(class_names)
    if len(names) != 45 or len(set(names)) != 45 or set(names) != set(mapping):
        raise ValueError('Class order must explicitly name each of the 45 original classes')
    try:
        values = [float(x) for x in probabilities]
    except (TypeError,ValueError) as exc:
        raise ValueError('One 45-element probability vector is required') from exc
    if len(values) != 45 or not all(math.isfinite(x) and 0 <= x <= 1 for x in values):
        raise ValueError('Probabilities must be 45 finite values in [0,1]')
    if not math.isclose(math.fsum(values),1.,rel_tol=0.,abs_tol=1e-6):
        raise ValueError('Supply normalized class probabilities, not logits')
    features = {}
    for k, feature in enumerate(FEATURES):
        mass = {v: math.fsum(p for name,p in zip(names,values) if mapping[name][k] == v)
                for v in (1,0,None)}
        reason = ('hand_not_observed' if not hand_present else
                  'hand_identity_unresolved' if not identity_resolved else None)
        estimate = None if reason else (1 if mass[1] >= THRESHOLD else 0 if mass[0] >= THRESHOLD else None)
        if reason is None and estimate is None:
            reason = 'insufficient_source_mapped_model_mass'
        features[feature] = {'state':estimate,'kind':'model_estimate','reason':reason,
                             'p_pos':mass[1],'p_neg':mass[0],'p_unknown':mass[None],
                             'measurement':None,'calibrated_probability':False}
    return {'hand_present':hand_present,'identity_resolved':identity_resolved,
            'input_kind':'model_estimate','features':features,
            'raw_class_names':names,'raw_class_probabilities':values,
            'source_mapping_sha256':mapping_hash,'educational_verdict':None}

def observed_hand(features, *, hand_present, identity_resolved, evidence_id):
    """Wrap externally supplied current observations, without certifying truth.

Each feature contains state 0/1/null and an optional measurement
{value:number, unit:string}. A model prediction must use the model adapter.
"""
    _boolean(hand_present,'hand_present'); _boolean(identity_resolved,'identity_resolved')
    if not isinstance(evidence_id,str) or not evidence_id:
        raise ValueError('An external current-observation evidence_id is required')
    if not isinstance(features,dict) or set(features)-set(FEATURES):
        raise ValueError('Unexpected feature keys')
    output = {}
    for f in FEATURES:
        raw = features.get(f,{'state':None})
        state = _state_value(raw.get('state'))
        measurement = copy.deepcopy(raw.get('measurement'))
        if measurement is not None:
            _number(measurement.get('value'),'measurement.value')
            if not isinstance(measurement.get('unit'),str) or not measurement['unit']:
                raise ValueError('Measurement unit required')
            measurement['kind'] = 'current_observed'
        reason = ('hand_not_observed' if not hand_present else 'hand_identity_unresolved'
                  if not identity_resolved else 'feature_unobserved' if state is None else None)
        output[f] = {'state':state if hand_present and identity_resolved else None,
                     'kind':'current_observed','reason':reason,
                     'measurement':measurement if hand_present and identity_resolved else None,
                     'original_observation':copy.deepcopy(raw)}
    return {'hand_present':hand_present,'identity_resolved':identity_resolved,
            'input_kind':'current_observed','evidence_id':evidence_id,
            'features':output,'educational_verdict':None}

def _validate_frames(frames):
    if not isinstance(frames,list):
        raise ValueError('Frames must be a list')
    previous = -math.inf
    for frame in frames:
        if not isinstance(frame,dict): raise ValueError('Frame object required')
        t = _number(frame.get('time_s'),'time_s')
        if t <= previous: raise ValueError('Frame times must strictly increase')
        previous = t
        hands = frame.get('hands',{})
        if not isinstance(hands,dict) or set(hands)-set(HANDS):
            raise ValueError('Hand keys must be externally assigned L/R')
        for hand in hands.values():
            _boolean(hand.get('hand_present'),'hand_present')
            _boolean(hand.get('identity_resolved'),'identity_resolved')
            if not isinstance(hand.get('features'),dict) or set(hand['features'])-set(FEATURES):
                raise ValueError('Invalid feature object')
            for cell in hand['features'].values():
                _state_value(cell.get('state'))
                if cell.get('kind') not in ('current_observed','model_estimate','unknown'):
                    raise ValueError('Evidence kind must remain explicit')
                if cell.get('kind') == 'unknown' and cell.get('state') is not None:
                    raise ValueError('Unknown evidence may not carry a determined state')
                m = cell.get('measurement')
                if m is not None:
                    _number(m.get('value'),'measurement.value')
                    if m.get('kind') != 'current_observed' or not isinstance(m.get('unit'),str) or not m['unit']:
                        raise ValueError('Scalar differences require named current-observed measurements')
                for k in ('p_pos','p_neg','p_unknown'):
                    if k in cell and not 0 <= _number(cell[k],k) <= 1+1e-6:
                        raise ValueError('Invalid source-mapped mass')

def _cell(frame, hand, feature):
    h = frame.get('hands',{}).get(hand)
    if h is None:
        return {'state':None,'kind':'unknown','reason':'hand_record_missing','measurement':None}
    if not h['hand_present'] or not h['identity_resolved']:
        return {'state':None,'kind':'unknown','reason':'hand_not_observed' if not h['hand_present']
                else 'hand_identity_unresolved','measurement':None}
    return h['features'].get(feature,{'state':None,'kind':'unknown','reason':'feature_record_missing','measurement':None})

def _path(path, nq, nr):
    if not isinstance(path,list): raise ValueError('Frozen path must be an explicit list')
    unique = []; previous = None; duplicates = 0
    for pair in path:
        if not isinstance(pair,(list,tuple)) or len(pair) != 2 or any(type(x) is not int for x in pair):
            raise ValueError('Path pairs must contain integer query/reference indices')
        q,r = pair
        if not (0 <= q < nq and 0 <= r < nr): raise ValueError('Path index outside supplied frames')
        if previous is not None:
            dq,dr = q-previous[0],r-previous[1]
            if (dq,dr) == (0,0): duplicates += 1; continue
            if (dq,dr) not in ((1,0),(0,1),(1,1)):
                raise ValueError('Expected monotone adjacent DTW steps; no reordering or skipped indices')
        unique.append((q,r)); previous=(q,r)
    return unique,duplicates

def _insufficient_runs(cells, times):
    """Reference-sample runs, never silently presented as continuous duration."""
    runs=[]; current=[]
    for c in cells:
        index=c['reference_index']
        if c['required'] and c['classification_comparison'] != 'complete':
            if current and index != current[-1]+1:
                runs.append(current);current=[]
            current.append(index)
        elif current:
            runs.append(current);current=[]
    if current:runs.append(current)
    return [{'reference_indices':r,'sample_times_s':[times[i] for i in r],
             'start_sample_s':times[r[0]],'end_sample_s':times[r[-1]],
             'continuous_interval_claim':False} for r in runs]

def compare_references(query_frames, references):
    """Compare only on given DTW path; retain each reference as its own case.

Each required reference sample counts once even if many query frames align to
it. Every distinct path pair is retained. Partial coverage cannot become a
complete match by selecting the best pair/reference. Empty query is allowed.
"""
    _validate_frames(query_frames)
    if not isinstance(references,list) or not references:
        raise ValueError('At least one separate reference is required')
    ids=[r.get('reference_id') for r in references]
    if any(not isinstance(x,str) or not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError('Unique reference IDs required')
    out=[]
    for ref in references:
        frames=ref['frames']; _validate_frames(frames)
        if not frames: raise ValueError('A reference must have a fixed nonempty frame axis')
        path,duplicates=_path(ref['path'],len(query_frames),len(frames))
        required=ref['required']
        if set(required) != set(HANDS) or any(set(required[h]) != set(FEATURES) for h in HANDS):
            raise ValueError('Explicit required masks for both hands and every feature are mandatory')
        for h in HANDS:
            for f in FEATURES:
                if len(required[h][f]) != len(frames) or any(type(x) is not bool for x in required[h][f]):
                    raise ValueError('Required mask must match reference length with booleans')
        links={i:[] for i in range(len(frames))}
        for q,r in path:links[r].append(q)
        groups={}
        for h in HANDS:
            groups[h]={}
            for f in FEATURES:
                cells=[]
                for ri,rframe in enumerate(frames):
                    rc=_cell(rframe,h,f); pairs=[]
                    for qi in links[ri]:
                        qc=_cell(query_frames[qi],h,f)
                        known=qc['state'] is not None and rc['state'] is not None
                        qm,rm=qc.get('measurement'),rc.get('measurement')
                        measured=(qm is not None and rm is not None and qm['unit']==rm['unit'])
                        masses=(qc.get('p_pos') is not None and rc.get('p_pos') is not None
                                and qc['state'] is not None and rc['state'] is not None)
                        pairs.append({'query_index':qi,'query_time_s':query_frames[qi]['time_s'],
                                      'reference_index':ri,'reference_time_s':rframe['time_s'],
                                      'query_state':qc['state'],'reference_state':rc['state'],
                                      'query_kind':qc['kind'],'reference_kind':rc['kind'],
                                      'query_reason':qc.get('reason'),'reference_reason':rc.get('reason'),
                                      'classification_state_difference':int(qc['state']!=rc['state']) if known else None,
                                      'observed_value_abs_difference':abs(qm['value']-rm['value']) if measured else None,
                                      'observed_value_unit':qm['unit'] if measured else None,
                                      'model_positive_mass_abs_difference':abs(qc['p_pos']-rc['p_pos']) if masses else None})
                    knownpairs=[p for p in pairs if p['classification_state_difference'] is not None]
                    measuredpairs=[p for p in pairs if p['observed_value_abs_difference'] is not None]
                    complete=bool(pairs) and len(knownpairs)==len(pairs)
                    kinds=[(p['query_kind'],p['reference_kind']) for p in knownpairs]
                    relations={p['classification_state_difference'] for p in knownpairs}
                    cells.append({'reference_index':ri,'reference_time_s':rframe['time_s'],
                                  'required':required[h][f][ri], 'aligned_pair_count':len(pairs),
                                  'known_pair_count':len(knownpairs),
                                  'measured_pair_count':len(measuredpairs),
                                  'complete_observed_value_comparison':bool(pairs) and len(measuredpairs)==len(pairs),
                                  'classification_comparison':'complete' if complete else 'partial' if knownpairs else 'unavailable',
                                  'state_relation':'different' if relations=={1} else 'same' if relations=={0} else 'mixed' if relations else None,
                                  'state_relation_scope':'all_aligned_pairs' if complete else 'known_pairs_only_not_whole_required_cell',
                                  'complete_observed_only':complete and all(a==b=='current_observed' for a,b in kinds),
                                  'complete_involving_estimate':complete and any('model_estimate' in k for k in kinds),
                                  'pairs':pairs,'educational_verdict':None})
                req=[c for c in cells if c['required']]
                count=len(req);complete=sum(c['classification_comparison']=='complete' for c in req)
                groups[h][f]={'required_sample_count':count,'complete_comparison_count':complete,
                              'partial_comparison_count':sum(c['classification_comparison']=='partial' for c in req),
                              'insufficient_count':count-complete,
                              'complete_observed_only_count':sum(c['complete_observed_only'] for c in req),
                              'complete_involving_estimate_count':sum(c['complete_involving_estimate'] for c in req),
                              'complete_observed_value_comparison_count':sum(c['complete_observed_value_comparison'] for c in req),
                              'complete_comparison_fraction':complete/count if count else None,
                              'sample_count_not_duration_fraction':True,
                              'insufficient_sample_runs':_insufficient_runs(cells,[x['time_s'] for x in frames]),
                              'cells':cells,'educational_verdict':None}
        out.append({'reference_id':ref['reference_id'],'path_original':copy.deepcopy(ref['path']),
                    'path_unique_pairs':[list(x) for x in path],'duplicate_pairs_ignored_for_count':duplicates,
                    'path_recomputed':False,'reference_frames':len(frames),'query_frames':len(query_frames),
                    'raw_reference_frames':copy.deepcopy(frames),
                    'required_mask_preserved':copy.deepcopy(required),'hands':groups,
                    'educational_verdict':None})
    return {'scope':'research_evidence_accounting_only','references':out,
            'raw_query_frames':copy.deepcopy(query_frames),
            'bridge_protocol_sha256':_sha(U/'BRIDGE_PROTOCOL.json'),
            'reference_splicing_or_best_selection':False,'educational_verdict':None,
            'educational_feedback_enabled':False,'model_outputs_are_ground_truth':False,
            'existing_alignment_modified':False,'probabilities_are_calibrated':False}
