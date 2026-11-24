"""Separate saved-result decoder. Never imports or reruns audit_presence.py.

This is a second implementation by the audit author; independent human/agent
review remains distinct. Verifies arithmetic and provenance, not RGB truth.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json

W = Path(__file__).resolve().parent.parent
KEYS = ('R_down','R_hold','L_down','L_hold')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def need(ok, text):
    if not ok: raise ValueError(text)


def frames_of(item, allowed):
    start,end=item['first_frame'],item['last_frame']
    need(type(start) is int and type(end) is int and start<=end,'strict interval frame type/order')
    represented=set(range(start,end+1))
    need(represented and represented<=set(allowed),'interval inside original frames')
    return represented


def raw_semantics(row, source):
    """Returns eligibility and contradiction identities using frame-set logic."""
    source_frames=source['frame_ids'];errors=[];roi={};blocked={}
    need(row['alias']==source['alias'] and row['actual_image_viewed'] is True and row['image_sha256']==source['image_sha256'],'image identity and actual view')
    for key in KEYS: blocked[key]=set()
    identity={h:set() for h in 'RL'}
    for item in row['unknown_intervals']:
        time=frames_of(item,source_frames)
        need(item['hand'] in ('R','L','both') and item['component'] in ('translation','identity','shape','orientation'),'unknown hand/component')
        hands='RL' if item['hand']=='both' else item['hand']
        if item['component']=='translation':
            kinds=item['affected_predicates']
            need(type(kinds) is list and kinds and len(set(kinds))==len(kinds) and set(kinds)<={'down','hold'},'affected predicates explicit')
        else:kinds=('down','hold') if item['component']=='identity' else ()
        for h in hands:
            if item['component']=='identity':identity[h]|=time
            for p in kinds:blocked[h+'_'+p]|=time
    for event in row['events']:
        span=frames_of(event,source_frames)
        need(event['hand'] in ('R','L','both') and event['kind'] in ('onset','offset','reversal','rotation','shape_change','hand_crossing'),'event type')
        need(event['evidence'] in ('direct','uncertain') and type(event['bounded_transition']) is bool,'event evidence')
        need(not event['bounded_transition'] or (event['evidence']=='direct' and source_frames[0] not in span and source_frames[-1] not in span),'bounded event context')
    def error(h,p,reason):errors.append((h,p,reason))
    for h in 'RL':
        full=row['full13_position_observable'][h];full_roi=row['full13_roi'][h]
        need((full is None or type(full) is bool) and (full_roi is None or type(full_roi) is bool),'observation flags bool/null')
        for p in ('down','hold'):
            key=h+'_'+p;value=row['presence'][h][p];segments=row['segments'][h][p]
            need(value is None or type(value) is int and value in (0,1),'presence int0/int1/null')
            need(type(segments) is list,'segments list')
            bounds=[]
            for segment in segments:
                span=frames_of(segment,source_frames)
                need(segment['in_roi'] is None or type(segment['in_roi']) is bool,'segment ROI')
                bounds.append((segment['first_frame'],segment['last_frame']))
                if len(span)<{'down':2,'hold':3}[p]:error(h,p,'positive_segment_too_short')
                # Retain multiplicity: each independently stated relevant unknown is a contradiction.
                for item in row['unknown_intervals']:
                    relevant=item['hand'] in (h,'both') and (item['component']=='identity' or item['component']=='translation' and p in item['affected_predicates'])
                    if relevant and span & frames_of(item,source_frames):error(h,p,'positive_witness_overlaps_relevant_unknown')
                if full_roi is True and segment['in_roi'] is False:error(h,p,'full13_roi_contradicts_segment_roi')
            if bounds!=sorted(bounds):error(h,p,'positive_segments_not_chronological')
            if value==1 and len(segments)==0:error(h,p,'positive_without_direct_witness')
            if value!=1 and len(segments)>0:error(h,p,'nonpositive_with_positive_witness')
            if value==0:
                if full is not True:error(h,p,'negative_without_full13_observation')
                if blocked[key]:error(h,p,'negative_with_relevant_unknown')
            roi[key]=(any(s['in_roi'] is True for s in segments) if value==1 else value==0 and full is True and full_roi is True)
        for d in row['segments'][h]['down']:
            for hold in row['segments'][h]['hold']:
                if len(frames_of(d,source_frames)&frames_of(hold,source_frames))>1:error(h,None,'down_hold_conflicting_observed_intervals')
        if full is True and identity[h]:error(h,None,'full13_identity_claim_contradicts_unknown')
    return roi,errors


def tally(items):
    c=Counter({name:0 for name in ('all','both_known','both_unknown','known_unknown_mismatch','known_conflicts','agreed_positive','agreed_negative','a_eligible','b_eligible','both_eligible','eligible_agreed_known','eligible_agreed_positive','eligible_agreed_negative')})
    for item in items:
        a,b,ea,eb=item['a'],item['b'],item['ea'],item['eb'];c['all']+=1
        c['a_eligible']+=ea;c['b_eligible']+=eb;c['both_eligible']+=ea and eb
        if a is None and b is None:c['both_unknown']+=1
        elif a is None or b is None:c['known_unknown_mismatch']+=1
        else:
            c['both_known']+=1
            if a!=b:c['known_conflicts']+=1
            else:
                label='positive' if a else 'negative';c['agreed_'+label]+=1
                if ea and eb:c['eligible_agreed_known']+=1;c['eligible_agreed_'+label]+=1
    out=dict(c)
    for name,numer,denom in [('known_conflict_rate','known_conflicts','both_known'),('known_unknown_mismatch_rate','known_unknown_mismatch','all'),('both_known_fraction','both_known','all'),('eligible_agreed_fraction','eligible_agreed_known','both_eligible'),('eligible_agreed_fraction_all','eligible_agreed_known','all')]:out[name]=c[numer]/c[denom] if c[denom] else None
    return out


def independent_expected(manifest, reviews, gates):
    sources=manifest['rows'];need(len(sources)==24 and len({s['sample_id'] for s in sources})==24,'source scope')
    maps=[]
    for review in reviews:
        need(review['status']=='COMPLETE' and len(review['rows'])==24,'complete24 review')
        m={r['sample_id']:r for r in review['rows']};need(set(m)=={s['sample_id'] for s in sources} and len(m)==24,'unique exact review scope');maps.append(m)
    need(len(maps)==2,'two reviewers')
    items={key:[] for key in KEYS};failures=[];rows=[];raw_patterns={h:[] for h in 'RL'}
    for source in sources:
        sid=source['sample_id'];frames=source['frame_ids'];times=source['timestamps_us']
        need(len(frames)==len(times)==13 and all(type(x) is int for x in frames+times),'13 integer source frames/times')
        need(frames==list(range(frames[0],frames[0]+13)) and all(b>a for a,b in zip(times,times[1:])),'native chronology')
        need(source['center_frame']==frames[6] and source['center_timestamp_us']==times[6] and source['role']=='fit','centre and fit identity')
        pair=[m[sid] for m in maps];validated=[raw_semantics(r,source) for r in pair]
        for ix,(_,errors) in enumerate(validated): failures.extend((sid,('A','B')[ix],*error) for error in errors)
        row=dict(sample_id=sid,alias=source['alias'],frame_ids=frames,timestamps_us=times,predicates={},ordered_evidence=[])
        for h in 'RL':raw_patterns[h].append(dict(sample_id=sid,a=[pair[0]['presence'][h][p] for p in ('down','hold')],b=[pair[1]['presence'][h][p] for p in ('down','hold')]))
        for key in KEYS:
            h,p=key.split('_');a,b=[r['presence'][h][p] for r in pair];ea,eb=[r[0][key] for r in validated]
            items[key].append(dict(a=a,b=b,ea=ea,eb=eb,source=source))
            same=a is not None and a==b;invalid=any(eh==h and ep in (None,p) for _,errors in validated for eh,ep,_ in errors)
            row['predicates'][key]=dict(a=a,b=b,a_eligible=ea,b_eligible=eb,agreed_value=a if same else None,model_label=a if same and ea and eb and not invalid else None,unknown_is_not_negative=True)
            for ix,record in enumerate(pair):
                for segment in record['segments'][h][p]:
                    row['ordered_evidence'].append(dict(reviewer=('A','B')[ix],hand=h,predicate=p,first_frame=segment['first_frame'],last_frame=segment['last_frame'],first_timestamp_us=times[frames.index(segment['first_frame'])],last_timestamp_us=times[frames.index(segment['last_frame'])],in_roi=segment['in_roi'],note=segment.get('note','')))
        row['ordered_evidence'].sort(key=lambda x:(x['first_frame'],x['last_frame'],x['reviewer'],x['hand'],x['predicate']));rows.append(row)
    metrics={key:tally(group) for key,group in items.items()}
    strata={name:{key:tally([i for i in group if (i['source']['strata']==['hash_ordinary'])==(name=='ordinary')]) for key,group in items.items()} for name in ('ordinary','challenge')}
    def group_support(group,value,model=False):
        sources=[i['source'] for i in group if i['a']==value and i['b']==value and (not model or i['ea'] and i['eb'])]
        return {name:sorted({s[name] for s in sources}) for name in ('sample_id','clip_id','person','word_id')}
    classes={key:{scope:{str(value):group_support(group,value,scope=='eligible') for value in (0,1)} for scope in ('physical','eligible')} for key,group in items.items()}
    families={p:group_support(items['R_'+p]+items['L_'+p],1) for p in ('down','hold')}
    patterns={}
    for h,values in raw_patterns.items():
        complete=[r for r in values if None not in r['a']+r['b']];conflict=[r for r in complete if r['a']!=r['b']]
        patterns[h]=dict(all=24,both_complete=len(complete),conflicts=len(conflict),conflict_rate=len(conflict)/len(complete) if complete else None,conflict_rows=conflict,raw_rows=values)
    checks={'annotation_contract':not failures}
    for key,m in metrics.items():
        tests={'known_conflict':m['both_known']>0 and m['known_conflicts']/m['both_known']<=gates['max_raw_known_conflict_rate_each_predicate'],
               'unknown_mismatch':m['known_unknown_mismatch']/24<=gates['max_raw_known_unknown_mismatch_rate_each_predicate'],
               'known_coverage':m['both_known']>=gates['min_both_known_each_predicate'],
               'ordinary_coverage':strata['ordinary'][key]['both_known']>=gates['min_both_known_ordinary_each_predicate'],
               'both_classes':all(m['agreed_'+v]>=gates['min_agreed_positive_and_negative_each_predicate'] for v in ('positive','negative')),
               'eligible_agreement_fraction':m['both_eligible']>0 and m['eligible_agreed_known']/m['both_eligible']>=gates['min_model_eligible_agreed_known_fraction_each_predicate'],
               'eligible_coverage':m['eligible_agreed_known']>=gates['min_model_eligible_agreed_known_each_predicate'],
               'eligible_both_classes':all(m['eligible_agreed_'+v]>=gates['min_model_eligible_agreed_positive_and_negative_each_predicate'] for v in ('positive','negative'))}
        checks.update({key+'_'+name:value for name,value in tests.items()})
    for h,m in patterns.items():
        checks[h+'_two_bit_coverage']=m['both_complete']>=gates['min_two_bit_both_known_each_hand']
        checks[h+'_two_bit_conflict']=m['both_complete']>0 and m['conflicts']/m['both_complete']<=gates['max_two_bit_hand_pattern_conflict_rate']
    for p,s in families.items():checks[p+'_positive_family_support']=len(s['sample_id'])>=gates['min_agreed_'+p+'_positive_windows'] and len(s['person'])>=gates['min_people_each_positive_family']
    return dict(status='PASS_PRESENCE_PILOT_ONLY_BROAD_LABELS_NEXT' if all(checks.values()) else 'REPAIR_PRESENCE_PILOT_BEFORE_BROAD_LABELS_OR_TRAINING',checks=checks,metrics=metrics,strata=strata,pattern_metrics=patterns,family_support=families,class_support=classes,rows=rows,contract_failure_identities=failures)


def compare(saved, expected):
    bad=[]
    for key,value in expected.items():
        if key=='contract_failure_identities':
            actual=[(r['sample_id'],r['reviewer'],r['hand'],r['predicate'],r['reason']) for r in saved['contract_failures']]
            same=Counter(actual)==Counter(value)
        else:same=saved.get(key)==value
        if not same:bad.append(key)
    return bad


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--audit',type=Path,default=W/'PRESENCE_PILOT_AUDIT.json');parser.add_argument('--output',type=Path,default=W/'review/PRESENCE_VERIFICATION.json');args=parser.parse_args()
    need(not args.output.exists(),'Verification already saved; never overwrite')
    checks=[]
    def check(name,value):checks.append(dict(name=name,passed=bool(value)))
    saved=read(args.audit);manifest=read(W/'PILOT_MANIFEST.json');gates=read(W/'PILOT_GATES.json')
    reviews=[read(W/'review'/name) for name in ('PRESENCE_A.json','PRESENCE_B.json')]
    for name,review in zip(('PRESENCE_A.json','PRESENCE_B.json'),reviews):
        need(review.get('rubric_sha256')==sha(W/'RUBRIC.json'),'Raw reviewer rubric hash mismatch: '+name)
    expected=independent_expected(manifest,reviews,gates);bad=compare(saved,expected)
    for key in expected:check('separate raw decoder: '+key,key not in bad)
    for path,digest in saved['source_pins'].items():check('saved source hash: '+path,sha(path)==digest)
    for name in ('PILOT_FROZEN.json','AUDIT_CODE_FROZEN.json'):
        path=W/name;check('freeze document pinned: '+name,saved['source_pins'].get(str(path))==sha(path))
        for path,digest in read(path)['pins'].items():check('freeze retained and current: '+path,saved['source_pins'].get(path)==digest==sha(path))
    for name in ('PRESENCE_A.json','PRESENCE_B.json'):
        path=W/'review'/name;check('raw label pinned: '+name,saved['source_pins'].get(str(path))==sha(path))
    check('executed code accurately pinned',saved['source_pins'].get(saved['audit_code'])==sha(saved['audit_code']))
    check('original dataset unchanged',sha(manifest['source'])==manifest['source_sha256']=='f2ef5dc927ae55c30794a4a600e2cbe8f0cea3f93b6026d50ff7c811b7a61d61')
    original=read(manifest['source']);by={r['sample_id']:r for r in original['rows']};blind={r['sample_id']:r for r in read(W/'BLIND_PILOT.json')['rows']}
    prior=W.parent/'word100_center_motion_pilot_20261101'
    check('same pilot selection as failed centre study',sha(W/'PILOT_MANIFEST.json')==sha(prior/'PILOT_MANIFEST.json'))
    check('same blind source metadata',sha(W/'BLIND_PILOT.json')==sha(prior/'BLIND_PILOT.json'))
    for row in manifest['rows']:
        sid=row['sample_id'];check(sid+' montage hash',sha(row['image'])==row['image_sha256'])
        check(sid+' blind fields',all(row[k]==v for k,v in blind[sid].items()))
        for key in ('clip_id','word_id','person','role','frame_ids','timestamps_us','center_frame','center_timestamp_us','image'):check(sid+' source '+key,row[key]==by[sid][key])
    for path,digest in {r['input']:r['input_sha256'] for r in original['rows']}.items():check('original prepared input: '+path,sha(path)==digest)
    for path,digest in original['pins'].items():check('original protected source: '+path,sha(path)==digest)
    check('24/48/96 denominators',(saved['all_windows'],saved['all_hand_windows'],saved['all_predicate_cells'])==(24,48,96))
    check('no training permission or accuracy claim',saved['training_allowed'] is False and saved['accuracy_or_performance_claim'] is None and saved['semantic_or_educational_truth'] is None)
    check('explicit occurrence versus ordering distinction',saved['presence_is_not_ordering'] is True)
    check('all original failures remain failed',saved['original_failed_protocols_unchanged'] is True and read(prior/'PILOT_AUDIT.json')['status']=='REPAIR_PILOT_BEFORE_BROAD_LABELS_OR_TRAINING')
    report=dict(status='PASS_SEPARATE_PRESENCE_DECODER_AND_IMMUTABILITY' if all(c['passed'] for c in checks) else 'FAIL_PRESENCE_VERIFICATION',pilot_status=saved['status'],checks_count=len(checks),checks=checks,failures=[c for c in checks if not c['passed']],
                author_relation='Same author, separately implemented decoder; independent reviewer assessment remains separate.',
                source_pins={str(p):sha(p) for p in (args.audit,Path(__file__),W/'review/PRESENCE_A.json',W/'review/PRESENCE_B.json')},
                labels_or_originals_modified=False,audit_writer_rerun=False,training_runs=0,inference_runs=0,
                scope='Raw-value/interval eligibility, gate arithmetic, saved evidence bounds/PTS, frozen source/input hashes. No independent RGB reannotation or semantic accuracy claim.')
    with args.output.open('x',encoding='utf8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:report[k] for k in ('status','pilot_status','checks_count','failures')},ensure_ascii=False))


if __name__=='__main__':main()
