import sys
sys.dont_write_bytecode=True
import argparse, copy, gzip, hashlib, json, os, time, traceback
from pathlib import Path
UNIT=Path(__file__).resolve().parent
OPS=UNIT.parent
PREV=OPS/'word100_independent_hand_evidence_20261101'
OLD=OPS/'word100_description_evidence_feedback_20261101'
PLAN=OPS/'word100_action_feedback_plan_20261101'

def read(p):
    p=Path(p);return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_text('utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise FileExistsError(p)
    b=json.dumps(v,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf-8')
    p.write_bytes(gzip.compress(b,mtime=0) if p.suffix=='.gz' else b)

def prepare():
    if (UNIT/'INPUTS.json').exists():raise RuntimeError('existing input manifest; inspect and reuse rather than recreate')
    saved=read(PREV/'INPUTS.json');old=read(OLD/'REPORT_DATA.json');manifest=read(PREV/'RESULT_MANIFEST.json')
    selected={'P036','P048','P066','P072'};records={r['id']:r for r in manifest['results']};pins={}
    def pin(p,h=None):
        p=str(Path(p).resolve());got=sha(p)
        if h:assert got==h,p
        pins[p]=got;return {'path':p,'sha256':got}
    for p in (PLAN/'PLAN.json',PLAN/'FINAL_PLAN_REVIEW.json',PREV/'INPUTS.json',PREV/'RESULT_MANIFEST.json',PREV/'EVALUATION.json',PREV/'FINAL_VERIFICATION.json',OLD/'REPORT_DATA.json',OLD/'SUMMARY.json',UNIT/'review/WINDOW_SELECTION.json',UNIT/'review/WINDOW_ACK.json'):
        pin(p)
    # Retain previously protected production/input/source hashes without rerunning any producer.
    for p,h in read(PREV/'FROZEN.json')['input_pins'].items():pin(p,h)
    natural=[]
    for t in saved['natural']:
        if t['query'] not in selected:continue
        ci=next(i for i,c in enumerate(old['cases']) if c['query']==t['query'] and c['reference']==t['reference'])
        c=old['cases'][ci];si=next(i for i,s in enumerate(c['stages']) if s['id']==t['parent']['segment_id']);s=c['stages'][si]
        assert s['reference_core']==t['reference_range']
        assert [min(s['query_frames']),max(s['query_frames'])]==t['query_range']
        assert s['path']==t['baseline_full_path']
        r=records[t['id']];pin(r['path'],r['sha256'])
        natural.append({'id':t['id'],'kind':'natural','query':t['query'],'reference':t['reference'],'label':c['label'],
            'query_range':t['query_range'],'reference_range':t['reference_range'],'parent_id':s['id'],
            'parent_physical_description':s['reference_rgb_description'],'hand_record':r,
            'legacy':{'path':str(OLD/'REPORT_DATA.json'),'sha256':sha(OLD/'REPORT_DATA.json'),'json_pointer':f'/cases/{ci}/stages/{si}','case_index':ci,'stage_index':si},
            'source_card_count':len(s['source_cards'])})
    technical=[]
    for t in saved['technical']:
        r=records[t['id']];pin(r['path'],r['sha256'])
        technical.append({'id':t['id'],'kind':'technical','hand_record':r,'condition':t['edit']['condition'],'hand':t['edit']['hand']})
    assert len(natural)==39 and len(technical)==810
    assert sum(bool(t['source_card_count']) for t in natural)==12 and sum(t['source_card_count'] for t in natural)==30
    assert len({(t['query'],t['reference']) for t in natural})==12
    write(UNIT/'INPUTS.json',{'natural':natural,'technical':technical,'rows':{k:v for k,v in saved['rows'].items()},'pins':pins,
        'counts':{'cases':4,'pairs':12,'parents':39,'linked_parents':12,'physical_only':27,'source_cards':30,'technical':810},
        'all_previous_reference_and_frame_scopes_preserved':True,'new_inference':0,'new_dtw':0})
    print(json.dumps({'natural':39,'technical':810,'pins':len(pins)}))

def freeze():
    inp=read(UNIT/'INPUTS.json');pins=dict(inp['pins'])
    for p,h in read(UNIT/'review/WINDOW_SELECTION.json')['source_pins'].items():pins[p]=h
    rgb=read(UNIT/'review/RGB_MANIFEST.json')
    assert rgb['selection_sha256']==sha(UNIT/'review/WINDOW_SELECTION.json')
    assert rgb['ack_sha256']==sha(UNIT/'review/WINDOW_ACK.json')
    for frame in rgb['full_frames']:
        pins[frame['image']]=frame['image_sha256']
        pins[frame['source']]=frame['source_sha256']
    for page in rgb['pages']:pins[page['file']]=page['sha256']
    for p,h in pins.items():assert sha(p)==h,p
    required=['AUTHORIZATION.json','INPUTS.json','pipeline.py','case_evidence.py','selftest.py','SELFTEST.json',
              'review/WINDOW_SELECTION.json','review/WINDOW_ACK.json','review/RGB_MANIFEST.json','review/VISUAL_REVIEW.json',
              'review/PREFLIGHT_METHOD_REVIEW.json','review/PREEXECUTION_REVIEW.json','review/verify_casebook.py']
    assert str(read(UNIT/'SELFTEST.json')['status']).startswith('PASS')
    assert str(read(UNIT/'review/PREEXECUTION_REVIEW.json')['status']).startswith('PASS')
    write(UNIT/'FROZEN.json',{'source_pins':pins,'code_and_review_pins':{str(UNIT/p):sha(UNIT/p) for p in required},
        'time_unix':time.time(),'scope':'saved evidence summary only:39natural+810technical. RGB reviews observational, not target labels.'})
    print('FROZEN')

def visual_for(t,review):
    matches=[]
    for i,w in enumerate(review['windows']):
        if (w['query'],w['reference'],w['parent_id'])==(t['query'],t['reference'],t['parent_id']):
            matches.append({'source':{'path':str(UNIT/'review/VISUAL_REVIEW.json'),'sha256':sha(UNIT/'review/VISUAL_REVIEW.json'),'json_pointer':f'/windows/{i}'},'review':w})
    if not matches:return None
    return {'status':'REVIEWED','source':{'path':str(UNIT/'review/VISUAL_REVIEW.json'),'sha256':sha(UNIT/'review/VISUAL_REVIEW.json'),'json_pointer':'/windows'},
        'windows':[m['review'] for m in matches],'observations':[m for m in matches],
        'observation_not_training_truth':True,'scope_is_only_named_pair_parent_and_observed_frames':True}

def run():
    import case_evidence
    if (UNIT/'RUNNING.json').exists() or (UNIT/'RESULT_MANIFEST.json').exists():raise RuntimeError('existing lock or completed results; inspect first')
    frozen=read(UNIT/'FROZEN.json');pins={**frozen['source_pins'],**frozen['code_and_review_pins']}
    for p,h in pins.items():assert sha(p)==h,p
    write(UNIT/'RUNNING.json',{'pid':os.getpid(),'started':time.time()})
    inp=read(UNIT/'INPUTS.json');old=read(OLD/'REPORT_DATA.json');visual=read(UNIT/'review/VISUAL_REVIEW.json')
    results=[];case_rows=[];started=time.perf_counter();compute=0.
    for t in inp['natural']+inp['technical']:
        r=read(t['hand_record']['path']);description=correspondence=vr=None
        if t['kind']=='natural':
            stage=old['cases'][t['legacy']['case_index']]['stages'][t['legacy']['stage_index']]
            description={'source':t['legacy'],'cards':stage['source_cards'],
                'legacy_decisions':{k:{f:v.get(f) for f in ('saved_metric','saved_decision','saved_range','feedback')} for k,v in stage['groups'].items()},
                'physical_description':stage['reference_rgb_description'],'old_trace_pointer':t['legacy']['json_pointer']+'/groups'}
            correspondence={'source':t['legacy'],'details':stage['alignment'],'saved_path':stage['path'],
                'actual_stage_correctness_verified':False,'new_path_computed':False}
            vr=visual_for(t,visual)
        tick=time.perf_counter();summary=case_evidence.summarize(r,description=description,correspondence=correspondence,visual_review=vr);compute+=time.perf_counter()-tick
        result={'id':t['id'],'kind':t['kind'],'source':t['hand_record'],'summary':summary,
            'frozen_sha256':sha(UNIT/'FROZEN.json'),'position_sources':r.get('position_sources',[]),'shape_sources':r.get('shape_sources',[]),
            'shape_guide_link':r.get('shape_guide_link'),'new_inference':False,'new_dtw':False,'new_coordinate_comparison':False,
            'educational_verdict':None,'training_target':False}
        if t['kind']=='natural':result.update(query=t['query'],reference=t['reference'],parent_id=t['parent_id'],query_range=t['query_range'],reference_range=t['reference_range'],legacy_source=t['legacy'])
        dest=UNIT/'results'/(t['id']+'.json.gz');write(dest,result)
        link={'id':t['id'],'kind':t['kind'],'path':str(dest),'sha256':sha(dest)};results.append(link)
        if t['kind']=='natural':case_rows.append(dict(link,query=t['query'],reference=t['reference'],parent_id=t['parent_id'],label=t['label'],visual_reviewed=vr is not None))
    cases=[]
    for q in sorted({r['query'] for r in case_rows}):
        rows=[r for r in case_rows if r['query']==q];refs=sorted({r['reference'] for r in rows});assert len(refs)==3
        cases.append({'query':q,'label':rows[0]['label'],'references':refs,'parents':rows,'all_three_references_retained':True})
    for p,h in pins.items():assert sha(p)==h,p
    write(UNIT/'CASEBOOK.json',{'cases':cases,'source_catalog':{'path':str(OLD/'REPORT_DATA.json'),'sha256':sha(OLD/'REPORT_DATA.json'),'json_pointer':'/source_clause_catalog'},
        'new_review':{'path':str(UNIT/'review/VISUAL_REVIEW.json'),'sha256':sha(UNIT/'review/VISUAL_REVIEW.json')},
        'historical_development_not_new_generalization':True,'educational_accuracy':None})
    write(UNIT/'RESULT_MANIFEST.json',{'status':'COMPLETE_PENDING_VERIFICATION','results':results,'natural':39,'technical':810,
        'compute_seconds':compute,'wall_seconds':time.perf_counter()-started,'end_to_end_user_latency':None,
        'new_model_inference':0,'new_training':0,'new_dtw':0,'new_coordinate_comparisons':0})
    (UNIT/'RUNNING.json').rename(UNIT/'RUN_FINISHED.json')
    print(json.dumps({'records':len(results),'cases':len(cases),'compute_seconds':compute,'wall_seconds':time.perf_counter()-started}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','freeze','run']);arg=parser.parse_args()
    try:globals()[arg.stage]()
    except Exception:
        p=UNIT/f'FIRST_ERROR_{arg.stage}.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf-8')
        raise
