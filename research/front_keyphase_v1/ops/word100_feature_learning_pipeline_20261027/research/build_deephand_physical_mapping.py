"""Source-diagram-derived partial physical labels. Does not alter original classes."""
from pathlib import Path
import hashlib,json,datetime,collections

HERE=Path(__file__).resolve().parent
U=HERE.parent
M={
 '1':((0,0,1),'Four non-thumb fingers drawn closed; thumb extended. Name 1 does not mean index-only.'),
 '2':((1,0,0),'Straight index and thumb; other three non-thumb fingers closed. Thumb excluded from target.'),
 '3':((0,0,0),'Index and middle extended; ring and little closed.'),
 '3_hook':((0,0,None),'Two raised hooked fingers; neither all-four extension nor index-only; full closure unclear.'),
 '4':((0,1,0),'Four non-thumb fingers extended and spread; thumb state irrelevant.'),
 '5':((0,1,0),'Four non-thumb fingers extended and spread; thumb extended but not target.'),
 'a':((0,0,1),'Four non-thumb fingers visibly closed toward palm.'),
 'ae':((0,0,None),'Broad curved/claw form; not broadly extended; degree of full closure not established.'),
 'b':((0,1,0),'Four non-thumb fingers straight together; thumb state irrelevant.'),
 'b_nothumb':((0,1,0),'Four non-thumb fingers straight together; thumb state irrelevant.'),
 'b_thumb':((0,1,0),'Four non-thumb fingers straight together; thumb extended but not target.'),
 'by':((0,None,0),'Four grouped fingers with substantial base flexion; broadly-open convention ambiguous, not closed toward palm.'),
 'c':((0,0,0),'Large C-shaped curved fingers with visible open gap; neither straight-four nor closed-four.'),
 'cbaby':((0,0,None),'Small curved opening with hooked index; closure of all four not unambiguous.'),
 'd':((1,0,0),'Straight index with remaining three non-thumb fingers folded; thumb contact differs but target ignores thumb.'),
 'e':((0,0,None),'Curled grouped fingers; whether this belongs to full closure versus hooked form is underdefined.'),
 'f':((0,0,0),'Index bends toward thumb while other non-thumb fingers remain raised.'),
 'f_open':((0,0,0),'Index curved near thumb; other non-thumb fingers raised. Not all-four extension.'),
 'fly':((0,0,0),'Index and little extended, middle and ring folded.'),
 'fly_nothumb':((0,0,0),'Index and little extended, middle and ring folded; thumb ignored.'),
 'g':((1,0,0),'Single straight index projects sideways; other three non-thumb fingers folded.'),
 'h':((0,0,0),'Index and middle straight together; not index-only.'),
 'h_hook':((0,0,None),'Two hooked fingers; full closure ambiguous.'),
 'h_thumb':((0,0,0),'Index and middle extended together; thumb ignored.'),
 'i':((0,0,0),'Little finger extended; remaining non-thumb fingers folded.'),
 'index':((1,0,0),'Single straight index; other three non-thumb fingers folded.'),
 'index_flex':((None,0,0),'Index base flexion is explicit in source variant; prior broadly-extended definition does not specify this boundary.'),
 'index_hook':((0,0,None),'Hooked index; not straight index-only, but complete four-finger closure not established.'),
 'ital':((0,None,None),'Grouped converging fingers; substantial base flexion and thumb-contact geometry underdefine broad extension/full closure.'),
 'ital_nothumb':((0,None,0),'Grouped projecting fingers with strong base flexion; not palm-closed, broad-open classification left unresolved.'),
 'ital_open':((0,None,0),'Grouped fingers projecting from substantially bent base joints; broad-open versus bent-open boundary unresolved.'),
 'ital_thumb':((0,None,0),'Grouped projecting fingers with strong base flexion; thumb ignored.'),
 'jesus':((0,0,0),'Multiple extended non-thumb fingers; not all four and not index-only.'),
 'k':((0,0,0),'Index and middle extended in different directions; thumb ignored.'),
 'l_hook':((0,0,None),'Index hooked with thumb extended; all-four full-closure status not unambiguous.'),
 'o':((0,0,None),'Round grouped fingertip contact; O/contact geometry is not equated with full closure.'),
 'obaby':((0,0,None),'Small round precision-contact form; full closure not assumed.'),
 'pincet':((None,0,None),'Precision pinch side view; source icon alone insufficient to decide index straightness or all-four closure.'),
 'r':((0,0,0),'Two extended crossed fingers; normal crossing not called an error.'),
 's':((0,0,1),'Four non-thumb fingers visibly closed toward palm; thumb ignored.'),
 'v':((0,0,0),'Index and middle extended and spread.'),
 'v_flex':((0,0,0),'Two projecting fingers with base flexion; not index-only, open-four or closed-four.'),
 'w':((0,0,0),'Three non-thumb fingers extended; little finger folded.'),
 'write':((None,0,None),'Compact precision grasp; index and full-closure boundary ambiguous from source icon.'),
 'y':((0,0,0),'Little finger extended and other three folded; thumb ignored.')
}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    target=HERE/'DEEPHAND_PHYSICAL_PARTIAL_MAPPING.json'
    if target.exists():raise RuntimeError('First mapping already exists; preserve rather than overwrite')
    chart_path=HERE/'taxonomy_source/CHART_MANIFEST.json'
    chart=json.loads(chart_path.read_text(encoding='utf-8'))
    exp=json.loads((U/'EXPERIMENT.json').read_text(encoding='utf-8-sig'))
    rubric=U.parent/'word100_shared_feature_bundle_20261027/RUBRIC.json'
    assert set(M)==set(exp['classes'])==set(c['class_name']for c in chart['classes'])
    features=['index_only','open_four','four_closed']
    entries=[]
    for c in chart['classes']:
        vals,note=M[c['class_name']]
        assert sha(Path(c['file']))==c['sha256']
        entries.append({**c,**dict(zip(features,vals)),'basis':'Visible author-linked taxonomy icon, not class-name inference','interpretation_note':note,'new_frame_manual_annotation':False})
    result={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'FROZEN_SOURCE_DERIVED_PARTIAL_MAPPING_BEFORE_THREE_FEATURE_TRAINING_OR_TEST_REVIEW','reviewer':'Codex source audit; diagram interpretation is not new independent human frame truth','source':chart['author_page'],'source_page_sha256':chart['author_page_sha256'],'chart_manifest_sha256':sha(chart_path),'original_rubric':str(rubric),'original_rubric_sha256':sha(rubric),'original_class_contract_sha256':sha(U/'EXPERIMENT.json'),'source_taxonomy':'Author adopts Danish lexicon team taxonomy; displayed icons credited to Dictionary of New Zealand Sign Language. DeepHand observes45 of60 model classes.','target_definitions':{'index_only':'Index clearly extended, middle/ring/little folded; thumb unconstrained. Matches previous RGB rubric.','open_four':'Four non-thumb fingers broadly extended together or spread, natural slight bend allowed. Matches previous RGB rubric.','four_closed':'All four non-thumb fingers clearly closed toward palm; thumb unjudged. New physical feature, NOT asserted equivalent to a whole-hand fist.'},'values':{'1':'Source diagram clearly supports feature','0':'Source diagram clearly supports absence','null':'Diagram or target boundary insufficient; do not convert to0 or derive by complement'},'null_policy':'Per-feature unknown; other features may remain known. No class or source row deleted. Unknown-target cells excluded only from that target supervised loss with coverage reported. Original45 labels retained.','relation_to_ground_truth':'Targets are inherited author-manual45-class labels combined with explicitly interpreted source diagrams. They are not a newly human-reviewed3-target per-frame dataset, hidden-finger ground truth, KSL lexical equivalence, or educational correctness labels. Class typical diagram may not settle every actual image transition/ambiguity.','original45_labels_modified':False,'model_results_or_predictions_used':False,'new_training':False,'user_data_used':False,'rows':entries}
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    byname={r['class_name']:r for r in entries}
    counts={}
    for role in ['fit','validation','test']:
        rr=[r for r in exp['rows']if r['role']==role]
        counts[role]={'total_rows':len(rr),'features':{f:dict(collections.Counter('null'if byname[r['label']][f]is None else str(byname[r['label']][f])for r in rr))for f in features}}
    check={'mapping_sha256':sha(target),'source_class_count':len(entries),'features':{f:{'positive_classes':[r['class_name']for r in entries if r[f]==1],'negative_classes':[r['class_name']for r in entries if r[f]==0],'unknown_classes':[r['class_name']for r in entries if r[f]is None]}for f in features},'role_target_support':counts,'original_files_unchanged':{'EXPERIMENT.json':sha(U/'EXPERIMENT.json')},'no_class_string_heuristic':True,'no_prediction_review':True}
    (HERE/'DEEPHAND_PHYSICAL_MAPPING_SUPPORT.json').write_text(json.dumps(check,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'mapping':str(target),'sha':sha(target),'positive':{f:check['features'][f]['positive_classes']for f in features},'support':counts},ensure_ascii=False))
if __name__=='__main__':main()
