from pathlib import Path
import json,hashlib,datetime,html
U=Path(__file__).resolve().parent;O=U.parent;D=O/'word100_dev_image_feedback_20261026';V=O/'word100_recovery_dev300_20261026'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open('x',encoding='utf8') as fp:json.dump(x,fp,ensure_ascii=False,allow_nan=False)

def build():
    assert read(D/'FEEDBACK_VERIFIED.json')['queries']==300
    assert read(D/'HTML_SCRIPT_VERIFIED.json')['states']==13800
    frozen=read(D/'FROZEN.json');fd=read(D/'FEEDBACK_DONE.json');lineage=read(D/'SAVED_RECOGNITION_LINEAGE.json')
    for p,h in lineage['source_pins'].items():assert sha(p)==h
    es=read(V/'E3_SCORES.json');assert es['weights']==[1/3]*3 and es['training']==0
    e3={x['clip_id']:x for x in es['predictions']['native']};view={x['query']:x for x in read(D/'VIEW_DATA.json')};rows=[]
    files=[D/'FROZEN.json',D/'FEEDBACK_DONE.json',D/'FEEDBACK_VERIFIED.json',D/'INTEGRATION_VERIFIED.json',D/'SUMMARY.json',D/'VIEW_DATA.json',D/'SAVED_RECOGNITION_LINEAGE.json',V/'E3_SCORES.json',V/'CLASSIFIER_VERIFIED.json',V/'CLASSIFIER_RESULT.json',U/'adapter.py']
    labelmap={x['label_index']:x['canonical_label'] for x in read(O/'word100_body_boundary_20261017/PREP_CONTEXT.json')['ordered_rows'] if x['role']=='development_only'}
    for row in frozen['rows']:
        cid=row['clip_id'];qp=D/'queries'/(cid+'.json');q=read(qp);saved=e3[cid]
        assert saved['signer_id']==row['source']['signer_id'] and q['query_id']==cid
        preds={k:dict(value) for k,value in view[cid]['predictions'].items()}
        preds['E3']={k:saved[k] for k in ('status','prediction','top5','correct','top5_correct')}
        preds['E3'].update(predicted_label=labelmap[saved['prediction']],top5_labels=[labelmap[i] for i in saved['top5']])
        files.append(qp);rows.append(dict(query_id=cid,recorded_person_id=row['source']['signer_id'],word_code=row['source']['word_code'],label=row['source']['label'],query_path=str(qp),predictions=preds,reference_count=len(q['references'])))
    assert len(rows)==300 and {x['query_id'] for x in rows}==set(e3) and all(x['reference_count']==2 for x in rows)
    manifest=dict(schema='public_recognition_feedback_v1',at=datetime.datetime.now().astimezone().isoformat(),
        condition='Read-only publicDEV300 replay; existing E3 fixed ensemble plus source-aware DTW feedback',
        models=['C','T3F','E3'],default_model='C',allow_personal_input=False,
        new_training=0,new_classifier_inference=0,new_ranking=0,operating_changed=False,
        e3_weights=es['weights'],scope='RecordedIDs11/12/13,100words; development data,not new heldout accuracy',
        feedback_target='Practice target, independent of classifier prediction',
        pins={str(p):sha(p) for p in files},receipt_pins={p:h for p,h in fd['pins'].items() if Path(p).parent.name=='receipts'},rows=rows,
        prior_recognition_summary=lineage['scores'],feedback_summary=read(D/'SUMMARY.json')['coverage'],
        rule='No calibrated semantic-normal range or direction-of-correction is available; do not convert confidence or coverage to skillscore')
    write(U/'MANIFEST.json',manifest)
    write(U/'AUTHORIZATION.json',dict(user_request='Continue autonomously improving a common research candidate, preserve C/T3F and personal recordings',scope=manifest['condition'],training=0,operating_changed=False,implementation='Codex local',prohibited=['personal/mentor/extra199/200 access','operating replacement','camera','network','training','new inference']))
    print('FROZEN_RESEARCH_BINDING',len(rows),len(manifest['receipt_pins']),flush=True)

if __name__=='__main__':build()
