"""Independent synthetic contract fixtures; contains no human observations."""
from copy import deepcopy
from datetime import datetime, timezone

def fixture():
    words=[{'word_id':f'W{i:02}','clip_id':f'CLIP_{i:02}','display_name':f'word{i}'} for i in range(18)]
    plan={'schema':'wis_fresh_self_capture_plan_v1','protocol_id':'synthetic_v1','participant_id':'self','purpose':'frozen_retest',
          'reference_set_id':'synthetic_refs','prepared_at_kst':'2026-01-01T00:00:00+00:00','rounds':3,'words':words,
          'slots':[{'slot':r*18+i+1,'round':r+1,'word_id':w['word_id']} for r in range(3) for i,w in enumerate(words)]}
    journal={'schema':'wis_frozen_capture_journal_v1','protocol_id':plan['protocol_id'],'client_id':'client_synthetic','session_id':'ses_synthetic',
             'status':'CLOSED','current_slot':9,'slots':[]}
    for slot in plan['slots']:
        n=slot['slot']; tid=f'tri_{n:02}' if n<=8 else None
        journal['slots'].append(dict(slot= n,round=slot['round'],requested_word_id=slot['word_id'],trial_id=tid,
          request_id=f"synthetic_v1:client_synthetic:{n}:{slot['word_id']}" if tid else None,
          terminal_state='FEEDBACK' if n in (1,2,3,4,6,7) else 'CANCELLED' if n==5 else 'ERROR' if n==8 else None,
          review_saved=False,explicit_skip=False,review_revision=None,transport_error=None))
    session={'session_id':'ses_synthetic','client_id':'client_synthetic','participant_id':'self','purpose':'frozen_retest','state':'CLOSED',
             'created_epoch':datetime(2026,1,1,0,0,10,tzinfo=timezone.utc).timestamp(),'store_video':False,'camera':False,
             'trial_ids':[f'tri_{n:02}' for n in range(1,9)]}
    results={};artifacts={};reviews=[]
    actual={1:'W00',2:'W01',3:'W00',4:'UNKNOWN',6:'W05',7:'W06'}
    for n in range(1,9):
        tid=f'tri_{n:02}'
        artifacts[tid]={'npz_exists':False,'result_exists':False,'sidecar_exists':False,'npz_sha256':None,'sidecar_observation_sha256':None}
        if n not in actual:continue
        word=words[n-1];digest=str(n)*64
        results[tid]={'trial_id':tid,'session_id':'ses_synthetic','origin':'camera','reference_set_id':plan['reference_set_id'],
                      'observation_sha256':digest,'reference_binding':{'status':'BOUND','reference_set_id':plan['reference_set_id'],
                      'requested_word_id':word['word_id'],'clip_id':word['clip_id'],'demonstration_clip_id':word['clip_id'],
                      'demonstration_word_id':word['word_id']},'timing':{'server_end_candidate_to_result_ms':n*10}}
        artifacts[tid]={'npz_exists':n!=6,'result_exists':True,'sidecar_exists':True,'npz_sha256':('f'*64 if n==7 else digest),
                        'sidecar_observation_sha256':digest}
        reviews.append({'session_id':'ses_synthetic','trial_id':tid,'revision':1,'source':'participant_self_report',
                        'actual_word_id':actual[n],'followed_guidance':'yes','followed_demonstration_form':'yes',
                        'note':f'protocol=synthetic_v1;slot={n}; synthetic only'})
    older=deepcopy(reviews[0]);older['actual_word_id']='W02'
    reviews[0]['revision']=2
    reviews.insert(0,older)
    events=[{'event':'trial_closed','session_id':'ses_synthetic','trial_id':'tri_05','state':'CANCELLED','outcome':'CANCELLED'}]
    return plan,journal,session,reviews,events,results,artifacts

def predictions():
    truth={1:('W00','W01','W00'),2:('W01','W01','W00'),3:('W02','W00','W00'),4:('W03','W03','W03')}
    rows={}
    for n,values in truth.items():
        rows[f'tri_{n:02}']={}
        for role,pred in zip(('parent','control','treatment'),values):
            rows[f'tri_{n:02}'][role]={'status':'ERROR' if (n,role)==(3,'treatment') else 'READY',
                                     'project18':pred,'full93':'L'+pred[1:],'elapsed_ms':float(n),'error':None}
    return rows
