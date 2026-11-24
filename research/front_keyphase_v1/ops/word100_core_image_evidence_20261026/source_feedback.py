"""Reusable research feedback receipts. Observation advice, never a skill verdict."""
import numpy as np
CODES={0:'outside_audit_scope',1:'query_coordinates_missing',2:'reference_coordinates_missing',3:'both_coordinates_missing',4:'not_rechecked',5:'query_recheck_uncertain',6:'reference_recheck_uncertain',7:'both_rechecks_uncertain',8:'current_image_corroboration'}
TEXT={1:'이 시간대는 촬영 영상의 손 모양을 확인하기 어려워 비교를 보류했어요. 동작을 틀렸다고 판단한 것은 아니에요.',2:'이 시간대는 참고 영상의 손 모양 근거가 부족해 비교를 보류했어요. 사용자에게 동작 수정을 요구하지 않아요.',3:'이 시간대는 촬영 영상과 참고 영상 모두 손 모양을 확인하기 어려워요.',4:'좌표 차이는 계산했지만 이 시간대는 이미지 재확인을 하지 않았어요.',5:'촬영 영상의 손 모양을 다시 확인한 결과가 불확실해요. 확인 가능한 차이만 표시해요.',6:'참고 영상의 손 모양 재확인 근거가 불확실해요. 사용자 동작이 틀렸다고 판단하지 않아요.',7:'촬영 영상과 참고 영상 모두 손 모양 재확인 근거가 불확실해요.',8:'두 영상의 현재 이미지 근거와 함께 좌표 차이를 표시했어요. 정답률이나 올바른 수행 판정은 아니에요.'}
def states_on_pair(pair,q,r,mode='after'):
 domain=pair['domain'][:,:,:15];idx=pair['phase_old'];lo=np.floor(idx).astype(int);hi=np.ceil(idx).astype(int);qa=q['available'][lo,:,:15]&q['available'][hi,:,:15];ra=r['available'][:,:,:15];qgood=q['quality_'+mode][lo,:,:15]&q['quality_'+mode][hi,:,:15];rgood=r['quality_'+mode][:,:,:15];qs=q['state_'+mode];rs=r['state_'+mode];qc=(qs[lo]==1)|(qs[hi]==1);rc=rs==1;out=np.zeros(domain.shape,np.uint8);out[domain&~qa&ra]=1;out[domain&qa&~ra]=2;out[domain&~qa&~ra]=3;both=domain&qa&ra;out[both&~qgood&rgood]=5;out[both&qgood&~rgood]=6;out[both&~qgood&~rgood]=7;out[both&(qc|rc)]=4;out[both&qgood&rgood]=8
 assert np.array_equal(out==8,pair['point_corroborated_'+mode][:,:,:15]);assert not (out[domain]==0).any();return out
def build(pair,q,r,query_id,reference_id,consumer,mode='after'):
 keys=['domain','point_ok','point_delta','sequence_ok','sequence_delta','source','reference_missing','reference_times_us'];args={k:pair[k] for k in keys};args['sequence_quality']=pair['sequence_corroborated_'+mode];receipt=consumer.build(**args,query_id=query_id,reference_id=reference_id,kind='public_same_word_source_separated',recognition=None);states=states_on_pair(pair,q,r,mode);times=pair['reference_times_us'];cards=[]
 for h in (0,1):
  angles=[]
  for a in range(15):
   ss=states[:,h,a];entries=[]
   for code in range(1,9):
    mask=ss==code
    if mask.any():entries.append(dict(code=CODES[code],message=TEXT[code],cells=int(mask.sum()),intervals=consumer.spans(mask,times),gesture_error=None,correction_required=None))
   angles.append(dict(angle_index=a,states=entries))
  cards.append(dict(side=('L','R')[h],angles=angles))
 receipt['source_separated_point_evidence']=cards;receipt['advice_policy']='Reference shortage never becomes user error. Observation shortage never becomes low difference or success. Evidence messages are not claims that both hands must be used.';receipt['image_methods']='MediaPipe original/full/crop common-image agreement. YOLO audit separate; no YOLO veto or precision truth.';receipt['yolo_audit_available']=bool(q['yolo_conflict'].any());receipt['whole_gesture_correct']=None;receipt['validated_correction']=None;return receipt
