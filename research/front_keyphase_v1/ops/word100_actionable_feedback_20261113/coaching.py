"""Evidence-bound Korean review cues. Does not infer correctness or change scores.

The stage association is an estimate from the existing model. Every correction
cue retains its source component and native time; unknown is never an error.
"""
from pathlib import Path
import copy,json,math
O=Path(__file__).resolve().parent.parent
GUIDES=json.loads((O/'word100_action_feedback_demo_20261108/CARDS.json').read_text('utf-8-sig'))['words']
CFG=json.loads((O/'word100_positive_direction_20261109/FROZEN.json').read_text('utf-8-sig'))
DIRS=json.loads((O/'word100_positive_direction_20261109/GUIDE_CONTRACTS.json').read_text('utf-8-sig'))['contracts']
SIDE={'R':'오른손','L':'왼손'};ARM={'R':'오른팔','L':'왼팔'}
SHAPES={1:('R','펴기',3),2:('L','펴기',4),3:('R','굽히기',1),4:('L','굽히기',2)}

def number(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def interval(x,n):
    return (isinstance(x,list) and len(x)==2 and all(isinstance(i,int) and not isinstance(i,bool) for i in x)
            and 0<=x[0]<=x[1]<n)
def overlaps(a,b):return not (a[1]<b[0] or b[1]<a[0])
def attach(result):
    out=copy.deepcopy(result);out['coaching']=compose(result);return out

def compose(result,guides_by_word=GUIDES,direction_contracts=DIRS):
    word=result.get('word');ts=result.get('timestamp_us',[]);n=len(ts)
    safe_time=bool(n>=2 and all(number(x) and x>=0 for x in ts) and all(a<b for a,b in zip(ts,ts[1:])))
    guides={c['id']:c for c in guides_by_word.get(word,[])};cues=[];notes=[];rejects=[]
    def emit(card,kind,span,hand,observation,action,basis,extra=None):
        key=(kind,hand,tuple(span),tuple(sorted((extra or {}).items())))
        for cue in cues:
            if cue['_key']==key:
                if card['id'] not in cue['card_ids']:cue['card_ids'].append(card['id'])
                return
        cues.append(dict(_key=key,kind=kind,card_ids=[card['id']],title=card['title'],hand=hand,
            start_seconds=ts[span[0]]/1e6,end_seconds=ts[span[1]]/1e6,focus_indices=span,
            observation=observation,action=action,basis=basis,association='model_estimate',
            educational_wrong=False,whole_action_complete=None,**(extra or {})))
    for card in result.get('cards',[]):
        cid=card.get('id');guide=guides.get(cid);status=card.get('display_status');span=card.get('query_indices')
        bound=guide is not None and all(card.get(k)==guide.get(k) for k in ['features','hands','groups','reference'])
        if not bound:
            rejects.append(dict(card_id=cid,reason='guide_component_binding_mismatch'))
            notes.append(dict(card_id=cid,status='uncertain',message='설명과 관측의 연결을 확인하지 못했어요.'))
            continue
        local=interval(span,n) and safe_time
        observed=not card.get('observation_quarantine_overlap',False)
        long_missing=result.get('long_missing_runs',[])
        def visible(s):return observed and not any(interval(m,n) and overlaps(s,m) for m in long_missing)
        before=len(cues)
        if status=='difference' and local and observed:
            for e in card.get('shape_evidence',[]):
                f=e.get('feature');spec=SHAPES.get(f) if type(f) is int else None
                if not spec or guide['features'].get(str(f))!=1:continue
                hand,action,opposite=spec
                ok=(type(e.get('expected')) is int and e.get('expected')==1 and e.get('state')=='contradicted' and e.get('opposite_feature')==opposite
                    and e.get('projection_consistency_required') is True and e.get('conflicting_predictions') is False
                    and number(e.get('available_fraction')) and e['available_fraction']>=CFG['positive']['min_available_fraction'])
                if not ok:continue
                for s in e.get('contradicted_spans',[]):
                    if not interval(s,n) or not (span[0]<=s[0]<=s[1]<=span[1]) or not visible(s):continue
                    opened=action=='펴기'
                    emit(guide,'handshape',s,hand,
                        f"{SIDE[hand]}의 네 손가락이 {'굽혀진' if opened else '펴진'} 모양으로 보여요.",
                        f"{SIDE[hand]}의 네 손가락을 {'펴고' if opened else '굽히고'} 이 부분을 다시 해 보세요.",
                        dict(feature=f,expected=1,opposite_feature=opposite,projection_checked=True,thumb_assessed=False))
            m=card.get('motion_evidence') or {};s=m.get('span');hand=m.get('moving_hand');arm=m.get('anchor_arm');sign=m.get('expected_sign')
            contract=direction_contracts.get(cid,{})
            ok=(all(m.get(k)==contract.get(k) for k in ['kind','moving_hand','anchor_arm','expected_sign'])
                and m.get('kind')=='along_forearm' and m.get('state')=='contradicted' and hand in SIDE and arm in ARM and hand!=arm
                and hand in guide['hands'] and arm in guide['hands'] and type(sign) is int and sign in [-1,1]
                and interval(s,n) and overlaps(span,s) and visible(s)
                and all(number(m.get(k)) for k in ['available_fraction','signed_change','trend','deadband'])
                and m['available_fraction']>=CFG['direction']['min_available_fraction'] and m['deadband']>=0
                and m['signed_change'] < -m['deadband'] and m['trend']<0)
            if ok:
                origin,dest=('팔꿈치','손목') if sign==1 else ('손목','팔꿈치')
                emit(guide,'direction',s,hand,f"{SIDE[hand]}이 참고와 반대 방향으로 움직인 것으로 보여요.",
                     f"{SIDE[hand]}을 {ARM[arm]}의 {origin} 쪽에서 {dest} 쪽으로 움직여 보세요.",
                     dict(signed_change=m['signed_change'],deadband=m['deadband'],contact_assessed=False),
                     dict(anchor_arm=arm,expected_sign=sign))
        if status=='similar' and local and observed:
            text='확인한 손모양·움직임 특징이 비슷해요.'
            if not card.get('checked_components'):text='이 구간은 참고와 함께 확인해 주세요.'
            notes.append(dict(card_id=cid,status='similar' if card.get('checked_components') else 'uncertain',message=text))
        elif status=='optional':notes.append(dict(card_id=cid,status='optional',message='마지막에 멈추는 시간까지 똑같이 맞출 필요는 없어요.'))
        elif len(cues)==before and not any(cid in c['card_ids'] for c in cues):
            message=('이 동작의 위치를 확실히 찾지 못했어요. 빠뜨렸다고 단정하지 않고 참고와 함께 확인할게요.' if not local else
                '이 순간의 손모양이나 움직임을 확실히 구분하지 못했어요. 틀렸다는 뜻은 아니에요.')
            notes.append(dict(card_id=cid,status='uncertain',message=message))
    # Same component at the same instant is one cue, even if stage cards overlap.
    cues.sort(key=lambda c:(c['start_seconds'],c['kind'],c['hand']))
    for i,cue in enumerate(cues):cue.pop('_key');cue['id']=f'cue_{i}'
    if cues:
        first=cues[0];summary=f"{first['start_seconds']:.1f}초 부근, ‘{first['title']}’ 부분을 확인해요. {first['action']}"
        status='review_difference'
    elif not notes or not safe_time or any(n['status']=='uncertain' for n in notes):summary='확인하지 못한 부분이 있어요. 해당 순간을 참고와 함께 살펴봐요.';status='uncertain'
    else:summary='이번에 확인한 특징은 참고와 비슷해요.';status='observed_similar'
    return dict(version='evidence_bound_coaching_v1',status=status,summary=summary,cues=cues,notes=notes,rejected=rejects,
        uses_new_model=False,uses_llm=False,does_not_change_model_scores=True,
        scope='observed component review cue; automatic stage binding remains estimated; not whole-sign correctness',
        educational_correctness_validated=False)
