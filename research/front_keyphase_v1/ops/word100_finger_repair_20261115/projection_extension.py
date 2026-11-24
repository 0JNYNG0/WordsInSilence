from collections import defaultdict
NAMES={'index':'검지','middle':'중지','ring':'약지','little':'새끼손가락'}
def project_fingers(base_project,row,guides,frames):
    result=base_project(row,guides,frames)
    for p in result:
        source=p['source_card'];details=source.get('finger_details',[])
        if p['status']!='difference' or source['state']!='attributable_feature_difference':continue
        groups=defaultdict(list)
        for i,e in enumerate(details):groups[e['hand'],e['expected'],tuple(e['span'])].append((i,e))
        extra=[]
        for (hand,expected,span),ee in groups.items():
            fingers=[e['finger']for i,e in ee];name='오른손'if hand=='R'else'왼손';part='엄지를 제외한 네 손가락'if len(fingers)==4 else '·'.join(NAMES[f]for f in fingers)
            final=(ord(part[-1])-0xAC00)%28 !=0
            message=f"{name}의 {part}{'이'if final else'가'} {'굽혀진'if expected=='extended'else'펴진'} 것으로 관측됐어요."
            action=f"{name}의 {part}{'을'if final else'를'} {'펴'if expected=='extended'else'굽혀'} 보세요."
            extra.append(dict(kind='finger',message=message,action=action,spans=[list(span)],hands=[hand],fingers=fingers,evidence='finger_details/'+','.join(str(i)for i,e in ee),whole_sign_correct=None))
        if extra:
            # Keep original coarse evidence in source_card; avoid duplicate cue buttons.
            hands={h for e in extra for h in e['hands']};p['cues']=[e for e in p['cues']if not(e['kind']=='shape'and set(e['hands'])<=hands)]+extra;p['message']=p['cues'][0]['message']
    return result
