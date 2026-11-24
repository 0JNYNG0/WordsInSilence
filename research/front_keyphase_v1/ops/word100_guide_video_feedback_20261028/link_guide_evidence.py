"""Explicit, partial text-feature links; no word-specific detector exceptions.

These are source annotations of what to look for, not handshape ground truth,
not requiredness labels, and not training data. Observations stay unchanged.
"""
from pathlib import Path
import json,hashlib
U=Path(__file__).resolve().parent;O=U.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(name,v):(U/name).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
# Data records annotate dictionary spans, rather than alter a detector for a word.
ATOMS=[
    ('NIKL_legacy:11290','두 주먹의 1지를 펴서',['R_index_only','L_index_only'],'두 손 검지 펴기'),
    ('NIKL_legacy:6280','오른 주먹의 1지를 펴서',['R_index_only'],'오른 검지 펴기'),
    ('NIKL_legacy:6700','오른 주먹의 1지를 펴서',['R_index_only'],'오른 검지 펴기'),
    ('NIKL_legacy:2372','펴서 모로 세운 오른손',['R_open_four'],'오른손 네 손가락 펼침'),
    ('NIKL_legacy:2372','편 왼 손등',['L_open_four'],'왼손 네 손가락 펼침'),
    ('NIKL_legacy:6163','편 오른손',['R_open_four'],'펼친 오른손 상태'),
]
def main():
    cards=read(U/'GUIDE_CARDS.json')['cards']
    draft=read(O/'word100_shared_feature_bundle_20261027/DRAFT_LABELS.json')['rows']
    data=[]
    for entry,span,features,ko in ATOMS:
        owner=[(c,e) for c in cards for e in c['candidates'] if e['entry_id']==entry]
        assert len(owner)==1,(entry,len(owner))
        card,e=owner[0];assert span in e['description']
        samples=[]
        for r in draft:
            if r['word_id']!=card['word_id']:continue
            for f in features:
                value=r['labels'][f]
                samples.append({'alias':r['alias'],'person':r['recorded_person_id'],'feature':f,
                    'pts_seconds':r['pts_us']/1e6,'frame_id':r['frame_id'],'image':r['image'],
                    'value':value,'state':'IMAGE_DRAFT_SUPPORT' if value==1 else ('IMAGE_DRAFT_DIFFERENT_AT_THIS_SAMPLE' if value==0 else 'UNKNOWN'),
                    'phase_correspondence_verified':False,'educational_error':None})
        data.append({'entry_id':entry,'word_id':card['word_id'],'label':card['label'],'source_url':e['url'],
            'source_span':span,'label_ko':ko,'shared_features':features,'expected_value':1,
            'source_status':'TEXT_FEATURE_LINK_DRAFT','time_scope':'at some described stage, not all frames',
            'samples':samples,'requiredness':None,'tolerance':None,'whole_shape_verified':False,
            'limitations':'A true sampled atom only supports that feature. A false sample may be preparation/transition; it is not proof of a wrong sign.'})
    save('GUIDE_CLAIM_LINKS.json',{'status':'PARTIAL_SOURCE_TO_IMAGE_DRAFT_LINK_NOT_LEARNED_ALIGNMENT',
        'atoms':data,'source_atom_count':len(data),'expressions':len(set(a['word_id'] for a in data)),
        'unlinked_families':['hand-facing 3D orientation','physical contact','mandatory repeat count','all required phases','linguistic tolerances'],
        'video_observations_changed':False,'learned_semantic_model':False})
    print({'partial_text_feature_links':len(data),'sample_links':sum(len(a['samples']) for a in data),'educational_verdicts':0})
if __name__=='__main__':main()
