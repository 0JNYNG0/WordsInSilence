"""Source-linked expectations, separate from actual video observations."""
import json,hashlib
from pathlib import Path
U=Path(__file__).resolve().parent;O=U.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(name,x):(U/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    profiles=read(O/'word100_shared_feature_bundle_20261027/SOURCE_PROFILES.json')['profiles']
    form={x['word_id']:x for x in read(O/'word100_source_token_alignment_20261027/SOURCE_FORM_LINKS.json')['links']}
    indices=read(O/'word100_shared_feature_bundle_20261027/visual_inputs/INDEX.json')['rows']
    cards=[]
    for p in profiles:
        link=form.get(p['word_id']); candidates=[]
        for e in p['candidate_entries']:
            key=e['source_id']+':'+str(e['entry_id'])
            rejected=bool(link and key in link['rejected_ids'])
            claims=[]
            for c in e.get('claims',[]):
                claims.append({'family':c['family'],'text':c.get('text',e.get('description')),'spans':c.get('spans',[]),
                    'source':c.get('source',{'url':e['url']}),'time_relation':'source text only; no video phase assignment',
                    'requiredness':None,'tolerance':None,'video_match_verified':False})
            candidates.append({'entry_id':key,'url':e['url'],'description':e.get('description'),
                'status':'REJECTED_FOR_CURRENT_VIDEO_FORM' if rejected else 'PROVISIONAL_EXPECTATION_ONLY',
                'claims':claims})
        cards.append({'word_id':p['word_id'],'label':p['display_label'],'candidates':candidates,
            'link':link or {'state':'SOURCE_CANDIDATE_NOT_FORM_VERIFIED','summary':'같은 이름의 설명 후보. 영상 표현형 세부 연결 미확인.'},
            'clips':[{k:r[k] for k in ['alias','clip_id','recorded_person_id','role','source','annotation_intervals']} for r in indices if r['word_id']==p['word_id']],
            'educational_correction_enabled':False,'full_guide_match_verified':False})
    # Keep the guide-vs-observation distinction visible in Korean.
    for c in cards:
        if c['link']['state']=='SOURCE_CANDIDATE_NOT_FORM_VERIFIED':c['link']['summary']='같은 이름의 설명 후보입니다. 실제 영상과 같은 표현형인지 세부 연결은 미확인입니다.'
    save('GUIDE_CARDS.json',{'status':'SOURCE_EXPECTATIONS_NOT_LABELS','cards':cards,
        'purpose':'Dictionary expectations and video evidence remain separate; unknown instructions do not fill observations.'})
    obs=read(O/'word100_source_token_alignment_20261027/TOKEN_OBSERVATIONS.json')
    save('EXISTING_STAGE_EVIDENCE.json',{'status':'REUSED_IMAGE_DRAFTS_NOT_PRETRAINED_INFERENCE',
        'groups':obs['annotations'],'warning':'Listed evidence frames only; groups may overlap/interleave. Do not treat group index as linguistic phase order.'})
    print({'guide_cards':len(cards),'clips':sum(len(c['clips']) for c in cards),'educational_rules_verified':0})
if __name__=='__main__':main()
