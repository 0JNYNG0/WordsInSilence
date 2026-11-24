"""Local inventory audit only. No new word training or game integration."""
from pathlib import Path
import json,hashlib
U=Path(__file__).resolve().parent;O=U.parents[2]
def read(p):return json.loads(p.read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
paths={
 'sources':O/'word100_rtmw_joint_learning_20261108/SOURCES.json',
 'labels':O/'word100_rtmw_joint_learning_20261108/FROZEN.json',
 'profiles':O/'word100_linguistic_feedback_pipeline_20261027/FEATURE_PROFILES.json',
 'form_links':O/'word100_source_token_alignment_20261027/SOURCE_FORM_LINKS.json',
 'guides':O/'word100_learned_guide_feedback_20261102/GUIDE_SUPPORT.json'}
data={k:read(p)for k,p in paths.items()}
rows=[]
for label,scene in [('잘하다','캐릭터의 성공을 보고 칭찬하기'),('감사','도움을 받은 뒤 고마움 표현하기'),('즐겁다','함께한 활동 뒤 기분 표현하기')]:
    ss=[r for r in data['sources']if r['label']==label];assert label in data['labels']['labels']and ss
    wid=ss[0]['word_id'];assert {r['word_id']for r in ss}=={wid}
    profile=next((r for r in data['profiles']['profiles']if r['word_id']==wid),None)
    link=next((r for r in data['form_links']['links']if r['word_id']==wid),None)
    guides=[r for r in data['guides']['guides']if r['word_id']==wid]
    rows.append(dict(label=label,word_id=wid,scene_proposal=scene,
        public_fit_sources=len({r['clip_id']for r in ss if r['role']=='fit'}),
        public_development_sources=len({r['clip_id']for r in ss if r['role']=='development'}),
        current_detailed_feedback_pilot=any(r['feedback']for r in ss),
        saved_guide_stage_count=len(guides),
        dictionary_candidates=[{k:r.get(k)for k in ['entry_id','entry_label','url','description','link_status']}for r in (profile or {}).get('candidate_entries',[])],
        saved_form_review=link,
        admission='NOT_YET_ADMITTED_FOR_AUTOMATIC_EDUCATIONAL_CORRECTION',
        remaining=(['현재 공통 판정부 안정화 및 통합 공개검증']if guides else['사전 뜻/표현형과 공개 영상의 일치 확인','기존 공통 특징으로 설명을 연결할 시간별 정답 검토','정상/속도/교차/손모양/방향/누락 비교']),
        source_ids=sorted({r['clip_id']for r in ss})))
out=dict(status='CANDIDATE_READINESS_ONLY_NOT_FINAL_SELECTION',rows=rows,
    proposal='Reuse existing four-word regressions; prepare 2-3 short game situations without implementing Unity now. First detailed pilot word is 잘하다 only after its existing gates pass. 감사/즐겁다 remain candidates, not ready feedback words.',
    rules=['Do not select words by best observed test score. Fix source/variant/feature eligibility first.',
           'No concatenated KSL sentence or linguistic naturalness claim; these are separate game situations.',
           'A word in the100-class classifier does not imply fine-feedback support.',
           'No private recordings, downloads, live camera, new training, web changes or game deployment in this audit.'],
    pins={str(p):sha(p)for p in paths.values()})
with (U/'DEMO_WORD_READINESS.json').open('x',encoding='utf-8')as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
for r in rows:print(r['label'],r['word_id'],'fit',r['public_fit_sources'],'dev',r['public_development_sources'],'guide stages',r['saved_guide_stage_count'],'dictionary candidates',len(r['dictionary_candidates']))
