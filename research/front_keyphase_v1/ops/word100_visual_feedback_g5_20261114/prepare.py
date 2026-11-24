from pathlib import Path
import json,hashlib,shutil,subprocess
import numpy as np
import imageio_ffmpeg
from adapter import project
U=Path(__file__).resolve().parent; O=U.parent; G4=O/'word100_public_g4_20261114'; G=O/'word100_web_completion_graph_20261111'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
LABELS={'WORD0739':'순종','WORD1133':'잘하다','WORD1567':'조상','WORD2523':'이집트'}
KINDS={'normal':'공개 원본','speed075':'느리게 재생한 시험','speed125':'빠르게 재생한 시험','prefix_before_goal':'중간에서 자른 시험','continuous_full_reverse':'전체 역재생 시험','all_hands_missing':'손 관측을 제거한 시험'}

def main():
    for d in ('data','media','site','before'): (U/d).mkdir(exist_ok=True)
    write(U/'PLAN.json',dict(stage='G5',goal='G4 저장 근거를 기존 웹에 연결: 정확한 시간·손·설명, 교정과 확인필요 구분',steps=['표시 계약과 근거 고정','공통 표시 어댑터와 재생 화면 구현','공개 원본/속도/역재생/부분/누락 결과 검산·브라우저 확인','복구 경로·그래프·기록 갱신'],checks=['evidence_text_visual_agree','no_unqualified_claim','rollback'],no_training=True,no_private_access=True,no_model_change=True,live_capture_acceptance='G6 pending',author='Codex',verifier='Codex separate script and browser inspection; not independent human review'))
    frozen={str(p):sha(p) for p in [G4/'CONFLICT_REGRESSION.json',G4/'EXTRACTED.json',O/'word100_action_episode_supervision_20261110/GUIDE_FAMILIES.json',G/'PROTECTIONS.json']}
    write(U/'FROZEN_INPUTS.json',frozen)
    results=read(G4/'CONFLICT_REGRESSION.json'); records=read(G4/'EXTRACTED.json')['rows']; sources={r['clip_id']:r for r in records}; guides=read(O/'word100_action_episode_supervision_20261110/GUIDE_FAMILIES.json')['words']
    rows=[r for group in ('public_normal','public_continuous','public_splice') for r in results['rows'][group] if r['condition']=='current_g3' and r['kind'] in KINDS]
    manifest=[]; routes={}; bank={}; all_data=[]; ff=imageio_ffmpeg.get_ffmpeg_exe()
    for cid,sr in sources.items():
        assert sha(sr['path'])==sr['sha256'] and sha(sr['raw_path'])==sr['raw_sha256']
        z=np.load(sr['raw_path']); ch=np.load(sr['channel_path']); xy=z['xy_px']; ok=ch['point_available'].astype(bool); w=int(z['source_width']);h=int(z['source_height'])
        ok &= np.isfinite(xy).all(-1)&(xy[:,:,0]>=0)&(xy[:,:,0]<w)&(xy[:,:,1]>=0)&(xy[:,:,1]<h)
        bank[cid]=(z,ch,xy,ok,w,h)
        routes[f'/feedback-g5/media/{cid}.mp4']=sr['path']
    for r in rows:
        cid=r['clip_id'];kind=r['kind'];word=cid.split('_')[2];z,ch,xy,ok,w,h=bank[cid];n=len(xy);rate={'speed075':.75,'speed125':1.25}.get(kind,1)
        if kind=='prefix_before_goal':
            n=read(G4/'auto'/f'{cid}_{kind}.json')['frames']
        mapping=list(range(n)); video=f'/feedback-g5/media/{cid}.mp4'
        if kind=='continuous_full_reverse':
            mapping=read(G4/'continuous'/f'{cid}_{kind}.json')['source_mapping'];assert mapping==list(range(len(xy)-1,-1,-1))
            name=f'{cid}_reverse.mp4'; dest=U/'media'/name
            if not dest.exists():
                subprocess.run([ff,'-hide_banner','-loglevel','error','-i',sources[cid]['path'],'-an','-vf','scale=960:-2,reverse','-c:v','libx264','-crf','19','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart','-y',str(dest)],check=True)
            video='/feedback-g5/media/'+name;routes[video]=str(dest)
        ts=np.rint(z['timestamp_us'][:n]/rate).astype('int64').tolist()
        coordinates=[]
        for f in mapping:
            a=[]
            for j in range(133):
                good=ok[f,j] and not(kind=='all_hands_missing' and j>=91)
                a.append([round(float(xy[f,j,0]),2),round(float(xy[f,j,1]),2)] if good else None)
            coordinates.append(a)
        cards=project(r,guides[word],n)
        refrow=next(x for x in rows if x['clip_id']==cid and x['kind']=='normal')
        refcards=project(refrow,guides[word],len(xy))
        refxy=[[([round(float(v[0]),2),round(float(v[1]),2)] if ok[f,j] else None) for j,v in enumerate(xy[f])] for f in range(len(xy))]
        key=cid+'__'+kind
        data=dict(id=key,word=word,label=LABELS[word],kind=kind,kind_label=KINDS[kind],clip_id=cid,record_id=sources[cid]['record_id'],source='public_saved_G4',synthetic=kind!='normal',video=video,playback_rate=rate,width=w,height=h,timestamp_us=ts,frame_count=n,source_mapping=mapping,xy=coordinates,cards=cards,reference=dict(video=f'/feedback-g5/media/{cid}.mp4',width=w,height=h,timestamp_us=z['timestamp_us'].tolist(),xy=refxy,cards=refcards),alignment='side_by_side_native_time_not_DTW',training_use=False,whole_sign_correct=None,provenance=dict(result=str(G4/'CONFLICT_REGRESSION.json'),result_sha256=frozen[str(G4/'CONFLICT_REGRESSION.json')],source_sha256=sources[cid]['sha256'],prior_use=sources[cid]['prior_use']))
        path=U/'data'/f'{key}.json';write(path,data);routes[f'/feedback-g5/data/{key}.json']=str(path)
        statuses=[c['status'] for c in cards]
        manifest.append(dict(id=key,label=LABELS[word],word=word,record_id=sources[cid]['record_id'],kind=kind,kind_label=KINDS[kind],url=f'/feedback-g5/data/{key}.json',difference='difference' in statuses,uncertain='uncertain' in statuses))
        all_data.append(data)
    write(U/'site/catalog.json',dict(cases=manifest,default='NIA_SL_WORD1133_REAL18_F__continuous_full_reverse',source='G4 공개 저장 결과',limits=['엄지·손끝 모으기·반복 전체·실제 접촉은 아직 판정하지 않음','확인한 특징을 보여주며 단어 전체의 정답·완료를 의미하지 않음','새 촬영에 대한 최신 판정부 연결과 촬영 검증은 G6에서 확인']))
    write(U/'ROUTES.json',routes)
    write(U/'BUILD.json',dict(cases=len(manifest),cards=sum(len(d['cards']) for d in all_data),cues=sum(len(c['cues']) for d in all_data for c in d['cards']),source_videos=len(sources),training_calls=0,neural_calls=0,private_data_read=False))
    print(json.dumps(read(U/'BUILD.json'),ensure_ascii=False))
if __name__=='__main__':main()
