"""Check saved-result display fidelity and read-only routes; no model evaluation."""
from pathlib import Path
import json,hashlib,urllib.request,urllib.error,traceback,time
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pointer(d,p):
    for k in p.strip('/').split('/'):d=d[int(k)] if isinstance(d,list) else d[k]
    return d
def main():
    build=read(U/'BUILD.json');cat=read(U/'data/catalog.json');checks=[]
    def check(name,ok):
        assert ok,name
        checks.append(name)
    for path,h in build['input_pins'].items():check('unchanged source '+Path(path).name,sha(path)==h)
    for path,h in build['output_pins'].items():check('display snapshot '+Path(path).name,sha(path)==h)
    cards=0;overlays=0
    for item in cat['cases']:
        d=read(U/'data'/f"{item['id']}.json")
        r=read(U.parent/'word100_feedback_discrimination_20261103/combined_results'/f"{item['id']}.json")
        check(item['id']+' no verdict',d['whole_sign_correctness'] is None and d['correction_certified'] is False)
        check(item['id']+' same word reference',d['reference']['id']==d['word_id']+'_reencoded_normal')
        check(item['id']+' identical stored overlay',d['query']['overlay']==r.get('overlay'))
        check(item['id']+' identical missing frames',d['query']['unavailable']==r['combined_evidence_feedback']['global_image_unavailable_frames'])
        overlays+=bool(d['query']['overlay'])
        for c in d['cards']:
            cards+=1;src=pointer(r,c['source_pointer']);qids=c['query_segments'][0]['span']['frames'] if c['query_segments'][0]['span'] else []
            if c['type']=='order':
                check(c['id']+' original order state',src['observation_status']==c['saved_status'])
                check(c['id']+' original evidence sets',sorted(sorted(s['span']['frames']) if s['span'] else [] for s in c['query_segments'])==sorted([src['first_evidence']['input_frames'],src['second_evidence']['input_frames']]))
            elif c['type']=='movement':check(c['id']+' original motion scope',qids==src['motion']['fixed_scope']['frame_ids'])
            elif c['type']=='feature':check(c['id']+' original observation frames',qids==[p['frame'] for p in src['native_points'] if p['observation']==c['saved_status']])
            else:check(c['id']+' original missing frames',set(qids)<=set(src))
            for key in ['query','reference']:
                for s in c[key+'_segments']:
                    if not s['span']:continue
                    x=s['span'];ts=d[key]['times']
                    check(c['id']+key+' native timestamps',x['start']==ts[x['frames'][0]] and x['end']==ts[x['frames'][-1]] and x['focus'] in [ts[i] for i in x['frames']])
            check(c['id']+' no invented correction',c['correction_certified'] is False and c['educational_error'] is None)
        req=urllib.request.Request('http://127.0.0.1:8811/media/'+item['id'],headers={'Range':'bytes=0-127'})
        with urllib.request.urlopen(req,timeout=10) as response:
            media=read(U/'MEDIA.json')[item['id']]
            with Path(media['path']).open('rb') as f:expected=f.read(128)
            check(item['id']+' exact public media range',response.status==206 and response.read()==expected)
    for path,method,status in [('/api/analyze','POST',405),('/api/case/unknown','GET',404),('/media/../../AGENTS.md','GET',404)]:
        try:urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8811'+path,method=method),timeout=5);raise AssertionError(path)
        except urllib.error.HTTPError as e:check('blocked '+method+path,e.code==status)
    with urllib.request.urlopen('http://127.0.0.1:8811/api/health') as r:health=json.load(r)
    check('read-only paused service',not health['inference'] and not health['training'] and health['goal_paused'])
    report=dict(status='PASS_SAVED_DISPLAY_FIDELITY',check_count=len(checks),cases=16,cards=cards,query_overlays=overlays,
        implementation='Codex',verification='Separate deterministic display/source check by same Codex; not human sign-language annotation',
        new_training=0,new_inference=0,new_DTW=0,new_scores=0,health=health,checks=checks)
    (U/'DISPLAY_VERIFIED.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['checks','health']},ensure_ascii=False))
if __name__=='__main__':
    try:main()
    except BaseException:
        (U/f'DISPLAY_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
