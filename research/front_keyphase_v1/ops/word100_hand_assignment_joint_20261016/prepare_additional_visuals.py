from pathlib import Path
U=Path(__file__).resolve().parent
s=(U/'render_changes.py').read_text('utf-8')
a=s.index('targets=[]');b=s.index('index=[]',a)
s=s[:a]+'''rows={(r['clip_id'],r['mode']):r for r in read(U/'REPLAY_RESULT.json')['rows']}
targets=[(rows[v['clip_id'],v['mode']],v['frame']) for v in read(U/'NUMERIC_VISUAL_TARGETS.json')]
'''+s[b:]
s=s.replace("sides=[np.load(d/n,allow_pickle=False)['sides'] for n in ['previous_assignment.npz','assignment.npz']]", "sides=[raw['baseline_side'],np.load(d/'assignment.npz',allow_pickle=False)['sides']]")
s=s.replace("'Previous v2'","'Production baseline'").replace("f'REVIEW_IMAGE_", "f'ADDITIONAL_IMAGE_")
s=s.replace("'REVIEW_IMAGE_INDEX.json'", "'ADDITIONAL_IMAGE_INDEX.json'")
with (U/'render_additional.py').open('x',encoding='utf-8') as f:f.write(s)
