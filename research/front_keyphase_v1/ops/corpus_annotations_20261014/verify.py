"""Independent SQL aggregation and direct source sample comparison of audit outputs."""
import collections as C
import hashlib
import json
from pathlib import Path
import sqlite3
import tarfile

OUT = Path(__file__).resolve().parent
def save(name, obj):
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf8')

a = json.loads((OUT/'FULL_AUDIT.json').read_text('utf8'))
w = json.loads((OUT/'WORD_DISTRIBUTION.json').read_text('utf8'))
con = sqlite3.connect(f'file:{(OUT/"ANNOTATION_INDEX.sqlite").as_posix()}?mode=ro', uri=True)
checks = []
def check(name, actual, expected):
    checks.append(dict(name=name, actual=actual, expected=expected, passed=actual==expected))
check('document_total', con.execute('SELECT COUNT(*) FROM documents').fetchone()[0], a['documents_parsed'])
check('raw_event_accounting', sum(sum(y['raw_tiers'].values()) for y in a['years'].values()),
      a['events_sql_count']+a['issues']['raw_valid_events_minus_exact_union_events']+a['issues']['invalid_time_or_empty_gloss'])
sql = '''SELECT d.year,e.gloss,COUNT(*),COUNT(DISTINCT e.doc),COUNT(DISTINCT d.signer)
 FROM events e JOIN documents d ON e.doc=d.doc WHERE (e.tiers & 1)=1 AND e.express='s'
 GROUP BY d.year,e.gloss'''
rows = con.execute(sql).fetchall()
expected = {(r['year'],r['gloss']):(r['events'],r['documents'],r['signer_count']) for r in w}
mismatch = [(y,g,n,nd,ns,expected.get((y,g))) for y,g,n,nd,ns in rows if (n,nd,ns)!=expected.get((y,g))]
check('word_groups', len(rows), len(expected))
check('all_word_event_document_signer_counts', len(mismatch), 0)
for year, ys in a['years'].items():
    check(f'{year}_primary_s', sum(n for y,g,n,nd,ns in rows if y==year), ys['valid_primary_s'])
    check(f'{year}_bytes_vs_csv', ys['bytes'], a['index_checks'][year]['copy_bytes'])
    check(f'{year}_speaker_count', con.execute('SELECT COUNT(DISTINCT signer) FROM documents WHERE year=?',(year,)).fetchone()[0], len(ys['signers']))

# Thirty documents, selected across year boundaries and interior positions, compared with raw JSON again.
sample = []
for year in sorted(a['years']):
    docs = con.execute('SELECT doc,path,video,signer FROM documents WHERE year=? ORDER BY path',(year,)).fetchall()
    sample += [docs[round(i*(len(docs)-1)/9)] for i in range(10)]
targets = {p:(doc,video,signer) for doc,p,video,signer in sample}
sample_checks = []
with tarfile.open(a['source'], 'r|gz') as tf:
    for m in tf:
        if m.name not in targets:
            continue
        doc,video,signer = targets[m.name]
        j = json.loads(tf.extractfile(m).read())
        ss = j['sign_script'].get('sign_gestures_strong') or []
        raw_events = {(str(t.get('gloss_id') or '').strip(),float(t['start']),float(t['end'])) for t in ss
            if t.get('express')=='s' and t.get('gloss_id') and float(t['start'])>=0 and float(t['end'])>float(t['start'])}
        saved_events = set(con.execute("SELECT gloss,start,end FROM events WHERE doc=? AND express='s' AND (tiers & 1)=1",(doc,)))
        sample_checks.append(dict(path=m.name,primary_s_match=raw_events==saved_events,
            metadata_match=(video==j['vido_file_nm'] and signer==j['potogrf']['sl_speaker_id'])))
check('stratified_source_samples',len(sample_checks),30)
check('sample_mismatches',sum(not (s['primary_s_match'] and s['metadata_match']) for s in sample_checks),0)
h=hashlib.file_digest(open(a['source'],'rb'),'sha256').hexdigest()
inventory=json.loads((OUT/'ARCHIVE_INVENTORY.json').read_text('utf8'))
check('source_sha256_preserved',h,inventory['sha256'])

supp = {'year_details':{}}
for year,ys in a['years'].items():
    ww=[r for r in w if r['year']==year]
    counts=ys['signers']; largest=max(counts.values())
    supp['year_details'][year]=dict(largest_signer_documents=largest,
        largest_signer_fraction=largest/ys['documents'],
        signer_profiles_with_multiple_metadata=sum(len(profiles)>1 for profiles in ys['signer_profiles'].values()),
        candidate_glosses_ge5_events_each_for_ge8_signers=sum(sum(n>=5 for n in r['signer_events'].values())>=8 for r in ww),
        candidate_glosses_ge10_events_each_for_ge8_signers=sum(sum(n>=10 for n in r['signer_events'].values())>=8 for r in ww),
        longest_gloss_examples=sorted([r['gloss'] for r in ww],key=lambda s:(-len(s),s))[:20],
        rare_gloss_single_event_count=sum(r['events']==1 for r in ww))
supp['cross_year_raw_signer_id_strings']=[r[0] for r in con.execute('SELECT DISTINCT signer FROM documents ORDER BY signer')]
supp['raw_id_strings_are_not_confirmed_unique_people']=True
supp['review_label_examples']=[{k:r[k] for k in ('year','gloss','events','documents','signer_count','signer_events','example')} for r in w
    if r['gloss'] in ['감사','감사(고마움)','고맙다1','고맙다2','수어','수어1','한국수어1','공부','공부1','먹다','먹다1','운동1','수영1','잘하다1']]
supp['current100_exact_gloss_matches_by_year']={}
labels=json.loads((OUT/'CURRENT100_LABEL_DISCOVERY.json').read_text('utf8'))['current100']
for y in sorted(a['years']):
    gl={r['gloss'] for r in w if r['year']==y}
    supp['current100_exact_gloss_matches_by_year'][y]=[l for l in labels if l in gl]
save('SUPPLEMENT.json',supp)
save('INDEPENDENT_CHECK.json',dict(status='PASS' if all(c['passed'] for c in checks) else 'FAIL',
     author='Codex; separate SQL aggregation/source-sample code, not external person or agent review',checks=checks,samples=sample_checks))
print(json.dumps({'checks':len(checks),'pass':all(c['passed'] for c in checks),'supplement':supp['year_details']},ensure_ascii=False),flush=True)
