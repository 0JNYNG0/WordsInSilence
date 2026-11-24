"""Independent SQL aggregation from the already verified metadata manifest."""
import hashlib
import json
import sqlite3
from html.parser import HTMLParser
from pathlib import Path

U=Path(__file__).resolve().parent
r=json.loads((U/'DISTRIBUTION.json').read_text(encoding='utf-8'))
source=Path(r['source']); raw=source.read_bytes()
assert hashlib.sha256(raw).hexdigest()==r['source_sha256']
con=sqlite3.connect(':memory:')
con.execute('CREATE TABLE clips(id TEXT PRIMARY KEY, code TEXT, signer INT, label TEXT, clean INT, role TEXT)')
for line in raw.decode('utf-8').splitlines():
    a=json.loads(line)
    con.execute('INSERT INTO clips VALUES (?,?,?,?,?,?)',(a['clip_id'],a['word_code'],a['filename_signer_id'],a['labels'][0] if a['metadata_clean'] else None,int(a['metadata_clean']),a['role']))
assert con.execute('SELECT COUNT(*),SUM(clean) FROM clips').fetchone()==(54000,53917)
label_sql={k:(n,p,t,tp) for k,n,p,t,tp in con.execute("SELECT label,COUNT(*),COUNT(DISTINCT signer),SUM(role='train_candidate'),COUNT(DISTINCT CASE WHEN role='train_candidate' THEN signer END) FROM clips WHERE clean GROUP BY label")}
assert len(label_sql)==len(r['label_ranking'])==2965
for item in r['label_ranking']:
    assert label_sql[item['label']]==(item['count'],item['people'],item['training_count'],item['training_people'])
ranked=con.execute('SELECT label,RANK() OVER (ORDER BY p DESC,n DESC) FROM (SELECT label,COUNT(DISTINCT signer) p,COUNT(*) n FROM clips WHERE clean GROUP BY label)').fetchall()
assert dict(ranked)=={x['label']:x['rank'] for x in r['label_ranking']}
code_sql={k:(raw,n,p,l,t) for k,raw,n,p,l,t in con.execute("SELECT code,COUNT(*),SUM(clean),COUNT(DISTINCT CASE WHEN clean THEN signer END),COUNT(DISTINCT label),SUM(clean AND role='train_candidate') FROM clips GROUP BY code")}
for x in r['codes']:
    assert code_sql[x['code']]==(x['raw_count'],x['count'],x['people'],len(x['labels']),x['training_count'])
assert sum(x['count'] for x in r['label_ranking'])==sum(x['count'] for x in r['codes'])==53917
for key,col in [('clean_code_count_distribution','count')]:
    actual={str(v):sum(x[col]==v for x in r['codes']) for v in {x[col] for x in r['codes']}}
    assert actual==r[key]
assert r['label_people_distribution']=={str(v):sum(x['people']==v for x in r['label_ranking']) for v in {x['people'] for x in r['label_ranking']}}
assert r['code18_single_label_count']==sum(x[1]==18 and x[3]==1 for x in code_sql.values())==2659
class Reader(HTMLParser):
    def __init__(self):super().__init__();self.rows=0;self.scripts=0
    def handle_starttag(self,t,a):
        if t=='tr':self.rows+=1
        if t=='script':self.scripts+=1
h=Reader();h.feed((U/'단어별_전체분포.html').read_text(encoding='utf-8'))
assert h.rows==2+2965+3000+3 and h.scripts==0
v=dict(status='PASS',method='Codex separate SQLite aggregation/SQL rank + HTML row/link structure; no separate agent claim',
       source_sha256_unchanged=True,unique_front_rows=54000,clean_rows=53917,code_rows_verified=3000,
       label_rows_verified=2965,ranks_verified=2965,existing_training_role_counts_verified=True,
       full_code_and_label_totals_agree=True,html_rows=h.rows,model_or_video_execution=0)
(U/'VERIFIED_RESULT.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(v,ensure_ascii=False))
