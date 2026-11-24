"""Bounded author chart acquisition. No training/user images or labels changed."""
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urljoin
from html.parser import HTMLParser
import json, hashlib, time

U=Path(__file__).resolve().parent
OUT=U/'taxonomy_source'
OUT.mkdir(exist_ok=True)
BASE='https://www-i6.informatik.rwth-aachen.de/~koller/1miohands-data/'
def fetch(url,filename,max_bytes):
    p=OUT/filename
    if p.exists():return p.read_bytes()
    try:
        with urlopen(Request(url,headers={'User-Agent':'Research-source-verification/1.0'}),timeout=30)as r:
            b=r.read(max_bytes+1)
            if len(b)>max_bytes:raise RuntimeError('Bound exceeded')
        p.write_bytes(b);return b
    except Exception as e:
        (OUT/('ERROR_'+str(time.time_ns())+'.json')).write_text(json.dumps({'url':url,'type':type(e).__name__,'message':str(e)}),encoding='utf-8')
        raise

class Table(HTMLParser):
    def __init__(self):super().__init__();self.rows=[];self.row=None;self.cell=None
    def handle_starttag(self,tag,attrs):
        if tag=='tr':self.row=[]
        if tag in ('td','th'):self.cell={'text':'','images':[]}
        if tag=='img'and self.cell is not None:self.cell['images'].append(dict(attrs).get('src'))
    def handle_data(self,data):
        if self.cell is not None:self.cell['text']+=data
    def handle_endtag(self,tag):
        if tag in ('td','th')and self.cell is not None:
            if self.row is not None:self.row.append(self.cell)
            self.cell=None
        if tag=='tr'and self.row is not None:self.rows.append(self.row);self.row=None

def main():
    page=fetch(BASE,'author_page.html',512000)
    table=Table();table.feed(page.decode('utf-8'))
    classes=json.loads((U.parent/'EXPERIMENT.json').read_text(encoding='utf-8-sig'))['classes']
    linked={}
    for row in table.rows:
        for i in range(len(row)-2):
            number=row[i]['text'].strip();images=row[i+1]['images'];name=row[i+2]['text'].strip()
            if number.isdigit()and images and name in classes:
                url=urljoin(BASE,images[0]);filename=url.rsplit('/',1)[-1]
                b=fetch(url,filename,256000)
                linked[name]={'author_class_id':int(number),'class_name':name,'url':url,'file':str(OUT/filename),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
                time.sleep(.05)
    assert len(linked)==45,(len(linked),set(classes)-set(linked))
    result={'author_page':BASE,'author_page_sha256':hashlib.sha256(page).hexdigest(),'statement':'Author table links all 45 observed class names to icons credited by author to Dictionary of New Zealand Sign Language. Mapping interpretation is separate, not author-supplied binary labels.','classes':[linked[x]for x in classes]}
    (OUT/'CHART_MANIFEST.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'images':len(linked),'bytes':sum(x['bytes']for x in linked.values()),'path':str(OUT/'CHART_MANIFEST.json')}))
if __name__=='__main__':main()
