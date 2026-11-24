from pathlib import Path
import json,html
U=Path(__file__).resolve().parent
def read(n):return json.loads((U/n).read_text(encoding='utf-8-sig'))
def main():
    payload={'summary':read('SUMMARY.json'),'observations':read('OBSERVATIONS.json')['clips'],
        'pairs':read('PAIRED_REPLAY.json')['pairs'],'guides':read('GUIDE_CARDS.json')['cards'],
        'atoms':read('GUIDE_CLAIM_LINKS.json')['atoms'],'controls':read('CONTROLS.json')['cases']}
    text=(U/'report_template.html').read_text(encoding='utf-8')
    text=text.replace('__PAYLOAD__',json.dumps(payload,ensure_ascii=False,allow_nan=False).replace('</','<\\/'))
    (U/'RESULT_KO.html').write_text(text,encoding='utf-8')
    print('Saved RESULT_KO.html',len(text),'characters')
if __name__=='__main__':main()
