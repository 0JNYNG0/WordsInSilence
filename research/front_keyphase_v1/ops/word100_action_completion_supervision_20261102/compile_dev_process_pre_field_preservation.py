"""Separate evaluation-only process annotations and unlabelled review views.

No FIT compiler, model, prediction, training or existing result is executed.
Rendering uses only already prepared public DEV native images. The new sheets
are review aids, never SignRep input or automatic transformed-action labels.
"""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
import argparse
import copy
import hashlib
import json
import os
import statistics
import traceback

U=Path(__file__).resolve().parent
OUT=Path('W:/WordsInSilence_derived/word100_action_completion_supervision_20261102/development_views')
WORDS=('0739','1133','1567','2523')
FILES=[U/f'REVIEW_DEV_{w}.json' for w in WORDS]
MAN=U/'DEV_PROCESS_REVIEW_MANIFEST.json'
TARGET=U/'DEV_PROCESS_TARGETS.json'
CONTRACT=U/'DEV_PARTIAL_CONTRACT.json'
INTENT=U/'DEV_PARTIAL_VIEWS_INTENT.json'
FINAL=U/'DEV_PARTIAL_VIEWS_MANIFEST.json'
HASH_CACHE={}

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    path=str(Path(path).resolve())
    if path not in HASH_CACHE:HASH_CACHE[path]=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return HASH_CACHE[path]
def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
def now():return datetime.now(timezone.utc).isoformat()
def key(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def fragments(ids):
    out=[]
    for f in ids:
        if not out or out[-1][1]+1!=f:out.append([f,f])
        else:out[-1][1]=f
    return out


def compile_targets():
    assert not TARGET.exists(), 'Existing DEV target compilation is immutable'
    assert all(p.exists() for p in FILES), 'Wait for all four direct review files; never compile partial coverage'
    m=read(MAN);intent=U/'DEV_PROCESS_REVIEW_INTENT.json'
    assert m['intent_sha256']==sha(intent)
    meta={r['id']:r for r in m['rows']}
    assert len(meta)==15 and len({r['clip_id'] for r in meta.values()})==12
    assert {r['person'] for r in meta.values()}=={11,12,13}
    frames_by_query={};unique={};pages=0
    for q in meta.values():
        assert q['role']=='development' and q['word_id'] in {'WORD'+w for w in WORDS}
        assert q['clip_id']==f"NIA_SL_{q['word_id']}_REAL{q['person']:02d}_F"
        assert sha(q['source'])==q['source_sha256']
        fs={f['frame']:f for f in q['frames']};lo,hi=q['display_bounds']
        assert sorted(fs)==list(range(lo,hi+1))
        assert all(fs[b]['timestamp_us']>fs[a]['timestamp_us'] for a,b in zip(sorted(fs),sorted(fs)[1:]))
        covered=[]
        for p in q['pages']:
            assert sha(p['path'])==p['sha256'];covered.extend(p['frames']);pages+=1
        assert covered==sorted(fs)
        for f in fs.values():
            assert sha(f['path'])==f['sha256']
            k=(q['clip_id'],f['frame']);v=(f['timestamp_us'],f['sha256'])
            assert k not in unique or unique[k]==v
            unique[k]=v
        frames_by_query[q['id']]=fs
    labels=[];counts=Counter();review_sources=[];original_preserved=0
    for file,w in zip(FILES,WORDS):
        doc=read(file)
        assert doc.get('model_predictions_used',False) is False
        assert doc.get('candidate_predictions_viewed',False) is False
        assert doc.get('new_training',0)==0
        assert doc.get('new_model_inference',0)==0 and doc.get('new_inference',0)==0
        if 'manifest_sha256' in doc:assert doc['manifest_sha256']==sha(MAN)
        rows=doc['rows']
        assert len(rows)==(6 if w=='1567' else 3)
        review_sources.append({'path':str(file),'sha256':sha(file)})
        for r in rows:
            q=meta[r['id']];fs=frames_by_query[r['id']]
            for k in ('clip_id','word_id','person','role','physical_query','source_sha256'):
                assert r[k]==q[k],(r['id'],k)
            assert r['word_id']=='WORD'+w
            assert {p['path']:p['sha256'] for p in r['reviewed_pages']}=={p['path']:p['sha256'] for p in q['pages']}
            declared=r.get('reviewed_native_frames',r.get('reviewed_all_native_frames'))
            if isinstance(declared,list):assert declared==sorted(fs)
            assert r['predicate_definitions']['setup_relation'] and r['predicate_definitions']['terminal_relation']
            assert r.get('whole_word_correctness') is None
            assert r.get('required_repetition_count') is None and r.get('required_extent') is None
            assert isinstance(r['review_bounds_sufficient'],bool)
            seen=set();relations=[]
            for p in r['relations']:
                f=p['frame'];assert type(f)==int and f in fs and f not in seen;seen.add(f)
                for k in ('setup_relation','terminal_relation'):
                    assert p[k] in (0,1,None);counts[k+':'+str(p[k])]+=1
                assert p['reason']
                if 'timestamp_us' in p:assert p['timestamp_us']==fs[f]['timestamp_us']
                z=copy.deepcopy(p);z['timestamp_us']=fs[f]['timestamp_us'];z['input_image']=copy.deepcopy(fs[f]);relations.append(z)
            assert relations
            motions=[]
            for p in r['motion_intervals']:
                lo,hi=p['start'],p['end'];assert type(lo)==type(hi)==int and lo<hi
                assert set(range(lo,hi+1))<=set(fs) and p['label'] in (0,1,None) and p['reason']
                times=[fs[lo]['timestamp_us'],fs[hi]['timestamp_us']]
                for field,value in [('start_us',times[0]),('end_us',times[1]),('timestamps_us',times)]:
                    if field in p:assert p[field]==value
                if 'reviewed_native_frames' in p:assert p['reviewed_native_frames']==list(range(lo,hi+1))
                z=copy.deepcopy(p);z['native_timestamps_us']=times;z['supervision_scope']='one reviewed interval, never dense per-frame targets'
                motions.append(z);counts['motion:'+str(p['label'])]+=1
            assert motions and len(r['sequence_links'])==1
            for seq in r['sequence_links']:
                assert seq['visible_chain'] in ('yes','partial','uncertain') and seq['reason']
                for f in seq['setup_frames']+seq['terminal_frames']:assert type(f)==int and f in fs
                lo,hi=seq['motion_interval'];assert lo<hi and set(range(lo,hi+1))<=set(fs)
                assert seq.get('linguistic_sufficiency') is None and seq.get('whole_clause_truth') is None
                counts['chain:'+seq['visible_chain']]+=1
                # A sparse frame omitted from relation points is unknown, not0.
                byframe={p['frame']:p for p in r['relations']}
                if seq['visible_chain']=='yes':
                    assert seq['setup_frames'] and seq['terminal_frames']
                    assert all(byframe.get(f,{}).get('setup_relation')!=0 for f in seq['setup_frames'])
                    assert all(byframe.get(f,{}).get('terminal_relation')!=0 for f in seq['terminal_frames'])
            row=copy.deepcopy(r)
            row.update(source=q['source'],relations=relations,motion_intervals=motions,
                review_bounds=q['display_bounds'],unlisted_is_unknown=True,linguistic_correctness=None,
                source_annotation=str(file),source_annotation_sha256=sha(file),
                repeated_development=True,independent_final_test=False,training_allowed=False)
            # Every original field is retained; enriched relations and intervals
            # contain every original nested field with unchanged values.
            for k,value in r.items():
                if k in ('relations','motion_intervals'):
                    for before,after in zip(value,row[k]):
                        assert all(after[x]==v for x,v in before.items())
                else:assert row[k]==value
            original_preserved+=1;labels.append(row)
    assert len(labels)==15 and {r['id'] for r in labels}==set(meta)
    pins={str(p):sha(p) for p in [MAN,intent,*FILES,U/'PROCESS_LABEL_PROTOCOL.json',U/'PHASE_LABEL_ADDENDUM.json',Path(__file__)]}
    out={'at':now(),'status':'DIRECT_DEV_PHYSICAL_PROCESS_TARGETS_PREPARED_EVALUATION_ONLY',
       'rows':labels,'queries':15,'source_clips':12,'registered_ids':[11,12,13],
       'unique_source_frames':len(unique),'direct_pages':pages,'counts':dict(counts),'review_sources':review_sources,'pins':pins,
       'training_allowed':False,'repeated_development':True,'independent_final_test':False,
       'original_annotation_rows_preserved':original_preserved,'intervals_not_densified':True,
       'source_variants_preserved':True,'independent_human_expert':False,
       'new_inference':0,'new_training':0,'whole_goal_complete':False}
    write_new(TARGET,out)
    print('Compiled DEV targets:',len(labels),dict(counts),flush=True)


def prepare_views():
    assert TARGET.exists() and not FINAL.exists() and not INTENT.exists()
    assert not OUT.exists(), 'Inspect partial view output before any recovery'
    targets=read(TARGET)
    for p,h in targets['pins'].items():assert sha(p)==h,p
    rows=targets['rows'];review={r['id']:r for r in read(MAN)['rows']}
    eligible=[];skipped=[]
    for r in rows:
        assert r['role']=='development' and r['person'] in (11,12,13) and r['training_allowed'] is False
        seq=r['sequence_links'][0];lo,hi=r['review_bounds']
        a=max(seq['setup_frames']) if seq['setup_frames'] else None
        b=min(seq['terminal_frames']) if seq['terminal_frames'] else None
        ok=r['review_bounds_sufficient'] and seq['visible_chain']=='yes' and a is not None and b is not None and lo<=a<b<=hi and b-a-1>=1
        if ok:eligible.append(r)
        else:skipped.append({'query_id':r['id'],'original_chain':copy.deepcopy(seq),'reason':'No supported separated anchors. Native observations/variants remain; no substitute cut or label.'})
    sentinels=[];sentinel_skips=[]
    all_types=sorted({r['physical_query'] for r in rows});assert len(all_types)==5
    for typ in all_types:
        fixed=sorted([r for r in rows if r['physical_query']==typ and r['person']==11],key=lambda r:r['id'])
        assert fixed
        r=fixed[0]
        if r in eligible:sentinels.append(r['id'])
        else:sentinel_skips.append({'query_id':r['id'],'reason':'Fixed ID11 lacks supported anchors; do not replace by a more convenient person.'})
    contract={'at':now(),'status':'PREPARE_ONLY_NO_AUTOMATIC_EDIT_LABELS','training_allowed':False,
      'evaluation_people':[11,12,13],'same_physical_protocol_as_fit':True,
      'native':'All15 query aliases, full original display_bounds, including uncertain variants.',
      'primary':'For directly supported yes with sufficient bounds: a=max(setup), b=min(terminal), require L<=a<b<=U and b-a-1>=1. prefix=L..a; suffix=b..U; splice=L..a+b..U. One fixed cut only.',
      'sentinels':'Lexicographically first query of each of5 physical queries at fixedID11 only. No outcome/prediction selection or fallback person.',
      'sentinel_types':{'speed_0_8':.8,'speed_1_25':1.25,'intact_context_trim':'min/max of existing setup+motion+terminal witnesses','missing_rgb':'first positive reviewed motion interval wholly inside sequence interval; uniform128 replaces those pixels'},
      'view_time':'Preserve native deltas within each contiguous fragment. At splice use median native frame period; speed divides elapsed deltas. Keep original source PTS separately.',
      'no_shortcuts_as_gold':['edit kind is not label','endpoints do not certify a process','deleted selected occurrence does not mean all occurrences absent','blank pixels are unknown','speed sheets are not speed-video validation'],
      'render':'All retained original full images, no crops; neutral header/no editkind label/no seam highlight. Only source frame and new time printed.',
      'labels':'All view labels null until actual view review; original labels stay separately linked.',
      'model_input':'Future inference must re-extract actual transformed pixels; never copy/reorder clean temporal features.',
      'eligible_queries':[r['id'] for r in eligible],'skipped':skipped,'fixed_sentinels':sentinels,'sentinel_skips':sentinel_skips,
      'pins':{str(p):sha(p) for p in [TARGET,MAN,U/'PROCESS_LABEL_PROTOCOL.json',U/'PHASE_LABEL_ADDENDUM.json',U/'PARTIAL_CONTROL_CONTRACT.json',Path(__file__)]},
      'new_inference':0,'new_training':0}
    assert not CONTRACT.exists();write_new(CONTRACT,contract)
    views=[];media={}
    def add(r,kind,ids,speed=1.,blank=()):
        q=review[r['id']];fs={f['frame']:f for f in q['frames']};lo,hi=r['review_bounds']
        assert ids and ids==sorted(set(ids)) and set(ids)<=set(fs)
        period=statistics.median(fs[f+1]['timestamp_us']-fs[f]['timestamp_us'] for f in range(lo,hi))
        mapping=[];elapsed=0.;seams=[];blank=set(blank)
        for j,f in enumerate(ids):
            x=fs[f]
            assert sha(x['path'])==x['sha256']
            if j:
                adjacent=f==ids[j-1]+1
                elapsed+=(x['timestamp_us']-fs[ids[j-1]]['timestamp_us'] if adjacent else period)/speed
                if not adjacent:seams.append(j)
            mapping.append({'output_index':j,'source_native_frame':f,'source_native_timestamp_us':x['timestamp_us'],
              'source_frame_path':x['path'],'source_frame_sha256':x['sha256'],'view_timestamp_us':round(elapsed),
              'pixel_operation':'blackout' if f in blank else 'copy','image_available':f not in blank})
        digest=key({'source_sha256':r['source_sha256'],'frames':[(x['source_native_frame'],x['pixel_operation'],x['view_timestamp_us']) for x in mapping]})
        retained=set(ids);account=[]
        for m in r['motion_intervals']:
            whole=set(range(m['start'],m['end']+1));part=whole&retained
            account.append({'original_interval':[m['start'],m['end']],'original_label_audit_only':m['label'],
              'retained_native_fragments':fragments(sorted(part)),'removed_native_spans':fragments(sorted(whole-part)),
              'retention':'fully_retained' if part==whole else 'wholly_removed' if not part else 'partially_retained',
              'remainder_label':None,'blackout_frames':sorted(part&blank)})
        seq=r['sequence_links'][0]
        v={'view_id':r['id']+'__'+kind,'query_id':r['id'],'clip_id':r['clip_id'],'recorded_person_id':r['person'],'role':'development',
          'physical_query':r['physical_query'],'source_path':r['source'],'source_sha256':r['source_sha256'],
          'original_review_path':r['source_annotation'],'original_review_sha256':r['source_annotation_sha256'],'sequence_link_index':0,
          'reviewed_bounds':[lo,hi],'a':max(seq['setup_frames']) if seq['setup_frames'] else None,'b':min(seq['terminal_frames']) if seq['terminal_frames'] else None,
          'construction_kind_audit_only':kind,'pixel_view_key':digest,
          'query_review_key':key([digest,r['physical_query'],key(r['predicate_definitions'])]),'ordered_frame_map':mapping,
          'retained_contiguous_native_fragments':fragments(ids),'removed_native_spans':fragments(sorted(set(fs)-retained)),
          'seam_output_indices':seams,'motion_retention_audit_only':account,'labels_status':'pending_direct_review','labels':None,
          'original_scope_variant':copy.deepcopy(r.get('actual_terminal_variant')),'speed':speed,
          'training_allowed':False,'repeated_development':True,'independent_final_test':False,
          'actual_retimed_video_prepared':False,'model_input_prepared':False,'review_overlay_not_model_input':True}
        views.append(v);media.setdefault(digest,v)
    good={r['id'] for r in eligible}
    for r in rows:
        lo,hi=r['review_bounds'];seq=r['sequence_links'][0];add(r,'native',list(range(lo,hi+1)))
        if r['id'] not in good:continue
        a=max(seq['setup_frames']);b=min(seq['terminal_frames'])
        add(r,'prefix',list(range(lo,a+1)));add(r,'suffix',list(range(b,hi+1)));add(r,'middle_splice',list(range(lo,a+1))+list(range(b,hi+1)))
        if r['id'] in sentinels:
            add(r,'speed_0_8',list(range(lo,hi+1)),.8);add(r,'speed_1_25',list(range(lo,hi+1)),1.25)
            anchors=seq['setup_frames']+seq['motion_interval']+seq['terminal_frames']
            add(r,'intact_context_trim',list(range(min(anchors),max(anchors)+1)))
            pos=sorted([m for m in r['motion_intervals'] if m['label']==1 and seq['motion_interval'][0]<=m['start']<m['end']<=seq['motion_interval'][1]],key=lambda m:(m['start'],m['end']))
            assert pos,'Fixed blackout needs explicit positive interval, never invent one'
            add(r,'missing_rgb',list(range(lo,hi+1)),blank=range(pos[0]['start'],pos[0]['end']+1))
    assert len(views)==15+3*len(eligible)+4*len(sentinels)
    pins={str(p):sha(p) for p in [CONTRACT,TARGET,MAN,*FILES,Path(__file__)]}
    write_new(INTENT,{'at':now(),'status':'FROZEN_DEV_REVIEW_VIEWS_NO_EDIT_LABELS','pins':pins,'views':views,'sentinels':sentinels,'skipped':skipped,'training_allowed':False,'new_inference':0,'new_training':0})
    from PIL import Image,ImageDraw,ImageFont
    OUT.mkdir(parents=True,exist_ok=False)
    font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',14);rendered={}
    for i,(digest,v) in enumerate(media.items()):
        entries=[];maps=v['ordered_frame_map']
        for j in range(0,len(maps),25):
            part=maps[j:j+25];cw,ch=300,224
            sheet=Image.new('RGB',(cw*5,76+ch*((len(part)+4)//5)),'#f5f6f8');d=ImageDraw.Draw(sheet)
            d.text((8,5),v['clip_id']+' | view '+digest[:12],font=font,fill='#111827')
            d.text((8,28),'FULL source frame | ALL displayed images | source frame / view time | NOT labels',font=font,fill='#111827')
            for k,e in enumerate(part):
                x=(k%5)*cw+4;y=76+(k//5)*ch
                if e['image_available']:
                    with Image.open(e['source_frame_path']) as src:image=src.copy().convert('RGB')
                    image.thumbnail((cw-8,ch-31))
                else:image=Image.new('RGB',(cw-8,ch-31),(128,128,128))
                sheet.paste(image,(x,y+29));d.text((x,y),f"f{e['source_native_frame']} / {e['view_timestamp_us']/1e6:.3f}s",font=font,fill='#111827')
            destination=OUT/(digest[:20]+f'_p{j//25+1}.jpg');sheet.save(destination,quality=92)
            entries.append({'path':str(destination),'sha256':sha(destination),'output_indices':[e['output_index'] for e in part],'source_native_frames':[e['source_native_frame'] for e in part]})
        rendered[digest]=entries
        if i%20==0:print('DEV review views',i+1,'/',len(media),flush=True)
    counts=Counter()
    for v in views:
        v['review_pages']=rendered[v['pixel_view_key']];v['native_comparison_pages']=review[v['query_id']]['pages'];counts[v['construction_kind_audit_only']]+=1
    for p,h in pins.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    write_new(FINAL,{'at':now(),'status':'PREPARED_PENDING_DIRECT_DEV_VIEW_REVIEW','views':views,'skipped':skipped,
       'counts':dict(counts),'queries':15,'source_clips':12,'unique_pixel_time_views':len(media),'pages':sum(map(len,rendered.values())),
       'intent_sha256':sha(INTENT),'pins':pins,'training_allowed':False,'repeated_development':True,'independent_final_test':False,
       'new_inference':0,'new_training':0,'actual_transformed_videos_prepared':0,'whole_goal_complete':False})
    print('Prepared DEV views:',dict(counts),'unique',len(media),'pages',sum(map(len,rendered.values())),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--compile-only',action='store_true');parser.add_argument('--render-only',action='store_true');args=parser.parse_args()
    assert not (args.compile_only and args.render_only)
    try:
        if not args.render_only:compile_targets()
        if not args.compile_only:prepare_views()
    except BaseException:
        p=U/'DEV_PROCESS_COMPILATION_FIRST_FAILURE.txt'
        if not p.exists():p.write_text(traceback.format_exc(),encoding='utf-8')
        raise
