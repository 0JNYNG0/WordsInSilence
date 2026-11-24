"""Offline runner, only after a separately approved local model acquisition.

No from_pretrained remote IDs, downloads, remote code, video uploads or training.
Use --preflight to validate inputs without importing Transformers or loading weights.
"""
import argparse,os,json,hashlib,time
from pathlib import Path
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model-dir',type=Path);ap.add_argument('--case',default='clip_01');ap.add_argument('--preflight',action='store_true');args=ap.parse_args()
    data=read(U/'VLM_INPUTS.json');rows=[r for r in data['rows'] if r['case_id']==args.case];assert len(rows)==1
    row=rows[0]
    for f in row['frames']:
        assert hashlib.sha256(Path(f['path']).read_bytes()).hexdigest()==f['file_sha256']
    if args.preflight:
        print(json.dumps({'input_valid':True,'case':args.case,'images':len(row['frames']),'model_loaded':False}));return
    if args.model_dir is None or not args.model_dir.is_dir():raise SystemExit('Local model not supplied. This runner will not download one.')
    approval=U/'DOWNLOAD_AUTHORIZATION.json'
    if not approval.exists() or read(approval).get('user_confirmed') is not True:raise SystemExit('Specific acquisition confirmation is not recorded. No inference started.')
    target=U/'vlm_outputs'/f'{args.case}.json'
    if target.exists():raise SystemExit('Stored trial output already exists; no automatic re-inference.')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',TOKENIZERS_PARALLELISM='false')
    from PIL import Image
    import torch
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    start=time.perf_counter()
    processor=AutoProcessor.from_pretrained(str(args.model_dir),local_files_only=True,trust_remote_code=False)
    model=Qwen3VLForConditionalGeneration.from_pretrained(str(args.model_dir),local_files_only=True,trust_remote_code=False,
        dtype=torch.float32,attn_implementation='sdpa').eval()
    load_seconds=time.perf_counter()-start
    content=[];images=[]
    for f in row['frames']:
        content.append({'type':'text','text':f"Time {f['timestamp_seconds']:.6f} seconds."})
        # Keep aspect ratio, enough pixels for a feasibility trial; downsampling
        # itself is a limitation for finger detail and is logged explicitly.
        im=Image.open(f['path']).convert('RGB');im.thumbnail((560,560));images.append(im)
        content.append({'type':'image','image':im})
    content.append({'type':'text','text':row['prompt']})
    messages=[{'role':'user','content':content}]
    t=time.perf_counter()
    inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt')
    with torch.inference_mode():out=model.generate(**inputs,do_sample=False,max_new_tokens=512)
    response=processor.batch_decode(out[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0]
    result={'case_id':args.case,'model_local':str(args.model_dir.resolve()),'model_id':data['model'],'dtype':'FP32','threads':4,
        'image_sizes':[list(im.size) for im in images],'source_frames':row['frames'],'prompt':row['prompt'],
        'evaluation_metadata_sent':False,'response':response,'model_load_seconds':load_seconds,
        'preprocess_and_generate_seconds':time.perf_counter()-t,'educational_correctness_verified':False}
    target.parent.mkdir(exist_ok=True);target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(target),'load_seconds':load_seconds,'trial_seconds':result['preprocess_and_generate_seconds']}))
if __name__=='__main__':main()
