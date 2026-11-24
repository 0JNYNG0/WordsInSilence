from train_motion import *

def main():
 torch.set_num_threads(2);new=RegionMotion();old=mm.Observer('mlp32');by={r['id']:r for r in all_sources()};cid='NIA_SL_WORD1290_REAL17_R';ch=channels(dict(np.load(by[cid]['raw'])));a=old.sequence(ch);b=new.sequence(ch);rows=[dict(frame=i,before=int(a[i]),candidate=int(b[i]))for i in range(89,102)]
 write(M/'motion/SHORT_RETURN_CHECK.json',dict(rows=rows,scope='Diagnostic reused DEV view; not independent validation'))
 print(rows,flush=True)
if __name__=='__main__':main()
