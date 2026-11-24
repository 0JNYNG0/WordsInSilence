"""Divide an ordered sequence into ceil(N/max_size) near-equal batches."""
def balanced_batches(order, max_size):
 if not isinstance(max_size,int) or isinstance(max_size,bool) or max_size<1:
  raise ValueError('max_size must be a positive integer')
 n=len(order)
 if n==0:return []
 count=(n+max_size-1)//max_size
 small,extra=divmod(n,count);out=[];start=0
 for i in range(count):
  size=small+(i<extra);out.append(order[start:start+size]);start+=size
 assert start==n
 return out
