"""Sequential follow-up; previous failed comparison preserved."""
import check_lesson as check
from visible_lesson import *
from target_region import region_distance

def main():
 torch.set_num_threads(2);dest=V/'extent_checks';dest.mkdir(exist_ok=True);assert not(dest/'FROZEN.json').exists()
 write(dest/'FROZEN.json',dict(
  cause='Arrival anchor was treated as final frame, cutting off the actual terminal hold; absolute positive duration also rejected a pure speed change.',
  repair='Observe same action through terminal hold, stop before learned departure/background/missing/next event. Positive shape requires three observed consecutive states independent of pace.',
  limitations='No smoothing or fabricated state fills; no promotion before distorted shape/plane and held source checks.',
  code={p.name:sha(p)for p in[V/'visible_lesson.py',V/'target_region.py']}))
 # Reuse exactly the prior fixed suite under a new output path.
 oldV=check.V
 class Location:
  def __truediv__(self,x):return dest if x=='visible_checks'else oldV/x
 check.V=Location();check.main()
if __name__=='__main__':main()
