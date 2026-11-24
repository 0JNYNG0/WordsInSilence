from apple_io import *
# Native RGB contact sheets reviewed BEFORE any candidate prediction.
# These are observed semantic anchors, NOT DTW-generated labels.
ANNOTATIONS={
 1:([43,45,48,50,53,55],[65,68,70,73,75,78,80,82],(43,55),(65,70),(75,78)),
 2:([47,50,52,55,58],[69,72,74,77,80,83,85,88,91,94],(47,58),(69,74),(80,85)),
 3:([44,47,50,54,57],[66,69,72,75,78,82,85,88,91],(44,57),(66,72),(78,85)),
 4:([63,66,69,72,75,78],[89,92,95,98,101,104,107,110,113,116],(63,78),(89,98),(101,107)),
 6:([25,28,31,34,38,41],[51,54,57,60,64,67,70,73,77,80,83],(25,41),(51,57),(64,70)),
 7:([52,55,58,61,64,66],[75,78,81,84,87,89,92,95,98],(52,66),(75,81),(87,89)),
 8:([58,61,64,67,70,73],[84,87,90,93,95,98,101],(58,73),(84,90),(93,98)),
 9:([51,53,56,58,61],[74,76,79,81,84,87,89,92],(51,61),(74,79),(84,87)),
 11:([47,50,53,56],[66,69,73,76,79,82,86,89,92,95,99],(47,56),(66,76),(82,89)),
 12:([46,49,52,54,57,60],[71,74,77,80,83,85,88,91,94],(46,60),(71,77),(83,88)),
 13:([43,45,48,50,53,55],[63,66,68,71,73,76,78,81,83,86,89],(43,55),(63,71),(76,81)),
 14:([34,37,40,44],[56,60,63,66,69,73,76,79,82,85,89],(34,44),(56,60),(69,73)),
 15:([49,52,55,59,62,65],[75,78,81,84,88,91,94,97,101,104],(49,65),(75,78),(88,91))}
def main():
 assert not(U/'SOURCE_TRUTH.json').exists()
 out=[]
 for r in read(U/'INPUTS.json')['rows']:
  a,b,sweep,wipe1,wipe2=ANNOTATIONS[r['person']]
  out.append(dict(clip_id=r['clip_id'],person=r['person'],role=r['role'],source_sha256=r['source_sha256'],
   mouth_anchors=a,pair_anchors=b,mouth_sweep=list(sweep),wipe_sweeps=[list(wipe1),list(wipe2)],
   first_only_end=sweep[1],wipe_before_descent=wipe1[0],
   right_first_fingers=[1,0,0,0],right_later='curved open palm, not tightly clenched',left_later='closed support hand',
   individual_later_fingers=None,exact_contact=None,review_image=str(U/'review'/f'p{r["person"]:02d}.jpg')))
 write(U/'SOURCE_TRUTH.json',dict(rows=out,reviewer='Codex native sequential RGB; no independent KSL expert',
  source=dict(title='국립국어원 수어, 또 하나의 언어',printed_page=11,
   url='https://www.korean.go.kr/common/download.do%3Bfront%3D412D56F0980C264000AB4D933878E108?c_file_name=c92dc946-eb7c-45bd-b867-26155dfb9e3e_0.pdf&file_path=reportData&o_file_name=%EC%88%98%EC%96%B4+%EB%98%90+%ED%95%98%EB%82%98%EC%9D%98+%EC%96%B8%EC%96%B4.pdf',
   paraphrase='검지로 입술 아래를 오른쪽으로 스친 뒤, 오른 손바닥으로 왼 주먹을 닦아 내린다.'),
  limits='Sparse manually reviewed anchors and endpoints. Unreviewed transition frames masked, no exact contact or invisible finger truth. Wipe endpoints will be checked against native close crops before completion admission.'))
 print('SOURCE_REVIEW',len(out))
if __name__=='__main__':main()
