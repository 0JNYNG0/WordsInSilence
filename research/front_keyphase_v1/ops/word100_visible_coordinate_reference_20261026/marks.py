"""RGB-only approximate regions, manually recorded by Codex before numerical scoring.
Each frame: anatomical left, anatomical right; each hand: wrist, thumb tip, index tip.
None explicitly means insufficient visible evidence, not a zero coordinate.
"""
MARKS={
 '1290':{
  52: [[[1008,449,1048,486],[892,469,913,492],None],[[850,404,889,444],[947,307,968,331],[1023,355,1046,378]]],
  63: [[[1004,469,1045,511],[889,509,915,532],None],[[848,416,887,459],[935,325,955,348],[1006,387,1030,411]]],
  80: [[[1008,504,1050,544],[891,529,915,551],None],[[849,402,890,446],[932,319,953,343],[1004,370,1030,394]]]
 },
 '0003':{
  46: [[[1076,544,1115,583],None,[931,612,956,640]],[[768,509,812,548],None,[940,471,962,490]]],
  56: [[[1105,545,1144,587],None,[951,593,979,622]],[[790,515,832,555],None,[950,481,976,501]]],
  72: [[[1107,527,1150,568],None,[969,496,993,516]],[[773,575,816,618],None,[944,579,971,603]]]
 },
 '0185':{
  42: [[[1097,894,1140,932],None,None],[[757,300,804,343],None,None]],
  56: [[[994,557,1034,600],[948,468,972,494],None],[[907,489,948,529],[999,415,1023,440],[950,359,974,386]]],
  78: [[[979,547,1024,586],None,None],[None,[1010,464,1035,490],[962,417,987,445]]]
 },
 '0920':{
  46: [[[977,388,1017,428],[1025,333,1050,355],None],[[883,482,926,526],[864,381,894,408],None]],
  60: [[None,[1027,331,1054,355],None],[[908,445,949,486],None,None]],
  83: [[[980,407,1024,451],[1040,345,1071,369],None],[[880,612,922,651],None,None]]
 },
 '1545':{
  57: [[[1090,540,1130,578],[1012,600,1041,627],[1065,590,1093,614]],[[1016,398,1054,442],None,[1143,356,1169,379]]],
  66: [[[1093,512,1135,550],[1016,576,1047,604],[1066,561,1093,585]],[[1023,440,1064,482],None,[1148,428,1174,453]]],
  80: [[None,[999,570,1031,598],[1050,560,1077,584]],[[954,468,997,510],None,[1090,473,1115,497]]]
 },
 '2029':{
  56: [[[984,393,1030,433],None,None],[[910,361,952,400],None,None]],
  64: [[None,[874,580,899,604],None],[[896,503,940,543],[1032,501,1056,524],[1040,569,1065,591]]],
  78: [[None,[878,572,902,597],None],[[896,495,937,536],[1033,492,1057,515],[1032,559,1064,583]]]
 }
}
NOTES={
 '1290':'Lower anatomical-left palm is visible, but the index fingertip cannot be separated confidently from other fingers in this view. Wrist regions include anatomical placement uncertainty.',
 '0003':'Visible pointing indices and wrist connections marked. Folded thumb endpoints are not consistently distinguishable from overlapping knuckles, so all thumbs withheld.',
 '0185':'Frame42 includes a resting left hand and raised right fist; tips not distinguishable. Frame56 left index folded/occluded. Frame78 right wrist covered by left hand; left fingertips uncertain. Visible right open-hand thumb/index preserved.',
 '0920':'Open palm/clustered fingers obscure individual index identity. Frame60 crossing hides left wrist and right fingertips. Frame83 right fingertips grouped tightly and thumb indistinguishable. Only distinguishable wrist/thumb endpoints marked.',
 '1545':'Right thumb folded and endpoint uncertain in all three frames. Frame80 left wrist junction overlapped by pointing right hand. Left thumb/index silhouettes and right pointing index visible.',
 '2029':'Frame56 fists are motion-blurred; tip identities withheld. Frames64/78 left wrist junction and index are overlapped by right hand; right thumb/index and left thumb silhouette remain visible.'
}
