import sys
import cv2
import numpy as np
np.set_printoptions(threshold=np.nan)
import scipy.misc as misc

fn_curProb = sys.argv[1]

arr = misc.imread(fn_curProb)
print arr.shape
print arr

#cur_prob = cv2.imread(fn_curProb,0)
#print 'cur_prob shape:', cur_prob.shape
#print cur_prob

#cur_prob = np.clip( (cur_prob.astype('f4'))/(np.iinfo(cur_prob.dtype).max),0.,1.)
#print 'clip'
#print cur_prob
