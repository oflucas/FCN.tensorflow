import sys,os
from glob import glob
import shutil
import numpy as np
import scipy.misc as misc

class Confg: 
    '''
    All the defines go in here!
    '''
    
    cats = ['um_lane', 'um_road', 'umm_road', 'uu_road']
    calib_end = '.txt'
    im_end = '.png'
    gt_end = '.png'
    prob_end = '.png'
    eval_propertyList = ['MaxF', 'AvgPrec', 'PRE_wp', 'REC_wp', 'FPR_wp', 'FNR_wp' ] 

    gt_dir = 'gt_image_2'
    an_road_dir = 'annotations_road'
    an_lane_dir = 'annotations_lane'

def annotate():
    """
    Now ground truth is stored as unsigned char color PNG
    - first/R plane contains valid evaluation area
        -> Exclusion of road areas that are not relevant for evaluation
    - third/B plane contains road area ground truth
    """
    for dr in [Confg.an_road_dir, Confg.an_lane_dir]:
        if not os.path.exists(dr):
            os.makedirs(dr)
    
    seen_shape=(0, 0)

    fn_search  = '*%s' % Confg.gt_end
    gt_fileList = glob(os.path.join(Confg.gt_dir, fn_search))
    assert len(gt_fileList)>0, 'Error reading ground truth'
    
    for fn_curGt in gt_fileList:
        file_key = fn_curGt.split('/')[-1].split('.')[0]
        cat, road_type, id_str = tuple(file_key.split('_'))

        an_dir = Confg.an_road_dir if road_type == 'road' else Confg.an_lane_dir

        gt = misc.imread(fn_curGt)
        anno = np.zeros_like(gt[:,:,2], dtype=np.uint8)
        anno[gt[:,:,2] > 0] = 1 # third/B place is G.T.
        if anno.shape != seen_shape:
            seen_shape = anno.shape
            print 'annotation shapes:', anno.shape

        an_file = os.path.join(an_dir, file_key + Confg.gt_end)
        misc.imsave(an_file, anno)


if __name__ == "__main__":
    annotate()
