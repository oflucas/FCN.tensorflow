__author__ = 'oflucas'
import numpy as np
import os
import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob

import TensorflowUtils as utils

DATA_URL = ''
TRAIN_DATA_DIR = 'Data_zoo/kitti_road/data_road/training'
# sub dirs: annotations_lane  annotations_road  calib  gt_image_2  image_2

def read_dataset(data_dir=TRAIN_DATA_DIR):
    """
    go to training data dir and retrive:
        training_records, validation_records
    where records are:
        [{'image': image_path, 'annotation': annotation_path},,,]
    """
    assert os.path.exists(data_dir), "Cannot find dir = " + data_dir
    
    anno_dir = os.path.join(data_dir, 'annotations_road')
    im_dir = os.path.join(data_dir, 'image_2')
    records = retrive_records(anno_dir, im_dir)

    training_records, validation_records = split_records(records)
    return training_records, validation_records


def retrive_records(anno_dir, im_dir):
    res = []
    annos = glob.glob(os.path.join(anno_dir, '*.png'))
    for anno in annos:
        fname = os.path.splitext(anno.split("/")[-1])[0]
        cat, road_type, idx_str = tuple(fname.split('_'))
        
        im = os.path.join(im_dir, cat+'_'+idx_str+'.png')
        if not os.path.exists(im):
            print '[ERROR] Annotation-Image Pair File Not Found:', im
            continue

        res.append({'image': im, 'annotation': anno})

    return res


def split_records(records, validation_ratio=0.2):
    random.shuffle(records)
    val_size = int(validation_ratio * len(records))
    return records[:val_size], records[val_size:]
