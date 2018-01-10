import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os, sys
from six.moves import urllib
import tarfile
import zipfile
import scipy.io

filepath = sys.argv[1]
data = scipy.io.loadmat(filepath)

for k, v in data.items():
    print k, ':', type(v), v.shape if isinstance(v, np.ndarray) else '.'
print 'dict:', data
