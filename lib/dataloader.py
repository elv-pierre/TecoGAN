import tensorflow as tf
from lib.ops import *

import cv2 as cv
import collections, os, math
import scipy.misc as sic
import numpy as np
from scipy import signal

class OD:
    def __init__(self, filepaths, loader_fn, max_in_mem_files):
        self.filepaths = filepaths
        self.loader_fn = loader_fn
        self.max_in_mem_files = max_in_mem_files
        self.fifo = []
        self.files = {}

    def _in(self, index):
        return index in self.fifo

    def _load(self, index):
        self.fifo.append(index)
        self.files[index] = self.loader_fn(self.filepaths[index])
    
    def _update(self, index):
        self.fifo.remove(index)
        self.fifo.append(index)

    def _pop(self):
        index = self.fifo.pop(0)
        del self.files[index]

    def __getitem__(self, index):
        if self._in(index):
            self._update(index)
        elif len(self.fifo) < self.max_in_mem_files:
            self._load(index)
        else:
            self._pop()
            self._load(index)

        return self.files[index]

    def __len__(self):
        return len(self.filepaths)

# The inference data loader. 
# should be a png sequence
def inference_data_loader(FLAGS):

    filedir = FLAGS.input_dir_LR
    downSP = False
    if (FLAGS.input_dir_LR is None) or (not os.path.exists(FLAGS.input_dir_LR)):
        if (FLAGS.input_dir_HR is None) or (not os.path.exists(FLAGS.input_dir_HR)):
            raise ValueError('Input directory not found')
        filedir = FLAGS.input_dir_HR
        downSP = True
        
    image_list_LR_temp = os.listdir(filedir)
    image_list_LR_temp = [_ for _ in image_list_LR_temp if _.endswith(".png")] 
    image_list_LR_temp = sorted(image_list_LR_temp) # first sort according to abc, then sort according to 123
    image_list_LR_temp.sort(key=lambda f: int(''.join(list(filter(str.isdigit, f))) or -1))
    image_list_LR = [os.path.join(filedir, _) for _ in image_list_LR_temp]

    # Read in and preprocess the images
    def preprocess_test(name):
        im = sic.imread(name, mode="RGB").astype(np.float32)
        
        if downSP:
            icol_blur = cv.GaussianBlur( im, (0,0), sigmaX = 1.5)
            im = icol_blur[::4,::4,::]
        im = im / 255.0 #np.max(im)
        return im

    # image_LR = [preprocess_test(_) for _ in image_list_LR]
    if True: # a hard-coded symmetric padding
        image_list_LR = image_list_LR[5:0:-1] + image_list_LR
        # image_LR = image_LR[5:0:-1] + image_LR

    image_LR = OD(image_list_LR, preprocess_test, 64)

    Data = collections.namedtuple('Data', 'paths_LR, inputs')
    return Data(
        paths_LR=image_list_LR,
        inputs=image_LR
    )





