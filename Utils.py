from __future__ import print_function

import copy
import cv2
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import os
import pandas

from CommonType import NP_FLOAT

def show_delimiter(title = "", c = "=", n = 50, leading = "\n", ending = "\n"):
    d = [c for i in range( int(n/2) )]
    s = "".join(d) + " " + title + " " + "".join(d)

    print("%s%s%s" % (leading, s, ending))

def test_dir(d):
    if ( False == os.path.isdir(d) ):
        os.makedirs(d)

def save_float_image(fn, img):
    img = (img - img.min()) / ( img.max() - img.min() ) * 255

    img = img.astype(np.uint8)

    cv2.imwrite( fn, img )

def estimate_loops(N, step):
    """
    N and step will be converted to integers.
    step must less than N.
    """

    N = (int)( N )
    step = (int)( step )

    loops = N / step

    if ( step * loops + 1 > N ):
        loops -= 1

    return loops