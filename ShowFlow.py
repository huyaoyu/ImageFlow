from __future__ import print_function

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from CommonType import NP_FLOAT

from ImageFlow import show, show_as_KITTI, calculate_angle_distance_from_du_dv
from ImageFlow import SELF_OCC, CROSS_OCC, OUT_OF_FOV_POSITIVE_Z
from Utils import test_dir

def save_sample_hsv_linear(fn, h=128, w=256):
    img = np.zeros((h, w, 3), dtype=NP_FLOAT)

    img[:, :, 1] = 255
    img[:, :, 2] = 255

    x = np.linspace(0, w-1, w, endpoint=True, dtype=NP_FLOAT) / ( w-1 ) * 255

    for i in range(h):
        img[i, :, 0] = x

    img = img.astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    cv2.imwrite(fn, img)

def save_sample_hsv(fn, h=256, w=256):
    x = np.linspace(0, w-1, w, endpoint=True, dtype=np.int)
    y = np.linspace(0, h-1, h, endpoint=True, dtype=np.int)

    x, y = np.meshgrid( x, y )

    du = x - w/2
    dv = y - h/2

    y = (h - 1) - y

    a, d, angleShift = calculate_angle_distance_from_du_dv( du, dv, flagDegree=False )

    parts = os.path.split(fn)

    hueMax = 179

    key = 0

    while ( key != 113 ):
        key = show_as_KITTI(a, d, None, 8, None, None, None, waitTime=None, hueMax=hueMax)

        if ( 106 == key ):
            hueMax -= 1
            if ( hueMax < 0 ):
                hueMax = 0
        elif ( 107 == key ):
            hueMax += 1
            if ( hueMax > 255 ):
                hueMax = 255

        print("hueMax = %d. " % (hueMax))

    key = show_as_KITTI(a, d, None, 8, None, parts[0], "%s_hue%d" % (parts[1], hueMax), waitTime=None, hueMax=hueMax)

    np.savetxt("%s.txt" % (fn), a, fmt="%+.2f")

    cv2.destroyAllWindows()

def plot_vector_field(du, dv, mask=None, outDir=None, outName="bgr", magFactor=1.0,):
    # Generate x and y coordinates.
    h, w = du.shape[:2]

    x = np.linspace(0, w-1, w, endpoint=True, dtype=np.int)
    y = np.linspace(0, h-1, h, endpoint=True, dtype=np.int)

    # Flip the y-axis.
    y = (h - 1) - y
    dv = dv.max() - dv

    x, y = np.meshgrid( x, y )

    # import ipdb; ipdb.set_trace()

    # plt.quiver( x, y, du*magFactor, dv*magFactor, units="width", width=1)

    h0, h1 = 28, 67
    w0, w1 = 390, 428

    plt.quiver( x[h0:h1, w0:w1], y[h0:h1, w0:w1], du[h0:h1, w0:w1]*magFactor, dv[h0:h1, w0:w1]*magFactor, units="xy")
    # plt.ion()
    plt.show()
    # plt.pause(0.001)

class Args(object):
    def __init__(self, flow):
        super(Args, self).__init__()

        self.flow                  = flow
        self.mask                  = ""
        self.mf                    = 1.0
        self.ignore_fov_mask       = False
        self.ignore_cross_occ_mask = False
        self.ignore_self_occ_mask  = False
        self.ignore_all_occ_mask   = False
        self.write_dir             = ""
        self.kitti                 = False
        self.max_f                 = 0
        self.vector                = False
        self.not_show              = False
        self.hue_max               = 159

    def copy_args(self, args):
        self.flow                  = args.flow
        self.mask                  = args.mask
        self.mf                    = args.mf
        self.ignore_fov_mask       = args.ignore_fov_mask
        self.ignore_cross_occ_mask = args.ignore_cross_occ_mask
        self.ignore_self_occ_mask  = args.ignore_self_occ_mask
        self.ignore_all_occ_mask   = args.ignore_all_occ_mask
        self.write_dir             = args.write_dir
        self.kitti                 = args.kitti
        self.max_f                 = args.max_f
        self.vector                = args.vector
        self.not_show              = args.not_show
        self.hue_max               = args.hue_max

    def make_parser(self):
        parser = argparse.ArgumentParser(description='Visualize an optical flow.')

        parser.add_argument("flow", type=str, \
            help="The filename of the flow.")

        parser.add_argument("--mask", type=str, default="", \
            help="The filename of the mask")
        
        parser.add_argument("--mf", type=float, default=1.0, \
            help="The amplification factor.")
        
        parser.add_argument("--ignore-fov-mask", action="store_true", default=False, \
            help="Ignore the out-of-FOV mask label.")
        
        parser.add_argument("--ignore-cross-occ-mask", action="store_true", default=False, \
            help="Ignore the cross-ossclusion mask label.")

        parser.add_argument("--ignore-self-occ-mask", action="store_true", default=False, \
            help="Ignore the self-occlusion mask label.")

        parser.add_argument("--ignore-all-occ-mask", action="store_true", default=False, \
            help="Equivalent to setting --ignore-self-occ-mask and --ignore-self-occ-mask at the same time.")
        
        parser.add_argument("--write-dir", type=str, default="", \
            help="Specify this argument for writing images to the file system. The file name will be determined with respect to the input file.")

        parser.add_argument("--kitti", action="store_true", default=False, \
            help="Show the image similary to KITTI dataset.")

        parser.add_argument("--max-f", type=float, default=0, \
            help="The max optical flow value for KITTI.")
        
        parser.add_argument("--vector", action="store_true", default=False, \
            help="Plot the vector field.")

        parser.add_argument("--not-show", action="store_true", default=False, \
            help="Set this flag to disable showing the images.")
        
        parser.add_argument("--hue-max", type=int, default=159, \
            help="The max hue value. Should be positive and under 180.")

        return parser

    def parse_args(self, parser):
        args = parser.parse_args()

        self.copy_args(args)

        return args

def run(args):
    # Open the flow file.
    flow = np.load(args.flow)

    # Open the mask file.
    if ( "" == args.mask ):
        # No mask file specified.
        mask = np.zeros(flow.shape[:2], dtype=np.uint8) + 255
    else:
        mask = np.load(args.mask)

    suffix = ""

    # Masking.
    if ( args.ignore_fov_mask ):
        tempMask = mask == OUT_OF_FOV_POSITIVE_Z
        mask[tempMask] = 255
        suffix += "_nf"

    if ( args.ignore_all_occ_mask ):
        tempMask = mask == SELF_OCC
        mask[tempMask] = 255
        tempMask = mask == CROSS_OCC
        mask[tempMask] = 255
        suffix += "_nc_ns"
    else:
        if ( args.ignore_self_occ_mask ):
            tempMask = mask == SELF_OCC
            mask[tempMask] = 255
            suffix += "_ns"
        
        if ( args.ignore_cross_occ_mask ):
            tempMask = mask == CROSS_OCC
            mask[tempMask] = 255
            suffix += "_nc"
    
    # Calculate the angle and distance.
    a, d, angleShift = calculate_angle_distance_from_du_dv( flow[:, :, 0], flow[:, :, 1], flagDegree=False )

    if ( "" != args.write_dir ):
        test_dir( args.write_dir )
        outDir  = args.write_dir
        outName = "%s" % ( os.path.splitext( os.path.split( args.flow )[1] )[0] )
    else:
        outDir  = None
        outName = ""

    if ( args.not_show ):
        flagShow = False
    else:
        flagShow = True

    # Plot the vector field.
    if ( args.vector ):
        plot_vector_field( flow[:, :, 0], flow[:, :, 1], mask )
    else:
        outName += suffix

        # Show the image.
        if ( args.kitti ):
            if ( args.max_f <= 0 ):
                maxF = None
            else:
                maxF = args.max_f
            
            show_as_KITTI(a, d, maxF, 8, mask, outDir, outName, waitTime=None, flagShowFigure=flagShow, hueMax=args.hue_max)
        else:
            show(a, d, mask, outDir, outName, waitTime=None, flagShowFigure=flagShow, magFactor=args.mf, hueMax=args.hue_max)

if __name__ == "__main__":
    args = Args(None)
    parser = args.make_parser()
    args.parse_args(parser)

    run(args)
