from __future__ import print_function

import argparse
import cv2
import numpy as np
import os

from CommonType import NP_FLOAT

from ImageStereo import STEREO_OUT_OF_FOV, STEREO_SELF_OCC, STEREO_CROSS_OCC
from Utils import test_dir, normalize_float_image, save_float_image_PNG

def visualize_disparity_HSV(disp, mask=None, outDir=None, outName=None, maxDisp=None, waitTime=None, flagShowFigure=True, maxHue=179, n=8):
    # Normalize the disparity.
    if ( maxDisp is None ):
        maxDisp = disp.max()
    
    hsv = np.zeros((disp.shape[0], disp.shape[1], 3), dtype=NP_FLOAT)

    hsv[:, :, 0] = np.clip(disp / maxDisp, 0, 1) * maxHue
    hsv[:, :, 1] = disp / maxDisp * n
    hsv[:, :, 2] = ( n - hsv[:, :, 1] ) / n

    hsv[:, :, 1] = np.clip( hsv[:, :, 1], 0, 1 ) * 255
    hsv[:, :, 2] = np.clip( hsv[:, :, 2], 0.75, 1 ) * 255

    hsv = hsv.astype(np.uint8)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if ( mask is not None ):
        mask = mask != 255
        bgr[mask] = np.array([0, 0 ,0], dtype=np.uint8)

    if ( outDir is not None ):
        cv2.imwrite(outDir + "/%s_vis.png" % (outName), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    key = None

    if ( True == flagShowFigure ):
        cv2.imshow("HSV2GBR", bgr)

        if ( waitTime is None ):
            key = cv2.waitKey()
        else:
            key = cv2.waitKey( waitTime )

    return key

def visualize_disparity_KITTI(disp, fn, maxDisp=None):
    """
    This function is the reproduction of the MATLAB code provided
    by the KITTI dataset.
    """

    h, w = disp.shape

    # Normalize the disparity.
    if ( maxDisp is None ):
        maxDisp = disp.max()

    disp = np.clip( disp/maxDisp, 0.0, 1.0 )

    mMap = np.array( [ \
         [ 0, 0, 0, 114 ],
         [ 0, 0, 1, 185 ],
         [ 1, 0, 0, 114 ],
         [ 1, 0, 1, 174 ],
         [ 0, 1, 0, 114 ],
         [ 0, 1, 1, 185 ],
         [ 1, 1, 0, 114 ], 
         [ 1, 1, 1, 0 ] ], dtype=NP_FLOAT )

    # import ipdb; ipdb.set_trace()

    bins  = mMap[0:-1, 3].reshape((-1, 1))
    cbins = np.cumsum(bins, axis=0)
    bins  = bins / cbins[-1, -1]
    cbins = cbins[:-1] / cbins[-1, -1]

    ind   = np.minimum( \
                np.sum( 
                    np.tile( disp.reshape((1, -1)), [6, 1] ) > np.tile( cbins, [ 1, disp.size ] ), axis=0 ),
                6 ).astype(np.int)
    bins  = 1.0 / bins
    cbins = np.insert( cbins, 0, 0 ).reshape((-1, 1))

    disp  = ( disp.reshape((1, -1)) - cbins[ind, 0] ) * bins[ind, 0]
    disp  = disp.reshape((-1, 1))

    disp = mMap[ ind, 0:3 ] * np.tile( 1.0 - disp, [1, 3] ) + mMap[ ind+1, 0:3 ] * np.tile( disp, [1, 3] )
    disp = np.clip( disp, 0, 1 ) * 255
    
    disp = disp.reshape((h, w, 3)).astype(np.uint8)

    cv2.imshow("disp", disp)
    cv2.waitKey()

def visualize_depth_as_disparity(depth, BF, mask=None, outDir=None, outName=None, maxDepth=None, waitTime=None, flagShowFigure=True, maxHue=179, n=8):
    if ( maxDepth is None ):
        BFMaxDepth = None
    else:
        BFMaxDepth = BF/maxDepth
    
    visualize_disparity_HSV( BF/depth, mask, outDir, outName, BFMaxDepth, waitTime, flagShowFigure, maxHue, n )

class Args(object):
    def __init__(self, disp):
        super(Args, self).__init__()

        self.disp                  = disp
        self.mask                  = ""
        self.ignore_fov_mask       = False
        self.ignore_cross_occ_mask = False
        self.ignore_self_occ_mask  = False
        self.ignore_all_occ_mask   = False
        self.write_dir             = ""
        self.style                 = "kitti"
        self.not_show              = False

    def copy_args(self, args):
        self.disp                  = args.disp
        self.mask                  = args.mask
        self.ignore_fov_mask       = args.ignore_fov_mask
        self.ignore_cross_occ_mask = args.ignore_cross_occ_mask
        self.ignore_self_occ_mask  = args.ignore_self_occ_mask
        self.ignore_all_occ_mask   = args.ignore_all_occ_mask
        self.write_dir             = args.write_dir
        self.style                 = args.style
        self.not_show              = args.not_show

    def make_parser(self):
        parser = argparse.ArgumentParser(description='Visualize an disparity.')

        parser.add_argument("disp", type=str, \
            help="The filename of the disparity.")

        parser.add_argument("--mask", type=str, default="", \
            help="The filename of the mask")
        
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

        parser.add_argument("--style", type=str, default="kitti", \
            help="Use kitti, hsv, gray to choose style.")

        parser.add_argument("--not-show", action="store_true", default=False, \
            help="Set this flag to disable showing the figure.")

        return parser

    def parse_args(self, parser):
        args = parser.parse_args()

        self.copy_args(args)

        return args

def run(args):
    # Open the disparity file.
    disp = np.load(args.disp).astype(NP_FLOAT)

    # Open the mask file.
    if ( "" == args.mask ):
        # No mask file specified.
        mask = np.zeros(disp.shape, dtype=np.uint8) + 255
    else:
        mask = np.load(args.mask)

    suffix = ""

    # Masking.
    if ( args.ignore_fov_mask ):
        tempMask = mask == STEREO_OUT_OF_FOV
        mask[tempMask] = 255
        suffix += "_nf"

    if ( args.ignore_all_occ_mask ):
        tempMask = mask == STEREO_SELF_OCC
        mask[tempMask] = 255
        tempMask = mask == STEREO_CROSS_OCC
        mask[tempMask] = 255
        
        suffix += "_nc_ns"
    else:
        if ( args.ignore_self_occ_mask ):
            tempMask = mask == STEREO_SELF_OCC
            mask[tempMask] = 255
            suffix += "_ns"

        if ( args.ignore_cross_occ_mask ):
            tempMask = mask == STEREO_CROSS_OCC
            mask[tempMask] = 255
            suffix += "_nc"

    if ( "" != args.write_dir ):
        test_dir( args.write_dir )
        outDir  = args.write_dir
        outName = "%s" % ( os.path.splitext( os.path.split( args.disp )[1] )[0] )
    else:
        outDir  = None
        outName = ""

    outName += suffix

    tempMask = mask != 255
    disp[tempMask] = 0.0

    outFn = "%s/%s.png" % ( outDir, outName )

    # Show the image.
    if ( args.style == "kitti" ):
        dispN = visualize_disparity_KITTI( disp, outFn )

        cv2.imshow(outName, dispN)
        cv2.waitKey()
    elif ( args.style == "hsv" ):
        visualize_disparity_HSV( disp, mask, outDir, outName, maxHue=159 )
    else:
        dispN = normalize_float_image( disp, 255 ).astype(np.uint8)

         # Save the image.
        if ( "" != args.write_dir ):
            save_float_image_PNG( outFn, disp )

        cv2.imshow(outName, dispN)
        cv2.waitKey()

if __name__ == "__main__":
    args = Args(None)
    parser = args.make_parser()
    args.parse_args(parser)

    run(args)
