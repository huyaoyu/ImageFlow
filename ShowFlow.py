from __future__ import print_function

import argparse
import numpy as np
import os

from ImageFlow import show, calculate_angle_distance_from_du_dv
from ImageFlow import SELF_OCC, CROSS_OCC, OUT_OF_FOV_POSITIVE_Z
from Utils import test_dir

if __name__ == "__main__":
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

    args = parser.parse_args()

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
    a, d, angleShift = calculate_angle_distance_from_du_dv( flow[:, :, 0], flow[:, :, 1], flagDegree=True )

    if ( "" != args.write_dir ):
        test_dir( args.write_dir )
        outDir  = args.write_dir
        outName = "%s" % ( os.path.splitext( os.path.split( args.flow )[1] )[0] )
    else:
        outDir  = None
        outName = ""

    outName += suffix

    # Show the image.
    show(a, d, mask, outDir, outName, waitTime=None, magFactor=args.mf, angShift=angleShift)
