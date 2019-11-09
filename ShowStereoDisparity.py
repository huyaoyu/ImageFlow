from __future__ import print_function

import argparse
import cv2
import numpy as np
import os

from CommonType import NP_FLOAT

from ImageStereo import STEREO_OUT_OF_FOV, STEREO_SELF_OCC, STEREO_CROSS_OCC
from Utils import test_dir, normalize_float_image, save_float_image_PNG

if __name__ == "__main__":
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

    args = parser.parse_args()

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
        tempMask = mask == STEREO_CROSS_OCC
        mask[tempMask] = 255
        tempMask = mask == STEREO_SELF_OCC
        mask[tempMask] = 255
        suffix += "_nc_ns"
    else:
        if ( args.ignore_cross_occ_mask ):
            tempMask = mask == STEREO_CROSS_OCC
            mask[tempMask] = 255
            suffix += "_nc"

        if ( args.ignore_self_occ_mask ):
            tempMask = mask == STEREO_SELF_OCC
            mask[tempMask] = 255
            suffix += "_ns"

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

    # Save the image.
    if ( "" != args.write_dir ):
        save_float_image_PNG( outFn, disp )

    # Show the image.
    dispN = normalize_float_image( disp, 255 ).astype(np.uint8)

    cv2.imshow(outName, dispN)
    cv2.waitKey()
    