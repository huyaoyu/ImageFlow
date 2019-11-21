from __future__ import print_function

import argparse
import glob
import os

from ShowStereoDisparity import Args as SSDA
from ShowStereoDisparity import run as SSDRun

if __name__ == "__main__":
    print("Local run ShowStereoDisparity.py. ")

    parser = argparse.ArgumentParser(description="Local run ShowStereoDisparity.py.")

    parser.add_argument("indir", type=str, \
        help="The input base directory. ")

    parser.add_argument("outdir", type=str, \
        help="The output directory. ")

    parser.add_argument("--pattern-disp", type=str, default="*_disp.npy", \
        help="The filename pattern for the disp files.")

    parser.add_argument("--pattern-mask", type=str, default="*_mask.npy", \
        help="The filename pattern for the mask files.")

    parser.add_argument("--style", type=str, default="hsv", \
        help="Could be kitti, hsv, gray.")
    
    parser.add_argument("--color-max-disp", type=float, default=0, \
        help="The maximum disparity value for color renderring. Set 0 to disable. Has no effect on the grayscale figures.")

    args = parser.parse_args()

    # Find all the files.

    disps = sorted( glob.glob( "%s/%s" % ( args.indir, args.pattern_disp ) ) )
    masks = sorted( glob.glob( "%s/%s" % ( args.indir, args.pattern_mask ) ) )

    nF = len( disps )
    nM = len( masks )

    if ( nF == 0 or nM == 0 ):
        raise Exception("Zero files found in %s with patters %s and %s. " % ( args.indir, args.pattern_disp, args.pattern_mask ))

    if ( nF != nM ):
        raise Exception("Numbers of files are not the same. nF = %d, nM = %d. " % ( nF, nM ))

    for i in range(nF):
        disp = disps[i]
        mask = masks[i]

        print("%d/%d: %s. " % ( i+1, nF, disp ))

        # Create a dummy Args object.
        ssdArgs = SSDA( disp )
        ssdArgs.mask                  = mask
        ssdArgs.ignore_fov_mask       = False
        ssdArgs.ignore_cross_occ_mask = False
        ssdArgs.ignore_self_occ_mask  = False
        ssdArgs.ignore_all_occ_mask   = False
        ssdArgs.write_dir             = "%s" % ( args.outdir )
        ssdArgs.style                 = args.style
        ssdArgs.not_show              = True
        ssdArgs.color_max_disp        = args.color_max_disp

        # Run Showdisp.py.
        SSDRun(ssdArgs)

    print("Done. ")