from __future__ import print_function

import argparse
import glob
import os

from ShowFlow import Args as SFA
from ShowFlow import run as SFRun

if __name__ == "__main__":
    print("Local run ShowFlow.py. ")

    parser = argparse.ArgumentParser(description="Local run ShowFlow.py.")

    parser.add_argument("indir", type=str, \
        help="The input base directory. ")

    parser.add_argument("outdir", type=str, \
        help="The output directory. ")

    parser.add_argument("--pattern-flow", type=str, default="*_flow.npy", \
        help="The filename pattern for the flow files.")

    parser.add_argument("--pattern-mask", type=str, default="*_mask.npy", \
        help="The filename pattern for the mask files.")

    parser.add_argument("--kitti", action="store_true", default=False, \
        help="Use kitti style.")

    args = parser.parse_args()

    # Find all the files.

    flows = sorted( glob.glob( "%s/%s" % ( args.indir, args.pattern_flow ) ) )
    masks = sorted( glob.glob( "%s/%s" % ( args.indir, args.pattern_mask ) ) )

    nF = len( flows )
    nM = len( masks )

    if ( nF == 0 or nM == 0 ):
        raise Exception("Zero files found in %s with patters %s and %s. " % ( args.indir, args.pattern_flow, args.pattern_mask ))

    if ( nF != nM ):
        raise Exception("Numbers of files are not the same. nF = %d, nM = %d. " % ( nF, nM ))

    for i in range(nF):
        flow = flows[i]
        mask = masks[i]

        print("%d/%d: %s. " % ( i+1, nF, flow ))

        # Create a dummy Args object.
        sfArgs = SFA( flow )
        sfArgs.mask                  = mask
        sfArgs.mf                    = 1.0
        sfArgs.ignore_fov_mask       = False
        sfArgs.ignore_cross_occ_mask = False
        sfArgs.ignore_self_occ_mask  = False
        sfArgs.ignore_all_occ_mask   = False
        sfArgs.write_dir             = "%s" % ( args.outdir )
        sfArgs.kitti                 = args.kitti
        sfArgs.max_f                 = 0
        sfArgs.vector                = False
        sfArgs.not_show              = True

        # Run ShowFlow.py.
        SFRun(sfArgs)

    print("Done. ")