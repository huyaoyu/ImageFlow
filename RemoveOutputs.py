
from __future__ import print_function

import argparse
import glob
import os
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete the output files from a working directory.")

    parser.add_argument("--dir", type=str, default="./", \
        help="The working directory.")
    
    parser.add_argument("--img-dir", type=str, default="left", \
        help="The image sub-directory.")

    parser.add_argument("--output-dir", type=str, default="ImageFlow", \
        help="The ImageFlow output sub-directory.")

    args = parser.parse_args()

    # Delete all the outputs in the image sub-directory.

    imgDir = "%s/%s" % ( args.dir, args.img_dir )

    if ( True == os.path.isdir( imgDir ) ):
        print( "Deleting files from %s." % (imgDir) )

        # Find all the output files in the image sub-directory.
        files = sorted( glob.glob( "%s/*_warp.*" % ( imgDir ) ) )

        for f in files:
            os.remove(f)

        files = sorted( glob.glob( "%s/*_error.*" % ( imgDir ) ) )

        for f in files:
            os.remove(f)
    else:
        print("%s not exists. " % ( imgDir ))
    
    # Delete al the outputs in the ImageFlow output sub-directory.

    outputDir = "%s/%s" % ( args.dir, args.output_dir )
    if ( True == os.path.isdir( outputDir ) ):
        print( "Deleting %s. " % (outputDir) )

        shutil.rmtree(outputDir)
    else:
        print( "%s not exits. " % ( outputDir ) )

    print("Done.")