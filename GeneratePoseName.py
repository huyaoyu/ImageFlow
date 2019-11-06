from __future__ import print_function

import argparse
import glob
import json
import os

def cprint(s, sf):
    if ( not sf ):
        print(s)

class DummyArgs(object):
    def __init__(self, datadir, img_dir="image_left", pattern="*.png", out_file="PoseName.json", silent=False):
        super(DummyArgs, self).__init__()

        self.datadir  = datadir
        self.img_dir  = img_dir
        self.pattern  = pattern
        self.out_file = out_file
        self.silent   = silent

def generate_pose_name_json(args):
    # Search for files.
    files = sorted( glob.glob( "%s/%s/%s" % ( args.datadir, args.img_dir, args.pattern ) ) )

    n = len(files)

    if ( 0 == n ):
        raise Exception("Zero files found at %s/%s with search pattern %s. " % ( args.datadir, args.img_dir, args.pattern ))
    else:
        cprint("%d files found." % (n), args.silent)

    # Converth the filenames into a list.
    poseNames = []

    for f in files:
        poseNames.append(os.path.splitext(os.path.split(f)[1])[0].split("_")[0] )

    # Save poseNames as a JSON file.
    d = {"poseName":poseNames}

    with open( "%s/%s" % (args.datadir, args.out_file), "w" ) as fp:
        json.dump( d, fp, indent=2 )

    cprint("Done.", args.silent)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a PoseName.json file.")

    parser.add_argument("datadir", type=str, \
        help="The data directory contains the image subfolder.")
    
    parser.add_argument("--img-dir", type=str, default="image_left", \
        help="The image sub-folder.")
    
    parser.add_argument("--pattern", type=str, default="*.png", \
        help="The search pattern.")
    
    parser.add_argument("--out-file", type=str, default="PoseName.json", \
        help="The output filename.")

    parser.add_argument("--silent", action="store_true", default=False, \
        help="Set this flag for silent processing.")

    args = parser.parse_args()

    # Do the job.
    generate_pose_name_json(args)
