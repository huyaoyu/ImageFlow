from __future__ import print_function

import argparse
import os

from ImageFlow import show

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