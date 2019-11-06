from __future__ import print_function

import argparse
import glob
import json
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a PoseName.json file.")

    parser.add_argument("datadir", type=str, \
        help="The data directory contains the image subfolder.")
    
    parser.add_argument("--imgDir", type=str, default="image_left", \
        help="The image sub-folder.")
    
    parser.add_argument("--pattern", type=str, default="*.png", \
        help="The search pattern.")

