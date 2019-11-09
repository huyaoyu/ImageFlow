from __future__ import print_function

import json
import numpy as np
import os

from GeneratePoseName import DummyArgs, generate_pose_name_json
from SimpleGeometory import from_quaternion_to_rotation_matrix

def load_IDs(fn):
    fp = open(fn, "r")

    if ( fp is None ):
        print("Could not open %s" % (fn))
        return -1
    
    lines = fp.readlines()

    fp.close()

    IDs = []

    for l in lines:
        IDs.append( l[:-2] )

    return 0, IDs

def load_IDs_JSON(fn, poseName = None):
    fp = open(fn, "r")

    if ( fp is None ):
        print("Could not open %s" % (fn))
        return -1
    
    dict = json.load(fp)

    fp.close()

    if ( poseName is None ):
        return 0, dict["ID"]
    else:
        return 0, dict[poseName]

def get_pose_from_line(poseDataLine):
    """
    poseDataLine is a 7-element NumPy array. The first 3 elements are 
    the translations. The remaining 4 elements are the orientation 
    represented as a quternion.
    """

    data = poseDataLine.reshape((-1, 1))
    t = data[:3, 0].reshape((-1, 1))
    q = data[3:, 0].reshape((-1, 1))
    R = from_quaternion_to_rotation_matrix(q)

    return R.transpose(), -R.transpose().dot(t), q

def get_pose_by_ID(ID, poseIDs, poseData):
    idxPose = poseIDs.index( ID )

    return get_pose_from_line( poseData[idxPose, :] )

def read_input_parameters_from_json(fn):
    fpJSON = open(fn, "r")
    if ( fpJSON is None ):
        print("%s could not be opened." % (fn))
        # Handle the error.

    inputParams = json.load(fpJSON)

    fpJSON.close()

    return inputParams

def create_pose_id_file(dataDir, imgDir, pattern, poseFileName):
    # Create dummy args.
    args = DummyArgs(dataDir, imgDir, pattern, out_file=poseFileName, silent=True)
    generate_pose_name_json(args)