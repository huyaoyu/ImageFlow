import cv2
import glob
import numpy as np
import os

# Installed packages.
from CommonPython.Filesystem import Filesystem

# Local packages.
import SimplePLY
from SimulatedLiDARModel import VELODYNE_VLP_32C, E50_32C
import SimulatedLiDAR

def read_depth_files(root):
    '''
    Arguments: 
    root (string): The root directory for the four directories.

    Returns: 
    A list of list of filenames.
    '''

    pattern = '%s/**/*_front_*.npy' % (root)
    frontFns = sorted( glob.glob( pattern, recursive=True ) )

    if ( ( n := len(frontFns) ) == 0 ):
        raise Exception('No files found with %s. ' % (pattern))

    return [ [ fn, fn.replace('front', 'right'), fn.replace('front', 'back'), fn.replace('front', 'left') ] 
            for fn in frontFns ]

def read_depths(fns):
    '''
    Arguments: 
    fns (list of strings): The 4 depth files.

    Returns: 
    A list of depth images.
    '''

    return [ np.load(fn).astype(np.float32) for fn in fns ]

def main():
    # Prepare the output directory.
    outDir = './lidartest_output'
    if ( not os.path.isdir(outDir) ):
        os.makedirs(outDir)

    # Find the files.
    fnList = read_depth_files('./lidartest')

    sld = SimulatedLiDAR.SimulatedLiDAR( 320, 480 )
    sld.set_description( E50_32C )
    sld.initialize()

    for fns in fnList:
        print(fns)
        # Read the depth files.
        depths = read_depths(fns)

        # Extract 3D points.
        lidarPoints = sld.extract( depths )
        lidarPoints = lidarPoints.reshape( (-1, 3) )
        xyz = SimulatedLiDAR.convert_DEA_2_XYZ( lidarPoints[:, 0], lidarPoints[:, 1], lidarPoints[:, 2] )

        # Compose name. 
        parts = Filesystem.get_filename_parts( fns[0] )
        outFn = os.path.join( outDir, '%s.ply' % (parts[1]) )

        # Write.
        SimplePLY.output_to_ply( outFn, xyz.transpose(), [ 1, xyz.shape[0] ], 50, np.array([0, 0, 0]).reshape((-1,1)) )

    return 0

if __name__ == '__main__':
    import sys
    print('Hello, %s! ' % ( os.path.basename(__file__) ))
    sys.exit(main())