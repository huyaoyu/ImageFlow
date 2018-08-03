
from __future__ import print_function

import argparse
import copy
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import os

from ColorMapping import color_map

# Global variables used as constants.

INPUT_JSON = "./IFInput.json"

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

PLY_COLORS = [\
    "#2980b9",\
    "#27ae60",\
    "#f39c12",\
    "#c0392b",\
    ]

PLY_COLOR_LEVELS = 20

def show_delimiter(title = "", c = "=", n = 50, leading = "\n", ending = "\n"):
    d = [c for i in range(n/2)]
    s = "".join(d) + " " + title + " " + "".join(d)

    print("%s%s%s" % (leading, s, ending))

def write_ply(fn, verts, colors):
    verts  = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts  = np.hstack([verts, colors])

    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

def depth_to_color(depth, limit = None):

    d  = copy.deepcopy(depth)
    if ( limit is not None ):
        d[ d>limit ] = limit

    color = np.zeros((depth.shape[0], depth.shape[1], 3), dtype = float)
    color[:, :, 0] = d
    color[:, :, 1] = d
    color[:, :, 2] = d

    color = ( color - d.min() ) / ( d.max() - d.min() ) * 255
    color = color.astype(np.uint8)

    return color

def output_to_ply(fn, X, imageSize, rLimit):
    vertices = np.zeros(( imageSize[0], imageSize[1], 3 ), dtype = np.float)
    vertices[:, :, 0] = X[0, :].reshape(imageSize)
    vertices[:, :, 1] = X[1, :].reshape(imageSize)
    vertices[:, :, 2] = X[2, :].reshape(imageSize)
    
    vertices = vertices.reshape((-1, 3))

    r = LA.norm(vertices, axis=1).reshape((-1,1))
    mask = r < rLimit
    mask = mask.reshape(( mask.size ))

    r = r[ mask ]

    cr, cg, cb = color_map(r, PLY_COLORS, PLY_COLOR_LEVELS)

    colors = np.zeros( (r.size, 3), dtype = np.uint8 )

    # import ipdb; ipdb.set_trace()

    colors[:, 0] = cr.reshape( cr.size )
    colors[:, 1] = cg.reshape( cr.size )
    colors[:, 2] = cb.reshape( cr.size )

    write_ply(fn, vertices[mask, :], colors)

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

def from_quaternion_to_rotation_matrix(q):
    """
    q: A numpy vector, 4x1.
    """

    qi2 = q[0, 0]**2
    qj2 = q[1, 0]**2
    qk2 = q[2, 0]**2

    qij = q[0, 0] * q[1, 0]
    qjk = q[1, 0] * q[2, 0]
    qki = q[2, 0] * q[0, 0]

    qri = q[3, 0] * q[0, 0]
    qrj = q[3, 0] * q[1, 0]
    qrk = q[3, 0] * q[2, 0]

    s = 1.0 / ( q[3, 0]**2 + qi2 + qj2 + qk2 )
    ss = 2 * s

    R = [\
        [ 1.0 - ss * (qj2 + qk2), ss * (qij - qrk), ss * (qki + qrj) ],\
        [ ss * (qij + qrk), 1.0 - ss * (qi2 + qk2), ss * (qjk - qri) ],\
        [ ss * (qki - qrj), ss * (qjk + qri), 1.0 - ss * (qi2 + qj2) ],\
    ]

    R = np.array(R, dtype = np.float)

    return R

def get_pose_by_ID(ID, poseIDs, poseData):
    idxPose = poseIDs.index( ID )
    data    = poseData[idxPose, :].reshape((-1, 1))
    t = data[:3, 0].reshape((-1, 1))
    q = data[3:, 0].reshape((-1, 1))
    R = from_quaternion_to_rotation_matrix(q)

    return LA.inv(R), -t, q

def du_dv(nu, nv, imageSize):
    wIdx = np.linspace( 0, imageSize[1] - 1, imageSize[1] )
    hIdx = np.linspace( 0, imageSize[0] - 1, imageSize[0] )

    u, v = np.meshgrid(wIdx, hIdx)

    return nu - u, nv - v

def show(ang, mag, outDir = None, waitTime = None, magFactor = 1.0, angShift = 0.0):
    """ang: degree"""
    # Use Hue, Saturation, Value colour model 
    hsv = np.zeros( ( ang.shape[0], ang.shape[1], 3 ) , dtype=np.uint8)
    hsv[..., 1] = 255

    # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1]

    hsv[..., 0] = (ang + angShift)/ 2
    hsv[..., 2] = np.clip(mag * magFactor, 0, 255).astype(np.uint8) #cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if ( outDir is not None ):
        cv2.imwrite(outDir + "/bgr.jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])

    cv2.imshow("HSV2GBR", bgr)
    if ( waitTime is None ):
        cv2.waitKey()
    else:
        cv2.waitKey( waitTime )

def estimate_loops(N, step):
    """
    N and step will be converted to integers.
    step must less than N.
    """

    N = (int)( N )
    step = (int)( step )

    loops = N / step

    if ( step * loops + 1 > N ):
        loops -= 1

    return loops

class CameraBase(object):
    def __init__(self, focal, imageSize):
        self.focal = focal
        self.imageSize = copy.deepcopy(imageSize) # List or tuple, (height, width)
        self.size = self.imageSize[0] * self.imageSize[1]

        self.pu = self.imageSize[1] / 2
        self.pv = self.imageSize[0] / 2

        self.cameraMatrix = np.eye(3, dtype = np.float)
        self.cameraMatrix[0, 0] = self.focal
        self.cameraMatrix[1, 1] = self.focal
        self.cameraMatrix[0, 2] = self.pu
        self.cameraMatrix[1, 2] = self.pv

        self.worldR = np.zeros((3,3), dtype = np.float)
        self.worldR[0, 1] = 1.0
        self.worldR[1, 2] = 1.0
        self.worldR[2, 0] = 1.0

        self.worldRI = np.zeros((3,3), dtype = np.float)
        self.worldRI[0, 2] = 1.0
        self.worldRI[1, 0] = 1.0
        self.worldRI[2, 1] = 1.0

    def from_camera_frame_to_image(self, coor):
        """
        coor: A numpy column vector, 3x1.
        return: A numpy column vector, 2x1.
        """
        
        coor = self.worldR.dot(coor)
        x = self.cameraMatrix.dot(coor)
        x = x / x[2,:]

        return x[0:2, :]

    def from_depth_to_x_y(self, depth):
        wIdx = np.linspace( 0, self.imageSize[1] - 1, self.imageSize[1] )
        hIdx = np.linspace( 0, self.imageSize[0] - 1, self.imageSize[0] )

        u, v = np.meshgrid(wIdx, hIdx)

        u = u.astype(np.float)
        v = v.astype(np.float)

        x = ( u - self.pu ) * depth / self.focal
        y = ( v - self.pv ) * depth / self.focal

        coor = np.zeros((3, self.size), dtype = np.float)
        coor[0, :] = x.reshape((1, -1))
        coor[1, :] = y.reshape((1, -1))
        coor[2, :] = depth.reshape((1, -1))

        coor = self.worldRI.dot(coor)

        return coor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute the image flow data from sequence of camera poses and their depth information.')

    parser.add_argument("--input", help = "The filename of the input JSON file.", default = INPUT_JSON)
    parser.add_argument("--mf",\
        help = "The iamge magnitude factor. If not specified, the value in the input JSON file will be used. Overwrite the value in the input JSON file is specifiec here.",\
        default = -1.0, type = float)
    parser.add_argument("--debug", help = "Debug information including 3D point clouds will be written addintionally.", action = "store_true", default = False)

    args = parser.parse_args()

    # Read the JSON input file.
    fpJSON = open(args.input, "r")
    if ( fpJSON is None ):
        print("%s could not be opened." % (args.input))
        # Handle the error.

    inputParams = json.load(fpJSON)

    fpJSON.close()

    # Data directory.
    dataDir = inputParams["dataDir"]

    # The magnitude factor.
    if ( args.mf < 0.0 ):
        mf = inputParams["imageMagnitudeFactor"]
    else:
        mf = args.mf

    # Load the pose filenames and the pose data.
    _, poseIDs = load_IDs_JSON(\
        dataDir + "/" + inputParams["poseFilename"], inputParams["poseName"])
    poseData   = np.load( dataDir + "/" + inputParams["poseData"] )
    if ( True == args.debug ):
        np.savetxt( dataDir + "/poseData.dat", poseData, fmt="%+.4e" )

    # print(poseData.shape)
    print("poseData and poseFilenames loaded.")

    # Camera.
    cam_0 = CameraBase(inputParams["camera"]["focal"], inputParams["camera"]["imageSize"])
    print(cam_0.imageSize)
    print(cam_0.cameraMatrix)

    # We are assuming that the cameras at the two poses are the same camera.
    cam_1 = cam_0

    # Get the number of loops.
    nPoses = len( poseIDs )

    # Loop over the poses.
    poseID_0, poseID_1 = None, None
    idxStep       = inputParams["idxStep"]
    outDirBase    = dataDir + "/" + inputParams["outDir"]
    depthDir      = dataDir + "/" + inputParams["depthDir"]
    depthTail     = inputParams["depthSuffix"] + inputParams["depthExt"]
    distanceRange = inputParams["distanceRange"]
    flagDegree    = inputParams["flagDegree"]

    estimatedLoops = estimate_loops( nPoses - inputParams["startingIdx"], idxStep )

    count = 0
    idxNumberRequest = inputParams["idxNumberRequest"]

    for i in range( inputParams["startingIdx"] + idxStep,\
        nPoses, idxStep ):

        show_delimiter( title = "%d / %d" % ( count + 1, estimatedLoops ) )

        poseID_0 = poseIDs[ i - idxStep ]
        poseID_1 = poseIDs[ i ]

        print("poseID_0 = %s, poseID_1 = %s" % (poseID_0, poseID_1))

        outDir = outDirBase + "/" + poseID_0

        if ( os.path.isdir(outDir) ):
            pass
        else:
            os.makedirs(outDir)

        # Get the pose of the first position.
        R0, t0, q0= get_pose_by_ID(poseID_0, poseIDs, poseData)
        R0Inv = LA.inv(R0)

        print("t0 = \n{}".format(t0))
        print("q0 = \n{}".format(q0))
        print("R0 = \n{}".format(R0))
        print("R0Inv = \n{}".format(R0Inv))

        # Get the pose of the second position.
        R1, t1, q1 = get_pose_by_ID(poseID_1, poseIDs, poseData)
        R1Inv = LA.inv(R1)

        print("t1 = \n{}".format(t1))
        print("q1 = \n{}".format(q1))
        print("R1 = \n{}".format(R1))
        print("R1Inv = \n{}".format(R1Inv))

        # Compute the rotation between the two camera poses.
        R = np.matmul( R1, R0Inv )
        print("R = \n{}".format(R))

        # Load the depth of the first image.
        depth_0 = np.load( depthDir + "/" + poseID_0 + depthTail )
        if ( True == args.debug ):
            np.savetxt( outDir + "/depth_0.dat", depth_0, fmt="%.2e")

        # Calculate the coordinates in the first camera's frame.
        X0 = cam_0.from_depth_to_x_y(depth_0)
        if ( True == args.debug ):
            output_to_ply(outDir + '/XInCam_0.ply', X0, cam_0.imageSize, distanceRange)

        # The coordinates in the world frame.
        XWorld_0  = R0Inv.dot(X0 - t0)
        if ( True == args.debug ):
            output_to_ply(outDir + "/XInWorld_0.ply", XWorld_0, cam_1.imageSize, distanceRange)

        # Load the depth of the second image.
        depth_1 = np.load( depthDir + "/" + poseID_1 + depthTail )
        if ( True == args.debug ):
            np.savetxt( outDir + "/depth_1.dat", depth_1, fmt="%.2e")

        # Calculate the coordinates in the second camera's frame.
        X1 = cam_1.from_depth_to_x_y(depth_1)
        if ( True == args.debug ):
            output_to_ply(outDir + "/XInCam_1.ply", X1, cam_1.imageSize, distanceRange)

        # The coordiantes in the world frame.
        XWorld_1 = R1Inv.dot( X1 - t1 )
        if ( True == args.debug ):
            output_to_ply(outDir + "/XInWorld_1.ply", XWorld_1, cam_1.imageSize, distanceRange)

        # ====================================
        # The coordinate in the seconde camera's frame.
        X_01 = R1.dot(XWorld_0) + t1
        if ( True == args.debug ):
            output_to_ply(outDir + '/X_01.ply', X_01, cam_0.imageSize, distanceRange)

        # The image coordinates in the second camera.
        c = cam_0.from_camera_frame_to_image(X_01)

        # Get new u anv v
        u = c[0, :].reshape(cam_0.imageSize)
        v = c[1, :].reshape(cam_0.imageSize)
        np.savetxt(outDir + "/u.dat", u, fmt="%4d")
        np.savetxt(outDir + "/v.dat", v, fmt="%4d")

        # Get the du and dv.
        du, dv = du_dv(u, v, cam_0.imageSize)

        # Save du and dv.
        np.savetxt(outDir + "/du.dat", du.astype(np.int), fmt="%+3d")
        np.savetxt(outDir + "/dv.dat", dv.astype(np.int), fmt="%+3d")

        dudv = np.zeros( ( cam_0.imageSize[0], cam_0.imageSize[1], 2), dtype = np.float32 )
        dudv[:, :, 0] = du
        dudv[:, :, 1] = dv
        np.save(outDir + "/dudv.npy", dudv)

        # Calculate the angle and distance.
        a = np.arctan2( dv, du )
        angleShift = np.pi
        if ( True == flagDegree ):
            a = a / np.pi * 180
            angleShift = 180
            print("Convert angle from radian to degree as demanded by the input file.")

        d = np.sqrt( du * du + dv * dv )

        # Save the angle and distance.
        np.savetxt(outDir + "/a.dat", a, fmt="%+.2e")
        np.savetxt(outDir + "/d.dat", d, fmt="%+.2e")

        angleAndDist = np.zeros( ( cam_0.imageSize[0], cam_0.imageSize[1], 2), dtype = np.float32 )
        angleAndDist[:, :, 0] = a
        angleAndDist[:, :, 1] = d
        np.save(outDir + "/ad.npy", angleAndDist)

        # Show and save the resulting HSV image.
        if ( 1 == estimatedLoops ):
            show(a, d, outDir, (int)(mf), angleShift)
        else:
            show(a, d, outDir, (int)(inputParams["imageWaitTimeMS"]), mf, angleShift)

        print("Done with i = %d" % ( i - idxStep ))

        count += 1

        if ( count >= idxNumberRequest ):
            print("Loop number hits the request number. Stop here.")
            break

    show_delimiter("Summary.")
    print("%d poses, starting at idx = %d, step = %d, %d steps in total. idxNumberRequest = %d\n" % (nPoses, inputParams["startingIdx"], idxStep, count, idxNumberRequest))

    if ( args.mf >= 0 ):
        print( "Command line argument --mf %f overwrites the parameter \"imageMagnitudeFactor\" (%f) in the input JSON file.\n" % (mf, inputParams["imageMagnitudeFactor"]) )
