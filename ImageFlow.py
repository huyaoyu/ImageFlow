#!/usr/bin/python

from __future__ import print_function

import argparse
import copy
import cv2
import json
import math
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

WORLD_ORIGIN  = np.zeros((3, 1))
CAMERA_ORIGIN = np.zeros((3, 1))

def show_delimiter(title = "", c = "=", n = 50, leading = "\n", ending = "\n"):
    d = [c for i in range( int(n/2) )]
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

def output_to_ply(fn, X, imageSize, rLimit, origin):
    # Check the input X.
    if ( X.max() <= X.min() ):
        raise Exception("X.max() = %f, X.min() = %f." % ( X.max(), X.min() ) )
    
    vertices = np.zeros(( imageSize[0], imageSize[1], 3 ), dtype = np.float)
    vertices[:, :, 0] = X[0, :].reshape(imageSize)
    vertices[:, :, 1] = X[1, :].reshape(imageSize)
    vertices[:, :, 2] = X[2, :].reshape(imageSize)
    
    vertices = vertices.reshape((-1, 3))
    rv = copy.deepcopy(vertices)
    rv[:, 0] = vertices[:, 0] - origin[0, 0]
    rv[:, 1] = vertices[:, 1] - origin[1, 0]
    rv[:, 2] = vertices[:, 2] - origin[2, 0]

    r = LA.norm(rv, axis=1).reshape((-1,1))
    mask = r < rLimit
    mask = mask.reshape(( mask.size ))
    # import ipdb; ipdb.set_trace()
    r = r[ mask ]

    cr, cg, cb = color_map(r, PLY_COLORS, PLY_COLOR_LEVELS)

    colors = np.zeros( (r.size, 3), dtype = np.uint8 )

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

    return R.transpose(), -R.transpose().dot(t), q

def du_dv(nu, nv, imageSize):
    wIdx = np.linspace( 0, imageSize[1] - 1, imageSize[1] )
    hIdx = np.linspace( 0, imageSize[0] - 1, imageSize[0] )

    u, v = np.meshgrid(wIdx, hIdx)

    return nu - u, nv - v

def show(ang, mag, outDir = None, waitTime = None, magFactor = 1.0, angShift = 0.0, flagShowFigure=True):
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

    if ( True == flagShowFigure ):
        cv2.imshow("HSV2GBR", bgr)

        if ( waitTime is None ):
            cv2.waitKey()
        else:
            cv2.waitKey( waitTime )

def save_float_image(fn, img):
    img = (img - img.min()) / ( img.max() - img.min() ) * 255

    img = img.astype(np.uint8)

    cv2.imwrite( fn, img )

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
        
        # coor = self.worldR.dot(coor)
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

        # coor = self.worldRI.dot(coor)

        return coor

def get_distance_from_coordinate_table(tab, idx):
    """
    tab: A 3-row table contains 3D coordinates.
    idx: The column idex.

    This funcion will return the distance of a point specified
    by idx measured from the origin.
    """

    # Get the x, y, z
    x = tab[0, idx]
    y = tab[1, idx]
    z = tab[2, idx]

    return math.sqrt( x**2 + y**2 + z**2 )

def create_warp_masks(imageSize, x01, x1, u, v):
    """
    imageSize: height x width.
    x01: The 3D coordinates of the pixels in the first image observed in the frame of the second camera. 3-row 2D array.
    x1: The 3D coordinates of the pixels in the second image observed in the frame of the second camera. 3-row 2D array.
    u: The u coordinates of the pixel in the second image plane. 1D array.
    v: The v coordinates of the pixel in the second image plane. 1D array.
    """
    # import ipdb;ipdb.set_trace()
    # Check dimensions.
    assert( u.shape[0] == v.shape[0] )
    assert( u.shape[1] == v.shape[1] )
    assert( u.shape[0] * u.shape[1] == x01.shape[1] )
    assert( x01.shape[0] == 3 )

    # Allocate memory.
    occupancyMap  = np.zeros( imageSize, dtype=np.int32 ) - 1
    maskOcclusion = np.zeros( imageSize, dtype=np.uint8 )
    maskFOV       = np.zeros( imageSize, dtype=np.uint8 )

    # Reshape input arguments.
    u = u.reshape((-1,))
    v = v.reshape((-1,))

    h = imageSize[0]
    w = imageSize[1]

    # Loop for every pixel index.
    for i in range( h*w ):
        # Get the u and v coordinate of the pixel in the image plane of the second camera.
        iu = int( u[i] )
        iv = int( v[i] )

        # Get the u and v coordinate of the pixel in the image plane of the original camera.
        iy = i // w
        ix = i % w

        # Check if the new index is out of boundary?
        if ( iu < 0 or iv < 0 or iu >= w or iv >= h ):
            # Update the FOV mask.
            maskFOV[iy, ix] = 1

            # Stop the current loop.
            continue

        # Get the current depth.
        d0 = get_distance_from_coordinate_table(x01, i)

        # Check if the new index is occupied.
        if ( -1 != occupancyMap[iv, iu] ):
            # This pixel is occupied.

            # Get the index registered in the occupancy map.
            opIndex = occupancyMap[iv, iu]

            # Get the depth at the registered index.
            dr = get_distance_from_coordinate_table(x01, opIndex)

            if ( d0 < dr ):
                # Current point nearer to the camera.
                # Update the occlusion mask.
                maskOcclusion[ opIndex // w, opIndex % w ] = 1
            elif ( d0 > dr ):
                # Current point farther.
                # Update the occlusion mask.
                maskOcclusion[ iy, ix ] = 1

                # Stop the current loop.
                continue
            else:
                raise Exception("%d pixel has same distance with %d pixel." % ( i, opIndex ))
        
        # Get the depth at x=iu, y=iv in the second image observed in the second camera.
        d1 = get_distance_from_coordinate_table(x1, iv*w + iu)

        if ( d0 <= d1 ):
            # Current point is nearer to the camera or equals the distance of the corresponding pixel in the second image.
            pass
        elif ( ( d0 > 1000 and d1 > 1000 ) or d0 - d1 < 0.05 * d0 ):
            pass
        else:
            # Current point is occluded by the corresponding pixel in the second image.
            # Update the occlusion mask.
            maskOcclusion[ iy, ix ] = 2

            if ( -1 != occupancyMap[iv, iu] ):
                raise Exception( "Current pixel %d, wins pre-registered %d but occlued by second image at x=%d, y=%d with d0=%f, dr=%f, d1=%f." \
                    % ( i, opIndex, iu, iv, d0, dr, d1 ) )

            continue

        # Update the occupancy map.
        occupancyMap[ iv, iu ] = i
        
    return maskOcclusion, maskFOV, occupancyMap

def evaluate_warp_error( img0, img1, x01, x1, u, v ):
    """
    x01: The 3D coordinates of the pixels in the first image observed in the frame of the second camera. 3-row 2D array.
    x1: The 3D coordinates of the pixels in the second image observed in the frame of the second camera. 3-row 2D array.
    u: The u coordinates of the pixel in the second image plane. 1D array.
    v: The v coordinates of the pixel in the second image plane. 1D array.
    """

    h = img0.shape[0]
    w = img0.shape[1]

    # Get the masks.
    maskOcclusion, maskFOV, occupancyMap = create_warp_masks( img0.shape[:2], x01, x1, u, v )

    # Make a mask for the occupancyMap
    mask = occupancyMap != -1

    # All indices need to be evaluated in img0.
    idx0 = occupancyMap[mask].astype(np.int32)

    # All u and v need to be evaluated in img1.
    u1 = u.reshape((-1,))[idx0]
    v1 = v.reshape((-1,))[idx0]

    u1 = u1.astype(np.int32)
    v1 = v1.astype(np.int32)

    # Convert u1 and v1 into linear index.
    idx1 = v1 * img1.shape[1] + u1
    idx1 = idx1.astype(np.int32)

    # Reshape the input image.
    img0 = img0.reshape((-1, img0.shape[2])).astype(np.int32)
    img1 = img1.reshape((-1, img1.shape[2])).astype(np.int32)

    # Absolute difference.
    diff = img0[idx0, :] - img1[idx1, :]
    diff = np.linalg.norm( diff, 2, axis=1 )

    # Make diff to be an image.
    dImg0 = np.zeros( h*w, dtype=np.float32 )
    dImg1 = np.zeros( h*w, dtype=np.float32 )
    dImg0[idx0] = diff
    dImg1[idx1] = diff

    return dImg0.reshape( (h, w) ), dImg1.reshape( (h, w) ), occupancyMap, mask

def warp_image(imgDir, poseID_0, poseID_1, imgSuffix, imgExt, X_01C, X1C, u, v):
    cam0ImgFn = "%s/%s%s%s" % ( imgDir, poseID_0, imgSuffix, imgExt )
    cam1ImgFn = "%s/%s%s%s" % ( imgDir, poseID_1, imgSuffix, imgExt )
    warpErrImgFn = "%s/%s%s%s%s" % ( imgDir, poseID_0, imgSuffix, "_error", imgExt )
    warpErrStaFn = "%s/%s%s%s%s" % ( imgDir, poseID_0, imgSuffix, "_error", ".dat" )

    # print("Warp %s." % (cam0ImgFn))
    cam0_img = cv2.imread( cam0ImgFn, cv2.IMREAD_UNCHANGED )
    
    # Evaluate warp error.
    cam1_img = cv2.imread( cam1ImgFn, cv2.IMREAD_UNCHANGED )
    
    dImg0, dImg1, occupancyMap, occupancyMask = evaluate_warp_error( cam0_img, cam1_img, X_01C, X1C, u, v )
    save_float_image( warpErrImgFn, dImg1 )

    # The mean warp error over the valid pixels in the seconde image.
    meanError = dImg1[occupancyMask].mean()

    np.savetxt( warpErrStaFn, \
        np.array([ dImg1[occupancyMask].min(), dImg1.max(), meanError ]).reshape((-1, 1)) )

    warppedImg = np.zeros_like(cam0_img)
    
    # for h in range(cam0_img.shape[0]):
    #     for w in range(cam0_img.shape[1]):
    #         u_w, v_w = int(round(u[h,w])), int(round(v[h,w]))
    #         if u_w < cam0_img.shape[1] and v_w < cam0_img.shape[0] and u_w >= 0 and v_w >= 0:
    #             warppedImg[v_w, u_w, :] = cam0_img[h, w, :]
    
    # validWarpMask = occupancyMap != -1
    validWarpMask = occupancyMask
    validWarpIdx  = occupancyMap[validWarpMask]

    cam0ImgCpy = copy.deepcopy(cam0_img).reshape( (-1, cam0_img.shape[2]) )
    warppedImg = warppedImg.reshape( (-1, cam0_img.shape[2]) )
    warppedImg[validWarpMask.reshape( (-1,) ), :] = cam0ImgCpy[ validWarpIdx, : ]
    warppedImg = warppedImg.reshape( cam0_img.shape )

    # Save the warpped image.
    cam0WrpFn = "%s/%s%s%s%s" % ( imgDir, poseID_0, imgSuffix, "_warp", imgExt )
    cv2.imwrite(cam0WrpFn, warppedImg)

    return warppedImg, meanError

def read_input_parameters_from_json(fn):
    fpJSON = open(fn, "r")
    if ( fpJSON is None ):
        print("%s could not be opened." % (fn))
        # Handle the error.

    inputParams = json.load(fpJSON)

    fpJSON.close()

    return inputParams

def get_magnitude_factor_from_input_parameters(params, args):
    if ( args.mf < 0.0 ):
        mf = params["imageMagnitudeFactor"]
    else:
        mf = args.mf

    return mf

def load_pose_id_pose_data(params, args):
    dataDir = params["dataDir"]

    _, poseIDs = load_IDs_JSON(\
        dataDir + "/" + params["poseFilename"], params["poseName"])
    poseData   = np.load( dataDir + "/" + params["poseData"] )
    if ( True == args.debug ):
        np.savetxt( dataDir + "/poseData.dat", poseData, fmt="%+.4e" )

    return poseIDs, poseData

def test_dir(d):
    if ( False == os.path.isdir(d) ):
        os.makedirs(d)

def calculate_angle_distance_from_du_dv(du, dv, flagDegree=False):
    a = np.arctan2( dv, du )

    angleShift = np.pi

    if ( True == flagDegree ):
        a = a / np.pi * 180
        angleShift = 180
        # print("Convert angle from radian to degree as demanded by the input file.")

    d = np.sqrt( du * du + dv * dv )

    return a, d, angleShift

def make_angle_distance(cam, a, d):
    angleAndDist = np.zeros( ( cam.imageSize[0], cam.imageSize[1], 2), dtype = np.float32 )
    angleAndDist[:, :, 0] = a
    angleAndDist[:, :, 1] = d

    return angleAndDist

def print_over_warp_error_list(overWarpErrList, t):
    # { "idx": i, "poseID_0": poseID_0, "poseID_1": poseID_1, "meanWarpError": meanWarpError }

    if ( 0 != len( overWarpErrList ) ):
        print( "%d over warp error threshold (%f). " % ( len( overWarpErrList ), t ) )
        print( "idx, poseID_0, poseID_1, meanWarpError" )
    else:
        print( "No warp error over the threshold (%f). " % (t) )
        return
    
    for entry in overWarpErrList:
        print( "%d, %s, %s, %f. " % ( entry["idx"], entry["poseID_0"], entry["poseID_1"], entry["meanWarpError"] ) )

def print_max_warp_error(entry):
    if ( entry["idx"] != -1 ):
        print( "Max mean warp error: " )
        print( "idx: %d, poseIDs: %s - %s, mean error: %f. " % \
            ( entry["idx"], entry["poseID_0"], entry["poseID_1"], entry["warpErr"] ) )
    else:
        raise Exception( "Wrong max warp error entry: idx: %d, poseIDs: %s - %s, mean error: %f. " % \
            ( entry["idx"], entry["poseID_0"], entry["poseID_1"], entry["warpErr"] ) )

def process_single_thread(name, inputParams, args, poseIDs, poseData, indexList, startII, endII, flagShowFigure=False):
    # Data directory.
    dataDir = inputParams["dataDir"]

    # The magnitude factor.
    mf = get_magnitude_factor_from_input_parameters( inputParams, args )

    # Camera.
    cam_0 = CameraBase(inputParams["camera"]["focal"], inputParams["camera"]["imageSize"])
    print(cam_0.imageSize)
    print(cam_0.cameraMatrix)

    # We are assuming that the cameras at the two poses are the same camera.
    cam_1 = cam_0

    # Loop over the poses.
    poseID_0, poseID_1 = None, None

    outDirBase    = dataDir + "/" + inputParams["outDir"]
    depthDir      = dataDir + "/" + inputParams["depthDir"]
    imgDir        = dataDir + "/" + inputParams["imageDir"]
    imgSuffix     = inputParams["imageSuffix"]
    imgExt        = inputParams["imageExt"]
    depthTail     = inputParams["depthSuffix"] + inputParams["depthExt"]
    distanceRange = inputParams["distanceRange"]
    flagDegree    = inputParams["flagDegree"]
    warpErrThres  = inputParams["warpErrorThreshold"]

    estimatedLoops = endII - startII + 1 - 1

    count = 0

    overWarpErrThresList = []
    warpErrMaxEntry = { "idx": -1, "poseID_0": "N/A", "poseID_1": "N/A", "warpErr": 0.0 }

    for i in range( startII+1, endII+1 ):
        # Show the delimiter.
        show_delimiter( title = "%s: %d / %d" % ( name, count + 1, estimatedLoops ) )

        idxPose0 = indexList[i - 1]
        idxPose1 = indexList[i]

        poseID_0 = poseIDs[ idxPose0 ]
        poseID_1 = poseIDs[ idxPose1 ]

        print("poseID_0 = %s, poseID_1 = %s" % (poseID_0, poseID_1))

        # Prepare output directory.
        outDir = outDirBase + "/" + poseID_0
        test_dir(outDir)

        # Get the pose of the first position.
        R0, t0, q0= get_pose_by_ID(poseID_0, poseIDs, poseData)
        R0Inv = LA.inv(R0)

        if ( True == args.debug ):
            print("t0 = \n{}".format(t0))
            print("q0 = \n{}".format(q0))
            print("R0 = \n{}".format(R0))
            print("R0Inv = \n{}".format(R0Inv))

        # Get the pose of the second position.
        R1, t1, q1 = get_pose_by_ID(poseID_1, poseIDs, poseData)
        R1Inv = LA.inv(R1)

        if ( True == args.debug ):
            print("t1 = \n{}".format(t1))
            print("q1 = \n{}".format(q1))
            print("R1 = \n{}".format(R1))
            print("R1Inv = \n{}".format(R1Inv))

        # Compute the rotation between the two camera poses.
        R = np.matmul( R1, R0Inv )

        if ( True == args.debug ):
            print("R = \n{}".format(R))

        # Load the depth of the first image.
        depth_0 = np.load( depthDir + "/" + poseID_0 + depthTail )
        
        if ( True == args.debug ):
            np.savetxt( outDir + "/depth_0.dat", depth_0, fmt="%.2e")

        # Calculate the coordinates in the first camera's frame.
        X0C = cam_0.from_depth_to_x_y(depth_0) # Coordinates in the camera frame. z-axis pointing forwards.
        X0  = cam_0.worldRI.dot(X0C)           # Corrdinates in the NED frame. z-axis pointing downwards.
        
        if ( True == args.debug ):
            try:
                output_to_ply(outDir + '/XInCam_0.ply', X0, cam_0.imageSize, distanceRange, CAMERA_ORIGIN)
            except Exception as e:
                print("Cannot write PLY file for X0. Exception: ")
                print(e)

        # The coordinates in the world frame.
        XWorld_0  = R0Inv.dot(X0 - t0)

        if ( True == args.debug ):
            try:
                output_to_ply(outDir + "/XInWorld_0.ply", XWorld_0, cam_1.imageSize, distanceRange, -R0Inv.dot(t0))
            except Exception as e:
                print("Cannot write PLY file for XWorld_0. Exception: ")
                print(e)

        # Load the depth of the second image.
        depth_1 = np.load( depthDir + "/" + poseID_1 + depthTail )

        if ( True == args.debug ):
            np.savetxt( outDir + "/depth_1.dat", depth_1, fmt="%.2e")

        # Calculate the coordinates in the second camera's frame.
        X1C = cam_1.from_depth_to_x_y(depth_1) # Coordinates in the camera frame. z-axis pointing forwards.
        X1  = cam_1.worldRI.dot(X1C)           # Corrdinates in the NED frame. z-axis pointing downwards.

        if ( True == args.debug ):
            try:
                output_to_ply(outDir + "/XInCam_1.ply", X1, cam_1.imageSize, distanceRange, CAMERA_ORIGIN)
            except Exception as e:
                print("Cannot write PLY file for X1. Exception: ")
                print(e)

        # The coordiantes in the world frame.
        XWorld_1 = R1Inv.dot( X1 - t1 )

        if ( True == args.debug ):
            try:
                output_to_ply(outDir + "/XInWorld_1.ply", XWorld_1, cam_1.imageSize, distanceRange, -R1Inv.dot(t1))
            except Exception as e:
                print("Cannot write PLY file for XWorld_1. Exception: ")
                print(e)

        # ====================================
        # The coordinate of the pixels of the first camera projected in the seconde camera's frame (NED).
        X_01 = R1.dot(XWorld_0) + t1

        if ( True == args.debug ):
            try:
                output_to_ply(outDir + '/X_01.ply', X_01, cam_0.imageSize, distanceRange, CAMERA_ORIGIN)
            except Exception as e:
                print("Cannot write PLY file for X_01. Exception: ")
                print(e)

        # The image coordinates in the second camera.
        X_01C = cam_0.worldR.dot(X_01)                  # Camera frame, z-axis pointing forwards.
        c     = cam_0.from_camera_frame_to_image(X_01C) # Image plane coordinates.

        # Get new u anv v
        u = c[0, :].reshape(cam_0.imageSize)
        v = c[1, :].reshape(cam_0.imageSize)
        np.savetxt(outDir + "/u.dat", u, fmt="%4d")
        np.savetxt(outDir + "/v.dat", v, fmt="%4d")

        # Get the du and dv.
        du, dv = du_dv(u, v, cam_0.imageSize)
        np.savetxt(outDir + "/du.dat", du.astype(np.int), fmt="%+4d")
        np.savetxt(outDir + "/dv.dat", dv.astype(np.int), fmt="%+4d")

        dudv = np.zeros( ( cam_0.imageSize[0], cam_0.imageSize[1], 2), dtype = np.float32 )
        dudv[:, :, 0] = du
        dudv[:, :, 1] = dv
        np.save(outDir + "/dudv.npy", dudv)

        # Calculate the angle and distance.
        a, d, angleShift = calculate_angle_distance_from_du_dv( du, dv, flagDegree )
        np.savetxt(outDir + "/a.dat", a, fmt="%+.2e")
        np.savetxt(outDir + "/d.dat", d, fmt="%+.2e")

        angleAndDist = make_angle_distance(cam_0, a, d)
        np.save(outDir + "/ad.npy", angleAndDist)

        # warp the image to see the result
        warppedImg, meanWarpError = warp_image(imgDir, poseID_0, poseID_1, imgSuffix, imgExt, X_01C, X1C, u, v)

        if ( meanWarpError > warpErrThres ):
            # print("meanWarpError (%f) > warpErrThres (%f). " % ( meanWarpError, warpErrThres ))
            overWarpErrThresList.append( { "idx": i, "poseID_0": poseID_0, "poseID_1": poseID_1, "meanWarpError": meanWarpError } )

        if ( meanWarpError > warpErrMaxEntry["warpErr"] ):
            warpErrMaxEntry["idx"] = i
            warpErrMaxEntry["poseID_0"] = poseID_0
            warpErrMaxEntry["poseID_1"] = poseID_1
            warpErrMaxEntry["warpErr"]  = meanWarpError

        cv2.imshow('img', warppedImg)
        # The waitKey() will be executed in show() later.
        # cv2.waitKey(0)

        # Show and save the resulting HSV image.
        if ( 1 == estimatedLoops ):
            show(a, d, outDir, None, angleShift, flagShowFigure=flagShowFigure)
        else:
            show(a, d, outDir, (int)(inputParams["imageWaitTimeMS"]), mf, angleShift, flagShowFigure=flagShowFigure)

        count += 1

        # if ( count >= idxNumberRequest ):
        #     print("Loop number hits the request number. Stop here.")
        #     break

    # show_delimiter("Summary.")
    # print("%d poses, starting at idx = %d, step = %d, %d steps in total. idxNumberRequest = %d\n" % (nPoses, inputParams["startingIdx"], idxStep, count, idxNumberRequest))

    # print_over_warp_error_list( overWarpErrThresList, warpErrThres )

    # print_max_warp_error( warpErrMaxEntry )

    # if ( args.mf >= 0 ):
    #     print( "Command line argument --mf %f overwrites the parameter \"imageMagnitudeFactor\" (%f) in the input JSON file.\n" % (mf, inputParams["imageMagnitudeFactor"]) )

    return overWarpErrThresList, warpErrMaxEntry

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute the image flow data from sequence of camera poses and their depth information.')

    parser.add_argument("--input", help = "The filename of the input JSON file.", default = INPUT_JSON)
    parser.add_argument("--mf",\
        help = "The iamge magnitude factor. If not specified, the value in the input JSON file will be used. Overwrite the value in the input JSON file is specifiec here.",\
        default = -1.0, type = float)
    parser.add_argument("--debug", help = "Debug information including 3D point clouds will be written addintionally.", action = "store_true", default = False)

    args = parser.parse_args()

    # Read the JSON input file.
    inputParams = read_input_parameters_from_json( args.input )

    # Check if use degree as the unit of angle
    flagDegree = inputParams["flagDegree"]
    if ( True == flagDegree ):
        print("Convert angle from radian to degree as demanded by the input file.")

    # Load the pose filenames and the pose data.
    poseIDs, poseData = load_pose_id_pose_data( inputParams, args )
    print("poseData and poseFilenames loaded.")

    # Get the number of poseIDs.
    nPoses = len( poseIDs )
    idxNumberRequest = inputParams["idxNumberRequest"]

    idxStep = inputParams["idxStep"]

    idxList = [ i for i in range( inputParams["startingIdx"], nPoses, idxStep ) ]
    if ( idxNumberRequest < len(idxList)-1 ):
        idxList = idxList[:idxNumberRequest+1]

    startII, endII = 0, len( idxList ) - 1

    # Process.
    overWarpErrThresList, warpErrMaxEntry = \
        process_single_thread("Single", inputParams, args, poseIDs, poseData, idxList, startII, endII, flagShowFigure=True)

    show_delimiter("Summary.")
    print("%d poses, starting at idx = %d, step = %d, %d steps in total. idxNumberRequest = %d\n" % (nPoses, inputParams["startingIdx"], idxStep, len( idxList )-1, idxNumberRequest))

    print_over_warp_error_list( overWarpErrThresList, inputParams["warpErrorThreshold"] )

    print_max_warp_error( warpErrMaxEntry )

    if ( args.mf >= 0 ):
        print( "Command line argument --mf %f overwrites the parameter \"imageMagnitudeFactor\" (%f) in the input JSON file.\n" % (mf, inputParams["imageMagnitudeFactor"]) )
