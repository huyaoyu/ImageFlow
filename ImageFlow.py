#!/usr/bin/python

from __future__ import print_function

import argparse
import copy
import cv2
import json
import math
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import numpy.linalg as LA
import os
import pandas
import pcl
import queue # python3.
import time

from CommonType import NP_FLOAT

from Camera import CameraBase
from SimpleGeometory import distance_of_2_points
from SimplePLY import output_to_ply 
import Utils
import WorkDirectory as WD

# NumPy dtype for PCL routines.
PCL_FLOAT=np.float32

# Global variables used as constants.

WORLD_ORIGIN  = np.zeros((3, 1), dtype=NP_FLOAT)
CAMERA_ORIGIN = np.zeros((3, 1), dtype=NP_FLOAT)

SELF_OCC  = 2
CROSS_OCC = 1

OUT_OF_FOV_POSITIVE_Z = 11
OUT_OF_FOV_NEGATIVE_Z = 12

def compose_m(R, t):
    m = np.zeros((4, 4), dtype=NP_FLOAT)
    m[0:3, 0:3] = R
    m[0:3, 3]   = t.reshape((-1,))
    m[3,3]      = 1.0

    return m

def inv_m(m):
    im = np.zeros_like(m, dtype=NP_FLOAT)

    R = m[0:3, 0:3]
    t = m[0:3, 3].reshape((3, 1))

    im[0:3, 0:3] = R.transpose()
    im[0:3, 3]   = -R.transpose().dot( t ).reshape((-1,))
    im[3, 3]     = 1.0

    return im

def du_dv(nu, nv, imageSize):
    wIdx = np.linspace( 0, imageSize[1] - 1, imageSize[1], dtype=np.int )
    hIdx = np.linspace( 0, imageSize[0] - 1, imageSize[0], dtype=np.int )

    u, v = np.meshgrid(wIdx, hIdx)

    return nu - u, nv - v

def show(ang, mag, mask=None, outDir=None, outName="bgr", waitTime=None, magFactor=1.0, angShift=0.0, flagShowFigure=True):
    """ang: degree"""
    # Use Hue, Saturation, Value colour model 
    hsv = np.zeros( ( ang.shape[0], ang.shape[1], 3 ) , dtype=np.uint8)
    hsv[..., 1] = 255

    # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1]

    hsv[..., 0] = (ang + angShift)/ 2
    hsv[..., 2] = np.clip(mag * magFactor, 0, 255).astype(np.uint8) #cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if ( mask is not None ):
        mask = mask != 255
        bgr[mask] = np.array([0, 0 ,0], dtype=np.uint8)

    if ( outDir is not None ):
        cv2.imwrite(outDir + "/%s_vis.png" % (outName), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    if ( True == flagShowFigure ):
        cv2.imshow("HSV2GBR", bgr)

        if ( waitTime is None ):
            cv2.waitKey()
        else:
            cv2.waitKey( waitTime )

from numba import cuda

@cuda.jit
def k_filter_mask(mask, v, nv):
    tx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    ty = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    xStride = cuda.blockDim.x * cuda.gridDim.x * 3
    yStride = cuda.blockDim.y * cuda.gridDim.y * 3

    for y in range(ty*3+1, mask.shape[0]-1, yStride):
        for x in range(tx*3+1, mask.shape[1]-1, xStride):
            for i in range(3):
                my = y + i
                for j in range(3):
                    mx = x + j

                    if ( mask[my, mx] == v ):
                        if ( mask[my,   mx-1] != v and \
                             mask[my,   mx+1] != v and \
                             mask[my-1, mx]   != v and \
                             mask[my+1, mx]   != v ):
                            mask[my, mx] = nv

def filter_mask(mask, v, nv):
    """
    v is the value for labeled pixel.
    nv is the value for non-labeled pixel.
    """

    h, w = mask.shape[0], mask.shape[1]

    for i in range(1,h-1):
        for j in range(1,w-1):
            if ( mask[i, j] == v):
                if ( mask[i, j-1] != v and \
                     mask[i, j+1] != v and \
                     mask[i-1, j] != v and \
                     mask[i+1, j] != v ):
                    mask[i, j] = nv

    # dMask = cuda.to_device(mask)

    # threadsOneDim = 16
    # coveragePerBlock = threadsOneDim * 3 + 2
    # blocksH = int(math.ceil( h / coveragePerBlock ))
    # blocksW = int(math.ceil( w / coveragePerBlock ))

    # cuda.synchronize()
    # k_filter_mask[[blocksW, blocksH, 1], [threadsOneDim, threadsOneDim, 1]](dMask, v, nv)
    # cuda.synchronize()

    # mask = dMask.copy_to_host()

    return mask

def get_center_and_neighbors(h, w, i, j):
    c = i*w+j
    idx = [c] # Center.
    
    if ( j == 0 ):
        idx.append( c+1 )
        idx.append( c+1 )
    elif ( j == w - 1 ):
        idx.append( c-1 )
        idx.append( c-1 )
    else:
        idx.append( c-1 )
        idx.append( c+1 )

    if ( i == 0 ):
        idx.append( c+w )
        idx.append( c+w )
    elif ( i == h - 1 ):
        idx.append( c-w )
        idx.append( c-w )
    else:
        idx.append( c-w )
        idx.append( c+w )

    return idx

def get_distance_from_coordinate_table(tab, h, w, i, j, showDetail=False):
    """
    tab: A 3-row table contains 3D coordinates.
    h, w: The height and width of the original image.
    i, j: The indices along the height and width directions.

    This funcion will return the distance of a point specified
    by i, j, and its 4 neighbors.
    """

    idx = get_center_and_neighbors( h, w, i, j )

    # Get the x, y, z.
    x = tab[0, idx]
    y = tab[1, idx]
    z = tab[2, idx]

    c = np.sqrt( x[0]**2 + y[0]**2 + z[0]**2 )
    relativeDist = np.sqrt( (x-x[0])**2 + (y-y[0])**2 + (z-z[0])**2 )

    ddMin = relativeDist.min()
    ddMax = relativeDist.max()

    if ( showDetail ):
        print("idx = {}. ".format(idx))
        print("x = {}. ".format(x))
        print("y = {}. ".format(y))
        print("z = {}. ".format(z))
        print("c = {}. ".format(c))
        print("relativeDist = {}. ".format(relativeDist))

    return c, ddMin, ddMax, np.array( ( x[0], y[0], z[0] ), dtype=NP_FLOAT )

def estimate_surface_normal(x):
    """
    x is a 3-column array contains the 3D points.

    This function is partially adopted from 
    https://github.com/strawlab/python-pcl/blob/master/examples/official/Features/NormalEstimationUsingIntegralImages.py
    """

    # Convert x to pcl point cloud.
    cloud = pcl.PointCloud(x.astype(PCL_FLOAT))

    ne = cloud.make_NormalEstimation()
    ne.set_RadiusSearch(1)
    normals = ne.compute()

    return normals.to_array()

def check_two_surface_normals(n00, n11, v01, tPar = 0.75, eOth = 0.1):
    """
    n00, n11, v01: 3-element NumPy arrays. 3D vectors.
    n00 and n11 are the surface normals at two locations.
    v01 is the vector starts from 0 points to 1.

    tPar and eOth are to control parameters. tPar is the threshould for
    recognizing two vectors as parallel. eOth is the limit for determining 
    if two vectors are othogonal.
    """

    # Normalization.
    n00 = n00.reshape((-1, 1)) / np.linalg.norm(n00, 2)
    n11 = n11.reshape((-1, 1)) / np.linalg.norm(n11, 2)
    v01 = v01.reshape((-1, 1)) / np.linalg.norm(v01, 2)

    # Use inner product as approximinity measure.
    pNN = np.absolute( n00.transpose().dot( n11 ) )
    pNV = np.absolute( n00.transpose().dot( v01 ) )

    if ( pNN > tPar ):
        flagParNN = True
    else:
        flagParNN = False

    if ( pNV > tPar ):
        flagParNV = True
    else:
        flagParNV = False

    if ( pNV < eOth ):
        flagOth = True
    else:
        flagOth = False

    return flagParNN, flagParNV, flagOth, pNN, pNV

def create_warp_masks(imageSize, x01, x1, u, v, cam_0, cam_1, M0, M1, p=0.001, D=1000):
    """
    imageSize: height x width.
    x01: The 3D coordinates of the pixels in the first image observed in the frame of the second camera. 3-row 2D array.
    x1: The 3D coordinates of the pixels in the second image observed in the frame of the second camera. 3-row 2D array.
    u: The u coordinates of the pixel in the second image plane. 1D array.
    v: The v coordinates of the pixel in the second image plane. 1D array.
    p: A coefficient controls the sensitivity of 0-1 occlustion. Unit percentage.
    D: Points that fall beyond this distance will not be checked for occlusion. Unit m.
    """
    # import ipdb;ipdb.set_trace()
    # Check dimensions.
    assert( u.shape[0] == v.shape[0] )
    assert( u.shape[1] == v.shape[1] )
    assert( u.shape[0] * u.shape[1] == x01.shape[1] )
    assert( x01.shape[0] == 3 )

    h = imageSize[0]
    w = imageSize[1]

    # Surface normal estimateion.
    normals01 = estimate_surface_normal(x01.transpose())
    # normals1  = estimate_surface_normal(x1.transpose())

    print("normals01 non nan number = {}. ".format( np.isfinite(normals01).sum() ))

    # Geometory.
    # H   = M0.dot( inv_m(M1) )
    # h00 = np.absolute( H[0, 0] / cam_1.focal )
    # h01 = np.absolute( H[0, 1] / cam_1.focal )
    # h10 = np.absolute( H[1, 0] / cam_1.focal )
    # h11 = np.absolute( H[1, 1] / cam_1.focal )
    # h20 = np.absolute( H[2, 0] / cam_1.focal )
    # h21 = np.absolute( H[2, 1] / cam_1.focal )

    # dxDxy = max( h00, h01 )
    # dyDxy = max( h10, h11 )
    # dzDxy = max( h20, h21 )

    # rMax = np.sqrt( dxDxy**2 + dyDxy**2 + dzDxy**2 )

    # Allocate memory.
    occupancyMap00 = np.zeros( imageSize, dtype=np.int32 ) - 1
    occupancyMap01 = np.zeros( imageSize, dtype=np.int32 ) - 1
    maskOcclusion  = np.zeros( imageSize, dtype=np.uint8 )
    maskFOV        = np.zeros( imageSize, dtype=np.uint8 )

    # Reshape input arguments.
    u = u.reshape((-1,))
    v = v.reshape((-1,))

    # Loop for every pixel index.
    for i in range( h*w ):
        # Get the u and v coordinate of the pixel in the image plane of the second camera.
        iu = int( round( u[i] ) )
        iv = int( round( v[i] ) )

        # Get the u and v coordinate of the pixel in the image plane of the original camera.
        iy = i // w
        ix = i % w

        # Check if the new index is out of boundary?
        if ( iu < 0 or iv < 0 or iu >= w or iv >= h ):
            # Update the FOV mask.
            maskFOV[iy, ix] = OUT_OF_FOV_POSITIVE_Z

            # Stop the current loop.
            continue
        
        # Check if the current point is on the opposite side of the image plane of the second camera.
        if ( x01[2, i] <= 0 ):
            # Update the FOV mask.
            maskFOV[iy, ix] = OUT_OF_FOV_NEGATIVE_Z

            # Stop the current loop.
            continue

        showDetail = False
        if ( iy == 348 and ix == 88 ):
            showDetail = True

        # if ( iu == 282 and iv == 123 ):
        #     showDetail = True

        if ( showDetail ):
            print("iy = %d, ix = %d. " % ( iy, ix ))
            print("iu = %d, iv = %d. " % ( iu, iv ))

        # Get the current depth.
        d0, ddMin0, ddMax0, coor0 = get_distance_from_coordinate_table(x01, h, w, iy, ix, showDetail)
        dr = 0.0
        opIndex = np.nan
        flagParNN, flagParNV, flagOth, pNN, pNV = None, None, None, None, None

        # Check if the new index is occupied.
        if ( -1 != occupancyMap00[iv, iu] ):
            # This pixel is occupied.

            # Get the index registered in the occupancy map.
            opIndex = occupancyMap00[iv, iu]
            opY = opIndex // w
            opX = opIndex % w

            # Get the depth at the registered index.
            dr, ddMin, ddMax, coorR = get_distance_from_coordinate_table(x01, h, w, opY, opX)

            if ( d0 < dr ):
                # Current point is nearer to the camera.
                # Update the occupancy map.
                occupancyMap00[ iv, iu ] = i

                if ( np.isfinite(normals01[i]).sum() >= 3 and \
                     np.isfinite(normals01[opIndex]).sum() >= 3 ):
                    # Check surface normal.
                    v01 = np.array([ coorR[0] - coor0[0], coorR[1] - coor0[1], coorR[2] - coor0[2] ], dtype=NP_FLOAT)
                    flagParNN, flagParNV, flagOth, pNN, pNV = check_two_surface_normals( \
                        normals01[opIndex][:3], normals01[i][:3], v01 )

                    if ( flagOth ):
                        # Not an occlusion.
                        pass
                    else:
                        # Update the occlusion mask.
                        maskOcclusion[ opY, opX ] = SELF_OCC
                else:
                    # Update the occlusion mask.
                    maskOcclusion[ opY, opX ] = SELF_OCC

            elif ( d0 > dr ):
                # Current point is farther.

                if ( np.isfinite(normals01[i]).sum() >= 3 and \
                     np.isfinite(normals01[opIndex]).sum() >= 3 ):
                    # Check surface normal.
                    v01 = np.array([ coorR[0] - coor0[0], coorR[1] - coor0[1], coorR[2] - coor0[2] ], dtype=NP_FLOAT)
                    flagParNN, flagParNV, flagOth, pNN, pNV = check_two_surface_normals( \
                        normals01[i][:3], normals01[opIndex][:3], v01 )

                    if ( flagOth ):
                        # Not an occlusion.
                        pass
                    else:
                        # Update the occlusion mask.
                        maskOcclusion[ iy, ix ] = SELF_OCC
                else:
                    # Update the occlusion mask.
                    maskOcclusion[ iy, ix ] = SELF_OCC

                # Stop the current loop.
                continue

            else:
                raise Exception("%d pixel has same distance with %d pixel." % ( i, opIndex ))

        else:
           # Update the occupancy map.
            occupancyMap00[ iv, iu ] = i 

        if (showDetail):
            print("d0 = %f, dr = %f. " % ( d0, dr ))
            print("coor0 = {}. ".format( coor0 ))
            if ( np.isfinite(opIndex) ):
                print("normals01[{}] = {}. ".format( i, normals01[i] ))
                print("normals01[{}] = {}. ".format( opIndex, normals01[opIndex] ))
                print("v01 = {}. ".format( v01 ))
                print("flagParNN = {}, flagParNV = {}, flagOth = {}. ".format( flagParNN, flagParNV, flagOth ))
                print("pNN = {}, pNV = {}. ".format(pNN, pNV))

        # Get the depth at x=iu, y=iv in the second image observed in the second camera.
        d1, ddMin1, ddMax1, coor1 = get_distance_from_coordinate_table(x1, h, w, iv, iu, showDetail)
        dist01 = distance_of_2_points( coor0, coor1 )

        if (showDetail):
            print("d0 = %f, dr = %f, d1 = %f, ddMin0 = %f, ddMax0 = %f. " % ( d0, dr, d1, ddMin0, ddMax0 ))
            print("coor0 = {}. ".format( coor0 ))
            print("coor1 = {}. ".format( coor1 ))
            print("dist01 = %f. " % (dist01))
            # print("M0 = {}".format(M0))
            # print("M1 = {}".format(M1))
            # print("H = {}".format(H))
            print("cam_1.forcal = %f. " % ( cam_1.focal ))
            # print("rMax = %f, rMax*coor1[2] = %f. " % ( rMax, rMax*coor1[2] ))
            print("d0 - d1 = %f." % (d0 - d1))
            print("0.6*ddMax = %f." % (ddMax0))

        if ( d0 >= D and d1 >= D ):
            # Points at infinity do not occlude each other.
            pass
        elif ( d0 <= d1 or d0 - d1 <= 1.1*ddMax0 ):
            # Current point is nearer to the camera or equals the distance of the corresponding pixel in the second image.
            # This is subject to the value of p. p is not check constantly.
            pass
        else:
            # Current point is occluded by the corresponding pixel in the second image.
            # Update the occlusion mask.
            if ( maskOcclusion[ iy, ix ] != SELF_OCC ):
                maskOcclusion[ iy, ix ] = CROSS_OCC

            if (showDetail):
                print( "(iy %d, ix %d) cross-occ." % (iy, ix) )

            # if ( -1 != occupancyMap01[iv, iu] ):
            #     # if ( occupancyMap01[iv, iu] == opIndex ):
            #     #     continue

            #     raise Exception( "Current pixel %d, wins pre-registered %d but occlued by second image at x=%d, y=%d (occupancyMap01: %d) with d0=%f, dr=%f, d1=%f, ddMin=%f, ddMax=%f. " \
            #         % ( i, opIndex, iu, iv, occupancyMap01[iv, iu], d0, dr, d1, ddMin, ddMax ) )

            continue

        # Update the occupancy map.
        occupancyMap01[ iv, iu ] = i
        
    maskOcclusion = filter_mask(maskOcclusion, SELF_OCC, 0)
    maskOcclusion = filter_mask(maskOcclusion, CROSS_OCC, 0)

    return maskOcclusion, maskFOV, occupancyMap00, occupancyMap01

def warp_error_by_index( img0, img1, u, v, idx0 ):
    h = img0.shape[0]
    w = img0.shape[1]
    
    # All u and v need to be evaluated in img1.
    u1 = u.reshape((-1,))[idx0]
    v1 = v.reshape((-1,))[idx0]

    u1 = np.around(u1).astype(np.int32)
    v1 = np.around(v1).astype(np.int32)

    # Convert u1 and v1 into linear index.
    idx1 = v1 * img1.shape[1] + u1
    idx1 = idx1.astype(np.int32)

    # Reshape the input image.
    img0 = img0.reshape((-1, img0.shape[2])).astype(np.int32)
    img1 = img1.reshape((-1, img1.shape[2])).astype(np.int32)

    # Absolute difference.
    diff = img0[idx0, :] - img1[idx1, :]
    diff = np.sqrt( np.linalg.norm( diff, 2, axis=1 ) )

    # Make diff to be an image.
    dImg0 = np.zeros( h*w, dtype=NP_FLOAT )
    dImg1 = np.zeros( h*w, dtype=NP_FLOAT )
    dImg0[idx0] = diff
    dImg1[idx1] = diff

    return dImg0.reshape( (h, w) ), dImg1.reshape( (h, w) )

def evaluate_warp_error( img0, img1, x01, x1, u, v, cam_0, cam_1, M0, M1 ):
    """
    x01: The 3D coordinates of the pixels in the first image observed in the frame of the second camera. 3-row 2D array.
    x1: The 3D coordinates of the pixels in the second image observed in the frame of the second camera. 3-row 2D array.
    u: The u coordinates of the pixel in the second image plane. 1D array.
    v: The v coordinates of the pixel in the second image plane. 1D array.
    """

    h = img0.shape[0]
    w = img0.shape[1]

    # Get the masks.
    maskOcclusion, maskFOV, occupancyMap00, occupancyMap01 = create_warp_masks( img0.shape[:2], x01, x1, u, v, cam_0, cam_1, M0, M1 )

    # Make a mask for the occupancyMap00
    mask00 = occupancyMap00 != -1 

    # All indices need to be evaluated in img0.
    idx0_00 = occupancyMap00[mask00].astype(np.int32)

    # Warp error by index.
    dImg0_00, dImg1_00 = warp_error_by_index(img0, img1, u, v, idx0_00)

    # 01.
    mask01  = occupancyMap01 != -1
    idx0_01 = occupancyMap01[mask01].astype(np.int32)
    dImg0_01, dImg1_01 = warp_error_by_index(img0, img1, u, v, idx0_01)

    return dImg0_00, dImg1_00, occupancyMap00, mask00, \
           dImg0_01, dImg1_01, occupancyMap01, mask01, maskOcclusion, maskFOV

def warp_image(imgDir, poseID_0, poseID_1, imgSuffix, imgExt, X_01C, X1C, u, v, \
        cam_0, cam_1, M0, M1):
    cam0ImgFn = "%s/%s%s%s" % ( imgDir, poseID_0, imgSuffix, imgExt )
    cam1ImgFn = "%s/%s%s%s" % ( imgDir, poseID_1, imgSuffix, imgExt )

    # print("Warp %s." % (cam0ImgFn))
    cam0_img = cv2.imread( cam0ImgFn, cv2.IMREAD_UNCHANGED )
    
    # Evaluate warp error.
    cam1_img = cv2.imread( cam1ImgFn, cv2.IMREAD_UNCHANGED )
    
    dImg0_00, dImg1_00, occupancyMap_00, occupancyMask_00, \
    dImg0_01, dImg1_01, occupancyMap_01, occupancyMask_01, \
    maskOcc, maskFOV \
         = evaluate_warp_error( cam0_img, cam1_img, X_01C, X1C, u, v, cam_0, cam_1, M0, M1 )

    # Warp the image.
    warppedImg = np.zeros_like(cam0_img, dtype=np.uint8)
    
    validWarpMask = occupancyMask_00
    validWarpIdx  = occupancyMap_00[validWarpMask]

    # Reshape to (HxW, C)
    cam0ImgCpy = copy.deepcopy(cam0_img).reshape( (-1, cam0_img.shape[2]) )
    warppedImg = warppedImg.reshape( (-1, cam0_img.shape[2]) )

    warppedImg[validWarpMask.reshape( (-1,) ), :] = cam0ImgCpy[ validWarpIdx, : ]
    warppedImg = warppedImg.reshape( cam0_img.shape )

    return maskOcc, maskFOV, warppedImg, dImg1_00, dImg1_01, occupancyMask_00, occupancyMask_01

def get_magnitude_factor_from_input_parameters(params, args):
    if ( args.mf < 0.0 ):
        mf = params["imageMagnitudeFactor"]
    else:
        mf = args.mf

    return mf

def load_pose_id_pose_data(params, args):
    dataDir = params["dataDir"]

    poseIDsFn = dataDir + "/" + params["poseFilename"]

    if ( not os.path.isfile(poseIDsFn) ):
        # File not exist. Create on the fly.
        WD.create_pose_id_file( dataDir, params["imageDir"], "*%s" % (params["imageExt"]), params["poseFilename"] )

    _, poseIDs = WD.load_IDs_JSON(\
        poseIDsFn, params["poseName"])

    poseDataFn = dataDir + "/" + params["poseData"]
    
    if ( ".txt" == os.path.splitext( os.path.split(poseDataFn)[1] )[1] ):
        poseData = np.loadtxt( poseDataFn, dtype=NP_FLOAT )
    else:
        poseData = np.load( poseDataFn ).astype(NP_FLOAT)

        if ( True == args.debug ):
            np.savetxt( dataDir + "/poseData.dat", poseData, fmt="%+.4e" )

    return poseIDs, poseData

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
    angleAndDist = np.zeros( ( cam.imageSize[0], cam.imageSize[1], 2), dtype = NP_FLOAT )
    angleAndDist[:, :, 0] = a
    angleAndDist[:, :, 1] = d

    return angleAndDist

def print_over_warp_error_list(overWarpErrList, t, fn):
    # { "idx": i, "poseID_0": poseID_0, "poseID_1": poseID_1, "meanWarpError": meanWarpError }

    if ( 0 != len( overWarpErrList ) ):
        print( "%d over warp error threshold (%f). " % ( len( overWarpErrList ), t ) )
        print( "idx, poseID_0, poseID_1, meanWarpError" )
    else:
        print( "No warp error over the threshold (%f). " % (t) )
        return
    
    fp = open(fn, "w")
    fp.write("idx, poseID_0, poseID_1, meanWarpError, meanWarpError_01\n")

    for entry in overWarpErrList:
        s = "%d, %s, %s, %f, %f" % ( entry["idx"], entry["poseID_0"], entry["poseID_1"], entry["meanWarpError"], entry["meanWarpError_01"] )
        print( s )

        s += "\n"
        fp.write(s)
    
    fp.close()

def print_max_warp_error(entry):
    if ( entry["idx"] != -1 ):
        print( "Max mean warp error: " )
        print( "idx: %d, poseIDs: %s - %s, mean error: %f, mean error 01: %f. " % \
            ( entry["idx"], entry["poseID_0"], entry["poseID_1"], entry["warpErr"], entry["warpErr_01"] ) )
    else:
        raise Exception( "Wrong max warp error entry: idx: %d, poseIDs: %s - %s, mean error: %f. " % \
            ( entry["idx"], entry["poseID_0"], entry["poseID_1"], entry["warpErr"], entry["warpErr_01"] ) )

def save_flow(fnBase, flowSuffix, maskSuffix, du, dv, maskOcc, maskFOV):
    """
    fnBase: The file name base.
    flowSuffix: Filename suffix of the optical flow file.
    maskSuffix: Filename suffix of the mask file.
    du, dv: The opitcal flow saved as NumPy array with dtype=numpy.float32.
    maskOcc: The occlusion mask. NumPy array with dtype=numpy.uint8.
    maskFOV: The FOV mask. NumPy array with dtype=numpy.uint8.

    Values saved in maskOcc and maskFOV are interpreted by referring to 
    SELF_OCC, CROSS_OCC, OUT_OF_FOV_POSITIVE_Z, OUT_OF_FOV_NEGATIVE_Z.

    Optical flow masked as OUT_OF_FOV_NEGATIVE_Z will be labeled as invalid.
    Other labels in maskOcc and maskFOV will be fused into a single mask file.

    The optical flow file will be saved as a binary NumPy output with dytpe=numpy.float32.
    The mask file will be saved as a binary NumPy output with dtype=numpy.uint8.
    - A 0 in the mask file means invalid pixel.
    - A 255 in the mask file means valid non-occluded pixel.
    - Values of SELF_OCC, CROSS_OCC, and OUT_OF_FOV_POSITIVE_Z are used to indicate general 
    occlusion. 
    The mask file is obtained by first setting the NumPy array to all 255. Then assign SELF_OCC, 
    CROSS_OCC, OUT_OF_FOV_POSITIVE_Z orderly. Finally, 0 value is assigned according to the 
    OUT_OF_FOV_NEGATIVE_Z distribution in maskFOV.

    All invalid flow is set to zero.
    """

    # Create a 2-channel NumPy array.
    flow = np.stack([du, dv], axis=-1).astype(NP_FLOAT)

    # Create a 1-channel NumPy array.
    mask = np.zeros_like(du, dtype=np.uint8) + 255

    tempMask = maskOcc == CROSS_OCC
    mask[tempMask] = CROSS_OCC

    tempMask = maskOcc == SELF_OCC
    mask[tempMask] = SELF_OCC

    tempMask = maskFOV == OUT_OF_FOV_POSITIVE_Z
    mask[tempMask] = OUT_OF_FOV_POSITIVE_Z

    tempMask = maskFOV == OUT_OF_FOV_NEGATIVE_Z
    mask[tempMask] = 0
    flow[tempMask] = np.array([0.0, 0.0], dtype=NP_FLOAT)

    # Save the files.
    np.save( "%s%s.npy" % (fnBase, flowSuffix), flow )
    np.save( "%s%s.npy" % (fnBase, maskSuffix), mask )

def process_single_process(name, outDir, \
    imgDir, poseID_0, poseID_1, imgSuffix, imgExt, 
    poseDataLine_0, poseDataLine_1, depth_0, depth_1, distanceRange, 
    cam_0, cam_1, 
    flagErrorExtra=False, errorExtraDir=None, 
    flagShowFigure=False, flagDebug=False, debugOutDir="./"):
    
    # Get the pose of the first position.
    R0, t0, q0 = WD.get_pose_from_line(poseDataLine_0)
    R0Inv = LA.inv(R0)
    M0 = compose_m(R0, t0)

    if ( flagDebug ):
        print("t0 = \n{}".format(t0))
        print("q0 = \n{}".format(q0))
        print("R0 = \n{}".format(R0))
        print("R0Inv = \n{}".format(R0Inv))

    # Get the pose of the second position.
    R1, t1, q1 = WD.get_pose_from_line(poseDataLine_1)
    R1Inv = LA.inv(R1)
    M1 = compose_m(R1, t1)

    if ( flagDebug ):
        print("t1 = \n{}".format(t1))
        print("q1 = \n{}".format(q1))
        print("R1 = \n{}".format(R1))
        print("R1Inv = \n{}".format(R1Inv))

    # Compute the rotation between the two camera poses.
    R = np.matmul( R1, R0Inv )

    if ( flagDebug ):
        print("R = \n{}".format(R))

    # Calculate the coordinates in the first camera's frame.
    # X0C = cam_0.from_depth_to_x_y(depth_0) # Coordinates in the camera frame. z-axis pointing forwards.
    X0C = cam_0.from_depth_to_x_y(depth_0, distanceRange) # Coordinates in the camera frame. z-axis pointing forwards.
    X0  = cam_0.worldRI.dot(X0C)           # Corrdinates in the NED frame. z-axis pointing downwards.

    if ( flagDebug ):
        try:
            output_to_ply(debugOutDir + '/XInCam_0.ply', X0C, cam_0.imageSize, distanceRange, CAMERA_ORIGIN)
        except Exception as e:
            print("Cannot write PLY file for X0. Exception: ")
            print(e)

    # The coordinates in the world frame.
    XWorld_0  = R0Inv.dot(X0 - t0)

    if ( flagDebug ):
        try:
            output_to_ply(debugOutDir + "/XInWorld_0.ply", XWorld_0, cam_1.imageSize, distanceRange, -R0Inv.dot(t0))
        except Exception as e:
            print("Cannot write PLY file for XWorld_0. Exception: ")
            print(e)

    # Calculate the coordinates in the second camera's frame.
    # X1C = cam_1.from_depth_to_x_y(depth_1) # Coordinates in the camera frame. z-axis pointing forwards.
    X1C = cam_1.from_depth_to_x_y(depth_1, distanceRange) # Coordinates in the camera frame. z-axis pointing forwards.

    if ( flagDebug ):
        X1  = cam_1.worldRI.dot(X1C)           # Corrdinates in the NED frame. z-axis pointing downwards.

        try:
            output_to_ply(debugOutDir + "/XInCam_1.ply", X1C, cam_1.imageSize, distanceRange, CAMERA_ORIGIN)
        except Exception as e:
            print("Cannot write PLY file for X1. Exception: ")
            print(e)

        # The coordiantes in the world frame.
        XWorld_1 = R1Inv.dot( X1 - t1 )

        try:
            output_to_ply(debugOutDir + "/XInWorld_1.ply", XWorld_1, cam_1.imageSize, distanceRange, -R1Inv.dot(t1))
        except Exception as e:
            print("Cannot write PLY file for XWorld_1. Exception: ")
            print(e)

    # ====================================
    # The coordinate of the pixels of the first camera projected in the second camera's frame (NED).
    X_01 = R1.dot(XWorld_0) + t1

    # The image coordinates in the second camera.
    X_01C = cam_0.worldR.dot(X_01)                  # Camera frame, z-axis pointing forwards.
    c     = cam_0.from_camera_frame_to_image(X_01C) # Image plane coordinates.

    if ( flagDebug ):
        try:
            output_to_ply(debugOutDir + '/X_01C.ply', X_01C, cam_0.imageSize, distanceRange, CAMERA_ORIGIN)
        except Exception as e:
            print("Cannot write PLY file for X_01. Exception: ")
            print(e)

    # Get new u anv v
    u = c[0, :].reshape(cam_0.imageSize)
    v = c[1, :].reshape(cam_0.imageSize)

    # Get the du and dv.
    du, dv = du_dv(u, v, cam_0.imageSize)

    dudv = np.zeros( ( cam_0.imageSize[0], cam_0.imageSize[1], 2), dtype = NP_FLOAT )

    # # Calculate the angle and distance.
    # a, d, angleShift = calculate_angle_distance_from_du_dv( du, dv, flagDegree )
    # angleAndDist     = make_angle_distance(cam_0, a, d)

    # warp the image to see the result
    maskOcc, maskFOV, warppedImg, dImg1_00, dImg1_01, occupancyMask_00, occupancyMask_01 \
         = warp_image(imgDir, poseID_0, poseID_1, imgSuffix, imgExt, X_01C, X1C, u, v, \
             cam_0, cam_1, M0, M1)

    save_flow( "%s/%s" % (outDir, poseID_0), "_flow", "_mask", du, dv, maskOcc, maskFOV )

    # The mean warp error over the valid pixels in the seconde image.
    minError_00  = dImg1_00[occupancyMask_00].min()
    maxError_00  = dImg1_00.max()
    meanError_00 = dImg1_00[occupancyMask_00].mean()
    minError_01  = dImg1_01[occupancyMask_01].min()
    maxError_01  = dImg1_01.max()
    meanError_01 = dImg1_01[occupancyMask_01].mean()

    if ( flagErrorExtra ):
        warpErrImgFn = "%s/%s_%s%s%s%s" % ( errorExtraDir, poseID_0, poseID_1, imgSuffix, "_error_00", imgExt )
        
        # Save the error image.
        Utils.save_float_image( warpErrImgFn, dImg1_00 )

        # Save the warpped image.
        cam0WrpFn = "%s/%s_%s%s%s%s" % ( errorExtraDir, poseID_0, poseID_1, imgSuffix, "_warp", imgExt )
        cv2.imwrite(cam0WrpFn, warppedImg)

    return warppedImg, ( minError_00, maxError_00, meanError_00, minError_01, maxError_01, meanError_01 )

def logging_worker(name, jq, p, workingDir):
    import logging

    logger = logging.getLogger("ImageFlow")
    logger.setLevel(logging.INFO)

    logFn = "%s/Log.log" % (workingDir)
    print(logFn)

    fh = logging.FileHandler( logFn, "w" )
    fh.setLevel(logging.INFO)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info("%s: Logger initialized." % (name))

    while (True):
        if (p.poll()):
            command = p.recv()

            print("%s: %s command received." % (name, command))

            if ("exit" == command):
                # print("%s: exit." % (name))
                break
        
        try:
            job = jq.get(False)
            # print("{}: {}.".format(name, jobStrList))

            logger.info(job)

            jq.task_done()
        except queue.Empty as exp:
            pass
    
    logger.info("Logger exited.")

def worker(name, jq, rq, lq, p, inputParams, args):
    """
    name: String, the name of this worker process.
    jq: A JoinableQueue. The job queue.
    rq: The report queue.
    lq: The logger queue.
    p: A pipe connection object. Only for receiving.
    """

    lq.put("%s: Worker starts." % (name))

    # ==================== Preparation. ========================

    # Data directory.
    dataDir = inputParams["dataDir"]

    # The magnitude factor.
    mf = get_magnitude_factor_from_input_parameters( inputParams, args )

    # Camera.
    cam_0 = CameraBase(inputParams["camera"]["focal"], inputParams["camera"]["imageSize"])

    # We are assuming that the cameras at the two poses are the same camera.
    cam_1 = cam_0

    outDir        = dataDir + "/" + inputParams["outDir"]
    depthDir      = dataDir + "/" + inputParams["depthDir"]
    imgDir        = dataDir + "/" + inputParams["imageDir"]
    imgSuffix     = inputParams["imageSuffix"]
    imgExt        = inputParams["imageExt"]
    depthTail     = inputParams["depthSuffix"] + inputParams["depthExt"]
    distanceRange = inputParams["distanceRange"]
    flagDegree    = inputParams["flagDegree"]
    warpErrThres  = inputParams["warpErrorThreshold"]

    if ( args.save_error_extra ):
        errExDir = "%s%s" % ( outDir, args.error_extra_dir_suffix )
    else:
        errExDir = None

    count = 0

    while (True):
        if (p.poll()):
            command = p.recv()

            lq.put("%s: %s command received." % (name, command))

            if ("exit" == command):
                # print("%s: exit." % (name))
                break

        try:
            job = jq.get(True, 1)
            # print("{}: {}.".format(name, jobStrList))

            poseID_0 = job["poseID_0"]
            poseID_1 = job["poseID_1"]
            poseDataLineList_0 = job["poseLineList_0"]
            poseDataLineList_1 = job["poseLineList_1"]

            poseDataLine_0 = np.array( poseDataLineList_0, dtype=NP_FLOAT )
            poseDataLine_1 = np.array( poseDataLineList_1, dtype=NP_FLOAT )

            # Load the depth.
            depth_0 = np.load( depthDir + "/" + poseID_0 + depthTail ).astype(NP_FLOAT)
            depth_1 = np.load( depthDir + "/" + poseID_1 + depthTail ).astype(NP_FLOAT)

            # If it is debugging.
            if ( args.debug ):
                debugOutDir = "%s/ImageFlow/%s" % ( dataDir, poseID_0 )
                Utils.test_dir(debugOutDir)
            else:
                debugOutDir = "./"

            # Process.
            warppedImg, errorTuple = \
                process_single_process(name, outDir, 
                    imgDir, poseID_0, poseID_1, imgSuffix, imgExt, 
                    poseDataLine_0, poseDataLine_1, depth_0, depth_1, distanceRange, 
                    cam_0, cam_1, 
                    flagErrorExtra=args.save_error_extra, errorExtraDir=errExDir,
                    flagShowFigure=False, flagDebug=args.debug, debugOutDir=debugOutDir)

            rq.put( { "idx": job["idx"], \
                "poseID_0": poseID_0, "poseID_1": poseID_1, 
                "minErr_00": errorTuple[0], "maxErr_00": errorTuple[1], "avgErr_00": errorTuple[2],
                "minErr_01": errorTuple[3], "maxErr_01": errorTuple[4], "avgErr_01": errorTuple[5] } )

            count += 1

            lq.put("%s: idx = %d. " % (name, job["idx"]))

            jq.task_done()
        except queue.Empty as exp:
            pass
    
    lq.put("%s: Done with %d jobs." % (name, count))

def reshape_idx_array(idxArray):
    N = idxArray.size
    idxArray = idxArray.astype(np.int)

    # Find the closest squre root.
    s = int(math.ceil(math.sqrt(N)))

    # Create a new index array.
    idx2D = np.zeros( (s*s, ), dtype=np.int ) + N
    idx2D[:N] = idxArray

    # Reshape and transpose.
    idx2D = idx2D.reshape((s, s)).transpose()

    # Flatten again.
    return idx2D.reshape((-1, ))

def process_report_queue(rq):
    """
    rq: A Queue object containing the report data.
    """

    count = 0

    mergedDict = { "idx": [], "poseID_0": [], "poseID_1": [], \
        "minErr_00": [], "maxErr_00": [], "avgErr_00": [], \
        "minErr_01": [], "maxErr_01": [], "avgErr_01": [] }

    try:
        while (True):
            r = rq.get(block=False)

            mergedDict['idx'].append( r["idx"] )
            mergedDict['poseID_0'].append( r["poseID_0"] )
            mergedDict['poseID_1'].append( r["poseID_1"] )
            mergedDict['minErr_00'].append( r["minErr_00"] )
            mergedDict['maxErr_00'].append( r["maxErr_00"] )
            mergedDict['avgErr_00'].append( r["avgErr_00"] )
            mergedDict['minErr_01'].append( r["minErr_01"] )
            mergedDict['maxErr_01'].append( r["maxErr_01"] )
            mergedDict['avgErr_01'].append( r["avgErr_01"] )

            count += 1
    except queue.Empty as ex:
        pass

    return mergedDict

def save_report(fn, report):
    """
    fn: the output filename.
    report: a dictonary contains the data.

    This function will use pandas package to output the report as a CSV file.
    """

    # Create the DataFrame object.
    df = pandas.DataFrame.from_dict(report, orient="columns")

    # Sort the rows according to the index column.
    df.sort_values(by=["idx"], ascending=True, inplace=True)

    # Save the file.
    df.to_csv(fn, index=False)

def handle_script_args(parser):
    parser.add_argument("input", type=str, \
        help = "The filename of the input JSON file.")

    parser.add_argument("--mf", type=float, default = -1.0, \
        help = "The iamge magnitude factor. If not specified, the value in the input JSON file will be used. Overwrite the value in the input JSON file is specifiec here.")
    
    parser.add_argument("--debug", action = "store_true", default = False, \
        help = "Debug information including 3D point clouds will be written addintionally.")
    
    parser.add_argument("--save-error-extra", action="store_true", default=False, \
        help="Save the extra error evaluation files.")

    parser.add_argument("--error-extra-dir-suffix", type=str, default="_error", \
        help="The folder name suffix which will be appended to the output folder name. The new folder name is used for saving extra error evaluations. This argument is only used when --save-error-extra is issued.")

    parser.add_argument("--np", type=int, default=1, \
        help="Number of threads.")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    # Script input arguments.
    parser = argparse.ArgumentParser(description='Compute the image flow data from sequence of camera poses and their depth information.')
    args   = handle_script_args(parser)

    # Read the JSON input file.
    inputParams = WD.read_input_parameters_from_json( args.input )

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

    idxArray = np.array(idxList, dtype=np.int)

    # Reshape the idxArray.
    idxArrayR = reshape_idx_array(idxArray)

    # Test the output directory.
    outDir = inputParams["dataDir"] + "/" + inputParams["outDir"]
    Utils.test_dir(outDir)

    # Check if we are saving the extra error evaluations.
    if ( args.save_error_extra ):
        errExDir = "%s%s" % ( outDir, args.error_extra_dir_suffix )
        Utils.test_dir(errExDir)
    else:
        errExDir = None

    startTime = time.time()

    print("Main: Main process.")

    jqueue  = multiprocessing.JoinableQueue() # The job queue.
    manager = multiprocessing.Manager()
    rqueue  = manager.Queue()         # The report queue.

    loggerQueue = multiprocessing.JoinableQueue()
    [conn1, loggerPipe] = multiprocessing.Pipe(False)

    loggerProcess = multiprocessing.Process( \
        target=logging_worker, args=["Logger", loggerQueue, conn1, inputParams["dataDir"]] )

    loggerProcess.start()

    processes   = []
    processStat = []
    pipes       = []

    loggerQueue.put("Main: Create %d processes." % (args.np))

    for i in range(args.np):
        [conn1, conn2] = multiprocessing.Pipe(False)
        processes.append( multiprocessing.Process( \
            target=worker, args=["P%03d" % (i), jqueue, rqueue, loggerQueue, conn1, \
                inputParams, args]) )
        pipes.append(conn2)

        processStat.append(1)

    for p in processes:
        p.start()

    loggerQueue.put("Main: All processes started.")
    loggerQueue.join()

    nIdx  = idxArray.size
    nIdxR = idxArrayR.size

    for i in range(nIdxR):
        # The index of cam_0.
        idx_0 = int(idxArrayR[i])

        if ( idx_0 == nIdx or idx_0 == nIdx - 1):
            continue

        idx_1 = idx_0 + 1

        # Get the poseIDs.
        poseID_0 = poseIDs[ idx_0 ]
        poseID_1 = poseIDs[ idx_1 ]

        # Get the poseDataLines.
        poseDataLine_0 = poseData[idx_0].reshape((-1, )).tolist()
        poseDataLine_1 = poseData[idx_1].reshape((-1, )).tolist()

        d = { "idx": idx_0, "poseID_0": poseID_0, "poseID_1": poseID_1, \
            "poseLineList_0": poseDataLine_0, "poseLineList_1": poseDataLine_1 }

        jqueue.put(d)
    
    loggerQueue.put("Main: All jobs submitted.")

    jqueue.join()

    loggerQueue.put("Main: Job queue joined.")
    loggerQueue.join()

    # Process the rqueue.
    report = process_report_queue(rqueue)

    # Save the report to file.
    reportFn = "%s/Report.csv" % (outDir)
    save_report(reportFn, report)
    loggerQueue.put("Report saved to %s. " % (reportFn))

    endTime = time.time()

    Utils.show_delimiter("Summary.")
    loggerQueue.put("%d poses, starting at idx = %d, step = %d, %d steps in total. idxNumberRequest = %d. Total time %ds. \n" % \
        (nPoses, inputParams["startingIdx"], idxStep, len( idxList )-1, idxNumberRequest, endTime-startTime))

    if ( args.mf >= 0 ):
        loggerQueue.put( "Command line argument --mf %f overwrites the parameter \"imageMagnitudeFactor\" (%f) in the input JSON file.\n" % (mf, inputParams["imageMagnitudeFactor"]) )

    # Stop all subprocesses.
    for p in pipes:
        p.send("exit")

    loggerQueue.put("Main: Exit command sent to all processes.")

    loggerQueue.join()

    nps = len(processStat)

    for i in range(nps):
        p = processes[i]

        if ( p.is_alive() ):
            p.join(timeout=1)

        if ( p.is_alive() ):
            loggerQueue.put("Main: %d subprocess (pid %d) join timeout. Try to terminate" % (i, p.pid))
            p.terminate()
        else:
            processStat[i] = 0
            loggerQueue.put("Main: %d subprocess (pid %d) joined." % (i, p.pid))

    if (not 1 in processStat ):
        loggerQueue.put("Main: All processes joined. ")
    else:
        loggerQueue.put("Main: Some process is forced to be terminated. ")

    # Stop the logger.
    loggerQueue.join()
    loggerPipe.send("exit")
    loggerProcess.join()

    print("Main: Done.")