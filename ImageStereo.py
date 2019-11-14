from __future__ import print_function

import argparse
import copy
import cv2
import json
import math
import multiprocessing
import numpy as np
import numpy.linalg as LA
import os
import pandas
import queue # python3.
import time

from CommonType import NP_FLOAT

from Camera import CameraBase
from SimplePLY import output_to_ply 
import Utils
import WorkDirectory as WD

STEREO_OUT_OF_FOV = 11

STEREO_SELF_OCC   = 2
STEREO_CROSS_OCC  = 1

def load_pose_data(fn):
    if ( ".txt" == os.path.splitext( os.path.split(fn)[1] )[1] ):
        poseData = np.loadtxt( fn, dtype=NP_FLOAT )
    else:
        poseData = np.load( fn ).astype(NP_FLOAT)

    return poseData

def load_pose_id_pose_data(params, args):
    dataDir = args.inputdir

    poseIDsFn = dataDir + "/" + params["poseFilename"]

    if ( not os.path.isfile(poseIDsFn) ):
        # File not exist. Create on the fly.
        WD.create_pose_id_file( dataDir, params["imageDir"], "*%s" % (params["imageExt"]), params["poseFilename"] )

    _, poseIDs = WD.load_IDs_JSON(\
        poseIDsFn, params["poseName"])

    poseDataFn_0 = dataDir + "/" + params["poseData"][0]
    poseDataFn_1 = dataDir + "/" + params["poseData"][1]

    poseData_0 = load_pose_data(poseDataFn_0)
    poseData_1 = load_pose_data(poseDataFn_1)

    return poseIDs, [ poseData_0, poseData_1 ]

def calculate_stereo_extrinsics_from_pose_lines( line_0, line_1 ):
    """
    line_0 and line_1 are NumPy arrays with 7 elements.
    The first 3 elements are the position of the camere measured in the world coordinate.
    The remaining 4 elemenets are the orientation of the camera measured in the world coordinate in terms of quaternions.

    This funcion returen the stereo extrinsics between the two cameras, namely the rotation R and translation T. R and T are
    defined in such a way that a point in the camera_0 frame, x0, could be transformed into x1 in camera_1 frame by

    R * x0 + T = x1

    """

    # Get the R and T for the cameras with respect to the world coordinate.
    R0, t0, q0 = WD.get_pose_from_line(line_0)
    R1, t1, q1 = WD.get_pose_from_line(line_1)

    # Make the stereo extrinsics.
    R = R1.dot( R0.transpose() )
    T = -R.dot( t0 ) + t1

    return R, T

def get_depth_with_neighbors(depth, i, j):
    h, w = depth.shape[0], depth.shape[1]

    dep = []

    idxH = np.clip( np.array([i, i-1, i-1, i+1, i+1], dtype=np.int), 0, h-1 )
    idxW = np.clip( np.array([j, j-1, j+1, j-1, j+1], dtype=np.int), 0, w-1 )

    for i in range(5):
        dep.append( depth[ idxH[i], idxW[i] ] )

    d = np.array( dep, dtype=NP_FLOAT )

    return d[0], d.min(), d.max()

def find_stereo_occlusions_naive(depth_0, depth_1, disp, BF):
    h, w = depth_0.shape[0], depth_0.shape[1]

    # Masks.
    maskFOV = np.zeros_like(depth_0, dtype=np.uint8)
    maskOcc = np.zeros_like(depth_0, dtype=np.uint8)

    occupancyMap_00 = np.zeros((h, w, 2), dtype=np.int) - 1

    for i in range(h):
        for j0 in range(w):
            y  = int(i)
            x0 = int(j0)

            showDetails = False
            # if ( y == 128 and x0 == 307 ):
            #     showDetails = True
            #     print("i = %d, j0 = %d. " % ( i, j0 ))

            # Disparity.
            d = disp[y, x0]

            # The correspondence
            x1 = int( round( x0 - d ) )

            if ( showDetails ):
                print("d = {}, x1 = {}. ".format(d, x1))

            if ( x1 < 0 ):
                # No corresponding pixel.
                maskFOV[y, x0] = STEREO_OUT_OF_FOV
                continue

            dep0 = depth_0[y, x0]
            depR = 0

            # Have a valid correspondence.
            if ( -1 != occupancyMap_00[y, x1][0] ):
                # A pixel already takes this place.

                # Get the index of the regeistered pixel.
                coorR = occupancyMap_00[y, x1]

                depR = depth_0[coorR[0], coorR[1]] # Previously registered pixel depth.

                if ( dep0 < depR ):
                    # The current pixel is closer.
                    occupancyMap_00[y, x1] = np.array([y, x0])
                    
                    if ( maskOcc[ coorR[0], coorR[1] ] == -1 ):
                        maskOcc[ coorR[0], coorR[1] ] = STEREO_SELF_OCC
                else:
                    # The current pixel is farther.
                    if ( maskOcc[ y, x0 ] == -1 ):
                        maskOcc[ y, x0 ] = STEREO_SELF_OCC
            else:
                # No pixel is previously registered here.
                occupancyMap_00[y, x1] = np.array([y, x0])

            # Cross-occlusion check.
            # dep1, ddMin, ddMax = get_depth_with_neighbors(depth_1, y, x1)
            dep1 = depth_1[y, x1]
            ddMax = BF/d**2

            if ( dep0 <= dep1 or dep0 - dep1 <= ddMax):
                # The pixel from cam_0 is closer.
                pass
            else:
                # Occlusion.
                maskOcc[ y, x0 ] = STEREO_CROSS_OCC

            if ( showDetails ):
                print("dep0 = {}, depR = {}, dep1 = {}, ddMax = {}. ".format(dep0, depR, dep1, ddMax))

    return maskFOV, maskOcc, occupancyMap_00

def merge_masks(maskFOV, maskOcc):
    mask = np.zeros_like(maskFOV, dtype=np.uint8) + 255

    # The order of the application of the masks matters.
    tempMask = maskOcc == STEREO_CROSS_OCC
    mask[tempMask] = STEREO_CROSS_OCC

    tempMask = maskOcc == STEREO_SELF_OCC
    mask[tempMask] = STEREO_SELF_OCC

    tempMask = maskFOV == STEREO_OUT_OF_FOV
    mask[tempMask] = STEREO_OUT_OF_FOV

    return mask

def calculate_stereo_disparity_naive(cam, \
    depth_0, depth_1, B):
    """
    B: The baseline of the stereo camera. Could be negative.

    This function calculate the disparity and occlusion in the naive way. We
    are assuming the camera poses are perfect and the depth images are rectified.
    So that we could constrain ourselves to find occlusions between scanlines
    with the same row index. And find correspodence only with the disparity
    values.
    """

    # Baseline * focal length.
    BF = np.absolute( B * cam.focal )

    # Disparity.
    disp = BF / depth_0

    # Find occlusions.
    maskFOV, maskOcc, occupancyMap_00 = find_stereo_occlusions_naive(depth_0, depth_1, disp, BF)

    # Merge the masks
    mask = merge_masks(maskFOV, maskOcc)

    return disp, mask

def save_disparity_mask(outDir, poseID, disp, mask):
    np.save( "%s/%s_disp.npy" % (outDir, poseID), disp )
    np.save( "%s/%s_mask.npy" % (outDir, poseID), mask )

def handle_script_args(parser):
    parser.add_argument("inputdir", type=str, \
        help = "The directory name that contains the ISInput.json file.")
   
    parser.add_argument("--debug", action = "store_true", default = False, \
        help = "Debug information including 3D point clouds will be written addintionally.")
    
    parser.add_argument("--np", type=int, default=1, \
        help="Number of CPU threads.")

    parser.add_argument("--cuda", action="store_true", default=False, \
        help="Set this flow to use CUDA accelerated methods. Could not used with --np with thread number larger than 1.")

    args = parser.parse_args()

    return args

def logging_worker(name, jq, p, workingDir):
    import logging

    logger = logging.getLogger("ImageStereo")
    logger.setLevel(logging.INFO)

    logFn = "%s/LogStereo.log" % (workingDir)
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
    dataDir = args.inputdir

    # Camera.
    cam_0 = CameraBase(inputParams["camera"]["focal"], inputParams["camera"]["imageSize"])

    # We are assuming that the cameras at the two poses are the same camera.
    cam_1 = cam_0

    outDir        = dataDir + "/" + inputParams["outDir"]
    depthDir      = [ "%s/%s" % ( dataDir, inputParams["depthDir"][0] ), \
                      "%s/%s" % ( dataDir, inputParams["depthDir"][1] ) ]
    imgDir        = [ "%s/%s" % ( dataDir, inputParams["imageDir"][0] ), \
                      "%s/%s" % ( dataDir, inputParams["imageDir"][1] ) ]
    imgSuffix     = inputParams["imageSuffix"]
    imgExt        = inputParams["imageExt"]
    depthTail     = [ "%s%s" % ( inputParams["depthSuffix"][0], inputParams["depthExt"] ), \
                      "%s%s" % ( inputParams["depthSuffix"][1], inputParams["depthExt"] ) ]
    distanceRange = inputParams["distanceRange"]

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

            poseID = job["poseID"]
            poseDataLineList_0 = job["poseLineList_0"]
            poseDataLineList_1 = job["poseLineList_1"]

            poseDataLine_0 = np.array( poseDataLineList_0, dtype=NP_FLOAT )
            poseDataLine_1 = np.array( poseDataLineList_1, dtype=NP_FLOAT )

            # Load the depth.
            depth_0 = np.load( depthDir[0] + "/" + poseID + depthTail[0] ).astype(NP_FLOAT)
            depth_1 = np.load( depthDir[1] + "/" + poseID + depthTail[1] ).astype(NP_FLOAT)

            # If it is debugging.
            if ( args.debug ):
                debugOutDir = "%s/ImageStereo/%s" % ( dataDir, poseID_0 )
                Utils.test_dir(debugOutDir)
            else:
                debugOutDir = "./"

            # Process.

            # Find the extrinsics of Cam_1 repsect to Cam_0.
            R, T = calculate_stereo_extrinsics_from_pose_lines( poseDataLine_0, poseDataLine_1 )
            lq.put("{}: idx = {}, T = {}. ".format(name, job["idx"], T.reshape((-1,)) ))

            # Calculate the disparity.
            # T[1] is the actual baseline.
            disp, mask = calculate_stereo_disparity_naive( cam_0, depth_0, depth_1, T[1] )

            # Save the disparity and mask.
            save_disparity_mask( outDir, poseID, disp, mask )

            rq.put( { "idx": job["idx"], \
                "poseID": poseID, 
                "T": np.array2string(T.reshape((-1,))) } )

            count += 1

            jq.task_done()
        except queue.Empty as exp:
            pass
    
    lq.put("%s: Done with %d jobs." % (name, count))

def process_report_queue(rq):
    """
    rq: A Queue object containing the report data.
    """

    count = 0

    mergedDict = { "idx": [], "poseID": [], "T": [] }

    try:
        while (True):
            r = rq.get(block=False)

            mergedDict['idx'].append( r["idx"] )
            mergedDict['poseID'].append( r["poseID"] )
            mergedDict['T'].append( r["T"] )

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

if __name__ == "__main__":
    # Script arguments.
    parser = argparse.ArgumentParser(description="Process the left-right stereo data.")
    args   = handle_script_args(parser)

    # Test the input file.
    inputJsonFn = "%s/ISInput.json" %( args.inputdir )
    if ( not os.path.isfile( inputJsonFn ) ):
        raise Exception( "Input file %s does not exist. " % ( inputJsonFn ))

    # Read the JSON input file.
    inputParams = WD.read_input_parameters_from_json( inputJsonFn )

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

    # Test the output directory.
    outDir = "%s/%s" % ( args.inputdir, inputParams["outDir"] )
    Utils.test_dir(outDir)

    # Start processing.
    startTime = time.time()

    print("Main: Main process.")

    jqueue  = multiprocessing.JoinableQueue() # The job queue.
    manager = multiprocessing.Manager()
    rqueue  = manager.Queue()         # The report queue.

    loggerQueue = multiprocessing.JoinableQueue()
    [conn1, loggerPipe] = multiprocessing.Pipe(False)

    loggerProcess = multiprocessing.Process( \
        target=logging_worker, args=["Logger", loggerQueue, conn1, args.inputdir] )

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

    # Submit jobs.
    for i in range(idxArray.size):
        idx = int(idxArray[i])

        # Get the poseIDs.
        poseID = poseIDs[ idx ]

        # Get the poseDataLines.
        poseDataLine_0 = poseData[0][idx].reshape((-1, )).tolist()
        poseDataLine_1 = poseData[1][idx].reshape((-1, )).tolist()

        d = { "idx": idx, "poseID": poseID, \
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

    # Stop all subprocesses.
    for p in pipes:
        p.send("exit")

    loggerQueue.put("Main: Exit command sent to all processes.")

    loggerQueue.join()

    nps = len(processStat)

    for i in range(nps):
        p = processes[i]

        if ( p.is_alive() ):
            p.join(timeout=2)

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