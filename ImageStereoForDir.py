from __future__ import print_function

import argparse
import copy
import cv2
import glob
import json
import math
import multiprocessing
import numba
import numpy as np
import numpy.linalg as LA
import os
import pandas
import queue # python3.
import time
import sys

from CommonType import NP_FLOAT, NP_INT

import Utils

STEREO_OUT_OF_FOV = 11

STEREO_SELF_OCC   = 2
STEREO_CROSS_OCC  = 1

@numba.jit(nopython=True)
def find_stereo_occlusions_naive(depth_0, depth_1, disp, BF):
    h, w = depth_0.shape[0], depth_0.shape[1]

    # Masks.
    maskFOV = np.zeros_like(depth_0, dtype=np.uint8)
    maskOcc = np.zeros_like(depth_0, dtype=np.uint8)

    occupancyMap_00 = np.zeros((h, w, 2), dtype=NP_INT) - 1

    # debugX = 3072
    # debugY = 2531

    for i in range(h):
        for j0 in range(w):
            y  = int(i)
            x0 = int(j0)

            showDetails = False

            # Disparity.
            d = disp[y, x0]

            # The correspondence
            x1 = int( round( x0 - d ) )

            # if ( y == debugY and x0 == debugX ):
            #     showDetails = True

            if ( showDetails ):
                print("i = ", i, ", j0 = ", j0, ". ")
                print("d = ", d, ", x1 = ", x1, ". ")
                print("occupancyMap_00[y, x1][1] = ", occupancyMap_00[y, x1][1])

            if ( x1 < 0 ):
                # No corresponding pixel.
                maskFOV[y, x0] = STEREO_OUT_OF_FOV
                continue

            dep0  = depth_0[y, x0]
            dep1  = depth_1[y, x1]
            ddMax = BF/d**2

            if ( dep0 <= dep1 or dep0 - dep1 <= 2*ddMax):
                # The pixel from cam_0 is closer.
                if ( showDetails ):
                    print("Cross-occlusion: Current pixel closer.")
            else:
                # Occlusion.
                if ( maskOcc[ y, x0 ] != STEREO_OUT_OF_FOV ):
                    maskOcc[ y, x0 ] = STEREO_CROSS_OCC

                if ( showDetails ):
                    print("Cross-occlusion: Current pixel farther.")

            if ( showDetails ):
                print("dep0 = ", dep0, ", ", \
                      "dep1 = ", dep1, ", ", \
                      "dep0 - dep1 = ", dep0 - dep1, ", ", \
                      "ddMax = ", ddMax, ". ")

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

def calculate_stereo_disparity_naive(depth_0, depth_1, BF):
    """
    BF: The baseline * focal length of the stereo camera. Could be negative.

    This function calculate the disparity and occlusion in the naive way. We
    are assuming the camera poses are perfect and the depth images are rectified.
    So that we could constrain ourselves to find occlusions between scanlines
    with the same row index. And find correspodence only with the disparity
    values.
    """

    # Disparity.
    disp = BF / depth_0

    # Find occlusions.
    maskFOV, maskOcc, occupancyMap_00 = find_stereo_occlusions_naive(depth_0, depth_1, disp, BF)

    # Merge the masks
    mask = merge_masks(maskFOV, maskOcc)

    return disp, mask

def save_disparity_mask(fn0, disp, mask):
    parts = Utils.get_filename_parts(fn0)

    np.save( "%s/%s_disp.npy" % (parts[0], parts[1]), disp )
    np.save( "%s/%s_mask.npy" % (parts[0], parts[1]), mask )

def handle_script_args(parser):
    parser.add_argument("inputdir", type=str, \
        help="The directory name that contains the ISInput.json file. ")

    parser.add_argument("--left-depth-pattern", type=str, default="*depth_0.npy", \
        help="The search pattern for the left depth. ")

    parser.add_argument("--right-depth-pattern", type=str, default="*depth_1.npy", \
        help="The search pattern for the right depth. ")

    parser.add_argument("--bf", type=float, default=80.0, \
        help="The baseline * focal length of the stereo.")

    parser.add_argument("--report-dir", type=str, default="", \
        help="The directory for saving the report. Leave blank to disable saving. ")

    parser.add_argument("--max-num", type=int, default=0, \
        help="The maximum number of stereo pairs to process. Debug use. Set 0 to disable. ")

    parser.add_argument("--debug", action = "store_true", default = False, \
        help="Debug information including 3D point clouds will be written addintionally. ")
    
    parser.add_argument("--np", type=int, default=1, \
        help="Number of CPU threads.")

    parser.add_argument("--cuda", action="store_true", default=False, \
        help="Set this flow to use CUDA accelerated methods. ")

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

def single_process(job, bf):
    """
    job: A dictionary contains the following keys:
        idx, depth0, depth1.
    bf: baseline * focal length.
    """
    # Load the depth data.
    depth0 = np.load(job["depth0"]).astype(NP_FLOAT)
    depth1 = np.load(job["depth1"]).astype(NP_FLOAT)

    # Calculate the disparity.
    disp, mask = calculate_stereo_disparity_naive( depth0, depth1, bf )

    # Save the disparity and mask.
    save_disparity_mask( job["depth0"], disp, mask )

def worker(name, jq, rq, lq, p, args):
    """
    name: String, the name of this worker process.
    jq: A JoinableQueue. The job queue.
    rq: The report queue.
    lq: The logger queue.
    p: A pipe connection object. Only for receiving.
    """

    lq.put("%s: Worker starts." % (name))

    # ==================== Preparation. ========================

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

            single_process(job, args.bf)

            rq.put( { "idx": job["idx"] } )

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

    mergedDict = { "idx": [] }

    try:
        while (True):
            r = rq.get(block=False)

            mergedDict['idx'].append( r["idx"] )

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

def find_intput_files(args):
    # Find the depth maps.
    depthList0 = sorted( glob.glob( 
        "%s/**/%s" % ( args.inputdir, args.left_depth_pattern ),  recursive=True ) )
    depthList1 = sorted( glob.glob( 
        "%s/**/%s" % ( args.inputdir, args.right_depth_pattern ), recursive=True ) )

    # Check the numbers.
    nDepth0 = len(depthList0)
    nDepth1 = len(depthList1)

    if ( nDepth0 != nDepth1 ):
        raise Exception( \
"""Wrong numbers of files: nDepth0 = {}, nDepth1 = {}\n\
inputdir: %s\n
left_depth_pattern:  %s\n
right_depth_pattern: %s\n""".format(nDepth0, nDepth1, \
                    args.inputdir, 
                    args.left_depth_pattern, args.right_depth_pattern) )

    if ( 0 == nDepth0 ):
        raise Exception( \
"""No files found. \n\
inputdir: %s\n
left_depth_pattern:  %s\n
right_depth_pattern: %s\n""".format(args.inputdir, \
                    args.left_depth_pattern, args.right_depth_pattern) )

    # Compose the dictionary.
    return { "depthList0": depthList0, "depthList1": depthList1 }

def max_num(nFiles, maxNum):
    if ( maxNum > 0 and maxNum < nFiles ):
        return maxNum
    else:
        return nFiles

def main():
    # Script arguments.
    parser = argparse.ArgumentParser(description="Process the left-right stereo data.")
    args   = handle_script_args(parser)

    # Find the input files.
    filesDict = find_intput_files(args)
    nFiles = len( filesDict["depthList0"] )
    nFiles = max_num(nFiles, args.max_num)

    # # Test the output directory.
    # Utils.test_dir(outDir)

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
            target=worker, args=[ \
            "P%03d" % (i), jqueue, rqueue, loggerQueue, conn1, args ]) )
        pipes.append(conn2)

        processStat.append(1)

    for p in processes:
        p.start()

    loggerQueue.put("Main: All processes started.")
    loggerQueue.join()

    # Submit jobs.
    for i in range(nFiles):
        d = { "idx": i, \
            "depth0": filesDict["depthList0"][i], 
            "depth1": filesDict["depthList1"][i] }

        jqueue.put(d)

    loggerQueue.put("Main: All jobs submitted.")

    jqueue.join()

    loggerQueue.put("Main: Job queue joined.")
    loggerQueue.join()

    # Process the rqueue.
    report = process_report_queue(rqueue)

    # Save the report to file.
    if ( "" != args.report_dir ):
        reportFn = "%s/Report.csv" % (args.report_dir)
        save_report(reportFn, report)
        loggerQueue.put("Report saved to %s. " % (reportFn))
    else:
        reportFn = "%s/Report.csv" % ("/tmp")
        save_report(reportFn, report)

    endTime = time.time()

    Utils.show_delimiter("Summary.")
    loggerQueue.put("%d stereo pairs processed in %f seconds. \n" % (nFiles, endTime-startTime))

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

    return 0

if __name__ == "__main__":
    sys.exit(main())