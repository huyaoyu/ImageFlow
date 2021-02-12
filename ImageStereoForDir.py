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

LOCAL_NP_FLOAT=np.float32

import Utils

from CommonPython.Filesystem import Filesystem
from CommonPython.ImageIO.ImageRead import read_compressed_float
from CommonPython.ImageIO.ImageWrite import write_compressed_float

STEREO_OUT_OF_FOV = 11

STEREO_SELF_OCC   = 2
STEREO_CROSS_OCC  = 1

STEREO_NON_OCC    = 255

def read_depth(fn, dtype=NP_FLOAT):
    if ( not os.path.isfile(fn) ):
        raise Exception('%s does not exist' % (fn))

    parts = Filesystem.get_filename_parts(fn)

    if ( '.npy' == parts[2] ):
        return np.load(fn).astype(dtype)
    elif ( '.png' == parts[2] ):
        return read_compressed_float(fn).astype(dtype)
    else:
        raise Exception('Unexpected extention {}. '.format(parts[2]))

def warp_by_disparity(img, disp):
    '''
    img (NumPy array): OpenCV format image array. 0/1-channel or 3-channel.
    disp (NumPy array): Disparity array. 0/1-channel.

    img is the image from which to sample. The sampling coordinates will be
    computed by subracting disp from the coordinates of an ordinary image
    coordinates.
    '''

    # ========== Modify the dimensions if needed. ==========
    if ( disp.ndim == 3 ):
        if ( disp.shape[2] != 1 ):
            raise Exception('Wrong dimension of disp. disp.shape = {}'.format(disp.shape))
        else:
            disp = disp.reshape( ( disp.shape[0], disp.shape[1] ) )

    if ( img.ndim == 3 ):
        if ( img.shape[2] == 1 ):
            img = img.reshape( ( img.shape[0], img.shape[1] ) )

    # ========== Prepare the sampling coordiantes. ===========
    H, W = img.shape[:2]

    x = np.arange(0, W, dtype=np.float32)
    y = np.arange(0, H, dtype=np.float32)

    xx, yy = np.meshgrid(x, y)
    xx = xx - disp
    xx = xx.astype(np.float32)
    # xx = xx.reshape( ( xx.shape[0], xx.shape[1], 1 ) )
    # yy = yy.reshape( ( yy.shape[0], yy.shape[1], 1 ) )
    # print('xx.dtype = {}'.format(xx.dtype))
    # print('yy.dtype = {}'.format(yy.dtype))
    # print('xx.shape = {}'.format(xx.shape))
    # print('yy.shape = {}'.format(yy.shape))

    # ========== Sample the warped image. ==========
    warped = cv2.remap(img, xx, yy, interpolation=cv2.INTER_LINEAR)

    return warped

@numba.jit(nopython=True)
def find_stereo_occlusions_naive(depth_0, depth_1, disp, BF):
    h, w = depth_0.shape[0], depth_0.shape[1]

    # Masks.
    maskFOV = np.zeros_like(depth_0, dtype=np.uint8)
    maskOcc = np.zeros_like(depth_0, dtype=np.uint8)

    occupancyMap_00 = np.zeros((h, w, 2), dtype=NP_INT) - 1

    # debugX = 1659
    # debugY = 1162
    debugX = -1
    debugY = -1

    for i in range(h):
        for j0 in range(w):
            y  = int(i)
            x0 = int(j0)

            showDetails = False

            # Disparity.
            d = disp[y, x0]

            # The correspondence
            x1 = int( round( x0 - d ) )

            if ( y == debugY and x0 == debugX ):
                showDetails = True

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

            if ( dep0 >= 1000 and dep1 >= 1000):
                # Both points are at infinity.
                if ( showDetails ):
                    print("Both points are at infinity. ")
            elif ( dep0 - dep1 >= 1000 and dep1 <= 1000 ):
                # pixel from cam_0 is at infinity.
                # Occlusion.
                if ( maskOcc[ y, x0 ] != STEREO_OUT_OF_FOV ):
                    maskOcc[ y, x0 ] = STEREO_CROSS_OCC

                if ( showDetails ):
                    print("Infinity-occlusion: Current pixel farther.")
            elif ( dep0 <= dep1 or dep0 - dep1 <= 2*ddMax):
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
    mask = np.zeros_like(maskFOV, dtype=np.uint8) + STEREO_NON_OCC

    # The order of the application of the masks matters.
    tempMask = maskOcc == STEREO_CROSS_OCC
    mask[tempMask] = STEREO_CROSS_OCC

    # tempMask = maskOcc == STEREO_SELF_OCC
    # mask[tempMask] = STEREO_SELF_OCC

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

    
def convert_single_channel_float_array_2_image(array):
    assert( 2 == array.ndim )

    limit0 = array.min()
    limit1 = array.max()

    array = array - limit0
    array = array / ( limit1 - limit0 ) * 255

    return array.astype( np.uint8 )

def convert_mask_2_image(mask):
    m = mask == STEREO_NON_OCC
    mask[m] = 255
    mask[np.logical_not(m)] = 0
    return mask
    
# def save_disparity_mask_warped(fn0, disp, mask, warped):
#     parts = Utils.get_filename_parts(fn0)

#     np.save( "%s/%s_disp.npy" % (parts[0], parts[1]), disp )
#     cv2.imwrite( "%s/%s_mask.png" % (parts[0], parts[1]), mask, [cv2.IMWRITE_PNG_COMPRESSION, 5] )

def save_disparity_mask_warped(fn0, disp, mask, warped):
    parts = Utils.get_filename_parts(fn0)

    # np.save( "%s/%s_disp.npy" % (parts[0], parts[1]), disp )
    write_compressed_float( 
        "%s/%s_disp_lu1.png" % (parts[0], parts[1]), disp )
    # np.save( "%s/%s_mask.npy" % (parts[0], parts[1]), mask )
    
    cv2.imwrite( "%s/%s_disp.png" % (parts[0], parts[1]), 
        convert_single_channel_float_array_2_image(disp) )

    cv2.imwrite( "%s/%s_mask.png" % (parts[0], parts[1]), 
        convert_mask_2_image(mask) )

    cv2.imwrite( "%s/%s_warped.png" % ( parts[0], parts[1] ), 
        warped, [cv2.IMWRITE_PNG_COMPRESSION, 5] )

def handle_script_args(parser):
    parser.add_argument("inputdir", type=str, \
        help="The directory name that contains the ISInput.json file. ")

    parser.add_argument("--depth-pattern-0", type=str, default="*depth_0.npy", \
        help="The search pattern for the left depth. ")

    parser.add_argument("--depth-pattern-1", type=str, default="*depth_1.npy", \
        help="The search pattern for the right depth. ")

    parser.add_argument("--image-pattern-0", type=str, default="*rgb_0.png", \
        help="The search pattern for the reference image. ")
    
    parser.add_argument("--image-pattern-1", type=str, default="*rgb_1.png", \
        help="The search pattern for the source/test image. ")

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
    depth0 = read_depth(job["depth0"], LOCAL_NP_FLOAT)
    depth1 = read_depth(job["depth1"], LOCAL_NP_FLOAT)

    # Calculate the disparity.
    disp, mask = calculate_stereo_disparity_naive( depth0, depth1, bf )

    # Load the source/test image.
    imgSrc = cv2.imread( job["image1"], cv2.IMREAD_UNCHANGED )

    # Warp the source/test image.
    warped = warp_by_disparity(imgSrc, disp)

    # Save the disparity and mask.
    save_disparity_mask_warped( job["depth0"], disp, mask, warped )

    # # Debut use.
    # parts = Utils.get_filename_parts(job["depth0"])
    # img1OutFn = os.path.join( parts[0], "%s_image1.png" % (parts[1]) )
    # cv2.imwrite(img1OutFn, imgSrc)

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
        "%s/**/%s" % ( args.inputdir, args.depth_pattern_0 ), recursive=True ) )
    depthList1 = sorted( glob.glob( 
        "%s/**/%s" % ( args.inputdir, args.depth_pattern_1 ), recursive=True ) )

    imageList0 = sorted( glob.glob(
        "%s/**/%s" % ( args.inputdir, args.image_pattern_0 ), recursive=True ) )
    imageList1 = sorted( glob.glob(
        "%s/**/%s" % ( args.inputdir, args.image_pattern_1 ), recursive=True ) )

    # Check the numbers.
    nDepth0 = len(depthList0)
    nDepth1 = len(depthList1)
    nImage0 = len(imageList0)
    nImage1 = len(imageList1)

    if ( nDepth0 != nDepth1 or nImage0 != nImage1 or nDepth0 != nImage0 ):
        raise Exception( \
"""Wrong numbers of files: nDepth0 = {}, nDepth1 = {}, nImage0 = {}, nImage1 = {}\
inputdir: {}
depth_pattern_0: {}
depth_pattern_1: {}
image_pattern_0: {}
image_pattern_1: {}
""".format(nDepth0, nDepth1, nImage0, nImage1, \
            args.inputdir, \
            args.depth_pattern_0, args.depth_pattern_1, \
            args.image_pattern_0, args.image_pattern_1 ) )

    if ( 0 == nDepth0 ):
        raise Exception( \
"""No files found.\
inputdir: {}
depth_pattern_0: {}
depth_pattern_1: {}
image_pattern_0: {}
image_pattern_1: {}
""".format(args.inputdir, \
        args.depth_pattern_0, args.depth_pattern_1, \
        arsg.image_pattern_0, args.image_pattern_1 ) )

    # Compose the dictionary.
    return { "depthList0": depthList0, "depthList1": depthList1, \
             "imageList0": imageList0, "imageList1": imageList1 }

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
            "depth1": filesDict["depthList1"][i],
            "image0": filesDict["imageList0"][i], 
            "image1": filesDict["imageList1"][i] }

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