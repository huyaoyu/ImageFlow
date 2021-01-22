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
from queue import Empty
import time
import sys

from CommonType import NP_FLOAT, NP_INT
from IO import readPFM

import Utils

STEREO_OUT_OF_FOV = 11

STEREO_SELF_OCC     = 2
STEREO_CROSS_OCC    = 1
STEREO_FILTERED_OCC = 10
STEREO_NON_MASK     = 255

class DirectoryCreator(object):
    def __init__(self, baseDir=None, baseSuffix=None):
        super(DirectoryCreator, self).__init__()

        self.baseDir = baseDir
        self.baseSuffix = baseSuffix

        if ( self.baseDir is not None ):
            self.prefix = self.baseDir
            if ( self.baseSuffix is not None ):
                self.prefix = os.path.join( self.prefix, self.baseSuffix )
        else:
            self.prefix = ''

        self.previous = None

    def create_by_filename(self, fn):
        parts = Utils.get_filename_parts(fn)

        if ( parts[0] == self.previous ):
            return
        
        d = os.path.join( self.prefix, parts[0] )
        if ( not os.path.isdir(d) ):
            os.makedirs(d)
        
        self.previous = parts[0]

def read_string_list(fn, prefix=""):
    """
    fn (string): The filename.
    prefix (string): Prefix to be added to every string in the string list.

    Read a file contains lines of strings. A list will be returned.
    Each element of the list contains a single entry of the input file.
    Leading and tailing white spaces, tailing carriage return will be stripped.
    """

    if ( False == os.path.isfile( fn ) ):
        raise Exception("%s does not exist." % (fn))
    
    with open(fn, "r") as fp:
        lines = fp.read().splitlines()

        n = len(lines)

        if ( "" == prefix ):
            for i in range(n):
                lines[i] = lines[i].strip()
        else:
            for i in range(n):
                lines[i] = "%s/%s" % ( prefix, lines[i].strip() )

    return lines

def extract_strings(line, expected, delimiter):
    '''
    line (string): A stirng contains sub-strings separated by delimiter.
    expected (int): The expected number of sub-strings to be extracted.
    delimiter (string): The delimiter.
    '''

    ss = line.split(delimiter)

    n = len(ss)

    assert (n == expected ), "{} strings extracted from {} with delimiter {}. Expected to be {}. ".format(n, line, delimiter, expected)

    result = []
    for s in ss:
        result.append( s.strip() )

    return result

def read_string_list_2D(fn, expCols, delimiter=",", prefix=""):
    """
    fn (string): The filename.
    expCols (int): The expected columns of each line. 
    delimiter (string): The separator between strings.
    prefix (string): If prefix is not empty, then the prefix string will be added to the front of each sub-string.
    """
    
    assert (int(expCols) > 0), "expCols = {}. ".format(expCols)
    expCols = int(expCols)

    if ( False == os.path.isfile( fn ) ):
        raise Exception("%s does not exist." % (fn))
    
    strings2D = []
    n = 0

    with open(fn, "r") as fp:
        lines = fp.read().splitlines()

        n = len(lines)

        if ( "" == prefix ):
            for i in range(n):
                line = extract_strings(lines[i].strip(), expCols, delimiter)
                strings2D.append( line )
        else:
            for i in range(n):
                line = extract_strings(lines[i].strip(), expCols, delimiter)

                for j in range(expCols):
                    col = line[j]
                    line[j] = "%s/%s" % ( prefix, col ) if col != 'None' else col

                strings2D.append( line )

    if ( n == 0 ):
        raise Exception("Read {} failed. ".format(fn))

    stringCols = []
    for i in range(expCols):
        col = []
        for j in range(n):
            col.append( strings2D[j][i] )

        stringCols.append(col)

    return stringCols

@numba.jit(nopython=True)
def find_stereo_occlusions_naive(depth_0, depth_1, disp, BF):
    h, w = depth_0.shape[0], depth_0.shape[1]

    # Masks.
    maskFOV = np.zeros_like(depth_0, dtype=np.uint8)
    maskOcc = np.zeros_like(depth_0, dtype=np.uint8)

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

            if ( x1 < 0 ):
                # No corresponding pixel.
                maskFOV[y, x0] = STEREO_OUT_OF_FOV
                continue

            dep0  = depth_0[y, x0]
            dep1  = depth_1[y, x1]
            ddMax = BF/d**2

            if ( dep0 <= dep1 or dep0 - dep1 <= 2*ddMax):
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

    return maskFOV, maskOcc

@numba.jit(nopython=True)
def find_stereo_occlusion_by_disp_naive( disp0, disp1=None ):
    h, w = disp0.shape[:2]

    # Masks.
    maskFOV = np.zeros_like(disp0, dtype=np.uint8)
    maskOcc = np.zeros_like(disp0, dtype=np.uint8)

    # Occupancy maps.
    occupancyMap_01 = np.zeros((h, w), dtype=np.int32) - 1
    occupancyMap_d  = np.zeros_like(disp0)

    # debugX = 258
    # debugY = 414

    showDetails = False

    for i in range(h):
        for j0 in range(w):
            y  = int(i)
            x0 = int(j0)

            # Disparity.
            d = disp0[y, x0]

            # The correspondence
            x1 = int( round( x0 - d ) )

            # showDetails = True if y == debugY and x0 == debugX else False

            if ( showDetails ):
                print("i = ", i, ", j0 = ", j0, ". ")
                print("d = ", d, ", x1 = ", x1, ". ")

            # FOV check.
            if ( x1 < 0 ):
                maskFOV[y, x0] = STEREO_OUT_OF_FOV
                continue

            # Occupancy check.
            if ( -1 == occupancyMap_01[y, x1] ):
                # Free.
                occupancyMap_01[y, x1] = x0
                occupancyMap_d[y, x1]  = d

                if ( showDetails ):
                    print('Free')
            else:
                # ========== The target location in image 1 is occupied. ==========

                d1c = occupancyMap_d[y, x1]  # The current disparity saved at the location [y, x1] of the occupancy map.
                x0c = occupancyMap_01[y, x1] # The current index in image 0 saved at the location [y, x1] of the occupancy map.

                if ( showDetails ):
                    print( 'd1c = ', d1c, 'x0c = ', x0c )

                if ( d - d1c > 0 ):
                    # Pixel x0 is closer.
                    if ( d - d1c > 1 ):
                        # Pixel x0 occludes x0c.
                        if ( maskOcc[ y, x0c ] != STEREO_OUT_OF_FOV ):
                            maskOcc[ y, x0c ] = STEREO_SELF_OCC
                    
                    occupancyMap_01[ y, x1 ] = x0
                    occupancyMap_d[ y, x1 ]  = d
                else:
                    # Pixel x0c is closer.
                    if ( d - d1c < -1 ):
                        if ( maskOcc[ y, x0 ] != STEREO_OUT_OF_FOV ):
                            maskOcc[ y, x0 ] = STEREO_SELF_OCC

            if ( disp1 is None ):
                continue

            d1 = np.abs( disp1[y, x1] )
            if ( showDetails ):
                print('d1 = ', d1)
                print('d1 - d = ', d1-d)

            if ( d1 - d > 1.5 ):
                # The target pixel in image1 is closer.
                if ( maskOcc[ y, x0 ] != STEREO_OUT_OF_FOV ):
                    maskOcc[ y, x0 ] = STEREO_CROSS_OCC

    return maskFOV, maskOcc, occupancyMap_01, occupancyMap_d

def merge_masks(maskFOV, maskOcc):
    mask = np.zeros_like(maskFOV, dtype=np.uint8) + STEREO_NON_MASK

    # The order of the application of the masks matters.
    tempMask = maskOcc == STEREO_CROSS_OCC
    mask[tempMask] = STEREO_CROSS_OCC

    tempMask = maskOcc == STEREO_SELF_OCC
    mask[tempMask] = STEREO_SELF_OCC

    tempMask = maskFOV == STEREO_OUT_OF_FOV
    mask[tempMask] = STEREO_OUT_OF_FOV

    return mask

@numba.jit(nopython=True)
def filter_mask(mask, disp, occupancyMap_d, nonMaskValue, newMaskValue, segmentLimit=3):
    '''
    mask (NumPy array): The mask array, 2D, valid (non-occlusion) pixels are nonMaskValue.
    disp (NumPy array): The disparity array, 2D.
    occupancyMap_d (NumPy array): The occupancy map storing the disparities of the occupied pixels.
    nonMaskValue (int): The value for non-mask pixels.
    newMaskValue (int): The value to be inserted into mask if new masked pixels are found. 
    segmentLimit (int): The maximum number of consecutive pixels to be considered as a segment.
    
    This function checks the validity of the mask and filter invalid masks. For
    a segment of un-masked pixels along the x-axis, if the two ending pixels are masked and the
    average interior disparty is smaller than the disparities retrieved by referring to occupancyMap_d, 
    then the interior of this segment are marked as masked.

    Segment contains, at maximum, segmentLimit number of pixels.

    It is assumed that disp has been checked and no infinite values are present.
    '''

    H, W = mask.shape[:2]

    for y in range(H):
        seg0 = -1
        seg1 = -1
        segCount = 0

        for x in range(1, W):
            # Accumulate segment.
            if ( segCount > 0 ):
                # Accumulating.
                if ( mask[y, x] == nonMaskValue ):
                    # Accumulate.
                    segCount += 1

                    if ( segCount > segmentLimit ):
                        # Segment too long.
                        seg0 = -1
                        segCount =0
                
                    continue
                
                else:
                    # End of the segment.
                    seg1 = x
            else:
                if ( mask[y, x] == nonMaskValue ):
                    if ( mask[y, x-1] != nonMaskValue ):
                        seg0 = x
                        segCount = 1
                    
                continue

            # Handle segment.
            if ( seg1 != -1 ):
                disp0 = disp[y, seg0-1] # Disparity at leading masked pixel.
                disp1 = disp[y, seg1]   # Disparity at ending masked pixel.
                x1_0 = int( round( seg0 - 1 - disp0 ) ) # x-coordinate of the leading masked pixel in occupancyMap_d.
                x1_1 = int( round( seg1     - disp1 ) ) # x-coordinate of the ending masked pixel in occupancyMap_d.

                avgDisp1 = ( occupancyMap_d[y, x1_0] + occupancyMap_d[y, x1_1] ) / 2
                avgDisp0 = ( disp0 + disp1 ) / 2
                
                if ( avgDisp0 < avgDisp1 ):
                    mask[y, seg0:seg1] = newMaskValue

                seg0 = -1
                seg1 = -1
                segCount = 0


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
    maskFOV, maskOcc = find_stereo_occlusions_naive(depth_0, depth_1, disp, BF)

    # Merge the masks
    mask = merge_masks(maskFOV, maskOcc)

    # filter_mask(mask, disp, STEREO_NON_MASK, STEREO_FILTERED_OCC)

    return mask

def calculate_stereo_mask_only_naive(disp0, disp1=None):

    maskFOV, maskOcc, occupancyMap_01, occupancyMap_d = \
        find_stereo_occlusion_by_disp_naive( disp0, disp1 )

    mask = merge_masks( maskFOV, maskOcc )

    filter_mask(mask, disp0, occupancyMap_d, STEREO_NON_MASK, STEREO_FILTERED_OCC)

    return mask

def compose_mask_filename(root, subDir, fn0):
    parts = Utils.get_filename_parts(fn0)

    outDir = os.path.join( root, subDir, parts[0] )
    # Utils.test_dir(outDir)

    return os.path.join( outDir, '%s.png' % (parts[1]) )

def save_mask(outFn, mask):
    cv2.imwrite( outFn, mask )

def write_masked_image(root, subDir, img0, mask):
    inFn = os.path.join( root, img0 )

    if ( not os.path.isfile(inFn) ):
        raise Exception('%s does not exist. ' % (inFn))

    img = cv2.imread(inFn, cv2.IMREAD_UNCHANGED)
    
    mask = mask != STEREO_NON_MASK

    if ( 3 == img.ndim and 3 == img.shape[2]):
        mask = np.stack( ( mask, mask, mask ), axis=2 )

    img[mask] = 0
    
    parts = Utils.get_filename_parts(img0)

    outDir = os.path.join( root, subDir, parts[0] )
    # Utils.test_dir(outDir)

    outFn = os.path.join( outDir, '%s.png' % (parts[1]) )
    cv2.imwrite(outFn, img)

def handle_script_args(parser):
    parser.add_argument("inputjson", type=str, \
        help="The input JSON file describing the dataset. ")

    parser.add_argument("jsonentry", type=str, \
        help="The JSON entry to read as the input filelist. ")

    parser.add_argument("occdir", type=str, \
        help="The output subdirectory. ")

    parser.add_argument("--max-num", type=int, default=0, \
        help="The maximum number of stereo pairs to process. Debug use. Set 0 to disable. ")

    parser.add_argument("--write-images", action = "store_true", default = False, \
        help="Set this flag to enable writing masked images. ")

    parser.add_argument("--no-overwrite", action="store_true", default = False, \
        help='Set this flag to skip a job if the target occlusion mask already exists. ')
    
    parser.add_argument("--np", type=int, default=1, \
        help="Number of CPU threads.")

    parser.add_argument("--cuda", action="store_true", default=False, \
        help="Set this flow to use CUDA accelerated methods. ")

    args = parser.parse_args()

    return args

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

def worker_rq(name, rq, lq, nFiles, resFn, timeoutCountLimit=100):
    resultDict = { 'idx':[], 'name': [], 'disp0Fn': [], 'depth0Fn':[], 'countMasked': [] }
    resultCount = 0
    timeoutCount = 0

    flagOK = True

    while(resultCount < nFiles):
        try:
            r = rq.get(block=True, timeout=1)
            resultDict['idx'].append(r['idx'])
            resultDict['name'].append(r['name'])
            resultDict['disp0Fn'].append(r['disp0Fn'])
            resultDict['depth0Fn'].append(r['depth0Fn'])
            resultDict['countMasked'].append(r['countMasked'])
            resultCount += 1

            timeoutCount = 0

            if (resultCount % 100 == 0):
                lq.put("%s: worker_rq collected %d results. " % (name, resultCount))
        except Empty as exp:
            lq.put("%s: Wait on rq-index %d. " % (name, resultCount))
            time.sleep(0.5)
            timeoutCount += 1

            if ( timeoutCount == timeoutCountLimit ):
                lq.put("%s: worker_rq reaches the timeout count limit (%d). Process abort. " % \
                    (name, timeoutCountLimit))
                flagOK = False
                break

    if (flagOK):
        lq.put("%s: All results processed. " % (name))

    if ( resultCount > 0 ):
        if ( resultCount != nFiles ):
            lq.put("%s: resultCount = %d, nFiles = %d. " % (name, resultCount, nFiles))

        save_report( resFn, resultDict)

def logging_worker(name, jq, p, workingDir, fn):
    import logging

    logger = logging.getLogger("ImageStereo")
    logger.setLevel(logging.INFO)

    logFn = os.path.join( workingDir, fn )
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

def load_depth(fn):
    assert( os.path.isfile(fn) ), '{} does not exist. '.format(fn)

    parts = Utils.get_filename_parts(fn)
    if ( parts[2] == '.npy' ):
        return np.load(fn).astype(NP_FLOAT)
    elif ( parts[2] == '.png' ):
        depth = cv2.imread(fn, cv2.IMREAD_UNCHANGED).view("<f4")
        return np.squeeze(depth, axis=-1).astype(NP_FLOAT)
    else:
        raise Exception('Not supported depth format: {}. '.format(parts[2]))

def single_process_depth(job):
    """
    job: A dictionary contains the following keys:
        idx, depth0, depth1.
    """

    # The target output filename.
    outFn = compose_mask_filename( job["datasetRoot"], job['occDir'], job["depth0"] )
    if ( job["flagNoOverwrite"] and os.path.isfile(outFn) ):
        return -1

    # Load the depth data.
    try:
        depth0 = load_depth( 
            os.path.join( job["datasetRoot"], job["depth0"] ) )
        depth1 = load_depth( 
            os.path.join( job["datasetRoot"], job["depth1"] ) )
        bf     = job['bf']
    except Exception as exp:
        return '%s read error. ' % ( os.path.join( job["datasetRoot"], job["depth0"] ) )

    # Calculate the disparity.
    mask = calculate_stereo_disparity_naive( depth0, depth1, bf )

    # Save the disparity and mask.
    save_mask( outFn, mask )

    if ( job['flagWriteImage'] ):
        write_masked_image(job['datasetRoot'], job['occDir'], job['img0Fn'], mask)

    countMask = mask != STEREO_NON_MASK
    return countMask.sum()

def read_disp(fn):
    parts = Utils.get_filename_parts(fn)

    if ( '.npy' == parts[2] ):
        disp = np.load( fn ).astype(NP_FLOAT)
    elif ( '.pfm' == parts[2] ):
        disp, _ = readPFM( fn )
        disp = disp.astype(NP_FLOAT)
    else:
        raise Exception('Unexpected disparity format %s from %s. ' % (parts[2], fn))

    return disp

def single_process_disp(job):
    '''
    job (dict): Contains the job description. 
    '''

    outFn = compose_mask_filename( job['datasetRoot'], job['occDir'], job['disp0Fn'] )
    if ( job["flagNoOverwrite"] and os.path.isfile(outFn) ):
        return -1

    disp0 = read_disp( os.path.join(job['datasetRoot'], job['disp0Fn']) )
    disp1 = read_disp( os.path.join(job['datasetRoot'], job['disp1Fn']) ) \
        if job['disp1Fn'] != 'None' else None

    mask = calculate_stereo_mask_only_naive( disp0, disp1 )

    save_mask( outFn, mask )

    if ( job['flagWriteImage'] ):
        write_masked_image(job['datasetRoot'], job['occDir'], job['img0Fn'], mask)

    countMask = mask != STEREO_NON_MASK
    return countMask.sum()

def single_process(job):
    '''
    job (dict): Contains the job description. Only "flagDepth" is used here.
    '''

    if ( job['flagDepth'] ):
        count = single_process_depth(job)
    else:
        count = single_process_disp(job)
    
    return count

def worker(name, jq, rq, lq, p):
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

            countMask = single_process(job)

            rq.put( { \
                "name": name, \
                "idx": job["idx"], \
                "disp0Fn": job["disp0Fn"], \
                "depth0Fn": job["depth0"], \
                "countMasked": countMask } )

            count += 1

            jq.task_done()
        except queue.Empty as exp:
            pass
    
    lq.put("%s: Done with %d jobs." % (name, count))

def load_dataset(fn, jsonEntry):
    '''
    fn (string): The filename of the input JSON.
    '''

    if ( not os.path.isfile(fn) ):
        raise Exception('%s does not exist. ' % (fn))

    with open(fn, 'r') as fp:
        dataset = json.load(fp)

    fileListFn = os.path.join( dataset['fileListDir'], dataset[jsonEntry] )

    if ( dataset['flagDepth'] ):
        img0FnList, _, depth0FnList = read_string_list_2D( 
            fileListFn, 3, delimiter=dataset['delimiter'] )
        depth1FnList = [ f.replace('_left', '_right') for f in depth0FnList ]
        filesDict = { \
            'flagDepth':True, \
            'img0FnList': img0FnList, \
            'depthList0': depth0FnList, \
            'depthList1': depth1FnList, \
            'bf': dataset['fb'], \
            'datasetRoot': dataset['datasetRoot'] }
    else:
        img0FnList, _, disp0FnList, disp1FnList = \
            read_string_list_2D( 
                fileListFn, 4, delimiter=dataset['delimiter'] )
        filesDict = { \
            'flagDepth': False, \
            'img0FnList': img0FnList, \
            'disp0FnList': disp0FnList, \
            'disp1FnList': disp1FnList, \
            'datasetRoot': dataset['datasetRoot'] }
    
    return filesDict

def max_num(nFiles, maxNum):
    if ( maxNum > 0 and maxNum < nFiles ):
        return maxNum
    else:
        return nFiles

def main():
    mainTimeStart = time.time()
    # ========== Entry preparation. ==========
    # Script arguments.
    parser = argparse.ArgumentParser(description="Create occlusion masks for a left-right stereo dataset.")
    args   = handle_script_args(parser)

    # Dataset description.
    filesDict = load_dataset( args.inputjson, args.jsonentry )
    if ( filesDict['flagDepth'] ):    
        nFiles = len( filesDict["depthList0"] )
    else:
        nFiles = len( filesDict['disp0FnList'] )
    
    nFiles = max_num(nFiles, args.max_num)

    # Start time.
    startTime = time.time()

    print("Main: Main process.")

    # ========== Create the queues. ==========
    jqueue      = multiprocessing.JoinableQueue() # The job queue.
    manager     = multiprocessing.Manager()
    rqueue      = manager.Queue()                 # The report queue.
    loggerQueue = multiprocessing.JoinableQueue() # The logger queue.

    # ========== Processes. ==========
    # Logger process.
    [conn1, loggerPipe] = multiprocessing.Pipe(False)
    loggerProcess = multiprocessing.Process( \
        target=logging_worker, args=["Logger", loggerQueue, conn1, filesDict['datasetRoot'], 'OccLog.log'] )

    loggerProcess.start()

    # Worker processes.
    processes   = []
    processStat = []
    pipes       = []

    loggerQueue.put("Main: Create %d processes." % (args.np))
    loggerQueue.join()

    for i in range(args.np):
        [conn1, conn2] = multiprocessing.Pipe(False)
        processes.append( multiprocessing.Process( \
            target=worker, args=[ \
            "P%03d" % (i), jqueue, rqueue, loggerQueue, conn1 ]) )
        pipes.append(conn2)

        processStat.append(1)

    for p in processes:
        p.start()

    loggerQueue.put("Main: All worker processes started.")
    loggerQueue.join()
    
    # Report queue worker process.
    resFn = os.path.join( filesDict['datasetRoot'], 'OccReport.txt' )
    pWorkerRQ = multiprocessing.Process( 
        target=worker_rq,
        args=["RQ", rqueue, loggerQueue, nFiles, resFn] )
    
    pWorkerRQ.start()
    
    loggerQueue.put('Main: result queue worker started. ')
    loggerQueue.join()

    # ========== Job submission. ==========
    maskDirCreator = DirectoryCreator( filesDict["datasetRoot"], args.occdir )
    imgDirCreator  = DirectoryCreator( filesDict["datasetRoot"], args.occdir )

    # Submit jobs.
    if ( filesDict['flagDepth'] ):
        for i in range(nFiles):
            depth0Fn = filesDict['depthList0'][i]
            maskDirCreator.create_by_filename(depth0Fn)
            
            imgFn = filesDict['img0FnList'][i]
            if ( args.write_images ):
                imgDirCreator.create_by_filename(imgFn)

            d = { "idx": i, \
                "flagDepth": True, \
                "disp0Fn": "", \
                "depth0": depth0Fn, \
                "depth1": filesDict["depthList1"][i], \
                "bf": filesDict["bf"], \
                "datasetRoot": filesDict["datasetRoot"], \
                "occDir": args.occdir, \
                "flagWriteImage": args.write_images, \
                "img0Fn": imgFn, \
                "flagNoOverwrite": args.no_overwrite }

            jqueue.put(d)
    else:
        for i in range(nFiles):
            disp0Fn = filesDict['disp0FnList'][i]
            disp1Fn = filesDict['disp1FnList'][i]
            maskDirCreator.create_by_filename(disp0Fn)
            
            imgFn = filesDict['img0FnList'][i]
            if ( args.write_images ):
                imgDirCreator.create_by_filename(imgFn)

            d = { 'idx': i, \
                'flagDepth': False, \
                'disp0Fn': disp0Fn, \
                'disp1Fn': disp1Fn, \
                'depth0': '', \
                'datasetRoot': filesDict['datasetRoot'], \
                'occDir': args.occdir, \
                'flagWriteImage': args.write_images, \
                'img0Fn': imgFn, \
                'flagNoOverwrite': args.no_overwrite }

            jqueue.put(d)

    loggerQueue.put("Main: All jobs submitted.")
    loggerQueue.join()
    
    # ========== Blocking for sub-processes. ==========
    # Report queue worker process.
    pWorkerRQ.join()
    loggerQueue.put('Result queue worker joined. ')
    
    # Job queue.
    jqueue.join()
    loggerQueue.put("Main: Job queue joined.")
    loggerQueue.join()

    endTime = time.time()

    Utils.show_delimiter("Summary.")
    loggerQueue.put("%d stereo pairs processed in %f seconds. \n" % (nFiles, endTime-startTime))
    loggerQueue.join()

    # Stop all worker processes.
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
    
    loggerQueue.join()

    # Stop the logger.
    loggerPipe.send("exit")
    loggerProcess.join()

    # ========== All done. ==========
    mainTimeEnd = time.time()
    print("Main: Done in %fh. " % ( ( mainTimeEnd - mainTimeStart ) / 3600.0 ))

    return 0

if __name__ == "__main__":
    sys.exit(main())
