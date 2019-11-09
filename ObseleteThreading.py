def process_single_thread(name, inputParams, args, poseIDs, poseData, indexList, startII, endII, flagShowFigure=False):
    # Data directory.
    dataDir = inputParams["dataDir"]

    # The magnitude factor.
    mf = get_magnitude_factor_from_input_parameters( inputParams, args )

    # Camera.
    cam_0 = CameraBase(inputParams["camera"]["focal"], inputParams["camera"]["imageSize"])
    # print(cam_0.imageSize)
    # print(cam_0.cameraMatrix)

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
    warpErrMaxEntry = { "idx": -1, "poseID_0": "N/A", "poseID_1": "N/A", "warpErr": 0.0, "warpErr_01": 0.0 }

    for i in range( startII+1, endII+1 ):
        # Show the delimiter.
        show_delimiter( title = "%s: %d / %d" % ( name, count + 1, estimatedLoops ), leading="", ending="" )

        idxPose0 = indexList[i - 1]
        idxPose1 = indexList[i]

        poseID_0 = poseIDs[ idxPose0 ]
        poseID_1 = poseIDs[ idxPose1 ]

        # print("poseID_0 = %s, poseID_1 = %s" % (poseID_0, poseID_1))

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
        depth_0 = np.load( depthDir + "/" + poseID_0 + depthTail ).astype(NP_FLOAT)
        
        if ( True == args.debug ):
            np.savetxt( outDir + "/depth_0.dat", depth_0, fmt="%.2e")

        # Calculate the coordinates in the first camera's frame.
        X0C = cam_0.from_depth_to_x_y(depth_0) # Coordinates in the camera frame. z-axis pointing forwards.
        X0  = cam_0.worldRI.dot(X0C)           # Corrdinates in the NED frame. z-axis pointing downwards.
        
        if ( True == args.debug ):
            try:
                output_to_ply(outDir + '/XInCam_0.ply', X0C, cam_0.imageSize, distanceRange, CAMERA_ORIGIN)
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
        depth_1 = np.load( depthDir + "/" + poseID_1 + depthTail ).astype(NP_FLOAT)

        if ( True == args.debug ):
            np.savetxt( outDir + "/depth_1.dat", depth_1, fmt="%.2e")

        # Calculate the coordinates in the second camera's frame.
        X1C = cam_1.from_depth_to_x_y(depth_1) # Coordinates in the camera frame. z-axis pointing forwards.
        X1  = cam_1.worldRI.dot(X1C)           # Corrdinates in the NED frame. z-axis pointing downwards.

        if ( True == args.debug ):
            try:
                output_to_ply(outDir + "/XInCam_1.ply", X1C, cam_1.imageSize, distanceRange, CAMERA_ORIGIN)
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
        # The coordinate of the pixels of the first camera projected in the second camera's frame (NED).
        X_01 = R1.dot(XWorld_0) + t1

        # The image coordinates in the second camera.
        X_01C = cam_0.worldR.dot(X_01)                  # Camera frame, z-axis pointing forwards.
        c     = cam_0.from_camera_frame_to_image(X_01C) # Image plane coordinates.

        if ( True == args.debug ):
            try:
                output_to_ply(outDir + '/X_01C.ply', X_01C, cam_0.imageSize, distanceRange, CAMERA_ORIGIN)
            except Exception as e:
                print("Cannot write PLY file for X_01. Exception: ")
                print(e)

        # Get new u anv v
        u = c[0, :].reshape(cam_0.imageSize)
        v = c[1, :].reshape(cam_0.imageSize)
        np.savetxt(outDir + "/u.dat", u, fmt="%+.2e")
        np.savetxt(outDir + "/v.dat", v, fmt="%+.2e")

        # Get the du and dv.
        du, dv = du_dv(u, v, cam_0.imageSize)
        np.savetxt(outDir + "/du.dat", du, fmt="%+.2e")
        np.savetxt(outDir + "/dv.dat", dv, fmt="%+.2e")

        dudv = np.zeros( ( cam_0.imageSize[0], cam_0.imageSize[1], 2), dtype = NP_FLOAT )
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
        warppedImg, meanWarpError, meanWarpError_01 = warp_image(imgDir, poseID_0, poseID_1, imgSuffix, imgExt, X_01C, X1C, u, v)

        if ( meanWarpError > warpErrThres ):
            # print("meanWarpError (%f) > warpErrThres (%f). " % ( meanWarpError, warpErrThres ))
            overWarpErrThresList.append( { "idx": i, "poseID_0": poseID_0, "poseID_1": poseID_1, "meanWarpError": meanWarpError, "meanWarpError_01": meanWarpError_01 } )

        if ( meanWarpError > warpErrMaxEntry["warpErr"] ):
            warpErrMaxEntry["idx"] = i
            warpErrMaxEntry["poseID_0"]   = poseID_0
            warpErrMaxEntry["poseID_1"]   = poseID_1
            warpErrMaxEntry["warpErr"]    = meanWarpError
            warpErrMaxEntry["warpErr_01"] = meanWarpError_01

        if ( True == flagShowFigure ):
            cv2.imshow('img', warppedImg)
            # The waitKey() will be executed in show() later.
            # cv2.waitKey(0)

        # Show and save the resulting HSV image.
        if ( 1 == estimatedLoops ):
            show(a, d, None, outDir, poseID_0, None, angleShift, flagShowFigure=flagShowFigure)
        else:
            show(a, d, None, outDir, poseID_0, (int)(inputParams["imageWaitTimeMS"]), mf, angleShift, flagShowFigure=flagShowFigure)

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

class ImageFlowThread(Thread):
    def __init__(self, name, inputParams, args, poseIDs, poseData, indexList, startII, endII, flagShowFigure=False):
        super(ImageFlowThread, self).__init__()

        self.setName( name )

        self.name           = name
        self.inputParams    = inputParams
        self.args           = args
        self.poseIDs        = poseIDs
        self.poseData       = poseData
        self.indexList      = indexList
        self.startII        = startII
        self.endII          = endII
        self.flagShowFigure = flagShowFigure

        self.overWarpErrThresList = None
        self.warpErrMaxEntry      = None

    def run(self):
        self.overWarpErrThresList, self.warpErrMaxEntry = \
            process_single_thread( \
                self.name, 
                self.inputParams, self.args, 
                self.poseIDs, self.poseData, 
                self.indexList, self.startII, self.endII, 
                flagShowFigure=self.flagShowFigure )