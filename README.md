# ImageFlow

Welcome to the ImageFlow wiki!

# Overview #

ImageFlow is a simple program written in Python to compute the optical flow between two images. We are using it to facilitate our research. 

The images are obtained from the [Unreal Engine](https://www.unrealengine.com/en-US/blog) and the [AirSim](https://github.com/Microsoft/AirSim) package. Each image should be associated with a camera pose and a depth map. The user specifies the input information through a JSON file. The outputs will be written to folders which are arranged in a pattern similar to the filenames of the input images.

# Example usage #

The data is arranged in such a way that it consists of several different files and folders. We design it to fit our needs, so maybe it is a little bit complex.

The user needs

* A JSON file as the top-level input (File A)
* A JSON file holds the "names" of the images as a list (File B)
* A camera pose file in NumPy binary format (.npy) (File C)
* A folder contains the images (Folder A)
* A folder contains the depth information (Folder B)

## Input JSON (File A) ##
Inputs are fed into ImageFlow through a JSON file. The user could find a sample input file with the name “IFInput.json” at the root of this package. The content of IFInput.json is really straightforward and detailed definitions are as follows:

* __dataDir__: A string represents the folder for the data. 
* __startingIdx__: An integer represents the starting index in the list recorded in "File B".
* __idxStep__: An integer represents the moving step in the list recorded in "File B".
* __idxNumberRequest__: An integer represents the requested number of steps to be performed. The user could specify a un-reasonable large number to just let the program process all the images starting from "startingIdx" with a step size of "idxStep".
* __poseFilename__: A string represents the filename of File B.
* __poseName__: A string represents the variable name of the list recorded in File B. File B contains only one list. poseName is the variable name of that very list.
* __poseData__: A string represents the filename of File C.
* __outDir__: A string shows the output directory.
* __imageDir__: A string shows the image directory, Folder A.
* __imageSuffix__: A string represents the filename suffix of the input image file.
* __imageExt__: A string represents the extension part of the filename of an image file, including the period in front of the extension.
* __depthDir__: A string shows the directory which contains all the depth files, Folder B.
* __depthSuffix__: A string represents a suffix in the filenames of the depth files. We were generating these depth files and naming them after each image. So there is a uniform suffix for each depth file.
* __depthExt__: A string represents the extension part of the filename of a depth file, including the period in front of the extension.
* __camera__: A dictionary like parameter which contains the focal length and the image size. All the values are in pixel units.
* __imageMagnitudeFactor__: A floating point number represents the magnification factor for the output optical flow image. Sometimes the camera poses are so similar to each other that the optical flow image looks very dark. We just use this parameter to boost up the brightness of the optical flow image. All the raw result data written to the file system is __NOT__ scaled. When the user is using the --mf command from the command line, this imageMagnitudeFactor gets overwritten.
* __iamgeWaitTimeMS__: An integer specifies the amount of time that the program waits while it is showing the resultant optical flow image on the screen. The unit is millisecond.
* __distanceRange__: An integer represents a threshold distance from the camera. This value is used when the user wants ImageFlow to generate 3D point clouds for detailed inspection. The point clouds will be generated in the PLY format, which could be imported by software like the MeshLab. Point clouds are only generated if the user specifies --debug on the command line.
* __flagDegree__: True or False. The pixel movement will be computed as moving direction and magnitude. For the direction, the user could choose use degree as the unit for the saved result and the optical flow image. Or the user could choose radian as the unit. Specify True for degree unit.

## Image name file (File B) ##

The user needs a JSON file which holds the "names" of the images as a list. This JSON file contains only one parameter. The identifier of this parameter is specified by "__poseName__" in File A. The value of the parameter must be a list of strings. These strings are the actual names for the input images. A sample File B could be found in the sample folder.

## Camera pose file (File C) ##

The camera pose file is a NumPy binary file that contains each camera pose for the input images. The user will get a NumPy 2D array after loading this file by numpy.load(). In this array, every row represents a pose entry. A pose entry consists of two components, the 3D position vector of the camera and the orientation of the camera. An entry has 7 columns, the first three is the position vector, the remaining 4 columns represent a quaternion which in turn represents a rotation. The last element of the quaternion is the scale factor.

The user is responsible for keeping the correspondence between the rows in the NumPy array and the list defined in File B. To be more specific, the order of the names listed in File B must be the same with the row order in File C.

## Depth files ##

Each depth file is a binary NumPy data file generated by numpy.save(). Once loaded by numpy.load(), the array has the same size of the associated image but only one channel. The __plane depth__ of the Unreal Engine should be used here.

## Folders ##

All the folders specified in File A except "__dataDir__" are relative paths. They are relative to "__dataDir__".

## Command line arguments ##

The user could invoke ImageFlow.py and specify some command line arguments in the meantime. There are three command line arguments.

* __--input \<input_filename\>__: The filename of File A, with its full path or relative path. If this argument is not specified in the command line, a default input file with a filename of "IFInput.json" will be used.
* __--mf \<magnitude_factor\>__: The magnification factor at runtime the user specified to overwrite "__imageMagnitudeFactor__" in File A.
* __--debug__: Use this argument to make ImageFlow to output all debugging data into "__outDir__" with the 3D point clouds included.

##  Outputs ##

There are some text outputs on the terminal. These outputs are mainly for user reference. __PLEASE NOTE__: The _t0_ vector and the _R0_ matrix, as well as _t1_ and _R1_ are __NOT__ the values originally found in File C. These output values are the converted translation vector and rotation matrix for coordinate transformation from the world coordinate system to the camera reference frame. For example, let's define a point in the world coordinate system as {__xw__},  and its coordinate with respect to the camera frame, {__xc__}. Then we have {__xc__} = [__R__] {__xw__} + {__t__}.

The other output text on the terminal is for progress tracking and final summary.

After a normal operation of ImageFlow without the --debug command line argument, some result files are written to the file system. Optical flow is calculated between two images taking the first image as the reference. The pixel movement is described by its moving angle and moving distance (measured in pixel). All of the above data are written into different files in folder \<dataDir\> \ \<outDir\> \ ImageFlow:

* __a.dat__: The moving angle for an individual pixel. Using degree or radian as the unit specified by File A.
* __d.dat__: The moving magnitude for an individual pixel.
* __u.dat__: The x-axis pixel location of a pixel in the first image observed in the second image.
* __v.dat__: The y-axis (downwards) pixel location of a pixel in the first image observed in the second image.
* __du.dat__: The x-axis change from the first image with respect to __u.dat__.
* __dv.dat__: The y-axis change from the first image with respect to __v.dat__.
* __ad.npy__: Combines the data in __a.dat__ and __d.dat__ into a 2-channel matrix.
* __dudy.dat__: Combines the data in __du.dat__ and __dv.dat__ into a 2-channel matrix.
* __bgr.jpg__: A color image as the optical flow image.

The warped images are written back to \<dataDir\> \ \<imageDir\> for easy comparision with the original input images.

Some 3D point clouds will be written if --debug command line argument is specified. These point clouds are used for debugging purpose, however, the user a welcome to use them anyway. These files are defined as follows:

* __XInCam_0.ply__: What's the objective looks like in the reference frame of camera 0 (the first camera).
* __XInCam_1.ply__: What's the objective looks like in the reference frame of camera 1 (the second camera).
* __XInWorld_0.ply__: What's the objective looks like in the world frame from the perspective of camera 0 (the first camera).
* __XInWorld_1.ply__: What's the same objective looks like in the world frame from the perspective of camera 1 (the second camera).

For all the point clouds, the vertex color is defined by the distance from the camera center, the red the further, the blue the nearer. "__distanceRange__" in File A is a threshold (measured in meter) that any point beyond this distance will be omitted in the PLY file.
