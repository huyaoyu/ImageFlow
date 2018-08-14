
from __future__ import print_function

import numpy as np

def from_camera_frame_to_image(coor, intrinsics):
        """
        coor: A numpy column vector, 3x1.
        return: A numpy column vector, 2x1.
        """
        
        x = intrinsics.dot( coor )
        x = x / x[2,:]

        return x[0:2, :]

def from_depth_to_x_y(depth, intrinsics):
        wIdx = np.linspace( 0, depth.shape[1] - 1, depth.shape[1] )
        hIdx = np.linspace( 0, depth.shape[0] - 1, depth.shape[0] )

        u, v = np.meshgrid(wIdx, hIdx)

        u = u.astype(np.float)
        v = v.astype(np.float)

        focal = intrinsics[0, 0]

        x = ( u - intrinsics[0, 2] ) * depth / focal
        y = ( v - intrinsics[1, 2] ) * depth / focal

        coor = np.zeros((3, depth.size), dtype = np.float)
        coor[0, :] = x.reshape((1, -1))
        coor[1, :] = y.reshape((1, -1))
        coor[2, :] = depth.reshape((1, -1))

        # Rearrange u and v into a two-channel matrix.
        uv = np.stack([u, v], axis = 0)

        return coor, uv

def of_one_to_zero(depth, pose, intrinsics):
    """
        Calculate the optical flow of image 1 with respect to image 0. Thus, how the pixel of iamge
        1 move if they are project onto the image plane of image 1.

    depth: Depth image of camera 1.
    pose: The camera pose measured in frame_0 (camera 0's reference frame). A 4x4 numpy matrix.
    intrinsics: A 3x3 numpy matrix.

    NOTE: The 3D reference frames of camera 0 and 1 have their z-axis pointing forward. This is NOT
    the same way in which AirSim represents a 3D point and orientation with respect to its global
    frame, which in turn has its z-axis pointing downwards. 
    """

    # Calculate the 3D coordinates of the points in reference frame 1.
    coor_1, uv_1 = from_depth_to_x_y(depth, intrinsics)

    # Transform.
    R = pose[0:3, 0:3]
    T = pose[0:3, 3:4]

    coor_0 = R.dot(coor_1) + T

    uv_0 = from_camera_frame_to_image(coor_0, intrinsics)

    uv_0 = uv_0.reshape((2, depth.shape[0], depth.shape[1]))

    dudv = uv_1 - uv_0

    return dudv

if __name__ == "__main__":
    # Load the depth file.
    depth_1 = np.load("/home/yyhu/Projects/ImageFlow/data/form_one_to_zero/depth_1.npy")

    # Compose the intrinsics.
    intrinsics = np.eye(3, dtype = np.float)
    intrinsics[0, 0] = 256.0
    intrinsics[1, 1] = 256.0
    intrinsics[0, 2] = 256.0
    intrinsics[1, 2] = 192.0

    # Load the pose matrix.
    R = np.loadtxt("/home/yyhu/Projects/ImageFlow/data/form_one_to_zero/R.txt", dtype = np.float)
    T = np.loadtxt("/home/yyhu/Projects/ImageFlow/data/form_one_to_zero/T.txt", dtype = np.float)

    r = np.zeros((3, 3), dtype = np.float)
    r[0, 0] =  1.0
    r[1, 2] =  1.0
    r[2, 1] = -1.0

    R = np.matmul( np.matmul(r, R), r.transpose())
    T = r.dot(T)

    print("R = \n{}".format(R))
    print("T = \n{}".format(T))

    # Compose the pose matrix.
    pose = np.eye(4, dtype = np.float)
    pose[0:3, 0:3] = R
    pose[0:3, 3]   = T

    # Calculate dudv.
    dudv = of_one_to_zero(depth_1, pose, intrinsics)

    # Save dudv.
    np.savetxt("/home/yyhu/Projects/ImageFlow/data/form_one_to_zero/du.txt", dudv[0, :, :], fmt = "%+4d")
    np.savetxt("/home/yyhu/Projects/ImageFlow/data/form_one_to_zero/dv.txt", dudv[1, :, :], fmt = "%+4d")
