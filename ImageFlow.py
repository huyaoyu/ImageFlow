
from __future__ import print_function

import copy
import numpy as np
import numpy.linalg as LA

import cv2
import matplotlib.pyplot as plt

DATA_DIR      = "./data/test"
POSE_FILENAME = DATA_DIR + "/pose_filename.txt"
POSE_DATA     = DATA_DIR + "/pose_data.txt"

DEPTH_SUFFIX = "_depth"
DEPTH_EXT    = ".npy"

POSE_ID_0 = "000211_316623"
POSE_ID_1 = "000212_317028"

CAM_FOCAL  = 320
IMAGE_SIZE = (360, 640)

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

def output_to_ply(fn, X, colors, imageSize, rLimit):
    vertices = np.zeros(( imageSize[0], imageSize[1], 3 ), dtype = np.float)
    vertices[:, :, 0] = X[0, :].reshape(imageSize)
    vertices[:, :, 1] = X[1, :].reshape(imageSize)
    vertices[:, :, 2] = X[2, :].reshape(imageSize)
    
    vertices = vertices.reshape((-1, 3))
    colors = colors.reshape((-1,3))

    r = LA.norm(vertices, axis=1).reshape((-1,1))
    mask = r < rLimit

    mask = mask.reshape(( mask.size ))

    write_ply(fn, vertices[mask, :], colors[mask, :])

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

def show(ang, mag, shape):
    # Use Hue, Saturation, Value colour model 
    hsv = np.zeros(shape, dtype=np.uint8)
    hsv[..., 1] = 255

    # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    a = ang * 180 / np.pi / 2
    np.savetxt(DATA_DIR + "/ang180.dat", a, fmt="%.2e")

    hsv[..., 0] = (ang+np.pi) * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    np.savetxt(DATA_DIR + "/rgb.dat", rgb[:, :, 0], fmt="%3d")

    plt.imshow(rgb)
    plt.show()

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
    _, poseIDs = load_IDs(POSE_FILENAME)
    poseData   = np.loadtxt(POSE_DATA, dtype = np.float)
    # print(poseData.shape)
    print("poseData and poseFilenames loaded.")

    # Camera.
    cam_0 = CameraBase(CAM_FOCAL, IMAGE_SIZE)
    print(cam_0.imageSize)
    print(cam_0.cameraMatrix)

    cam_1 = cam_0

    # # Test the projection of the camera.
    # X = np.array([ cam.imageSize[1], cam.imageSize[0], 2*cam.focal ]).reshape(3,1)
    # x = cam.from_camera_frame_to_image(X)
    # print(X)
    # print(x)

    # Get the pose of the first position.
    R0, t0, q0= get_pose_by_ID(POSE_ID_0, poseIDs, poseData)
    R0Inv = LA.inv(R0)

    print("t0 = \n{}".format(t0))
    print("q0 = \n{}".format(q0))
    print("R0 = \n{}".format(R0))
    print("R0Inv = \n{}".format(R0Inv))

    # Get the pose of the second position.
    R1, t1, q1 = get_pose_by_ID(POSE_ID_1, poseIDs, poseData)
    R1Inv = LA.inv(R1)

    R = np.matmul( R1, R0Inv )
    print("R = \n{}".format(R))

    # Load the depth of the first image.
    depth_0 = np.load( DATA_DIR + "/" + POSE_ID_0 + DEPTH_SUFFIX + DEPTH_EXT )
    np.savetxt( DATA_DIR + "/depth_0.dat", depth_0, fmt="%.2e")

    # Calculate the x and y coordinates int the first camera's frame.
    X0 = cam_0.from_depth_to_x_y(depth_0)

    colors = depth_to_color(depth_0, 200)
    output_to_ply(DATA_DIR + '/XInCam_0.ply', X0, colors, cam_0.imageSize, 200)

    # The coordinates in the world frame.
    XWorld_0  = R0Inv.dot(X0 - t0)
    output_to_ply(DATA_DIR + "/XInWorld_0.ply", XWorld_0, colors, cam_1.imageSize, 200)

    # Load the depth of the second image.
    depth_1 = np.load( DATA_DIR + "/" + POSE_ID_1 + DEPTH_SUFFIX + DEPTH_EXT )
    np.savetxt( DATA_DIR + "/depth_1.dat", depth_1, fmt="%.2e")

    X1 = cam_1.from_depth_to_x_y(depth_1)

    colors = depth_to_color(depth_1, 200)
    output_to_ply(DATA_DIR + "/XInCam_1.ply", X1, colors, cam_1.imageSize, 200)

    XWorld_1 = R1Inv.dot( X1 - t1 )
    output_to_ply(DATA_DIR + "/XInWorld_1.ply", XWorld_1, colors, cam_1.imageSize, 200)

    # ===

    # X1 = R1.dot(X) + t1

    # output_to_ply(DATA_DIR + '/X1.ply', X1, colors, cam.imageSize, 200)

    # # The image coordinates in the second camera.
    # c = cam.from_camera_frame_to_image(X1)

    # # Get new u anv v
    # u = c[0, :].reshape(cam.imageSize)
    # v = c[1, :].reshape(cam.imageSize)

    # # Get the du and dv.
    # du, dv = du_dv(u, v, cam.imageSize)

    # np.savetxt(DATA_DIR + "/du.dat", du.astype(np.int), fmt="%3d")
    # np.savetxt(DATA_DIR + "/dv.dat", dv.astype(np.int), fmt="%3d")

    # a = np.arctan2( dv, du )

    # d = np.sqrt( du * du + dv * dv )

    # np.savetxt(DATA_DIR + "/a.dat", a, fmt="%.2e")
    # np.savetxt(DATA_DIR + "/d.dat", d, fmt="%.2e")

    # show(a, d, (cam.imageSize[0], cam.imageSize[1], 3))
