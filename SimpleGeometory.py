from __future__ import print_function

import copy
import numpy as np

from CommonType import NP_FLOAT

def distance_of_2_points(p0, p1):
    d = p0.reshape((-1,)) - p1.reshape((-1,))

    return np.linalg.norm( d, 2 )

def clip_distance(coor, dMax):
    """
    coor is a 3xn coordinate array.
    """

    if ( dMax <= 0 ):
        raise Exception("dMax should be positive. dMax = {}. ".format( dMax ))

    if ( coor.shape[0] != 3 ):
        raise Exception("d.shape = {}. ".format(coor.shape))

    # Calculate the distance of every points.
    d = np.linalg.norm( coor, axis=0 )
    r = d / dMax
    mask = r > 1.0

    newCoor = copy.deepcopy(coor)

    newCoor[:, mask] = coor[:, mask] / r[mask]

    return newCoor

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

    R = np.array(R, dtype = NP_FLOAT)

    return R
