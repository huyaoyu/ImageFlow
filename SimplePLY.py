from __future__ import print_function

import copy
import numpy as np
import numpy.linalg as LA

from CommonType import NP_FLOAT

from ColorMapping import color_map

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

PLY_COLORS = [\
    "#2980b9",\
    "#27ae60",\
    "#f39c12",\
    "#c0392b",\
    ]

PLY_COLOR_LEVELS = 20

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

    color = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=NP_FLOAT)
    color[:, :, 0] = d
    color[:, :, 1] = d
    color[:, :, 2] = d

    color = ( color - d.min() ) / ( d.max() - d.min() ) * 255
    color = color.astype(np.uint8)

    return color

def output_to_ply(fn, X, imageSize, rLimit, origin):
    # Check the input X.
    if ( X.max() <= X.min() ):
        raise Exception("X.max() = %f, X.min() = %f." % ( X.max(), X.min() ) )
    
    vertices = np.zeros(( imageSize[0], imageSize[1], 3 ), dtype = NP_FLOAT)
    vertices[:, :, 0] = X[0, :].reshape(imageSize)
    vertices[:, :, 1] = X[1, :].reshape(imageSize)
    vertices[:, :, 2] = X[2, :].reshape(imageSize)
    
    vertices = vertices.reshape((-1, 3))
    rv = copy.deepcopy(vertices)
    rv[:, 0] = vertices[:, 0] - origin[0, 0]
    rv[:, 1] = vertices[:, 1] - origin[1, 0]
    rv[:, 2] = vertices[:, 2] - origin[2, 0]

    r = LA.norm(rv, axis=1).reshape((-1,1))
    mask = r < rLimit
    mask = mask.reshape(( mask.size ))
    # import ipdb; ipdb.set_trace()
    r = r[ mask ]

    cr, cg, cb = color_map(r, PLY_COLORS, PLY_COLOR_LEVELS)

    colors = np.zeros( (r.size, 3), dtype = np.uint8 )

    colors[:, 0] = cr.reshape( cr.size )
    colors[:, 1] = cg.reshape( cr.size )
    colors[:, 2] = cb.reshape( cr.size )

    write_ply(fn, vertices[mask, :], colors)
