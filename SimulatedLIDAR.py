from __future__ import print_function

import argparse
import copy
import numpy as np

def convert_DEA_2_XYZ(d, e, a):
    """
    e and a are defined in the camera frame with z-axis pointing forward.
    """

    # Shift a with -90 degrees.
    a = -a + 90
    c = np.cos( e/180.0*np.pi )
    x = d * c * np.cos( a/180.0*np.pi )
    z = d * c * np.sin( a/180.0*np.pi )
    y = d * np.sin( e/180.0*np.pi )

    return np.stack((x, y, z), axis=1)

def convert_Velodyne_DEA_2_XYZ(d, e, a):
    """
    e and a are defined in the camera frame with z-axis pointing forward.
    """

    dc = d * np.cos( e/180.0*np.pi )
    x = dc * np.sin( a/180.0*np.pi )
    y = dc * np.cos( a/180.0*np.pi )
    z = d * np.sin( e/180.0*np.pi )

    return np.stack((x, y, z), axis=1)

def convert_DEA_from_camera_2_Velodyne(d, e, a):
    """
    e, and a are defined in the camera frame with z-axis pointing forward.
    Velodyne has its y-axis pointing forward. Both of these devices have
    the x-axis pointing to the right.

    For Velodyne, a is measured from y-axis to x-axis. e is measured from 
    the xy-plane to z-axis.
    """

    return copy.deepcopy(d), -e, copy.deepcopy(a)

def convert_DEA_from_camera_2_Velodyne_XYZ(d, e, a):
    d, e, a = convert_DEA_from_camera_2_Velodyne(d, e, a)
    return convert_Velodyne_DEA_2_XYZ(d, e, a)

def convert_lidar_point_list_2_array(pointList):
    return np.concatenate( pointList, axis=0 ) 

class LIDARDescription(object):
    def __init__(self, desc=None):
        super(LIDARDescription, self).__init__()

        self.desc = desc

VELODYNE_VLP_32C = LIDARDescription( \
    [ \
        { "id":  0, "E":-25,     "flagFlip":True, "resA": 0.1, "offA":  1.4 },
        { "id":  1, "E":-1,      "flagFlip":True, "resA": 0.1, "offA": -4.2 },
        { "id":  2, "E":-1.667,  "flagFlip":True, "resA": 0.1, "offA":  1.4 },
        { "id":  3, "E":-15.639, "flagFlip":True, "resA": 0.1, "offA": -1.4 },
        { "id":  4, "E":-11.31,  "flagFlip":True, "resA": 0.1, "offA":  1.4 },
        { "id":  5, "E": 0,      "flagFlip":True, "resA": 0.1, "offA": -1.4 },
        { "id":  6, "E":-0.667,  "flagFlip":True, "resA": 0.1, "offA":  4.2 },
        { "id":  7, "E":-8.843,  "flagFlip":True, "resA": 0.1, "offA": -1.4 },
        { "id":  8, "E":-7.254,  "flagFlip":True, "resA": 0.1, "offA":  1.4 },
        { "id":  9, "E": 0.333,  "flagFlip":True, "resA": 0.1, "offA": -4.2 },
        { "id": 10, "E":-0.333,  "flagFlip":True, "resA": 0.1, "offA":  1.4 }, 
        { "id": 11, "E":-6.148,  "flagFlip":True, "resA": 0.1, "offA": -1.4 }, 
        { "id": 12, "E":-5.333,  "flagFlip":True, "resA": 0.1, "offA":  4.2 }, 
        { "id": 13, "E": 1.333,  "flagFlip":True, "resA": 0.1, "offA": -1.4 }, 
        { "id": 14, "E": 0.667,  "flagFlip":True, "resA": 0.1, "offA":  4.2 }, 
        { "id": 15, "E":-4,      "flagFlip":True, "resA": 0.1, "offA": -1.4 }, 
        { "id": 16, "E":-4.667,  "flagFlip":True, "resA": 0.1, "offA":  1.4 }, 
        { "id": 17, "E": 1.667,  "flagFlip":True, "resA": 0.1, "offA": -4.2 }, 
        { "id": 18, "E": 1,      "flagFlip":True, "resA": 0.1, "offA":  1.4 }, 
        { "id": 19, "E":-3.667,  "flagFlip":True, "resA": 0.1, "offA": -4.2 }, 
        { "id": 20, "E":-3.333,  "flagFlip":True, "resA": 0.1, "offA":  4.2 }, 
        { "id": 21, "E": 3.333,  "flagFlip":True, "resA": 0.1, "offA": -1.4 }, 
        { "id": 22, "E": 2.333,  "flagFlip":True, "resA": 0.1, "offA":  1.4 }, 
        { "id": 23, "E":-2.667,  "flagFlip":True, "resA": 0.1, "offA": -1.4 }, 
        { "id": 24, "E":-3,      "flagFlip":True, "resA": 0.1, "offA":  1.4 }, 
        { "id": 25, "E": 7,      "flagFlip":True, "resA": 0.1, "offA": -1.4 }, 
        { "id": 26, "E": 4.667,  "flagFlip":True, "resA": 0.1, "offA":  1.4 }, 
        { "id": 27, "E":-2.333,  "flagFlip":True, "resA": 0.1, "offA": -4.2 }, 
        { "id": 28, "E":-2,      "flagFlip":True, "resA": 0.1, "offA":  4.2 }, 
        { "id": 29, "E": 15,     "flagFlip":True, "resA": 0.1, "offA": -1.4 }, 
        { "id": 30, "E": 10.333, "flagFlip":True, "resA": 0.1, "offA":  1.4 }, 
        { "id": 31, "E":-1.333,  "flagFlip":True, "resA": 0.1, "offA": -1.4 }, 
         ] )

class ScanlineParams(object):
    def __init__( self, f, height, a=None, e=None, h=None, \
            bx0=None, bx1=None, by0=None, by1=None, tx=None, ty=None, idxX=None, idxY=None, \
            w00=None, w01=None, w10=None, w11=None ):
        super(ScanlineParams, self).__init__()

        self.f = f
        self.width  = f + f # Assuming 90 degrees FOV.
        self.height = height

        self.a   = a
        self.e   = e
        self.bx0 = bx0
        self.bx1 = bx1
        self.by0 = by0
        self.by1 = by1
        self.tx  = tx
        self.ty  = ty
        self.idxX = idxX
        self.idxY = idxY

        self.w00 = w00
        self.w01 = w01
        self.w10 = w10
        self.w11 = w11
    
    def check_interval(self, interval):
        """
        Assumint interval is a 1D array.
        """

        if ( interval.size < 3 ):
            raise Exception("interval = {}. ".format( interval ))

        d = interval[1:] - interval[:-1]

        if ( d.min() < 1 ):
            return False
        else:
            return True

    def find_intervals(self, E, resA, offA):
        """
        E: The elevation angle of the scanline. Unit degree.
        resA: The azimuth resolution. Unit degree.
        offA: The azimuth offset. Unit degree.
        """

        assert resA > 0

        # The number of points along the azimuthal direction.
        N = int(90.0 / resA)

        if ( N < 2 ):
            raise Exception( "N is too small. N = {}. ".format(N) )

        # Pad the aximuthal range.
        a0 = -45 - np.absolute(offA)
        a1 =  45 + np.absolute(offA)

        # Find aximuthal points.
        a = np.linspace( a0, a1, int( (a1-a0)/resA + 1 ), endpoint=False )
        mask = np.logical_and( a >= -45, a <  45 )
        a = a[mask]

        if ( a.size < N ):
            raise Exception("a.size(%d) < N(%d). " % ( a.size, N ))

        a = a[:N]

        # Find the pixel index.
        p = self.f * np.tan( a / 180.0 * np.pi )

        p = p + self.width/2

        bx0 = np.floor(p)
        bx1 = np.ceil(p)

        if ( not self.check_interval(bx0) ):
            raise Exception("bx0 has bad interval.")
        
        if ( not self.check_interval(bx1) ):
            raise Exception("bx1 has bad interval.")

        tx = p - bx0

        if ( bx0[0] < 0 or bx1[-1] >= self.width ):
            raise Exception("bx0[0] = %f, bx1[-1] = %f. " % ( bx0[0], bx1[-1] ))

        # The h index.
        p = self.f / np.cos( a / 180.0 * np.pi ) * np.tan( E / 180.0 * np.pi ) + self.height/2

        by0 = np.floor(p)
        by1 = np.ceil(p)

        ty = p - by0

        if ( by0[0] < 0 or by1[-1] >= self.height ):
            raise Exception("by0[0] = %f, by1[-1] = %f. " % ( by0[0], by1[-1] ))

        self.tx  = tx
        self.ty  = ty

        self.w00 = ( 1.0 - self.tx ) * ( 1.0 - self.ty )
        self.w01 = self.tx * ( 1.0 - self.ty )
        self.w10 = ( 1.0 - self.tx ) * self.ty
        self.w11 = self.tx * self.ty

        self.a   = a
        self.e   = np.tile(E, [N])
        self.bx0 = bx0.astype(np.int)
        self.bx1 = bx1.astype(np.int)
        self.by0 = by0.astype(np.int)
        self.by1 = by1.astype(np.int)

        # Find the closet index for the LIDAR point.
        self.idxX = np.round( self.bx0 + self.tx ).astype(np.int)
        self.idxY = np.round( self.by0 + self.ty ).astype(np.int)

class SimulatedLIDAR(object):
    def __init__(self, f, h, desc=None, varThreshold=None):
        super(SimulatedLIDAR, self).__init__()

        self.f = f 
        self.w = f + f # We are assuming the horizontal FOV is 90 degrees.
        self.h = h
        self.desc = None
        self.scanlines = []
        self.varThreshold = varThreshold

    def set_description(self, desc):
        self.desc = desc
    
    def set_variance_threshold(self, vt):
        assert vt > 0

        self.varThreshold = vt

    def clear_variance_threshold(self):
        self.varThreshold = None

    def initialize(self):
        if ( self.desc is None ):
            raise Exception("self.desc is None.")

        for d in self.desc:
            E = d["E"]
            flagFlip = d["flagFlip"]
            resA = d["resA"]
            offA = d["offA"]

            if ( flagFlip ):
                E = -E

            sp = ScanlineParams(self.f, self.h)
            sp.find_intervals( E, resA, offA )

            self.scanlines.append( sp )

    def convert_depth_2_distance(self, u, v, depth):
        x = ( u - self.w/2 ) / self.f * depth
        y = ( v - self.h/2 ) / self.f * depth

        return np.sqrt( x**2 + y**2 + depth**2 )
        
    def extract_single(self, depth, aShift, sp):
        """
        depth: The 2D depth image.
        aShift: The shift of azimuthal angle for this depth image.
        sp: A ScanlineParams object.
        """

        # Get the depth for the bounds.
        dB00 = depth[ sp.by0, sp.bx0 ]
        dB01 = depth[ sp.by0, sp.bx1 ]
        dB10 = depth[ sp.by1, sp.bx0 ]
        dB11 = depth[ sp.by1, sp.bx1 ]

        # Interpolate.
        d =  sp.w00 * dB00 + sp.w01 * dB01 + sp.w10 * dB10 + sp.w11 * dB11

        # Distance.
        dist = self.convert_depth_2_distance( sp.bx0 + sp.tx , sp.by0 + sp.ty, d )

        return dist, copy.deepcopy(sp.e), sp.a + aShift

    def extract_single_with_vt(self, depth, aShift, sp):
        """
        depth: The 2D depth image.
        aShift: The shift of azimuthal angle for this depth image.
        sp: A ScanlineParams object.
        """

        if ( self.varThreshold is None ):
            raise Exception("self.varThreshold is None")

        # Get the depth for the bounds.
        dB00 = depth[ sp.by0, sp.bx0 ]
        dB01 = depth[ sp.by0, sp.bx1 ]
        dB10 = depth[ sp.by1, sp.bx0 ]
        dB11 = depth[ sp.by1, sp.bx1 ]

        # Interpolate.
        d =  sp.w00 * dB00 + sp.w01 * dB01 + sp.w10 * dB10 + sp.w11 * dB11

        # Stack the bounds.
        B = np.stack( (dB00, dB01, dB10, dB11), axis=-1 )

        # Find the variance of B.
        vb = np.var( B, axis=-1 )

        # Find the vaiance over the threshold.
        mask = vb > self.varThreshold

        # Find the indices.
        d[mask] = depth[ sp.idxY[mask], sp.idxX[mask] ]

        # Distance.
        dist = self.convert_depth_2_distance( sp.bx0 + sp.tx , sp.by0 + sp.ty, d )

        return dist, copy.deepcopy(sp.e), sp.a + aShift

    def extract(self, depthList, maxDist=None):
        """
        depthList is a 4-element list. Contains the depth image in the order of 
        increasing azimuthal angle.
        """

        lidarPoints = []

        # Loop over every scanline.
        for sp in self.scanlines:
            aShift = 0

            distList = []
            eList    = []
            aList    = []

            for depth in depthList:
                # Extract from this depth image.
                if ( self.varThreshold is not None ):
                    d, e, a = self.extract_single_with_vt( depth, aShift, sp )
                else:
                    d, e, a = self.extract_single( depth, aShift, sp )

                if ( maxDist is not None and maxDist > 0):
                    d = np.clip(d, 0, maxDist)

                distList.append( d )
                eList.append( e )
                aList.append( a )

                aShift += 90

            # Compose single extraction.
            d = np.concatenate( distList, axis=0 )
            e = np.concatenate( eList, axis=0 )
            a = np.concatenate( aList, axis=0 )

            lidarPoints.append( np.stack( (d, e, a), axis=1 ) )

        return lidarPoints

def save_LIDAR_poinst(fn, desc, lidarPoints, flagXYZ=False):
    """
    fn: Filename, without the extension.
    desc: The description object.
    lidarPoints: The list of LIDAR points.
    """

    if ( len(desc.desc) != len(lidarPoints) ):
        raise Exception("len(desc.desc) != len(p), len(desc.desc) = %d, len(p) = %d. " % ( len(desc.desc), len(lidarPoints) ))

    if ( flagXYZ ):
        xyzList = []

        for p in lidarPoints:
            xyzList.append( convert_DEA_2_XYZ( p[:, 0], p[:, 1], p[:, 2] ) )
        
        points = np.concatenate( xyzList, axis=0 )
    else:
        points = np.concatenate( lidarPoints, axis=0 )

    # Get all the IDs.
    ids = []
    for d in desc.desc:
        ids.append( d["id"] )

    ids = np.array(ids, dtype=np.int)

    # Save the file.
    outFn = "%s.npz" % (fn)
    np.savez(outFn, points=points, ids=ids, flagXYZ=flagXYZ)

def test_dummy_object():
    import SimplePLY

    sld = SimulatedLIDAR( 580, 768 )
    sld.set_description( VELODYNE_VLP_32C.desc )

    sld.initialize()

    # Test with dummy detph images.
    depth = np.zeros( (768, 1160), dtype=np.float64 ) + 10

    depthList = [ depth, depth, depth, depth ]

    lidarPoints = sld.extract( depthList )

    xyzList = []

    for p in lidarPoints:
        xyzList.append( convert_DEA_2_XYZ( p[:, 0], p[:, 1], p[:, 2] ) )

    xyz = np.concatenate( xyzList, axis=0 ) 

    SimplePLY.output_to_ply( "./DummyLidar.ply", xyz.transpose(), [ len(xyzList), xyzList[0].shape[0]], 1000, np.array([0, 0, 0]).reshape((-1,1)) )

def save_points_as_ply(fn, xyz, lines, pointsPerLine, maxDistColor):
    """
    xyz must have a shape of 3xN.
    """

    import SimplePLY

    if ( 2 != len(xyz.shape) or xyz.shape[0] != 3 ):
        raise Exception("xyz.shape = {}. ".format(xyz.shape))

    assert lines > 0
    assert pointsPerLine > 0
    assert maxDistColor > 0

    SimplePLY.output_to_ply( fn, xyz, [ lines, pointsPerLine], maxDistColor, np.array([0, 0, 0]).reshape((-1,1)) )

def test_with_depth_file(inputFn):
    import os

    sld = SimulatedLIDAR( 580, 768 )
    sld.set_description( VELODYNE_VLP_32C.desc )

    sld.initialize()

    depths = np.load(inputFn)

    if ( len( depths.files ) != 4 ):
        raise Exception("depths.files = {}. ".format( depths.files ) )

    depthList = [ depths["d0"], depths["d1"], depths["d2"], depths["d3"] ]
    # depthList = [ depths["d0"] ]

    lidarPoints = sld.extract( depthList )

    # Output preparation.
    parts = os.path.split(inputFn)

    # Save the lidar points.
    save_LIDAR_poinst( "%s/ExtractedLIDARPoints" % ( parts[0] ), VELODYNE_VLP_32C, lidarPoints )
    save_LIDAR_poinst( "%s/ExtractedLIDARPointsXYZ" % ( parts[0] ), VELODYNE_VLP_32C, lidarPoints, flagXYZ=True )

    xyzList = []

    for p in lidarPoints:
        xyzList.append( convert_DEA_2_XYZ( p[:, 0], p[:, 1], p[:, 2] ) )

    xyz = np.concatenate( xyzList, axis=0 ) 

    SimplePLY.output_to_ply( "%s/ExtractedLIDARPoints.ply"  % ( parts[0] ), xyz.transpose(), [ len(xyzList), xyzList[0].shape[0]], 50, np.array([0, 0, 0]).reshape((-1,1)) )

    # Velodyne frame.
    xyzList = []

    for p in lidarPoints:
        d, e, a = convert_DEA_from_camera_2_Velodyne( p[:, 0], p[:, 1], p[:, 2] )
        xyzList.append( convert_Velodyne_DEA_2_XYZ( d, e, a ) )

    xyz = np.concatenate( xyzList, axis=0 ) 

    # SimplePLY.output_to_ply( "%s/ExtractedLIDARPoints_Velodyne.ply" % ( parts[0] ), xyz.transpose(), [ len(xyzList), xyzList[0].shape[0]], 50, np.array([0, 0, 0]).reshape((-1,1)) )
    save_points_as_ply( "%s/ExtractedLIDARPoints_Velodyne.ply" % ( parts[0] ), xyz.transpose(), len(xyzList), xyzList[0].shape[0], 50 )

if __name__ == "__main__":
    print("Test SimulatedLIDAR.")

    parser = argparse.ArgumentParser(description="Test SimulatedLIDAR.")

    parser.add_argument("--input", type=str, default="", \
        help="The input npz file. Leave blank for not testing.")

    args = parser.parse_args()

    # test_dummy_object()

    if ( args.input != "" ):
        print("Test with %s. " % ( args.input ))
        test_with_depth_file( args.input )