from __future__ import print_function

import copy
import numpy as np

def convert_DEA_2_XYZ(d, e, a):
    # Shift a with -90 degrees.

    a = a - 90
    c = np.cos( e/180.0*np.pi )
    x = d * c * np.cos( a/180.0*np.pi )
    z = d * c * np.sin( a/180.0*np.pi )
    y = d * np.sin( e/180.0*np.pi )

    return np.stack((x, y, z), axis=1)

class ScanlineParams(object):
    def __init__(self, f, height, a=None, e=None, h=None, b0=None, b1=None, t=None):
        super(ScanlineParams, self).__init__()

        self.f = f
        self.width  = f + f # Assuming 90 degrees FOV.
        self.height = height

        self.a  = a
        self.e  = e
        self.h  = h
        self.b0 = b0
        self.b1 = b1
        self.t  = t
    
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

        b0 = np.floor(p)
        b1 = np.ceil(p)

        if ( not self.check_interval(b0) ):
            raise Exception("b0 has bad interval.")
        
        if ( not self.check_interval(b1) ):
            raise Exception("b1 has bad interval.")

        t = p - b0

        if ( b0[0] < 0 or b1[-1] >= self.width ):
            raise Exception("b0[0] = %f, b1[-1] = %f. " % ( b0[0], b1[-1] ))

        # The h index.
        idxH = np.floor( self.f * np.tan(E / 180.0 * np.pi) + self.height/2 )

        self.a  = a
        self.e  = np.tile(E, [N])
        self.h  = np.tile(idxH, [N]).astype(np.int)
        self.b0 = b0.astype(np.int)
        self.b1 = b1.astype(np.int)
        self.t  = t

class SimulatedLIDAR(object):
    def __init__(self, f, h, desc=None):
        super(SimulatedLIDAR, self).__init__()

        self.f = f 
        self.w = f + f # We are assuming the horizontal FOV is 90 degrees.
        self.h = h
        self.desc = None
        self.scanlines = []

    def set_description(self, desc):
        self.desc = desc

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
        dB0 = depth[ sp.h, sp.b0 ]
        dB1 = depth[ sp.h, sp.b1 ]

        # Interpolate.
        d = dB0 + sp.t * ( dB1 - dB0 )

        # Distance.
        dist = self.convert_depth_2_distance( sp.b0 + sp.t, sp.h, d )

        return dist, copy.deepcopy(sp.e), copy.deepcopy(sp.a) + aShift

    def extract(self, depthList):
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
                d, e, a = self.extract_single( depth, aShift, sp )

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

if __name__ == "__main__":
    import SimplePLY

    print("Test SimulatedLIDAR.")

    sld = SimulatedLIDAR( 580, 672 )
    sld.set_description( [ \
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

    sld.initialize()

    # Test with dummy detph images.
    depth = np.zeros( (672, 1160), dtype=np.float64 ) + 10

    depthList = [ depth, depth, depth, depth ]

    lidarPoints = sld.extract( depthList )

    xyzList = []

    for p in lidarPoints:
        xyzList.append( convert_DEA_2_XYZ( p[:, 0], p[:, 1], p[:, 2] ) )

    xyz  = np.concatenate( xyzList, axis=0 ) 

    SimplePLY.output_to_ply( "./DummyLidar.ply", xyz.transpose(), [ len(xyzList), xyzList[0].shape[0]], 1000, np.array([0, 0, 0]).reshape((-1,1)) )