
import copy
import cv2
import numpy as np

# Local packages.
from SimulatedLiDARModel import VELODYNE_VLP_32C, E50_32C

# Global constants
PI_180 = np.pi / 180

def convert_DEA_2_XYZ(d, e, a):
    """
    e and a are defined in the camera frame with z-axis pointing forward, y-axis downwards.

    Arguments: 
    d (array): The distance.
    e (array): The elivation angle. Unit degree.
    a (array): The azimuthal angle. Unit degree.

    Returns: 
    A N by 3 array.
    """

    # Shift a with -90 degrees.
    a  = a  * PI_180
    e  = e  * PI_180
    dc = d  * np.cos( e )
    x  = dc * np.sin( a )
    z  = dc * np.cos( a )
    y  = d  * np.sin( e )

    return np.stack((x, y, z), axis=1)

def convert_Velodyne_DEA_2_XYZ(d, e, a):
    """
    e and a are defined in the Velodyne frame with y-axis pointing forward, z-axis upwards.

    Arguments: 
    d (array): Distance.
    e (array): Elivation angle. Unit degree.
    a (array): Azimuthal angle. Unit degree.

    Returns: 
    A N by 3 array.
    """

    e  = e  * PI_180
    a  = a  * PI_180
    dc = d  * np.cos( e )
    x  = dc * np.sin( a )
    y  = dc * np.cos( a )
    z  = d  * np.sin( e )

    return np.stack((x, y, z), axis=1)

def convert_DEA_from_camera_2_Velodyne(d, e, a):
    """
    e, and a are defined in the camera frame with z-axis pointing forward.
    Velodyne has its y-axis pointing forward. Both of these devices have
    the x-axis pointing to the right.

    For Velodyne, a is measured from y-axis to x-axis. e is measured from 
    the xy-plane to z-axis.

    Arguments: 
    d (array): Distance.
    e (array): Elivation angle. Unit degree.
    a (array): Azimuthal angle. Unit degree.

    Returns: 
    d, e, a.
    """

    return copy.deepcopy(d), -e, copy.deepcopy(a)

def convert_DEA_from_camera_2_Velodyne_XYZ(d, e, a):
    '''
    This function converts d, e, a represented points from the camera frame to the xyz Coordinates
    under the Velodyne frame.
    '''
    d, e, a = convert_DEA_from_camera_2_Velodyne(d, e, a)
    return convert_Velodyne_DEA_2_XYZ(d, e, a)

class ScanlineParams(object):
    def __init__( self, f, height, aIntervalCheck=False):
        '''
        Arguments: 
        f (int): Focal length. 
        height (int): The height of the depth images.
        aIntervalCheck (bool): Set true to enable the interval check.
        '''
        super(ScanlineParams, self).__init__()

        self.f = f
        self.width  = f + f # Assuming 90 degrees FOV.
        self.height = height
        self.flagAIntCheck = aIntervalCheck

        self.a  = None
        self.e  = None
        self.a4 = None
        self.e4 = None

        self.pX = None # X pixel coordinate in the first view.
        self.pY = None # Y pixel coordinate in the first view.
        self.x4 = None # X pixel coordinate in the combined view.
        self.y4 = None # Y pixel coordinate in the combined view.

        self.xCycle = None
        
    def check_interval(self, interval):
        """
        This function checks an interval array to see if the neighboring indices 
        stored in interval have deltas larger than or equal to 1.
        Assuming interval is a 1D array.
        """

        if ( interval.size < 3 ):
            raise Exception("interval = {}. ".format( interval ))

        d = interval[1:] - interval[:-1]

        return d.min() >= 1

    def find_coordinates(self, E, resA, offA):
        """
        This function sets up self.pX, self.pY, self.x4, self.y4, self.xCycle, 
        self.a, self.e, self.a4, self.e4.

        Arguments:
        E (float): The elevation angle of the scanline. Unit degree.
        resA (float): The azimuthal resolution. Unit degree.
        offA (float): The azimuthal offset. Unit degree.
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
        a    = np.linspace( a0, a1, int( (a1-a0)/resA + 1 ), endpoint=False )
        mask = np.logical_and( a >= -45, a < 45 )
        a    = a[mask]

        if ( a.size < N ):
            raise Exception("a.size(%d) < N(%d). " % ( a.size, N ))

        a = a[:N]

        # Find the pixel index.
        self.pX = self.f * np.tan( a * PI_180 ) + self.width/2
        self.pX = self.pX.astype(np.float32)

        if ( self.flagAIntCheck ):
            bx0 = np.floor(self.pX)
            bx1 = np.ceil(self.pX)
            if ( not self.check_interval(bx0) ):
                raise Exception("bx0 has bad interval.")
            
            if ( not self.check_interval(bx1) ):
                raise Exception("bx1 has bad interval.")
        # import ipdb; ipdb.set_trace()
        # The h index.
        self.pY = self.f / np.cos( a * PI_180 ) * np.tan( E * PI_180 ) + self.height/2
        self.pY = self.pY.astype(np.float32)

        if ( self.pY[0] < 0 or self.pY[-1] >= self.height ):
            raise Exception("self.pY[0] = %f, self.pY[-1] = %f, self.height = %f" \
                % ( self.pY[0], self.pY[-1], self.height ))

        # Shift the x and y pixel coordinates to the other 3 views.

        self.x4 = np.concatenate( 
            ( self.pX, self.pX + self.width, self.pX + 2*self.width, self.pX + 3*self.width ),
            axis=0 )
        
        self.y4 = np.tile( self.pY, [4] )

        self.xCycle = np.tile( self.pX, [4] )

        # Save the a and e.
        self.a = a
        self.e = np.tile(E, [N])

        self.a4 = np.concatenate( 
            (self.a, self.a + 90, self.a + 180, self.a + 270),
            axis=0 )
        self.e4 = np.tile( self.e, [4] )

class SimulatedLiDAR(object):
    def __init__(self, f, h, desc=None):
        '''
        Arguments: 
        f (int): Integer focal length.
        h (int): Image height. 
        desc (list of dictionaries): The LiDAR device descrition.
        '''
        super(SimulatedLiDAR, self).__init__()

        self.f = f 
        self.w = f + f # We are assuming the horizontal FOV is 90 degrees.
        self.h = h
        self.desc = desc
        self.scanlineMaps   = [ None, None ] # The two maps, x and y.
        self.xCycleArray    = None
        self.azimuthArray   = None
        self.elivationArray = None

    def set_description(self, desc):
        assert ( desc is not None ), 'desc is None. '
        self.desc = desc
    
    def initialize(self):
        '''
        This function computes the values for self.scanlineMaps, 
        self.xCycleArray, self.aximuthArray, and self.elivationArray.
        '''
        if ( self.desc is None ):
            raise Exception("self.desc is None.")

        xMaps   = []
        yMaps   = []
        aList   = []
        eList   = []
        xCycles = []

        for d in self.desc:
            E        = d["E"]
            flagFlip = d["flagFlip"]
            resA     = d["resA"]
            offA     = d["offA"]

            if ( flagFlip ):
                E = -E

            sp = ScanlineParams(self.f, self.h)
            sp.find_coordinates( E, resA, offA )

            xMaps.append( sp.x4 )
            yMaps.append( sp.y4 )
            xCycles.append( sp.xCycle )
            aList.append( sp.a4 )
            eList.append( sp.e4 )

        self.scanlineMaps[0] = np.stack( xMaps, axis=0 )
        self.scanlineMaps[1] = np.stack( yMaps, axis=0 )
        self.xCycleArray     = np.stack( xCycles, axis=0 )
        self.azimuthArray    = np.stack( aList, axis=0 )
        self.elivationArray  = np.stack( eList, axis=0 )

    def convert_depth_2_distance(self, u, v, depth):
        '''
        Convert depth to distance in a camera frame.

        Arguments:
        u (array): The x pixel coordinates.
        v (array): The y pixel coordinates.
        depth (array): The depth.

        Returns: 
        A array of distance.
        '''
        x = ( u - self.w/2 ) / self.f
        y = ( v - self.h/2 ) / self.f
        return np.sqrt( x**2 + y**2 + 1 ) * depth

    def extract(self, depthList, maxDist=None):
        """
        Extract point cloud from 4 depth images.

        Arguments:
        depthList (4-element of arrays): Contains the depth image in the order of 
            increasing azimuthal angle.
        maxDist (float): The clipping threshold. Set None to disable.

        Returns: 
        A N by 3 array, with the columns defined as distance, elivation, and azimuth.
        """
        
        # Merge the 4 depth images.
        depth = np.concatenate( depthList, axis=1 )

        # Sample depth.
        sdLN = cv2.remap( depth, self.scanlineMaps[0], self.scanlineMaps[1], interpolation=cv2.INTER_LINEAR )
        sdNR = cv2.remap( depth, self.scanlineMaps[0], self.scanlineMaps[1], interpolation=cv2.INTER_NEAREST )

        # Filter.
        sdNR = np.clip(sdNR, 1e-6, sdNR.max())
        diff = np.abs( sdLN - sdNR )
        mask = diff > 0.1
        sdLN[mask] = sdNR[mask]

        # Convert depth to distance.
        dist = self.convert_depth_2_distance( self.xCycleArray, self.scanlineMaps[1], sdLN )

        # Threshold the maximum distance.
        if ( maxDist is not None and maxDist > 0 ):
            dist = np.clip(dist, 0, maxDist)

        return np.stack( ( dist, self.elivationArray, self.azimuthArray ), axis=-1 )

def test_dummy_object():
    import SimplePLY

    sld = SimulatedLiDAR( 580, 768 )
    sld.set_description( VELODYNE_VLP_32C )
    sld.initialize()

    # Test with dummy detph images.
    depth = np.zeros( (768, 1160), dtype=np.float64 ) + 10

    depthList = [ depth, depth, depth, depth ]

    lidarPoints = sld.extract( depthList )
    lidarPoints = lidarPoints.reshape( (-1, 3) )

    xyz = convert_DEA_2_XYZ( lidarPoints[:, 0], lidarPoints[:, 1], lidarPoints[:, 2] )
    xyz = xyz.reshape((-1, 1, 3))

    SimplePLY.output_to_ply( "./DummyLidar.ply", xyz.transpose(), [ 1, xyz.shape[0] ], 1000, np.array([0, 0, 0]).reshape((-1,1)) )

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
    import SimplePLY

    # Output preparation.
    parts = os.path.split(inputFn)
    if ( not os.path.isdir(parts[0]) ):
        os.makdirs( parts[0] )

    sld = SimulatedLiDAR( 580, 768 )
    sld.set_description( VELODYNE_VLP_32C )
    sld.initialize()

    depths = np.load(inputFn)

    if ( len( depths.files ) != 4 ):
        raise Exception("depths.files = {}. ".format( depths.files ) )

    depthList = [ depths["d0"], depths["d1"], depths["d2"], depths["d3"] ]
    # depthList = [ depths["d0"] ]

    lidarPoints = sld.extract( depthList )
    lidarPoints = lidarPoints.reshape( (-1, 3) )
    xyz = convert_DEA_2_XYZ( lidarPoints[:, 0], lidarPoints[:, 1], lidarPoints[:, 2] )
    SimplePLY.output_to_ply( "%s/ExtractedLIDARPoints.ply"  % ( parts[0] ), xyz.transpose(), [ 1, xyz.shape[0] ], 1000, np.array([0, 0, 0]).reshape((-1,1)) )

    # Velodyne frame.
    d, e, a = convert_DEA_from_camera_2_Velodyne( lidarPoints[:, 0], lidarPoints[:, 1], lidarPoints[:, 2] )
    xyz = convert_Velodyne_DEA_2_XYZ( d, e, a )
    SimplePLY.output_to_ply( 
        "%s/ExtractedLIDARPoints_Velodyne.ply" % ( parts[0] ), 
        xyz.transpose(), [1, xyz.shape[0] ], 100, np.array([0, 0, 0]).reshape((-1,1)) )

if __name__ == "__main__":
    import argparse
    print("Test SimulatedLiDAR.")

    parser = argparse.ArgumentParser(description="Test SimulatedLiDAR.")

    parser.add_argument("--input", type=str, default="", \
        help="The input npz file. Leave blank for not testing.")

    args = parser.parse_args()

    test_dummy_object()

    if ( args.input != "" ):
        print("Test with %s. " % ( args.input ))
        test_with_depth_file( args.input )