import os, sys, glob, time, warnings, math, copy
import numpy as np
import matplotlib.path as mPath
from tqdm import tqdm
from scipy.spatial import Delaunay
from pyevtk.hl import gridToVTK, pointsToVTK
from shapely.geometry import LineString, Point
from shapely.ops import substring
from geopandas import GeoSeries
from scipy.spatial.distance import cdist
from pykrige.ok import OrdinaryKriging


# FUNCTIONS ===================================================================
def read_control():
    '''Read control file with settings'''
    ctrl = []
    global A1, A2, B1, B2, B3, B4, I1, I2, I3, I4, P1, P2, P3, P4, Q1
    A1 = A2 = np.nan        # raster resolution
    I3 = 2.5                # interpolation window length (I3 x dC)
    B2 = B3 = 1             # numer of Chaikin's iterations
    B4 = 0.5                # thalweg spacing (IW longitudinal resolution)
    
    try:
        with open('control.txt', 'r') as file:
            for line in file:
                ctrl.append(line.split())
            file.close()
            for i in range(len(ctrl)):
                if ctrl[i]:
                    # Meausrement and output resolution
                    if ctrl[i][0] == 'A':
                        if int(ctrl[i][1]) == 1:
                            A1 = float(ctrl[i][2])
                        if int(ctrl[i][1]) == 2:
                            A2 = float(ctrl[i][2])
                    
                    # Interpolation settings
                    if ctrl[i][0] == 'I':
                        if int(ctrl[i][1]) == 1:
                            I1 = float(ctrl[i][2])
                        if int(ctrl[i][1]) == 2:
                            I2 = int(ctrl[i][2])
                        if int(ctrl[i][1]) == 3:
                            I3 = float(ctrl[i][2])
                        if int(ctrl[i][1]) == 4:
                            I4 = float(ctrl[i][2])

                    # Printer settings
                    if ctrl[i][0] == 'P':
                        if int(ctrl[i][1]) == 1:
                            P1 = int(ctrl[i][2])
                        if int(ctrl[i][1]) == 2:
                            P2 = int(ctrl[i][2])
                        if int(ctrl[i][1]) == 3:
                            P3 = int(ctrl[i][2])
                        if int(ctrl[i][1]) == 4:
                            P4 = int(ctrl[i][2])
                    
                    if ctrl[i][0] == 'Q':
                        if int(ctrl[i][1]) == 1:
                            Q1 = int(ctrl[i][2])
    except IOError:
        print('\n', 'ERROR :: wave_ts.dat not found!')
        quit()


def cca(coords, refinements):
    '''Chaikin's corner cutting algorithm based on Chaikin G. (1974)'''
    coords = np.array(coords)
    for _ in range(refinements):
        L = coords.repeat(2, axis=0)
        R = np.empty_like(L)
        R[0] = L[0]
        R[2::2] = L[1:-1:2]
        R[1:-1:2] = L[2::2]
        R[-1] = L[-1]
        coords = L * 0.75 + R * 0.25
    return coords


def redist_vertices(xy, dr):
    '''Redistribute vertices'''
    line = GeoSeries(map(Point, zip(xy[:, 0], xy[:, 1])))
    line = LineString(line)
    
    fst = True
    for i in np.arange(0, line.length, dr):
        s = substring(line, i, i+dr)
        
        pts = s.boundary
        pts = shapely_to_numpy(pts)
        pts = np.reshape(pts, (-1, 2))

        if fst == True:
            pts_all = pts
            fst = False
        else:
            pts = pts[1, :]
            pts_all = np.vstack((pts_all, pts))        
    return pts_all


def shapely_to_numpy(pts):
    '''Convert shapely feature to numpy array'''
    pts = np.asanyarray(pts)
    pts = str(pts)
    pts = pts.replace(' ', ',')
    pts = pts.replace(',,', ',')
    pts = pts.replace('(', '')
    pts = pts.replace(')', '')
    pts = pts.split(',')
    pts = pts[1:]
    pts = np.array(pts, dtype=float)
    return pts

    
def thalweg_ang(tw):
    '''Thalweg angles finder'''
    tw = np.c_[tw, np.zeros((len(tw), 2))]
    for i in range(len(tw)):
        dx = []
        dy = []
        if i == 0:
            dx = tw[i+1, 0] - tw[i, 0]
            dy = tw[i+1, 1] - tw[i, 1]
        elif i == len(tw)-1:
            dx = tw[i, 0] - tw[i-1, 0]
            dy = tw[i, 1] - tw[i-1, 1]
        else:
            dx = 0.5 * (tw[i+1, 0] - tw[i-1, 0])
            dy = 0.5 * (tw[i+1, 1] - tw[i-1, 1])
        tw[i, 2] = dy / dx
        tw[i, 3] = -np.arctan(1 / tw[i, 2])
    del dx, dy
    return tw


def alpha_shape(points, alpha, only_outer=True):
    '''convex hull'''
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        if (i, j) in edges or (j, i) in edges:
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                edges.remove((j, i))
            return
        edges.add((i, j))
    tri = Delaunay(points)
    edges = set()
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        # print(circum_r)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    boundary = []
    for i, j in edges:
        boundary.append([i, j])
    boundary = np.array(boundary)
    b = np.zeros((len(boundary), 2))
    b_points = np.zeros((len(boundary), 2))
    for i in range(len(b)):
        if i == 0:
            b[i, :] = boundary[i, :]
        else:
            ind = np.argwhere(boundary[:, 0] == b[i-1, 1])
            b[i, :] = boundary[ind[0], :]
        b_points[i, :] = points[int(b[i, 0]), :]
    return b_points


def spline(xi, yi, tw, i):
    '''Local interpolation spline --> PATH'''
    D_tw = np.sqrt((tw[:, 0] - xi[i])*(tw[:, 0] - xi[i]) + (tw[:, 1] - yi[i])*(tw[:, 1] - yi[i]))
    tw0_ind = np.argmin(D_tw)
    theta = tw[tw0_ind, 3]

    global dx_node
    global dy_node
    dx_node = xi[i] - tw[tw0_ind, 0]
    dy_node = yi[i] - tw[tw0_ind, 1]
    
    n_dist = math.ceil(A2*I3/B4/2)
    
    i1 = int(max(tw0_ind-n_dist, 0))
    i2 = int(min(tw0_ind+n_dist, len(tw)-1))

    global s0
    s0 = tw[i1:i2+1, 0:2]
    s0[:, 0] = s0[:, 0] #+ dx_node
    s0[:, 1] = s0[:, 1] #+ dy_node
    
    dx_spline = s0 * 0 + np.cos(theta) * (I4 / 2)
    dy_spline = s0 * 0 + np.sin(theta) * (I4 / 2)
    dr_spline = np.concatenate((np.expand_dims(dx_spline[:, 0], -1),
                                np.expand_dims(dy_spline[:, 1], -1)), axis=1)
    s_merge = np.concatenate((s0 -dr_spline, np.flipud(s0) +dr_spline), axis=0)
    s_merge = np.vstack([s_merge, [s_merge[0,0], s_merge[0,1]]])
    s_merge[:, 0] = s_merge[:, 0] + dx_node
    s_merge[:, 1] = s_merge[:, 1] + dy_node
    path_spline = mPath.Path(s_merge)
    return path_spline


def dist2d(xi, yi, xyz):
    '''Distance vector calculation'''
    xyz = np.array(xyz)
    if xyz.ndim == 1:
        D = np.sqrt(np.power((xi - xyz[0]), 2.0) + np.power((yi - xyz[1]), 2.0))
    else:
        D = np.sqrt(np.power((xi - xyz[:, 0]), 2.0) + np.power((yi - xyz[:, 1]), 2.0))
    return D


def spline_distance():
    '''Calculation of distances along the spline-based coordinate system'''
    s1 = copy.deepcopy(s0)
    s1[:, 0] = s1[:, 0] + dx_node
    s1[:, 1] = s1[:, 1] + dy_node

    DM = cdist(xyz[in_ind, 0:2], s1)        # distance matrix
    DM_min_ind = np.argmin(DM, axis=1)      # index of closest thalweg point
    
    D1 = abs(DM_min_ind - np.argmin(dist2d(x0, y0, s1))) * B4   # Lognitudinal distance
    
    D2 = np.empty((len(D1)))                # Lateral distance
    for i in range(DM.shape[0]):
        D2[i] = abs(DM[i, DM_min_ind[i]])

    D2_norm = (D2 - np.min(D2)) / (np.max(D2) - np.min(D2))
    D2_relax = 1.00001 - (np.exp(D2_norm**2.0) - 1) / (np.e - 1)
    
    D = np.sqrt(D1*D1 + D2*D2)              # 'Euclidean' distance
    D = D / D2_relax
    return D + 1e-5
    
def WALID(ind):
    '''Interpolation mainloop'''
    if inpoly_flag[ind] == True:
        global x0
        global y0
        global in_ind
        x0 = xi[ind]
        y0 = yi[ind]
        z_int = []

        # Definition of local interpolation window
        path = spline(xi, yi, tw, ind)        
        in_IW = path.contains_points(xyz[:, 0:2])
        in_ind = [l for l, x in enumerate(in_IW) if x]    
        
        if not in_ind or len(in_ind) < 6: # No points in windows –> regular IDW
            D = dist2d(xi[ind], yi[ind], xyz)
            dist = 1 / np.power(D[np.argpartition(D, I2)], I1)
            vals = xyz[np.argpartition(D, I2), 2]
            z_int = np.sum(dist[0:I2] * vals[0:I2] / np.sum(dist[0:I2]))
        else:
            in_xyz = np.empty((len(in_ind), 7))
            in_xyz[:, 0:3] = xyz[in_ind, :]
            in_xyz[:, 3] = spline_distance()
            in_xyz[:, 4] = 1 / np.power(in_xyz[:, 3], I1)
            in_xyz[:, 5] = in_xyz[:, 2] * in_xyz[:, 4]
            z_int = np.sum(in_xyz[:, 5]) / np.sum(in_xyz[:, 4])
            
        if z_int.size == 0 or np.isnan(z_int) == True:
            D = dist2d(xi[ind], yi[ind], xyz)
            dist = 1 / np.power(D[np.argpartition(D, I2)], I1)
            vals = xyz[np.argpartition(D, I2), 2]
            z_int = np.sum(dist[0:I2] * vals[0:I2] / np.sum(dist[0:I2]))

    else:
        z_int = np.nan
    return z_int


###################################################################################################
########################################## MAIN ###################################################
###################################################################################################
warnings.filterwarnings("ignore")

# Initialize working directory
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
    os.chdir(application_path)

# Start program
t0 = time.time()
print('\n')
print('River DTM generator')
print('===================')
print('Reading input files ...')


# READ INPUTS =================================================================
# Read control
if not glob.glob('control.txt'):
    print('\n ERROR :: control file not found! \n')
    quit()
else:
    read_control()
    print('  Control file found and read!')

# Ready *.xyz
fn_data = glob.glob('*.xyz')
if not fn_data:
    print('\n ERROR :: no measurement (.xyz) files found! \n')
    quit()
else:
    xyz = np.loadtxt(fn_data[0])
    print('  Measurement file [' + fn_data[0] + '] found and read!')

# Read thalweg
fn_tw = 'thalweg.xy'
if not fn_tw:
    print('\n ERROR :: thalweg file (thalweg.xy) not found! \n')
    quit()
else:
    tw = np.loadtxt(fn_tw)
    print('  Thalweg file [' + fn_tw + '] found and read!')

# Read query file
if Q1 == 1:
    fn_que = 'query.xy'
    if not fn_que:
        print('\n ERROR :: query file (query.xy) not found! \n')
        quit()
    else:
        que = np.loadtxt(fn_que)
        print('  Query file [' + fn_que + '] found and read!')



# PRE–PROC ====================================================================
# Initializing thalweg
print('Initializing thalweg geometry ...')
tw = cca(tw, B3)                # Chaikin's iterations
tw = redist_vertices(tw, B4)
tw = thalweg_ang(tw)

if Q1 == 0:
    # Generate nodes for interpolation
    print('Finding concave hull ...')
    boundary = alpha_shape(xyz[:, 0:2], A2, only_outer=True)
    boundary = cca(boundary[0::B1, :], B2)  # smoothing
    path_bound = mPath.Path(boundary)       # find path for raycasting

    # Generate raster
    print('Raster generation ...')
    rast_x = np.arange(min(boundary[:, 0]), max(boundary[:, 0]) + A1, A1)
    rast_y = np.arange(min(boundary[:, 1]), max(boundary[:, 1]) + A1, A1)

    # Number of rows/cols
    nr1 = len(rast_x)       # cols
    nr2 = len(rast_y)       # rows
    ind = range(nr1*nr2)    # number of elemeents

    print('Raster size [rows x cols]: ' + str(nr2) + ' x ' + str(nr1))

    Rx, Ry = np.meshgrid(rast_x, rast_y)
    Rz = np.zeros((nr2, nr1))

    xi = np.reshape(Rx, (nr1*nr2))
    yi = np.reshape(Ry, (nr1*nr2))
    ind = range(len(xi))
    inpoly_flag = (path_bound.contains_points(np.transpose([xi, yi])))

elif Q1 == 1:
    xi = que[:, 0]
    yi = que[:, 1]
    ind = range(len(xi))
    inpoly_flag = np.full(len(xi), True, dtype=bool)

# INTERPOLATION ===============================================================
print('Interpolation started ...')
x = map(WALID, tqdm(ind, ncols=64))
zi = np.array(list(x))

# Reshape files – results in x|y|z format (containing NaNs at zi)
print('Reshaping results ...')
if Q1 == 0:                 # Interp to grid
    dtm = np.empty((nr1*nr2, 3))
    dtm[:, 0] = xi
    dtm[:, 1] = yi
    dtm[:, 2] = zi
    Rz = np.reshape(zi, (nr2, nr1), order='C')

#if Q1 == 1:                 # Interp to query 
if Q1 == 0:
    nan0 = np.argwhere(~np.isnan(zi))
    dtm = np.empty((len(nan0), 3))
    for i in range(len(nan0)):
        dtm[i, 0] = xi[nan0[i]]
        dtm[i, 1] = yi[nan0[i]]
        dtm[i, 2] = zi[nan0[i]]

if Q1 == 1:
    dtm = np.empty((len(xi), 3))
    for i in range(len(xi)):
        dtm[i, 0] = xi[i]
        dtm[i, 1] = yi[i]

        if np.isnan(zi[i]) == True:
            D = dist2d(xi[i], yi[i], xyz)
            dist = 1 / np.power(D[np.argpartition(D, I2)], I1)
            vals = xyz[np.argpartition(D, I2), 2]
            dtm[i, 2] = np.sum(dist[0:I2] * vals[0:I2] / np.sum(dist[0:I2]))
        else:
            dtm[i, 2] = zi[i]

# PRINTER  ====================================================================
if P1 == 1 or P2 == 1 or P3 == 1 or P4 == 1:
    if not os.path.exists('out'):
        os.makedirs('out')

    # Print output files
    if P1 == 1:
        np.savetxt(('out/WALID' +  '_'  + fn_data[0][0:-4] + '.dat'), dtm, delimiter='\t', fmt='%.3f')

    if P2 == 1 and Q1 == 0:
        gx = np.array(np.reshape(Rx, (nr2, nr1, 1)))
        gy = np.array(np.reshape(Ry, (nr2, nr1, 1)))
        gz = np.array(np.reshape(Rz, (nr2, nr1, 1)))
        z = np.array(np.reshape(Rz, (nr2, nr1, 1)))
        gridToVTK(('out/WALID' + '_' + fn_data[0][0:-4]), gx, gy, gz, pointData = {'z [m]' : z})

    if P3 == 1:
        x = np.array(dtm[:, 0])
        y = np.array(dtm[:, 1])
        z = np.array(dtm[:, 2])
        pointsToVTK(('out/WALID' + '_' + fn_data[0][0:-4]), x, y, z, data = {'z [m]' : z})

    if P4 == 1:
        x = np.array(xyz[:, 0])
        y = np.array(xyz[:, 1])
        z = np.array(xyz[:, 2])
        pointsToVTK(('out/'+fn_data[0][0:-4]), x, y, z, data = {'z [m]' : z})

# Calculation time
t1 = time.time()
print('Program completed in %.1f' % (t1-t0), ' s')

quit()