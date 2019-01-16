# imports
import numpy as np
import itertools
from numpy import linalg as LA
from . import dissimilarities


def sphere(N=400, dimension=2, factor=1, return_cover_radius=False):
    """
    Random points on a sphere.

    Parameters
    -----------
    N : int (default : 400)
        Number of points
    dimension : int (default : 2)
        Intrinsic dimension of the sphere
    factor : float (default : 1)
        Oversampling facto used to get more regular polytope.
    return_cover_radius : boolean (default : False)
        Return minimal distance between points.
    """
    M = int(N * factor ** dimension)
    columns = []
    for k in range(dimension+1):
        columns.append(np.random.normal(size=M).reshape(M, 1))
    X = np.hstack(columns)
    X = X/LA.norm(X, axis=1).reshape(M, 1)
    insertion_times = dissimilarities.get_farthest_insertion_times(
        X=X,
        dissimilarity='euclidean',
        initial_point=0,
        n_samples=N,
        resolution=0)
    point_sample = insertion_times > -np.inf
    X = X[point_sample]
    if return_cover_radius:
        return X, np.min(insertion_times[point_sample])
    else:
        return X


def regular_polygon(N=100):
    """
    Regular polygon with N vertices

    Parameters
    -----------
    N : int (default : 100)
        Number of points
    """
    return clifford_torus_grid(N=N, dimension=1)


def clifford_torus(N=100):
    '''
    Spiral on the Clifford torus

    Parameters
    -----------
    N : int (default : 100)
        Number of points
    '''
    x = np.linspace(0, 2*np.pi, num=N, endpoint=False).reshape(N, 1)
    y = np.sqrt(N) * x
    return np.hstack((np.cos(x), np.sin(x), np.cos(y), np.sin(y)))


def clifford_embedding(t):
    embedding = np.empty((t.shape[0], 2 * t.shape[1]))
    for k in range(t.shape[1]):
        embedding[:, 2 * k] = np.cos(2 * np.pi * t[:, k])
        embedding[:, 2 * k + 1] = np.sin(2 * np.pi * t[:, k])
    return embedding


def clifford_torus_grid(N=100, dimension=2):
    """
    Regular grid of points on the Clifford torus.

    Parameters
    -----------
    N : int (default : 100)
        Number of points
    dimension : int (default : 2)
        Intrinsic dimension of the torus
    """
    subdivisions = int(N**(1./dimension))
    return clifford_embedding(regular_cubic_grid(
        dimension=dimension,
        subdivisions=subdivisions - 1,
        endpoint=False))


def regular_cubic_grid(N=27, dimension=3, subdivisions=None, endpoint=True):
    """"
    Regular cubical grid.

    Parameters
    -----------
    N : int (default : 27)
        Number of points
    dimension : int (default : 2)
        Dimension of the boundary of the cube
    subdivisions : int (default : None)
        Number of subdivisions of underlying intervals \
        Overwrites N if specified
    endpoint : boolean (default : True)
        Include endpoints of intervals
    """
    if subdivisions is None:
        subdivisions = int(N**(1./dimension))
    segment = [np.linspace(0, 1, subdivisions + 1, endpoint=endpoint)]
    return np.array(list(itertools.product(*(segment * dimension))))


def regular_cube_boundary_grid(dimension=2, subdivisions=2):
    """
    Regular grid on the boundary of a cube

    Parameters
    -----------
    dimension : int (default : 2)
        Dimension of the boundary of the cube
    subdivisions : int (default : 2)
        Number of subdivisions of underlying intervals
    """
    segment = [np.linspace(0, 1, subdivisions + 1)]
    endpoints = [np.linspace(0, 1, 2)]
    top_bottom = segment * dimension + endpoints
    vertices = []
    for k in range(dimension + 1):
        vertices.extend(itertools.product(*(
            top_bottom[k:] + top_bottom[:k])))
    return np.array(vertices)


def linked_twist_map(N=1000, 
                     r=4.0, 
                     x0=None, 
                     y0=None):
    """
    Linked twist map

    Parameters
    -----------
    N : int (default : 1000)
        Number of points
    r : float (default : 4.0)
        Linkage parameter
    x0,y0 : float (default : None)
        Initial point. Randomly generated if None. 
    """
    if x0 is None:
        x0=np.random.rand(1)
    if y0 is None:
        y0=np.random.rand(1)
    x = np.zeros(N)
    y = np.zeros(N)
    x[0] = x0
    y[0] = y0
    for i in range(1, N):
        x[i] = np.modf(x[i-1] + r*y[i-1]*(1-y[i-1]))[0]
        y[i] = np.modf(y[i-1] + r*x[i]*(1-x[i]))[0] 
    return np.vstack((x, y)).transpose() 
