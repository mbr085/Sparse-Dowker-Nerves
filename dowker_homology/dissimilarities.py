'''
Python routines used to test claims in the papers
*** SDN = Sparse Dowker Nerves ***
*** SFN = Sparse Filtered Nerves ***
Copyright: Applied Topology Group at University of Bergen.
No commercial use of the software is permitted without proper license.
'''
from scipy.spatial import distance
import numpy as np
import networkx as nx
import scipy.sparse as sps
from miniball import Miniball


def distance_function(X, dissimilarity):
    if dissimilarity in ['precomputed', 'dowker']:
        assert X.shape[0] == X.shape[1], 'X must be a square matrix'

        def dist_to_points(point_list):
            return X[point_list]
    else:
        def dist_to_points(point_list):
            return distance.cdist(X[point_list],
                                  X,
                                  metric=dissimilarity)
    return dist_to_points


def get_farthest_insertion_times(X,
                                 dissimilarity,
                                 initial_point=0,
                                 n_samples=None,
                                 resolution=0):
    if n_samples is None:
        n_samples = len(X)
    # initialize
    distances_to_points = distance_function(
        X, dissimilarity)
    farthest_point_list = np.zeros(n_samples, dtype=int)
    farthest_point_list[0] = initial_point
    insertion_times = np.full(len(X), -np.inf)
    insertion_times[initial_point] = np.inf
    cluster_dist = distances_to_points([initial_point])
    cluster_dist = cluster_dist.reshape(-1)
    # update sample
    for point_index in range(1, n_samples):
        if np.all(cluster_dist == -np.inf):
            break
        farthest_point = np.argmax(cluster_dist)
        farthest_point_list[point_index] = farthest_point
        insertion_times[farthest_point] = cluster_dist[farthest_point]
        if insertion_times[farthest_point] <= resolution:
            break
        cluster_dist = np.minimum(
            cluster_dist,
            distances_to_points([farthest_point]).reshape(-1))
    return insertion_times


def Dissimilarity(X,
                  dissimilarity='dowker',
                  resolution=0,
                  n_samples=None,
                  isolated_points=True,
                  initial_point=0,
                  **kwargs):
    """
    This function returns a class storing the dowker
    dissimilarity also used to calculate filtration values.

    Parameters
    ----------
    X : ndarray or spmatrix or Graph
    **kwargs : dict, optional
        Extra arguments depending on class of X. Possible arguments
        are:
        dissimilarity : str (default: 'dowker')
        Used if the X is a numpy array. The default assumes that the
        input is a dowker dissimilarity.  Any valid argument for the
        'metric' of `scipy.spatial.distance.cdist` is is valid. In
        that case, you can also specify additional arguments to the
        `scipy.spatial.distance.cdist` function.

    Examples
    --------
    >>> data = np.random.randn(100, 5)
    >>> dd = Dissimilarity(data, dissimilarity='minkowski', p=1)
    """
    if dissimilarity == 'ambient':
        return DissimilarityAmbient(X,
                                    resolution=resolution,
                                    n_samples=n_samples,
                                    initial_point=initial_point)
    elif dissimilarity == 'alpha':
        return DissimilarityAlpha(X,
                                  dissimilarity=dissimilarity,
                                  resolution=resolution,
                                  n_samples=n_samples,
                                  isolated_points=isolated_points,
                                  initial_point=initial_point,
                                  **kwargs)
    if isinstance(X, np.ndarray):
        return DissimilarityNumpy(X,
                                  dissimilarity=dissimilarity,
                                  resolution=resolution,
                                  n_samples=n_samples,
                                  isolated_points=isolated_points,
                                  initial_point=initial_point,
                                  **kwargs)
    elif isinstance(X, sps.spmatrix):
        return DissimilaritySparse(X.tolil(),
                                   isolated_points=isolated_points)
    elif isinstance(X, nx.classes.graph.Graph):
        cutoff = kwargs.pop('cutoff', np.inf)
        n = X.number_of_nodes()
        C = sps.lil_matrix((n, n))
        for x in nx.all_pairs_dijkstra_path_length(X, cutoff=cutoff):
            for j, weight in x[1].items():
                if weight == 0:
                    C[x[0], j] = -np.inf
                else:
                    C[x[0], j] = weight
        return DissimilaritySparse(C,
                                   isolated_points=isolated_points)
    else:
        raise TypeError("input must be an ndarray, spmatrix or Graph")


class DissimilarityNumpy:
    """
    Parameters
    ----------
    X : ndarray
    dissimilarity : str (default: 'dowker')
    **kwargs : dict, optional
        Additional arguments passed on to the
        `scipy.spatial.distance.cdist` function.
    """

    def __init__(self, X,
                 dissimilarity,
                 resolution=0,
                 n_samples=None,
                 isolated_points=True,
                 initial_point=0,
                 **kwargs):
        self.dissimilarity = dissimilarity
        self.resolution = resolution
        self.n_samples = n_samples
        self.isolated_points = isolated_points
        self.initial_point = initial_point
        assert(isinstance(X, np.ndarray))
        if self.dissimilarity in distance._METRIC_ALIAS.keys():
            self.set_dissimilarity_from_point_cloud(
                X, metric=self.dissimilarity, **kwargs)
        elif self.dissimilarity == 'metric':
            self.set_dissimilarity_from_metric(X)
        else:
            self.X = X
            self.cover_radius = 0
        self.X_orig = self.X
        self.shape = self.X.shape

    def set_dissimilarity_from_point_cloud(self, X, metric, **kwargs):
        self.cover_radius = 0
        coords = X
        insertion_times = get_farthest_insertion_times(
            X=coords,
            dissimilarity=metric,
            initial_point=self.initial_point,
            n_samples=self.n_samples,
            resolution=self.resolution)
        self.farthest_point_list = np.flatnonzero(insertion_times > -np.inf)
        if len(self.farthest_point_list) < len(insertion_times):
            self.cover_radius = np.min(
                insertion_times[self.farthest_point_list])
        self.X = distance.cdist(
            coords[self.farthest_point_list], coords,
            metric=metric, **kwargs)
        self.coords = coords[self.farthest_point_list]

    def set_dissimilarity_from_metric(self, X):
        self.cover_radius = 0
        insertion_times = get_farthest_insertion_times(
            X=X,
            dissimilarity='precomputed',
            initial_point=self.initial_point,
            n_samples=self.n_samples,
            resolution=self.resolution)
        self.farthest_point_list = np.flatnonzero(insertion_times > -np.inf)
        if len(self.farthest_point_list) < len(insertion_times):
            self.cover_radius = np.min(
                insertion_times[self.farthest_point_list])
        self.X = X[self.farthest_point_list]
        # [:, self.farthest_point_list]

    def len(self):
        return self.X.shape[0]

    def max(self):
        return np.max(self.X[self.X < np.inf])

    def subsample(self, point_list):
        self.X = self.X[point_list]
        self.shape = self.X.shape

    def filtration_value(
            self,
            node_indices):
        '''Filtration value of a simplex'''
        if len(node_indices) == 1:
            if self.isolated_points:
                return np.min(self.X[node_indices[0]])
            else:
                return np.inf
        return np.min(np.max(self.X[node_indices], axis=0))

    def witness_face(self, w):
        return frozenset(np.flatnonzero(self.X[:, w] < np.inf))

    def get_neares_point_list(self):
        self.nearest_point_list = np.empty(self.X_orig.shape[0])
        for idx, x in enumerate(self.X_orig):
            self.nearest_point_list[idx] = np.argmin(
                np.min(np.maximum(x, self.X), axis=1))

    def expand_cluster(self, cluster):
        return np.flatnonzero(np.isin(self.nearest_point_list, cluster))


class DissimilarityAmbient(DissimilarityNumpy):

    def __init__(self, X,
                 resolution=0,
                 n_samples=None,
                 # isolated_points=True,
                 initial_point=0,
                 **kwargs):
        self.resolution = resolution
        self.n_samples = n_samples
        self.initial_point = initial_point
        assert(isinstance(X, np.ndarray))
        self.set_dissimilarity_from_point_cloud(X=X, metric='euclidean')
        self.shape = self.X.shape
        self.X_orig = X

    def get_neares_point_list(self):
        self.nearest_point_list = None

    def subsample(self, point_list):
        super().subsample(point_list)
        self.coords = self.coords[point_list]

    def filtration_value(
            self,
            node_indices):
        '''Filtration value of a simplex'''
        if len(node_indices) == 1:
            simplex_filtration_value = 0.0
        else:
            mb = Miniball(self.coords[node_indices])
            simplex_filtration_value = np.sqrt(mb.squared_radius())

        return simplex_filtration_value


class DissimilarityAlpha(DissimilarityNumpy):

    def __init__(self, X,
                 dissimilarity,
                 resolution=0,
                 n_samples=None,
                 isolated_points=True,
                 initial_point=0):
        super().__init__(X=X,
                         dissimilarity=dissimilarity,
                         resolution=resolution,
                         n_samples=n_samples,
                         isolated_points=isolated_points,
                         initial_point=initial_point)
        assert(isinstance(X, np.ndarray))
        self.set_dissimilarity_from_point_cloud(X, metric='euclidean')
        self.shape = tuple([len(self.coords)] * 2)
        mb = Miniball(self.coords)
        self.radius = np.sqrt(mb.squared_radius())
        self.X = np.array([[0]])

    def len(self):
        return self.coords.shape[0]

    def max(self):
        return 2 * self.radius

    def subsample(self, point_list):
        self.coords = self.coords[point_list]

    def filtration_value(
            self,
            node_indices):
        '''Filtration value of a simplex'''
        if len(node_indices) == 1:
            if self.isolated_points:
                simplex_filtration_value = 0.0
            else:
                simplex_filtration_value = np.inf
        else:
            mb = Miniball(self.coords[node_indices])
            simplex_filtration_value = np.sqrt(mb.squared_radius())

        return simplex_filtration_value


class DissimilaritySparse(DissimilarityNumpy):

    def __init__(self,
                 X,
                 isolated_points=True):
        self.isolated_points = isolated_points
        self.shape = X.shape
        self.X = X
        self.X_orig = self.X
        self.W = self.X.transpose()
        self.cover_radius = 0

    def get_neares_point_list(self):
        self.nearest_point_list = None

    def max(self):
        return np.max([item for row_data in self.X.data for
                       item in row_data if item < np.inf])

    def witness_face(self, w):
        return frozenset(self.W.rows[w])

    def filtration_value(
            self,
            node_indices):
        '''Filtration value of a simplex'''
        if len(node_indices) == 1:
            if len(self.X.data[node_indices[0]]) == 0:
                return np.inf
            if self.isolated_points:
                return max(0., min(self.X.data[node_indices[0]]))
            else:
                return np.inf
        common_witnesses = list(set.intersection(
            *(set(row) for row in self.X.rows[node_indices])))
        if not common_witnesses:
            return np.inf
        distances_to_witnesses = np.zeros(self.X.shape[1])
        for index in node_indices:
            distances_to_witnesses[self.X.rows[index]] = np.maximum(
                distances_to_witnesses[self.X.rows[index]],
                self.X.data[index])
        return max(0., np.min(distances_to_witnesses[common_witnesses]))
