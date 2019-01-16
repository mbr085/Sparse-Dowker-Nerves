import numpy as np
import networkx as nx
from dowker_homology import datasets
from scipy import sparse as sps
from scipy.spatial.distance import cdist
from dowker_homology.persistence import Persistence
from dowker_homology.tests.test_examples import product_dict


def numpy_to_sparse_for_dowker(dists):
    dists = np.array(dists, dtype=float)
    dists[dists == 0] = -np.inf
    A = sps.lil_matrix(dists)
    return A


def test_sparse(N=50,
                additive_interleaving=[0.1, 1.5],
                multiplicative_interleaving=[1.001, 3.0]):
    # data
    data_dict = {'clifford': datasets.clifford_torus(N=N),
                 'sphere': datasets.sphere(N=N),
                 'circle': datasets.sphere(N=N, dimension=1),
                 'MVnormal': np.random.randn(N, 5)}

    # parameter set
    parameter_dicts = list(
        product_dict(dataset_name=data_dict.keys(),
                     additive_interleaving=additive_interleaving,
                     multiplicative_interleaving=multiplicative_interleaving,
                     restriction_method=['Sheehy', 'Parent', 'Dowker'],
                     truncation_method=['Sheehy', 'Dowker', 'Canonical'],
                     dissimilarity=['dowker']))

    # check interleaving guarantees
    for parameters in parameter_dicts:
        print(parameters)
        data = data_dict[parameters.pop('dataset_name')]
        dists = cdist(data, data)
        dists_sparse = numpy_to_sparse_for_dowker(dists)

    # calculate persistent homology
        try:
            persistence_Numpy = Persistence(**parameters)
            pers_Numpy = persistence_Numpy.persistent_homology(X=dists)
        except(ValueError):
            break
        # calculate sparse persistent homology
        persistence_Sparse = Persistence(**parameters)
        pers_Sparse = persistence_Sparse.persistent_homology(X=dists_sparse)

        # check that they are the same
        assert all(
            [np.allclose(pers_Numpy[i],
                         pers_Sparse[i])
             for i in range(len(pers_Numpy))])


def test_sparse_base(N=50):
    # data
    data_dict = {'cycle': nx.cycle_graph(n=N),
                 'star': nx.star_graph(n=N),
                 'erdos-reni': nx.gnp_random_graph(n=N, p=0.2)}

    # check interleaving guarantees
    for key, data in data_dict.items():
        print(key)

        # calculate sparse persistent homology
        persistence_Base = Persistence(dissimilarity='dowker',
                                       restriction_method='no_restriction')
        pers_Base = persistence_Base.persistent_homology(X=data)

        persistence_Sparse = Persistence(dissimilarity='dowker',
                                         restriction_method='dowker')
        pers_Sparse = persistence_Sparse.persistent_homology(X=data)

        # check that they are the same
        assert all(
            [np.allclose(pers_Base[i],
                         pers_Sparse[i])
             for i in range(len(pers_Base))])
