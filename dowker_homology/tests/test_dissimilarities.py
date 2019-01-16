import numpy as np
import networkx as nx
import scipy.sparse as sps
from dowker_homology import dissimilarities
import pytest


def test_dowker_dissimilarity():
    nrow = 50
    ncol = 5

    # generate random data
    data_numpy = np.random.randn(nrow, ncol)
    data_sparse = sps.rand(nrow, nrow)
    data_graph = nx.erdos_renyi_graph(nrow, 0.25)

    dd_numpy = dissimilarities.Dissimilarity(
        data_numpy, dissimilarity='minkowski', p=1)
    dd_sparse = dissimilarities.Dissimilarity(data_sparse)
    dd_graph = dissimilarities.Dissimilarity(data_graph)

    for dd in [dd_numpy, dd_sparse, dd_graph]:
        # correct length
        assert dd.len() == nrow

        # maximum is float
        assert isinstance(dd.max(), float)

        # filtration_value returns floats
        assert np.all([isinstance(dd.filtration_value([i]), float)
                       for i in range(50)])
        assert np.all([isinstance(dd.filtration_value([i, j]), float)
                       for i in range(50) for j in range(50)])


def test_dowker_dissimilarity_from_unknown():
    with pytest.raises(TypeError):
        dissimilarities.Dissimilarity("a")

    with pytest.raises(TypeError):
        dissimilarities.Dissimilarity((2, 4))
