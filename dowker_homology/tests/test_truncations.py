import numpy as np
import scipy.sparse as sps
from dowker_homology import dissimilarities
from dowker_homology import truncations
from dowker_homology import datasets
import pytest


def test_truncation():
    # create data
    data_numpy = [np.random.randn(100, 5),
                  datasets.regular_polygon(100),
                  datasets.clifford_torus(100)]
    dd_numpy = [
        dissimilarities.Dissimilarity(dta, dissimilarity='minkowski', p=1)
        for dta in data_numpy]
    data_sparse = [sps.rand(100, 100)]
    for mat in data_sparse:
        mat = mat.tocsr()
        mat[mat > 0] += 1
    dd_sparse = [dissimilarities.Dissimilarity(dta)
                 for dta in data_sparse]

    # translation functions
    def multiplicative_translation_function(time): return 1.5 * time

    def additive_translation_function(time): return time + 0.2

    def affine_translation_function(time): return 1.5 * time + 0.2
    translation_functions = [
        multiplicative_translation_function,
        additive_translation_function,
        affine_translation_function]

    for tf in translation_functions:
        # test numpy truncations
        for dta in dd_numpy:
            trunc = truncations.Truncation(dta, translation_function=tf)
            assert np.all(trunc.DD.X <= trunc.Y)
            assert np.all(trunc.Y <= tf(trunc.DD.X))
        # test sparse truncations
        for dta in dd_sparse:
            trunc = truncations.Truncation(dta, translation_function=tf)
            X = trunc.DD.X.todense()
            Y = trunc.Y.todense()
            assert np.all(X <= Y)
            assert np.all(Y <= tf(X))


def test_truncation_errors():
    with pytest.raises(TypeError):
        truncations.Truncation("a")

    with pytest.raises(TypeError):
        truncations.Truncation(np.random.randn(5, 5))
