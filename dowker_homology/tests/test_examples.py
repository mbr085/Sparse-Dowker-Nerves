import numpy as np
from itertools import product
from gudhi import bottleneck_distance
from dowker_homology import datasets
from dowker_homology.persistence import Persistence


def persistence_tuples(pers, dim, transform=lambda x: x):
    return [(transform(pers_row[0]), transform(pers_row[1]))
            for pers_row in pers[dim]]


def product_dict(**kwargs):
    'Source: https://stackoverflow.com/a/5228294/2591234'
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))


def test_persistent_homology(N=50,
                             additive_interleaving=[0.1, 1.5],
                             multiplicative_interleaving=[1.1, 3.0],
                             tol=np.finfo(np.float32).eps):
    # data
    data_dict = {'clifford': datasets.clifford_torus(N=N),
                 'sphere': datasets.sphere(N=N),
                 'circle': datasets.sphere(N=N, dimension=1),
                 'MVnormal': np.random.randn(N, 5)}

    # baseline persistence
    pers_euclidean = dict()
    pers_ambient = dict()
    for key, data in data_dict.items():
        persistence = Persistence(restriction_method='no_restriction',
                                  dissimilarity='euclidean')
        pers_euclidean[key] = (
            persistence.get_simplex_tree(X=data).persistent_homology())
        persistence = Persistence(restriction_method='no_restriction',
                                  dissimilarity='ambient')
        pers_ambient[key] = (
            persistence.persistent_homology(X=data))

    # parameter set
    parameter_dicts = list(
        product_dict(dataset_name=data_dict.keys(),
                     additive_interleaving=[0],
                     multiplicative_interleaving=multiplicative_interleaving,
                     restriction_method=[
                         'Sheehy',
                         'Parent',
                         'Dowker'],
                     truncation_method=[
                         'Sheehy',
                         'Canonical',
                         'Dowker'],
                     dissimilarity=['euclidean', 'ambient'])) + list(
        product_dict(dataset_name=data_dict.keys(),
                     additive_interleaving=additive_interleaving,
                     multiplicative_interleaving=[1],
                     restriction_method=['Sheehy', 'Parent', 'Dowker'],
                     truncation_method=['Sheehy', 'Dowker', 'Canonical'],
                     dissimilarity=['euclidean', 'ambient']))

    # check interleaving guarantees
    for parameters in parameter_dicts:
        print(parameters)
        dataset_name = parameters.pop('dataset_name')
        data = data_dict[dataset_name]
        if parameters.get('dissimilarity') == 'euclidean':
            pers_base = pers_euclidean[dataset_name]
        if parameters.get('dissimilarity') == 'ambient':
            pers_base = pers_ambient[dataset_name]
        try:
            persistence = Persistence(**parameters)
            pers = (
                persistence.persistent_homology(X=data))
        except(ValueError):
            break

        # calculate 0d bottleneck distance
        bn0 = bottleneck_distance(
            persistence_tuples(pers_base, dim=0),
            persistence_tuples(pers, dim=0))

        if parameters.get('multiplicative_interleaving') <= 1:
            # calculate 1d bottleneck distance
            bn1 = bottleneck_distance(
                persistence_tuples(pers_base, dim=1),
                persistence_tuples(pers, dim=1))

            # check if results are as expected
            assert bn0 < parameters.get('additive', 0) + tol
            assert bn1 < parameters.get('additive', 0) + tol

        if parameters.get('multiplicative_interleaving') > 1:
            # calculate 1d bottleneck distance
            bn1 = bottleneck_distance(
                persistence_tuples(pers_base, dim=1, transform=np.log),
                persistence_tuples(pers, dim=1, transform=np.log))

            # check if results are as expected
            assert bn0 < tol
            assert bn1 < np.log(parameters.get(
                'multiplicative_interleaving')) + tol
