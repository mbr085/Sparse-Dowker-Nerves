import numpy as np
from dowker_homology import datasets
from dowker_homology.persistence import Persistence
from dowker_homology.tests.test_examples import product_dict


def test_restrictions(N=50,
                      additive_interleaving=[0.1, 1.5],
                      multiplicative_interleaving=[1.001, 3.0]):
    # data
    data_dict = {'clifford': datasets.clifford_torus(N=N),
                 'sphere': datasets.sphere(N=N),
                 'circle': datasets.sphere(N=N, dimension=1),
                 # 'MVnormal': np.random.randn(N, 5)
                 }

    # parameter set
    parameter_dicts = list(
        product_dict(dataset_name=data_dict.keys(),
                     additive_interleaving=additive_interleaving,
                     multiplicative_interleaving=multiplicative_interleaving,
                     truncation_method=['Sheehy', 'dowker'],
                     dissimilarity=['euclidean', 'ambient']))

    # check interleaving guarantees
    for parameters in parameter_dicts:
        print(parameters)
        data = data_dict[parameters.pop('dataset_name')]

        # calculate persistent homology for 3 restriction methods
        persistence_Dowker = Persistence(
            **parameters, restriction_method='dowker')
        pers_Dowker = persistence_Dowker.persistent_homology(X=data)

        persistence_Parent = Persistence(
            **parameters, restriction_method='Parent')
        pers_Parent = persistence_Parent.persistent_homology(X=data)

        assert all(
            [np.allclose(pers_Dowker[i],
                         pers_Parent[i])
             for i in range(len(pers_Dowker))])

        try:
            persistence_Sheehy = Persistence(
                **parameters, restriction_method='Sheehy')
            pers_Sheehy = persistence_Sheehy.persistent_homology(X=data)
            assert all(
                [np.allclose(pers_Dowker[i],
                             pers_Sheehy[i])
                 for i in range(len(pers_Dowker))])
        except(ValueError):
            break
