import numpy as np
import networkx as nx
import dowker_homology as dh


def test_pentagon():
    # setup
    data = np.array([[0, 1, 2, 2, 1],
                     [1, 0, 1, 2, 2],
                     [2, 1, 0, 1, 2],
                     [2, 2, 1, 0, 1],
                     [1, 2, 2, 1, 0]], dtype=float)
    edges = [(0, 1, 1), (0, 2, 2), (0, 3, 2), (0, 4, 1),
             (1, 2, 1), (1, 3, 2), (1, 4, 2),
             (2, 3, 1), (2, 4, 2),
             (3, 4, 1)]
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    # expected homology
    homology = [
        np.array([
            [0, np.inf],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1]]),
        np.array([
            [1., 2.]])]
    # homology = np.array([[0, 0, np.inf],
    #                      [0, 0, 1],
    #                      [0, 0, 1],
    #                      [0, 0, 1],
    #                      [0, 0, 1],
    #                      [1, 1, 2]])
    # calculate persistent homology
    persistence = dh.persistence.Persistence(
        restriction_method='no_restriction',
        dissimilarity='dowker')
    assert persistence.maximal_faces(
        X=data) == [
            frozenset(list(range(5)))]

    persistence = dh.persistence.Persistence(
        restriction_method='no_restriction',
        cutoff=2)
    assert persistence.maximal_faces(X=G) == [
        frozenset(list(range(5)))]
    persistence = dh.persistence.Persistence(
        restriction_method='no_restriction',
        dissimilarity='dowker')
    persistence_dgms = persistence.persistent_homology(X=data)
    assert all(
        [np.allclose(persistence_dgms[i],
                     homology[i])
         for i in range(len(persistence_dgms))])
    persistence = dh.persistence.Persistence(
        restriction_method='no_restriction',
        cutoff=3)
    persistence_dgms = persistence.persistent_homology(X=G)
    assert all(
        [np.allclose(persistence_dgms[i],
                     homology[i])
         for i in range(len(persistence_dgms))])
