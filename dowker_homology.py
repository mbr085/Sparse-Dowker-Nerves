'''
Python routines used to test claims in the paper
*** SDN = Sparse Dowker Nerves ***
Copyright: Applied Topology Group at University of Bergen.
No commercial use of the software is permitted without proper license.
'''
import numpy as np
import igraph as ig
import itertools
import phat
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special
from scipy.spatial.distance import cdist
import miniball


def get_farthest_point_sample(distance_matrix,
                              n_samples=None,
                              initial_point=0):
    '''Farthest point sample

    Parameters
    -----------
    distance_matrix : ndarray, shape = (n_data, n_data)
        Full distance matrix
    n_samples : int, optional (default=n_data)
        Number of farthest points to sample
    initial_point: int, optional (default=0)
        First point in sample

    Returns
    -----------
    farthest_point : ndarray
        Indices of farthest points, shape = (n_samples, )
    cover_radii : ndarray
        Cover radii of all partial samples (SDN Def 6.9)
    '''
    # initialize
    if n_samples is None:
        n_samples = len(distance_matrix)
    farthest_point_list = np.zeros(n_samples, dtype=int)
    farthest_point_list[0] = initial_point
    cover_radii = np.zeros(n_samples)
    cluster_dist = distance_matrix[initial_point]
    # update sample
    for point_index in range(1, n_samples):
        farthest_point = np.argmax(cluster_dist)
        farthest_point_list[point_index] = farthest_point
        cover_radii[point_index - 1] = cluster_dist[farthest_point]
        cluster_dist = np.minimum(
            cluster_dist,
            distance_matrix[farthest_point])
    cover_radii[n_samples - 1] = np.max(cluster_dist)
    cover_radius = cover_radii[-1]
    cover_radii[1:] = cover_radii[:-1]
    cover_radii[0] = np.inf
    return farthest_point_list, cover_radii, cover_radius


def get_dowker_nerve_edge_matrix(dowker_dissimilarity_matrix):
    '''Weight matrix for the one-skeleton of the Dowker nerve
        (SDN Def 6.8)

    Parameters
    -----------
    dowker_dissimilarity_matrix : ndarray, shape = (n_samples, n_data)
        Distances between landmarks and witnesses (SDN Def 6.4)

    Returns
    -----------
    dowker_nerve_edge_matrix : ndarray, shape = (n_samples, n_samples)
        Weight matrix for the one-skeleton of the Dowker nerve

    '''
    dowker_nerve_edge_matrix = np.zeros((
        dowker_dissimilarity_matrix.shape[0],
        dowker_dissimilarity_matrix.shape[0]))
    for _i in range(len(dowker_dissimilarity_matrix)):
        _i_values = dowker_dissimilarity_matrix[_i]
        for _j in range(_i):
            _j_values = dowker_dissimilarity_matrix[_j]
            dowker_nerve_edge_matrix[_i, _j] \
                = dowker_nerve_edge_matrix[_j, _i] \
                = np.min(np.maximum(_i_values, _j_values))
    return dowker_nerve_edge_matrix


def get_truncated_ambient_dowker_matrix(dowker_matrix, truncation_times):
    '''Truncation of Dowker Dissimilarity Matrix with
    respect to Euclidean space

    Parameters
    -----------
    dowker_matrix : ndarray, shape = (n_samples, n_data)

    truncation_times : ndarray, shape = (n_samples, )
        Truncation time of vertices

    Returns
    -----------
    truncated_matrix : ndarray,
                              shape = (n_samples, n_data)
        Truncation of Dowker Dissimilarity Matirx

    '''
    _n, _N = dowker_matrix.shape
    if (_n != len(truncation_times) or _n != _N):
        raise Exception('Wrong matrix dimensions')
    truncation = np.minimum(
        np.repeat(truncation_times.reshape(1, _n), _n, axis=0),
        np.repeat(truncation_times.reshape(_n, 1), _n, axis=1))
    truncation_sum = (
        np.repeat(truncation_times.reshape(1, _n), _n, axis=0)
        + np.repeat(truncation_times.reshape(_n, 1), _n, axis=1))
    truncated_matrix = np.zeros(dowker_matrix.shape)
    selected_slice = truncation >= 0.5 * dowker_matrix
    truncated_matrix[selected_slice] = 0.5 * dowker_matrix[selected_slice]
    selected_slice = np.logical_and(truncation < 0.5 * dowker_matrix,
                                    truncation_sum >= dowker_matrix)
    truncated_matrix[selected_slice] = dowker_matrix[selected_slice] - \
        truncation[selected_slice]
    truncated_matrix[truncation_sum < dowker_matrix] = np.inf
    return truncated_matrix


def get_truncated_dowker_matrix(dowker_matrix, truncation_times):
    '''Truncation of Dowker Dissimilarity Matrix

    Parameters
    -----------
    dowker_matrix : ndarray, shape = (n_samples, n_data)

    truncation_times : ndarray, shape = (n_samples, )
        Truncation time of vertices

    Returns
    -----------
    truncated_dowker_matrix : ndarray,
                              shape = (n_samples, n_data)
        Truncation of Dowker Dissimilarity Matirx

    '''
    if dowker_matrix.shape[0] != len(truncation_times):
        raise Exception("Wrong matrix dimensions")
    truncation = np.repeat(
        truncation_times.reshape(
            len(truncation_times), 1),
        dowker_matrix.shape[1],
        axis=1)
    truncated_dowker_matrix = dowker_matrix
    truncated_dowker_matrix[dowker_matrix > truncation] = np.inf
    return truncated_dowker_matrix


def get_sparse_dowker_nerve_edge_matrix(dowker_nerve_edge_matrix,
                                        death_times):
    '''Weight matrix for the one-skeleton of the sparse Dowker nerve
        (SDN Def 9.1)

    Parameters
    -----------
    dowker_nerve_edge_matrix : ndarray, shape = (n_samples, n_samples)
        Weight matrix for the one-skeleton of the Dowker nerve
    death_times : ndarray, shape = (n_samples, )
        Death time of voronoi cells
    cover_radius : float
        Cover radius of sample in data

    Returns
    -----------
    sparse_dowker_nerve_edge_matrix : ndarray,
                                      shape = (n_samples, n_samples)
        Weight matrix for the one-skeleton of the sparse Dowker nerve

    '''
    sparse_dowker_nerve_edge_matrix = dowker_nerve_edge_matrix
    sparse_edges = np.logical_and(dowker_nerve_edge_matrix
                                  <= death_times.reshape(len(death_times), 1),
                                  dowker_nerve_edge_matrix
                                  <= death_times.reshape(1, len(death_times)))
    sparse_dowker_nerve_edge_matrix[
        ~sparse_edges] = np.inf
    return sparse_dowker_nerve_edge_matrix


def sort_filtered_nerve(nerve_of_cover, nerve_values):
    # sort filtered nerve
    filtered_nerve = list(zip(nerve_of_cover, nerve_values))
    filtered_nerve.sort(key=lambda obj: len(obj[0]))
    filtered_nerve.sort(key=lambda obj: obj[1])
    filtered_nerve, nerve_values = list(zip(*filtered_nerve))
    return filtered_nerve, nerve_values


def powerset(some_set, max_card):
    return itertools.chain(*[
        list(itertools.combinations(some_set, dim + 1))
        for dim in range(max_card)])


def get_intrinsic_filtration_value(node_indices,
                                   dowker_nerve_edge_matrix,
                                   dowker_dissimilarity_matrix,
                                   restriction_values,
                                   max_filtration_value):
    '''Filtration value of a simplex
    (SDN Def 9.1)

    Parameters
    -----------
    node_indices : tuple
        Indices of simplex vertices
    dowker_nerve_edge_matrix : ndarray, shape = (n_samples, n_samples)
        Weight matrix for the one-skeleton of the Dowker nerve
    dowker_dissimilarity_matrix : ndarray, shape = (n_samples, n_data)
        Distances between landmarks and witnesses
    max_filtration_value : float
        Maximal filtration value to include in the nerve

    Returns
    -----------
    filtration_value : float
        Filtration value
    '''
    if len(node_indices) == 1:
        simplex_filtration_value = 0.0
    elif len(node_indices) == 2:
        simplex_filtration_value = dowker_nerve_edge_matrix[
            node_indices[0], node_indices[1]]
    else:
        dists_to_simplex = np.max(
            dowker_dissimilarity_matrix[node_indices, ], axis=0)
        simplex_restriction_value = np.min(restriction_values[node_indices])
        simplex_filtration_value = np.min(dists_to_simplex)
        if simplex_filtration_value > simplex_restriction_value:
            simplex_filtration_value = np.inf
    if 0 in node_indices:
        simplex_filtration_value = np.min((
            simplex_filtration_value,
            max_filtration_value))
    return simplex_filtration_value


def get_ambient_filtration_value(node_indices,
                                 data,
                                 dowker_nerve_edge_matrix,
                                 truncation_values,
                                 restriction_values,
                                 max_filtration_value,
                                 method):
    '''Filtration value of a simplex in the situation
    where the witness set is euclidean space

    Parameters
    -----------
    node_indices : tuple
        Indices of simplex vertices
    dowker_nerve_edge_matrix : ndarray, shape = (n_samples, n_samples)
        Distances between voroni cells
    max_filtration_value : float
        Maximal filtration value to include in the nerve

    Returns
    -----------
    filtration_value : float
        Filtration value
    '''
    if len(node_indices) == 1:
        simplex_filtration_value = 0.0
    elif len(node_indices) == 2:
        simplex_filtration_value = dowker_nerve_edge_matrix[
            node_indices[0], node_indices[1]]
    else:
        if method == 'ambient':
            mb = miniball.Miniball(data[[node_indices]])
            radius = np.sqrt(mb.squared_radius())
        elif method == 'rips':
            radius = 0
        simplex_filtration_value = np.max((
            radius,
            np.max(dowker_nerve_edge_matrix[
                node_indices, :][:, node_indices])))
        simplex_restriction_value = np.min(restriction_values[[node_indices]])
        if simplex_filtration_value > simplex_restriction_value:
            simplex_filtration_value = np.inf
    if 0 in node_indices:
        simplex_filtration_value = np.min((
            simplex_filtration_value,
            2 * max_filtration_value))

    return simplex_filtration_value


def get_filtered_nerve(dowker_nerve_edge_matrix,
                       dowker_dissimilarity_matrix,
                       restriction_values=None,
                       truncation_values=None,
                       data=None,
                       dimension=1,
                       max_filtration_value=np.inf,
                       verbose=False,
                       method='intrinsic'):
    '''Filtered nerve

    WARNING: dowker_nerve_edge_matrix is destroyed
    make a copy first if you need to keep it.

    Parameters
    -----------
    dowker_nerve_edge_matrix : ndarray, shape = (n_samples, n_samples)
        Distances between voroni cells
    dowker_dissimilarity_matrix : ndarray, shape = (n_samples, n_data)
        Distances between voroni cells and data
    truncation_values : ndarray, shape = (n_samples,)
        Value where each sample gets truncated.
    data : ndarray, shape = (n_samples, d)
        Coordinate matrix of euclidean data points.
    dimension : int, optional (default = 1)
        Maxmial homology dimension
    max_filtration_value : float, optional (default = np.inf)
        Maximal filtration value to include in the nerve
    verbose : bool, optional (defalut = False)
        Print information about size of nerve

    Returns
    -----------
    filtered_nerve : tuple of tuples
        Tuple of tuples of indices of vertices of faces in the nerve
    nerve_filtration_values: ndarray, shape = (len(filtered_nerve), )
        Filtration values of faces in the filtered nerve.
    '''
    # convert igraph graph
    edge_weight_matrix = dowker_nerve_edge_matrix
    edge_weight_matrix[~np.isfinite(edge_weight_matrix)] = 0

    if verbose:
        print('Unreduced nerve has cardinality ' + str(
            np.sum([scipy.special.binom(
                    len(edge_weight_matrix), _i)
                    for _i in range(1, dimension + 3)])))

    nerve_of_cover = get_cliques(edge_weight_matrix, dimension)

    if verbose:
        print('Clique reduced nerve of cover has cardinality ' +
              str(len(nerve_of_cover)))

    # get filtration values
    if method == 'intrinsic':
        sparse_filtration_values = np.array([
            get_intrinsic_filtration_value(list(simplex),
                                           dowker_nerve_edge_matrix,
                                           dowker_dissimilarity_matrix,
                                           restriction_values,
                                           max_filtration_value)
            for simplex in nerve_of_cover])
    elif method in ['ambient', 'rips']:
        sparse_filtration_values = np.array([
            get_ambient_filtration_value(simplex,
                                         data,
                                         dowker_nerve_edge_matrix,
                                         restriction_values,
                                         truncation_values,
                                         max_filtration_value,
                                         method)
            for simplex in nerve_of_cover])
    finite_indices = np.isfinite(sparse_filtration_values)
    # sparse_filtration_values[finite_indices] = max_filtration_value

    # remove infinite values
    sparse_filtration_values = sparse_filtration_values[finite_indices]
    nerve_of_cover = nerve_of_cover[finite_indices]

    if verbose:
        print('Sparse nerve of cover has cardinality '
              + str(len(nerve_of_cover)))

    # sort filtered nerve
    filtered_nerve, nerve_filtration_values = sort_filtered_nerve(
        nerve_of_cover, sparse_filtration_values)

    return filtered_nerve, nerve_filtration_values


def get_cliques(edge_weight_matrix, dimension):
    g = ig.Graph.Adjacency((edge_weight_matrix > 0).tolist(),
                           mode=ig.ADJ_UNDIRECTED)
    g.es['weight'] = edge_weight_matrix[edge_weight_matrix.nonzero()]
    maximal_cliques = g.maximal_cliques()
    cliques = (frozenset((tuple(sorted(clique))
                          for clique in
                          powerset(maximal_clique, dimension + 2)))
               for maximal_clique in maximal_cliques)
    return np.array(list(frozenset(
        itertools.chain.from_iterable(cliques))))


def get_boundary_matrix(filtered_complex):
    di = dict((filtered_complex[k], k)
              for k in range(len(filtered_complex)))
    b_matrix = []
    for i in range(len(filtered_complex)):
        ni = len(filtered_complex[i])
        if (ni == 1):
            b_matrix.append(tuple([0, []]))
            continue
        boundary = []
        for a in range(ni):
            dimma = list(range(ni))
            dimma.pop(a)
            da = tuple([filtered_complex[i][k] for k in dimma])
            boundary.append(di[da])
        b_matrix.append(tuple(
            [ni - 1, list(sorted(boundary))]))
    return b_matrix


def persistent_homology(filtered_nerve, nerve_filtration_values):
    boundary_matrix = phat.boundary_matrix(
        representation=phat.representations.vector_vector)
    b_matrix = get_boundary_matrix(filtered_nerve)
    boundary_matrix.columns = b_matrix
    # compute persistence
    pairs = boundary_matrix.compute_persistence_pairs()
    pairs.sort()

    # return persistentce pairs
    pers = [[b_matrix[p[0]][0], nerve_filtration_values[p[0]],
             nerve_filtration_values[p[1]]] for
            p in pairs if p[0] < p[1]]
    pers.sort(key=lambda pair: pair[1] - pair[2])

    return np.array(pers)


def interleaving_function(death_times, interleaving_factor):
    return interleaving_factor * death_times


def dowker_persistent_homology(coords,
                               n_samples=None,
                               interleaving=1,
                               dimension=1,
                               method="dowker",
                               verbose=False):
    '''Dowker persistent homology

    Parameters
    -----------
    coords : ndarray, shape = (n_data, d)
        Coordinate matrix of euclidean data points
    n_samples : int, optional (default=n_data)
        Number of farthest points to sample
    interleaving : float, optional (default = 1)
        Multiplicative interleaving
    dimension : int, optional (default = 1)
        Maxmial homology dimension
    method : string, optional (default = "dowker")
        Either "dowker" or "sheehy"
    verbose : bool, optional (defalut = False)
        Print information about size of nerve

    Returns
    -----------
    homology : ndarray (n_pers, 3)
        Persistent homology with coordinates dimension, birth, death
    '''
    # defaults
    if n_samples is None:
        n_samples = len(coords)
    # interleaving
    interleaving_factor = interleaving / (interleaving - 1)
    # farthest point sampling
    distance_matrix = cdist(coords, coords, metric='euclidean')
    farthest_points, death_times, cover_radius = (
        get_farthest_point_sample(distance_matrix,
                                  n_samples=n_samples))
    # sample distance matrix
    sample_data_distance_matrix = distance_matrix[
        farthest_points][:, farthest_points]
    # truncation
    truncation_times = interleaving_function(
        death_times, interleaving_factor)
    # parent points
    t_times = truncation_times[1:]
    f_points = farthest_points[1:]
    parent_point_list = np.tril(
        np.repeat(t_times.reshape(len(t_times), 1),
                  len(t_times), axis=1)
        + distance_matrix[f_points, :][:, f_points]
        - np.repeat(t_times.reshape(1, len(t_times)),
                    len(t_times), axis=0) <= 0, k=-1)
    parent_point_list = np.max(
        parent_point_list
        * (np.arange(parent_point_list.shape[0]) + 1),
        axis=1)
    parent_point_list = np.append(0, parent_point_list)
    # restriction times
    if method == "dowker":
        restriction_times = interleaving_function(
            death_times[parent_point_list], interleaving_factor)
    if method == "sheehy":
        restriction_times = death_times * interleaving * \
            interleaving / (interleaving - 1)
    # truncate
    truncated_dowker_matrix = get_truncated_ambient_dowker_matrix(
        sample_data_distance_matrix, truncation_times)
    # sparsify
    sparse_nerve_edge_matrix = get_sparse_dowker_nerve_edge_matrix(
        truncated_dowker_matrix, restriction_times)
    # del truncated_dowker_matrix
    # filtered nerves
    filtered_nerve, nerve_values = get_filtered_nerve(
        sparse_nerve_edge_matrix,
        sample_data_distance_matrix,
        restriction_values=restriction_times,
        truncation_values=truncation_times,
        data=coords[farthest_points],
        dimension=dimension,
        verbose=verbose,
        method='ambient')
    # persistent homology
    homology = persistent_homology(filtered_nerve, nerve_values)
    # additional outputs for plotting
    max_filtration_value = interleaving_function(
        death_times[1], interleaving_factor)
    # return
    return homology, cover_radius, max_filtration_value


def plot_persistence(pers, s=0.5,
                     interleaving=1,
                     cover_radius=0,
                     max_filtration_value=None,
                     title="Persistence diagram",
                     ticks=None):
    if len(pers) == 0:
        raise Exception("Cannot plot empty persistence diagram.")
    # extract data
    pers = pd.DataFrame(dict(dim=pers[:, 0].astype("int"),
                             birth=pers[:, 1],
                             death=pers[:, 2]))
    pers = pers.ix[pers.ix[:, "death"] != pers.ix[:, "birth"], :]
    groups = pers.groupby("dim")
    # plot
    fig, ax = plt.subplots()
    for name, group in groups:
        ax.plot(group.birth, group.death, marker="o",
                linestyle="", ms=2, label=name)
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_title(title)
    fig.tight_layout()
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    # diagonal
    t = np.array([0, np.max([plt.xlim()[1], plt.ylim()[1]])])
    ax.plot(t, t, color="black", linewidth=1.0)

    # interleaving diagonal
    if interleaving == 1:
        x_points = np.array([0, max_filtration_value - cover_radius])
        y_points = cover_radius + x_points
    elif (cover_radius / (interleaving - 1)
          > max_filtration_value / interleaving):
        x_points = np.array([
            0,
            max_filtration_value])
        y_points = np.array([
            cover_radius,
            max_filtration_value + cover_radius])
    else:
        x_points = np.array([
            0,
            cover_radius / (interleaving - 1),
            max_filtration_value / interleaving])
        y_points = cover_radius + np.array([
            cover_radius,
            interleaving * cover_radius / (interleaving - 1),
            max_filtration_value])

    ax.plot(x_points, y_points, color="black", linewidth=1.0)

    if ticks is not None:
        ax.xaxis.set_ticks(ticks)
        ax.yaxis.set_ticks(ticks)
        ax.grid()
    ax.legend(loc="lower right")

    return plt
