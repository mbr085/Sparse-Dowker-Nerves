import numpy as np
import pandas as pd
import itertools
import collections
import operator
import functools
from scipy.spatial.distance import cdist
from miniball import Miniball
from pynverse import inversefunc


def get_slope(parent_point_list, death_times):
    forest = get_parent_forest(parent_point_list)
    slope = np.zeros(len(parent_point_list), dtype=bool)
    for index in range(len(parent_point_list)):
        if len(forest[index]) > 0:
            slope[index] = (death_times[index] >
                            np.max(death_times[
                                list(forest[index])]))
        else:
            slope[index] = True
    return slope


def get_parent_forest(parent_point_list):
    forest = {index: set() for index in
              range(len(parent_point_list))}
    for index, parent in enumerate(parent_point_list):
        if index != parent:
            forest[parent].add(index)
    return forest


def get_parent_point_list(death_times, cover_matrix):
    death_order = np.argsort(death_times)[::-1]
    qsets = []
    parent_point_list = np.zeros(len(death_times), dtype=int)
    for index, point in enumerate(death_order):
        qset = death_times[point] == cover_matrix[
            point, death_order[: index]]
        if np.sum(qset):
            parent_point_list[point] = death_order[
                np.min(np.arange(len(qset))[qset])]
        elif index > 0:
            ldeath_times = np.min(
                cover_matrix[point, death_order[: index]])
            qset = ldeath_times == cover_matrix[
                point, death_order[: index]]
            if np.sum(qset):
                parent_point_list[point] = death_order[
                    np.min(np.arange(len(qset))[qset])]
            else:
                parent_point_list[point] = point
        else:
            parent_point_list[point] = point
            qsets.append(death_order[: index][qset])
    return parent_point_list


def translation_function_from_interleaving(
        additive_interleaving,
        multiplicative_interleaving):
    def translation_function(time):
        return multiplicative_interleaving * time + additive_interleaving
    return translation_function


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


def invert_permutation(p):
    '''The argument p is assumed to be some permutation of
    0, 1, ..., len(p)-1.
    Returns an array s, where s[i] gives the index of i in p.
    '''
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return s


def get_nearest_insertion_times(X,
                                dissimilarity,
                                n_samples=None,
                                resolution=0):
    if n_samples is None:
        n_samples = len(X)
    # initialize
    distances_to_points = X.copy()
    np.fill_diagonal(distances_to_points, np.inf)
    insertion_times = np.zeros(n_samples)
    unsampled_points = list(range(n_samples))
    for point_index in range(n_samples):
        nearest_point = np.argmin(distances_to_points)
        nearest_point = np.unravel_index(
            nearest_point, distances_to_points.shape)
        insertion_time = distances_to_points[nearest_point]
        remaining_points = list(range(len(distances_to_points)))
        remaining_points.pop(nearest_point[1])
        distances_to_points = distances_to_points[
            remaining_points][:, remaining_points]
        nearest_point = unsampled_points.pop(nearest_point[1])
        insertion_times[nearest_point] = insertion_time
    # insertion_order = np.argsort(insertion_times)[::-1]
    # distances_to_points = X[insertion_order, :][:, insertion_order].copy()
    # distances_to_points = np.triu(distances_to_points, 1)
    # new_insertion_times = np.empty_like(insertion_times)
    # new_insertion_times[0] = np.inf
    # for point_index in range(1, n_samples):
    #     new_insertion_times[point_index] = np.max(np.min(
    #         distances_to_points[:point_index, point_index:],
    #         axis=0))
    # insertion_times = new_insertion_times[invert_permutation(insertion_order)]
    return insertion_times


def get_truncated_dowker_matrix(X,
                                truncation_times,
                                max_filtration_value,
                                truncation_list=None,
                                dissimilarity=None):
    if truncation_list is None:
        truncation_list = range(len(X))
    if dissimilarity is None:
        dissimilarity = 'precomputed'
    dowker_matrix = distance_function(
        X, dissimilarity)(truncation_list)
    truncation = np.repeat(
        truncation_times.reshape(
            len(truncation_times), 1),
        dowker_matrix.shape[1],
        axis=1)
    truncated_dowker_matrix = np.full(dowker_matrix.shape,
                                      max_filtration_value)
    finite_indices = dowker_matrix <= truncation
    truncated_dowker_matrix[
        finite_indices] = dowker_matrix[
            finite_indices]
    return truncated_dowker_matrix


def get_cover_matrix(X, Y):
    cover_matrix = np.zeros((len(X), len(X)))
    for w in range(X.shape[1]):
        Xw = X[:, w].reshape((len(X), 1))
        Yw = Y[:, w].reshape((1, len(X)))
        cover_matrix = np.maximum(
            Xw * (Xw > Yw), cover_matrix)
    return cover_matrix.transpose()


def get_truncation_times(X, translation_function):
    times = np.unique(X)
    times = times[times > 0]
    times = np.append(times, np.inf)
    T = np.zeros(len(X))
    T[0] = np.inf
    for l in range(len(X)):
        for time in times:
            witnesses = np.flatnonzero(X[l] < time)
            neighbors = np.flatnonzero(np.max(X[:, witnesses], axis=1) <
                                       translation_function(time))
            lowest_neighbor = neighbors[np.argmin(
                np.max(X[neighbors, :][:, witnesses], axis=1) - T[neighbors])]
            farthest_witness = np.argmax(
                X[lowest_neighbor, witnesses] - T[lowest_neighbor])
            if X[lowest_neighbor, farthest_witness] - T[lowest_neighbor] > 0:
                # T[lowest_neighbor] = X[lowest_neighbor, farthest_witness]
                T[l] = time
    return T


def get_intrinsic_filtration_value(node_indices,
                                   truncated_dissimilarity_matrix):
    '''Filtration value of a simplex
    (SDN Def 9.1)

    Parameters
    -----------
    node_indices : tuple
        Indices of simplex vertices
    dowker_dissimilarity_matrix : ndarray, shape = (n_samples, n_data)
        Distances between landmarks and witnesses

    Returns
    -----------
    filtration_value : float
        Filtration value
    '''
    if len(node_indices) == 1:
        return np.min(truncated_dissimilarity_matrix[node_indices[0]])
    return np.min(
        np.max(truncated_dissimilarity_matrix[node_indices], axis=0))


def get_ambient_filtration_value(node_indices,
                                 farthest_point_sample,
                                 coords,
                                 restriction_times):
    '''Filtration value of a simplex in the situation
    where the witness set is euclidean space

    Parameters
    -----------
    node_indices : tuple
        Indices of simplex vertices
    coords : ndarray
        Cartesian coordinates

    Returns
    -----------
    simplex_filtration_value : float
        Filtration value
    '''
    if len(node_indices) == 1:
        simplex_filtration_value = 0.0
    else:
        mb = Miniball(coords[
            farthest_point_sample][node_indices])
        simplex_filtration_value = np.sqrt(mb.squared_radius())
        if simplex_filtration_value > np.min(restriction_times[node_indices]):
            simplex_filtration_value = np.inf

    return simplex_filtration_value


def get_nerve_from_maximal_faces(maximal_faces,
                                 dimension):
    '''Returns nerve given a list of maximal faces

    Parameters
    -----------
    maximal_face : list of frozensets of integers
    dimension : int
        Maximal homology dimension

    Returns
    -----------
    nerve : ndarray of lists of integers
        List of all simplices up to desired dimension
    '''
    # This needs to be optimized.
    # First we find the maximal faces of the dimension + 1 skeleton.
    # Then we fill in the rest of the faces.
    faces = ((frozenset(face)
              for face in
              powerset(maximal_face,
                       max_card=dimension + 2,
                       min_card=min(len(maximal_face),
                                    dimension + 2)))
             for maximal_face in maximal_faces)
    faces = frozenset(
        itertools.chain.from_iterable(faces))
    faces = ((frozenset(face)
              for face in
              powerset(maximal_face,
                       max_card=dimension + 2))
             for maximal_face in faces)
    return np.array(list(frozenset(
        itertools.chain.from_iterable(faces))))


def is_power_of_two(n):
    """Returns True iff n is a power of two.  Assumes n > 0."""
    return (n & (n - 1)) == 0


def eliminate_subsets(sequence_of_sets):
    """I did not write this. Far too clever for me.
    Return a list of the elements of `sequence_of_sets`, removing all
    elements that are subsets of other elements.  Assumes that each
    element is a set or frozenset."""
    # The code below does not handle the case of a sequence containing
    # only the empty set, so let's just handle all easy cases now.
    sequence_of_sets = list(frozenset(sequence_of_sets))
    if len(sequence_of_sets) <= 1:
        return list(sequence_of_sets)
    # We need an indexable sequence so that we can use a bitmap to
    # represent each set.
    if not isinstance(sequence_of_sets, collections.Sequence):
        sequence_of_sets = list(sequence_of_sets)
    # For each element, construct the list of all sets containing that
    # element.
    sets_containing_element = {}
    for i, s in enumerate(sequence_of_sets):
        for element in s:
            try:
                sets_containing_element[element] |= 1 << i
            except KeyError:
                sets_containing_element[element] = 1 << i
    # For each set, if the intersection of all of the lists in which it is
    # contained has length != 1, this set can be eliminated.
    out = [s for s in sequence_of_sets
           if s and is_power_of_two(functools.reduce(
               operator.and_, (sets_containing_element[x] for x in s)))]
    return out
  

def get_boundary_matrix(filtered_complex):
    '''
    Return the boundary matrix of filtered complex. The format of the
    filtered complex is a tuple of tuples of integers. The returned boundary
    matrix is a list of tuples (dimension, boundary).
    Example:
    filtered_complex = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2))
    get_boundary_matrix(filtered_complex)
    '''
    simplex_indices = dict((face, k)
                           for k, face in enumerate(filtered_complex))
    boundary_matrix = []
    for simplex in filtered_complex:
        boundary = (simplex_indices[frozenset(face)] for face in
                    itertools.combinations(simplex, len(simplex) - 1)
                    if len(simplex) > 1)
        boundary_matrix.append((len(simplex) - 1, tuple(sorted(boundary))))
    return tuple(boundary_matrix)


def get_boundary_matrix_nophat(filtered_complex):
    '''
    Return the boundary matrix of filtered complex. The format of the
    filtered complex is a tuple of tuples of integers. The returned boundary
    matrix is a list of tuples (dimension, boundary).
    Example:
    filtered_complex = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2))
    get_boundary_matrix(filtered_complex)
    '''



def get_persistent_homology(filtered_nerve,
                            nerve_filtration_values,
                            method='dual'):
    '''
    Compute persistence pairs from a filtered nerve and its nerve
    filtration values.

    Arguments:
    filtered_nerve: tuple of tuples of integers specifying simplices
    nerve_filtration_values: list of filtration values
    method: string. Possible values 'standard' and 'dual'

    Example:
    my_homology = get_persistent_homology(
        filtered_nerve, nerve_values, method='dual')

    Note: the dual method seems to be slower than the standard method.
    '''

    simplex_indices = dict((face, k)
                           for k, face in enumerate(filtered_nerve))
    degrees = []
    boundaries = []
    for simplex in filtered_nerve:
        cardinality = len(simplex)
        degrees.append(cardinality - 1)
        boundary = (simplex_indices[frozenset(face)] for face in
                    itertools.combinations(simplex, len(simplex) - 1)
                    if len(simplex) > 1)
        boundaries.append(set(boundary))

    if method in ['dual']:
        boundaries = get_dual_boundary_matrix(boundaries)
        degrees = [-degree for degree in degrees[::-1]]
        nerve_filtration_values = nerve_filtration_values[::-1]

    degree_dict = {}
    for j, degree in enumerate(degrees):
        try:
            degree_dict[degree].append(j)
        except KeyError:
            degree_dict[degree] = [j]
    pairs = get_image_twist_reduced_boundary_matrix(
        boundaries, degree_dict)

    # return persistentce pairs
    if method in ['dual', 'kernel_dual', 'cohomology']:
        pers = [[-degrees[p[1]], nerve_filtration_values[p[1]],
                 nerve_filtration_values[p[0]]] for
                p in pairs]
    else:
        pers = [[degrees[p[0]], nerve_filtration_values[p[0]],
                 nerve_filtration_values[p[1]]] for
                p in pairs]
    pers = [per for per in pers if per[1] < per[2]]
    pers.sort(key=lambda pair: (pair[1] - pair[2], pair[0]))

    return np.array(pers)


def sort_filtered_nerve(nerve, nerve_values):
    # sort filtered nerve
    filtered_nerve = list(zip(nerve,
                              nerve_values))
    filtered_nerve.sort(key=lambda obj: len(obj[0]))
    filtered_nerve.sort(key=lambda obj: obj[1])
    filtered_nerve, nerve_values = list(zip(*filtered_nerve))
    return filtered_nerve, nerve_values


def powerset(some_set, max_card, min_card=1):
    return itertools.chain(*[
        list(itertools.combinations(some_set, dim + 1))
        for dim in range(min_card - 1, max_card)])


def get_dual_boundary_matrix(boundary_matrix):
    '''
    comments please
    '''
    res = dict((j, set()) for j in
               range(len(boundary_matrix)))
    n = len(boundary_matrix) - 1
    for j, boundary in enumerate(boundary_matrix):
        for i in boundary:
            res[n - i].add(n - j)
    return [res[i] for i in range(len(boundary_matrix))]


def get_image_twist_reduced_boundary_matrix(boundary_matrix,
                                            degree_dict):
    '''In place column reduction of boundary_matrix using the
    cohomological degree
    twist trick. Retruns list of persistence pairs.

    Arguments:
    boundary_matrix : list, tuple or dictionary of set
        objects.
    degree_dict: dictionary of signature (degree, index_list) where
        degree is the
        homological degree and index_list is the list of indices of
        columns in the
        boundary matrix of the given degree.

    Example:
    degrees, boundaries = zip(*get_boundary_matrix(filtered_nerve))
    degree_dict = {(degree, j) for (j, degree) in enumerate(degrees)}
    boundaries = [BTrees.IIBTree.IISet(b) for b in boundaries]
    persistence_pairs = get_twist_reduced_boundary_matrix(
        boundaries, degree_dict)

    This implementation seems to be approximately as fast as phat.
    '''

    # initialize dictionary of persistence pairs
    pairs = {}

    for degree in range(max(degree_dict.keys()),
                        min(degree_dict.keys()), -1):
        # print(degree, flush=True)
        low = set()
        for j in degree_dict[degree]:
            col = boundary_matrix[j]
            while True:
                try:
                    low_j = max(col.intersection(low))
                except ValueError:
                    if not bool(col):
                        break
                    low_j = max(col)
                    pairs[low_j] = j
                    low.add(low_j)
                    boundary_matrix[low_j] = set()
                    break
                col.symmetric_difference_update(
                    boundary_matrix[pairs[low_j]])

            boundary_matrix[j] = col
    return zip(pairs.keys(), pairs.values())


def distance_function(X, dissimilarity):
    """Distance function"""
    if dissimilarity in ['precomputed', 'dowker']:
        def dist_to_points(point_list):
            return X[point_list]
    else:
        def dist_to_points(point_list):
            return cdist(X[point_list],
                         X,
                         metric=dissimilarity)
    return dist_to_points


def inverse_beta(alpha):
    def beta(time):
        return(alpha(time) - time)

    def inverse_beta_function(time):
        return inversefunc(beta)(time)
    return inverse_beta_function


def _summarize_one(dowker):
    class_name = dowker.__class__.__name__
    split_name = class_name.split("_")
    if len(split_name) == 2:
        split_name += ['SFN', 'SFN']
    dowker.cardinality_information(verbose=False)
    return pd.DataFrame({'space': [split_name[1]],
                         'truncation': [split_name[2]],
                         'reduction': [split_name[3]],
                         'unreduced': [dowker.card_unreduced], 
                         'reduced': [dowker.card_reduced]}, 
                        columns = ['space', 'truncation', 'reduction', 'unreduced', 'reduced'])


def summarize_dowker(*args):
    res = pd.concat([_summarize_one(arg) for arg in args])
    return res
