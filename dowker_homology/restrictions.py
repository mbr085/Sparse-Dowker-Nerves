'''
Python routines used to test claims in the papers
*** SDN = Sparse Dowker Nerves ***
*** SFN = Sparse Filtered Nerves ***
Copyright: Applied Topology Group at University of Bergen.
No commercial use of the software is permitted without proper license.
'''
import numpy as np
from . import truncations
from . import nerves


def Restriction(truncation, restriction_method):
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
    if isinstance(truncation, truncations.TruncationSparse):
        return RestrictionSparse(truncation)
    elif (isinstance(truncation, truncations.TruncationNone) or
          isinstance(truncation, truncations.TruncationNumpy) or
          isinstance(truncation, truncations.TruncationSheehy)):
        if restriction_method == 'Sheehy':
            return RestrictionSheehy(truncation)
        elif restriction_method == 'Parent':
            return RestrictionParent(truncation)
        elif restriction_method == 'no_restriction':
            return RestrictionNone(truncation)
        else:
            return RestrictionNumpy(truncation)
    else:
        raise TypeError("unknown input")


class RestrictionNumpy(object):
    """Documentation for Restriction

    """

    def __init__(self, truncation):
        self.truncation = truncation
        self.get_restriction_times(truncation)

    def get_restriction_times(self, truncation):
        X = truncation.Y  # [:, truncation.witness_points()]
        self.X = X
        cover_matrix = truncation.get_cover_matrix(
            X=X, Y=X)
        # self.cm = cover_matrix.copy()
        cover_matrix[cover_matrix == 0] = np.inf
        # self.max_filtration_value
        self.cover_matrix = cover_matrix
        # self.restriction_times = np.full(truncation.DD.len(),
        #                                  truncation.DD.max())
        self.restriction_times = np.min(cover_matrix, axis=1)
        self.parent_point_list = truncation.get_parent_point_list(
            self.restriction_times, self.cover_matrix)

        self.restriction_times = cover_matrix[
            np.arange(len(cover_matrix)), self.parent_point_list]
        self.restriction_times[
            self.parent_point_list[
                np.arange(len(self.parent_point_list))] ==
            np.arange(len(self.parent_point_list))] = np.inf
        self.parent_point_list = truncation.get_parent_point_list(
            self.restriction_times, self.cover_matrix)
        self.force_monotone_restriction()

    def force_monotone_restriction(self):
        root = np.flatnonzero(
            self.parent_point_list == np.arange(len(self.parent_point_list)))
        restriction_tree = nerves.get_parent_forest(
            self.parent_point_list)
        for l in nerves.iterative_topological_sort(
                restriction_tree, root[0]):
            bucket = list(restriction_tree[l])
            bucket.append(l)
            self.restriction_times[l] = np.max(
                self.restriction_times[bucket])

    def get_Y_column(self, rf, w):
        return self.truncation.to_numpy_col(self.truncation.Y[rf, w])

    def witness_face(self, w, l, rf):
        if self.restriction_times[l] < np.inf:
            face = set(tuple(sorted(
                rf[np.flatnonzero(
                    self.get_Y_column(rf, w) <=
                    np.full_like(self.restriction_face_indices(l),
                                 self.restriction_times[l],
                                 dtype=float))])))
        else:
            face = set(tuple(sorted(
                rf[np.flatnonzero(
                    self.get_Y_column(rf, w) <
                    np.full_like(self.restriction_face_indices(l),
                                 self.restriction_times[l],
                                 dtype=float))])))
        return face

    def restriction_face_indices(self, l):
        return np.flatnonzero(
            self.restriction_times[l] <= self.restriction_times)

    def l_witnesses(self, l, X):
        return np.flatnonzero(
            self.truncation.Y[l] <= self.restriction_times[l])

    def slope_vertices(self, slope, w, rf):
        slope_face = rf[slope[rf]]
        return set(tuple(sorted(
            slope_face[np.flatnonzero(
                self.get_Y_column(slope_face, w) <
                self.restriction_times[slope_face])])))

    def non_slope_vertices(self, slope, w, rf):
        non_slope_face = rf[~slope[rf]]
        return set(tuple(sorted(
            non_slope_face[np.flatnonzero(
                self.get_Y_column(non_slope_face, w) <=
                self.restriction_times[non_slope_face])])))


class RestrictionNone(RestrictionNumpy):
    def get_restriction_times(self, X=None):
        """Calculate restriction times"""
        self.parent_point_list = np.full_like(
            self.truncation.truncation_times,
            self.truncation.initial_point,
            dtype=int)
        self.restriction_times = np.full_like(
            self.truncation.truncation_times,
            np.inf)


class RestrictionSheehy(RestrictionNumpy):

    def get_restriction_times(self, X=None):
        """Calculate restriction times"""
        self.parent_point_list = np.full_like(
            self.truncation.truncation_times,
            self.truncation.initial_point,
            dtype=int)
        self.restriction_times = self.truncation.translation_function(
            self.truncation.truncation_times)


class RestrictionParent(RestrictionNumpy):

    def get_restriction_times(self, X=None):
        """Calculate restriction times"""

        Y = self.truncation.Y
        truncation_times = self.truncation.truncation_times
        n = len(truncation_times)
        self.parent_point_list = np.zeros(n, dtype=int)
        self.restriction_times = np.full(n, np.inf)
        truncation_order = np.argsort(truncation_times)[::-1]
        for point_index in range(1, n):
            ordered_point_index = truncation_order[point_index]
            check_points = Y[ordered_point_index, :] < np.inf
            max_dist = [np.max(Y[truncation_order[parent_index], check_points])
                        for parent_index in range(point_index)]
            self.restriction_times[point_index] = np.min(max_dist)
            self.parent_point_list[point_index] = np.argmin(max_dist)
            # self.parent_point_list[point_index] = np.flatnonzero(check_points)[
            #    np.argmin(max_dist)]


class RestrictionSparse(RestrictionNumpy):
    # def get_Y_column(self, rf, w):
    #     res = self.truncation.Y[
    #         rf, w].toarray().flatten()
    #     res[res == 0] = np.inf
    #     return res

    def l_witnesses(self, l, X):
        if self.restriction_times[l] < np.inf:
            Wl = set(
                list(np.array(self.truncation.Y.rows[l])[
                    np.array(self.truncation.Y.data[l]) <=
                    self.restriction_times[l]]))
            Wl.intersection_update(set(list(
                X.rows[l])))
        else:
            Wl = set(range(X.shape[1]))
        return Wl

    def get_restriction_times(self, truncation):
        """Calculate restriction times"""
        X = truncation.Y  # [:, self.witness_points()]
        self.X = X
        cover_matrix = truncation.get_cover_matrix(
            X=X, Y=X)
        cover_matrix[cover_matrix == -np.inf] = np.inf
        self.cm = cover_matrix.copy()
        self.cover_matrix = cover_matrix
        self.restriction_times = np.full(truncation.DD.len(),
                                         truncation.DD.max())
        for index in range(len(self.restriction_times)):
            if len(cover_matrix.rows[index]) > 0:
                self.restriction_times[index] = min(
                    cover_matrix.data[index])
            else:
                self.restriction_times[index] = np.inf
        self.restriction_times1 = self.restriction_times
        self.parent_point_list = truncation.get_parent_point_list(
            self.restriction_times, self.cover_matrix)
        self.restriction_times = cover_matrix[
            np.arange(cover_matrix.shape[0]),
            self.parent_point_list].toarray().flatten()
        self.restriction_times[
            self.restriction_times == 0] = np.inf
        self.restriction_times[
            self.parent_point_list[
                np.arange(len(self.parent_point_list))] ==
            np.arange(len(self.parent_point_list))] = np.inf
        self.force_monotone_restriction()

    def get_restriction_times2(self, truncation):
        X = truncation.Y  # [:, truncation.witness_points()]
        cover_matrix = truncation.get_cover_matrix(
            X=X, Y=X)
        cover_matrix[cover_matrix == 0] = np.inf
        # self.max_filtration_value
        self.cover_matrix = cover_matrix
        self.restriction_times = np.full(truncation.DD.len(),
                                         truncation.DD.max())
        self.restriction_times = np.min(cover_matrix, axis=1)
        self.parent_point_list = truncation.get_parent_point_list(
            self.restriction_times, self.cover_matrix)

        # self.restriction_tree = nerves.get_parent_forest(
        #     self.parent_point_list)

        self.restriction_times = cover_matrix[
            np.arange(len(cover_matrix)), self.parent_point_list]
        self.restriction_times[
            self.parent_point_list[
                np.arange(len(self.parent_point_list))] ==
            np.arange(len(self.parent_point_list))] = np.inf
