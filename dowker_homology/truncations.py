"""
Python routines used to test claims in the papers
*** SDN = Sparse Dowker Nerves ***
*** SFN = Sparse Filtered Nerves ***
Copyright: Applied Topology Group at University of Bergen.
No commercial use of the software is permitted without proper license.
"""
import numpy as np
from . import dissimilarities
from . import nerves
import scipy.sparse
from pynverse import inversefunc


def Truncation(
    DD,
    translation_function,
    truncation_method=None,
    n_samples=None,
    initial_point=0,
):
    """
    This function returns a class storing the truncation
    class.

    Parameters
    ----------
    DD : Dissimilarity
    translation_function : function
    truncation_method : str
        The desired truncation method.
    n_samples : int
    initial_point : int

    Examples
    --------
    >>> data = np.random.randn(100, 5)
    >>> dd = Dissimilarity(data, dissimilarity='minkowski', p=1)
    >>> def translation_function(time): return 1.5 * time + 0.1
    >>> trunc = Truncation(dd, translation_function = translation_function)
    """
    if isinstance(DD, dissimilarities.DissimilaritySparse):
        return TruncationSparse(
            DD=DD,
            translation_function=translation_function,
            n_samples=n_samples,
            initial_point=initial_point,
        )
    elif isinstance(DD, dissimilarities.DissimilarityNumpy):
        if truncation_method == "no_truncation":
            return TruncationNone(DD=DD)
        if truncation_method == "Canonical":
            return TruncationCanonical(
                DD=DD,
                translation_function=translation_function,
                n_samples=n_samples,
                initial_point=initial_point,
            )
        elif truncation_method == "Sheehy":
            return TruncationSheehy(
                DD=DD,
                translation_function=translation_function,
                n_samples=n_samples,
                initial_point=initial_point,
            )
        # elif truncation_method == 'Sheehy_Full':
        #     return TruncationSheehyFull(
        #         DD=DD,
        #         translation_function=translation_function,
        #         n_samples=n_samples,
        #         initial_point=initial_point)
        else:
            return TruncationNumpy(
                DD=DD,
                translation_function=translation_function,
                n_samples=n_samples,
                initial_point=initial_point,
            )
    else:
        raise TypeError("unknown input")


class TruncationNumpy(object):
    """Documentation for Truncation_Numpy"""

    def __init__(
        self, DD, translation_function, n_samples=None, initial_point=0
    ):
        # initialize
        self.X = DD.X.copy()
        self.DD = DD
        self.translation_function = translation_function
        self.n_samples = n_samples
        self.initial_point = initial_point
        # calculate truncation times and truncated dissimilarity
        self.get_truncation_times()
        self.offset = 0

    def get_cover_list(self, x, Y):
        mask = x < np.inf
        cover_list = np.max(x[mask] * (Y[:, mask] < x[mask]), axis=1)
        cover_list[np.any(np.isfinite(Y[:, ~mask]), axis=1)] = np.inf
        return cover_list

    # def modified_translation_function(self, translation_function, t):
    #     return translation_function(t)

    def matrix_translation_function(self, translation_function, X):
        return translation_function(X)

    def to_numpy_row(self, array):
        return array

    def to_numpy_col(self, array):
        return array

    def get_parent_point_list(self, death_times, cover_matrix):
        death_order = np.argsort(death_times, kind="stable")[::-1]
        qsets = []
        parent_point_list = np.zeros(len(death_times), dtype=int)
        for index, point in enumerate(death_order):
            qset = death_times[point] == self.to_numpy_row(
                cover_matrix[point, death_order[:index]]
            )
            if np.sum(qset):
                parent_point_list[point] = death_order[
                    np.min(np.arange(len(qset))[qset])
                ]
            elif index > 0:
                ldeath_times = np.min(
                    self.to_numpy_row(cover_matrix[point, death_order[:index]])
                )
                qset = ldeath_times == self.to_numpy_row(
                    cover_matrix[point, death_order[:index]]
                )
                if np.sum(qset):
                    parent_point_list[point] = death_order[
                        np.min(np.arange(len(qset))[qset])
                    ]
                else:
                    parent_point_list[point] = point
            else:
                parent_point_list[point] = point
                qsets.append(death_order[:index])
        return parent_point_list

    def get_cover_matrix(self, X, Y):
        cover_matrix = np.zeros((len(X), len(X)))
        for index in range(X.shape[0]):
            cover_matrix[index] = self.get_cover_list(X[index], Y)
        # for w in range(X.shape[1]):
        #     Xw = X[:, w].reshape((len(X), 1))
        #     Yw = Y[:, w].reshape((1, len(X)))
        #     cover_matrix = np.maximum(
        #         Xw * (Xw > Yw), cover_matrix)
        return cover_matrix.transpose()

    def minimum(self, X, Y):
        return np.minimum(X, Y)

    def maximum(self, X, Y):
        return np.maximum(X, Y)

    def get_truncation_times(self, X=None):
        """Calculate truncation times"""
        # X = self.DD.X
        if self.n_samples is None:
            self.n_samples = self.DD.X.shape[0]
        Y = self.matrix_translation_function(
            self.translation_function, self.DD.X
        )
        farthest_point = self.initial_point
        self.farthest_point_list = [farthest_point]
        self.truncation_times = np.zeros(self.DD.X.shape[0])
        self.truncation_times[self.initial_point] = np.inf
        remaining_points = list(range(self.DD.X.shape[0]))
        remaining_points.pop(self.initial_point)
        cover_list = self.get_cover_list(
            self.DD.X[farthest_point], Y[remaining_points]
        )
        self.cover_list1 = cover_list
        while remaining_points:
            farthest_point = np.argmax(cover_list)
            Tvalue = cover_list[farthest_point]
            if len(self.farthest_point_list) >= self.n_samples:
                self.offset = Tvalue
                break
            # if Tvalue <= self.modified_translation_function(
            #         self.translation_function, 0):
            if Tvalue <= self.translation_function(0):
                break
            cover_list = np.delete(cover_list, farthest_point)
            farthest_point = remaining_points.pop(farthest_point)
            self.truncation_times[farthest_point] = Tvalue
            self.farthest_point_list.append(farthest_point)
            cover_list = np.minimum(
                self.get_cover_list(
                    self.DD.X[farthest_point], Y[remaining_points]
                ),
                cover_list,
            )
        self.farthest_point_list = np.array(self.farthest_point_list)
        self.point_sample = np.flatnonzero(
            self.truncation_times > self.translation_function(0)
        )
        self.truncation_times = self.truncation_times[self.point_sample]
        self.Y = Y[self.point_sample]
        self.DD.subsample(self.point_sample)
        # self.DD.X = self.DD.X[self.point_sample]
        self.get_truncated_dissimilarity()

    def get_truncated_dissimilarity(self):
        X = self.DD.X
        death_times = self.truncation_times

        Y = self.Y
        # translation_function = self.translation_function
        cover_matrix = self.get_cover_matrix(X=X, Y=Y)
        parent_point_list = self.get_parent_point_list(
            death_times, cover_matrix
        )
        graph = nerves.get_parent_forest(parent_point_list)
        self.graph = graph
        start = np.argmax(death_times)
        for index in nerves.iterative_topological_sort(
            graph=graph, start=start
        ):
            for child in graph[index]:
                Y[index] = self.minimum(Y[index], Y[child])
            Y[index] = self.maximum(Y[index], X[index])
        self.Y = Y


class TruncationCanonical(TruncationNumpy):
    # self.get_cover_list does not belong in this class.
    def get_truncation_times(self, X=None):
        insertion_times = dissimilarities.get_farthest_insertion_times(
            X=self.DD.X,
            dissimilarity="precomputed",
            initial_point=self.initial_point,
        )
        self.farthest_point_sample = np.flatnonzero(insertion_times > -np.inf)
        self.cover_radius = np.min(insertion_times[self.farthest_point_sample])

        Y = self.translation_function(self.X)
        farthest_point = self.initial_point
        self.farthest_point_list = [farthest_point]
        self.truncation_times = np.zeros(len(self.X))
        self.truncation_times[self.initial_point] = np.inf
        remaining_points = list(range(len(self.X)))
        remaining_points.pop(self.initial_point)
        cover_list = self.get_cover_list(
            self.X[farthest_point], Y[remaining_points]
        )
        while remaining_points:
            farthest_point = np.argmax(cover_list)
            Tvalue = cover_list[farthest_point]
            if Tvalue == -np.inf:
                break
            cover_list = np.delete(cover_list, farthest_point)
            farthest_point = remaining_points.pop(farthest_point)
            self.truncation_times[farthest_point] = Tvalue
            self.farthest_point_list.append(farthest_point)
            # X[farthest_point][
            #     X[farthest_point] > Tvalue] = self.max_filtration_value
            cover_list = np.minimum(
                self.get_cover_list(
                    self.X[farthest_point], Y[remaining_points]
                ),
                cover_list,
            )
        self.farthest_point_list = np.array(self.farthest_point_list)
        self.point_sample = np.flatnonzero(self.truncation_times > 0)
        self.truncation_times = self.truncation_times[self.point_sample]

        self.X = self.X[self.farthest_point_sample]
        self.max_filtration_value = np.max(self.X)

        truncation = np.repeat(
            self.truncation_times.reshape(len(self.truncation_times), 1),
            self.X.shape[1],
            axis=1,
        )
        self.Y = np.full(self.X.shape, self.max_filtration_value)
        finite_indices = self.X <= truncation
        self.Y[finite_indices] = self.X[finite_indices]


class TruncationSheehy(TruncationNumpy):
    def truncation_function(self, time):
        return self.translation_function(inversefunc(self.beta)(time))
        # result = np.full_like(time, np.inf)
        # finite_times = np.isfinite(time)
        # result[finite_times] = self.translation_function(
        #     inversefunc(self.beta)(time[finite_times]))
        # return result

    def beta(self, time):
        return self.translation_function(time) - time

    def get_truncation_times(self, X=None):
        insertion_times = dissimilarities.get_farthest_insertion_times(
            X=self.DD.X,
            dissimilarity="precomputed",
            initial_point=self.initial_point,
        )
        # farthest point sample
        self.farthest_point_sample = np.flatnonzero(insertion_times > -np.inf)
        self.cover_radius = np.min(insertion_times)

        self.truncation_times = np.full_like(insertion_times, np.inf)
        finite_insertions = insertion_times < np.inf
        self.truncation_times[finite_insertions] = self.truncation_function(
            insertion_times[finite_insertions]
        )

        self.X = self.X[self.farthest_point_sample]
        self.max_filtration_value = np.max(self.X)

        truncation = np.repeat(
            self.truncation_times.reshape(len(self.truncation_times), 1),
            self.X.shape[1],
            axis=1,
        )
        self.Y = np.full(self.X.shape, np.inf)
        finite_indices = self.X <= truncation
        self.Y[finite_indices] = self.X[finite_indices]


class TruncationNone(TruncationNumpy):
    """Documentation for Truncation_None"""

    def __init__(self, DD):
        # initialize
        self.X = DD.X.copy()
        self.DD = DD
        # calculate truncation times and truncated dissimilarity
        self.truncation_times = np.full(self.X.shape[0], np.inf)
        self.Y = self.X


class TruncationSparse(TruncationNumpy):
    """Documentation for Truncation_Sparse"""

    def to_numpy_row(self, array):
        return _sparse_row_to_numpy(array)

    def to_numpy_col(self, array):
        return self.to_numpy_row(array.transpose()).transpose()

    def get_cover_list(self, x, Y):
        x_indices = x.rows[0]
        cover_list = np.empty(Y.shape[0])
        for index, yrow_indices in enumerate(Y.rows):
            if set(yrow_indices).difference(x_indices):
                cover_list[index] = np.inf
                continue
            xrow = _sparse_row_to_numpy(x[0, yrow_indices])
            Yrow = _sparse_row_to_numpy(Y[index, yrow_indices])
            Yrowlessxrow = Yrow < xrow
            if np.any(Yrowlessxrow):
                cover_list[index] = np.max(xrow * Yrowlessxrow)
            else:
                cover_list[index] = 0.0
        return cover_list

    # def modified_translation_function(self, translation_function, t):
    #     return translation_function(t)

    def matrix_translation_function(self, translation_function, X):
        Y = scipy.sparse.lil_matrix(X.shape)
        for index, data in enumerate(X.data):
            data = np.array(data)
            Y.data[index] = list(self.translation_function(data))
            Y.rows[index] = X.rows[index]
        return Y

    def get_cover_matrix(self, X, Y):
        cover_matrix = scipy.sparse.lil_matrix((X.shape[0], Y.shape[0]))
        for index in range(X.shape[0]):
            cover_list = self.get_cover_list(X[index], Y)
            cover_list[cover_list == 0] = -np.inf
            cover_matrix[index] = cover_list
        return cover_matrix.transpose()

    def minimum(self, A, B):
        Arow = set(A.rows[0])
        Brow = set(B.rows[0])
        intersection = Arow.intersection(Brow)
        onlyA = sorted(list(Arow.difference(intersection)))
        onlyB = sorted(list(Brow.difference(intersection)))
        intersection = sorted(list(intersection))
        res = scipy.sparse.lil_matrix(A.shape)
        if intersection:
            res[0, intersection] = A[0, intersection].minimum(
                B[0, intersection]
            )
        res[0, onlyA] = A[0, onlyA]
        res[0, onlyB] = B[0, onlyB]
        return res

    def maximum(self, A, B):
        Arow = set(A.rows[0])
        Brow = set(B.rows[0])
        intersection = list(Arow.intersection(Brow))
        res = scipy.sparse.lil_matrix(A.shape)
        if intersection:
            res[0, intersection] = A[0, intersection].maximum(
                B[0, intersection]
            )
        return res


def _sparse_row_to_numpy(array):
    if array.shape[0] > 0:
        res = np.full(array.shape[1], np.inf)
        res[array.rows[0]] = array.data[0]
        res[res == -np.inf] = 0
    else:
        res = np.zeros(array.shape)
    return res
