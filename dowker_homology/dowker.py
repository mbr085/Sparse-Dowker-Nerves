"""
Python routines used to test claims in the paper
*** SDN = Sparse Dowker Nerves ***
*** SFN = Sparse Filtered Nerves *** 
Copyright: Applied Topology Group at University of Bergen.
No commercial use of the software is permitted without proper license.
"""
import numpy as np
from . import dowker_functions as df
from .dowker_restricted import Dowker_Restricted


def get_cover_list(x, Y):
    return np.max(x * (Y < x), axis=1)


class Dowker(Dowker_Restricted):
    """Dowker persistent homology class

    Parameters
    ----------
    dimension : maximal persistence dimension (default 1)
    translation_function : an optional translation function, 
        which automatically gets calculated from 
        multiplicative and additive interleaving if None 
        (default None)
    initial_point : Initial point for farthest point
        sampling (default 0)
    resolution : Resolution for farthest point sampling
        (default 0)
    n_samples : Number of samples for farthest point 
        sampling (default None)
    multiplicative_interleaving : multiplicative 
        interleaving (default 1)
    additive_interleaving : additive interleaving
        (default 1)
    homology_method : what method should be used to
        calculate persistent homology (see phat)
        (default 'dual)
    max_simplex_size : maximum size biggest simplex 
        in the nerve to calculate persistent homology
        (default 2e5)

    Examples
    --------
    # generate cyclic network
    def cyclic_network(n_nodes):
        m = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j in range(n_nodes):
                m[i, j] = (j - i) % n_nodes
        return m
    dowker_dissimilarity = cyclic_network(100)
    # initiate dowker homology object
    dowker = dh.Dowker(dimension=1)
    # calculate and plot persistent homology
    dowker.persistence(X=dowker_dissimilarity)
    """

    def get_truncation_times(self, X=None):
        """Calculate truncation times"""
        self.update_X(X)
        self.max_filtration_value = np.max(self.X)
        Y = self.translation_function(self.X)
        farthest_point = self.initial_point
        self.farthest_point_list = [farthest_point]
        self.truncation_times = np.zeros(len(self.X))
        self.truncation_times[self.initial_point] = np.inf
        remaining_points = list(range(len(self.X)))
        remaining_points.pop(self.initial_point)
        cover_list = get_cover_list(self.X[farthest_point],
                                    Y[remaining_points])
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
                get_cover_list(self.X[farthest_point], Y[remaining_points]),
                cover_list)
        self.farthest_point_list = np.array(self.farthest_point_list)
        self.point_sample = np.flatnonzero(self.truncation_times > 0)
        self.truncation_times = self.truncation_times[self.point_sample]
        self.X = df.get_truncated_dowker_matrix(
            X=df.distance_function(self.X, dissimilarity='precomputed')(
                self.point_sample),
            truncation_times=self.truncation_times,
            max_filtration_value=self.max_filtration_value)

    def get_restriction_times(self, X=None):
        """Calculate restriction times"""
        self.update_X(X)
        if not hasattr(self, 'truncation_times'):
            self.get_truncation_times()

        cover_matrix = df.get_cover_matrix(
            X=self.X, Y=self.X)
        cover_matrix[cover_matrix == 0] = self.max_filtration_value
        self.cover_matrix = cover_matrix
        self.restriction_times = np.full(len(self.X),
                                         self.max_filtration_value)
        self.restriction_times = np.min(cover_matrix, axis=1)
        self.parent_point_list = df.get_parent_point_list(
            self.restriction_times, self.cover_matrix)

        self.restriction_tree = df.get_parent_forest(
            self.parent_point_list)

        self.restriction_times = cover_matrix[
            np.arange(len(cover_matrix)), self.parent_point_list]
        self.restriction_times[
            self.parent_point_list[
                np.arange(len(self.parent_point_list))] ==
            np.arange(len(self.parent_point_list))] = np.inf

        # slope = np.zeros(len(self.parent_point_list), dtype=bool)
        # for index in range(len(self.parent_point_list)):
        #     if len(self.restriction_tree[index]) > 0:
        #         slope[index] = (self.restriction_times[index] >
        #                         np.max(self.restriction_times[
        #                             list(self.restriction_tree[index])]))
        #     else:
        #         slope[index] = True
        self.slope = df.get_slope(self.parent_point_list,
                                  self.restriction_times)
