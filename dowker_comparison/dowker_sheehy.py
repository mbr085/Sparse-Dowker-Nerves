'''
Python routines used to test claims in the paper
*** SDN = Sparse Dowker Nerves ***
Copyright: Applied Topology Group at University of Bergen.
No commercial use of the software is permitted without proper license.
'''
import numpy as np
from dowker_homology import dowker_functions as df
from dowker_homology.dowker_intrinsic import Dowker_Intrinsic
from dowker_homology.dowker_ambient import Dowker_Ambient


class Dowker_Intrinsic_Sheehy_Sheehy(Dowker_Intrinsic):
    '''Dowker persistent homology class'''

    def get_truncation_times(self, X=None):
        """Calculate truncation times"""
        self.update_X(X)

        insertion_times = df.get_farthest_insertion_times(
            X=self.X,
            dissimilarity=self.dissimilarity,
            initial_point=self.initial_point,
            n_samples=self.n_samples,
            resolution=self.resolution)
        self.farthest_point_sample = np.flatnonzero(insertion_times > -np.inf)
        insertion_times = insertion_times[self.farthest_point_sample]
        self.cover_radius = np.min(insertion_times)
        inverse_beta_function = df.inverse_beta(
            self.translation_function)

        def truncation_function(time):
            return self.translation_function(inverse_beta_function(time))

        self.truncation_times = np.full_like(insertion_times, np.inf)
        finite_insertions = insertion_times < np.inf
        self.truncation_times[finite_insertions] = truncation_function(
            insertion_times[finite_insertions])
        distX = df.distance_function(self.X, self.dissimilarity)(
            self.farthest_point_sample)
        self.max_filtration_value = np.max(distX)
        self.X = df.get_truncated_dowker_matrix(
            X=distX,
            truncation_times=self.truncation_times, 
	    max_filtration_value=self.max_filtration_value)
        self.parent_point_list = df.get_parent_point_list(self.truncation_times, 
                                                          self.X)

    def get_restriction_times(self, X=None):
        """Calculate restriction times"""
        self.update_X(X)
        if not hasattr(self, 'truncation_times'):
            self.get_truncation_times()

        self.restriction_times = self.translation_function(
            self.truncation_times)

class Dowker_Ambient_Sheehy_Sheehy(Dowker_Ambient):
    '''Dowker persistent homology class'''

    def get_truncation_times(self, X=None):
        """Calculate truncation times"""
        self.update_X(X)
        self.coords = self.X
        self.ambient_dimension = self.coords.shape[1]

        insertion_times = df.get_farthest_insertion_times(
            X=self.X,
            dissimilarity='euclidean',
            initial_point=self.initial_point,
            n_samples=self.n_samples,
            resolution=self.resolution)
        self.farthest_point_list = np.flatnonzero(insertion_times > -np.inf)
        insertion_times = insertion_times[self.farthest_point_list]
        self.cover_radius = np.min(insertion_times)
        # insertion_order = point_sample[
        #     np.argsort(insertion_times[point_sample])[::-1]]
        # insertion_times = insertion_times[insertion_order]
        inverse_beta_function = df.inverse_beta(
            self.translation_function)

        def truncation_function(time):
            return self.translation_function(inverse_beta_function(time))

        self.truncation_times = np.full_like(insertion_times, np.inf)
        finite_insertions = insertion_times < np.inf
        self.truncation_times[finite_insertions] = truncation_function(
            insertion_times[finite_insertions])
        self.point_sample = np.flatnonzero(self.truncation_times > -np.inf)
        distX = df.distance_function(self.X, dissimilarity='euclidean')(
            self.farthest_point_list)
        self.max_filtration_value = np.max(distX)
        self.X = df.get_truncated_dowker_matrix(
            X=distX,
            truncation_times=self.truncation_times,
            max_filtration_value=self.max_filtration_value)

    def get_restriction_times(self, X=None):
        """Calculate restriction times"""
        self.update_X(X)
        if not hasattr(self, 'truncation_times'):
            self.get_truncation_times()

        self.restriction_times = self.translation_function(
            self.truncation_times)
