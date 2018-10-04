"""
Python routines used to test claims in the paper
*** SDN = Sparse Dowker Nerves ***
*** SFN = Sparse Filtered Nerves *** 
Copyright: Applied Topology Group at University of Bergen.
No commercial use of the software is permitted without proper license.
"""
import numpy as np
import scipy.special
from scipy.spatial.distance import cdist
from . import dowker_functions as df
from .dowker import Dowker


class Dowker_Ambient(Dowker):
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
    # import packages
    import numpy as np
    import dowker_homology as dh
    
    # generate data
    N = 1000
    x = np.linspace(0, 2*np.pi, num=N, endpoint=False).reshape(N,1)
    y = 20*x
    coords = np.hstack((np.cos(x), np.sin(x), np.cos(y), np.sin(y))) 

    # initiate dowker homology object
    dowker = dh.Dowker_Ambient(
        additive_interleaving=0.2,
        multiplicative_interleaving=1.3)
    # calculate persistent homology
    dowker.persistence(X=coords)
    """

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
        self.cover_radius = np.min(insertion_times[self.farthest_point_list])
        self.coords = self.X[self.farthest_point_list]
        self.X = cdist(self.coords, self.coords, 
                       metric='euclidean')

        super().get_truncation_times()

    def get_nerve(self, X=None):
        """Compute filtration values of underlying simplicial complex"""
        self.update_X(X)
        infinity_nerve = self.get_infinity_nerve()
        self.nerve_values = np.array([
            df.get_ambient_filtration_value(
                list(simplex),
                self.farthest_point_list[self.point_sample],
                self.coords,
                self.restriction_times)
            for simplex in infinity_nerve])

        # sort filtered nerve
        finite_nerve = np.flatnonzero(
            self.nerve_values < self.max_filtration_value)

        self.filtered_nerve, self.nerve_values = df.sort_filtered_nerve(
            infinity_nerve[finite_nerve], self.nerve_values[finite_nerve])

    def get_maximal_infinity_faces(self, X=None):
        """Maximal faces of nerve of dowker matrix without filtration values.
        """
        self.update_X(X)
        if not hasattr(self, 'restriction_times'):
            self.get_restriction_times()

        dim_const = np.sqrt(
            2 * self.ambient_dimension / (self.ambient_dimension + 1))
        restriction_order = np.argsort(self.restriction_times)[::-1]
        self.maximal_faces = frozenset()
        for index, landmark in enumerate(restriction_order):
            landmark_witnesses = np.flatnonzero(
                self.X[landmark, :] <=
                dim_const * self.truncation_times[landmark])
            landmark_restriction_face = (np.flatnonzero(
                self.restriction_times[landmark] <= self.restriction_times))
            landmark_faces = []
            for witness in landmark_witnesses:
                face = tuple(sorted(
                    landmark_restriction_face[np.flatnonzero(
                        self.X[landmark_restriction_face, witness] <=
                        np.minimum(
                            dim_const * np.full_like(
                                landmark_restriction_face,
                                self.restriction_times[landmark],
                                dtype=float),
                            self.truncation_times[
                                landmark_restriction_face] +
                            np.min(self.truncation_times[
                                landmark_restriction_face])))]))
                landmark_faces.append(face)
            landmark_faces = df.eliminate_subsets(
                list(frozenset(landmark_faces)))
            self.maximal_faces = self.maximal_faces.union(landmark_faces)
        self.maximal_faces = df.eliminate_subsets(list(self.maximal_faces))

        max_cardinality = max([len(x) for x in self.maximal_faces])
        self.actual_max_simplex_size = np.sum(
            [scipy.special.binom(
                max_cardinality, _i)
             for _i in range(1, self.dimension + 3)])
