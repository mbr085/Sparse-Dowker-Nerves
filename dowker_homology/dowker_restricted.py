"""
Python routines used to test claims in the paper
*** SDN = Sparse Dowker Nerves ***
*** SFN = Sparse Filtered Nerves *** 
Copyright: Applied Topology Group at University of Bergen.
No commercial use of the software is permitted without proper license.
"""
import numpy as np
import scipy.special
import scipy.sparse
from . import dowker_functions as df
from .dowker_base import Dowker_Base


class Dowker_Restricted(Dowker_Base):
    """
    Dowker restricted persistent homology class
    
    The minimal requirement for subclasses is that 
    they provide a get_restriction_times method. 
    """

    def __init__(self,
                 dimension=1,
                 translation_function=None,
                 initial_point=0,
                 resolution=0,
                 n_samples=None,
                 multiplicative_interleaving=1,
                 additive_interleaving=0,
                 homology_method='dual',
                 max_simplex_size=2e5):
        """
        Initalize a new dowker homology object.
        """
        super().__init__(dimension,
                         homology_method,
                         max_simplex_size)

        if translation_function is None:
            translation_function = df.translation_function_from_interleaving(
                additive_interleaving=additive_interleaving,
                multiplicative_interleaving=multiplicative_interleaving)
        self.initial_point = initial_point
        self.resolution = resolution
        self.n_samples = n_samples
        self.translation_function = translation_function

    def clone_or_reset(self, other):

        reset_list = {'_Dowker_Restricted__initial_point',
                      '_Dowker_Restricted__resolution',
                      '_Dowker_Restricted__n_samples',
                      '_Dowker_Restricted__translation_function'}
        clone_properties = {property: self.__dict__[
            property] for property in
            reset_list.intersection(set(self.__dict__.keys()))}
        super().clone_or_reset(other)
        other.__dict__.update(clone_properties)

    @property
    def initial_point(self):
        return self.__initial_point

    @initial_point.setter
    def initial_point(self, initial_point):
        self.reset()
        self.__initial_point = initial_point

    @property
    def resolution(self):
        return self.__resolution

    @resolution.setter
    def resolution(self, resolution):
        self.reset()
        self.__resolution = resolution

    @property
    def n_samples(self):
        return self.__n_samples

    @n_samples.setter
    def n_samples(self, n_samples):
        self.reset()
        self.__n_samples = n_samples

    @property
    def translation_function(self):
        return self.__translation_function

    @translation_function.setter
    def translation_function(self, translation_function):
        self.reset()
        self.__translation_function = translation_function

    def get_interleaving_lines(self, X=None, n_points=100):
        """
        Calculate interleaving guarantee
        """
        super().get_interleaving_lines()
        x_points = np.linspace(0, self.max_filtration_value, n_points)
        if self.n_samples is not None:
            resolution = self.cover_radius
        else:
            resolution = self.resolution
        y_points = self.translation_function(x_points + resolution)
        self.interleaving_lines.append((x_points, y_points))

    def get_maximal_infinity_faces(self, X=None):
        """Maximal faces of nerve of dowker matrix without filtration values"""
        self.update_X(X)
        if not hasattr(self, 'restriction_times'):
            self.get_restriction_times()
        restriction_order = np.argsort(self.restriction_times)[::-1]
        dowker_matrix = self.X
        slope = df.get_slope(self.parent_point_list,
                             self.restriction_times)

        self.maximal_faces = frozenset()
        for landmark in restriction_order:
            landmark_witnesses = np.flatnonzero(
                dowker_matrix[landmark, :] <= self.restriction_times[landmark])
            landmark_restriction_face = (np.flatnonzero(
                self.restriction_times[landmark] <= self.restriction_times))
            landmark_faces = []
            for witness in landmark_witnesses:
                face = set(tuple(sorted(
                    landmark_restriction_face[np.flatnonzero(
                        dowker_matrix[landmark_restriction_face, witness] <=
                        np.full_like(landmark_restriction_face,
                                     self.restriction_times[landmark],
                                     dtype=float))])))
                slope_face = landmark_restriction_face[
                    slope[landmark_restriction_face]]
                witness_face = set(tuple(sorted(
                    slope_face[np.flatnonzero(
                        dowker_matrix[slope_face, witness] <
                        self.restriction_times[slope_face])])))
                non_slope_face = landmark_restriction_face[
                    ~slope[landmark_restriction_face]]
                witness_face = witness_face.union(tuple(sorted(
                    non_slope_face[np.flatnonzero(
                        dowker_matrix[non_slope_face, witness] <=
                        self.restriction_times[non_slope_face])])))
                face.intersection_update(witness_face)
                landmark_faces.append(frozenset(face))
            landmark_faces = df.eliminate_subsets(
                list(frozenset(landmark_faces)))
            self.maximal_faces = self.maximal_faces.union(landmark_faces)
        self.maximal_faces = df.eliminate_subsets(list(self.maximal_faces))

        max_cardinality = max([len(x) for x in self.maximal_faces])
        self.actual_max_simplex_size = np.sum(
            [scipy.special.binom(
                max_cardinality, _i)
             for _i in range(1, self.dimension + 3)])
