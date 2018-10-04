"""
Python routines used to test claims in the paper
*** SDN = Sparse Dowker Nerves ***
*** SFN = Sparse Filtered Nerves *** 
Copyright: Applied Topology Group at University of Bergen.
No commercial use of the software is permitted without proper license.
"""
import numpy as np
from . import dowker_functions as df
from .dowker import Dowker


class Dowker_Intrinsic(Dowker):
    """Dowker intrinsic persistent homology class

    Parameters
    ----------
    dimension : maximal persistence dimension (default 1)
    dissimilarity : A dissimilarity from 
        scipy.spatial.distance.cdist (default 'euclidean')
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
    dowker = dh.Dowker_Intrinsic(
        additive_interleaving=0.2,
        multiplicative_interleaving=1.3)
    # calculate persistent homology
    dowker.persistence(X=coords)

    """

    def __init__(self,
                 dimension=1,
                 dissimilarity='euclidean',
                 translation_function=None,
                 initial_point=0,
                 resolution=0,
                 n_samples=None,
                 multiplicative_interleaving=1,
                 additive_interleaving=0,
                 homology_method='dual',
                 max_simplex_size=2e5):
        super().__init__(
            dimension=dimension,
            translation_function=translation_function,
            initial_point=initial_point,
            resolution=resolution,
            n_samples=n_samples,
            multiplicative_interleaving=multiplicative_interleaving,
            additive_interleaving=additive_interleaving,
            homology_method=homology_method,
            max_simplex_size=max_simplex_size)

        self.dissimilarity = dissimilarity

    def get_truncation_times(self, X=None):
        """Calculate truncation times"""
        self.update_X(X)
        insertion_times = df.get_farthest_insertion_times(
            X=self.X,
            dissimilarity=self.dissimilarity,
            initial_point=self.initial_point,
            resolution=self.resolution,
            n_samples=self.n_samples)
        self.farthest_point_sample = np.flatnonzero(insertion_times > -np.inf)
        self.X = df.distance_function(self.X, self.dissimilarity)(
            self.farthest_point_sample)
        self.cover_radius = np.min(insertion_times[self.farthest_point_sample])

        super().get_truncation_times()

    def clone_or_reset(self, other):

        reset_list = {'_Dowker_Intrinsic__dissimilarity'}
        clone_properties = {property: self.__dict__[
            property] for property in
            reset_list.intersection(set(self.__dict__.keys()))}
        super().clone_or_reset(other)
        other.__dict__.update(clone_properties)

    @property
    def dissimilarity(self):
        return self.__dissimilarity

    @dissimilarity.setter
    def dissimilarity(self, dissimilarity):
        self.reset()
        self.__dissimilarity = dissimilarity
