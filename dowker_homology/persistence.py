'''
Python routines used to test claims in the papers
*** SDN = Sparse Dowker Nerves ***
*** SFN = Sparse Filtered Nerves ***
Copyright: Applied Topology Group at University of Bergen.
No commercial use of the software is permitted without proper license.
'''
import collections
import numpy as np
from . import nerves
from .simplex_tree import SimplexTreeFromNerve
from scipy.spatial.distance import _METRIC_ALIAS


class PersistenceTransformer(object):
    """
    This class implements the fit and transform methods that are
    needed when calculating persistent homology in a sklearn-pipline.

    Parameters
    ----------
    dimension : int (default : 1)
        Homology dimension to calculate.
    dissimilarity : str (default : 'dowker')
        The default assumes that the input is a dowker dissimilarity. \
        Otherwise, any valid argument for the 'metric' of \
        `scipy.spatial.distance.cdist` is is valid. To calculate the \
        ambient Cech complex in Euclidean space, specify 'ambient'.
    additive_interleaving : float (default : 0.0)
        Additive interleaving guarantee.
    multiplicative_interleaving : float (default : 1.0)
        Multiplicative interleaving guarantee.
    translation_function : function (default : None)
        Translation function. If not specified, the additive and \
        multiplicative interleavings are used. If specified, overwrites \
        additive and multiplicative interleavings.
    truncation_method : str (default : None)
        The truncation method used.
    restriction_method : str (default : None)
        The restriction method used
    resolution : float
    n_samples : int
    isolated_points : bool (default : True)
        Should isolated points be plotted in the persistence diagram?
    initial_point : int (default : 0)
    coef_field : int (default : 11)
        Characteristic p of the coefficient field Z/pZ for computing homology.
    max_simplex_size : int (default : 2e5)
        Maximal size of a simplex. Use to make sure that \
        computation time is finite.
    **kwargs : dict, optional
        cutoff : int (default : np.inf) \
            Cutoff for Graph dissimilarities. \
        Additional arguments to the \
        `scipy.spatial.distance.cdist` function.
    """

    def __init__(self,
                 dimension=1,
                 dissimilarity=None,
                 additive_interleaving=0.0,
                 multiplicative_interleaving=1.0,
                 translation_function=None,
                 truncation_method=None,
                 restriction_method=None,
                 resolution=0.0,
                 n_samples=None,
                 isolated_points=True,
                 initial_point=0,
                 coeff_field=11,
                 max_simplex_size=2e5,
                 **kwargs):
        self.persistence = Persistence(
            dimension=dimension,
            dissimilarity=dissimilarity,
            additive_interleaving=additive_interleaving,
            multiplicative_interleaving=multiplicative_interleaving,
            translation_function=translation_function,
            truncation_method=truncation_method,
            resolution=resolution,
            n_samples=n_samples,
            isolated_points=isolated_points,
            initial_point=initial_point,
            coeff_field=coeff_field,
            max_simplex_size=max_simplex_size,
            **kwargs)

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        """
        Transform data into persistence diagrams.

        Parameters
        ----------
        X : ndarray or list of ndarrays
            data

        Returns
        -------
        dgms : list of list of ndarrays
            One list for each dataset containing a list with an array for each
            dimension containing birth- and death values.
        """
        if isinstance(X, collections.abc.Sequence):
            self.transformed_data = list()
            for data in X:
                self.persistence.persistent_homology(X=data)
                self.transformed_data.append(
                    self.persistence.simplex_tree.dgms)
        else:
            self.persistence.persistent_homology(X=X)
            self.transformed_data = self.persistence.simplex_tree.dgms
        return self.transformed_data

    def fit_transform(self, X, y=None):
        return self.transform(X=X, y=y)


class Persistence(object):
    """
    This class is used for calculating and plotting persistent homology.

    Parameters
    ----------
    dimension : int (default : 1)
        Homology dimension to calculate.
    dissimilarity : str (default : 'dowker')
        The default assumes that the input is a dowker dissimilarity. \
        Otherwise, any valid argument for the 'metric' of \
        `scipy.spatial.distance.cdist` is is valid. To calculate the \
        ambient Cech complex in Euclidean space, specify 'ambient'.
    additive_interleaving : float (default : 0.0)
        Additive interleaving guarantee.
    multiplicative_interleaving : float (default : 1.0)
        Multiplicative interleaving guarantee.
    translation_function : function (default : None)
        Translation function. If not specified, the additive and \
        multiplicative interleavings are used. If specified, overwrites \
        additive and multiplicative interleavings.
    truncation_method : str (default : None)
        The truncation method used.
    restriction_method : str (default : None)
        The restriction method used
    resolution : float
    n_samples : int
    isolated_points : bool (default : True)
        Should isolated points be plotted in the persistence diagram?
    initial_point : int (default : 0)
    coef_field : int (default : 11)
        Characteristic p of the coefficient field Z/pZ for computing homology.
    max_simplex_size : int (default : 2e5)
        Maximal size of a simplex. Use to make sure that \
        computation time is finite.
    **kwargs : dict, optional
        cutoff : int (default : 1) \
            Cutoff for Graph dissimilarities. \
        Additional arguments to the \
        `scipy.spatial.distance.cdist` function.
    """

    def __init__(self,
                 dimension=1,
                 dissimilarity=None,
                 additive_interleaving=0.0,
                 multiplicative_interleaving=1.0,
                 translation_function=None,
                 truncation_method=None,
                 restriction_method=None,
                 resolution=0.0,
                 n_samples=None,
                 isolated_points=True,
                 initial_point=0,
                 coeff_field=11,
                 max_simplex_size=2e5,
                 **kwargs):
        # initialize
        self.dimension = dimension
        self.dissimilarity = dissimilarity
        self.additive_interleaving = additive_interleaving
        self.multiplicative_interleaving = multiplicative_interleaving
        self.translation_function = translation_function
        self.truncation_method = truncation_method
        self.restriction_method = restriction_method
        self.resolution = resolution
        self.n_samples = n_samples
        self.isolated_points = isolated_points
        self.initial_point = initial_point
        self.coeff_field = coeff_field
        self.max_simplex_size = max_simplex_size
        self.kwargs = kwargs
        # check arguments
        self.check_arguments()
        # set defaults
        self.set_defaults()

    def check_arguments(self):
        # Interleavings for Sheehy truncation
        if self.truncation_method is 'Sheehy':
            if self.translation_function is None:
                if self.multiplicative_interleaving <= 1.0:
                    raise ValueError(
                        'Sheehy truncation requires that ' +
                        'interleaving grows faster than the identity.')
        # Sheehy restriction
        if self.restriction_method is 'Sheehy':
            if self.truncation_method not in {'Sheehy', None}:
                raise ValueError(
                    'Sheehy restriction is only compatible ' +
                    'with Sheehy truncation')
            self.truncation = 'Sheehy'
        # restriction methods
        restriction_methods = {None, 'dowker', 'Sheehy', 'alpha',
                               'no_restriction', 'Parent'}
        if (self.restriction_method not in restriction_methods):
            raise ValueError(
                'Allowed values for restriction_method are',
                restriction_methods)
        # truncation methods
        truncation_methods = {'Sheehy', 'dowker',
                              'Canonical', 'no_truncation'}
        if (self.truncation_method is not None and
                self.truncation_method not in truncation_methods):
            raise ValueError(
                'Allowed values for truncation_method are',
                truncation_methods)
        # alpha complex
        if (self.restriction_method == 'alpha'
                and self.dissimilarity not in {'alpha', None}):
            raise ValueError(
                'Alpha complexes must have dissimilarity set to alpha')
        # check dissimilarity
        if self.dissimilarity is not None:
            if self.dissimilarity not in [
                    'dowker', 'metric', 'ambient', 'alpha',
                    'filtered_complex'] + list(_METRIC_ALIAS.keys()):
                raise ValueError('Unknown dissimilarity', self.dissimilarity)
            if self.dissimilarity == 'alpha':
                self.restriction_method = 'alpha'
        allowed_kwargs = set(_METRIC_ALIAS.keys()).union({'cutoff'})
        for argument in self.kwargs.keys():
            if argument not in allowed_kwargs:
                raise ValueError('Unknown argument', argument)

    def set_defaults(self):
        if self.truncation_method is None:
            if self.restriction_method is 'no_restriction':
                self.truncation_method = 'no_truncation'
            else:
                self.truncation_method = 'dowker'
        if self.restriction_method is None:
            self.restriction_method = 'dowker'
        if self.dissimilarity is None:
            if self.restriction_method in {
                    'dowker', 'Sheehy', 'no_restriction',
                    'Parent'}:
                self.dissimilarity = 'dowker'
            elif self.restriction_method in {'alpha'}:
                self.dissimilarity = 'alpha'

    def set_nerve(self, X):
        if X is None:
            if hasattr(self, 'nerve'):
                return
            else:
                raise ValueError(
                    'The first time this function is called, X should not be None')
        self.__dict__.pop('nerve', None)
        self.__dict__.pop('homology', None)
        self.__dict__.pop('simplex_tree', None)

        if (self.restriction_method == 'no_restriction' and
                self.truncation_method == 'no_truncation'):
            self.nerve = nerves.DowkerNerveBase(
                X=X,
                dimension=self.dimension,
                dissimilarity=self.dissimilarity,
                additive_interleaving=self.additive_interleaving,
                multiplicative_interleaving=self.multiplicative_interleaving,
                translation_function=self.translation_function,
                truncation_method=self.truncation_method,
                restriction_method=self.restriction_method,
                resolution=self.resolution,
                n_samples=self.n_samples,
                isolated_points=self.isolated_points,
                initial_point=self.initial_point,
                coeff_field=self.coeff_field,
                max_simplex_size=self.max_simplex_size,
                **self.kwargs)
        elif self.restriction_method == 'alpha':
            self.nerve = nerves.AlphaNerve(
                X=X,
                dimension=self.dimension,
                dissimilarity=self.dissimilarity,
                truncation_method=self.truncation_method,
                restriction_method=self.restriction_method,
                resolution=self.resolution,
                n_samples=self.n_samples,
                isolated_points=self.isolated_points,
                initial_point=self.initial_point,
                coeff_field=self.coeff_field,
                max_simplex_size=self.max_simplex_size,
                **self.kwargs)
        else:
            if self.dissimilarity is 'ambient':
                self.nerve = nerves.DowkerNerveAmbient(
                    X=X,
                    dimension=self.dimension,
                    additive_interleaving=self.additive_interleaving,
                    multiplicative_interleaving=(
                        self.multiplicative_interleaving),
                    translation_function=self.translation_function,
                    truncation_method=self.truncation_method,
                    restriction_method=self.restriction_method,
                    resolution=self.resolution,
                    n_samples=self.n_samples,
                    initial_point=self.initial_point,
                    coeff_field=self.coeff_field,
                    max_simplex_size=self.max_simplex_size,
                    **self.kwargs)
            else:
                self.nerve = nerves.DowkerNerve(
                    X=X,
                    dimension=self.dimension,
                    dissimilarity=self.dissimilarity,
                    additive_interleaving=self.additive_interleaving,
                    multiplicative_interleaving=(
                        self.multiplicative_interleaving),
                    translation_function=self.translation_function,
                    truncation_method=self.truncation_method,
                    restriction_method=self.restriction_method,
                    resolution=self.resolution,
                    n_samples=self.n_samples,
                    isolated_points=self.isolated_points,
                    initial_point=self.initial_point,
                    coeff_field=self.coeff_field,
                    max_simplex_size=self.max_simplex_size,
                    **self.kwargs)

    def maximal_faces(self, X=None):
        self.set_nerve(X)
        self.nerve.get_maximal_faces()
        return self.nerve.maximal_faces

    def cardinality_information(self, verbose=False):
        return self.nerve.cardinality_information(verbose=verbose)

    def get_simplex_tree(self, X=None):
        self.set_nerve(X)
        if hasattr(self, 'simplex_tree'):
            return self.simplex_tree
        if not hasattr(self.nerve, 'nerve_values'):
            self.nerve.get_filtered_nerve()
        self.simplex_tree = SimplexTreeFromNerve(
            self.nerve,
            self.coeff_field)
        return self.simplex_tree

    def persistent_homology(self, X=None):
        """
        Computing persistent homology using gudhi.

        Parameters
        ----------
        X : data

        Returns
        -------
        dgms : list of ndarrays
            One array for each dimension containing birth- and
            death values.
        """
        self.get_simplex_tree(X)
        if not hasattr(self.nerve.DD, 'nearest_point_list'):
            self.nerve.DD.get_nearest_point_list()
            self.simplex_tree.nearest_point_list = (
                self.nerve.DD.nearest_point_list)
        if not hasattr(self.simplex_tree, 'dgms'):
            self.simplex_tree.persistent_homology()
        return self.simplex_tree.dgms

    def plot_persistence(self,
                         X=None,
                         plot_only=None,
                         title=None,
                         xy_range=None,
                         labels=None,
                         colormap="default",
                         size=10,
                         alpha=0.5,
                         add_multiplicity=False,
                         ax_color=np.array([0.0, 0.0, 0.0]),
                         colors=None,
                         diagonal=True,
                         lifetime=False,
                         legend=True,
                         show=False,
                         return_plot=False):
        """
        Show or save persistence diagram

        Parameters
        ----------
        X : data
        plot_only: list of numeric
            If specified, an array of only the diagrams that should be plotted.
        title: string, default is None
            If title is defined, add it as title of the plot.
        xy_range: list of numeric [xmin, xmax, ymin, ymax]
            User provided range of axes. This is useful for comparing \
            multiple persistence diagrams.
        labels: string or list of strings
            Legend labels for each diagram. \
            If none are specified, we use H_0, H_1, H_2,... by default.
        colormap: str (default : 'default')
            Any of matplotlib color palettes. \
            Some options are 'default', 'seaborn', 'sequential'.
        size: numeric (default : 10)
            Pixel size of each point plotted.
        alpha: numeric (default : 0.5)
            Transparency of each point plotted.
        add_multiplicity: boolean (default : False)
            Add the multiplicity of each point plotted. 
        ax_color: any valid matplotlib color type.
            See https://matplotlib.org/api/colors_api.html for complete API.
        colors : list of colors
            color list for different homology dimensions
        diagonal : bool (default : True)
            Plot the diagonal x=y line.
        lifetime : bool (default : False). If True, diagonal is turned False.
            Plot life time of each point instead of birth and death.  \
            Essentially, visualize (x, y-x).
        legend : bool (default : True)
            If true, show the legend.
        show : bool (default : False)
            Call plt.show() after plotting. If you are using self.plot() as \
            part of a subplot, set show=False and call plt.show() only once \
            at the end.
        return_plot : bool (default : False)
            Should plt be returned?
        """
        self.persistent_homology(X)
        plt = self.simplex_tree.plot_persistence(
            plot_only=plot_only,
            title=title,
            xy_range=xy_range,
            labels=labels,
            colormap=colormap,
            size=size,
            alpha=alpha,
            add_multiplicity=add_multiplicity,
            ax_color=ax_color,
            colors=colors,
            diagonal=diagonal,
            lifetime=lifetime,
            legend=legend,
            show=show,
            return_plot=True)
        if return_plot:
            return plt

    def components(self,
                   X=None,
                   persistence_function=0,
                   max_birth_value=np.inf):
        self.persistent_homology(X)
        return self.simplex_tree.components(
            persistence_function, max_birth_value)

    def persistence_points(self, X=None, persistence_function=0, depth=None):
        self.persistent_homology(X)
        return self.simplex_tree.persistence_points(
            persistence_function, depth)

    def cycle_components(self, X=None, persistence_function=0):
        self.persistent_homology(X)
        return self.simplex_tree.cycle_components(persistence_function)

    def cycle_representatives(self, X=None, persistence_function=0):
        self.persistent_homology(X)
        if not hasattr(self.nerve.DD, 'nearest_point_list'):
            self.nerve.DD.get_nearest_point_list()
        cycles = self.simplex_tree.cycle_representatives(
            persistence_function=persistence_function)
        return [[self.nerve.truncation.point_sample[list(edge)] for
                 edge in cycle] for cycle in cycles]
