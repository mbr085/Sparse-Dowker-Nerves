"""
Python routines used to test claims in the papers
*** SDN = Sparse Dowker Nerves ***
*** SFN = Sparse Filtered Nerves *** 
Copyright: Applied Topology Group at University of Bergen.
No commercial use of the software is permitted without proper license.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import binom
from inspect import signature
from . import dowker_functions as df


class Dowker_Base:
    """Dowker base persistent homology class"""

    def __init__(self,
                 dimension=1,
                 homology_method='dual',
                 max_simplex_size=2e5):
        """
        Initalize a new dowker_base object.

        Parameters
        ----------
        dimension : maximal persistence dimension
        homology_method : what method should be used to
            calculate persistent homology (see phat)
        max_simplex_size : maximum size biggest simplex 
            in the nerve to calculate persistent homology
        """
        self.dimension = dimension
        self.homology_method = homology_method
        self.max_simplex_size = max_simplex_size

    @property
    def dimension(self):
        return self.__dimension

    @dimension.setter
    def dimension(self, dimension):
        self.reset()
        self.__dimension = dimension

    def get_param_names(self):
        """Get parameter names"""
        return list(signature(self.__init__).parameters.keys())

    def get_params(self):
        """Get parameters"""
        return {key: getattr(self, key, None) for
                key in self.get_param_names()}

    def __repr__(self):
        class_name = self.__class__.__name__
        dowker_params = ', '.join([key + '=' + str(value)
                                   for key, value in
                                   self.get_params().items()])
        return '%s(%s)' % (class_name, dowker_params)

    def clone_or_reset(self, other):

        reset_list = {'_Dowker_Base__dimension',
                      'homology_method',
                      'max_simplex_size'}
        # print(reset_list)
        # print(set(self.__dict__.keys()))
        # if reset_list.issubset(set(self.__dict__.keys())):
        other.__dict__ = {property: self.__dict__[
            property] for property in
            reset_list.intersection(set(self.__dict__.keys()))}

    def clone(self):
        new_dowker = self.__class__()
        self.clone_or_reset(new_dowker)
        return new_dowker

    def reset(self):
        self.clone_or_reset(self)

    def update_X(self, X=None):
        if X is not None:
            self.reset()
            self.X = X

    def get_maximal_infinity_faces(self, X=None):
        """Maximal faces of nerve of dowker matrix without filtration values"""
        self.update_X(X)
        self.maximal_faces = [frozenset(range(len(self.X)))]
        max_cardinality = len(self.X) #max([len(x) for x in self.maximal_faces])
        self.actual_max_simplex_size = np.sum(
            [binom(max_cardinality, _i)
             for _i in range(1, self.dimension + 3)])

    def get_infinity_nerve(self, X=None):
        """Calculate underlynig simplicial complex"""
        self.update_X(X)
        self.max_filtration_value = np.max(self.X)
        if not hasattr(self, 'maximal_faces'):
            self.get_maximal_infinity_faces()
        if self.actual_max_simplex_size > self.max_simplex_size:
            raise ValueError('Filtered complex will be of size at least ' +
                             str(self.actual_max_simplex_size) +
                             '.\nIf you want to compute it ' +
                             'increase the parameter\n' +
                             'max_simplex_size from its current value ' +
                             str(self.max_simplex_size) +
                             '\nto something bigger than ' +
                             str(self.actual_max_simplex_size) + '.')
        return df.get_nerve_from_maximal_faces(
            self.maximal_faces, self.dimension)

    def get_nerve(self, X=None):
        """Compute filtration values of underlying simplicial complex"""
        self.update_X(X)
        infinity_nerve = self.get_infinity_nerve()
        self.nerve_values = np.array([
            df.get_intrinsic_filtration_value(
                list(simplex), self.X)
            for simplex in infinity_nerve])

        # sort filtered nerve
        self.filtered_nerve, self.nerve_values = df.sort_filtered_nerve(
            infinity_nerve, self.nerve_values)

    def persistence(self, homology_method=None, X=None):
        """
        Use phat to compute persistence pairs.
        """
        if homology_method is not None:
            self.homology_method = homology_method
        self.update_X(X)
        if not hasattr(self, 'filtered_nerve'):
            self.get_nerve()
        if self.homology_method in ['image_twist', 'dual_image_twist']:
            self.homology = df.get_persistent_homology(
                self.filtered_nerve,
                self.nerve_values,
                method=self.homology_method)
            return None

        try:
            import phat
        except ModuleNotFoundError:
            print('Warning: phat not found\n' +
                  'falling back to built in persistence computation')
            self.homology = df.get_persistent_homology(
                self.filtered_nerve,
                self.nerve_values,
                method=self.homology_method)
            return None

        boundary_matrix = phat.boundary_matrix(
            representation=phat.representations.vector_vector)
        b_matrix = df.get_boundary_matrix(self.filtered_nerve)
        boundary_matrix.columns = b_matrix
        # compute persistence
        if self.homology_method == 'dual':
            pairs = boundary_matrix.compute_persistence_pairs_dualized()
        else:
            pairs = boundary_matrix.compute_persistence_pairs()
        pers = [[b_matrix[p[0]][0], self.nerve_values[p[0]],
                 self.nerve_values[p[1]]] for
                p in pairs]
        pers = [per for per in pers if per[1] < per[2]]
        pers.sort(key=lambda pair: (pair[1] - pair[2], pair[0]))
        self.homology = np.array(pers)

    def get_interleaving_lines(self, X=None,
                               n_points=100):
        """
        Calculate interleaving guarantee
        """
        self.update_X(X)
        if not hasattr(self, 'homology'):
            self.persistence()
        self.interleaving_lines = []

    def plot_persistence(self,
                         X=None,
                         title="Persistence diagram",
                         ticks=None,
                         xmax=None,
                         ymax=None,
                         zoom=False,
                         alpha=0.5,
                         ms=10):
        """Plot persistence diagram"""
        self.update_X(X)
        if not hasattr(self, 'interleaving_lines'):
            self.get_interleaving_lines()
        if len(self.homology) == 0:
            print("The persistence diagram is empty.")
            return
        # extract data
        pers = pd.DataFrame(dict(dim=self.homology[:, 0].astype("int"),
                                 birth=self.homology[:, 1],
                                 death=self.homology[:, 2]))
        pers = pers.loc[pers.loc[:, "death"] != pers.loc[:, "birth"], :]
        groups = pers.groupby("dim")
        # plot
        fig, ax = plt.subplots()
        for name, group in groups:
            ax.plot(group.birth, group.death, marker="o",
                    linestyle="", ms=ms, alpha=alpha,
                    color='C' + str(name),
                    label=name)
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        ax.set_title(title)
        fig.tight_layout()
        if not zoom:
            xmax = self.max_filtration_value
            ymax = self.max_filtration_value
        plt.xlim(xmin=0, xmax=xmax)
        plt.ylim(ymin=0, ymax=ymax)
        # diagonal
        t = np.array([0, self.max_filtration_value])
        ax.plot(t, t, color="black", linewidth=1.0)

        # interleaving lines
        for (x_points, y_points) in self.interleaving_lines:
            ax.plot(x_points, y_points, color="black", linewidth=1.0)

        if ticks is not None:
            ax.xaxis.set_ticks(ticks)
            ax.yaxis.set_ticks(ticks)
            ax.grid()
        ax.legend(loc="lower right")
        plt.show()

        return None

    def cardinality_information(self, X=None):
        self.update_X(X)
        card_unred = np.sum([binom(len(self.X), _i) 
                             for _i in range(1, self.dimension + 3)])
        print('Unreduced nerve has cardinality ' + str(int(card_unred)))
        if hasattr(self, 'filtered_nerve'):
            card_red = len(self.filtered_nerve)
            print('Reduced nerve has cardinality ' + str(int(card_red)))
        elif hasattr(self, 'actual_max_simplex_size'):
            card_dim = self.dimension + 1
            card_actual = self.actual_max_simplex_size
            print('The ' +
                  str(int(card_dim)) +
                  '-skeleton of the biggest simplex in the nerve ' +
                  'has cardinality ' + str(int(card_actual)))

