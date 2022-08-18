"""
Python routines used to test claims in the papers
*** SDN = Sparse Dowker Nerves ***
*** SFN = Sparse Filtered Nerves ***
Copyright: Applied Topology Group at University of Bergen.
No commercial use of the software is permitted without proper license.
"""
import numbers
import itertools
import collections
import collections.abc
import operator
import functools
import numpy as np
import scipy.special
from . import dissimilarities
from . import truncations
from . import restrictions
from scipy.spatial import Delaunay


class DowkerNerveBase(object):
    """Documentation for ClassName"""

    def __init__(
        self,
        X,
        dimension=1,
        dissimilarity="dowker",
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
        **kwargs
    ):
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
        # set dowker dissimilarity
        self.set_dowker_dissimilarity(X)

    def interleaving_line(self, n_points=100):
        return None

    def reset(self):
        self.__dict__.pop("nerve", None)
        self.__dict__.pop("restriction", None)
        self.__dict__.pop("truncation", None)

    def set_dowker_dissimilarity(self, X):
        if X is None:
            return
        self.reset()

        self.DD = dissimilarities.Dissimilarity(
            X,
            dissimilarity=self.dissimilarity,
            resolution=self.resolution,
            n_samples=self.n_samples,
            isolated_points=self.isolated_points,
            initial_point=self.initial_point,
            **self.kwargs
        )

    def get_maximal_faces(self, X=None):
        """Maximal faces of nerve of dowker matrix without filtration values."""
        self.set_dowker_dissimilarity(X)
        maximal_faces = set()
        for index in range(self.DD.shape[1]):
            maximal_faces.add(self.DD.witness_face(index))
        self.maximal_faces = eliminate_subsets(maximal_faces)

    def get_nerve(self, X=None):
        """Calculate underlynig simplicial complex"""
        self.set_dowker_dissimilarity(X)
        if not hasattr(self, "maximal_faces"):
            self.get_maximal_faces()
        if self.actual_max_simplex_size() > self.max_simplex_size:
            raise ValueError(
                "Filtered complex will be of size at least "
                + str(self.actual_max_simplex_size())
                + ".\nIf you want to compute it "
                + "increase the parameter\n"
                + "max_simplex_size from its current value "
                + str(self.max_simplex_size)
                + "\nto something bigger than "
                + str(self.actual_max_simplex_size())
                + "."
            )
        return get_nerve_from_maximal_faces(self.maximal_faces, self.dimension)

    def get_filtered_nerve(self, X=None):
        """Compute filtration values of underlying simplicial complex"""
        self.set_dowker_dissimilarity(X)
        infinity_nerve = self.get_nerve()
        self.nerve_values = np.array(
            [
                self.DD.filtration_value(list(simplex))
                for simplex in infinity_nerve
            ]
        )
        # sort filtered nerve
        self.nerve, self.nerve_values = sort_nerve(
            infinity_nerve, self.nerve_values
        )
        self.max_filtration_value = self.DD.max()

    def cardinality_information(self, verbose=True):
        if not hasattr(self, "DD"):
            if verbose:
                print("No data added")
            return {}
        if hasattr(self, "nerve"):
            unreduced_cardinality = len(self.nerve)
            if verbose:
                print("Nerve has cardinality ", unreduced_cardinality)
            return {"unreduced_cardinality": unreduced_cardinality}
        else:
            if verbose:
                print("Nerve not yet computed")
            return {}

    def actual_max_simplex_size(self):
        self.max_cardinality = max([len(x) for x in self.maximal_faces])
        return np.sum(
            [
                scipy.special.binom(self.max_cardinality, _i)
                for _i in range(1, self.dimension + 3)
            ]
        )


class AlphaNerve(DowkerNerveBase):
    def get_maximal_faces(self, X=None):
        """Delaunay maximal faces"""
        tri = Delaunay(self.DD.coords)
        self.maximal_faces = set(frozenset(face) for face in tri.simplices)

    def cardinality_information(self, verbose=True):
        if not hasattr(self, "DD"):
            if verbose:
                print("No data added")
            return {}
        cardinality = {
            "unreduced_cardinality": np.sum(
                [
                    scipy.special.binom(self.DD.len(), _i)
                    for _i in range(1, self.dimension + 3)
                ]
            )
        }
        if verbose:
            print(
                "Unreduced nerve has cardinality "
                + str(cardinality["unreduced_cardinality"])
            )
        if hasattr(self, "nerve"):
            cardinality["reduced_cardinality"] = len(self.nerve)
            if verbose:
                print(
                    "Reduced nerve has cardinality "
                    + str(cardinality["reduced_cardinality"])
                )
        elif hasattr(self, "maximal_faces"):
            cardinality["maximal_faces"] = self.actual_max_simplex_size()
            if verbose:
                print(
                    "The "
                    + str(self.dimension + 1)
                    + "-skeletion of the biggest simplex in the nerve "
                    + "has cardinality "
                    + str(cardinality["maximal_faces"])
                )
        cardinality["dimension"] = self.max_cardinality - 1
        if verbose:
            print(
                "The simplicial complex has dimension",
                cardinality["dimension"],
            )
        return cardinality


class DowkerNerve(DowkerNerveBase):
    def set_dowker_dissimilarity(self, X):
        super().set_dowker_dissimilarity(X)
        self.set_translation_function()

    def set_translation_function(self):
        if self.translation_function is None:
            self.translation_function = translation_function_from_interleaving(
                a=self.additive_interleaving,
                c=self.multiplicative_interleaving,
                threshold=self.kwargs.get("threshold", None),
                cover_radius=self.DD.cover_radius,
            )

    def cardinality_information(self, verbose=True):
        if not hasattr(self, "DD"):
            print("No data added")
            return {}
        cardinality = {
            "unreduced_cardinality": np.sum(
                [
                    scipy.special.binom(self.DD.len(), _i)
                    for _i in range(1, self.dimension + 3)
                ]
            )
        }
        if verbose:
            print(
                "Unreduced nerve has cardinality "
                + str(cardinality["unreduced_cardinality"])
            )
        if hasattr(self, "nerve"):
            cardinality["reduced_cardinality"] = len(self.nerve)
            if verbose:
                print(
                    "Reduced nerve has cardinality "
                    + str(cardinality["reduced_cardinality"])
                )
        elif hasattr(self, "maximal_faces"):
            cardinality["maximal_faces"] = self.actual_max_simplex_size()
            if verbose:
                print(
                    "The "
                    + str(self.dimension + 1)
                    + "-skeletion of the biggest simplex in the nerve "
                    + "has cardinality "
                    + str(cardinality["maximal_faces"])
                )
        if hasattr(self, "max_cardinality"):
            cardinality["dimension"] = self.max_cardinality - 1
        if verbose:
            print(
                "The simplicial complex has dimension",
                cardinality["dimension"],
            )
        return cardinality

    def get_maximal_faces(self, X=None):
        """Maximal faces of nerve of dowker matrix without filtration values."""
        self.set_dowker_dissimilarity(X)
        if not hasattr(self, "restriction"):
            self.get_restriction()
        restriction = self.restriction
        self.slope = get_slope(
            restriction.parent_point_list, restriction.restriction_times
        )
        maximal_faces = frozenset()
        restriction_times = restriction.restriction_times
        for l in np.argsort(restriction_times)[::-1]:
            l_faces = []
            rf = restriction.restriction_face_indices(l)
            for w in restriction.l_witnesses(l, self.DD.X):
                face = restriction.witness_face(w, l, rf)
                slope_vertices = restriction.slope_vertices(self.slope, w, rf)
                non_slope_vertices = restriction.non_slope_vertices(
                    self.slope, w, rf
                )
                face.intersection_update(
                    slope_vertices.union(non_slope_vertices)
                )
                l_faces.append(frozenset(face))
            l_faces = eliminate_subsets(list(frozenset(l_faces)))
            maximal_faces = maximal_faces.union(l_faces)
        self.maximal_faces = eliminate_subsets(list(maximal_faces))

    def interleaving_line(self, n_points=100):
        """
        Calculate interleaving guarantee
        """
        ymax = 1.05 * (self.max_filtration_value)
        xmax = self.max_filtration_value
        x_points = np.linspace(0, 1.01 * xmax, n_points)
        y_points = np.maximum(
            self.translation_function(x_points + self.DD.cover_radius),
            x_points + self.truncation.offset,
        )
        y_points[x_points > xmax] = ymax
        return x_points, y_points

    def get_truncation_times(self, X=None):
        """Calculate truncation times"""
        self.set_dowker_dissimilarity(X)
        DD = self.__dict__.pop("DD")
        self.truncation = truncations.Truncation(
            DD,
            translation_function=self.translation_function,
            truncation_method=self.truncation_method,
            n_samples=self.n_samples,
            initial_point=self.initial_point,
        )
        self.DD = self.truncation.DD

    def get_restriction(self, X=None):
        """Calculate restriction times"""
        self.set_dowker_dissimilarity(X)
        if not hasattr(self, "truncation_times"):
            self.get_truncation_times()
        self.restriction = restrictions.Restriction(
            self.truncation, self.restriction_method
        )


class DowkerNerveAmbient(DowkerNerve):
    def __init__(
        self,
        X,
        dimension=1,
        dissimilarity="dowker",
        translation_function=None,
        additive_interleaving=0.0,
        multiplicative_interleaving=1.0,
        truncation_method=None,
        restriction_method=None,
        resolution=0.0,
        n_samples=None,
        initial_point=0,
        coeff_field=11,
        max_simplex_size=2e5,
        **kwargs
    ):
        if isinstance(additive_interleaving, numbers.Number):
            additive_interleaving *= 2
        else:
            additive_interleaving = tuple(2 * a for a in additive_interleaving)
        if "threshold" in kwargs.keys():
            kwargs["threshold"] *= 2
        super().__init__(
            X,
            dimension=dimension,
            dissimilarity=dissimilarity,
            translation_function=translation_function,
            additive_interleaving=additive_interleaving,
            multiplicative_interleaving=multiplicative_interleaving,
            truncation_method=truncation_method,
            restriction_method=restriction_method,
            resolution=resolution,
            n_samples=n_samples,
            isolated_points=True,
            initial_point=initial_point,
            coeff_field=coeff_field,
            max_simplex_size=max_simplex_size,
            **kwargs
        )

    def set_dowker_dissimilarity(self, X):
        if X is None:
            return
        self.reset()
        self.DD = dissimilarities.DissimilarityAmbient(
            X,
            resolution=self.resolution,
            n_samples=self.n_samples,
            initial_point=self.initial_point,
        )
        self.set_translation_function()

    def interleaving_line(self, n_points=100):
        """
        Calculate interleaving guarantee
        """
        ymax = 1.05 * (self.max_filtration_value)
        xmax = self.max_filtration_value
        x_points = np.linspace(0, 1.01 * xmax, n_points)
        y_points = np.maximum(
            0.5
            * self.translation_function(2 * (x_points + self.DD.cover_radius)),
            x_points + self.truncation.offset,
        )
        y_points[x_points > xmax] = ymax
        return x_points, y_points

    def get_filtered_nerve(self, X=None):
        """Compute filtration values of underlying simplicial complex"""
        super().get_filtered_nerve(X)
        finite_faces = np.array(
            [
                self.nerve_values[index]
                <= np.min(self.restriction.restriction_times[list(simplex)])
                for index, simplex in enumerate(self.nerve)
            ]
        )
        self.nerve = [
            simplex
            for index, simplex in enumerate(self.nerve)
            if finite_faces[index]
        ]
        # self.nerve = self.nerve[finite_faces]
        self.nerve_values = self.nerve_values[finite_faces]

    def get_restriction(self, X=None):
        """Calculate restriction times"""
        self.set_dowker_dissimilarity(X)
        if not hasattr(self, "truncation_times"):
            self.get_truncation_times()

        self.restriction = restrictions.Restriction(
            self.truncation, self.restriction_method
        )
        # a_dim = self.DD.coords.shape[1]
        self.restriction.restriction_times *= 2
        # np.sqrt(2 * a_dim / (a_dim + 1))


def powerset(some_set, max_card, min_card=1):
    return itertools.chain(
        *[
            list(itertools.combinations(some_set, dim + 1))
            for dim in range(min_card - 1, max_card)
        ]
    )


def get_nerve_from_maximal_faces(maximal_faces, dimension):
    """Returns nerve given a list of maximal faces

    Parameters
    -----------
    maximal_face :
    dimension : int
        Maximal homology dimension

    Returns
    -----------
    nerve : ndarray of lists of integers
        List of all simplices up to desired dimension
    """
    # This needs to be optimized.
    # First we find the maximal faces of the dimension + 1 skeleton.
    # Then we fill in the rest of the faces.
    faces = (
        (
            frozenset(face)
            for face in powerset(
                maximal_face,
                max_card=dimension + 2,
                min_card=min(len(maximal_face), dimension + 2),
            )
        )
        for maximal_face in maximal_faces
    )
    faces = frozenset(itertools.chain.from_iterable(faces))
    faces = (
        (
            frozenset(face)
            for face in powerset(maximal_face, max_card=dimension + 2)
        )
        for maximal_face in faces
    )
    return np.array(list(frozenset(itertools.chain.from_iterable(faces))))


def sort_nerve(nerve, nerve_values):
    # sort filtered nerve
    sorted_nerve = list(zip(nerve, nerve_values))
    sorted_nerve.sort(key=lambda obj: len(obj[0]))
    sorted_nerve.sort(key=lambda obj: obj[1])
    sorted_nerve, nerve_values = list(zip(*sorted_nerve))
    return sorted_nerve, np.array(nerve_values)


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
    if not isinstance(sequence_of_sets, collections.abc.Sequence):
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
    out = [
        s
        for s in sequence_of_sets
        if s
        and is_power_of_two(
            functools.reduce(
                operator.and_, (sets_containing_element[x] for x in s)
            )
        )
    ]
    return out


def translation_function_from_interleaving(
    a,  # additive interleaving
    c,  # multiplicative interleaving
    threshold=None,
    cover_radius=0,
):
    if threshold is not None:
        if isinstance(a, numbers.Number):
            a0, a1 = (a, a + threshold)
        else:
            a0, a1 = (a[0], a[1] + threshold)

    def translation_function(t):
        if threshold is not None:
            additive_line = a0 + t * (a1 - a0) / threshold
            threshold_line = c * t + a1 - threshold * c
            ft = np.maximum(threshold_line, additive_line)
            ft = np.maximum(ft, t)
        else:
            ft = a + t * c
        return ft

    return translation_function


def get_slope(parent_point_list, death_times):
    forest = get_parent_forest(parent_point_list)
    slope = np.zeros(len(parent_point_list), dtype=bool)
    for index in range(len(parent_point_list)):
        if len(forest[index]) > 0 and death_times[index] < np.inf:
            slope[index] = death_times[index] > np.max(
                death_times[list(forest[index])]
            )
        else:
            slope[index] = True
    return slope


def get_parent_forest(parent_point_list):
    forest = {index: set() for index in range(len(parent_point_list))}
    for index, parent in enumerate(parent_point_list):
        if index != parent:
            forest[parent].add(index)
    return forest


def iterative_topological_sort(graph, start):
    # taken from https://stackoverflow.com/questions/47192626/
    # deceptively-simple-implementation-of-topological-sorting-in-python
    seen = set()
    stack = []
    order = []
    q = [start]
    while q:
        v = q.pop()
        if v not in seen:
            seen.add(v)
            q.extend(graph[v])

            while stack and v not in graph[stack[-1]]:
                order.append(stack.pop())
            stack.append(v)

    return order + stack[::-1]
