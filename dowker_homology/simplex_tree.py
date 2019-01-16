'''
Python routines used to test claims in the papers
*** SDN = Sparse Dowker Nerves ***
*** SFN = Sparse Filtered Nerves ***
Copyright: Applied Topology Group at University of Bergen.
No commercial use of the software is permitted without proper license.
'''
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import gudhi
from itertools import cycle
import matplotlib as mpl


def SimplexTreeFromNerve(
        nerve,
        coeff_field=11,
        min_persistence=0):

    nerve.get_filtered_nerve()
    geometric_dimension = max(
        (len(x) for x in nerve.maximal_faces)) - 1
    persistence_dim_max = geometric_dimension <= nerve.dimension

    simplex_tree = SimplexTree(
        interleaving_line=nerve.interleaving_line,
        coeff_field=coeff_field,
        persistence_dim_max=persistence_dim_max,
        min_persistence=min_persistence)

    for index, simplex in enumerate(nerve.nerve):
        simplex_tree.insert(simplex=list(simplex),
                            filtration=nerve.nerve_values[index])

    return simplex_tree


class SimplexTree(gudhi.SimplexTree):

    def __init__(self,
                 interleaving_line=None,
                 coeff_field=11,
                 persistence_dim_max=False,
                 min_persistence=0):
        super().__init__()
        if interleaving_line is None:
            interleaving_line = empty_interleaving_line
        self.interleaving_line = interleaving_line
        self.coeff_field = coeff_field
        self.persistence_dim_max = persistence_dim_max
        self.min_persistence = min_persistence

    def persistent_homology(
            self,
            interleaving_line=None,
            coeff_field=None,
            persistence_dim_max=None,
            min_persistence=None):
        """
        Computing persistent homology using gudhi.

        Parameters
        ----------
        nerve : object of Nerve type

        Returns
        -------
        dgms : list of ndarrays
            One array for each dimension containing birth- and
            death values.
        """
        if interleaving_line is None:
            interleaving_line = self.interleaving_line
        elif not interleaving_line:
            interleaving_line = empty_interleaving_line
        if coeff_field is None:
            coeff_field = self.coeff_field
        if persistence_dim_max is None:
            persistence_dim_max = self.persistence_dim_max
        if min_persistence is None:
            min_persistence = self.min_persistence
        persistence = self.persistence(
            homology_coeff_field=coeff_field,
            min_persistence=min_persistence,
            persistence_dim_max=persistence_dim_max)
        max_homology_dimension = self.dimension() - 1 + persistence_dim_max
        self.dgms = [[] for dim in range(max_homology_dimension + 1)]
        for hclass in persistence:
            self.dgms[hclass[0]].append((hclass[1][0], hclass[1][1]))
        self.dgms = [np.array(classes) for classes in self.dgms]
        return self.dgms

    def plot_persistence(
            self,
            plot_only=None,
            title=None,
            xy_range=None,
            labels=None,
            colormap="default",
            size=10,
            alpha=0.5,
            ax_color=np.array([0.0, 0.0, 0.0]),
            colors=None,
            diagonal=True,
            lifetime=False,
            legend=True,
            show=False,
            return_plot=False
    ):
        """
        Show or save persistence diagram

        Parameters
        ----------
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
        ax_color: any valid matplotlib color type.
            See https://matplotlib.org/api/colors_api.html for complete API.
        diagonal: bool (default : True)
            Plot the diagonal x=y line.
        lifetime: bool (default : False). If True, diagonal is turned to False.
            Plot life time of each point instead of birth and death.  \
            Essentially, visualize (x, y-x).
        legend: bool (default : True)
            If true, show the legend.
        show: bool (default : False)
            Call plt.show() after plotting. If you are using self.plot() as \
            part of a subplot, set show=False and call plt.show() only once \
            at the end.
        """
        if not hasattr(self, 'dgms'):
            self.persistent_homology()
        if hasattr(self, 'interleaving_line'):
            interleaving_line = self.interleaving_line
        else:
            interleaving_line = empty_interleaving_line
        plt = plot_dgms(
            diagrams=self.dgms,
            interleaving_line=interleaving_line,
            plot_only=plot_only,
            title=title,
            xy_range=xy_range,
            labels=labels,
            colormap=colormap,
            size=size,
            ax_color=ax_color,
            colors=colors,
            diagonal=diagonal,
            lifetime=lifetime,
            legend=legend,
            show=show,
            alpha=alpha
        )

        if return_plot:
            return plt


def plot_dgms(
        diagrams,
        interleaving_line=None,
        plot_only=None,
        title=None,
        xy_range=None,
        labels=None,
        colormap="default",
        size=10,
        ax_color=np.array([0.0, 0.0, 0.0]),
        colors=None,
        diagonal=True,
        lifetime=False,
        legend=True,
        show=False,
        alpha=0.5,
        rips_dimension=None,
):
    # Originally from https://github.com/scikit-tda/ripser.py/blob/master/\
    # ripser/ripser.py
    xlabel, ylabel = "Birth", "Death"

    if labels is None:
        # Provide default labels for diagrams if using self.dgm_
        labels = [
            "$H_0$",
            "$H_1$",
            "$H_2$",
            "$H_3$",
            "$H_4$",
            "$H_5$",
            "$H_6$",
            "$H_7$",
            "$H_8$",
        ]

    if not isinstance(diagrams, list):
        # Must have diagrams as a list for processing downstream
        diagrams = [diagrams]

    if plot_only is None:
        plot_only = [i for i, dgm in enumerate(diagrams) if len(dgm) > 0]
    if not isinstance(plot_only, list):
        plot_only = [plot_only]

    if not isinstance(labels, list):
        labels = [labels] * len(diagrams)

    if colors is None:
        mpl.style.use(colormap)
        colors = cycle(["C0", "C1", "C2", "C3", "C4",
                        "C5", "C6", "C7", "C8", "C9"])
        colors = [next(colors) for i in range(len(diagrams))]
        colors = [colors[i] for i in plot_only]

    diagrams = [diagrams[i] for i in plot_only]
    labels = [labels[i] for i in plot_only]
    # Construct copy with proper type of each diagram
    # so we can freely edit them.
    diagrams = [dgm.astype(np.float32, copy=True) for dgm in diagrams]

    # find min and max of all visible diagrams
    concat_dgms = np.concatenate(diagrams).flatten()
    has_inf = np.any(np.isinf(concat_dgms))
    finite_dgms = concat_dgms[np.isfinite(concat_dgms)]

    if not xy_range:
        # define bounds of diagram
        ax_min, ax_max = np.min(finite_dgms), np.max(finite_dgms)
        x_r = ax_max - ax_min

        # Give plot a nice buffer on all sides.
        # ax_range=0 when only one point,
        buffer = 1 if xy_range == 0 else x_r / 5

        ax = ax_min - buffer / 2
        bx = ax_max + buffer

        ay, by = ax, bx
    else:
        ax, bx, ay, by = xy_range
    if max(ax - bx, ay - by) >= 0:
        raise ValueError('Please specify a non-degenerate xy_range')
    yr = by - ay

    if rips_dimension is not None:
        interleaving_line = rips_interleaving_line(rips_dimension, ax, bx)
    if interleaving_line() is not None:
        interleaving_line_lists = list(interleaving_line())
    else:
        interleaving_line_lists = None

    if lifetime:

        # Don't plot landscape and diagonal at the same time.
        diagonal = False

        # reset y axis so it doesn't go much below zero
        ay = -yr * 0.05
        by = ay + yr

        # set custom ylabel
        ylabel = "Lifetime"

        # set diagrams to be (x, y-x)
        for dgm in diagrams:
            dgm[:, 1] -= dgm[:, 0]
        if interleaving_line_lists is not None:
            interleaving_line_lists[1] -= interleaving_line_lists[0]

        # plot horizon line
        plt.plot([ax, bx], [0, 0], c=ax_color)

    # Plot diagonal
    if diagonal:
        plt.plot([ax, bx], [ax, bx], "-", c=ax_color)

    # Plot interleaving line
    if interleaving_line_lists is not None:
        x_points, y_points = interleaving_line_lists
        # color="black", linewidth=1.0)
        plt.plot(x_points, y_points, "--", c=ax_color)

    # Plot inf line
    b_inf = ay + yr * 0.95
    if has_inf:
        # put inf line slightly below top

        plt.plot([ax, bx], [b_inf, b_inf], "--", c="k", label=r"$\infty$")

        # convert each inf in each diagram with b_inf
        for dgm in diagrams:
            dgm[np.isinf(dgm)] = b_inf

    plot_h0 = True
    if plot_h0:
        first_dgm = 0
    else:
        first_dgm = 1
    # Plot each diagram
    for dgm, color, label in list(zip(diagrams, colors, labels))[first_dgm:]:

        # plot persistence pairs
        plt.plot(dgm[:, 0], dgm[:, 1], ms=size, color=color,
                 label=label, linestyle='',  # edgecolor="none",
                 alpha=alpha, marker='o')

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    plt.xlim([ax, bx])
    plt.ylim([ay, by])

    if title is not None:
        plt.title(title)

    if legend is True:
        plt.legend(loc="lower right")

    if show is True:
        plt.show()

    return plt


def rips_interleaving_line(n, ax, bx):
    def interleaving_line():
        dim_const = np.sqrt(2 * n / (n + 1))
        return np.array([ax, bx / dim_const]), np.array([ax, bx])
    return interleaving_line


def persistence_clusters(persistence_threshold,
                         max_birth_value,
                         simplex_tree):
    # simplex_tree.persistence(
    #     homology_coeff_field=2,
    #     min_persistence=0,
    #     persistence_dim_max=0)
    if not hasattr(simplex_tree, 'dgms'):
        simplex_tree.persistent_homology()
    filtration_values = dict(
        (frozenset(face), value)
        for (face, value) in simplex_tree.get_skeleton(1))
    filtration_values[frozenset()] = np.inf
    birth_points = []
    death_values = []
    birth_values = []
    for pair in simplex_tree.persistence_pairs():
        if len(pair[0]) == 1:
            birth = frozenset(pair[0])
            death = frozenset(pair[1])
            birth_value = filtration_values[birth]
            death_value = filtration_values[death]
            if (death_value - birth_value >= persistence_threshold
                and
                    birth_value <= max_birth_value):
                birth_points.append(pair[0][0])
                birth_values.append(birth_value)
                death_values.append(death_value)
    cluster_dta = dict()
    bd_clusters = {birth_point: dict() for birth_point in birth_points}
    edges = [list(face) for face in filtration_values.keys() if len(face) == 2]
    for death_value in death_values:
        graph = nx.Graph()
        selected_edges = [
            edge for edge in edges
            if filtration_values[frozenset(edge)] < death_value]
        graph.add_edges_from(selected_edges)
        graph.add_nodes_from(birth_points)
        born_points = [birth_point for birth_point, birth_values in
                       zip(birth_points, birth_values) if
                       birth_value <= death_value]
        for birth_point in born_points:
            selected_cluster = frozenset(
                nx.shortest_path(graph, birth_point).keys())
            if selected_cluster in cluster_dta.keys():
                cluster_dta[selected_cluster].append(
                    (birth_point, death_value))
            else:
                cluster_dta[selected_cluster] = [
                    (birth_point, death_value)]
            bd_clusters[birth_point][death_value] = selected_cluster
    core_points = {}
    for cluster, dta in cluster_dta.items():
        core_points[cluster] = frozenset(
            (birth_point for birth_point, death_value in dta))

    return [list(bd_clusters[birth_point][max(
        (death_value for death_value, cluster in
         bd_clusters[birth_point].items() if
         len(core_points[cluster]) <= 1))]) for
        birth_point in birth_points]


def representing_cycles(persistence_threshold,
                        simplex_tree,
                        coeff_field=11):
    # simplex_tree.persistence(
    #     homology_coeff_field=coeff_field,
    #     min_persistence=0,
    #     persistence_dim_max=1)
    if not hasattr(simplex_tree, 'dgms'):
        simplex_tree.persistent_homology()
    filtration_values = dict(
        (frozenset(face), value)
        for (face, value) in simplex_tree.get_skeleton(2))
    filtration_values[frozenset()] = np.inf
    birth_edges = []
    persistence_values = {}
    for pair in simplex_tree.persistence_pairs():
        if len(pair[0]) == 2:
            birth = frozenset(pair[0])
            death = frozenset(pair[1])
            persistence = filtration_values[death] - filtration_values[birth]
            if persistence >= persistence_threshold:
                birth_edges.append(list(birth))
                persistence_values[frozenset(list(birth))] = persistence
    subgraphs = []
    birth_edges.sort(key=lambda x: -persistence_values[frozenset(x)])
    edges = [list(face) for face in filtration_values.keys() if len(face) == 2]
    for birth_edge in birth_edges:
        graph = nx.Graph()
        selected_edges = [
            edge for edge in edges
            if filtration_values[frozenset(edge)] <
            filtration_values[frozenset(birth_edge)]]
        graph.add_edges_from(selected_edges)
        graph.add_nodes_from(birth_edge)
        try:
            subgraphs.append(list(graph.subgraph(
                nx.shortest_path(
                    graph, birth_edge[0], birth_edge[1])).edges()) +
                [birth_edge])
        except nx.NetworkXNoPath:
            graph = nx.Graph()
            selected_edges = [
                edge for edge in edges
                if filtration_values[frozenset(edge)] <=
                filtration_values[frozenset(birth_edge)]]
            graph.add_edges_from(selected_edges)
            graph.remove_edge(*birth_edge)
            graph.add_nodes_from(birth_edge)
            try:
                subgraphs.append(list(graph.subgraph(
                    nx.shortest_path(
                        graph, birth_edge[0], birth_edge[1])).edges()) +
                    [birth_edge])
            except nx.NetworkXNoPath:
                pass
    return subgraphs


def cycle_clusters(persistence_threshold,
                   simplex_tree,
                   coeff_field=11):
    # simplex_tree.persistence(
    #     homology_coeff_field=11,
    #     min_persistence=0,
    #     persistence_dim_max=1)
    if not hasattr(simplex_tree, 'dgms'):
        simplex_tree.persistent_homology()
    filtration_values = dict(
        (frozenset(face), value)
        for (face, value) in simplex_tree.get_skeleton(2))
    filtration_values[frozenset()] = np.inf
    birth_edges = []
    persistence_values = {}
    for pair in simplex_tree.persistence_pairs():
        if len(pair[0]) == 2:
            birth = frozenset(pair[0])
            death = frozenset(pair[1])
            persistence = filtration_values[death] - filtration_values[birth]
            if persistence >= persistence_threshold:
                birth_edges.append(list(birth))
                persistence_values[frozenset(list(birth))] = persistence
    birth_edges.sort(key=lambda x: -persistence_values[frozenset(x)])
    clusters = []
    edges = [list(face) for face in filtration_values.keys() if len(face) == 2]
    for birth_edge in birth_edges:
        graph = nx.Graph()
        selected_edges = [
            edge for edge in edges
            if filtration_values[frozenset(edge)] <=
            filtration_values[frozenset(birth_edge)]]
        graph.add_edges_from(selected_edges)
        graph.add_nodes_from(birth_edge)
        clusters.append(
            list(nx.shortest_path(graph, birth_edge[0]).keys()))
    return clusters


def depth_clusters(persistence_threshold,
                   depth,
                   simplex_tree):
    if depth is None:
        depth = persistence_threshold
    if not hasattr(simplex_tree, 'dgms'):
        simplex_tree.persistent_homology()
    # simplex_tree.persistence(
    #     homology_coeff_field=2,
    #     min_persistence=0,
    #     persistence_dim_max=1)
    filtration_values = dict(
        (frozenset(face), value)
        for (face, value) in simplex_tree.get_skeleton(2))
    filtration_values[frozenset()] = np.inf
    birth_points = []
    persistence_values = {}
    for pair in simplex_tree.persistence_pairs():
        if len(pair[0]) == 1:
            birth = frozenset(pair[0])
            death = frozenset(pair[1])
            persistence = filtration_values[death] - filtration_values[birth]
            if persistence >= persistence_threshold:
                birth_points.append(list(birth))
                persistence_values[frozenset(list(birth))] = persistence
    birth_points.sort(key=lambda x: -persistence_values[frozenset(x)])
    clusters = []
    edges = [list(face) for face in filtration_values.keys() if len(face) == 2]
    for birth_point in birth_points:
        graph = nx.Graph()
        selected_edges = [
            edge for edge in edges
            if filtration_values[frozenset(edge)] <=
            filtration_values[frozenset(birth_point)] + depth]
        graph.add_edges_from(selected_edges)
        graph.add_nodes_from(birth_point)
        clusters.append(
            list(nx.shortest_path(graph, birth_point[0]).keys()))
    return clusters


def empty_interleaving_line(n_points=100):
    return None
