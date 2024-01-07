"""
Python routines used to test claims in the papers
*** SDN = Sparse Dowker Nerves ***
*** SFN = Sparse Filtered Nerves ***
Copyright: Applied Topology Group at University of Bergen.
No commercial use of the software is permitted without proper license.
"""
import numpy as np
import matplotlib.pyplot as plt
import gudhi
from itertools import cycle
import matplotlib as mpl


def SimplexTreeFromNerve(nerve, coeff_field=11, min_persistence=0):

    nerve.get_filtered_nerve()
    geometric_dimension = max((len(x) for x in nerve.maximal_faces)) - 1
    persistence_dim_max = geometric_dimension <= nerve.dimension

    simplex_tree = SimplexTree(
        interleaving_line=nerve.interleaving_line,
        coeff_field=coeff_field,
        persistence_dim_max=persistence_dim_max,
        min_persistence=min_persistence,
    )

    for index, simplex in enumerate(nerve.nerve):
        simplex_tree.st.insert(
            simplex=list(simplex), filtration=nerve.nerve_values[index]
        )

    return simplex_tree


class SimplexTree():
    def __init__(
        self,
        interleaving_line=None,
        coeff_field=11,
        persistence_dim_max=False,
        min_persistence=0,
    ):
        self.st = gudhi.SimplexTree()
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
        min_persistence=None,
    ):
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
        persistence = self.st.persistence(
            homology_coeff_field=coeff_field,
            min_persistence=min_persistence,
            persistence_dim_max=persistence_dim_max,
        )
        max_homology_dimension = max(
            1, self.st.dimension() - 1 + persistence_dim_max
        )
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
        add_multiplicity=False,
        ax_color=np.array([0.0, 0.0, 0.0]),
        colors=None,
        diagonal=True,
        lifetime=False,
        legend=True,
        show=False,
        return_plot=False,
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
        add_multiplicity: boolean (default : False)
            Show multiplicity of points plotted.
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
        if not hasattr(self.st, "dgms"):
            self.persistent_homology()
        if hasattr(self, "interleaving_line"):
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
            alpha=alpha,
            add_multiplicity=add_multiplicity,
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
    add_multiplicity=False,
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
    if sum((len(dgm) for dgm in diagrams)) <= 1:
        print("Persistence diagram is empty!\n Nothing to plot.")
        return

    if plot_only is None:
        plot_only = [i for i, dgm in enumerate(diagrams) if len(dgm) > 0]
    if not isinstance(plot_only, list):
        plot_only = [plot_only]

    if not isinstance(labels, list):
        labels = [labels] * len(diagrams)

    if colors is None:
        mpl.style.use(colormap)
        colors = cycle(
            ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
        )
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
        raise ValueError("Please specify a non-degenerate xy_range")
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
        plt.plot(
            dgm[:, 0],
            dgm[:, 1],
            ms=size,
            color=color,
            label=label,
            linestyle="",  # edgecolor="none",
            alpha=alpha,
            marker="o",
        )

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if add_multiplicity:
            xy_arr, s_arr = np.unique(dgm, return_counts=True, axis=0)
            for xy, s in zip(xy_arr, s_arr):
                plt.text(xy[0], xy[1], s)

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


def empty_interleaving_line(n_points=100):
    return None
