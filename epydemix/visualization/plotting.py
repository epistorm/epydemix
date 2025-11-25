from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def get_black_to_grey(n):
    """Generate `n` grayscale colors starting with pure black."""
    if n < 1:
        raise ValueError("n must be at least 1")
    greys = np.linspace(0, 255, n, dtype=int)
    greys[0] = 0
    return [(g, g, g) for g in greys]


def get_timeseries_data(
    df_quantiles: pd.DataFrame, column: str, quantile: float
) -> pd.DataFrame:
    """
    Extracts the time series data for a specific column (compartment or demographic group) and quantile.

    Args:
        df_quantiles (pd.DataFrame): DataFrame containing quantile data for compartments and demographic groups.
        column (str): The name of the column to extract data for.
        quantile (float): The quantile to extract data for.

    Returns:
        pd.DataFrame: A DataFrame containing the time series data for the specified column and quantile.
    """
    return df_quantiles.loc[(df_quantiles["quantile"] == quantile)][["date", column]]


def plot_quantiles(
    df_quantiles: pd.DataFrame,
    columns: Union[List[str], str],
    data: Optional[pd.DataFrame] = None,
    data_date_column: str = "date",
    ax: Optional[plt.Axes] = None,
    lower_q: float = 0.05,
    upper_q: float = 0.95,
    show_median: bool = True,
    show_data: bool = False,
    ci_alpha: float = 0.3,
    title: str = "",
    ylabel: str = "",
    xlabel: str = "",
    show_legend: bool = True,
    legend_loc: str = "upper left",
    palette: str = "Dark2",
    colors: Optional[Union[List[str], str]] = None,
    labels: Optional[Union[List[str], str]] = None,
    y_scale: str = "linear",
    show_grid: bool = True,
) -> plt.Axes:
    """
    Plots quantiles for compartments over time with optional observed data.

    Args:
        df_quantiles: DataFrame with columns: 'date', 'quantile', and data columns
        columns: Column name(s) to plot
        data: Optional DataFrame containing observed data
        data_date_column: Column name for date in data
        ax: Matplotlib axes to plot on
        lower_q: Lower quantile value (0.05 = 5th percentile)
        upper_q: Upper quantile value (0.95 = 95th percentile)
        show_median: Whether to show median line
        show_data: Whether to show observed data points
        ci_alpha: Alpha value for confidence interval shading
        title: Plot title
        ylabel: Y-axis label
        xlabel: X-axis label
        show_legend: Whether to show legend
        legend_loc: Legend location
        palette: Color palette name
        colors: Custom colors for lines
        labels: Custom labels for legend
        y_scale: Scale for y-axis ('linear' or 'log')
        show_grid: Whether to show grid lines

    Returns:
        plt.Axes: The matplotlib axes object
    """
    if not isinstance(columns, list):
        columns = [columns]

    if ax is None:
        _, ax = plt.subplots(dpi=300, figsize=(10, 4))

    if colors is None:
        colors = sns.color_palette(palette, len(columns))
    elif not isinstance(colors, list):
        colors = [colors]

    if labels is None:
        labels = columns
    elif not isinstance(labels, list):
        labels = [labels]

    pleg, handles = [], []
    for t, (column, color, label) in enumerate(zip(columns, colors, labels)):
        if show_median:
            df_med = get_timeseries_data(df_quantiles, column, 0.5)
            (p1,) = ax.plot(
                df_med.date, df_med[column].values, color=color, label=label, zorder=2
            )

        df_q1 = get_timeseries_data(df_quantiles, column, lower_q)
        df_q2 = get_timeseries_data(df_quantiles, column, upper_q)
        p2 = ax.fill_between(
            df_q1.date,
            df_q1[column].values,
            df_q2[column].values,
            alpha=ci_alpha,
            color=color,
            linewidth=0.0,
            zorder=1,
        )

        if show_median:
            pleg.append((p1, p2))
            handles.append(
                f"{label} (median, {np.round((1 - lower_q * 2) * 100, 0)}% CI)"
            )
        else:
            pleg.append(p2)
            handles.append(f"{label} ({np.round((1 - lower_q * 2) * 100, 0)}% CI)")

    if show_data and data is not None:
        data_colors = get_black_to_grey(len(columns))
        for column, data_color, label in zip(columns, data_colors, labels):
            p_actual = ax.scatter(
                data[data_date_column],
                data[column],
                s=10,
                color=data_color,
                zorder=3,
                label=f"observed ({label})",
            )
            if show_legend:
                pleg.append(p_actual)
                handles.append(f"observed ({label})")

    # Style improvements
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if show_grid:
        ax.grid(axis="y", linestyle="--", linewidth=0.3, alpha=0.5, zorder=0)

    # Labels and formatting
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_yscale(y_scale)

    if show_legend and pleg:
        ax.legend(pleg, handles, loc=legend_loc, frameon=False)

    plt.tight_layout()

    return ax


def plot_posterior_distribution(
    posterior: pd.DataFrame,
    parameter: str,
    ax: Optional[plt.Axes] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    kind: str = "hist",
    color: str = "dodgerblue",
    xlim: Optional[Tuple[float, float]] = None,
    prior: Optional[Any] = None,
    prior_range: bool = False,
    title: Optional[str] = None,
    fontsize: int = 10,
    show_grid: bool = True,
    show_kde: bool = True,
    show_rug: bool = False,
    figsize: Tuple[int, int] = (10, 4),
    stat: str = "density",
    bins: Union[int, str] = "auto",
    alpha: float = 0.4,
    vertical_lines: Optional[Dict[str, Dict[str, Any]]] = None,
    **kwargs,
) -> plt.Axes:
    """
    Plots the distribution of a parameter.

    Args:
        posterior: DataFrame containing the parameter values
        parameter: The parameter to plot
        ax: Matplotlib axes to plot on. Creates new figure if None
        xlabel: X-axis label. If None, uses parameter name
        ylabel: Y-axis label
        kind: Type of plot ('hist', 'kde', or 'ecdf')
        color: Color for the plot
        xlim: Tuple of (min, max) for x-axis limits
        prior: Prior distribution object with ppf method
        prior_range: Whether to set x-axis limits to prior range
        title: Plot title. If None, auto-generates
        fontsize: Base font size for labels and ticks
        show_grid: Whether to show grid lines
        show_kde: Whether to show KDE curve with histogram (only for kind='hist')
        show_rug: Whether to show rug plot
        figsize: Figure size if creating new figure
        stat: Statistic to plot ('count', 'density', 'probability')
        bins: Number of bins or method for histogram
        alpha: Transparency of the plot
        vertical_lines: Dict of vertical lines to add, format:
            {
                'name': {
                    'x': value,
                    'color': 'color',
                    'linestyle': '--',
                    'label': 'label'
                }
            }
        **kwargs: Additional arguments passed to plotting functions

    Returns:
        plt.Axes: The matplotlib axes object

    Raises:
        ValueError: If kind is not 'hist', 'kde', or 'ecdf'
        ValueError: If prior_range is True but no prior is provided
    """
    if ax is None:
        _, ax = plt.subplots(dpi=300, figsize=figsize)

    # Set default labels
    xlabel = xlabel or parameter

    if kind == "hist":
        sns.histplot(
            data=posterior,
            x=parameter,
            ax=ax,
            color=color,
            stat=stat,
            bins=bins,
            alpha=alpha,
            kde=show_kde,
            **kwargs,
        )
    elif kind == "kde":
        sns.kdeplot(
            data=posterior,
            x=parameter,
            ax=ax,
            color=color,
            fill=True,
            alpha=alpha,
            **kwargs,
        )
    elif kind == "ecdf":
        sns.ecdfplot(data=posterior, x=parameter, ax=ax, color=color, **kwargs)
    else:
        raise ValueError(
            f"Unknown kind for plot: {kind}. Must be 'hist', 'kde', or 'ecdf'"
        )

    # Add rug plot if requested
    if show_rug:
        sns.rugplot(data=posterior, x=parameter, ax=ax, color=color, alpha=alpha / 2)

    # Add vertical lines if specified
    if vertical_lines:
        for line_specs in vertical_lines.values():
            ax.axvline(
                x=line_specs["x"],
                color=line_specs.get("color", "red"),
                linestyle=line_specs.get("linestyle", "--"),
                label=line_specs.get("label", None),
                alpha=line_specs.get("alpha", 1.0),
            )
            if line_specs.get("label"):
                ax.legend(frameon=False)

    # Style improvements
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if show_grid:
        ax.grid(axis="y", linestyle="--", linewidth=0.3, alpha=0.5)

    # Set axis limits based on prior range or explicit limits
    if prior_range:
        if prior is None:
            raise ValueError("prior must be provided when prior_range is True")
        ax.set_xlim(prior.ppf(0), prior.ppf(1))
    elif xlim is not None:
        ax.set_xlim(xlim)

    # Labels and title
    ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize + 2, pad=20)

    # Tick formatting
    ax.tick_params(axis="both", which="major", labelsize=fontsize - 2)

    # Adjust layout
    plt.tight_layout()

    return ax


def plot_posterior_distribution_2d(
    posterior: pd.DataFrame,
    parameter_x: str,
    parameter_y: str,
    ax: Optional[plt.Axes] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    kind: str = "hist",
    palette: str = "Blues",
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    prior_x: Optional[Any] = None,
    prior_y: Optional[Any] = None,
    prior_range: bool = False,
    title: Optional[str] = None,
    fontsize: int = 10,
    cmap: Optional[str] = None,
    show_grid: bool = True,
    levels: int = 10,
    figsize: Tuple[int, int] = (6, 6),
    scatter: bool = False,
    scatter_alpha: float = 0.5,
    scatter_size: int = 20,
    scatter_color: str = "k",
    **kwargs,
) -> plt.Axes:
    """
    Plots the 2D joint distribution of two parameters.

    Args:
        posterior: DataFrame containing the parameter values
        parameter_x: Parameter to plot on x-axis
        parameter_y: Parameter to plot on y-axis
        ax: Matplotlib axes to plot on. Creates new figure if None
        xlabel: X-axis label. If None, uses parameter_x
        ylabel: Y-axis label. If None, uses parameter_y
        kind: Plot type ('hist', 'kde', or 'scatter')
        palette: Color palette for histogram/kde
        xlim: Tuple of (min, max) for x-axis limits
        ylim: Tuple of (min, max) for y-axis limits
        prior_x: Prior distribution object for x parameter
        prior_y: Prior distribution object for y parameter
        prior_range: Whether to set axis limits to prior ranges
        title: Plot title. If None, auto-generates
        fontsize: Base font size for labels and ticks
        cmap: Colormap for 2D histogram/kde
        show_grid: Whether to show grid lines
        levels: Number of contour levels for kde
        figsize: Figure size if creating new figure
        scatter: Whether to overlay scatter plot on kde
        scatter_alpha: Alpha value for scatter points
        scatter_size: Size of scatter points
        scatter_color: Color for scatter points
        **kwargs: Additional arguments passed to sns.histplot/kdeplot

    Returns:
        plt.Axes: The matplotlib axes object

    Raises:
        ValueError: If kind is not 'hist', 'kde', or 'scatter'
        ValueError: If prior_range is True but priors are not provided
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, dpi=300)

    # Set default labels if not provided
    xlabel = xlabel or parameter_x
    ylabel = ylabel or parameter_y

    # Set default colormap
    if cmap is None:
        cmap = palette if kind == "hist" else sns.color_palette(palette, as_cmap=True)

    # Plot based on kind
    if kind == "hist":
        sns.histplot(
            data=posterior, x=parameter_x, y=parameter_y, ax=ax, cmap=cmap, **kwargs
        )

    elif kind == "kde":
        sns.kdeplot(
            data=posterior,
            x=parameter_x,
            y=parameter_y,
            ax=ax,
            cmap=cmap,
            levels=levels,
            fill=True,
            **kwargs,
        )

    elif kind == "scatter":
        ax.scatter(
            posterior[parameter_x],
            posterior[parameter_y],
            alpha=scatter_alpha,
            s=scatter_size,
            c=scatter_color,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown plot kind: {kind}. Must be 'hist', 'kde', or 'scatter'"
        )

    # Add scatter points if requested (for hist/kde)
    if scatter and kind != "scatter":
        ax.scatter(
            posterior[parameter_x],
            posterior[parameter_y],
            alpha=scatter_alpha,
            s=scatter_size,
            c=scatter_color,
            zorder=2,
        )

    # Style improvements
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if show_grid:
        ax.grid(linestyle="--", linewidth=0.3, alpha=0.5)

    # Set axis limits based on prior ranges or explicit limits
    if prior_range:
        if prior_x is None or prior_y is None:
            raise ValueError(
                "Both prior_x and prior_y must be provided when prior_range is True"
            )
        ax.set_xlim(prior_x.ppf(0), prior_x.ppf(1))
        ax.set_ylim(prior_y.ppf(0), prior_y.ppf(1))
    else:
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

    # Labels and title
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if title is None:
        title = f"Joint Distribution\n{parameter_x} vs {parameter_y}"
    ax.set_title(title, fontsize=fontsize + 2, pad=20)

    # Tick formatting
    ax.tick_params(axis="both", which="major", labelsize=fontsize - 2)

    # Adjust layout
    plt.tight_layout()

    return ax


def plot_selected_trajectories(
    calibration_results,
    ax=None,
    show_data=True,
    columns="data",
    lower_q=0.05,
    upper_q=0.95,
    show_median=True,
    ci_alpha=0.3,
    title="",
    show_legend=True,
    ylabel="",
    palette="Dark2",
):
    """
    TODO
    """
    return 0


def plot_contact_matrix(
    population: Any,
    layer: str,
    ax: Optional[plt.Axes] = None,
    cmap: str = "YlOrRd",
    show_values: bool = False,
    title: Optional[str] = None,
    show_colorbar: bool = True,
    fmt: str = ".1f",
    fontsize: int = 8,
    rotation: int = 45,
    figsize: Tuple[int, int] = (10, 8),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    origin: str = "lower",
) -> plt.Axes:
    """
    Plot a contact matrix heatmap.

    Args:
        population: Population object containing contact matrices
        layer: Name of the contact layer to plot
        ax: Matplotlib axes to plot on. Creates new figure if None
        cmap: Colormap for the heatmap
        show_values: Whether to show numerical values in cells
        title: Plot title. If None, uses layer name
        colorbar: Whether to show the colorbar
        fmt: Format string for cell values
        fontsize: Font size for labels and values
        rotation: Rotation angle for x-axis labels
        figsize: Figure size if creating new figure
        vmin: Minimum value for colormap scaling
        vmax: Maximum value for colormap scaling
        origin: Whether to plot the matrix with the origin at the bottom left (default) or top left

    Returns:
        plt.Axes: The matplotlib axes object

    Raises:
        KeyError: If specified layer doesn't exist
    """
    if layer not in population.contact_matrices:
        raise KeyError(
            f"Layer '{layer}' not found. Available layers: {population.layers}"
        )

    # Get contact matrix
    matrix = population.contact_matrices[layer]

    # Create figure if needed
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(
        matrix, cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax, origin=origin
    )

    # Show colorbar
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Contacts per day", fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)

    # Show values in cells
    if show_values:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                text = format(value, fmt)
                # Determine text color based on background
                text_color = "white" if value > (vmax or matrix.max()) / 2 else "black"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=fontsize,
                )

    # Set labels and ticks
    ax.set_xticks(np.arange(len(population.Nk_names)))
    ax.set_yticks(np.arange(len(population.Nk_names)))
    ax.set_xticklabels(population.Nk_names, rotation=rotation, ha="center")
    ax.set_yticklabels(population.Nk_names)

    # Adjust tick label sizes
    ax.tick_params(axis="both", which="major", labelsize=fontsize)

    # Set title
    if title is None:
        title = f"Contact Matrix - {layer}"
    ax.set_title(title, fontsize=fontsize + 2)

    # Add labels
    ax.set_xlabel("Age group (contacted)", fontsize=fontsize)
    ax.set_ylabel("Age group (contacting)", fontsize=fontsize)

    # Add grid to better separate cells
    ax.set_xticks(np.arange(-0.5, len(population.Nk_names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(population.Nk_names), 1), minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)

    # Adjust layout
    plt.tight_layout()

    return ax


def plot_population(
    population: Any,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    color: str = "dodgerblue",
    show_perc: bool = False,
    fontsize: int = 10,
    rotation: int = 45,
    figsize: Tuple[int, int] = (10, 6),
    bar_width: float = 0.8,
    show_grid: bool = True,
    ylabel: Optional[str] = None,
    xlabel: str = "Age group",
    show_values: bool = True,
    fmt: str = ".1f",
    value_fontsize: Optional[int] = None,
) -> plt.Axes:
    """
    Plot the population distribution across demographic groups.

    Args:
        population: Population object containing demographic data
        ax: Matplotlib axes to plot on. Creates new figure if None
        title: Plot title. If None, uses default title
        color: Color for the bars
        show_perc: Whether to show population as percentages
        fontsize: Base font size for labels and ticks
        rotation: Rotation angle for x-axis labels
        figsize: Figure size if creating new figure
        bar_width: Width of the bars (between 0 and 1)
        show_grid: Whether to show grid lines
        ylabel: Y-axis label. If None, uses default based on show_perc
        xlabel: X-axis label
        show_values: Whether to show values above bars
        fmt: Format string for values
        value_fontsize: Font size for bar values. If None, uses fontsize

    Returns:
        plt.Axes: The matplotlib axes object
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, dpi=300)

    # Calculate values
    values = population.Nk
    if show_perc:
        values = 100 * values / np.sum(values)

    # Create bars
    bars = ax.bar(population.Nk_names, values, color=color, width=bar_width)

    # Show values on bars
    if show_values:
        value_fontsize = value_fontsize or fontsize
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                format(height, fmt),
                ha="center",
                va="bottom",
                fontsize=value_fontsize,
            )

    # Style improvements
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if show_grid:
        ax.grid(axis="y", linestyle="--", linewidth=0.3, alpha=0.5)

    # Labels
    if ylabel is None:
        ylabel = "Population (%)" if show_perc else "Population"
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)

    # Title
    if title is None:
        title = f"Population Distribution - {population.name}"
    ax.set_title(title, fontsize=fontsize + 2, pad=20)

    # Tick formatting
    ax.tick_params(axis="both", which="major", labelsize=fontsize)
    plt.setp(ax.get_xticklabels(), rotation=rotation, ha="right")

    # Adjust y-axis to start at 0
    ax.set_ylim(bottom=0)

    # Add some padding above highest bar for values
    if show_values:
        ax.set_ylim(top=ax.get_ylim()[1] * 1.1)

    # Adjust layout
    plt.tight_layout()

    return ax


def plot_spectral_radius(
    epimodel: Any,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    color: str = "k",
    show_perc: bool = False,
    layer: str = "overall",
    show_interventions: bool = True,
    interventions_palette: str = "Dark2",
    interventions_colors: Optional[List[str]] = None,
    fontsize: int = 10,
    date_format: str = "%Y-%m-%d",
    ylabel: Optional[str] = None,
    xlabel: str = "Date",
    show_grid: bool = True,
    alpha: float = 0.2,
    legend_loc: str = "upper left",
) -> plt.Axes:
    """
    Plots the spectral radius of the contact matrices over time.

    Args:
        epimodel: The EpiModel object containing contact matrices and interventions
        ax: Matplotlib axes to plot on. Creates new figure if None
        title: Plot title. If None, uses default title
        color: Color of the spectral radius line
        show_perc: Whether to show as percentage change with respect to the initial value
        layer: Contact matrix layer to analyze
        show_interventions: Whether to show intervention periods
        interventions_palette: Color palette for interventions
        interventions_colors: Custom colors for interventions
        fontsize: Base font size for labels and ticks
        date_format: Format string for date labels
        ylabel: Y-axis label. If None, auto-generated based on normalize/show_perc
        xlabel: X-axis label
        show_grid: Whether to show grid lines
        alpha: Transparency for intervention highlights
        legend_loc: Location of the legend

    Returns:
        plt.Axes: The matplotlib axes object

    Raises:
        ValueError: If no contact matrices are defined or layer doesn't exist
    """
    if len(epimodel.Cs) == 0:
        raise ValueError("No contact matrices defined over time")

    if layer not in epimodel.population.layers + ["overall"]:
        raise ValueError(
            f"Layer '{layer}' not found. Available layers: {epimodel.population.layers + ['overall']}"
        )

    # Create figure if needed
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Compute spectral radius
    dates = list(epimodel.Cs.keys())
    rho = [np.linalg.eigvals(epimodel.Cs[date][layer]).max().real for date in dates]

    # Normalize and convert to percentage if requested
    if show_perc:
        rho = np.array(rho) / rho[0]
        rho = (rho - 1) * 100

    # Plot spectral radius
    ax.plot(dates, rho, color=color, label="Spectral radius", linewidth=2)

    # Show interventions if requested
    if show_interventions and hasattr(epimodel, "interventions"):
        # Select interventions for the layer (if layer is "overall", all interventions are selected)
        if layer == "overall":
            interventions = epimodel.interventions
        else:
            interventions = [
                intervention
                for intervention in epimodel.interventions
                if intervention["layer"] == layer
            ]

        # get colors
        colors = (
            interventions_colors
            if interventions_colors
            else sns.color_palette(interventions_palette, len(interventions))
        )

        for intervention, color in zip(interventions, colors):
            ax.axvspan(
                intervention["start_date"],
                intervention["end_date"],
                alpha=alpha,
                color=color,
                label=intervention["name"],
            )

    # Style improvements
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if show_grid:
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Format dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Labels
    if ylabel is None:
        ylabel = "Change in spectral radius (%)" if show_perc else "Spectral radius"
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)

    if title is None:
        title = f"Contact Pattern Intensity - {layer.title()} Layer"
    ax.set_title(title, fontsize=fontsize + 2, pad=20)

    # Legend if interventions are shown
    if show_interventions and hasattr(epimodel, "interventions"):
        ax.legend(loc=legend_loc, fontsize=fontsize - 2)

    # Adjust layout
    plt.tight_layout()

    return ax


def plot_distance_distribution(
    distances: Union[np.ndarray, List[float], pd.Series],
    ax: Optional[plt.Axes] = None,
    kind: str = "hist",
    color: str = "dodgerblue",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    fontsize: int = 10,
    show_grid: bool = True,
    show_kde: bool = True,
    show_rug: bool = False,
    figsize: Tuple[int, int] = (10, 4),
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    stat: str = "density",
    bins: Union[int, str] = "auto",
    alpha: float = 0.4,
    vertical_lines: Optional[Dict[str, Dict[str, Any]]] = None,
    **kwargs,
) -> plt.Axes:
    """
    Plots the distribution of distances/errors from calibration.

    Args:
        distances: Array-like object containing the distance/error values
        ax: Matplotlib axes to plot on. Creates new figure if None
        kind: Type of plot ('hist', 'kde', or 'ecdf')
        color: Color for the plot
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        fontsize: Base font size for labels and ticks
        show_grid: Whether to show grid lines
        show_kde: Whether to show KDE curve with histogram (only for kind='hist')
        show_rug: Whether to show rug plot
        figsize: Figure size if creating new figure
        xlim: Tuple of (min, max) for x-axis limits
        ylim: Tuple of (min, max) for y-axis limits
        stat: Statistic to plot for histogram ('count', 'density', 'probability')
        bins: Number of bins or method for histogram
        alpha: Transparency of the plot
        vertical_lines: Dict of vertical lines to add, format:
            {
                'name': {
                    'x': value,
                    'color': 'color',
                    'linestyle': '--',
                    'label': 'label'
                }
            }
        **kwargs: Additional arguments passed to plotting functions

    Returns:
        plt.Axes: The matplotlib axes object

    Raises:
        ValueError: If kind is not 'hist', 'kde', or 'ecdf'
    """
    if ax is None:
        _, ax = plt.subplots(dpi=300, figsize=figsize)

    # Convert input to pandas Series for consistent handling
    if not isinstance(distances, pd.Series):
        distances = pd.Series(distances)

    # Set default labels if not provided
    if xlabel is None:
        xlabel = "Distance"
    if ylabel is None:
        ylabel = {
            "hist": "Density" if stat == "density" else "Count",
            "kde": "Density",
            "ecdf": "Cumulative Probability",
        }.get(kind, "")

    # Plot based on kind
    if kind == "hist":
        sns.histplot(
            data=distances,
            ax=ax,
            color=color,
            stat=stat,
            bins=bins,
            alpha=alpha,
            kde=show_kde,
            **kwargs,
        )
    elif kind == "kde":
        sns.kdeplot(
            data=distances, ax=ax, color=color, fill=True, alpha=alpha, **kwargs
        )
    elif kind == "ecdf":
        sns.ecdfplot(data=distances, ax=ax, color=color, **kwargs)
    else:
        raise ValueError(
            f"Unknown kind for plot: {kind}. Must be 'hist', 'kde', or 'ecdf'"
        )

    # Add rug plot if requested
    if show_rug:
        sns.rugplot(data=distances, ax=ax, color=color, alpha=alpha / 2)

    # Add vertical lines if specified
    if vertical_lines:
        for line_specs in vertical_lines.values():
            ax.axvline(
                x=line_specs["x"],
                color=line_specs.get("color", "red"),
                linestyle=line_specs.get("linestyle", "--"),
                label=line_specs.get("label", None),
                alpha=line_specs.get("alpha", 1.0),
            )
            if line_specs.get("label"):
                ax.legend(frameon=False)

    # Style improvements
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if show_grid:
        ax.grid(axis="y", linestyle="--", linewidth=0.3, alpha=0.5)

    # Set axis limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Labels and title
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if title is not None:
        ax.set_title(title, fontsize=fontsize + 2, pad=20)

    # Tick formatting
    ax.tick_params(axis="both", which="major", labelsize=fontsize - 2)

    # Adjust layout
    plt.tight_layout()

    return ax


def plot_trajectories(
    stacked: Dict[str, np.ndarray],
    columns: Union[List[str], str],
    data: Optional[pd.DataFrame] = None,
    ax: Optional[plt.Axes] = None,
    show_data: bool = False,
    alpha: float = 0.1,
    title: str = "",
    ylabel: str = "",
    xlabel: str = "",
    show_legend: bool = True,
    legend_loc: str = "upper left",
    palette: str = "Dark2",
    colors: Optional[Union[List[str], str]] = None,
    labels: Optional[Union[List[str], str]] = None,
    y_scale: str = "linear",
    show_grid: bool = True,
    dates: Optional[np.ndarray] = None,
) -> plt.Axes:
    """
    Plots individual trajectories over time with optional observed data.

    Args:
        stacked: Dictionary mapping column names to arrays of shape (n_simulations, timesteps)
        columns: Column name(s) to plot
        data: Optional DataFrame containing observed data
        ax: Matplotlib axes to plot on
        show_data: Whether to show observed data points
        alpha: Alpha value for individual trajectories
        title: Plot title
        ylabel: Y-axis label
        xlabel: X-axis label
        show_legend: Whether to show legend
        legend_loc: Legend location
        palette: Color palette name
        colors: Custom colors for lines
        labels: Custom labels for legend
        y_scale: Scale for y-axis ('linear' or 'log')
        show_grid: Whether to show grid lines
        dates: Array of dates for x-axis. If None, uses range(timesteps)

    Returns:
        plt.Axes: The matplotlib axes object
    """
    if not isinstance(columns, list):
        columns = [columns]

    if ax is None:
        _, ax = plt.subplots(dpi=300, figsize=(10, 4))

    if colors is None:
        colors = sns.color_palette(palette, len(columns))
    elif not isinstance(colors, list):
        colors = [colors]

    if labels is None:
        labels = columns
    elif not isinstance(labels, list):
        labels = [labels]

    # Create x-axis values
    if dates is None:
        x = np.arange(stacked[columns[0]].shape[1])
    else:
        x = dates

    # Plot each trajectory for each column
    pleg = []
    for column, color, label in zip(columns, colors, labels):
        trajectories = stacked[column]

        # Plot individual trajectories
        for traj in trajectories:
            line = ax.plot(x, traj, color=color, alpha=alpha, linewidth=0.5, zorder=1)

        # Plot median trajectory with higher alpha
        mean_traj = np.median(trajectories, axis=0)
        (line,) = ax.plot(
            x, mean_traj, color=color, alpha=1.0, linewidth=2, label=label, zorder=2
        )
        pleg.append(line)

    handles_data = []
    if show_data and data is not None:
        data_colors = get_black_to_grey(len(columns))
        for column, data_color, label in zip(columns, data_colors, labels):
            p_actual = ax.scatter(
                x,
                data[column],
                s=10,
                color=data_color,
                zorder=3,
                label=f"observed ({label})",
            )
            if show_legend:
                pleg.append(p_actual)
                handles_data.append(f"observed ({label})")

    # Style improvements
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if show_grid:
        ax.grid(axis="y", linestyle="--", linewidth=0.3, alpha=0.5, zorder=0)

    # Labels and formatting
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_yscale(y_scale)

    if show_legend and pleg:
        ax.legend(
            pleg,
            labels + (handles_data if show_data and data is not None else []),
            loc=legend_loc,
            frameon=False,
        )

    plt.tight_layout()

    return ax
