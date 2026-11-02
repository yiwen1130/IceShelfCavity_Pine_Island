#!/usr/bin/env python3

"""
Plot the width-integrated overturning circulation from a full 3-D
IceShelfCavity / Oceananigans NetCDF output file.

The script reads:

    <directory>/<simname>.nc

It must receive the full 3-D file, not *_midy.nc or *_yavg.nc.

The overturning streamfunction is

              y=Ly    z
    Psi(x,z) = ∫       ∫ u(x,y,z') dz' dy
              y=0     zb

where u is the along-cavity velocity, positive from the grounding
line toward the ice-shelf front.

Sign convention used here:

    positive Psi = counterclockwise circulation
    negative Psi = clockwise circulation

The plotted streamfunction is in Sverdrups:

    1 Sv = 1,000,000 m3/s
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


# ================================================================
# Command-line arguments
# ================================================================

parser = argparse.ArgumentParser(
    description=(
        "Plot width-integrated 3-D overturning circulation, "
        "temperature anomaly and salinity anomaly."
    )
)

parser.add_argument(
    "simname",
    help=(
        "Simulation name without .nc. Use the full 3-D run name, "
        "not a name ending in _midy or _yavg."
    ),
)

parser.add_argument(
    "--dir",
    default="output",
    help="Directory containing <simname>.nc",
)

parser.add_argument(
    "--geometry",
    default="data/pineislandbath.csv",
    help="Geometry CSV containing x_km, topo and bathy",
)

parser.add_argument(
    "--Tlim",
    type=float,
    default=None,
    help=(
        "Symmetric temperature-anomaly color limit [deg C]. "
        "Default: determine automatically."
    ),
)

parser.add_argument(
    "--Slim",
    type=float,
    default=None,
    help=(
        "Symmetric salinity-anomaly color limit [psu]. "
        "Default: determine automatically."
    ),
)

parser.add_argument(
    "--psilim",
    type=float,
    default=None,
    help=(
        "Symmetric streamfunction color limit [Sv]. "
        "Default: determine automatically."
    ),
)

parser.add_argument(
    "--time-index",
    type=int,
    default=-1,
    help=(
        "Time index used for the cross-section. "
        "Default -1 means the final saved time."
    ),
)

parser.add_argument(
    "--dpi",
    type=int,
    default=200,
    help="Output PNG resolution",
)

args = parser.parse_args()


# ================================================================
# Open the full 3-D model output
# ================================================================

ncfile = os.path.join(
    args.dir,
    f"{args.simname}.nc",
)

if not os.path.exists(ncfile):
    raise SystemExit(
        f"\nCould not find:\n\n"
        f"    {ncfile}\n\n"
        "Use the full 3-D NetCDF file, not _midy.nc or _yavg.nc."
    )

ds = xr.open_dataset(
    ncfile,
    decode_times=False,
)

print()
print(f"Reading: {ncfile}")
print(f"Variables: {list(ds.data_vars)}")
print(f"Dimensions: {dict(ds.sizes)}")


required_variables = ["u", "T", "S"]

for variable in required_variables:
    if variable not in ds:
        raise SystemExit(
            f"Required variable {variable!r} is missing from {ncfile}"
        )


# ================================================================
# Dimension helpers
# ================================================================

def find_dimension(data_array, first_letter):
    """Find a dimension beginning with x, y or z."""

    matching = [
        dim for dim in data_array.dims
        if dim.lower().startswith(first_letter.lower())
    ]

    if not matching:
        raise ValueError(
            f"Could not find a {first_letter}-dimension "
            f"in {data_array.dims}"
        )

    return matching[0]


def find_metric_for_dimension(prefix, dimension_name):
    """Find an Oceananigans grid metric such as Δy_aca or Δz_aac."""

    direct_name = f"Δ{prefix}_{dimension_name.split('_', 1)[-1]}"

    if direct_name in ds:
        metric = ds[direct_name]

        if dimension_name in metric.dims:
            return metric

    candidates = []

    for name in ds.data_vars:
        if not name.startswith(f"Δ{prefix}"):
            continue

        metric = ds[name]

        if (
            dimension_name in metric.dims
            and metric.sizes[dimension_name]
            == ds.sizes[dimension_name]
        ):
            candidates.append(metric)

    if candidates:
        return candidates[0]

    return None


# ================================================================
# Confirm this is full 3-D output
# ================================================================

u_field = ds["u"]

u_xdim = find_dimension(u_field, "x")
u_ydim = find_dimension(u_field, "y")
u_zdim = find_dimension(u_field, "z")

if u_field.sizes[u_ydim] <= 1:
    raise SystemExit(
        "\nThe selected file does not contain a resolved y dimension.\n"
        "Use the full 3-D file:\n\n"
        f"    {args.simname}.nc\n\n"
        "Do not use the _midy.nc file. A _yavg.nc file contains "
        "only mean transport per metre width, not the directly "
        "width-integrated 3-D transport."
    )

print()
print("Full 3-D velocity grid")
print(f"  x dimension: {u_xdim} ({u_field.sizes[u_xdim]} points)")
print(f"  y dimension: {u_ydim} ({u_field.sizes[u_ydim]} points)")
print(f"  z dimension: {u_zdim} ({u_field.sizes[u_zdim]} points)")


# ================================================================
# Read the model stratification
# ================================================================

stratification_file = os.path.join(
    "scripts",
    "stratification.jl",
)

if not os.path.exists(stratification_file):
    raise SystemExit(
        f"Could not find {stratification_file}.\n"
        "Run this script from the IceShelfCavity repository."
    )

with open(stratification_file) as file:
    stratification_text = file.read()


def read_four_constants(names):
    """Read four constants from a Julia const assignment."""

    pattern = (
        r"const\s+"
        + r",\s*".join(names)
        + r"\s*=\s*([^\n]+)"
    )

    match = re.search(
        pattern,
        stratification_text,
    )

    if match is None:
        raise RuntimeError(
            f"Could not find constants {names} "
            f"in {stratification_file}"
        )

    numbers = re.findall(
        r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
        match.group(1),
    )

    if len(numbers) < 4:
        raise RuntimeError(
            f"Could not parse four numbers for {names}"
        )

    return tuple(
        float(number)
        for number in numbers[:4]
    )


aT, cT, z0T, hT = read_four_constants(
    ["aT", "cT", "z0T", "hT"]
)

aS, cS, z0S, hS = read_four_constants(
    ["aS", "cS", "z0S", "hS"]
)


def ambient_temperature(z):
    """Initial/ambient temperature profile used by the model."""

    return aT + cT * np.tanh(
        (z - z0T) / hT
    )


def ambient_salinity(z):
    """Initial/ambient salinity profile used by the model."""

    return aS + cS * np.tanh(
        (z - z0S) / hS
    )


# ================================================================
# Read the ice-base and seafloor geometry
# ================================================================

if not os.path.exists(args.geometry):
    raise SystemExit(
        f"Could not find geometry file: {args.geometry}"
    )

geometry = pd.read_csv(args.geometry)

required_geometry_columns = [
    "x_km",
    "topo",
    "bathy",
]

for column in required_geometry_columns:
    if column not in geometry:
        raise SystemExit(
            f"Geometry file is missing column {column!r}"
        )

geometry_x = (
    geometry["x_km"].to_numpy(dtype=float)
    * 1000.0
)

geometry_top = geometry["topo"].to_numpy(
    dtype=float
)

geometry_bottom = geometry["bathy"].to_numpy(
    dtype=float
)


def interpolated_geometry(x):
    """Return ice-base and seafloor elevation at x."""

    top = np.interp(
        x,
        geometry_x,
        geometry_top,
    )

    bottom = np.interp(
        x,
        geometry_x,
        geometry_bottom,
    )

    return top, bottom


def center_wet_mask(x, z):
    """True at cell centers inside the ocean cavity."""

    top, bottom = interpolated_geometry(x)

    return (
        (z[:, None] < top[None, :])
        & (z[:, None] > bottom[None, :])
    )


def face_wet_mask(x, z):
    """True at vertical faces inside the ocean cavity."""

    top, bottom = interpolated_geometry(x)

    return (
        (z[:, None] <= top[None, :])
        & (z[:, None] >= bottom[None, :])
    )


# ================================================================
# Grid metrics
# ================================================================

dy = find_metric_for_dimension(
    "y",
    u_ydim,
)

if dy is None:
    print()
    print(
        "Warning: saved Δy metric was not found. "
        "Estimating width weights from y coordinates."
    )

    y_coordinate = (
        u_field[u_ydim]
        .to_numpy()
        .astype(float)
    )

    dy_values = np.gradient(y_coordinate)

    dy = xr.DataArray(
        dy_values,
        dims=(u_ydim,),
        coords={
            u_ydim: u_field[u_ydim],
        },
    )
else:
    dy = dy.astype(float)


channel_width = float(
    dy.sum(dim=u_ydim).to_numpy()
)

print()
print(f"Resolved channel width: {channel_width:.2f} m")
print(f"Resolved channel width: {channel_width / 1000:.3f} km")


# ================================================================
# Width operations
# ================================================================

def width_integral(data_array):
    """Integrate a 3-D field across y.

    For velocity u [m/s], the result has units [m2/s].
    """

    ydim = find_dimension(
        data_array,
        "y",
    )

    y_metric = find_metric_for_dimension(
        "y",
        ydim,
    )

    if y_metric is None:
        y_values = (
    data_array[ydim]
    .to_numpy()
    .astype(float)
)

        y_metric = xr.DataArray(
            np.gradient(y_values),
            dims=(ydim,),
            coords={
                ydim: data_array[ydim],
            },
        )

    return (
        data_array.fillna(0.0)
        * y_metric
    ).sum(dim=ydim)


def width_mean(data_array):
    """Calculate a grid-weighted mean across y."""

    ydim = find_dimension(
        data_array,
        "y",
    )

    y_metric = find_metric_for_dimension(
        "y",
        ydim,
    )

    if y_metric is None:
        y_values = (
            data_array[ydim]
            .to_numpy()
            .astype(float)
        )

        y_metric = xr.DataArray(
            np.gradient(y_values),
            dims=(ydim,),
            coords={
                ydim: data_array[ydim],
            },
        )

    numerator = (
        data_array.fillna(0.0)
        * y_metric
    ).sum(dim=ydim)

    denominator = y_metric.sum(dim=ydim)

    return numerator / denominator


# ================================================================
# Final width-mean temperature and salinity
# ================================================================

def final_width_mean_field(variable):
    """Return final y-mean field shaped as (z, x)."""

    field = ds[variable].isel(
        time=args.time_index
    )

    field_mean = width_mean(field)

    xdim = find_dimension(
        field_mean,
        "x",
    )

    zdim = find_dimension(
        field_mean,
        "z",
    )

    field_mean = field_mean.transpose(
        zdim,
        xdim,
    )

    x = (
        field_mean[xdim]
        .to_numpy()
        .astype(float)
    )

    z = (
        field_mean[zdim]
        .to_numpy()
        .astype(float)
    )

    values = field_mean.to_numpy().astype(
        float
    )

    if z[0] > z[-1]:
        z = z[::-1]
        values = values[::-1, :]

    return x, z, values


xT, zT, temperature_mean = final_width_mean_field(
    "T"
)

xS, zS, salinity_mean = final_width_mean_field(
    "S"
)

temperature_mask = center_wet_mask(
    xT,
    zT,
)

salinity_mask = center_wet_mask(
    xS,
    zS,
)

temperature_anomaly = (
    temperature_mean
    - ambient_temperature(zT)[:, None]
)

salinity_anomaly = (
    salinity_mean
    - ambient_salinity(zS)[:, None]
)

temperature_anomaly[
    ~temperature_mask
] = np.nan

salinity_anomaly[
    ~salinity_mask
] = np.nan


# ================================================================
# Width-integrated 3-D overturning streamfunction
# ================================================================

def calculate_3d_streamfunction(
    u_data_array,
    print_diagnostics=False,
):
    """Calculate width-integrated overturning.

    Parameters
    ----------
    u_data_array
        Three-dimensional u(z, y, x) at one model time.

    Returns
    -------
    x
        x-face coordinates [m].
    zf
        Vertical face coordinates [m].
    psi_sv
        Width-integrated overturning streamfunction [Sv].
    closure_fraction
        Maximum top residual divided by maximum interior
        streamfunction magnitude.
    """

    # Integrate velocity across the complete y width.
    #
    # u [m/s] * dy [m] = width transport density [m2/s].
    u_width_integrated = width_integral(
        u_data_array
    )

    xdim = find_dimension(
        u_width_integrated,
        "x",
    )

    zdim = find_dimension(
        u_width_integrated,
        "z",
    )

    u_width_integrated = (
        u_width_integrated.transpose(
            zdim,
            xdim,
        )
    )

    x = (
        u_width_integrated[xdim]
        .to_numpy()
        .astype(float)
    )

    zc = (
        u_width_integrated[zdim]
        .to_numpy()
        .astype(float)
    )

    u_transport = (
        u_width_integrated
        .to_numpy()
        .astype(float)
    )

    # Ensure that z increases from bottom toward the ice.
    reverse_vertical = zc[0] > zc[-1]

    if reverse_vertical:
        zc = zc[::-1]
        u_transport = u_transport[::-1, :]

    # Obtain vertical cell thickness.
    dz_metric = find_metric_for_dimension(
        "z",
        zdim,
    )

    if dz_metric is not None:
        dz = (
            dz_metric
            .to_numpy()
            .astype(float)
            .squeeze()
        )

        if reverse_vertical:
            dz = dz[::-1]

        dz = np.abs(dz)
    else:
        print()
        print(
            "Warning: saved Δz metric was not found. "
            "Estimating vertical thickness from z coordinates."
        )

        dz = np.abs(
            np.gradient(zc)
        )

    if dz.size != zc.size:
        raise ValueError(
            "Vertical thickness does not match "
            "the u vertical dimension."
        )

    # Construct vertical-face coordinates.
    vertical_face_candidates = [
        coord
        for coord in ds.coords
        if (
            coord.lower().startswith("z")
            and ds[coord].size == zc.size + 1
        )
    ]

    if vertical_face_candidates:
        zface_name = vertical_face_candidates[0]

        zf = (
            ds[zface_name]
            .to_numpy()
            .astype(float)
        )

        if zf[0] > zf[-1]:
            zf = zf[::-1]
    else:
        zf = np.empty(
            zc.size + 1
        )

        zf[0] = (
            zc[0]
            - 0.5 * dz[0]
        )

        zf[1:] = (
            zf[0]
            + np.cumsum(dz)
        )

    # Eliminate transport inside ice and bedrock.
    ocean_center_mask = center_wet_mask(
        x,
        zc,
    )

    u_transport_wet = np.where(
        (
            ocean_center_mask
            & np.isfinite(u_transport)
        ),
        u_transport,
        0.0,
    )

    # Integrate upward from the lowest vertical face.
    #
    # u_transport has units m2/s.
    # u_transport * dz has units m3/s.
    psi_m3s = np.zeros(
        (
            zc.size + 1,
            x.size,
        )
    )

    psi_m3s[1:, :] = np.cumsum(
        u_transport_wet
        * dz[:, None],
        axis=0,
    )

    # Convert volume transport to Sverdrups.
    psi_sv = psi_m3s / 1.0e6

    # Mask faces outside the local water column.
    ocean_face_mask = face_wet_mask(
        x,
        zf,
    )

    psi_sv[
        ~ocean_face_mask
    ] = np.nan

    # Check whether transport returns close to zero at the
    # upper cavity boundary.
    top_residual_m3s = np.sum(
        u_transport_wet
        * dz[:, None],
        axis=0,
    )

    maximum_interior_m3s = np.nanmax(
        np.abs(psi_m3s)
    )

    maximum_residual_m3s = np.nanmax(
        np.abs(top_residual_m3s)
    )

    if maximum_interior_m3s > 0:
        closure_fraction = (
            maximum_residual_m3s
            / maximum_interior_m3s
        )
    else:
        closure_fraction = np.nan

    if print_diagnostics:
        print()
        print("Overturning consistency check")
        print(
            "  maximum interior |Psi| = "
            f"{maximum_interior_m3s / 1e6:.6e} Sv"
        )
        print(
            "  maximum top residual = "
            f"{maximum_residual_m3s / 1e6:.6e} Sv"
        )
        print(
            "  residual / strength = "
            f"{closure_fraction:.3%}"
        )

    return (
        x,
        zf,
        psi_sv,
        closure_fraction,
    )


# Calculate the final streamfunction.
u_final = ds["u"].isel(
    time=args.time_index
)

xpsi, zpsi, psi_sv, final_closure = (
    calculate_3d_streamfunction(
        u_final,
        print_diagnostics=True,
    )
)


# ================================================================
# Streamfunction strengths through time
# ================================================================

time_seconds = (
    ds["time"]
    .to_numpy()
    .astype(float)
)

time_days = time_seconds / 86400.0

counterclockwise_strength = np.zeros(
    time_seconds.size
)

clockwise_strength = np.zeros(
    time_seconds.size
)

maximum_absolute_strength = np.zeros(
    time_seconds.size
)

closure_fraction = np.zeros(
    time_seconds.size
)


for time_index in range(
    time_seconds.size
):
    _, _, psi_at_time, closure_at_time = (
        calculate_3d_streamfunction(
            ds["u"].isel(
                time=time_index
            ),
            print_diagnostics=False,
        )
    )

    psi_min = float(
        np.nanmin(psi_at_time)
    )

    psi_max = float(
        np.nanmax(psi_at_time)
    )

    # Positive Psi is counterclockwise.
    counterclockwise_strength[
        time_index
    ] = max(
        0.0,
        psi_max,
    )

    # Negative Psi is clockwise.
    clockwise_strength[
        time_index
    ] = max(
        0.0,
        -psi_min,
    )

    maximum_absolute_strength[
        time_index
    ] = max(
        abs(psi_min),
        abs(psi_max),
    )

    closure_fraction[
        time_index
    ] = closure_at_time


# ================================================================
# Automatic symmetric color limits
# ================================================================

def symmetric_limit(
    values,
    requested_limit,
    percentile=98.0,
    minimum=1.0e-12,
):
    """Return requested or robust automatic symmetric limit."""

    if requested_limit is not None:
        return requested_limit

    finite_values = np.abs(
        values[np.isfinite(values)]
    )

    if finite_values.size == 0:
        return minimum

    calculated = np.nanpercentile(
        finite_values,
        percentile,
    )

    return max(
        float(calculated),
        minimum,
    )


temperature_limit = symmetric_limit(
    temperature_anomaly,
    args.Tlim,
    minimum=1.0e-8,
)

salinity_limit = symmetric_limit(
    salinity_anomaly,
    args.Slim,
    minimum=1.0e-8,
)

streamfunction_limit = symmetric_limit(
    psi_sv,
    args.psilim,
    minimum=1.0e-12,
)


# ================================================================
# Determine displayed model time
# ================================================================

selected_time_seconds = float(
    ds["time"].isel(
        time=args.time_index
    ).to_numpy()
)

selected_time_days = (
    selected_time_seconds
    / 86400.0
)


# ================================================================
# Plot
# ================================================================

fig, axes = plt.subplots(
    4,
    1,
    figsize=(13, 16),
    constrained_layout=True,
)

solid_color = "0.65"


# ----------------------------------------------------------------
# Panel 1: width-mean temperature anomaly
# ----------------------------------------------------------------

temperature_plot = axes[0].pcolormesh(
    xT / 1000.0,
    zT,
    temperature_anomaly,
    cmap="RdBu_r",
    vmin=-temperature_limit,
    vmax=temperature_limit,
    shading="auto",
)

axes[0].set_title(
    "Width-mean temperature anomaly: "
    r"$\overline{T}^{\,y}-T_{\mathrm{ambient}}(z)$"
)

axes[0].set_ylabel(
    "Elevation z (m)"
)

fig.colorbar(
    temperature_plot,
    ax=axes[0],
    label="Temperature anomaly (°C)",
)


# ----------------------------------------------------------------
# Panel 2: width-mean salinity anomaly
# ----------------------------------------------------------------

salinity_plot = axes[1].pcolormesh(
    xS / 1000.0,
    zS,
    salinity_anomaly,
    cmap="RdBu_r",
    vmin=-salinity_limit,
    vmax=salinity_limit,
    shading="auto",
)

axes[1].set_title(
    "Width-mean salinity anomaly: "
    r"$\overline{S}^{\,y}-S_{\mathrm{ambient}}(z)$"
)

axes[1].set_ylabel(
    "Elevation z (m)"
)

fig.colorbar(
    salinity_plot,
    ax=axes[1],
    label="Salinity anomaly (psu)",
)


# ----------------------------------------------------------------
# Panel 3: width-integrated overturning
# ----------------------------------------------------------------

streamfunction_levels = np.linspace(
    -streamfunction_limit,
    streamfunction_limit,
    31,
)

streamfunction_plot = axes[2].contourf(
    xpsi / 1000.0,
    zpsi,
    psi_sv,
    levels=streamfunction_levels,
    cmap="RdBu_r",
    extend="both",
)

# Thin transport contours.
axes[2].contour(
    xpsi / 1000.0,
    zpsi,
    psi_sv,
    levels=streamfunction_levels[::3],
    colors="black",
    linewidths=0.35,
    alpha=0.45,
)

# Zero contour separates opposite circulation directions.
if (
    np.nanmin(psi_sv) < 0
    and np.nanmax(psi_sv) > 0
):
    axes[2].contour(
        xpsi / 1000.0,
        zpsi,
        psi_sv,
        levels=[0.0],
        colors="black",
        linewidths=1.0,
    )

axes[2].set_title(
    "Width-integrated overturning streamfunction "
    "(positive = counterclockwise)"
)

axes[2].set_ylabel(
    "Elevation z (m)"
)

fig.colorbar(
    streamfunction_plot,
    ax=axes[2],
    label="Volume transport Ψ (Sv)",
)


# ----------------------------------------------------------------
# Panel 4: cell strength through time
# ----------------------------------------------------------------

axes[3].plot(
    time_days,
    counterclockwise_strength,
    color="steelblue",
    linewidth=1.7,
    label="Counterclockwise: max(Ψ)",
)

axes[3].plot(
    time_days,
    clockwise_strength,
    color="darkorange",
    linewidth=1.7,
    label="Clockwise: −min(Ψ)",
)

axes[3].plot(
    time_days,
    maximum_absolute_strength,
    color="black",
    linewidth=1.0,
    linestyle="--",
    alpha=0.65,
    label="Strongest absolute cell",
)

axes[3].set_title(
    "Width-integrated overturning-cell strength"
)

axes[3].set_xlabel(
    "Model time (days)"
)

axes[3].set_ylabel(
    "Cell strength (Sv)"
)

axes[3].grid(
    alpha=0.25
)

axes[3].legend(
    loc="best"
)


# ================================================================
# Geometry overlays and common spatial axes
# ================================================================

for axis in axes[:3]:
    axis.fill_between(
        geometry_x / 1000.0,
        geometry_top,
        geometry_top.max() + 150.0,
        color=solid_color,
        zorder=10,
    )

    axis.fill_between(
        geometry_x / 1000.0,
        geometry_bottom.min() - 150.0,
        geometry_bottom,
        color=solid_color,
        zorder=10,
    )

    axis.plot(
        geometry_x / 1000.0,
        geometry_top,
        color="0.25",
        linewidth=0.7,
        zorder=11,
    )

    axis.plot(
        geometry_x / 1000.0,
        geometry_bottom,
        color="0.25",
        linewidth=0.7,
        zorder=11,
    )

    axis.set_xlim(
        geometry_x.min() / 1000.0,
        geometry_x.max() / 1000.0,
    )

    axis.set_xlabel(
        "Distance from grounding line (km)"
    )


fig.suptitle(
    (
        f"{args.simname}: 3-D width-integrated overturning\n"
        f"Cross-section at model day {selected_time_days:.4f}"
    ),
    fontsize=15,
)


# ================================================================
# Save output
# ================================================================

output_file = os.path.join(
    args.dir,
    f"{args.simname}_overturning_3d.png",
)

fig.savefig(
    output_file,
    dpi=args.dpi,
    bbox_inches="tight",
)

print()
print(f"Saved: {output_file}")


# ================================================================
# Save a second transport-closure diagnostic
# ================================================================

closure_figure, closure_axis = plt.subplots(
    figsize=(10, 4.5),
    constrained_layout=True,
)

closure_axis.plot(
    time_days,
    100.0 * closure_fraction,
    color="black",
    linewidth=1.5,
)

closure_axis.axhline(
    5.0,
    color="darkorange",
    linestyle="--",
    linewidth=1.0,
    label="5% reference",
)

closure_axis.set_title(
    "Overturning transport-closure diagnostic"
)

closure_axis.set_xlabel(
    "Model time (days)"
)

closure_axis.set_ylabel(
    "Maximum top residual / maximum |Ψ| (%)"
)

closure_axis.grid(
    alpha=0.25
)

closure_axis.legend(
    loc="best"
)

closure_output = os.path.join(
    args.dir,
    f"{args.simname}_overturning_3d_closure.png",
)

closure_figure.savefig(
    closure_output,
    dpi=args.dpi,
    bbox_inches="tight",
)

print(f"Saved: {closure_output}")

plt.show()