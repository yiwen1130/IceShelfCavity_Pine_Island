#!/usr/bin/env python3

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


# ---------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Plot T/S anomalies and 2-D overturning streamfunction."
)

parser.add_argument(
    "simname",
    help="Simulation name, for example constant_10",
)

parser.add_argument(
    "--dir",
    default="output",
    help="Directory containing <simname>.nc",
)

parser.add_argument(
    "--Tlim",
    type=float,
    default=0.2,
    help="Symmetric temperature-anomaly color limit [deg C]",
)

parser.add_argument(
    "--Slim",
    type=float,
    default=0.05,
    help="Symmetric salinity-anomaly color limit [psu]",
)

parser.add_argument(
    "--psilim",
    type=float,
    default=None,
    help="Optional symmetric streamfunction color limit [m2/s]",
)

args = parser.parse_args()


# ---------------------------------------------------------------------
# Open model output
# ---------------------------------------------------------------------

ncfile = os.path.join(args.dir, f"{args.simname}.nc")

if not os.path.exists(ncfile):
    raise SystemExit(f"Could not find {ncfile}")

ds = xr.open_dataset(ncfile, decode_times=False)

print(f"Reading: {ncfile}")
print(f"Available variables: {list(ds.data_vars)}")


# ---------------------------------------------------------------------
# Read the exact ambient profiles used by the Julia model
# ---------------------------------------------------------------------

stratification_file = os.path.join("scripts", "stratification.jl")

if not os.path.exists(stratification_file):
    raise SystemExit(
        f"Could not find {stratification_file}. "
        "Run this script from the IceShelfCavity directory."
    )

with open(stratification_file) as f:
    stratification_text = f.read()


def read_four_constants(names):
    """Read a Julia line such as:
       const aT, cT, z0T, hT = 0.30, -0.9, -390.0, 125.0
    """

    pattern = (
        r"const\s+"
        + r",\s*".join(names)
        + r"\s*=\s*([^\n]+)"
    )

    match = re.search(pattern, stratification_text)

    if match is None:
        raise RuntimeError(
            f"Could not find constants {names} in {stratification_file}"
        )

    numbers = re.findall(
        r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
        match.group(1),
    )

    if len(numbers) < 4:
        raise RuntimeError(f"Could not parse values for {names}")

    return tuple(float(value) for value in numbers[:4])


aT, cT, z0T, hT = read_four_constants(
    ["aT", "cT", "z0T", "hT"]
)

aS, cS, z0S, hS = read_four_constants(
    ["aS", "cS", "z0S", "hS"]
)


def ambient_temperature(z):
    return aT + cT * np.tanh((z - z0T) / hT)


def ambient_salinity(z):
    return aS + cS * np.tanh((z - z0S) / hS)


# ---------------------------------------------------------------------
# Read ice-base and seafloor geometry
# ---------------------------------------------------------------------

geometry = pd.read_csv("data/pineislandbath.csv")

geometry_x = geometry["x_km"].to_numpy() * 1000.0
geometry_top = geometry["topo"].to_numpy()
geometry_bottom = geometry["bathy"].to_numpy()


# ---------------------------------------------------------------------
# Helpers for Oceananigans staggered coordinates
# ---------------------------------------------------------------------

def spatial_dimensions(data_array):
    """Find x and z dimension names, such as xC/xF and zC/zF."""

    xdim = next(
        dim for dim in data_array.dims
        if dim.lower().startswith("x")
    )

    zdim = next(
        dim for dim in data_array.dims
        if dim.lower().startswith("z")
    )

    return xdim, zdim


def last_xz_field(variable):
    """Return x, z, and the final field as an array shaped (z, x)."""

    data_array = ds[variable].isel(time=-1).squeeze(drop=True)
    xdim, zdim = spatial_dimensions(data_array)

    data_array = data_array.transpose(zdim, xdim)

    x = data_array[xdim].to_numpy()
    z = data_array[zdim].to_numpy()
    values = data_array.to_numpy().astype(float)

    # Make z run from bottom to top.
    if z[0] > z[-1]:
        z = z[::-1]
        values = values[::-1, :]

    return x, z, values


def wet_mask(x, z):
    """True inside the ocean cavity; false inside ice or seafloor."""

    top = np.interp(x, geometry_x, geometry_top)
    bottom = np.interp(x, geometry_x, geometry_bottom)

    Z = z[:, None]

    return (Z < top[None, :]) & (Z > bottom[None, :])


# ---------------------------------------------------------------------
# Temperature and salinity anomalies at the final model time
# ---------------------------------------------------------------------

xT, zT, temperature = last_xz_field("T")
xS, zS, salinity = last_xz_field("S")

temperature_wet = wet_mask(xT, zT)
salinity_wet = wet_mask(xS, zS)

temperature_anomaly = (
    temperature - ambient_temperature(zT)[:, None]
)

salinity_anomaly = (
    salinity - ambient_salinity(zS)[:, None]
)

temperature_anomaly[~temperature_wet] = np.nan
salinity_anomaly[~salinity_wet] = np.nan


# ---------------------------------------------------------------------
# Overturning streamfunction
# ---------------------------------------------------------------------

def calculate_streamfunction(u_data_array):
    """Calculate the overturning streamfunction in the x-z plane.

    Definition:
        psi(x,z) = - integral from the seafloor to z of u(x,z') dz'

    Sign convention:
        positive psi = clockwise circulation
        negative psi = counterclockwise circulation

    Returns
    -------
    x : one-dimensional array [m]
    z : one-dimensional array [m]
    psi : two-dimensional array shaped (z, x) [m2/s]
    """

    # Remove dimensions of length one, such as the unresolved y dimension.
    u_data_array = u_data_array.squeeze(drop=True)

    # Identify the horizontal and vertical dimension names.
    xdim, zdim = spatial_dimensions(u_data_array)

    # Arrange the velocity array so its shape is (z, x).
    u_data_array = u_data_array.transpose(zdim, xdim)

    # Extract coordinates and velocity as NumPy arrays.
    x = u_data_array[xdim].to_numpy()
    z = u_data_array[zdim].to_numpy()
    u = u_data_array.to_numpy().astype(float)

    # Ensure the vertical coordinate runs from the bottom upward.
    if z[0] > z[-1]:
        z = z[::-1]
        u = u[::-1, :]

    # True in ocean cells and false inside the ice or seabed.
    mask = wet_mask(x, z)

    # Solid cells and missing values contribute zero transport.
    u_for_integration = np.where(
        mask & np.isfinite(u),
        u,
        0.0,
    )

    # Calculate the thickness represented by each vertical level.
    dz = np.gradient(z)

    # Vertically integrate u from the bottom upward.
    #
    # The leading minus sign makes clockwise circulation positive.
    psi = np.cumsum(
        u_for_integration * dz[:, None],
        axis=0,
    )

    # Hide the streamfunction inside the ice and seabed.
    psi[~mask] = np.nan

    return x, z, psi


# Calculate the streamfunction at the final saved model time.
u_final = ds["u"].isel(time=-1)
xpsi, zpsi, psi = calculate_streamfunction(u_final)

# Find the largest streamfunction values of both signs.
psi_min = float(np.nanmin(psi))
psi_max = float(np.nanmax(psi))

# Direction-specific cell strengths.
counterclockwise_strength = max(0.0, psi_max)
clockwise_strength = max(0.0, -psi_min)

# Strongest cell, regardless of its direction.
maximum_absolute_psi = float(np.nanmax(np.abs(psi)))

print()
print("Final overturning diagnostic")
print("Sign convention: positive psi is clockwise")
print(f"  minimum psi                  = {psi_min:+.6e} m2/s")
print(f"  maximum psi                  = {psi_max:+.6e} m2/s")
print(f"  clockwise-cell strength      = {clockwise_strength:.6e} m2/s")
print(f"  counterclockwise-cell strength = {counterclockwise_strength:.6e} m2/s")
print(f"  maximum absolute psi         = {maximum_absolute_psi:.6e} m2/s")
# ---------------------------------------------------------------------
# Calculate overturning-cell strength at every saved time
#
# Sign convention:
#   positive psi = counterclockwise circulation
#   negative psi = clockwise circulation
# ---------------------------------------------------------------------

times = ds["time"].to_numpy().astype(float)

clockwise_strength = np.empty(times.size)
counterclockwise_strength = np.empty(times.size)
maximum_absolute_psi = np.empty(times.size)

for index in range(times.size):
    _, _, psi_at_time = calculate_streamfunction(
        ds["u"].isel(time=index)
    )

    psi_min_at_time = np.nanmin(psi_at_time)
    psi_max_at_time = np.nanmax(psi_at_time)

    # Strongest clockwise cell: largest positive psi
    counterclockwise_strength[index] = max(
        0.0,
        psi_max_at_time,
    )

    # Strongest counterclockwise cell: magnitude of most negative psi
    clockwise_strength[index] = max(
        0.0,
        -psi_min_at_time,
    )

    # Strongest circulation of either sign
    maximum_absolute_psi[index] = np.nanmax(
        np.abs(psi_at_time)
    )

time_days = times / 86400.0
# ---------------------------------------------------------------------
# Domain-mean temperature and salinity
# ---------------------------------------------------------------------

T_data = ds["T"]
T_xdim, T_zdim = spatial_dimensions(T_data)

T_x = T_data[T_xdim].to_numpy()
T_z = T_data[T_zdim].to_numpy()

T_ocean_mask = xr.DataArray(
    wet_mask(T_x, T_z),
    dims=(T_zdim, T_xdim),
    coords={
        T_zdim: T_data[T_zdim],
        T_xdim: T_data[T_xdim],
    },
)

domain_mean_T = (
    T_data.where(T_ocean_mask)
    .mean(dim=(T_zdim, T_xdim))
    .to_numpy()
)


S_data = ds["S"]
S_xdim, S_zdim = spatial_dimensions(S_data)

S_x = S_data[S_xdim].to_numpy()
S_z = S_data[S_zdim].to_numpy()

S_ocean_mask = xr.DataArray(
    wet_mask(S_x, S_z),
    dims=(S_zdim, S_xdim),
    coords={
        S_zdim: S_data[S_zdim],
        S_xdim: S_data[S_xdim],
    },
)

domain_mean_S = (
    S_data.where(S_ocean_mask)
    .mean(dim=(S_zdim, S_xdim))
    .to_numpy()
)


# ---------------------------------------------------------------------
# Choose streamfunction color scale
# ---------------------------------------------------------------------

if args.psilim is None:
    finite_psi = np.abs(psi[np.isfinite(psi)])

    if finite_psi.size == 0:
        psi_limit = 1.0e-12
    else:
        psi_limit = np.nanpercentile(finite_psi, 98)

    psi_limit = max(psi_limit, 1.0e-12)
else:
    psi_limit = args.psilim


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

fig, axes = plt.subplots(
    4,
    1,
    figsize=(12, 15),
    constrained_layout=True,
)

solid_color = "0.65"

# Temperature anomaly
temperature_plot = axes[0].pcolormesh(
    xT / 1000,
    zT,
    temperature_anomaly,
    cmap="RdBu_r",
    vmin=-args.Tlim,
    vmax=args.Tlim,
    shading="auto",
)

axes[0].set_title(
    "Temperature anomaly: T − ambient T(z)"
)

axes[0].set_ylabel("z (m)")

fig.colorbar(
    temperature_plot,
    ax=axes[0],
    label="T anomaly (°C)",
)


# Salinity anomaly
salinity_plot = axes[1].pcolormesh(
    xS / 1000,
    zS,
    salinity_anomaly,
    cmap="RdBu_r",
    vmin=-args.Slim,
    vmax=args.Slim,
    shading="auto",
)

axes[1].set_title(
    "Salinity anomaly: S − ambient S(z)"
)

axes[1].set_ylabel("z (m)")

fig.colorbar(
    salinity_plot,
    ax=axes[1],
    label="S anomaly (psu)",
)


# ---------------------------------------------------------------------
# Streamfunction cross-section
# ---------------------------------------------------------------------

levels = np.linspace(
    -psi_limit,
    psi_limit,
    31,
)

psi_plot = axes[2].contourf(
    xpsi / 1000,
    zpsi,
    psi,
    levels=levels,
    cmap="RdBu_r",
    extend="both",
)

# Add black streamfunction contours.
axes[2].contour(
    xpsi / 1000,
    zpsi,
    psi,
    levels=levels[::3],
    colors="k",
    linewidths=0.35,
    alpha=0.45,
)

# Emphasize the zero contour, which separates cells with opposite signs.
axes[2].contour(
    xpsi / 1000,
    zpsi,
    psi,
    levels=[0.0],
    colors="k",
    linewidths=0.9,
)

axes[2].set_title(
    "Overturning streamfunction: positive = counterclockwise"
)

axes[2].set_ylabel("z (m)")

fig.colorbar(
    psi_plot,
    ax=axes[2],
    label="psi (m²/s per metre width)",
)


# ---------------------------------------------------------------------
# Overturning-cell strength through time
# ---------------------------------------------------------------------

# axes[3].plot(
#     time_days,
#     clockwise_strength,
#     color="firebrick",
#     linewidth=1.5,
#     label="Clockwise cell: max(psi)",
# )

# axes[3].plot(
#     time_days,
#     counterclockwise_strength,
#     color="steelblue",
#     linewidth=1.5,
#     label="Counterclockwise cell: -min(psi)",
# )

axes[3].plot(
    time_days,
    counterclockwise_strength,
    color="black",
    linewidth=1.2,
    linestyle="--",
    label="Strongest cell: max(psi)",
)

axes[3].set_title(
    "Positive streamfunction strengths through time"
)

axes[3].set_xlabel("Model time (days)")
axes[3].set_ylabel(
    "Cell strength (m²/s per metre width)"
)

axes[3].grid(alpha=0.25)
axes[3].legend()

# Geometry overlay on the spatial panels
for axis in axes[:3]:
    axis.fill_between(
        geometry_x / 1000,
        geometry_top,
        geometry_top.max() + 100,
        color=solid_color,
        zorder=10,
    )

    axis.fill_between(
        geometry_x / 1000,
        geometry_bottom.min() - 100,
        geometry_bottom,
        color=solid_color,
        zorder=10,
    )

    axis.set_xlim(
        geometry_x.min() / 1000,
        geometry_x.max() / 1000,
    )

    axis.set_xlabel(
        "Distance from grounding line (km)"
    )


fig.suptitle(
    f"{args.simname}: anomalies and overturning",
    fontsize=15,
)

output_file = os.path.join(
    args.dir,
    f"{args.simname}_overturning.png",
)

fig.savefig(
    output_file,
    dpi=180,
    bbox_inches="tight",
)

print()
print(f"Saved: {output_file}")

# ---------------------------------------------------------------------
# Plot domain-mean temperature and salinity
# ---------------------------------------------------------------------

fig2, axes2 = plt.subplots(
    2,
    1,
    figsize=(10, 7),
    sharex=True,
    constrained_layout=True,
)

axes2[0].plot(
    time_days,
    domain_mean_T,
    color="firebrick",
    linewidth=1.5,
)

axes2[0].set_title(
    "Domain-mean temperature"
)

axes2[0].set_ylabel("Mean T (°C)")
axes2[0].grid(alpha=0.25)


axes2[1].plot(
    time_days,
    domain_mean_S,
    color="steelblue",
    linewidth=1.5,
)

axes2[1].set_title(
    "Domain-mean salinity"
)

axes2[1].set_xlabel("Model time (days)")
axes2[1].set_ylabel("Mean S (psu)")
axes2[1].grid(alpha=0.25)


mean_TS_output = os.path.join(
    args.dir,
    f"{args.simname}_mean_TS.png",
)

fig2.savefig(
    mean_TS_output,
    dpi=180,
    bbox_inches="tight",
)

print(f"Saved: {mean_TS_output}")