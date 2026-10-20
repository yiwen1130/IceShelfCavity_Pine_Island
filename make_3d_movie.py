#!/usr/bin/env python3
"""Render three intersecting slices of a 3-D cavity NetCDF file as an MP4."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
import xarray as xr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("simname", help="simulation name, without .nc")
    parser.add_argument("--dir", default="output", help="directory containing the NetCDF file")
    parser.add_argument("--out", default=None, help="output MP4 path")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--y-exaggeration", type=float, default=8.0)
    parser.add_argument("--z-exaggeration", type=float, default=15.0)
    args = parser.parse_args()

    source = Path(args.dir) / f"{args.simname}.nc"
    destination = Path(args.out) if args.out else Path(args.dir) / f"{args.simname}_3d.mp4"
    if not source.exists():
        raise SystemExit(f"Input file not found: {source}")

    ds = xr.open_dataset(source, decode_times=False)
    required = {"time", "T", "mask", "x_caa", "y_aca", "z_aac"}
    missing = required.difference(ds.variables)
    if missing:
        raise SystemExit(f"Missing variables: {', '.join(sorted(missing))}")

    x = ds.x_caa.values / 1000.0
    y = ds.y_aca.values / 1000.0 * args.y_exaggeration
    z = ds.z_aac.values / 1000.0 * args.z_exaggeration
    mask = ds["mask"].values > 0.5  # stored as (z, y, x)
    valid_temperature = ds["T"].values[:, mask]
    lower, upper = np.nanpercentile(valid_temperature, [2, 98])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
        lower, upper = -1.5, 1.0

    norm = colors.Normalize(lower, upper)
    cmap = cm.plasma.copy()
    cmap.set_bad(alpha=0)
    ix, iy, iz = len(x) // 2, len(y) // 2, len(z) // 2
    Yx, Zx = np.meshgrid(y, z)
    Xy, Zy = np.meshgrid(x, z)
    Xz, Yz = np.meshgrid(x, y)

    fig = plt.figure(figsize=(14, 8), dpi=120)
    writer = animation.FFMpegWriter(fps=args.fps, bitrate=5000)
    frames = ds.sizes["time"]

    with writer.saving(fig, str(destination), dpi=120):
        for frame, seconds in enumerate(ds.time.values):
            fig.clear()
            ax = fig.add_subplot(111, projection="3d")
            temperature = ds["T"].isel(time=frame).values  # (z, y, x)
            data = np.ma.masked_where(~mask, temperature)

            ax.plot_surface(
                np.full_like(Yx, x[ix]), Yx, Zx,
                facecolors=cmap(norm(data[:, :, ix])), shade=False,
                rstride=1, cstride=1,
            )
            ax.plot_surface(
                Xy, np.full_like(Xy, y[iy]), Zy,
                facecolors=cmap(norm(data[:, iy, :])), shade=False,
                rstride=1, cstride=1,
            )
            ax.plot_surface(
                Xz, Yz, np.full_like(Xz, z[iz]),
                facecolors=cmap(norm(data[iz, :, :])), shade=False,
                rstride=1, cstride=1,
            )

            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(y.min(), y.max())
            ax.set_zlim(z.min(), z.max())
            ax.set_box_aspect((x.ptp(), y.ptp(), z.ptp()))
            ax.set_xlabel("x: grounding line to shelf front (km)")
            ax.set_ylabel(f"y (km, x{args.y_exaggeration:g})")
            ax.set_zlabel(f"z (km, x{args.z_exaggeration:g})")
            ax.view_init(elev=25, azim=-58)
            ax.set_title(
                f"{args.simname}: 3-D temperature slices\n"
                f"t = {seconds / 3600:.2f} hours   frame {frame + 1}/{frames}"
            )
            scalar = cm.ScalarMappable(norm=norm, cmap=cmap)
            fig.colorbar(scalar, ax=ax, shrink=0.65, pad=0.08, label="Temperature (°C)")
            fig.tight_layout()
            writer.grab_frame()

    ds.close()
    plt.close(fig)
    print(f"wrote {destination} ({frames} frames)")


if __name__ == "__main__":
    main()
