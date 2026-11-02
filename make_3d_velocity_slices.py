#!/usr/bin/env python3

"""Create a 3-D cavity movie with velocity inside the ocean domain.

The ice-shelf draft and bathymetry are read from the same geometry CSV used
by the model. They are displayed as neutral boundary surfaces. Velocity
magnitude is displayed on a vertical x-z slice through the middle of the
periodic y domain.
"""

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
import xarray as xr


def read_velocity_magnitude(ds, frame):
    """Return velocity magnitude at tracer-cell centers as (z, y, x)."""

    u = (
        ds["u"]
        .isel(time=frame)
        .transpose("z_aac", "y_aca", "x_faa")
        .values
    )

    v = (
        ds["v"]
        .isel(time=frame)
        .transpose("z_aac", "y_afa", "x_caa")
        .values
    )

    w = (
        ds["w"]
        .isel(time=frame)
        .transpose("z_aaf", "y_aca", "x_caa")
        .values
    )

    # Convert face-centered velocity components to tracer-cell centers.
    u_center = 0.5 * (
        u[:, :, :-1] + u[:, :, 1:]
    )

    # The y direction is periodic.
    v_center = 0.5 * (
        v + np.roll(v, -1, axis=1)
    )

    w_center = 0.5 * (
        w[:-1, :, :] + w[1:, :, :]
    )

    return np.sqrt(
        u_center**2
        + v_center**2
        + w_center**2
    )


def load_geometry(path, x_metres):
    """Read and interpolate the geometry onto the model x coordinates."""

    table = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
    )

    required_columns = {
        "x_km",
        "bathy",
        "topo",
    }

    if (
        table.dtype.names is None
        or not required_columns.issubset(table.dtype.names)
    ):
        raise SystemExit(
            f"Geometry file {path} must contain "
            "x_km, bathy, and topo columns."
        )

    geometry_x = (
        np.asarray(
            table["x_km"],
            dtype=float,
        )
        * 1000.0
    )

    seabed = np.asarray(
        table["bathy"],
        dtype=float,
    )

    ice_draft = np.asarray(
        table["topo"],
        dtype=float,
    )

    order = np.argsort(geometry_x)

    geometry_x = geometry_x[order]
    seabed = seabed[order]
    ice_draft = ice_draft[order]

    seabed_on_grid = np.interp(
        x_metres,
        geometry_x,
        seabed,
    )

    ice_draft_on_grid = np.interp(
        x_metres,
        geometry_x,
        ice_draft,
    )

    return seabed_on_grid, ice_draft_on_grid


def make_extruded_surface(
    x_km,
    y_km,
    height_km,
):
    """Extrude an x-z profile through the model y direction."""

    xx, yy = np.meshgrid(
        x_km,
        y_km,
        indexing="ij",
    )

    zz = np.broadcast_to(
        height_km[:, None],
        xx.shape,
    )

    return pv.StructuredGrid(
        xx,
        yy,
        zz,
    ).extract_surface(
        algorithm="dataset_surface",
    )


def make_velocity_slices(
    x_km,
    y_km,
    z_km,
    wet,
    first_speed,
    number_of_slices=5,
):
    """Construct several wet x-z velocity slices across the y direction."""

    # Avoid placing slices directly on the periodic boundaries.
    y_indices = np.linspace(
        1,
        len(y_km) - 2,
        number_of_slices,
    ).round().astype(int)

    y_indices = np.unique(y_indices)

    slices = []

    for j in y_indices:
        slice_x, slice_z = np.meshgrid(
            x_km,
            z_km,
            indexing="ij",
        )

        slice_y = np.full_like(
            slice_x,
            y_km[j],
        )

        slice_grid = pv.StructuredGrid(
            slice_x,
            slice_y,
            slice_z,
        )

        # Model arrays are (z, y, x), whereas this grid is (x, z).
        slice_speed = (
            first_speed[:, j, :]
            .T
            .ravel(order="F")
        )

        slice_wet = (
            wet[:, j, :]
            .T
            .ravel(order="F")
        )

        slice_grid.point_data["speed"] = slice_speed
        slice_grid.point_data["wet"] = slice_wet.astype(float)
        slice_grid.point_data["point_id"] = np.arange(
            slice_grid.n_points,
            dtype=int,
        )

        wet_slice = slice_grid.threshold(
            value=0.5,
            scalars="wet",
            preference="point",
            all_scalars=True,
        )

        point_ids = np.asarray(
            wet_slice.point_data["point_id"],
            dtype=int,
        )

        slices.append(
            {
                "y_index": j,
                "grid": wet_slice,
                "point_ids": point_ids,
            }
        )

    return slices


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "simname",
        help="Simulation name without .nc",
    )

    parser.add_argument(
        "--dir",
        default="output",
        help="Directory containing the NetCDF file",
    )

    parser.add_argument(
        "--out",
        default=None,
        help="Output MP4 or PNG path",
    )

    parser.add_argument(
        "--geometry",
        default=str(
            Path(__file__).resolve().parent
            / "data"
            / "pineislandbath.csv"
        ),
        help="Geometry CSV with x_km, bathy, and topo columns",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="Movie frame rate",
    )

    parser.add_argument(
        "--speed-limit",
        type=float,
        default=None,
        help="Maximum velocity color value in m/s",
    )

    parser.add_argument(
        "--y-exaggeration",
        type=float,
        default=1.0,
        help="Visual y-direction exaggeration",
    )

    parser.add_argument(
        "--z-exaggeration",
        type=float,
        default=20.0,
        help="Visual vertical exaggeration",
    )

    parser.add_argument(
        "--preview",
        action="store_true",
        help="Write one PNG instead of an MP4",
    )

    args = parser.parse_args()

    source = (
        Path(args.dir)
        / f"{args.simname}.nc"
    )

    geometry_path = Path(args.geometry)

    if args.out is not None:
        destination = Path(args.out)
    elif args.preview:
        destination = (
            Path(args.dir)
            / f"{args.simname}_velocity_preview.png"
        )
    else:
        destination = (
            Path(args.dir)
            / f"{args.simname}_velocity.mp4"
        )

    if not source.exists():
        raise SystemExit(
            f"Input file not found: {source}"
        )

    if not geometry_path.exists():
        raise SystemExit(
            f"Geometry file not found: {geometry_path}"
        )

    ds = xr.open_dataset(
        source,
        decode_times=False,
    )

    required_variables = {
        "time",
        "u",
        "v",
        "w",
        "mask",
        "x_caa",
        "y_aca",
        "z_aac",
    }

    missing = required_variables.difference(
        ds.variables
    )

    if missing:
        raise SystemExit(
            "Missing variables: "
            + ", ".join(sorted(missing))
        )

    # Model coordinates in metres.
    x_m = ds["x_caa"].values
    y_m = ds["y_aca"].values
    z_m = ds["z_aac"].values

    # Plot coordinates in kilometres.
    x_km = x_m / 1000.0
    y_km = y_m / 1000.0
    z_km = z_m / 1000.0

    seabed_m, ice_draft_m = load_geometry(
        geometry_path,
        x_m,
    )

    # Oceananigans GridFittedBoundary convention:
    # 0 = ocean and 1 = solid.
    solid = (
        ds["mask"]
        .transpose(
            "z_aac",
            "y_aca",
            "x_caa",
        )
        .values
        > 0.5
    )

    wet = ~solid

    if not wet.any():
        raise SystemExit(
            "The mask contains no wet cells."
        )

    if not solid.any():
        raise SystemExit(
            "The mask contains no solid cells."
        )

    ice_surface = make_extruded_surface(
        x_km,
        y_km,
        ice_draft_m / 1000.0,
    )

    bed_surface = make_extruded_surface(
        x_km,
        y_km,
        seabed_m / 1000.0,
    )

    first_speed = read_velocity_magnitude(
        ds,
        frame=0,
    )

    finite_speed = first_speed[
        wet & np.isfinite(first_speed)
    ]

    if finite_speed.size == 0:
        raise SystemExit(
            "No finite velocity values were found "
            "inside the ocean cavity."
        )

    if args.speed_limit is None:
        speed_limit = np.nanpercentile(
            finite_speed,
            98,
        )
    else:
        speed_limit = args.speed_limit

    if (
        not np.isfinite(speed_limit)
        or speed_limit <= 0.0
    ):
        speed_limit = 1.0e-6

    velocity_slices = make_velocity_slices(
    x_km,
    y_km,
    z_km,
    wet,
    first_speed,
    number_of_slices=5,
    )

    plotter = pv.Plotter(
        window_size=(1400, 900),
        off_screen=args.preview,
    )

    # The ice and seabed are geometry, not velocity fields.
    plotter.add_mesh(
        ice_surface,
        color="lightblue",
        opacity=0.15,
        smooth_shading=True,
        show_edges=False,
        label="ice-shelf base",
    )

    plotter.add_mesh(
        bed_surface,
        color="saddlebrown",
        opacity=0.75,
        smooth_shading=True,
        show_edges=False,
        label="sea floor",
    )

    velocity_actors = []

    for slice_number, item in enumerate(velocity_slices):
        actor = plotter.add_mesh(
            item["grid"],
            scalars="speed",
            preference="point",
            cmap="Blues",
            clim=(0.0, speed_limit),
            opacity=0.2,
            n_colors=256,
            smooth_shading=False,
            lighting=False,
            show_edges=False,
            show_scalar_bar=(slice_number == 0),
            scalar_bar_args={
                "title": "Velocity magnitude (m/s)",
            },
            label=f"velocity slice y={y_km[item['y_index']]:.2f} km",
        )
        velocity_actors.append(actor)

    plotter.set_scale(
        xscale=1.0,
        yscale=args.y_exaggeration,
        zscale=args.z_exaggeration,
    )

    plotter.add_axes()

    plotter.show_grid(
        xtitle="x (km)",
        ytitle="y (km)",
        ztitle=(
            f"z display (km; "
            f"{args.z_exaggeration:g}x exaggerated)"
        ),
    )

    title = plotter.add_text(
        (
            f"{args.simname}: velocity across five y slices\n"
            f"vertical exaggeration = {args.z_exaggeration:g}x"
        ),
        position="upper_edge",
        font_size=13,
    )

    plotter.camera_position = "iso"
    plotter.camera.azimuth = -20
    plotter.camera.elevation = 12
    plotter.reset_camera()

    if args.preview:
        plotter.show(
            screenshot=str(destination),
            auto_close=True,
        )

        ds.close()

        print(
            f"Wrote: {destination}"
        )

        return

    plotter.open_movie(
        str(destination),
        framerate=args.fps,
        quality=8,
    )

    plotter.show(
        auto_close=False,
        interactive_update=True,
    )

    times = ds["time"].values

    for frame, seconds in enumerate(times):
        speed = read_velocity_magnitude(
            ds,
            frame=frame,
        )

        for item, actor in zip(
            velocity_slices,
            velocity_actors,
        ):
            j = item["y_index"]

            slice_speed = (
                speed[:, j, :]
                .T
                .ravel(order="F")
            )

            item["grid"].point_data["speed"] = (
                slice_speed[item["point_ids"]]
            )

            item["grid"].GetPointData().GetArray(
                "speed"
            ).Modified()

            actor.mapper.scalar_range = (
                0.0,
                speed_limit,
            )

        title.SetText(
            0,
            (
                f"{args.simname}: velocity across five y slices\n"
                f"time = {seconds / 86400.0:.2f} days   "
                f"frame {frame + 1}/{len(times)}"
            ),
        )

        plotter.render()
        plotter.write_frame()

    plotter.close()
    ds.close()

    print(
        f"Wrote: {destination}"
    )


if __name__ == "__main__":
    main()