"""Generate a smooth exponential ice base on an existing 2-D cavity bed.

The output keeps the source CSV's x_km coordinates and bathy values. Only
topo (the ice-shelf underside) is replaced by

    topo(x) = z_f + (z_gl - z_f) * exp(-x / lambda)

This script uses only Python's standard library.
"""

from argparse import ArgumentParser
import csv
from math import exp, isclose, isfinite, log
from pathlib import Path
from statistics import median


REPOSITORY = Path(__file__).resolve().parent
DEFAULT_SOURCE = REPOSITORY / "data" / "pineislandbath.csv"
DEFAULT_OUTPUT = REPOSITORY / "data" / "exponential_shelf.csv"
REQUIRED_COLUMNS = ("x_km", "bathy", "topo")


def parse_arguments():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--z-gl",
        type=float,
        default=None,
        help=(
            "Grounding-line ice-base elevation in metres. The default is "
            "the first source topo value, preserving ice-bed contact."
        ),
    )
    parser.add_argument(
        "--z-f",
        type=float,
        default=-350.0,
        help="Far-downstream asymptotic ice-base elevation in metres.",
    )
    parser.add_argument(
        "--lambda-m",
        type=float,
        default=None,
        help=(
            "Exponential e-folding length in metres. By default, choose "
            "the positive value that minimizes the mean squared difference "
            "between the idealized and source topo values."
        ),
    )
    return parser.parse_args()


def load_source(path):
    with path.open(newline="", encoding="utf-8") as source_file:
        reader = csv.DictReader(source_file)

        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row.")

        missing = set(REQUIRED_COLUMNS).difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{path} is missing required columns: "
                + ", ".join(sorted(missing))
            )

        rows = []
        for line_number, row in enumerate(reader, start=2):
            try:
                parsed = {
                    column: float(row[column]) for column in REQUIRED_COLUMNS
                }
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid geometry value at {path}:{line_number}."
                ) from error

            if not all(isfinite(parsed[column]) for column in REQUIRED_COLUMNS):
                raise ValueError(
                    f"Non-finite geometry value at {path}:{line_number}."
                )

            rows.append(parsed)

    if len(rows) < 2:
        raise ValueError(f"{path} must contain at least two geometry rows.")

    if any(
        next_row["x_km"] <= row["x_km"]
        for row, next_row in zip(rows, rows[1:])
    ):
        raise ValueError("Source x_km coordinates must be strictly increasing.")

    return rows


def write_output(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=REQUIRED_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def fit_lambda(source, x_origin_km, z_gl, z_f):
    """Return the least-squares exponential length scale in metres.

    The grounding-line elevation and far-field asymptote are held fixed. The
    optimization is performed in log(lambda), which guarantees lambda > 0.
    """
    distances_m = [
        (row["x_km"] - x_origin_km) * 1000.0 for row in source
    ]
    observed_topo = [row["topo"] for row in source]
    domain_length_m = distances_m[-1]

    if domain_length_m <= 0.0:
        raise ValueError("Source domain length must be positive.")

    def mean_squared_error(log_lambda):
        lambda_m = exp(log_lambda)
        squared_errors = []
        for distance_m, observed in zip(distances_m, observed_topo):
            predicted = z_f + (z_gl - z_f) * exp(
                -distance_m / lambda_m
            )
            squared_errors.append((predicted - observed) ** 2)
        return sum(squared_errors) / len(squared_errors)

    # Search a deliberately broad interval: 1 m to 1000 domain lengths.
    # Golden-section search is dependency-free and sufficient for this
    # one-parameter, smooth least-squares problem.
    lower = log(1.0)
    upper = log(1000.0 * domain_length_m)
    golden_ratio = (5.0 ** 0.5 - 1.0) / 2.0

    left = upper - golden_ratio * (upper - lower)
    right = lower + golden_ratio * (upper - lower)
    left_error = mean_squared_error(left)
    right_error = mean_squared_error(right)

    for _ in range(200):
        if left_error < right_error:
            upper = right
            right = left
            right_error = left_error
            left = upper - golden_ratio * (upper - lower)
            left_error = mean_squared_error(left)
        else:
            lower = left
            left = right
            left_error = right_error
            right = lower + golden_ratio * (upper - lower)
            right_error = mean_squared_error(right)

    return exp((lower + upper) / 2.0)


def main():
    args = parse_arguments()

    if args.lambda_m is not None and args.lambda_m <= 0.0:
        raise ValueError("--lambda-m must be positive.")

    source = load_source(args.source)
    x_origin_km = source[0]["x_km"]
    z_gl = source[0]["topo"] if args.z_gl is None else args.z_gl
    lambda_was_fitted = args.lambda_m is None
    lambda_m = (
        fit_lambda(source, x_origin_km, z_gl, args.z_f)
        if lambda_was_fitted
        else args.lambda_m
    )

    output = []
    for row in source:
        distance_from_gl = (row["x_km"] - x_origin_km) * 1000.0
        idealized_topo = args.z_f + (z_gl - args.z_f) * exp(
            -distance_from_gl / lambda_m
        )
        output.append(
            {
                "x_km": row["x_km"],
                "bathy": row["bathy"],
                "topo": idealized_topo,
            }
        )

    if not isclose(output[0]["topo"], output[0]["bathy"], abs_tol=1e-8):
        raise ValueError(
            "The generated ice base does not meet the unchanged bed at the "
            f"grounding line: topo={output[0]['topo']:.3f} m, "
            f"bathy={output[0]['bathy']:.3f} m. Omit --z-gl to preserve "
            "the source grounding line, or use a source with a matching bed."
        )

    invalid = next(
        (
            row
            for row in output[1:]
            if row["bathy"] >= row["topo"]
        ),
        None,
    )
    if invalid is not None:
        raise ValueError(
            "The idealized ice base intersects the seafloor at "
            f"x={invalid['x_km']:.3f} km: "
            f"bathy={invalid['bathy']:.3f} m, "
            f"topo={invalid['topo']:.3f} m. Adjust z_f or lambda."
        )

    if any(
        next_row["topo"] <= row["topo"]
        for row, next_row in zip(output, output[1:])
    ):
        raise ValueError(
            "The generated ice base is not strictly rising downstream. "
            "For this orientation, z_f must be greater than z_gl."
        )

    write_output(args.output, output)

    spacings_m = [
        (next_row["x_km"] - row["x_km"]) * 1000.0
        for row, next_row in zip(source, source[1:])
    ]
    thicknesses = [
        row["topo"] - row["bathy"] for row in output[1:]
    ]
    domain_length_km = source[-1]["x_km"] - x_origin_km

    print(f"Wrote: {args.output}")
    print(f"Source coordinates retained: {len(output)}")
    print(
        "Source spacing: "
        f"min={min(spacings_m):.3f} m, "
        f"median={median(spacings_m):.3f} m, "
        f"max={max(spacings_m):.3f} m"
    )
    print(f"Domain length: {domain_length_km:.3f} km")
    print(f"z_gl: {z_gl:.3f} m")
    print(f"z_f: {args.z_f:.3f} m")
    lambda_source = "least-squares fit" if lambda_was_fitted else "manual"
    print(f"lambda: {lambda_m / 1000.0:.3f} km ({lambda_source})")
    print(f"Shelf-front ice base: {output[-1]['topo']:.3f} m")
    print(
        "Minimum downstream cavity thickness: "
        f"{min(thicknesses):.3f} m"
    )


if __name__ == "__main__":
    main()
