import csv
from pathlib import Path

import matplotlib.pyplot as plt


REPOSITORY = Path(__file__).resolve().parent

ORIGINAL_FILE = REPOSITORY / "data" / "pineislandbath.csv"
IDEALIZED_FILE = REPOSITORY / "data" / "exponential_shelf.csv"
OUTPUT_FILE = REPOSITORY / "idealized_geometry_comparison.png"


def read_geometry(path):
    x_km = []
    bathy = []
    topo = []

    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)

        for row in reader:
            x_km.append(float(row["x_km"]))
            bathy.append(float(row["bathy"]))
            topo.append(float(row["topo"]))

    return x_km, bathy, topo


original_x, original_bathy, original_topo = read_geometry(
    ORIGINAL_FILE
)

idealized_x, idealized_bathy, idealized_topo = read_geometry(
    IDEALIZED_FILE
)


figure, axis = plt.subplots(figsize=(11, 5))

axis.plot(
    original_x,
    original_topo,
    color="tab:blue",
    linewidth=1.5,
    label="Original ice base",
)

axis.plot(
    idealized_x,
    idealized_topo,
    color="tab:red",
    linewidth=2.5,
    label="Exponential ice base",
)

axis.plot(
    original_x,
    original_bathy,
    color="black",
    linewidth=1.5,
    label="Seafloor",
)

axis.fill_between(
    idealized_x,
    idealized_bathy,
    idealized_topo,
    color="tab:cyan",
    alpha=0.15,
)

axis.set_xlabel("Distance from grounding line (km)")
axis.set_ylabel("Elevation (m)")
axis.set_title("Original and exponential ice-shelf geometries")
axis.grid(alpha=0.25)
axis.legend()

figure.tight_layout()
figure.savefig(OUTPUT_FILE, dpi=200)

print(f"Wrote: {OUTPUT_FILE}")