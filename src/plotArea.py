import matplotlib.pyplot as plt
import tilemapbase as tmb
import numpy as np
from matplotlib.patches import Polygon
import os

areas = {
    "ramp": [(29.533153, -98.474240), (29.527496, -98.466663), (29.525458, -98.468694),
             (29.526207, -98.469913), (29.527392, -98.468886), (29.532072, -98.475626)],
    # "gate": [(29.529930, -98.472447), (29.527392, -98.468886), (29.523794, -98.472543), (29.526706, -98.476224)],
    "runway1": [(29.542771, -98.487069), (29.537275, -98.479589), (29.536158, -98.480835), (29.541730, -98.488152)],
    "runway2": [(29.525458, -98.468694), (29.526207, -98.469913), (29.523821, -98.472556), (29.522807, -98.470954)],
}


def plotArea(output_path="../output1"):
    """
    Plot the boundary of area.
    """

    sat = {"latitude": 29.533689, "longitude": -98.469704}

    os.makedirs(output_path, exist_ok=True)
    tmb.start_logging()
    tmb.init(create=True)

    t = tmb.tiles.build_OSM()
    lon_range = 0.02
    lat_range = 0.012

    # create the figure
    extent = tmb.Extent.from_lonlat(
        longitude_min=sat["longitude"] - lon_range,
        longitude_max=sat["longitude"] + lon_range,
        latitude_min=sat["latitude"] - lat_range,
        latitude_max=sat["latitude"] + lat_range,
    )
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    plotter = tmb.Plotter(extent, t, width=600)
    plotter.plot(ax, t)

    # draw polygon on the figure
    x, y = zip(*[tmb.project(lon, lat) for lat, lon in areas["ramp"]])
    polygon = Polygon(np.column_stack((x, y)), closed=True,
                    edgecolor="red", facecolor='none', linestyle='dashed')
    ax.add_patch(polygon)


    x, y = zip(*[tmb.project(lon, lat) for lat, lon in areas["runway1"]])
    polygon = Polygon(np.column_stack((x, y)), closed=True,
                    edgecolor="blue", facecolor='none', linestyle='dashed')
    ax.add_patch(polygon)

    x, y = zip(*[tmb.project(lon, lat) for lat, lon in areas["runway2"]])
    polygon = Polygon(np.column_stack((x, y)), closed=True,
                    edgecolor="green", facecolor='none', linestyle='dashed')
    ax.add_patch(polygon)

    plt.savefig(f"{output_path}/areas.png")


plotArea()