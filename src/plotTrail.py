import matplotlib.pyplot as plt
import tilemapbase as tmb
from FlightRadar24.api import FlightRadar24API
import numpy as np
import time


fr_api = FlightRadar24API()
sat = {"latitude": 29.533689, "longitude": -98.469704}


deptFlights = []
arrvFlights = []

allFlights = fr_api.get_flights()

# tweak this to get a flight that has an actual trail - tricky in real time
for f in allFlights:
    if f.origin_airport_iata == "SAT" and f.on_ground == 1:
        deptFlights.append(f)
    if f.destination_airport_iata == "SAT" and f.on_ground == 1:
        arrvFlights.append(f)

# depature
tmb.start_logging()
tmb.init(create=True)

t = tmb.tiles.build_OSM()
lon_range = 0.02
lat_range = 0.012

extent = tmb.Extent.from_lonlat(
    longitude_min=sat["longitude"] - lon_range, 
    longitude_max=sat["longitude"] + lon_range,
    latitude_min=sat["latitude"] - lat_range, 
    latitude_max=sat["latitude"] + lat_range, 
)
# extent = extent.to_aspect(1.0)
plt.figure()
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

plotter = tmb.Plotter(extent, t, width=600)
plotter.plot(ax, t)
if len(deptFlights) != 0:
    deptTrail = fr_api.get_flight_details(deptFlights[0].id)["trail"]
else:
    deptTrail = []

latlong = np.zeros((len(deptTrail), 2))
for i, point in enumerate(deptTrail):
    latlong[i, :] = [point["lng"], point["lat"]]

for i, point in enumerate(latlong):
    print(point)

    # do this part to associate the map with the data!
    x, y = tmb.project(*point)
    ax.scatter(x, y, color="black", s=5)
    # if i > 10:
    #     break
plt.savefig(f"../output/depature_path/{int(time.time())}.png")



# arrival
tmb.start_logging()
tmb.init(create=True)

t = tmb.tiles.build_OSM()
lon_range = 0.02
lat_range = 0.012

extent = tmb.Extent.from_lonlat(
    longitude_min=sat["longitude"] - lon_range, 
    longitude_max=sat["longitude"] + lon_range,
    latitude_min=sat["latitude"] - lat_range, 
    latitude_max=sat["latitude"] + lat_range, 
)
# extent = extent.to_aspect(1.0)
plt.figure()
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

plotter = tmb.Plotter(extent, t, width=600)
plotter.plot(ax, t)
if len(arrvFlights) != 0:
    arrvTrail = fr_api.get_flight_details(arrvFlights[0].id)["trail"]
else:
    arrvTrail = []

latlong = np.zeros((len(arrvTrail), 2))
for i, point in enumerate(arrvTrail):
    latlong[i, :] = [point["lng"], point["lat"]]

for i, point in enumerate(latlong):
    print(point)

    # do this part to associate the map with the data!
    x, y = tmb.project(*point)
    ax.scatter(x, y, color="white", s=5)
    # if i > 10:
    #     break


plt.savefig(f"../output/arrival_path/{int(time.time())}.png")