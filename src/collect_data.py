import csv
from datetime import datetime
import time
from FlightRadar24.api import FlightRadar24API


def convert_unix_to_utc(timestamp):
    return datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H')


def save_to_csv(data, filename):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)


fr_api = FlightRadar24API()
sat = {"latitude": 29.533689, "longitude": -98.469704}


def in_sat(f):
    return sat["latitude"] - 0.1 < f.latitude < sat["latitude"] + 0.1 \
        and sat["longitude"] - 0.1 < f.longitude < sat["longitude"] + 0.1 \
        and f.on_ground == 1


def handle_f(f, type):
    timestamp = f.time
    utc_time = convert_unix_to_utc(timestamp)
    curr_t = utc_time[:13]

    data_to_save = [
        f.id, f.latitude, f.longitude, f.altitude, f.ground_speed, f.time, f.heading
    ]

    save_to_csv(data_to_save, f"../data/{type}/{curr_t}.csv")


print(f"Collecting data...")
while True:
    allFlights = fr_api.get_flights(airport="SAT")
    deptFlights = []
    arrvFilghts = []

    for f in allFlights:
        if f.destination_airport_iata == "SAT" and in_sat(f):
            arrvFilghts.append(f)
        elif f.origin_airport_iata == "SAT" and in_sat(f):
            deptFlights.append(f)
    for flight in deptFlights:
        handle_f(flight, "depature")
    for flight in arrvFilghts:
        handle_f(flight, "arrival")

    time.sleep(1)
