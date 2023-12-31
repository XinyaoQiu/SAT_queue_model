import argparse
from process import DataLoader
from model import QueueNetwork

def main(args):

    areas = {
        "ramp": [(29.533153, -98.474240), (29.527496, -98.466663), (29.525458, -98.468694),
                (29.526207, -98.469913), (29.527392, -98.468886), (29.532072, -98.475626)],
        "gate": [(29.529930, -98.472447), (29.527392, -98.468886), (29.523794, -98.472543), (29.526706, -98.476224)],
        "runway1": [(29.542771, -98.487069), (29.537275, -98.479589), (29.536158, -98.480835), (29.541730, -98.488152)],
        "runway2": [(29.525458, -98.468694), (29.526207, -98.469913), (29.523821, -98.472556), (29.522807, -98.470954)],
    }

    # time periods can add any time pairs as long as they don't include lost tables.
    time_periods = [
        ("2023-12-07 04", "2023-12-15 04"),
    ]

    data = DataLoader(areas=areas, time_periods=time_periods, data_path="../data/depature")
    data.process_dept()

    model = QueueNetwork(data, output_path="../output1")
    model.solve_dept("ramp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main(args)
