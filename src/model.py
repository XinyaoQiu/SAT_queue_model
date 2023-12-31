from process import DataLoader
from pandas import DataFrame
import numpy as np
import math
from scipy.integrate import quad
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime
import os

class QueueNetwork:
    def __init__(self, data: DataLoader, bin_size=300, percentile=0.7, output_path="../output", tz_diff=5, max_x=5):
        """
        Queueing network model.

        Parameters:
        ----------
        data: the flight data
        bin_size: the size of one bin (default is 300s)
        percentile: the percentile of service time bar (default is 70%)
        output_path: the directory storing output figures
        tz_diff: the time difference between two time zone, UTC and local time (default is 5h)
        max_x: the maximum queue size expected in the system

        TODO: This model only include departure flights and the ramp area. Arrival flights and 
                other areas like runway1 and runway2 need to be done later.
        """

        self.day_seconds = 24 * 60 * 60
        self.hour_seconds = 60 * 60
        self.day_hours = 24
        self.deptFlights = data.deptFlights
        self.arrvFlights = data.arrvFlights
        self.deptDays = data.deptDays
        self.arrvDays = data.arrvDays
        self.bin_size = bin_size
        self.percentile = percentile
        self.output_path = output_path
        self.tz_diff = tz_diff
        self.max_x = max_x

        os.makedirs(output_path, exist_ok=True)


    def handle_area_service(self, df: DataFrame, area):
        """
        Get service time and first time based on flight data table.

        Input:
        ----------
        df: the data table (columns include area, time and so on)
        area: selected area

        Output:
        ----------
        Service time and first time
        """

        serv_t, first_t = None, None
        # find flights in area
        first = df.query(f'area=="{area}"')
        if not first.empty:
            # find the first flight in and out of the area
            first_t = first.iloc[0]["time"]
            last = df.query(f'area!="{area}" & time>{first_t}')
            if not last.empty:
                last_t = last.iloc[0]["time"]
            else:
                last_t = first.iloc[-1]["time"]
            # service time is the time difference between first and last
            serv_t = last_t - first_t
            serv_t = serv_t if serv_t > 0 else None
        return serv_t, first_t

    def cal_serv_first_t(self, area, flights):
        """
        Get service time and first time from flight data.

        Input:
        ----------
        area: selected area
        flights: flights data (depature or arrival)

        Output:
        ----------
        List of (service time, first time)
        """

        serv_first_t = []
        for ls in flights:
            for df in ls:
                if df.iloc[0]["id"] == "31f9ded6":
                    print(df)
                s, f = self.handle_area_service(df, area)
                if s is not None:
                    serv_first_t.append((s, f))
        return serv_first_t

    def cal_mean_serv_rate(self, serv_t):
        """
        Calculate mean service rate.

        Input:
        ----------
        serv_t: service time list

        Output:
        ----------
        Mean service rate.
        """

        serv_r_list = [1 / x for x in serv_t]
        return np.mean(serv_r_list)

    def cal_serv_bar(self, serv_t):
        """
        Calculate the service time bar (for example over 70%)

        Input:
        ----------
        serv_t: service time list

        Output:
        ----------
        The service time bar.
        """

        bin_num = self.day_seconds // self.bin_size
        hist, edges = np.histogram(serv_t, bins=bin_num, density=True)
        cdf = np.cumsum(hist * np.diff(edges))
        index = np.argmax(cdf >= self.percentile)

        return edges[index]

    def cal_pushback_rate(self, first_t, days):
        """
        Calculate pushback rate.

        Input: 
        ----------
        first_t: first time list
        days: total number of days

        Output:
        ----------
        Pushback rate list.
        """

        # the difference between first times is pushback time
        pushback_t = np.diff(sorted(first_t))
        pushback_r = 1 / pushback_t

        bin_num = self.day_seconds // self.bin_size

        counts = [[] for _ in range(bin_num)]
        bins = np.arange(0, self.day_seconds+self.bin_size, self.bin_size)

        # add to count if in the bin
        for t, r in zip(first_t, pushback_r):
            idx = np.searchsorted(bins, t % self.day_seconds) - 1
            counts[idx].append(r * self.bin_size)

        pushback_rates = [
            np.mean(count) / days if len(count) else 0 for count in counts]

        return pushback_rates

    def cal_C_value(self, serv_t, max_x):
        """
        Calculate the value of C (refer to the paper "BadrinathLiBalakrishnan_JGCD2019.pdf").
        TODO: This is a simplified version.

        Input:
        ----------
        serv_t: service time list
        max_x: the maximum queue size expected in the system

        Output:
        ----------
        The value of C
        """

        Cv = np.std(serv_t) / np.mean(serv_t)

        def integrad(x, C):
            return (x + 1 - math.sqrt(x**2 + 2 * Cv**2 * x + 1)) / (1 - Cv**2) - (C * x) / (1 + C * x)

        def objective(C):
            re, _ = quad(integrad, 0.0, max_x, args=(C,))
            return abs(re)
        initial = 0.0
        C = minimize(objective, initial, bounds=[(0, None)]).x[0]

        return C

    def plot_serv_dist(self, area, serv_t):
        """
        Plot distribution of service time.

        Input:
        ----------
        area: selected area
        serv_t: service time list

        Output:
        ----------
        The figure of service time distribution.
        """

        plt.figure()
        plt.hist(serv_t, 40)
        plt.xlabel("Service time (s)")
        plt.title(f"Service time distribution for taxi-out {area}")
        plt.savefig(f"{self.output_path}/{area}_serv_t.png")

    def plot_pushback_rate(self, area, pushback_rates):
        """
        Plot the figure of pushback rate, just for reference.

        Input:
        ----------
        area: the selected area
        pushback_rates: pushback rate

        Output:
        ----------
        The figure
        """

        bin_num = self.day_seconds // self.bin_size
        t = np.arange(bin_num)
        x_ticks = np.array([4, 8, 12, 16, 20, 24]) * bin_num // 24
        x_tick_labels = ["4h", "8h", "12h", "16h", "20h", "24h"]
        plt.figure()
        plt.plot(t, pushback_rates)
        plt.xticks(x_ticks, x_tick_labels)
        plt.xlabel("Local time (5min window)")
        plt.ylabel("Pushback rate (ac / 5min)")
        plt.title(f"Pushback rate for taxi-out {area} queue")
        plt.savefig(f"{self.output_path}/{area}_pushback_rate.png")

    def plot_queueLen_data(self, days, area, serv_t, first_t):
        """
        Plot the figure of queue length (data version)

        Input:
        ----------
        days: number of days
        area: the selected area
        serv_t: list of service time 
        first_t: list of sorted first time entering area

        Output:
        ----------
        The figure of queue length (data version)
        """

        bin_num = self.day_seconds // self.bin_size
        counts = np.zeros(bin_num)
        bins = np.arange(0, self.day_seconds+self.bin_size, self.bin_size)

        # check the range [first_t, first_t + serv_t] located in which bins,
        # add 1 to those bins 
        for f, s in zip(first_t, serv_t):
            idx = np.searchsorted(bins, f % self.day_seconds) - 1
            for k in range(idx, idx + 1 + s // self.bin_size):
                counts[k % bin_num] += 1

        # the counts got before is the total flight numbers over 
        # the time period, so divide it by total days
        queue_len = counts / days

        t = np.arange(bin_num)
        x_ticks = np.array([0, 4, 8, 12, 16, 20, 24]) * bin_num // 24
        x_tick_labels = ["0h", "4h", "8h", "12h", "16h", "20h", "24h"]
        plt.figure()
        plt.plot(t, queue_len)
        plt.xticks(x_ticks, x_tick_labels)
        plt.xlabel("Local Time")
        plt.ylabel("Length")
        plt.title(f"Taxi-out {area} queue (data)")
        plt.savefig(f"{self.output_path}/{area}_queue_len_data.png")

    def plot_queueLen_model(self, area, pushback_rates, parameters):
        """
        Plot the figure of queue length (model version)

        Input:
        ----------
        area: the selected area
        pushback_rates: the pushback rates of queue
        parameters: parameters required by solving ODEs, contain "mean_serv_rate" and "C"

        Output:
        ----------
        The figure of queue length (model version)
        """

        bin_num = self.day_seconds // self.bin_size
        x = np.zeros(bin_num)
        t = np.arange(bin_num)

        # TODO: this is a simplified ODE, maybe use a new one later
        for i in range(len(pushback_rates) - 1):
            dxdt = -parameters["mean_serv_rate"] * self.bin_size * parameters["C"] * x[i] / \
                (1 + parameters["C"] * x[i]) + pushback_rates[i]
            x[i+1] = x[i] + dxdt
            if x[i+1] < 0:
                x[i+1] = 0

        plt.figure()
        plt.plot(t, x)
        plt.xlabel("Local Time")
        plt.ylabel("Length")
        plt.title(f"Taxi-out {area} queue (model)")
        x_ticks = np.array([0, 4, 8, 12, 16, 20, 24]) * bin_num // 24
        x_tick_labels = ["0h", "4h", "8h", "12h", "16h", "20h", "24h"]
        plt.xticks(x_ticks, x_tick_labels)
        plt.savefig(f"{self.output_path}/{area}_queue_len_model.png")

    def plot_queueLen_model_data(self, days, area, serv_t, first_t, pushback_rates, parameters):
        """
        Plot the figures of queue length (model version and data version)

        Input:
        ----------
        days: number of days
        area: the selected area
        serv_t: list of service time 
        first_t: list of sorted first time entering area
        pushback_rates: the pushback rates of queue
        parameters: parameters required by solving ODEs, contain "mean_serv_rate" and "C"

        Output:
        ----------
        The figure of queue length (model version and data version)
        """

        bin_num = self.day_seconds // self.bin_size
        t = np.arange(bin_num)

        # data version
        counts = np.zeros(bin_num)
        bins = np.arange(0, self.day_seconds+self.bin_size, self.bin_size)
        # check the range [first_t, first_t + serv_t] located in which bins,
        # add 1 to those bins 
        for f, s in zip(first_t, serv_t):
            idx = np.searchsorted(bins, f % self.day_seconds) - 1
            for k in range(idx, idx + 1 + s // self.bin_size):
                counts[k % bin_num] += 1
        # the counts got before is the total flight numbers over 
        # the time period, so divide it by total days
        queue_len_data = counts / days

        # model version
        x = np.zeros(bin_num)
        # TODO: this is a simplified ODE, maybe use a new one later
        for i in range(len(pushback_rates) - 1):
            dxdt = -parameters["mean_serv_rate"] * self.bin_size * parameters["C"] * x[i] / \
                (1 + parameters["C"] * x[i]) + pushback_rates[i]
            x[i+1] = x[i] + dxdt
            if x[i+1] < 0:
                x[i+1] = 0

        queue_len_mode = x

        plt.figure()
        plt.plot(t, queue_len_mode)
        plt.plot(t, queue_len_data)
        plt.xlabel("Local Time")
        plt.ylabel("Length")
        plt.title(f"Taxi-out {area} queue")
        x_ticks = np.array([0, 4, 8, 12, 16, 20, 24]) * bin_num // 24
        x_tick_labels = ["0h", "4h", "8h", "12h", "16h", "20h", "24h"]
        plt.xticks(x_ticks, x_tick_labels)
        plt.legend(["model", "data"])
        plt.savefig(f"{self.output_path}/{area}_queue_len.png")


    def convert_t(self, timestamp):
        """
        Convert timestamp (utc time) to the local time of SAT

        Input:
        ----------
        timestamp: the time column in tables

        Output:
        ----------
        the local time of hours in string type (like "2023-12-17 10")
        """

        return (
            datetime
            .utcfromtimestamp(timestamp)
            .strftime('%Y-%m-%d %H')
        )


    def plot_value_per_hour(self, serv_t, first_t):
        """
        Plot mean service time based on a time line per hour.
        
        Input:
        ----------
        serv_t: list of service time 
        first_t: list of sorted first time entering area

        Output:
        ----------
        1. figure of mean service time per hour based on a time line
        2. figure of C value per hour based on a time line
        """

        t, mean_serv_t_ph, C_value_ph = [], [], []

        # add the first hour into t
        t.append(first_t[0] // self.hour_seconds)

        # part_serv_t is the temp service time list per hour
        part_serv_t = []
        for s, f in zip(serv_t, first_t):
            if self.convert_t(f) in t:
                # if the hour is in list t, add s into part_serv_t
                part_serv_t.append(s)
            else:
                # if the hour is not in list t, add the hour into t
                t.append(self.convert_t(f))
                # calculate the mean service time on this hour
                mean_serv_t_ph.append(np.mean(part_serv_t) if len(part_serv_t) else 0)
                # calculate the C value on this hour
                C_value_ph.append(self.cal_C_value(part_serv_t, self.max_x) if len(part_serv_t) else 0)
                # renew part_serv_t
                part_serv_t = []
        # the last hour's value are not considered yet, add them into lists
        mean_serv_t_ph.append(np.mean(part_serv_t) if len(part_serv_t) else 0)
        C_value_ph.append(self.cal_C_value(part_serv_t, self.max_x) if len(part_serv_t) else 0)

        plt.figure()
        plt.bar(range(len(t)), mean_serv_t_ph)
        plt.xlabel("Hour")
        plt.ylabel("Mean Service Time (s)")
        plt.title("Mean service time per hour")
        plt.savefig(f"{self.output_path}/mean_serv_time_per_hour.png")

        plt.figure()
        plt.bar(range(len(t)), C_value_ph)
        plt.xlabel("Hour")
        plt.ylabel("C")
        plt.title("Value of parameter C per hour")
        plt.savefig(f"{self.output_path}/C_value_per_hour.png")


    def solve_dept(self, area):
        """
        Solve the ODE of depature flights and plot the figures.

        Input:
        ----------
        area: the selected area

        Output:
        ----------
        figures of service time distribution, 
                   pushback rates, 
                   length of queue (data version),
                   length of queue (model version), 
                   two versions into one figure, 
                   service time and C value per hour over a time line
        """

        # get service time and first time
        serv_first_t = self.cal_serv_first_t(area, self.deptFlights)
        print(serv_first_t)
        serv_t = [s for s, _ in serv_first_t]

        # calculate the 'target_percent' bar
        bar = self.cal_serv_bar(serv_t)
        # only flights with service time more than bar are considered in queue
        serv_t, first_t = [], []
        for s, f in serv_first_t:
            if bar < s:
                serv_t.append(s)
                # get local first time
                first_t.append(f - self.tz_diff * self.hour_seconds)

        # calculate mean service rate and pushback rates
        mean_serv_r = self.cal_mean_serv_rate(serv_t)
        pushback_rates = self.cal_pushback_rate(first_t, self.deptDays)
        # calculate C value
        C = self.cal_C_value(serv_t, self.max_x)

        # plot figure of service time distribution, pushback rates, and length of queue (data version)
        self.plot_serv_dist(area, serv_t)
        self.plot_pushback_rate(area, pushback_rates)
        self.plot_queueLen_data(self.deptDays, area, serv_t, first_t)

        # get parameters, prepare to solve the ODE        
        parameters = {"mean_serv_rate": mean_serv_r, "C": C}
        # solve the ODE and plot length of queue (model version)
        self.plot_queueLen_model(area, pushback_rates, parameters)
        # plot two versions into one figure
        self.plot_queueLen_model_data(
            self.deptDays, area, serv_t, first_t, pushback_rates, parameters)
        # plot service time and C value per hour over a time line
        self.plot_value_per_hour(serv_t, first_t)