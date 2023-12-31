import pandas as pd
from datetime import datetime, timedelta

class DataLoader:
    def __init__(self, areas, time_periods, timeDiff=300, data_path="../data/depature"):
        """
        Load and process data from flight tables.

        Parameters:
        -----------
        areas: the coordinates of area polygons
        time_periods: selected time periods
        timeDiff: if two continuous rows with same id have a difference larger than timeDiff, 
                  then they are considered as two different flights
        data_path: the path storing data tables

        TODO: This DataLoader can only load depature data, the arrival data need to be done later
        """

        self.DAY_HOURS = 24
        self.areas = areas
        self.timeDiff = timeDiff
        self.time_periods = time_periods
        self.deptFlights = []
        self.arrvFlights = []
        self.deptDays = None
        self.arrvDays = None
        self.data_path = data_path

    def is_inside(self, x, y, polygon):
        """
        Define a function that takes a point (x, y) and a polygon as arguments.
        Return whether this point is inside the polygen or not.
        
        Input:
        ----------
        x, y: the coordinates of the point
        polygon: the coordinates of polygon points

        Output:
        Whether the point is in the polygon or not
        """

        # Initialize the number of crossings to zero
        crossings = 0
        # Loop through each edge of the polygon
        for i in range(len(polygon)):
            # Get the current and next vertices
            p1x, p1y = polygon[i]
            p2x, p2y = polygon[(i+1) % len(polygon)]
            # Check if the point is within the vertical range of the edge
            if y > min(p1y, p2y) and y <= max(p1y, p2y):
                # Check if the point is to the left of the edge
                if x <= max(p1x, p2x):
                    # Compute the intersection point of the ray and the edge
                    if p1y != p2y:
                        xint = (y-p1y) * (p2x-p1x) / (p2y-p1y)+p1x
                    # If the ray is collinear with the edge, the point is on the boundary
                    if p1x == p2x or x <= xint:
                        crossings += 1
        # Return True if the number of crossings is odd, False otherwise
        return crossings % 2 == 1

    def area_info(self, flight):
        """
        Reture which area the coordinates of this row is in.

        Input:
        ----------
        flight: one row of flight data table

        Output:
        ----------
        area name 
        """

        # Define areas as a dictionary of vertices
        for area_name, area_polygon in self.areas.items():
            if self.is_inside(flight["latitude"], flight["longitude"], area_polygon):
                return area_name

        return "none"

    def separate_flights(self, df):
        """
        Seperate flights by time difference.
        
        Input:
        ----------
        df: one data table

        Output:
        ----------
        sepa_df: separated flight tables (List of DataFrame)
        """

        # Calculate the time difference between consecutive rows
        df["time_diff"] = df["time"].diff()

        # Initialize a list to store separated flights (DataFrames)
        sepa_df = []
        current_group = []

        for index, row in df.iterrows():
            if not current_group:
                # If the current group is empty, add the first row to it
                current_group.append(row)
            else:
                if row["time_diff"] > self.timeDiff:
                    # Start a new group if the time difference exceeds the threshold
                    sepa_df.append(pd.DataFrame(current_group))
                    current_group = [row]
                else:
                    # Add the row to the current group
                    current_group.append(row)

        # Append the last group
        if current_group:
            sepa_df.append(pd.DataFrame(current_group))

        # Remove the 'time_diff' column from each DataFrame
        for df in sepa_df:
            df.drop(columns=["time_diff"], inplace=True)

        return sepa_df


    def get_file_list(self, start_time, end_time):
        """
        Get the file name list.

        Input:
        ----------
        start_time: start time (timestamp)
        end_time: end time (timestamp)

        Output:
        ----------
        file_list: list of file names
        """

        start_date_time = datetime.strptime(start_time, "%Y-%m-%d %H")
        end_date_time = datetime.strptime(end_time, "%Y-%m-%d %H")
        file_list = []
        current_date_time = start_date_time
        while current_date_time <= end_date_time:
            file_name = current_date_time.strftime("%Y-%m-%d %H") + ".csv"
            file_list.append(file_name)
            current_date_time += timedelta(hours=1)
        return file_list

    def read_csv(self, col_names, filename):
        """
        Read .csv files.

        Input:
        ----------
        col_names: column names
        filename: file name

        Output:
        df: data table (DataFrame)
        """

        try:
            # if exists, return it
            df = pd.read_csv(f"{self.data_path}/{filename}", names=col_names)
        except:
            # if not exist, return empty DataFrame
            df = pd.DataFrame()

        return df

    def get_flights(self):
        """
        Group the whole table by "id" and separate each one into lists.

        Input: 
        ----------
        None
        
        Output:
        ---------
        num_days: total days
        sepa_lst: grouped and separated flight data
        """

        # get file name list
        file_list = []
        for start_time, end_time in self.time_periods:
            file_list.extend(self.get_file_list(start_time, end_time))

        # number of files is number of hours
        num_days = len(file_list) // self.DAY_HOURS
        # get the whole table
        column_names = ["id", "latitude", "longitude", "altitude", "ground_speed", "time", "heading"]
        df = pd.concat(
            [self.read_csv(column_names, filename) for filename in file_list],
            ignore_index=True
        )
        # delete duplicate rows
        df = df.drop_duplicates()
        # add column "area"
        df["area"] = df.apply(
            lambda row: self.area_info(row), axis=1)
        # only remain three columns
        df = df[["id", "time", "area"]]
        
        print("The DataFrame of flight data is:")
        print(df)
        
        # group by id
        grouped_list = [group_df for _, group_df in df.groupby("id")]
        # separate each grouped DataFrame
        sepa_lst = [self.separate_flights(df.sort_values(by="time")) for df in grouped_list]

        return num_days, sepa_lst


    def process_dept(self):
        """Process depature flight data."""

        print("Process data...")
        dept_num_days, sepa_dept_flights = self.get_flights()
        self.deptFlights = sepa_dept_flights
        self.deptDays = dept_num_days

