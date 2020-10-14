import csv
import pandas as pd


class ZipCoords:
    """
    weighted coordinates were calculate in the notebook dataanalysis,ipynb and stored in a csv
    this class simply reads from the csv
    """
    def __init__(self):
        zip_coords = pd.read_csv("zip_coords.csv")
        self.zip_coords = zip_coords.set_index(['customer ZIP'])

    def get_lat(self, zip_name: str):
        zip_name = zip_name.replace(" ", "")
        lat = self.zip_coords.loc[int(zip_name), 'weighted_latitude']
        return lat

    def get_lon(self, zip_name: str):
        zip_name = zip_name.replace(" ", "")
        lon = self.zip_coords.loc[int(zip_name), 'weighted_longitude']
        return lon
