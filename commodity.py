from node import Node
from zip import Zip


class Commodity:
    def __init__(self, origin: Node, dest: Zip):
        """

        :param origin: origin node
        :param dest: dest zip
        :param quantity: number of packages
        """
        self.name = origin.name+dest.name
        self.origin_node = origin
        self.dest = dest
        # assigning default values
        self.quantity = 0
        self.scenario_num = -1

    def set_quantity(self,quantity):
        self.quantity = quantity

    def get_quantity(self):
        return self.quantity

    def set_scenario(self, scen_num):
        self.scenario_num = scen_num

    def get_scenario_num(self):
        return self.scenario_num

