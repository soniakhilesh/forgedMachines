from node import Node
from zip import Zip


class Commodity:
    def __init__(self, origin: Node, dest: Zip, quantity):
        """

        :param origin: origin node
        :param dest: dest zip
        :param quantity: number of packages
        """
        self.origin_node = origin
        self.dest = zip
        self.quantity = quantity
