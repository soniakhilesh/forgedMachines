from node import Node


class Arc:
    def __init__(self, origin: Node, dest: Node):
        """

        :param origin: Node
        :param dest: Node
        """
        self.origin = origin
        self.dest = dest
        self.distance = 69.5*abs(self.origin.lat - self.dest.lat) + 57.3*abs(self.origin.lon - self.dest.lon)
        self.name=origin.name+dest.name

