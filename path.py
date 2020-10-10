from commodity import Commodity


class Path:
    def __init__(self, arcs, commodity: Commodity):
        """
        path is defined wrt commodity
        :param arcs: ordered list of arcs object
        """
        self.commodity = commodity
        self.origin = arcs[0].origin
        self.dest = arcs[-1].dest
        self.distance = 0.0
        self.arcs = []
        self.nodes = []
        for arc in arcs:
            self.distance += arc.distance
            self.arcs.append(arc)
            if arc.origin not in self.nodes:
                self.nodes.append(arc.origin)
            if arc.dest not in self.nodes:
                self.nodes.append(arc.dest)
