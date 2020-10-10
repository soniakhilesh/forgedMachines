
class Node:
    def __init__(self, name: str,  Lat: float, Lon: float):
        """

        :param nodetype: O T or D
        :param Lon: float
        :param Lat: float
        """
        self.name = name
        self.lon = Lon
        self.lat = Lat
        if name[0] == 'O':
            self.nodetype="O"
        if name[0] == 'T':
            self.nodetype="T"
        if name[0] == 'D':
            self.nodetype="D"



