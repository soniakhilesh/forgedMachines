from node import Node
from arc import Arc
from zip import Zip
from path import Path
from commodity import Commodity


class Network:

    def __init__(self):
        self.nodes = []
        self.zips = []
        self.arcs = []
        self.commodities = []
        self.origin_nodes = []
        self.trans_nodes = []
        self.dest_nodes = []
        self.paths = []

    def add_node(self, name: str, lon: float, lat: float):
        """
        add node to the network
        :param name:
        :param lon:
        :param lat:
        :return:
        """
        node = Node(name, lon, lat)
        self.nodes.append(node)
        if node.nodetype == "O":
            self.origin_nodes.append(node)
        if node.nodetype == "T":
            self.trans_nodes.append(node)
        if node.nodetype == "D":
            self.dest_nodes.append(node)

    def add_zip(self, name: str, lon: float, lat: float):
        """
        add zip to the network
        :param name:
        :param lon:
        :param lat:
        :return:
        """
        zipcode = Zip(name, lon, lat)
        self.zips.append(zipcode)

    def add_arc(self, origin: Node, dest: Node):
        """
        add arc to the network
        :param origin:
        :param dest:
        :return:
        """
        arc = Arc(origin, dest)
        self.arcs.append(arc)

    def add_commodity(self, origin: Node, dest: Zip, quantity):
        """
        add commodity to the network
        :param origin:
        :param dest:
        :param quantity:
        :return:
        """
        commodity = Commodity(origin, dest, quantity)
        self.commodities.append(commodity)

    def add_path(self, arcs, commodity):
        """
        add path to the network
        :param arcs:
        :param commodity:
        :return:
        """
        path = Path(arcs, commodity)
        self.paths.append(path)

    # NEW
    def get_commodity_dest(self, commodity):
        """
        get destination zip corresponding to given commodity
        :param commodity:
        :return:
        """
        for k in self.commodities:
            if commodity == k:
                destination = k.dest
                return destination

    def get_commodity_paths(self, commodity):
        """
        get paths corresponding to given commodity
        :param commodity:
        :return:
        """
        commodity_paths = []
        for path in self.paths:
            if path.commodity == commodity:
                commodity_paths.append(path)
        return commodity_paths

    def get_arc_paths(self, arc):
        """
        get all paths which contain given arc
        :param arc:
        :return:
        """
        arc_paths = []
        for path in self.paths:
            if arc in path.arcs:
                arc_paths.append(path)
        return arc_paths

    # NEW
    def get_dest_node_paths(self, node):
        """
        get all apaths which end at given destination node
        :param node:
        :return:
        """
        dest_node_paths = []
        for path in self.paths:
            if node == path.dest:
                dest_node_paths.append(path)
        return dest_node_paths

    # NEW
    def get_lat(self, n):
        """
        get node corresponding to given node name
        :type n: str
        :param n:
        :return:
        """
        for node in self.nodes:
            if node == n:
                return node.lat
        for z in self.zips:
            if z == n:
                return z.lat

    # NEW
    def get_zip_lat(self, n):
        """
        get node corresponding to given node name
        :type n: str
        :param n:
        :return:
        """
        for z in self.zips:
            if z == n:
                return z.lat

    # NEW
    def get_lon(self, n: str):
        """
        get node corresponding to given node name
        :type n: str
        :param n:
        :return:
        """
        for node in self.nodes:
            if node.name == n:
                return node.lon
        for z in self.zips:
            if z == n:
                return z.lat

    # NEW
    def get_zip_lon(self, n: str):
        """
        get node corresponding to given node name
        :type n: str
        :param n:
        :return:
        """
        for z in self.zips:
            if z == n:
                return z.lat

    def get_arc_dist(self, arc):
        """
        get node corresponding to given node name
        :param name:
        :return:
        """
        for a in self.arcs:
            if a == arc:
                return a.distance

    def get_node(self, name: str):
        """
        get node corresponding to given node name
        :param name:
        :return:
        """
        for node in self.nodes:
            if node.name == name:
                return node

    def get_origin_nodes(self):
        """
        get all origin nodes
        :return:
        """
        return self.origin_nodes

    def get_trans_nodes(self):
        """
        get all trans shipment nodes
        :return:
        """
        return self.trans_nodes

    def get_dest_nodes(self):
        """
        get all destination nodes
        :return:
        """
        return self.dest_nodes

    def get_all_nodes(self):
        """
        get all nodes
        :return:
        """
        return self.nodes

    def get_arc(self, name: str):
        """
        getting arc object for the given arc name
        :param name:
        :return:
        """
        for arc in self.arcs:
            if arc.name == name:
                return arc

    def get_arcs(self):
        """
        getting all arcc objects in the network
        :return:
        """
        return self.arcs

    def get_zip(self, name: str):
        """
        getting a zip object for the given zip name
        :param name:
        :return:
        """
        for zipcode in self.zips:
            if zipcode.name == name:
                return zipcode

    def get_zips(self):
        """
        getting all zip objects in the network
        :return:
        """
        return self.zips

    def get_commodities(self):
        """
        gettin all commodities object in the network
        :return:
        """
        return self.commodities

