from node import Node
from arc import Arc
from zip import Zip
from path import Path
from commodity import Commodity

def dist(lat1, lat2, lon1, lon2):
    dist =69.5 * abs(lat1 - lat2) + 57.3 * abs(lon1 - lon2)
    return dist

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

    def add_node(self, name: str, lat: float, lon: float, ):
        """
        add node to the network
        :param name:
        :param lon:
        :param lat:
        :return:
        """
        node = Node(name, lat, lon)
        self.nodes.append(node)
        if node.nodetype == "O":
            self.origin_nodes.append(node)
        if node.nodetype == "T":
            self.trans_nodes.append(node)
        if node.nodetype == "D":
            self.dest_nodes.append(node)

    def add_zip(self, name: str, lat: float, lon: float, ):
        """
        add zip to the network
        :param name:
        :param lon:
        :param lat:
        :return:
        """
        zipcode = Zip(name, lat, lon)
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

    def add_commodity(self, origin: Node, dest: Zip, quantity=0, scen_num=-1):
        """
        add commodity to the network
        :param scen_num: scenario number. -1 means determinsitic problem hence default value
        :param origin:
        :param dest:
        :param quantity: default 0 if not given
        :return:
        """
        commodity = Commodity(origin, dest)
        commodity.set_scenario(scen_num)
        commodity.set_quantity(quantity)
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

    def get_closest_dest_nodes(self, zip_object, num_dest_nodes):
        distance = {}
        for d in self.dest_nodes:
            distance[d] = dist(zip_object.lat, d.lat,zip_object.lon,zip_object.lon)
        sorted_distance = sorted(distance.items(), key=lambda kv: kv[1])
        nodes_to_keep = []
        for i in sorted_distance[:num_dest_nodes]:
            nodes_to_keep.append(i[0])
        return nodes_to_keep

    def get_dest_node_zip_paths(self, node, zip_code: Zip):
        """
        get all apaths which end at given destination node
        :param node:
        :return:
        """
        paths_to_return = []
        for path in self.paths:
            if node == path.dest and zip_code == path.commodity.dest:
                paths_to_return.append(path)
        return paths_to_return


    def get_commodity_dest_node_paths(self, commodity, node):
        """
        get all apaths for the given commodity which end at given destination node
        :param commodity:
        :param node:
        :return:
        """
        paths_to_return = []
        for path in self.paths:
            if node == path.dest and path.commodity == commodity:
                paths_to_return.append(path)
        return paths_to_return

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


    def get_commodity_by_name(self, name):
        for comm in self.commodities:
            if comm.name == name:
                return comm

    def get_scenario_commodities(self, scen_num):
        scen_commodities = []
        for comm in self.commodities:
            if comm.scenario_num == scen_num:
                scen_commodities.append(comm)
        return scen_commodities
