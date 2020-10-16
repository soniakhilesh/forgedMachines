import pandas as pd
from network import Network


def write_arcs_to_csv(y, network: Network, filename):
    """

    :param network:
    :param y: arc variables gurobi dict of truck variables
    :return:
    """
    df = pd.DataFrame(columns=["Arc", "Number of Trucks"])
    for arc in network.get_arcs():
        arc_name = arc.origin.name + "->" + arc.dest.name
        arc_value = y[arc].x
        df = df.append({'Arc': arc_name, 'Number of Trucks': arc_value}, ignore_index=True)
    df.to_csv(filename, index=False)

def write_zd_to_csv(u, network: Network, filename):
    """

    :param network:
    :param y: arc variables gurobi dict of truck variables
    :return:
    """
    df = pd.DataFrame(columns=["Assigned Destination Node", "ZIP"])
    for zip_code in network.get_zips():
        for d in network.get_dest_nodes():
            zip_name = zip_code.name
            d_name = d.name
            if u[zip_code,d].x>=0.5:      # i.e. == 1
                df=df.append({'Assigned Destination Node':d_name,'ZIP':zip_name}, ignore_index=True)
    df.to_csv(filename, index=False)