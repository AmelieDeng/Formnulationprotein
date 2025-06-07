import os
import torch
from torch_geometric.nn import knn_graph
from biopandas.pdb import PandasPdb


#output_path = "combined_edge_index.pt"  # 输出文件路径


def pos_chain_pos(pdb_path):
    pdb_to_pandas = PandasPdb().read_pdb(pdb_path)
    pdb_df = pdb_to_pandas.df['ATOM']
    pdb_df = pdb_df[pdb_df['atom_name'] == 'CA']
    ca_coord = torch.tensor(pdb_df[['x_coord', 'y_coord', 'z_coord']].values, dtype=torch.float32)

    chain_coordinates = {}
    for chain_id, chain_df in pdb_df.groupby('chain_id'):
        coordinates = chain_df[['x_coord', 'y_coord', 'z_coord']].to_numpy()
        coords_tensor = torch.tensor(coordinates, dtype=torch.float)
        chain_coordinates[chain_id] = coords_tensor
        
    return ca_coord, chain_coordinates


