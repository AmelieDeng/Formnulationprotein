import torch_geometric
from torch_geometric.transforms import KNNGraph
from torch_geometric.utils import to_undirected
import torch

class myKNNGraph(KNNGraph):
    r"""Creates a k-NN graph based on node positions :obj:`pos`.
    """
    def __init__(self, name: str, k=6, loop=False, force_undirected=False,
                 flow='source_to_target'):

        super().__init__(k, loop, force_undirected, flow)
        self.name = name

    def __call__(self, data):
        
        combined_edges = []
        node_offset = 0
        for chain_id, coords_tensor in data['atoms'].chainpos.items():
            print(f"Generating KNN graph for chain {chain_id} with {coords_tensor.size(0)} nodes.")
            

            
            data['atoms', self.name, 'atoms'].edge_attr = None
            batch = data.batch if 'batch' in data else None
            edge_index = torch_geometric.nn.knn_graph(coords_tensor,
                                                      self.k, batch,
                                                      loop=self.loop,
                                                      flow=self.flow)
            if self.force_undirected:
                edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)
            
            adjusted_edge_index = edge_index + node_offset
            combined_edges.append(adjusted_edge_index)
            node_offset += coords_tensor.size(0)
            
            #print(adjusted_edge_index)
            #print(node_offset)
            
            #print('adjusted_edge_index',adjusted_edge_index)
            #print('combined_edges',combined_edges)
                        
        combined_edge_index = torch.cat(combined_edges, dim=1)
        #print('combined_edge_index',combined_edge_index)
        
        data['atoms', self.name, 'atoms'].edge_index = combined_edge_index
        
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(k={self.k})'
