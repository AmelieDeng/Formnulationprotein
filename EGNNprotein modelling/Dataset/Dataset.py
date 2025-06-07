import math
import os
import pickle
import subprocess
import torch
import os.path as osp
from torch_geometric.data import Dataset, download_url, HeteroData
from biopandas.pdb import PandasPdb
import Constants
from Dataset.distanceTransform import myDistanceTransform
from Dataset.myKnn import myKNNGraph
from Dataset.myRadiusGraph import myRadiusGraph
from Dataset.utils import find_files, process_pdbpandas, get_knn, generate_Identity_Matrix
import torch_geometric.transforms as T
from torch_geometric.data import Data
from Dataset.AdjacencyTransform import AdjacencyFeatures
from preprocessing.utils import pickle_load, pickle_save, get_sequence_from_pdb, fasta_to_dictionary, collect_test, \
    read_test_set, read_test, cafa_fasta_to_dictionary
import pandas as pd
import random
from Dataset.cla1 import embedding_construction
from Dataset.cla2 import pos_chain_pos

class PDBDataset(Dataset):

    def __init__(self, root, esm_p, formdata_p, transform=None,pre_transform=None, pre_filter=None):

        self.root = root
        self.formdata_p = formdata_p
        # formulation data          
        form = pd.read_csv(formdata_p)
        self.prot_ids = form['Protein'].tolist()
        
        self.data = self.prot_ids
        self.esm_path = esm_p
        
        
        
        super().__init__(self.root, transform, pre_transform, pre_filter)

    def len(self):
        return len(self.data)

    def get(self, idx):
    
        formulationdata = pd.read_csv(self.formdata_p)
        labels = formulationdata.set_index('Protein')['Label'].to_dict()
        
        rep = self.data[idx]       
        
        label = labels[rep]
        
        inputdata = torch.load(osp.join(self.root + "/processed_data/{}/{}.pt".format(self.esm_path, rep)), weights_only=False)
        #print('inputdata', inputdata['atoms'].chainpos)

        if 'chainpos' in inputdata['atoms']:
            del inputdata['atoms'].chainpos
        
          
                
        return inputdata, label 

def load_dataset(root, esm_p, formdata_p):

    if root == None:
        raise ValueError('Root path is empty, specify root directory')
    dataset = PDBDataset(root, esm_p, formdata_p)
    
    return dataset




